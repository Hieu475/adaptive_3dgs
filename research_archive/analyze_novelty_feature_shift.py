#!/usr/bin/env python3
"""Describe the frozen novelty threshold's input-feature support across scenes."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAL = ROOT / "results/phase14_generalization_postfix/continuous_age_quota/diagnostics_tum_fr1_desk/rerun"
EVAL = ROOT / "results/phase14_generalization_postfix/continuous_age_quota/novelty_rule_transfer/confirmatory"
SCENES = {
    "calibration: tum_fr1_desk": (CAL / "tum_fr1_desk" / "seed42_global_importance.json", True),
    "held-out: replica_room0": (EVAL / "replica_room0" / "seed42_novelty_rule.json", False),
    "held-out: tum_fr1_room": (EVAL / "tum_fr1_room" / "seed42_novelty_rule.json", False),
}
WINDOWS = ((1, 30), (31, 60), (61, 90), (91, 119))
THRESHOLD = .06


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def quantile(xs, q):
    xs = sorted(xs)
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] + (xs[hi] - xs[lo]) * (p - lo)


def main():
    scene_series = {}
    hashes = {}
    for label, (path, diagnostic_calibration) in SCENES.items():
        record = json.loads(path.read_text())
        rows = record["frames"]
        vals = [float(r["new_surface_ratio_voxel_5cm"]) for r in rows]
        scene_series[label] = (rows, vals, diagnostic_calibration)
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        if not diagnostic_calibration:
            if any(abs(r["requested_mature_fraction"] - (.25 if v < THRESHOLD else .75)) > 1e-9
                   for r, v in zip(rows, vals)):
                raise ValueError(f"Recorded action does not match frozen threshold: {path}")

    summaries = []
    for label, (rows, vals, is_cal) in scene_series.items():
        summaries.append({
            "scene": label,
            "n_frames": len(vals),
            "novelty_mean": sum(vals) / len(vals),
            "novelty_quantiles": {str(q): quantile(vals, q) for q in (.1, .25, .5, .75, .9)},
            "fraction_below_0_06": sum(v < THRESHOLD for v in vals) / len(vals),
            "actual_fraction_routed_to_25pct": (sum(r["requested_mature_fraction"] == .25 for r in rows) / len(rows)
                                                if not is_cal else None),
        })

    windows = []
    for label, (rows, vals, is_cal) in scene_series.items():
        lookup = {int(r["frame"]): (r, float(r["new_surface_ratio_voxel_5cm"])) for r in rows}
        if not is_cal:
            scene_key = label.split(": ")[-1]
            global_rows = json.loads((EVAL / scene_key / "seed42_global_importance.json").read_text())["frames"]
            global_by_frame = {int(r["frame"]): r for r in global_rows}
        for lo, hi in WINDOWS:
            subset = [(lookup[f]) for f in range(lo, hi + 1) if f in lookup]
            if not subset:
                continue
            nv = [x[1] for x in subset]
            rows_win = [x[0] for x in subset]
            route = sum(v < THRESHOLD for v in nv) / len(nv)
            row = {"scene": label, "frame_window": [lo, hi],
                   "n_frames": len(nv), "mean_novelty": sum(nv) / len(nv),
                   "median_novelty": quantile(nv, .5),
                   "fraction_below_threshold": route}
            if not is_cal:
                row["actual_fraction_routed_to_25pct"] = sum(
                    r["requested_mature_fraction"] == .25 for r in rows_win) / len(rows_win)
                row["rule_minus_global_mean_psnr_delta_db"] = sum(
                    r["psnr"] - global_by_frame[int(r["frame"])]["psnr"] for r in rows_win) / len(rows_win)
            windows.append(row)

    result = {
        "status": "descriptive feature-support audit; no inferential test",
        "rule_threshold": THRESHOLD,
        "rule": "new_surface_ratio_voxel_5cm < 0.06 -> request mature fraction 0.25; otherwise 0.75",
        "scene_summaries": summaries,
        "window_summaries": windows,
        "provenance": {"record_sha256": hashes,
                       "calibration_diagnostic_manifest_sha256": sha256(CAL / "manifest.json"),
                       "heldout_manifest_sha256": sha256(EVAL / "manifest.json"),
                       "analysis_script_sha256": sha256(Path(__file__).resolve())},
        "limitations": [
            "Input novelty depends on the ground-truth pose trajectory and uses an offline running voxel set.",
            "One trajectory per scene is enough for this deterministic input-feature distribution check but not inferential scene generalization.",
            "Held-out outcome data are shown only as context; this is not a threshold validation or causal analysis.",
        ],
    }
    out_json = EVAL / "feature_shift_analysis.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Novel-surface feature support across calibration and held-out scenes", "",
        "Descriptive audit of the frozen 0.06 threshold. Calibration trajectory is TUM fr1 desk; held-out trajectories are Replica room0 and TUM fr1 room. The threshold remains unchanged.", "",
        "| Scene | Frames | Novelty mean | p10 / p25 / p50 / p75 / p90 | Frames below 0.06 | Actual frames routed to 25% |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for s in summaries:
        q = s["novelty_quantiles"]
        actual = "—" if s["actual_fraction_routed_to_25pct"] is None else f"{s['actual_fraction_routed_to_25pct']:.1%}"
        lines.append(f"| `{s['scene']}` | {s['n_frames']} | {s['novelty_mean']:.4f} | "
                     f"{q['0.1']:.3f} / {q['0.25']:.3f} / {q['0.5']:.3f} / {q['0.75']:.3f} / {q['0.9']:.3f} | "
                     f"{s['fraction_below_0_06']:.1%} | {actual} |")
    lines += ["", "## Fixed-window view", "",
              "| Scene | Frames | Mean novelty | Median novelty | Below-threshold / routed to 25% | Rule − global PSNR delta (dB; descriptive) |",
              "|:--|:--:|--:|--:|--:|--:|"]
    for r in windows:
        psnr = r.get("rule_minus_global_mean_psnr_delta_db")
        psnr_text = "—" if psnr is None else f"{psnr:+.3f}"
        actual_route = r.get("actual_fraction_routed_to_25pct")
        actual_text = "—" if actual_route is None else f"{actual_route:.1%}"
        lines.append(f"| `{r['scene']}` | {r['frame_window'][0]}–{r['frame_window'][1]} | {r['mean_novelty']:.4f} | "
                     f"{r['median_novelty']:.4f} | {r['fraction_below_threshold']:.1%} / "
                     f"{actual_text} | {psnr_text} |")
    lines += ["", "## Interpretation", "",
              "A target scene can improve over global importance without the novelty switch helping: on TUM fr1 room the feature remains above threshold almost throughout, so the rule behaves like fixed 75%. Feature-support overlap is limited by the small number of trajectories and the GT-pose-only novelty computation. This audit explains routing frequency; it does not establish why either quota improves quality or justify changing the threshold.", "",
              "Per-frame series, hashes, and limits are in `feature_shift_analysis.json`.", ""]
    (EVAL / "feature_shift_report.md").write_text("\n".join(lines))
    print(EVAL / "feature_shift_report.md")


if __name__ == "__main__":
    main()
