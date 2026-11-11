#!/usr/bin/env python3
"""Retrospective paired-seed summary of the existing quota dose-response sweeps.

This is descriptive/exploratory: it does not turn selected scene-states into an
independent held-out generalization set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = [
    "office1_quota_dose_response_seeds282_284",
    "office1_quota_dose_response_frame70_seeds285_287",
    "office1_quota_dose_response_frame90_seeds288_290",
    "office4_quota_dose_response_frame70_seeds291_293",
    "office3_quota_dose_response_frame70_seeds294_296",
]
T95_DF2 = 4.3026527299  # two-sided 95% t critical value, n=3 independent seeds


def mean(xs):
    return statistics.mean(xs)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_stats(values):
    n = len(values)
    avg = mean(values)
    if n < 2:
        return {"n_seeds": n, "mean": avg, "sd": None, "ci95_low": None, "ci95_high": None}
    sd = statistics.stdev(values)
    critical = T95_DF2 if n == 3 else 1.96  # current registered sweeps all have 3 seeds
    half = critical * sd / math.sqrt(n)
    return {"n_seeds": n, "mean": avg, "sd": sd,
            "ci95_low": avg - half, "ci95_high": avg + half}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path,
                    default=ROOT / "results/legacy/phase14_generalization_postfix")
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS,
                    help="Dose-response result subdirectories relative to --results-root")
    args = ap.parse_args()
    result_root = args.results_root.resolve()
    states = []
    for name in args.roots:
        path = result_root / name
        summary = json.loads((path / "summary.json").read_text())
        prereg = json.loads((path / "preregistration.json").read_text())
        seed_rows = summary["per_seed"]
        assert len(seed_rows) == 3, f"expected 3 independent seeds in {path}"
        assert all(set(row["policies"]) >= {"fixed25", "fixed49", "fixed75", "adaptive"}
                   for row in seed_rows)
        frame = int(seed_rows[0]["frame"])
        assert all(int(row["frame"]) == frame for row in seed_rows)
        deltas = {}
        for policy in ("fixed40", "fixed49", "fixed60", "fixed75", "adaptive"):
            if not all(policy in row["policies"] for row in seed_rows):
                continue
            psnr = [row["policies"][policy]["final_map_eval_psnr"] -
                    row["policies"]["fixed25"]["final_map_eval_psnr"] for row in seed_rows]
            depth = [row["policies"][policy]["final_map_eval_depth_l1"] -
                     row["policies"]["fixed25"]["final_map_eval_depth_l1"] for row in seed_rows]
            deltas[policy] = {"psnr_db": paired_stats(psnr), "depth_l1_m": paired_stats(depth),
                              "per_seed_psnr_db": psnr, "per_seed_depth_l1_m": depth}
        fixed = ["fixed25", "fixed40", "fixed49", "fixed60", "fixed75"]
        means = {p: mean([row["policies"][p]["final_map_eval_psnr"] for row in seed_rows])
                 for p in fixed}
        best = max(fixed, key=means.get)
        states.append({"result_dir": name, "scene": prereg["scene"], "frame": frame,
                       "seeds": [int(row["seed"]) for row in seed_rows],
                       "best_fixed_by_mean_psnr": best,
                       "best_fixed_gain_vs25_db": means[best] - means["fixed25"],
                       "fixed_quota_mean_psnr_db": means, "contrasts_vs_fixed25": deltas,
                       "branch_count": summary["branch_count"],
                       "source_summary_sha256": sha256(path / "summary.json"),
                       "source_preregistration_sha256": sha256(path / "preregistration.json"),
                       "source_manifest_sha256": sha256(path / "manifest.json"),
                       "preregistration_sha256": summary["preregistration_sha256"],
                       "manifest_sha256": summary["manifest_sha256"]})

    counts = {}
    for p in ("fixed49", "fixed75", "adaptive"):
        vals = [s["contrasts_vs_fixed25"][p]["psnr_db"]["mean"] for s in states]
        counts[p] = {"states_positive": sum(v > 0 for v in vals),
                     "states_negative": sum(v < 0 for v in vals),
                     "states_tie": sum(v == 0 for v in vals),
                     "unweighted_mean_state_gain_db": mean(vals)}
    output = {"title": "Retrospective quota dose-response transfer summary",
              "status": "exploratory_retrospective_descriptive_analysis",
              "independent_unit": "seed within each scene-state; scene-states are not independent samples",
              "interval_note": "Per-state 95% t intervals use n=3 seed-paired contrasts (df=2); intervals are descriptive and low-powered.",
              "states": states, "state_sign_counts_vs_fixed25": counts,
              "limitations": ["Selected existing states; not a preregistered held-out predictor test.",
                              "One RGB-D evaluation view/state; no geometry completeness or trajectory metric.",
                              "Matched update count, not wall-clock budget.",
                              "PSNR winner is a numerical maximum; no multiplicity correction across quota arms."]}
    outdir = result_root / "quota_dose_response_transfer_analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    lines = ["# Retrospective quota dose-response transfer summary", "",
             "Exploratory aggregation of five existing matched-snapshot dose sweeps. Seeds are the independent units within each scene-state; scene-states are not pooled as independent samples.", "",
             "| Scene-state | Seeds | Best fixed dose | Best gain vs 25% | 49%−25% PSNR (95% t CI) | 75%−25% PSNR (95% t CI) | Adaptive−25% |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in states:
        def fmt(p):
            x = s["contrasts_vs_fixed25"][p]["psnr_db"]
            return f"{x['mean']:+.4f} [{x['ci95_low']:+.4f}, {x['ci95_high']:+.4f}]"
        lines.append(f"| {s['scene']} f{s['frame']} | {', '.join(map(str,s['seeds']))} | {s['best_fixed_by_mean_psnr']} | {s['best_fixed_gain_vs25_db']:+.4f} dB | {fmt('fixed49')} | {fmt('fixed75')} | {fmt('adaptive')} dB |")
    lines += ["", "## Across-state pattern", ""]
    for p, v in counts.items():
        lines.append(f"- **{p} vs fixed25:** {v['states_positive']} positive, {v['states_negative']} negative, {v['states_tie']} tied states; unweighted mean state gain {v['unweighted_mean_state_gain_db']:+.4f} dB.")
    lines += ["", "## Interpretation and limits", "",
              "The numerical best fixed quota varies by state. Fixed-75% does not have a consistent direction, and the adaptive arm is identical to its realized quota (49% at office1 frame45; 75% in the other four states), so this table does not establish an adaptive-policy advantage. Fixed-49% is a useful intermediate dose in some states, but these selected snapshots do not identify a transferable decision rule.", "",
              "The per-state intervals are paired across three seeds and are not corrected for the five quota comparisons. They describe seed variation for these particular snapshots; they do not account for scene/state selection and should not be read as broad generalization intervals. This retrospective aggregation is not causal evidence about a feature-based controller.", "",
              "All runs use same-snapshot replay and matched update counts, but not matched wall time. Each reports same-view RGB-D quality only. Geometry, trajectory-level outcomes, and a preregistered held-out prediction test remain outstanding.", "",
              "## Provenance", "",
              "Input summary, preregistration, and manifest SHA-256 values are recorded per state in `summary.json`.", ""]
    (outdir / "report.md").write_text("\n".join(lines))
    print(f"wrote {outdir / 'report.md'}; {len(states)} scene-states; sign counts {counts}")


if __name__ == "__main__":
    main()
