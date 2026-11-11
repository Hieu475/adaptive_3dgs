#!/usr/bin/env python3
"""Pair policy-independent office1 input features with exact-snapshot quota curves."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


RUNS = {
    45: ("office1_quota_dose_response_seeds282_284", (282, 283, 284)),
    70: ("office1_quota_dose_response_frame70_seeds285_287", (285, 286, 287)),
    90: ("office1_quota_dose_response_frame90_seeds288_290", (288, 289, 290)),
}
FIXED = (25, 40, 49, 60, 75)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mean(xs):
    return statistics.mean(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()
    repo = args.repo.resolve()
    results = repo / "results/legacy/phase14_generalization_postfix"
    feature_root = results / "frame_diagnostics"
    feature_manifest = json.loads((feature_root / "manifest.json").read_text())
    feature_path = feature_root / "replica_office1/ours/seed42.json"
    feature_data = json.loads(feature_path.read_text())
    feature_by_frame = {int(r["frame"]): r for r in feature_data["frames"]}
    feature_fields = ("camera_translation_m_per_frame", "camera_angular_deg_per_frame",
                      "new_surface_ratio_voxel_5cm", "depth_valid_ratio_sampled")

    # These four features depend only on RGB-D and GT poses. Verify their values
    # are invariant across the available input-sequence seeds and policies.
    feature_invariance = {}
    for frame in RUNS:
        values = set()
        for policy in ("ours", "error_only"):
            for seed in range(42, 50):
                p = feature_root / f"replica_office1/{policy}/seed{seed}.json"
                x = json.loads(p.read_text())
                row = next(r for r in x["frames"] if int(r["frame"]) == frame)
                values.add(tuple(row[k] for k in feature_fields))
        assert len(values) == 1, f"policy-independent input features differ across seeds/policies at frame {frame}"
        feature_invariance[str(frame)] = True

    per_state = {}
    provenance = {"input_feature_file_sha256": sha256(feature_path),
                  "input_feature_manifest_sha256": sha256(feature_root / "manifest.json"),
                  "dose_response_runs": {}}
    for frame, (run_name, seeds) in RUNS.items():
        run_root = results / run_name
        prereg = json.loads((run_root / "preregistration.json").read_text())
        manifest = json.loads((run_root / "manifest.json").read_text())
        assert prereg["scene"] == "replica_office1" and prereg["snapshot_frames"] == [frame]
        assert prereg["seeds"] == list(seeds)
        assert prereg["intervention"]["fixed_mature_fractions"] == [0.25, 0.4, 0.49, 0.6, 0.75]
        assert prereg["intervention"]["total_micro_updates_per_action"] == 5000
        by_seed = {}
        for seed in seeds:
            cell = json.loads((run_root / "replica_office1" / f"seed{seed}_frame{frame}.json").read_text())
            assert cell["all_count_and_snapshot_gates_passed"] and len(cell["arms"]) == 60
            policies = defaultdict(list)
            for arm in cell["arms"]:
                assert arm["valid"] and arm["snapshot_hash"] == cell["base_snapshot_hash"]
                assert arm["n_spawned"] == 128 and arm["n_micro_updates"] == 5000
                name = arm["action"]["name"]
                if arm["quota_policy"] == "coverage_saturating":
                    label = "adaptive"
                    assert arm["realized_mature_micro_updates"] == min(3750, arm["mature_eligible"])
                else:
                    frac = float(arm["action"]["mature_fraction"])
                    pct = round(100 * frac)
                    label = f"fixed{pct}"
                    assert pct in FIXED and arm["realized_mature_micro_updates"] == round(frac * 5000)
                policies[label].append(arm)
            assert set(policies) == {f"fixed{x}" for x in FIXED} | {"adaptive"}
            assert all(len(policies[p]) == 10 for p in policies)
            # Snapshot-level diagnostics are emitted from the same common
            # pre-spawn render and must not depend on action order.
            diag = cell["arms"][0]["age_stratified_diagnostics"]
            assert all(a["age_stratified_diagnostics"] == diag for a in cell["arms"])
            by_seed[seed] = {
                p: {"psnr": mean([a["final_map_eval_psnr"] for a in aa]),
                    "depth_l1": mean([a["final_map_eval_depth_l1"] for a in aa]),
                    "opt_ms": mean([a["opt_stage_sync_ms"] for a in aa]),
                    "mature_share": mean([a["realized_mature_fraction"] for a in aa]),
                    "mature_coverage": mean([a["mature_update_coverage"] for a in aa])}
                for p, aa in policies.items()
            }
            by_seed[seed]["diagnostics"] = diag
            by_seed[seed]["snapshot_hash"] = cell["base_snapshot_hash"]

        pooled = {}
        labels = [f"fixed{x}" for x in FIXED] + ["adaptive"]
        for label in labels:
            pooled[label] = {metric: mean([by_seed[s][label][metric] for s in seeds])
                             for metric in ("psnr", "depth_l1", "opt_ms", "mature_share", "mature_coverage")}
        fixed_best = max((f"fixed{x}" for x in FIXED), key=lambda p: pooled[p]["psnr"])
        diagnostics = [by_seed[s]["diagnostics"] for s in seeds]
        young_color = mean([d["young_lt_10_frames"]["visible_color_error_mean"] for d in diagnostics])
        mature_color = mean([d["mature_ge_10_frames"]["visible_color_error_mean"] for d in diagnostics])
        young_depth = mean([d["young_lt_10_frames"]["visible_depth_error_mean"] for d in diagnostics])
        mature_depth = mean([d["mature_ge_10_frames"]["visible_depth_error_mean"] for d in diagnostics])
        view_coverage = mean([d["view_coverage_valid_depth"] for d in diagnostics])
        # Aggregate only the oldest mature revisit band, retaining its actual
        # visible population so a small stratum is not mistaken for a stable mean.
        oldest = [d["mature_revisit_strata"]["staleness_20_plus_frames"] for d in diagnostics]
        stale_diag = {"visible_count_mean": mean([x["visible_attributed"] for x in oldest]),
                      "staleness_frames_mean": mean([x["mean_staleness_frames"] for x in oldest]),
                      "update_count_mean": mean([x["mean_update_count"] for x in oldest]),
                      "color_error_mean": mean([x["visible_color_error_mean"] for x in oldest]),
                      "depth_error_mean": mean([x["visible_depth_error_mean"] for x in oldest])}
        input_row = feature_by_frame[frame]
        response = {"delta_psnr_vs25": {p: pooled[p]["psnr"] - pooled["fixed25"]["psnr"] for p in labels},
                    "delta_depth_l1_vs25": {p: pooled[p]["depth_l1"] - pooled["fixed25"]["depth_l1"] for p in labels}}
        per_state[str(frame)] = {
            "input_features": {k: input_row[k] for k in feature_fields},
            "map_diagnostics": {"view_coverage_valid_depth": view_coverage,
                                "young_color_error_mean": young_color,
                                "mature_color_error_mean": mature_color,
                                "young_to_mature_color_ratio": young_color / mature_color,
                                "young_depth_error_mean": young_depth,
                                "mature_depth_error_mean": mature_depth,
                                "young_to_mature_depth_ratio": young_depth / mature_depth,
                                "mature_staleness_20plus": stale_diag},
            "policy_means": pooled, "delta_vs25": response,
            "best_fixed_policy": fixed_best,
            "seed_level": {str(s): by_seed[s] for s in seeds},
            "all_seeds_best_fixed_same": all(max((f"fixed{x}" for x in FIXED), key=lambda p: by_seed[s][p]["psnr"]) == fixed_best for s in seeds),
        }
        provenance["dose_response_runs"][str(frame)] = {
            "run": run_name,
            "preregistration_sha256": sha256(run_root / "preregistration.json"),
            "manifest_sha256": sha256(run_root / "manifest.json"),
            "seeds": list(seeds),
            "snapshot_hashes": {str(s): by_seed[s]["snapshot_hash"] for s in seeds},
        }

    result = {"title": "Office1 state diagnostics paired with same-snapshot quota response",
              "status": "exploratory; three states on one scene, new seed cohorts by state",
              "analysis_script_sha256": sha256(Path(__file__).resolve()),
              "feature_definition": feature_manifest["surface_novelty_definition"],
              "feature_seed_policy_invariance": feature_invariance,
              "input_feature_provenance": provenance,
              "states": per_state}

    lines = ["# Replica office1 state diagnostics paired with quota response", "",
             "## Matched protocol", "",
             "Three frame states (45, 70, 90) were paired with dose-response replays on the same per-seed map snapshot. Every cell passed exact snapshot, candidate, spawn, update and quota checks; each quota mean averages ten repeats. Camera motion, sampled depth validity and surface novelty are policy-independent input features, verified identical across the eight available input-feature seeds and both policies. Map coverage, age residuals and staleness come from the exact quota-replay snapshot.", "",
             f"Surface novelty is `{feature_manifest['surface_novelty_definition']}`. These three states are along one office1 trajectory, and each frame uses a separate three-seed cohort. This is descriptive interaction analysis, not a predictive validation.", "",
             "## State-response comparison", "",
             "| Frame | Translation m/frame | Rotation deg/frame | New-surface ratio | Valid depth | Map view coverage | Young/mature color residual | Young/mature depth residual | ΔPSNR 75−25 | Best fixed quota |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for frame in RUNS:
        s = per_state[str(frame)]
        x, d = s["input_features"], s["map_diagnostics"]
        delta75 = s["delta_vs25"]["delta_psnr_vs25"]["fixed75"]
        lines.append(f"| {frame} | {x['camera_translation_m_per_frame']:.4f} | {x['camera_angular_deg_per_frame']:.2f} | {x['new_surface_ratio_voxel_5cm']:.4f} | {x['depth_valid_ratio_sampled']:.3f} | {d['view_coverage_valid_depth']:.3f} | {d['young_to_mature_color_ratio']:.2f}× | {d['young_to_mature_depth_ratio']:.2f}× | {delta75:+.4f} dB | {s['best_fixed_policy']} |")
    lines += ["", "## Quota-response curves", "", "| Frame | Fixed 25% | Fixed 40% | Fixed 49% | Fixed 60% | Fixed 75% | Adaptive |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for frame in RUNS:
        s = per_state[str(frame)]
        values = [f"{s['delta_vs25']['delta_psnr_vs25'][f'fixed{x}']:+.4f}" for x in FIXED]
        values.append(f"{s['delta_vs25']['delta_psnr_vs25']['adaptive']:+.4f}")
        lines.append("| " + str(frame) + " | " + " | ".join(values) + " |")
    lines += ["", "Entries are ΔPSNR versus fixed-25% in dB. These are mean differences across three seeds; see the raw per-seed summary for direction and depth-L1 results.", "", "## Oldest mature revisit stratum", "", "| Frame | Visible Gaussian count | Mean staleness (frames) | Mean prior update count | Color residual | Depth residual |", "|---:|---:|---:|---:|---:|---:|"]
    for frame in RUNS:
        d = per_state[str(frame)]["map_diagnostics"]["mature_staleness_20plus"]
        lines.append(f"| {frame} | {d['visible_count_mean']:.0f} | {d['staleness_frames_mean']:.1f} | {d['update_count_mean']:.2f} | {d['color_error_mean']:.5f} | {d['depth_error_mean']:.5f} |")
    lines += ["", "## Interpretation", "", "At frame 45, high surface novelty (0.190) and lower map view coverage (0.767) coincide with the strongest preference for a low mature quota: the fixed quota curve falls as mature allocation rises. At frame 70, novelty is intermediate (0.054), and fixed-49% is the numerical peak but its edge over 25% is negligible; 75% regresses. At frame 90, novelty is low (0.012) and view coverage is near saturation (0.996); higher mature quotas improve the same-view quality, with 75% best. This pattern is consistent with a novelty/coverage interaction, but frame, map age, revisit structure, and camera rotation all change together, so it is not causal evidence for a threshold.", "", "Young-group average residual exceeds mature-group average at all three states, including frame 90 where 75% wins. Aggregate age residual ratio therefore fails as a standalone quota predictor. The oldest mature staleness band also changes in size and residual; notably, frame 90 has higher color residual in this band than frame 70. This is a candidate explanation for heterogeneous mature utility, not a tested causal result.", "", "Do not fit or claim a router from these three states. The useful next test is a preregistered predictor evaluated on held-out scenes/states under matched update count and a small fixed-quota sweep, with state features frozen before outcomes are read.", "", "## Provenance", "", f"- Input feature manifest SHA-256: `{provenance['input_feature_manifest_sha256']}`", f"- Input feature file SHA-256: `{provenance['input_feature_file_sha256']}`", "- Dose-run manifests, preregistrations and paired snapshot hashes: `summary.json`", ""]
    out = results / "office1_state_quota_interactions"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "report.md").write_text("\n".join(lines))
    print(f"Wrote {out/'report.md'}")
    for frame in RUNS:
        s = per_state[str(frame)]
        print(f"frame={frame} novelty={s['input_features']['new_surface_ratio_voxel_5cm']:.4f} best={s['best_fixed_policy']} Δ75−25={s['delta_vs25']['delta_psnr_vs25']['fixed75']:+.4f} dB; young/mature residual ratios={s['map_diagnostics']['young_to_mature_color_ratio']:.2f}x color, {s['map_diagnostics']['young_to_mature_depth_ratio']:.2f}x depth")


if __name__ == "__main__":
    main()
