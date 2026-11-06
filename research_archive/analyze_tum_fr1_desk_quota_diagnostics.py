#!/usr/bin/env python3
"""Descriptive, seed-clustered summaries of preregistered TUM age diagnostics."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/continuous_age_quota/diagnostics_tum_fr1_desk/rerun"
SEEDS = tuple(range(42, 50))
ARMS = ("global_importance", "mature75")
WINDOWS = ((1, 30), (31, 60), (61, 90), (91, 119))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs, seed, n=30000):
    import random
    rng = random.Random(seed)
    vals = sorted(mean([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n))
    return [vals[int(.025 * (n - 1))], vals[int(.975 * (n - 1))]]


def main():
    manifest_path = BASE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("capture_age_diagnostics"):
        raise ValueError("Diagnostic instrumentation is not enabled in the manifest")
    data, hashes = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            p = BASE / "tum_fr1_desk" / f"seed{seed}_{arm}.json"
            record = json.loads(p.read_text())
            if record["summary"]["seed"] != seed or record["summary"]["arm"] != arm:
                raise ValueError(f"Record identity mismatch: {p}")
            if not record["summary"]["update_count_gate_pass"]:
                raise ValueError(f"Update count mismatch: {p}")
            if any(not x.get("age_stratified_diagnostics") for x in record["frames"]):
                raise ValueError(f"Missing age diagnostic: {p}")
            if any(x["n_micro_updates"] != 5000 for x in record["frames"]):
                raise ValueError(f"Not exactly 5k updates: {p}")
            data[(seed, arm)] = record["frames"]
            hashes[str(p.relative_to(ROOT))] = sha256(p)

    rows = []
    for lo, hi in WINDOWS:
        per_seed = []
        feature_keys = [
            "camera_translation_m_per_frame", "camera_angular_deg_per_frame",
            "new_surface_ratio_voxel_5cm", "depth_valid_ratio_sampled",
            "growth_pressure",
        ]
        by_feature = {k: [] for k in feature_keys}
        age_keys = [
            "view_coverage_valid_depth",
            "young_lt_10_frames.visible_fraction_attributed",
            "mature_ge_10_frames.visible_fraction_attributed",
            "young_lt_10_frames.visible_depth_error_mean",
            "mature_ge_10_frames.visible_depth_error_mean",
            "young_lt_10_frames.visible_color_error_mean",
            "mature_ge_10_frames.visible_color_error_mean",
        ]
        by_age = {k: [] for k in age_keys}
        quotas = []
        update_gain = []
        for seed in SEEDS:
            ref_by_frame = {x["frame"]: x for x in data[(seed, "global_importance")]}
            trt_by_frame = {x["frame"]: x for x in data[(seed, "mature75")]}
            frames = [f for f in range(lo, hi + 1) if f in ref_by_frame and f in trt_by_frame]
            deltas = [trt_by_frame[f]["psnr"] - ref_by_frame[f]["psnr"] for f in frames]
            per_seed.append(mean(deltas))
            update_gain.append(mean([trt_by_frame[f]["psnr_post"] - trt_by_frame[f]["psnr_pre"] for f in frames]))
            quotas.append(mean([
                float(trt_by_frame[f]["realized_mature_updates"] == 3750
                      and trt_by_frame[f]["realized_young_updates"] == 1250)
                for f in frames
            ]))
            for key in feature_keys:
                by_feature[key].append(mean([ref_by_frame[f][key] for f in frames]))
            for key in age_keys:
                parent, child = key.split(".", 1) if "." in key else (key, None)
                vals = []
                for f in frames:
                    diag = ref_by_frame[f]["age_stratified_diagnostics"]
                    vals.append(diag[parent] if child is None else diag[parent][child])
                by_age[key].append(mean(vals))
        rows.append({
            "frame_window": [lo, hi], "seed_clusters": len(SEEDS),
            "trajectory_psnr_delta_db_mean": mean(per_seed),
            "trajectory_psnr_delta_db_seed_bootstrap_95_ci": bootstrap_ci(per_seed, 501 + lo),
            "positive_seed_clusters": sum(x > 0 for x in per_seed),
            "treatment_mean_post_minus_pre_psnr_db": mean(update_gain),
            "treatment_exact_quota_compliance": mean(quotas),
            "global_importance_input_and_resource_features": {
                k: {"mean": mean(v), "seed_cluster_values": v} for k, v in by_feature.items()
            },
            "global_importance_age_diagnostics": {
                k: {"mean": mean(v), "seed_cluster_values": v} for k, v in by_age.items()
            },
            "seed_trajectory_psnr_deltas_db": dict(zip(map(str, SEEDS), per_seed)),
        })

    out = {
        "status": "exploratory diagnostics complete; no inferential feature selection",
        "scene": "tum_fr1_desk", "arms": list(ARMS), "seeds": list(SEEDS),
        "diagnostic_summary_source": "global_importance arm pre-action state; state trajectories diverge after interventions",
        "diagnostic_runs_are_not_merged_with_confirmatory_runs": True,
        "window_results": rows,
        "provenance": {
            "manifest_sha256": sha256(manifest_path),
            "runner_sha256": manifest["runner_sha256"],
            "source_revision": manifest["source_revision"],
            "source_file_sha256": manifest["source_file_sha256"],
            "record_sha256": hashes,
        },
        "limitations": [
            "Only one sequence is diagnosed; frame windows are repeated measures, not independent samples.",
            "Pre-action map diagnostics are policy-dependent after the arms diverge; global-importance arm is used as the reference trajectory.",
            "Concurrent changes are descriptive and do not identify causal mediation or validate a router.",
            "Instrumentation changes host work; timing from this rerun is excluded from systems claims.",
        ],
    }
    json_path = BASE / "diagnostics_analysis.json"
    json_path.write_text(json.dumps(out, indent=2) + "\n")
    lines = [
        "# TUM fr1 desk: exploratory quota-regression diagnostics", "",
        "Separate instrumented rerun; eight seeds per arm, 120 frames, 5,000 updates/frame. These rows are descriptive seed-cluster summaries, not independent frame samples. Age and map diagnostics are taken from the global-importance reference trajectory; branches diverge after their first intervention.", "",
        "| Frames | Mean paired PSNR delta (dB; 95% seed CI) | Positive seeds | Exact treatment quota compliance | Mature / young visible fraction | Mature / young color residual | Mature / young depth residual | Novel surface ratio | Camera translation m/frame | Angular rate deg/frame | Depth valid ratio | Growth pressure |",
        "|:--:|--:|:--:|--:|:--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        ci = r["trajectory_psnr_delta_db_seed_bootstrap_95_ci"]
        ag = r["global_importance_age_diagnostics"]
        ip = r["global_importance_input_and_resource_features"]
        lines.append(
            f"| {r['frame_window'][0]}–{r['frame_window'][1]} | {r['trajectory_psnr_delta_db_mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{r['positive_seed_clusters']}/8 | {r['treatment_exact_quota_compliance']:.1%} | "
            f"{ag['mature_ge_10_frames.visible_fraction_attributed']['mean']:.3f} / {ag['young_lt_10_frames.visible_fraction_attributed']['mean']:.3f} | "
            f"{ag['mature_ge_10_frames.visible_color_error_mean']['mean']:.3f} / {ag['young_lt_10_frames.visible_color_error_mean']['mean']:.3f} | "
            f"{ag['mature_ge_10_frames.visible_depth_error_mean']['mean']:.3f} / {ag['young_lt_10_frames.visible_depth_error_mean']['mean']:.3f} | "
            f"{ip['new_surface_ratio_voxel_5cm']['mean']:.3f} | {ip['camera_translation_m_per_frame']['mean']:.4f} | "
            f"{ip['camera_angular_deg_per_frame']['mean']:.2f} | {ip['depth_valid_ratio_sampled']['mean']:.3f} | "
            f"{ip['growth_pressure']['mean']:.4f} |"
        )
    lines += [
        "", "## Reading", "",
        "The main study's sign change reappears in this separate rerun: quota-75% gains are positive through frame 60 and negative afterward. The change coincides with lower novel-surface ratio, higher mature visibility, and higher mature color residual in frames 61–90; young depth residual remains larger than mature depth residual. These concurrent patterns do not identify what caused the sign change and do not justify a threshold rule.", "",
        "Pre-action state is policy-dependent once branches diverge. This report uses the global-importance branch as a reference and does not pool the two arm states. The diagnostic instrumentation adds host work; its timing is not a systems benchmark.", "",
        "Seed-level values, hashes, and limitations are in `diagnostics_analysis.json`.", "",
    ]
    report_path = BASE / "diagnostics_report.md"
    report_path.write_text("\n".join(lines))
    print(report_path)


if __name__ == "__main__":
    main()
