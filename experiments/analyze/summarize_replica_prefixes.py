#!/usr/bin/env python3
"""Summarize matched policy results across short Replica prefixes and two seeds."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = [
    ("office0", 100, ROOT / "results/replica/long_horizon_replica_500/run1"),
    ("office1", 120, ROOT / "results/replica/replica_office1_120/run1"),
    ("office2", 120, ROOT / "results/replica/replica_office2_120/run1"),
    ("office3", 120, ROOT / "results/replica/replica_office3_120/run1"),
    ("office4", 120, ROOT / "results/replica/replica_office4_120/run1"),
    ("room0", 100, ROOT / "results/replica/replica_room0_120/run1"),
    ("room1", 120, ROOT / "results/replica/replica_room1_120/run1"),
    ("room2", 120, ROOT / "results/replica/replica_room2_120/run1"),
]


def load_result(run_dir, expected_frames):
    results = json.loads((run_dir / "long_horizon_results.json").read_text())
    policies = {p["policy"]: p for p in results["policies"]}
    diagnostics = json.loads((run_dir / "long_horizon_diagnostics.json").read_text())
    return {
        "sequence_frames": results["metadata"]["n_frames"],
        "expected_frames": expected_frames,
        "ours": policies["ours"],
        "error_influence": policies["error_influence"],
        "completion_and_finite_gate": diagnostics["all_completion_and_finite_metric_gates_pass"],
        "frame_count_matches_available": results["metadata"]["n_frames"] == expected_frames,
    }


def main():
    rows = []
    for scene, expected_frames, seed43_dir in RUNS:
        seed42_dir = ROOT / "results" / "replica" / "replica_seed42" / scene / "run1"
        rows.append({
            "scene": scene,
            "sequence_frames": expected_frames,
            "seed42": load_result(seed42_dir, expected_frames),
            "seed43": load_result(seed43_dir, expected_frames),
            "scope": "short prefix; not a 500-frame run",
        })

    metric_keys = ("mean_psnr", "mean_ssim", "mean_lpips", "mean_depth_l1",
                   "fps", "p95_frame_time_ms", "N_final", "peak_vram_mb")
    paired = []
    for row in rows:
        for seed in ("seed42", "seed43"):
            a, b = row[seed]["ours"], row[seed]["error_influence"]
            paired.append({"scene": row["scene"], "seed": int(seed[-2:]), **{
                k: float(a[k]) - float(b[k]) for k in metric_keys if k != "N_final"
            }, "N_final_ours_minus_error_influence": int(a["N_final"]) - int(b["N_final"])})

    def win_count(items, key, lower_is_better=False):
        return sum((d[key] < 0 if lower_is_better else d[key] > 0) for d in items)

    def count_metrics(items):
        return {
            "mean_psnr": win_count(items, "mean_psnr"),
            "mean_ssim": win_count(items, "mean_ssim"),
            "mean_lpips": win_count(items, "mean_lpips", True),
            "mean_depth_l1": win_count(items, "mean_depth_l1", True),
            "p95_frame_time_ms": win_count(items, "p95_frame_time_ms", True),
            "final_map_size": win_count(items, "N_final_ours_minus_error_influence", True),
        }

    paired_by_seed = {
        seed: [d for d in paired if d["seed"] == seed]
        for seed in (42, 43)
    }
    aggregate_means = {
        key: sum(d[key] for d in paired) / len(paired)
        for key in ("mean_psnr", "mean_ssim", "mean_lpips", "mean_depth_l1",
                    "p95_frame_time_ms")
    }
    aggregate_means["mean_final_gaussian_reduction_fraction"] = sum(
        1.0 - row[seed]["ours"]["N_final"] / max(row[seed]["error_influence"]["N_final"], 1)
        for row in rows for seed in ("seed42", "seed43")
    ) / (2 * len(rows))

    summary = {
        "title": "Replica short-prefix cross-scene diagnostics across two seeds",
        "seeds": [42, 43],
        "scenes": rows,
        "paired_differences_ours_minus_error_influence": paired,
        "scene_win_counts_for_ours": {
            "per_seed": {seed: count_metrics(items) for seed, items in paired_by_seed.items()},
            "all_scene_seed_pairs": count_metrics(paired),
        },
        "mean_paired_difference_ours_minus_error_influence": aggregate_means,
        "interpretation": "Descriptive paired comparisons over short prefixes on all eight available Replica scenes and two seeds; not a 500-frame gate and not sufficient for a broad generalization claim.",
    }

    n_scenes = len(rows)
    lines = [
        "# Replica cross-scene summary across two seeds", "",
        "Seeds 42 and 43; ground-truth poses; 320x240; matched policy configs. Every scene uses its complete available prefix (100–120 frames), so none satisfies a 500-frame gate.", "",
        "| Scene | Frames | Gates (42/43) | Δ PSNR seed42 / seed43 | Δ depth L1 seed42 / seed43 | Map reduction % seed42 / seed43 |",
        "|:--|--:|:--:|--:|--:|--:|",
    ]
    for row in rows:
        ds = {d["seed"]: d for d in paired if d["scene"] == row["scene"]}
        gates = [all(row[seed][k] for k in ("completion_and_finite_gate", "frame_count_matches_available"))
                 for seed in ("seed42", "seed43")]
        gate_text = "PASS/PASS" if all(gates) else "/".join("PASS" if g else "FAIL" for g in gates)
        reductions = [100.0 * (1.0 - row[seed]["ours"]["N_final"] /
                      max(row[seed]["error_influence"]["N_final"], 1)) for seed in ("seed42", "seed43")]
        lines.append(
            f"| {row['scene']} | {row['sequence_frames']} | {gate_text} | {ds[42]['mean_psnr']:+.3f} / {ds[43]['mean_psnr']:+.3f} | "
            f"{ds[42]['mean_depth_l1']:+.5f} / {ds[43]['mean_depth_l1']:+.5f} | {reductions[0]:.1f}% / {reductions[1]:.1f}% |"
        )
    wins = summary["scene_win_counts_for_ours"]["all_scene_seed_pairs"]
    lines += ["", f"## Wins across {n_scenes} scenes × 2 seeds", "",
              f"`ours` wins: PSNR {wins['mean_psnr']}/{2*n_scenes}, SSIM {wins['mean_ssim']}/{2*n_scenes}, LPIPS {wins['mean_lpips']}/{2*n_scenes}, depth L1 {wins['mean_depth_l1']}/{2*n_scenes}, p95 latency {wins['p95_frame_time_ms']}/{2*n_scenes}, smaller final map {wins['final_map_size']}/{2*n_scenes}.", "",
              "The seed-level paired results are in summary.json. These comparisons use one run per scene and seed, with unequal prefix lengths (office0 and room0 have 100 frames; the other scenes have 120). They do not quantify statistical uncertainty or establish broad generalization.", ""]
    out_dir = ROOT / "results/replica/replica_cross_scene_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "summary.md").write_text("\n".join(lines))
    print(out_dir / "summary.md")


if __name__ == "__main__":
    main()
