#!/usr/bin/env python3
"""Summarize fixed-decision direct/separable SSIM checkpoint rollouts.

The unit of summary is a checkpoint: repeats are averaged within implementation,
then implementation deltas are computed per frame and at the final frame. This
script deliberately reports descriptive results only; adjacent frames are not
treated as independent statistical samples.
"""
import argparse
import json
import statistics
from pathlib import Path


METRICS = ("psnr", "ssim", "depth_l1")
CHECKPOINTS = (
    ("replica_room1", 279, "ssim_paired_rollout_clean_checkout_seed279"),
    ("replica_room1", 281, "ssim_paired_rollout_clean_checkout_seed281"),
    ("replica_office1", 264, "ssim_paired_rollout_clean_checkout_office1_seed264"),
    ("replica_office1", 265, "ssim_paired_rollout_clean_checkout_office1_seed265"),
    ("replica_office1", 266, "ssim_paired_rollout_clean_checkout_office1_seed266"),
)


def mean(values):
    return statistics.mean(values) if values else 0.0


def summarize_checkpoint(path, scene, seed):
    data = json.loads(path.read_text())
    if not data.get("fixed_decision_gates_passed"):
        raise ValueError(f"Fixed-decision gates failed or absent: {path}")
    if data["scene"] != scene or int(data["seed"]) != seed:
        raise ValueError(f"Scene/seed mismatch in {path}")
    if not data.get("spawn_and_update_decisions_frozen"):
        raise ValueError(f"Spawn/update decisions were not frozen: {path}")
    if not data.get("scheduler_profiling_feedback_frozen"):
        raise ValueError(f"Scheduler profiling feedback was not frozen: {path}")
    branches = data["branches"]
    by_impl = {
        impl: [branch for branch in branches if branch["implementation"] == impl]
        for impl in ("direct", "separable")
    }
    if any(len(items) != 3 for items in by_impl.values()):
        raise ValueError(f"Expected 3 repeats per implementation: {path}")
    frames = list(range(data["rollout_start_frame"], data["rollout_end_frame"] + 1))
    trajectory = []
    for frame in frames:
        row = {"frame": frame}
        for metric in METRICS:
            values = {
                impl: [next(r[metric] for r in b["frames"] if r["frame"] == frame)
                       for b in by_impl[impl]]
                for impl in by_impl
            }
            row[f"direct_{metric}"] = mean(values["direct"])
            row[f"separable_{metric}"] = mean(values["separable"])
            row[f"delta_{metric}"] = row[f"separable_{metric}"] - row[f"direct_{metric}"]
        trajectory.append(row)
    final = trajectory[-1]
    result = {
        "scene": scene,
        "seed": seed,
        "artifact": str(path),
        "source_commit": data["git_commit"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "snapshot_map_sha256": data["snapshot_map_sha256"],
        "schedule_sha256": data["decision_schedule_sha256"],
        "frames": len(frames),
        "frame_range": [frames[0], frames[-1]],
        "map_counts": sorted({int(r["n_gaussians"]) for b in branches for r in b["frames"]}),
        "micro_updates": sorted({int(r["n_micro_updates"]) for b in branches for r in b["frames"]}),
        "birth_counts": sorted({int(r["n_spawned"]) for b in branches for r in b["frames"]}),
        "final_delta": {metric: final[f"delta_{metric}"] for metric in METRICS},
        "mean_frame_delta": {
            metric: mean([r[f"delta_{metric}"] for r in trajectory]) for metric in METRICS
        },
        "mean_absolute_frame_delta": {
            metric: mean([abs(r[f"delta_{metric}"]) for r in trajectory]) for metric in METRICS
        },
        "final_repeat_sd": {
            impl: {
                metric: (statistics.stdev([
                    next(r[metric] for r in b["frames"] if r["frame"] == frames[-1])
                    for b in by_impl[impl]
                ]) if len(by_impl[impl]) > 1 else 0.0)
                for metric in METRICS
            }
            for impl in by_impl
        },
        "trajectory": trajectory,
    }
    if result["birth_counts"] != [0] or len(result["map_counts"]) != 1 or len(result["micro_updates"]) != 1:
        raise ValueError(f"Map/budget invariants failed in {path}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path,
                        default=Path("results/phase14_generalization_postfix"))
    parser.add_argument("--out-json", type=Path,
                        default=Path("results/phase14_generalization_postfix/ssim_locked_rollout_analysis.json"))
    parser.add_argument("--out-md", type=Path,
                        default=Path("results/phase14_generalization_postfix/ssim_locked_rollout_analysis.md"))
    args = parser.parse_args()
    checkpoints = [
        summarize_checkpoint(args.results_root / folder / "paired_rollout.json", scene, seed)
        for scene, seed, folder in CHECKPOINTS
    ]
    scenes = sorted({item["scene"] for item in checkpoints})
    scene_summary = {}
    for scene in scenes:
        cohort = [item for item in checkpoints if item["scene"] == scene]
        scene_summary[scene] = {
            "checkpoint_count": len(cohort),
            "mean_final_delta": {
                metric: mean([item["final_delta"][metric] for item in cohort])
                for metric in METRICS
            },
            "positive_negative_zero_psnr_deltas": {
                "positive": sum(item["final_delta"]["psnr"] > 0 for item in cohort),
                "negative": sum(item["final_delta"]["psnr"] < 0 for item in cohort),
                "zero": sum(item["final_delta"]["psnr"] == 0 for item in cohort),
            },
        }
    overall = {
        "checkpoint_count": len(checkpoints),
        "mean_final_delta": {
            metric: mean([item["final_delta"][metric] for item in checkpoints])
            for metric in METRICS
        },
        "psnr_sign_counts": {
            "positive": sum(item["final_delta"]["psnr"] > 0 for item in checkpoints),
            "negative": sum(item["final_delta"]["psnr"] < 0 for item in checkpoints),
            "zero": sum(item["final_delta"]["psnr"] == 0 for item in checkpoints),
        },
        "mean_absolute_per_frame_delta": {
            metric: mean([item["mean_absolute_frame_delta"][metric] for item in checkpoints])
            for metric in METRICS
        },
    }
    output = {
        "schema": "ssim-fixed-decision-rollout-analysis-v1",
        "contrast": "separable minus direct",
        "statistical_note": "Descriptive checkpoint-level summaries; repeated frames are not independent samples.",
        "overall": overall,
        "by_scene": scene_summary,
        "checkpoints": checkpoints,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# Locked-decision SSIM rollout summary",
        "",
        "Deltas are separable minus direct. Repeats are averaged within each implementation and checkpoint. Frame observations are summarized descriptively and are not treated as independent samples.",
        "",
        "| Scene / seed | Frames | ΔPSNR (dB) | ΔSSIM | ΔDepth L1 (m) | Direct repeat SD, PSNR | Separable repeat SD, PSNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in checkpoints:
        lines.append(
            f"| {item['scene']} / {item['seed']} | {item['frames']} | "
            f"{item['final_delta']['psnr']:+.7f} | {item['final_delta']['ssim']:+.7f} | "
            f"{item['final_delta']['depth_l1']:+.8f} | "
            f"{item['final_repeat_sd']['direct']['psnr']:.7f} | "
            f"{item['final_repeat_sd']['separable']['psnr']:.7f} |"
        )
    lines += ["", "## Scene-level summaries", "",
              "| Scene | Checkpoints | Mean final ΔPSNR (dB) | PSNR sign counts (+/−) |",
              "|---|---:|---:|---:|"]
    for scene, item in scene_summary.items():
        signs = item["positive_negative_zero_psnr_deltas"]
        lines.append(f"| {scene} | {item['checkpoint_count']} | {item['mean_final_delta']['psnr']:+.7f} | {signs['positive']}/{signs['negative']} |")
    signs = overall["psnr_sign_counts"]
    lines += [
        "",
        f"Across all five checkpoints, mean final deltas were PSNR {overall['mean_final_delta']['psnr']:+.7f} dB, SSIM {overall['mean_final_delta']['ssim']:+.7f}, and Depth L1 {overall['mean_final_delta']['depth_l1']:+.8f} m. PSNR was positive in {signs['positive']} checkpoints and negative in {signs['negative']}.",
        "",
        "The PSNR sign differs by scene and checkpoint. The shifts are small relative to repeat variation in several checkpoints; this descriptive cohort does not establish a quality advantage for either implementation. Do not use frame count as an inferential sample size.",
        "",
        "All included runs passed fixed-decision gates, with zero births, fixed Gaussian roster per checkpoint, and 5,000 allocated micro-updates per frame.",
    ]
    args.out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md),
                      "overall": overall, "by_scene": scene_summary}, indent=2))


if __name__ == "__main__":
    main()
