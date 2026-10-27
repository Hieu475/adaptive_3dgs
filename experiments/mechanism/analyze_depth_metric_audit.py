#!/usr/bin/env python3
"""Summarize GT-valid/overlap depth errors and timing from metric-v2 replays."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize_view(npz_path):
    with np.load(npz_path) as data:
        gt = data["depth_ground_truth"].astype(np.float64)
        pred = data["depth_prediction"].astype(np.float64)
        gt_valid = data["gt_valid_mask"].astype(bool)
        support = data["pred_depth_coverage_mask"].astype(bool)
        alpha = data["render_alpha"].astype(np.float64)
        accumulated = data["depth_accumulated"].astype(np.float64)
    overlap = gt_valid & support
    err_gt = np.abs(pred[gt_valid] - gt[gt_valid])
    err_overlap = np.abs(pred[overlap] - gt[overlap])
    if overlap.any():
        target, estimate = gt[overlap], pred[overlap]
        corr = float(np.corrcoef(target, estimate)[0, 1]) if (
            np.std(target) > 0 and np.std(estimate) > 0) else None
        mean_alpha_overlap = float(alpha[overlap].mean())
        accum_to_expected_median = float(np.median(
            np.abs(accumulated[overlap] - estimate * alpha[overlap])))
        gt_median = float(np.median(target))
        pred_median = float(np.median(estimate))
    else:
        corr = mean_alpha_overlap = accum_to_expected_median = gt_median = pred_median = None
    return {
        "image": str(npz_path),
        "valid_gt_pixels": int(gt_valid.sum()),
        "overlap_pixels": int(overlap.sum()),
        "pred_depth_coverage": float(overlap.sum() / max(gt_valid.sum(), 1)),
        "depth_l1_gt_valid_m": float(err_gt.mean()) if err_gt.size else None,
        "depth_l1_overlap_m": float(err_overlap.mean()) if err_overlap.size else None,
        "depth_rmse_overlap_m": float(np.sqrt(np.square(pred[overlap] - gt[overlap]).mean())) if overlap.any() else None,
        "median_gt_depth_overlap_m": gt_median,
        "median_pred_depth_overlap_m": pred_median,
        "pearson_depth_overlap": corr,
        "mean_alpha_on_overlap": mean_alpha_overlap,
        "accumulated_vs_expected_identity_median_abs_error": accum_to_expected_median,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--visualization-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    scene, seed = run["scene"], int(run["seed"])
    arms = []
    for arm in run["arms"]:
        name = arm["arm"]["name"]
        views = []
        for frame in arm["fixed_heldout_views"]:
            npz_path = (args.visualization_root / scene / f"seed{seed}" / name
                        / f"frame_{int(frame['frame']):04d}.npz")
            if not npz_path.is_file():
                raise FileNotFoundError(npz_path)
            views.append({"frame": int(frame["frame"]), **summarize_view(npz_path)})
        arms.append({
            "name": name,
            "logical_micro_updates": arm["total_micro_updates"],
            "time_opt_cuda_ms_mean": arm["mean_time_opt_cuda_ms"],
            "time_opt_cuda_ms_total": arm["total_time_opt_cuda_ms"],
            "time_opt_stage_wall_ms_mean": arm["mean_opt_stage_sync_ms"],
            "time_frame_wall_ms_mean": arm["mean_time_frame_wall_ms"],
            "modeled_optimization_budget_ms": run["protocol"]["budget_ms_modeled"],
            "requested_stale_fraction": run["protocol"]["stale_rescue"]["fraction"],
            "realized_stale_update_fraction": arm["realized_stale_update_fraction"],
            "reported_metric_means": {
                key: arm[key] for key in (
                    "fixed_heldout_mean_psnr", "fixed_heldout_mean_ssim",
                    "fixed_heldout_mean_ssim_legacy", "fixed_heldout_mean_depth_l1",
                    "fixed_heldout_mean_depth_l1_overlap",
                    "fixed_heldout_mean_pred_depth_coverage")
            },
            "views": views,
        })
    report = {
        "schema": "depth-metric-v2-audit-v1",
        "run": str(args.run),
        "scene": scene,
        "seed": seed,
        "metric_protocol": run.get("metric_protocol"),
        "pose_convention": run["protocol"].get("pose_convention",
            "Phase 10 sequence loader ground-truth W2C poses"),
        "depth_unit": "meters, as loaded from Replica depth.npy",
        "prediction_support_alpha_threshold": run["protocol"]["prediction_support_rule"]["alpha_threshold"],
        "arms": arms,
    }
    out_dir = args.out_dir or args.run.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "depth_metric_audit.json"
    md_path = out_dir / "depth_metric_audit.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Depth metric v2 audit", "",
        f"Scene `{scene}`, seed {seed}. GT depth is in meters. Prediction support uses alpha ≥ {report['prediction_support_alpha_threshold']:.2f}.",
        "",
        "## Arm summary", "",
        "| Arm | Updates | CUDA optimization stage mean/total (ms) | Stage wall mean (ms) | Frame wall mean (ms) | GT-valid L1 (m) | Overlap L1 (m) | Coverage | Stale share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        q = arm["reported_metric_means"]
        lines.append(
            f"| {arm['name']} | {arm['logical_micro_updates']} | "
            f"{arm['time_opt_cuda_ms_mean']:.3f}/{arm['time_opt_cuda_ms_total']:.1f} | "
            f"{arm['time_opt_stage_wall_ms_mean']:.3f} | {arm['time_frame_wall_ms_mean']:.3f} | "
            f"{q['fixed_heldout_mean_depth_l1']:.4f} | "
            f"{q['fixed_heldout_mean_depth_l1_overlap']:.4f} | "
            f"{q['fixed_heldout_mean_pred_depth_coverage']:.3f} | "
            f"{arm['realized_stale_update_fraction']:.3f} (requested {arm['requested_stale_fraction']:.2f}) |"
        )
    lines += ["", "## Held-out views", "",
              "Per-view pixel counts, coverage, depth correlation, alpha, and accumulated-to-expected identity checks are in `depth_metric_audit.json`.",
              "", "These runs are a one-seed substrate audit, not confirmatory evidence. `ssim_standard` uses torchmetrics spatial windows; `ssim_legacy` preserves the previous global-statistics score for comparison."]
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
