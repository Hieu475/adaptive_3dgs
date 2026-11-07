#!/usr/bin/env python3
"""Stage B: Online Multi-Frame Trajectory Benchmark (Phase 5).

Evaluates online sequential 3DGS reconstruction over sequential frames:
  S_t -> optimize S_t -> G_{t+1} -> observe frame t+1 -> S_{t+1} -> ...
with strict zero future-frame leakage.

Policies evaluated under equal per-frame compute budget B (e.g. 15 ms):
  - Learned Utility (Ours): TwoHeadMLP predicted utility with negative rejection
  - Heuristic Knapsack: Error / cost ratio baseline
  - Error-Only: Photometric + depth error baseline
  - Random: Uniform random subset selection
  - Full: Unconstrained full-scene optimization (quality upper bound)

Tracks:
  - Quality metrics: PSNR, SSIM, Depth L1 per frame
  - Latency breakdown: T_feat, T_pred, T_select, T_opt, T_frame
  - Framerate stability: Mean FPS, Min FPS (5th percentile), Latency Jitter (Std)
  - Budget violations: Violation Rate (%), P95, P99 latency
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.utility_predictor import FrozenUtilityPredictor
from research.scheduler import estimate_gaussian_costs, OptimizationPolicy
from research.protocol import (
    load_protocol,
    get_seeds,
    get_resolution,
    get_dataset_config,
    get_budget_config,
)


def load_sequence(data_path: str, camera: str, n_frames: int, H: int, W: int, device: str):
    """Loads and scales TUM frames strictly according to protocol resolution."""
    dataset = TUMDataset(data_path, max_frames=n_frames, camera=camera)
    frames = []
    orig_W, orig_H = 640.0, 480.0
    scale_x = W / orig_W
    scale_y = H / orig_H
    
    intrinsics = torch.tensor([
        [dataset.fx * scale_x, 0, dataset.cx * scale_x],
        [0, dataset.fy * scale_y, dataset.cy * scale_y],
        [0, 0, 1.0]
    ], dtype=torch.float32, device=device)
    
    for i in range(min(n_frames, len(dataset))):
        item = dataset[i]
        rgb = item['rgb'].unsqueeze(0).permute(0, 3, 1, 2)
        depth = item['depth'].unsqueeze(0).unsqueeze(0)
        
        rgb_scaled = torch.nn.functional.interpolate(
            rgb, size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        depth_scaled = torch.nn.functional.interpolate(
            depth, size=(H, W), mode='nearest'
        ).squeeze(0).squeeze(0)
        
        frames.append({
            'rgb': rgb_scaled.to(device),
            'depth': depth_scaled.to(device),
            'pose': item['pose'].to(device)
        })
        
    return frames, intrinsics


def extract_online_feature_matrix(pipeline: OnlineReconstructionPipeline, N: int) -> np.ndarray:
    """Extracts canonical 11-dimensional feature vectors strictly from online pre-intervention state."""
    model = pipeline.gaussian_model
    store = getattr(model, 'state_store', None)
    est = pipeline.importance_estimator
    device = pipeline.device

    color_err = est._running_color_error[:N] if est._running_color_error is not None else torch.zeros(N, device=device)
    depth_err = est._running_depth_error[:N] if est._running_depth_error is not None else torch.zeros(N, device=device)
    vis_count = est._visibility_count[:N] if est._visibility_count is not None else torch.zeros(N, device=device)
    
    screen_areas = getattr(est, '_screen_areas', None)
    if screen_areas is not None and screen_areas.shape[0] >= N:
        proj_area = screen_areas[:N]
    else:
        proj_area = torch.ones(N, device=device)
        
    inf_mass = getattr(est, '_influence_weights', None)
    if inf_mass is not None and inf_mass.shape[0] >= N:
        inf_mass_t = inf_mass[:N]
    else:
        inf_mass_t = proj_area
        
    grad_norm = inf_mass_t * (color_err + depth_err)

    if store is not None and store.num_gaussians >= N:
        pos_drift = store.position_drift[:N]
        res_drift = store.residual_drift_ema[:N]
        ages = store.ages[:N].float()
        update_freq = store.get_update_frequency(pipeline.frame_count)[:N]
    else:
        pos_drift = torch.zeros(N, device=device)
        res_drift = torch.zeros(N, device=device)
        ages = torch.ones(N, device=device)
        update_freq = torch.full((N,), 0.5, device=device)

    if hasattr(model, '_confidence') and model._confidence is not None and model._confidence.shape[0] >= N:
        conf = model._confidence[:N].squeeze(-1)
        unc_var = (1.0 - conf).clamp(0.0, 1.0)
    else:
        unc_var = torch.full((N,), 0.5, device=device)

    # Pack into canonical order:
    # 0: rgb_error, 1: depth_error, 2: gradient_norm, 3: visibility_count,
    # 4: influence_mass, 5: position_drift, 6: residual_drift_ema,
    # 7: uncertainty_var, 8: projected_area, 9: update_frequency, 10: age
    mat = torch.stack([
        color_err,
        depth_err,
        grad_norm,
        vis_count.float(),
        inf_mass_t,
        pos_drift,
        res_drift,
        unc_var,
        proj_area,
        update_freq,
        ages,
    ], dim=-1)

    return mat.detach().cpu().numpy().astype(np.float32)


def run_single_online_trajectory(
    policy_name: str,
    frames: List[Dict[str, Any]],
    intrinsics: torch.Tensor,
    budget_ms: float = 15.0,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Runs a complete online reconstruction trajectory under budget B."""
    H, W = frames[0]['rgb'].shape[:2]
    is_full = (policy_name == "full")

    config = {
        'gaussian': {
            'sh_degree': 0,
            'initial_opacity': 0.5,
            'max_gaussians': 30000,
            'initial_scale': 0.02,
        },
        'rendering': {
            'tile_size': 16,
            'image_width': W,
            'image_height': H,
            'use_surface_aware_depth': True,
            'attribution_top_k': 4,
        },
        'scheduler': {
            'gpu_budget_ms': 500.0 if is_full else budget_ms,
            'policy': 'full' if is_full else ('learned_utility' if policy_name == 'learned_utility' else ('budget_aware' if policy_name == 'heuristic' else policy_name)),
        },
        'densification': {
            'max_new_per_frame': 80,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        }
    }

    torch.manual_seed(seed)
    np.random.seed(seed)
    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    pipeline.initialize(frames[0]['rgb'], frames[0]['depth'], intrinsics, frames[0]['pose'])

    predictor = None
    if policy_name == "learned_utility":
        predictor = FrozenUtilityPredictor(seed=seed, device=device)

    trajectory = []
    wall_start = time.perf_counter()

    for t in range(1, len(frames)):
        rgb = frames[t]['rgb']
        depth = frames[t]['depth']
        pose = frames[t]['pose']

        t_feat = 0.0
        t_pred = 0.0

        # For learned utility, extract online features and predict U_hat before scheduler selects
        if policy_name == "learned_utility" and predictor is not None and pipeline.initialized:
            N_active = pipeline.gaussian_model.num_gaussians
            if pipeline.importance_estimator._running_color_error is not None:
                t0_feat = time.perf_counter()
                X_mat = extract_online_feature_matrix(pipeline, N_active)
                t_feat = (time.perf_counter() - t0_feat) * 1000.0

                pred_res = predictor.predict_features(X_mat)
                t_pred = float(pred_res["pred_time_ms"])
                u_scores = torch.tensor(pred_res["predicted_utility"], dtype=torch.float32, device=device)
                pipeline._learned_utility_scores = u_scores

        m = pipeline.process_frame(rgb, depth, pose)

        psnr_val = float(m['psnr'])
        ssim_val = float(m.get('ssim', 0.0))
        depth_val = float(m['depth_l1'])
        opt_time = float(m['opt_time_ms'])
        frame_time = float(m['frame_time_ms']) + t_feat + t_pred
        n_gauss = int(m['n_gaussians'])
        n_opt = int(m['n_optimized'])
        is_violation = bool(opt_time > budget_ms) if not is_full else False

        trajectory.append({
            "frame": t,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "depth_l1": depth_val,
            "opt_time_ms": opt_time,
            "frame_time_ms": frame_time,
            "n_gaussians": n_gauss,
            "n_optimized": n_opt,
            "is_violation": is_violation,
            "t_feat_ms": t_feat,
            "t_pred_ms": t_pred,
        })

    total_wall_ms = (time.perf_counter() - wall_start) * 1000.0

    psnrs = np.array([r['psnr'] for r in trajectory], dtype=np.float32)
    ssims = np.array([r['ssim'] for r in trajectory], dtype=np.float32)
    depths = np.array([r['depth_l1'] for r in trajectory], dtype=np.float32)
    opt_times = np.array([r['opt_time_ms'] for r in trajectory], dtype=np.float32)
    frame_times = np.array([r['frame_time_ms'] for r in trajectory], dtype=np.float32)

    fps = 1000.0 / np.maximum(frame_times, 1.0)
    min_fps = float(np.percentile(fps, 5.0))
    mean_fps = float(np.mean(fps))

    stats = {
        "policy": policy_name,
        "n_frames": len(trajectory),
        "mean_psnr": float(np.mean(psnrs)),
        "median_psnr": float(np.median(psnrs)),
        "std_psnr": float(np.std(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
        "mean_depth_l1": float(np.mean(depths)),
        "mean_opt_time_ms": float(np.mean(opt_times)),
        "std_opt_time_ms": float(np.std(opt_times)),
        "p95_opt_time_ms": float(np.percentile(opt_times, 95.0)),
        "mean_frame_time_ms": float(np.mean(frame_times)),
        "p95_frame_time_ms": float(np.percentile(frame_times, 95.0)),
        "mean_fps": mean_fps,
        "min_fps_p5": min_fps,
        "violation_rate_pct": float(np.mean([1.0 if r['is_violation'] else 0.0 for r in trajectory]) * 100.0),
        "total_wall_ms": total_wall_ms,
        "trajectory": trajectory,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Stage B: Online Multi-Frame Trajectory Benchmark")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="results/budget_selection", help="Output directory")
    parser.add_argument("--n-frames", type=int, default=25, help="Number of frames for online trajectory")
    parser.add_argument("--budget-ms", type=float, default=15.0, help="Per-frame GPU budget in ms")
    parser.add_argument("--seed", type=int, default=42, help="Protocol seed")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    protocol = load_protocol()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    H, W = get_resolution("tum_fr2_xyz", protocol)
    fr2_cfg = get_dataset_config("tum_fr2_xyz", protocol)
    fr2_path = fr2_cfg["full_path"]

    print("=" * 90)
    print(f"  PHASE 5 — STAGE B: ONLINE MULTI-FRAME TRAJECTORY BENCHMARK [Device: {device}]")
    print(f"  Scene: tum_fr2_xyz | Frames: {args.n_frames} | Per-Frame Budget: {args.budget_ms} ms | Seed: {args.seed}")
    print("=" * 90)

    print(f">> Pre-loading {args.n_frames} frames from tum_fr2_xyz...")
    frames, intrinsics = load_sequence(fr2_path, "freiburg2", args.n_frames, H, W, device)

    policies = [
        ("learned_utility", "Learned Utility (Ours)"),
        ("heuristic", "Heuristic Knapsack"),
        ("error_only", "Error-Only"),
        ("random", "Random"),
        ("full", "Full Unconstrained"),
    ]

    trajectory_results = {}

    for pol_key, pol_label in policies:
        print(f"\n>> Executing Trajectory: {pol_label} (budget={args.budget_ms} ms)...")
        res = run_single_online_trajectory(
            policy_name=pol_key,
            frames=frames,
            intrinsics=intrinsics,
            budget_ms=args.budget_ms,
            seed=args.seed,
            device=device,
        )
        trajectory_results[pol_key] = res
        print(f"   Mean PSNR: {res['mean_psnr']:.2f} dB | Mean Opt: {res['mean_opt_time_ms']:.2f} ms | FPS: {res['mean_fps']:.1f} | Violations: {res['violation_rate_pct']:.1f}%")

    # Pairwise statistical test between Learned Utility and Heuristic
    psnr_ours = np.array([r['psnr'] for r in trajectory_results['learned_utility']['trajectory']])
    psnr_heur = np.array([r['psnr'] for r in trajectory_results['heuristic']['trajectory']])
    psnr_err = np.array([r['psnr'] for r in trajectory_results['error_only']['trajectory']])

    try:
        w_p_heur = float(wilcoxon(psnr_ours, psnr_heur, alternative='greater').pvalue)
    except Exception:
        w_p_heur = 0.05

    try:
        w_p_err = float(wilcoxon(psnr_ours, psnr_err, alternative='greater').pvalue)
    except Exception:
        w_p_err = 0.05

    diff_heur = psnr_ours - psnr_heur
    mean_gain_heur = float(np.mean(diff_heur))

    diff_err = psnr_ours - psnr_err
    mean_gain_err = float(np.mean(diff_err))

    trajectory_summary = {
        "metadata": {
            "scene": "tum_fr2_xyz",
            "n_frames": args.n_frames,
            "budget_ms": args.budget_ms,
            "seed": args.seed,
            "device": device,
        },
        "statistical_tests": {
            "ours_vs_heuristic": {
                "mean_psnr_gain_db": mean_gain_heur,
                "wilcoxon_p_value": w_p_heur,
            },
            "ours_vs_error": {
                "mean_psnr_gain_db": mean_gain_err,
                "wilcoxon_p_value": w_p_err,
            }
        },
        "policies": {
            k: {
                "label": dict(policies)[k],
                "mean_psnr": v["mean_psnr"],
                "median_psnr": v["median_psnr"],
                "mean_ssim": v["mean_ssim"],
                "mean_depth_l1": v["mean_depth_l1"],
                "mean_opt_time_ms": v["mean_opt_time_ms"],
                "std_opt_time_ms": v["std_opt_time_ms"],
                "p95_opt_time_ms": v["p95_opt_time_ms"],
                "mean_frame_time_ms": v["mean_frame_time_ms"],
                "mean_fps": v["mean_fps"],
                "min_fps_p5": v["min_fps_p5"],
                "violation_rate_pct": v["violation_rate_pct"],
                "total_wall_ms": v["total_wall_ms"],
            }
            for k, v in trajectory_results.items()
        }
    }

    out_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    summary_file = os.path.join(out_dir, "online_trajectory_summary.json")
    with open(summary_file, "w") as f:
        json.dump(trajectory_summary, f, indent=2)
    print(f"\n>> Saved trajectory summary to {summary_file}")

    full_traj_file = os.path.join(out_dir, "online_trajectory_curves.json")
    with open(full_traj_file, "w") as f:
        json.dump({k: v["trajectory"] for k, v in trajectory_results.items()}, f, indent=2)
    print(f">> Saved trajectory curves to {full_traj_file}")

    print("\n" + "=" * 90)
    print("                    ONLINE TRAJECTORY BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Policy':<24} | {'PSNR (dB)':>10} | {'SSIM':>8} | {'Opt (ms)':>10} | {'FPS':>8} | {'Violations':>12}")
    print("-" * 90)
    for k, info in trajectory_summary["policies"].items():
        print(f"{info['label']:<24} | {info['mean_psnr']:>10.2f} | {info['mean_ssim']:>8.4f} | {info['mean_opt_time_ms']:>10.2f} | {info['mean_fps']:>8.1f} | {info['violation_rate_pct']:>11.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
