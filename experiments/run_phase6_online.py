#!/usr/bin/env python3
"""Phase 6: Online Reconstruction Trajectory Benchmark (RQ6).

Compares online trajectory of competing policies under equal compute budget:
  - NO_OP
  - RANDOM
  - ERROR_INFLUENCE
  - HEURISTIC
  - PHASE4_LEARNED
  - PHASE6_ADAPTIVE (OURS)

Tracks:
  - PSNR (pre and post optimization)
  - SSIM
  - Depth L1
  - Optimization latency per frame (ms)
  - Cumulative compute time (ms)
  - Quality per compute (PSNR / cumulative seconds)
  - Selection churn between consecutive frames
  - Budget violation rate (%)

Usage:
    python experiments/run_phase6_online.py --frames 10 --budget 15.0
    python experiments/run_phase6_online.py --frames 20 --budget 15.0
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.phase6_context import ContextConfig
from research.phase6_model import FrozenContextPredictor
from research.phase6_selection import (
    select_phase6_subset,
    map_candidate_to_active_index,
    Phase6PolicyName,
)
from research.utility_predictor import FrozenUtilityPredictor
from research.scheduler_metrics import compute_extended_churn
from research.protocol import (
    load_protocol,
    get_resolution,
    get_dataset_config,
)


def load_tum_sequence(data_path: str, camera: str, n_frames: int, H: int, W: int, device: str):
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


def build_reconstruction_pipeline(H: int, W: int, device: str) -> OnlineReconstructionPipeline:
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
            'gpu_budget_ms': 15.0,
            'policy': 'budget_aware',
        },
        'densification': {
            'max_new_per_frame': 80,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        }
    }
    return OnlineReconstructionPipeline(config=config, device=device)


def extract_online_canonical_features(pipeline: OnlineReconstructionPipeline, N: int) -> np.ndarray:
    """Extract canonical 11 features from pipeline state for N Gaussians."""
    device = pipeline.device
    est = pipeline.importance_estimator
    model = pipeline.gaussian_model

    color_err = est._running_color_error[:N] if est._running_color_error is not None else torch.zeros(N, device=device)
    depth_err = est._running_depth_error[:N] if est._running_depth_error is not None else torch.zeros(N, device=device)
    inf_mass = getattr(est, '_influence_weights', None)
    inf_mass_t = inf_mass[:N] if inf_mass is not None and inf_mass.shape[0] >= N else torch.ones(N, device=device)

    vis_count = getattr(est, '_pixel_counts', None)
    vis_count = vis_count[:N] if vis_count is not None and vis_count.shape[0] >= N else torch.zeros(N, device=device)
    grad_norm = (color_err + depth_err) * inf_mass_t

    store = getattr(model, 'state_store', None)
    if store is not None:
        pos_drift = store.position_drift[:N] if len(store.position_drift) >= N else torch.zeros(N, device=device)
        res_drift = store.residual_drift_ema[:N] if len(store.residual_drift_ema) >= N else torch.zeros(N, device=device)
        unc_var = store.uncertainty[:N] if len(store.uncertainty) >= N else torch.full((N,), 0.5, device=device)
        ages = store.ages[:N].float() if len(store.ages) >= N else torch.ones(N, device=device)
        up_counts = store.update_counts[:N].float() if len(store.update_counts) >= N else torch.zeros(N, device=device)
        update_freq = up_counts / torch.clamp(ages, min=1.0)
    else:
        pos_drift = torch.zeros(N, device=device)
        res_drift = torch.zeros(N, device=device)
        unc_var = torch.full((N,), 0.5, device=device)
        ages = torch.ones(N, device=device)
        update_freq = torch.zeros(N, device=device)

    proj_area = inf_mass_t

    mat = torch.stack([
        color_err, depth_err, grad_norm, vis_count.float(), inf_mass_t,
        pos_drift, res_drift, unc_var, proj_area, update_freq, ages
    ], dim=-1)
    return mat.detach().cpu().numpy().astype(np.float32)


def run_policy_online_trajectory(
    policy: str,
    frames: List[Dict[str, Any]],
    intrinsics: torch.Tensor,
    budget_ms: float,
    p6_predictor: FrozenContextPredictor,
    p4_predictor: FrozenUtilityPredictor,
    safety_factor: float = 1.10,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Runs sequential online trajectory for a single policy."""
    H, W = frames[0]['rgb'].shape[:2]
    torch.manual_seed(seed)
    np.random.seed(seed)

    pipe = build_reconstruction_pipeline(H, W, device)
    pipe.initialize(frames[0]['rgb'], frames[0]['depth'], intrinsics, frames[0]['pose'])
    pipe.config['scheduler']['gpu_budget_ms'] = budget_ms
    pipe.scheduler.gpu_budget_ms = budget_ms

    def online_selector(pipeline_obj: OnlineReconstructionPipeline, N_gaussians: int) -> torch.Tensor:
        mask = torch.zeros(N_gaussians, dtype=torch.bool, device=device)
        if N_gaussians == 0 or policy == "no_op":
            return mask

        # Extract features
        X = extract_online_canonical_features(pipeline_obj, N_gaussians)

        # Predict costs using Phase 4 predictor (unified cost baseline)
        res_p4 = p4_predictor.predict_features(X)
        pred_t = res_p4["predicted_delta_t"]
        pred_u_p4 = res_p4["predicted_utility"]

        est = pipeline_obj.importance_estimator
        imp_scores = est.compute_importance()[:N_gaussians].detach().cpu().numpy()
        pids = getattr(pipeline_obj.gaussian_model, "persistent_ids", None)

        # Build candidate representations (subsample visible pool if large for speed)
        cand_list = []
        max_pool = min(N_gaussians, 150)
        pool_indices = np.random.choice(N_gaussians, max_pool, replace=False) if N_gaussians > max_pool else list(range(N_gaussians))

        for idx in pool_indices:
            pid = int(pids[idx].item()) if (pids is not None and idx < len(pids)) else idx
            c_dict = {
                "gaussian_id": int(idx),
                "persistent_id": pid,
                "features": {
                    "rgb_error": float(X[idx, 0]),
                    "depth_error": float(X[idx, 1]),
                    "influence_mass": float(X[idx, 4]),
                },
                "predicted_importance": float(imp_scores[idx]),
                "measured_trial_cost_ms": float(pred_t[idx]),
                "predicted_delta_t": float(pred_t[idx]),
                "predicted_utility": float(pred_u_p4[idx]),
            }
            cand_list.append(c_dict)

        positions = pipeline_obj.gaussian_model.positions

        sel_res = select_phase6_subset(
            candidates=cand_list,
            policy=policy,
            budget=budget_ms,
            seed=seed + pipeline_obj.frame_count,
            safety_factor=safety_factor,
            reject_negative=False,
            use_predicted_cost=True,
            positions=positions,
            all_features=X,
            phase6_predictor=p6_predictor,
        )

        for s_idx in sel_res.selected_indices:
            act_idx = map_candidate_to_active_index(cand_list[s_idx], pipeline_obj.gaussian_model)
            if act_idx is not None and 0 <= act_idx < N_gaussians:
                mask[act_idx] = True

        return mask

    pipe._custom_selector_fn = online_selector

    frame_logs = []
    prev_selected_set = set()
    churn_history = []
    cumulative_compute_ms = 0.0
    initial_psnr = None

    for t in range(1, len(frames)):
        rgb = frames[t]['rgb']
        depth = frames[t]['depth']
        pose = frames[t]['pose']

        m = pipe.process_frame(rgb, depth, pose)
        opt_time = float(m['opt_time_ms'])
        cumulative_compute_ms += opt_time
        is_viol = bool(opt_time > budget_ms)

        psnr_val = float(m.get('psnr_post', m.get('psnr', 0.0)))
        if initial_psnr is None:
            initial_psnr = float(m.get('psnr_pre', psnr_val))

        opt_mask = getattr(pipe, '_last_optimize_mask', None)
        cur_selected = set(torch.where(opt_mask)[0].cpu().numpy().tolist()) if opt_mask is not None else set()

        ext_churn = compute_extended_churn(cur_selected, prev_selected_set) if t > 1 else {
            "selection_churn": 0.0,
            "selected_count": len(cur_selected),
            "retained_count": 0,
            "new_selected_count": len(cur_selected),
        }
        churn_history.append(ext_churn["selection_churn"])
        prev_selected_set = cur_selected

        total_sec = max(1e-4, cumulative_compute_ms / 1000.0)
        frame_logs.append({
            "frame": t,
            "psnr": psnr_val,
            "psnr_pre": float(m.get('psnr_pre', 0.0)),
            "psnr_post": psnr_val,
            "ssim": float(m.get('ssim', 0.0)),
            "depth_l1": float(m['depth_l1']),
            "opt_time_ms": opt_time,
            "cumulative_compute_ms": cumulative_compute_ms,
            "quality_per_compute": float(psnr_val / total_sec),
            "n_gaussians": int(m.get('n_gaussians', 0)),
            "is_violation": is_viol,
            "churn": ext_churn["selection_churn"],
            "selected_count": ext_churn["selected_count"],
        })

    total_sec = max(1e-4, cumulative_compute_ms / 1000.0)
    mean_p = float(np.mean([r["psnr"] for r in frame_logs])) if frame_logs else 0.0
    last_p = float(frame_logs[-1]["psnr"]) if frame_logs else 0.0

    return {
        "policy": policy,
        "mean_psnr": mean_p,
        "final_psnr": last_p,
        "mean_ssim": float(np.mean([r["ssim"] for r in frame_logs])) if frame_logs else 0.0,
        "mean_depth_l1": float(np.mean([r["depth_l1"] for r in frame_logs])) if frame_logs else 0.0,
        "mean_opt_time_ms": float(np.mean([r["opt_time_ms"] for r in frame_logs])) if frame_logs else 0.0,
        "total_compute_ms": cumulative_compute_ms,
        "mean_churn": float(np.mean(churn_history[1:])) if len(churn_history) > 1 else 0.0,
        "violation_rate_pct": float(np.mean([1.0 if r["is_violation"] else 0.0 for r in frame_logs]) * 100.0) if frame_logs else 0.0,
        "quality_per_compute": float(mean_p / total_sec),
        "frame_logs": frame_logs,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Online Trajectory (RQ6)")
    parser.add_argument("--frames", type=int, default=15, help="Number of frames to run")
    parser.add_argument("--budget", type=float, default=15.0, help="Per-frame budget in ms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(repo_root, "results", "phase6_context_utility", "online")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    protocol = load_protocol()
    ds_cfg = get_dataset_config("tum_fr2_xyz", protocol)
    H, W = get_resolution("tum_fr2_xyz", protocol)

    print("=" * 80)
    print("  PHASE 6: RQ6 ONLINE RECONSTRUCTION TRAJECTORY BENCHMARK")
    print("=" * 80)
    print(f"  Scene: tum_fr2_xyz | Frames: {args.frames} | Budget: {args.budget:.1f}ms | Device: {device}")

    # 1. Load sequence
    frames, intrinsics = load_tum_sequence(ds_cfg["full_path"], "freiburg2", args.frames, H, W, device)
    print(f"  Loaded {len(frames)} frames at {W}x{H}")

    # 2. Load predictors
    p6_ckpt = os.path.join(repo_root, "results", "phase6_context_utility", "checkpoints", f"context_mlp_V11_seed_{args.seed}.pt")
    p6_norm = os.path.join(repo_root, "results", "phase6_context_utility", "normalization_V11.json")
    p6_pred = FrozenContextPredictor(p6_ckpt, p6_norm, device=device)
    p4_pred = FrozenUtilityPredictor(seed=args.seed, device=device)

    # 3. Competing policies
    policies = [
        "no_op",
        "random",
        "error_influence",
        "heuristic",
        "phase4_learned",
        "phase6_adaptive",
    ]

    summaries = {}
    print(f"\n[Online] Running {len(policies)} policies over {len(frames)-1} online steps...")

    for pol in policies:
        t0 = time.perf_counter()
        res = run_policy_online_trajectory(
            policy=pol,
            frames=frames,
            intrinsics=intrinsics,
            budget_ms=args.budget,
            p6_predictor=p6_pred,
            p4_predictor=p4_pred,
            seed=args.seed,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        summaries[pol] = res
        print(f"  {pol:<18} | PSNR: {res['mean_psnr']:.2f}dB (Final: {res['final_psnr']:.2f}dB) | SSIM: {res['mean_ssim']:.3f} | Latency: {res['mean_opt_time_ms']:.1f}ms | Churn: {res['mean_churn']:.2f} | Time: {elapsed:.1f}s")

    # 4. Print Summary Table
    print("\n" + "=" * 95)
    print("  ONLINE TRAJECTORY SUMMARY TABLE (RQ6)")
    print("=" * 95)
    print(f"{'Policy':<18} | {'Mean PSNR':<10} | {'Final PSNR':<11} | {'Mean SSIM':<10} | {'Latency(ms)':<12} | {'Quality/Comp':<14} | {'Viol %':<8}")
    print("-" * 95)
    for pol in policies:
        s = summaries[pol]
        print(f"{pol:<18} | {s['mean_psnr']:<10.2f} | {s['final_psnr']:<11.2f} | {s['mean_ssim']:<10.3f} | {s['mean_opt_time_ms']:<12.1f} | {s['quality_per_compute']:<14.1f} | {s['violation_rate_pct']:<8.1f}")
    print("=" * 95)

    # 5. Save Artifacts
    out_file = os.path.join(output_dir, f"online_trajectory_seed_{args.seed}.json")
    with open(out_file, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n[Saved] Online Trajectory Artifacts: {out_file}")


if __name__ == "__main__":
    main()
