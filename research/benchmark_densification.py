"""Benchmark Harness: Compare Densification Strategies & Adaptive Thresholds (Milestone R5).

Ablation groups:
    Strategy 1: Uniform Random Densification
    Strategy 2: Error-driven Densification (P ∝ E_c + E_d)
    Strategy 3: Importance-weighted Densification (P ∝ λ_c·E_c + λ_d·E_d + λ_t·T)
    Strategy 4: Importance-weighted + Dynamic Adaptive Thresholds (δ(t) = k·σ(t))

Metrics measured:
    - Reconstruction Quality: PSNR (dB), Depth L1
    - Geometric Efficiency: Final # Gaussians, Densified per Frame
    - Timing & Latency: Densification Time (ms), Total Frame Time (ms)
"""
import time
import torch
from typing import Dict, List, Optional, Any
import numpy as np

from .pipeline import OnlineReconstructionPipeline


def run_densification_experiment(
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    strategy: str = 'importance',
    use_adaptive_thresholds: bool = True,
    max_new_per_frame: int = 200,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Run pipeline with specific densification configuration."""
    cfg = {
        'densification': {
            'strategy': strategy,
            'use_adaptive_thresholds': use_adaptive_thresholds,
            'max_new_per_frame': max_new_per_frame,
            'error_threshold_color': 0.1,
            'error_threshold_depth': 0.05,
            'transmission_threshold': 0.5,
            'adaptive_k': 2.0,
        },
        'rendering': {
            'image_width': frames[0]['rgb'].shape[1],
            'image_height': frames[0]['rgb'].shape[0],
            'tile_size': 16,
            'use_surface_aware_depth': True,
        }
    }

    pipeline = OnlineReconstructionPipeline(config=cfg, device=device)

    pipeline.initialize(
        rgb=frames[0]['rgb'],
        depth=frames[0]['depth'],
        intrinsics=intrinsics,
        pose=frames[0].get('pose', None),
    )

    per_frame_records = []

    for f_idx in range(1, len(frames)):
        metrics = pipeline.process_frame(
            rgb=frames[f_idx]['rgb'],
            depth=frames[f_idx]['depth'],
            gt_pose=frames[f_idx].get('pose', None),
        )
        per_frame_records.append(metrics)

    avg_psnr = float(np.mean([m['psnr'] for m in per_frame_records])) if per_frame_records else 0.0
    avg_depth_l1 = float(np.mean([m['depth_l1'] for m in per_frame_records])) if per_frame_records else 0.0
    avg_time = float(np.mean([m['frame_time_ms'] for m in per_frame_records])) if per_frame_records else 0.0

    return {
        'strategy': strategy,
        'use_adaptive_thresholds': use_adaptive_thresholds,
        'avg_psnr': avg_psnr,
        'avg_depth_l1': avg_depth_l1,
        'final_n_gaussians': pipeline.gaussian_model.num_gaussians,
        'avg_frame_time_ms': avg_time,
        'per_frame_records': per_frame_records,
    }


def run_full_densification_ablation(
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Run all 4 densification ablation configurations."""
    experiments = []

    # 1. Uniform random with fixed thresholds
    res_uniform = run_densification_experiment(
        frames, intrinsics, strategy='uniform', use_adaptive_thresholds=False, device=device
    )
    res_uniform['name'] = "Uniform (Fixed Thresh)"
    experiments.append(res_uniform)

    # 2. Error-driven with fixed thresholds
    res_error = run_densification_experiment(
        frames, intrinsics, strategy='error_driven', use_adaptive_thresholds=False, device=device
    )
    res_error['name'] = "Error-Driven (Fixed Thresh)"
    experiments.append(res_error)

    # 3. Importance-weighted (Composite P(u)) with fixed thresholds
    res_imp_fixed = run_densification_experiment(
        frames, intrinsics, strategy='importance', use_adaptive_thresholds=False, device=device
    )
    res_imp_fixed['name'] = "Importance-Weighted (Fixed Thresh)"
    experiments.append(res_imp_fixed)

    # 4. Importance-weighted + Dynamic Adaptive Thresholds (δ(t) = k·σ(t))
    res_imp_adaptive = run_densification_experiment(
        frames, intrinsics, strategy='importance', use_adaptive_thresholds=True, device=device
    )
    res_imp_adaptive['name'] = "Importance-Weighted + Adaptive δ(t)"
    experiments.append(res_imp_adaptive)

    return {'experiments': experiments}


def format_densification_table(ablation_results: Dict[str, Any]) -> str:
    """Format densification ablation results as markdown table."""
    lines = []
    lines.append("| Densification Strategy | Threshold Type | Final Gaussians | PSNR (dB) | Depth L1 | Frame Time (ms) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for exp in ablation_results['experiments']:
        name = exp['name']
        thresh_type = "Adaptive δ(t)" if exp['use_adaptive_thresholds'] else "Fixed"
        n_g = f"{exp['final_n_gaussians']}"
        psnr = f"{exp['avg_psnr']:.2f}"
        depth_l1 = f"{exp['avg_depth_l1']:.4f}"
        time_ms = f"{exp['avg_frame_time_ms']:.1f}"

        lines.append(f"| **{name}** | `{thresh_type}` | {n_g} | {psnr} | {depth_l1} | {time_ms} |")

    return "\n".join(lines)
