"""Benchmark Harness: Fixed Latency Budgets & Adaptive Budget Experiments (Milestone R6).

Evaluates the system across strict latency constraints:
    Budget ∈ {2ms, 4ms, 8ms, 16ms, Unconstrained (∞)}

Metrics measured:
    - Reconstruction Quality: PSNR (dB), Depth L1
    - Latency & FPS: Mean Latency (ms), P95 Latency (ms), Jitter (std ms), Mean FPS, Min FPS
    - Budget Compliance: Violation Rate (% frames exceeding budget), Over-budget penalty
    - Compute Allocation: Average # Gaussians optimized per frame
"""
import time
import torch
from typing import Dict, List, Optional, Any, Union
import numpy as np

from .pipeline import OnlineReconstructionPipeline
from .scheduler import OptimizationPolicy


def run_budget_experiment(
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    budget_ms: Optional[float] = 16.6,
    policy: str = 'budget_aware',
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Run pipeline under a specific GPU latency budget."""
    cfg = {
        'scheduler': {
            'policy': policy,
            'gpu_budget_ms': budget_ms if budget_ms is not None else -1.0,
            'cost_per_gaussian_us': 0.5,
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

    summary = pipeline.get_metrics_summary()
    lat_stats = summary.get('latency_stats', {})

    return {
        'budget_ms': budget_ms,
        'budget_label': f"{int(budget_ms)}ms" if budget_ms is not None and budget_ms > 0 else "Unconstrained",
        'avg_psnr': summary.get('avg_psnr', 0.0),
        'avg_depth_l1': summary.get('avg_depth_l1', 0.0),
        'avg_n_optimized': float(np.mean([m['n_optimized'] for m in per_frame_records])) if per_frame_records else 0.0,
        'final_n_gaussians': summary.get('final_n_gaussians', 0),
        'mean_frame_time_ms': summary.get('avg_frame_time_ms', 0.0),
        'p95_frame_time_ms': lat_stats.get('p95_frame_time_ms', 0.0),
        'std_frame_time_ms': lat_stats.get('std_frame_time_ms', 0.0),
        'avg_fps': summary.get('avg_fps', 0.0),
        'min_fps': lat_stats.get('min_fps', 0.0),
        'budget_violation_rate': summary.get('budget_violation_rate', 0.0),
        'per_frame_records': per_frame_records,
    }


def run_full_budget_matrix(
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    budgets: Optional[List[Optional[float]]] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Run evaluation across all target latency budgets."""
    if budgets is None:
        budgets = [2.0, 4.0, 8.0, 16.0, None]

    experiments = []

    for b in budgets:
        res = run_budget_experiment(
            frames, intrinsics, budget_ms=b, device=device
        )
        experiments.append(res)

    return {'experiments': experiments}


def format_budget_table(ablation_results: Dict[str, Any]) -> str:
    """Format latency budget evaluation as markdown table."""
    lines = []
    lines.append("| Latency Budget | PSNR (dB) | Depth L1 | Opt Gaussians | Mean Latency (ms) | P95 Latency (ms) | Mean FPS | Min FPS | Violation Rate |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for exp in ablation_results['experiments']:
        b_label = f"**{exp['budget_label']}**"
        psnr = f"{exp['avg_psnr']:.2f}"
        depth_l1 = f"{exp['avg_depth_l1']:.4f}"
        n_opt = f"{exp['avg_n_optimized']:.0f}"
        mean_t = f"{exp['mean_frame_time_ms']:.1f}"
        p95_t = f"{exp['p95_frame_time_ms']:.1f}"
        fps = f"{exp['avg_fps']:.1f}"
        min_fps = f"{exp['min_fps']:.1f}"
        viol = f"{exp['budget_violation_rate']:.1%}"

        lines.append(f"| {b_label} | {psnr} | {depth_l1} | {n_opt} | {mean_t} | {p95_t} | {fps} | {min_fps} | {viol} |")

    return "\n".join(lines)
