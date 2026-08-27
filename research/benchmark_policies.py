"""Benchmark Harness: Compare Optimization Selection Policies (Milestone R4).

Evaluates the research hierarchy:
    Policy 0: Full (100% optimization)
    Policy 1: Random selection (ratio r)
    Policy 2: Binary stable/unstable (RTG-SLAM baseline)
    Policy 3: Continuous Importance Top-K (ratio r)
    Policy 4: Budget-Aware (Importance / Cost knapsack)

Metrics measured:
    - Quality: PSNR, Depth L1
    - Compute: # Optimized Gaussians, Optimization Time (ms), Total Frame Time (ms)
    - Tradeoff: Quality Retained (% of Full), Compute Saved (% of Full), Pareto Frontier
"""
import time
import torch
from typing import Dict, List, Optional, Any, Union
import numpy as np

from .pipeline import OnlineReconstructionPipeline
from .scheduler import OptimizationPolicy


def run_policy_experiment(
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    policy: Union[str, OptimizationPolicy],
    ratio: float = 0.5,
    budget_ms: float = 16.6,
    device: str = 'cpu',
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a full reconstruction sequence under a specific optimization policy.
    
    Args:
        frames: list of {'rgb': (H,W,3), 'depth': (H,W), 'pose': (4,4)}
        intrinsics: (3,3) camera intrinsic matrix
        policy: policy name or OptimizationPolicy enum
        ratio: fraction of Gaussians to optimize (for random, top_k)
        budget_ms: GPU budget in ms (for budget_aware)
        device: 'cpu' or 'cuda'
        config_overrides: optional dict with config overrides
        
    Returns:
        Dict with experiment metrics:
            'policy': str
            'ratio': float
            'budget_ms': float
            'avg_psnr': float
            'avg_depth_l1': float
            'avg_n_optimized': float
            'final_n_gaussians': int
            'avg_frame_time_ms': float
            'per_frame_metrics': list
    """
    policy_str = str(policy).lower()
    if hasattr(policy, "value"):
        policy_str = policy.value

    cfg = {
        'scheduler': {
            'policy': policy_str,
            'optimize_ratio': ratio,
            'gpu_budget_ms': budget_ms,
        },
        'rendering': {
            'image_width': frames[0]['rgb'].shape[1],
            'image_height': frames[0]['rgb'].shape[0],
            'tile_size': 16,
            'use_surface_aware_depth': True,
        }
    }
    if config_overrides:
        # Merge overrides
        for k, v in config_overrides.items():
            if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    pipeline = OnlineReconstructionPipeline(config=cfg, device=device)

    # Initialize from first frame
    pipeline.initialize(
        rgb=frames[0]['rgb'],
        depth=frames[0]['depth'],
        intrinsics=intrinsics,
        pose=frames[0].get('pose', None),
    )

    per_frame_records = []

    # Process consecutive frames
    for f_idx in range(1, len(frames)):
        metrics = pipeline.process_frame(
            rgb=frames[f_idx]['rgb'],
            depth=frames[f_idx]['depth'],
            gt_pose=frames[f_idx].get('pose', None),
        )
        per_frame_records.append(metrics)

    if not per_frame_records:
        return {
            'policy': policy_str,
            'ratio': ratio,
            'budget_ms': budget_ms,
            'avg_psnr': 0.0,
            'avg_depth_l1': 0.0,
            'avg_n_optimized': 0.0,
            'final_n_gaussians': pipeline.gaussian_model.num_gaussians,
            'avg_frame_time_ms': 0.0,
            'per_frame_records': [],
        }

    avg_psnr = float(np.mean([m['psnr'] for m in per_frame_records]))
    avg_depth_l1 = float(np.mean([m['depth_l1'] for m in per_frame_records]))
    avg_n_opt = float(np.mean([m['n_optimized'] for m in per_frame_records]))
    avg_time = float(np.mean([m['frame_time_ms'] for m in per_frame_records]))

    return {
        'policy': policy_str,
        'ratio': ratio,
        'budget_ms': budget_ms,
        'avg_psnr': avg_psnr,
        'avg_depth_l1': avg_depth_l1,
        'avg_n_optimized': avg_n_opt,
        'final_n_gaussians': pipeline.gaussian_model.num_gaussians,
        'avg_frame_time_ms': avg_time,
        'per_frame_records': per_frame_records,
    }


def run_full_policy_ablation_matrix(
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    ratios: Optional[List[float]] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Run all 5 research policies across the standard ratio spectrum.
    
    Matrix:
        1. Full Optimization (r = 1.0) -> Baseline upper bound
        2. Random Selection (r in [0.10, 0.25, 0.50, 0.75])
        3. Binary Stable/Unstable (RTG-SLAM style)
        4. Top-K Continuous Importance (r in [0.10, 0.25, 0.50, 0.75])
        5. Budget-Aware (B in [2ms, 4ms, 8ms, 16ms])
    
    Args:
        frames: sequence of RGB-D frames
        intrinsics: camera intrinsics
        ratios: ratios to test for random and top_k (default: [0.10, 0.25, 0.50, 0.75])
        device: device
        
    Returns:
        Structured results dict with comparisons, speedups, and Pareto analysis.
    """
    if ratios is None:
        ratios = [0.10, 0.25, 0.50, 0.75]

    results = []

    # 1. Full baseline (r=1.0)
    full_res = run_policy_experiment(
        frames, intrinsics, policy=OptimizationPolicy.FULL, ratio=1.0, device=device
    )
    full_res['name'] = "Full (100%)"
    full_psnr = full_res['avg_psnr']
    full_time = full_res['avg_frame_time_ms']
    results.append(full_res)

    # 2. Binary baseline
    bin_res = run_policy_experiment(
        frames, intrinsics, policy=OptimizationPolicy.BINARY, device=device
    )
    bin_res['name'] = "Binary (RTG-SLAM)"
    results.append(bin_res)

    # 3. Random baseline across ratios
    for r in ratios:
        rand_res = run_policy_experiment(
            frames, intrinsics, policy=OptimizationPolicy.RANDOM, ratio=r, device=device
        )
        rand_res['name'] = f"Random ({int(r*100)}%)"
        results.append(rand_res)

    # 4. Top-K Continuous Importance across ratios
    for r in ratios:
        topk_res = run_policy_experiment(
            frames, intrinsics, policy=OptimizationPolicy.TOP_K, ratio=r, device=device
        )
        topk_res['name'] = f"Top-K Imp ({int(r*100)}%)"
        results.append(topk_res)

    # 5. Budget-Aware
    for b in [2.0, 4.0, 8.0, 16.0]:
        budget_res = run_policy_experiment(
            frames, intrinsics, policy=OptimizationPolicy.BUDGET_AWARE, budget_ms=b, device=device
        )
        budget_res['name'] = f"Budget-Aware ({int(b)}ms)"
        results.append(budget_res)

    # Compute quality retained and speedup relative to full baseline
    for item in results:
        psnr_delta = item['avg_psnr'] - full_psnr
        speedup = full_time / max(item['avg_frame_time_ms'], 1e-5)
        item['psnr_delta_db'] = psnr_delta
        item['speedup'] = speedup
        opt_ratio = item['avg_n_optimized'] / max(item['final_n_gaussians'], 1)
        item['actual_opt_ratio'] = opt_ratio

    return {
        'full_psnr': full_psnr,
        'full_time_ms': full_time,
        'experiments': results,
    }


def format_benchmark_table(ablation_results: Dict[str, Any]) -> str:
    """Format benchmark results into a clear markdown table."""
    lines = []
    lines.append("| Policy | Strategy | Opt Ratio | Opt Gaussians | PSNR (dB) | ΔPSNR (dB) | Depth L1 | Time (ms) | Speedup |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for exp in ablation_results['experiments']:
        name = exp.get('name', exp['policy'])
        pol = exp['policy']
        ratio_str = f"{exp['actual_opt_ratio']:.1%}"
        n_opt = f"{exp['avg_n_optimized']:.0f}"
        psnr = f"{exp['avg_psnr']:.2f}"
        delta_psnr = f"{exp.get('psnr_delta_db', 0.0):+.2f}"
        depth_l1 = f"{exp['avg_depth_l1']:.4f}"
        time_ms = f"{exp['avg_frame_time_ms']:.1f}"
        speedup = f"{exp.get('speedup', 1.0):.2f}x"

        lines.append(f"| **{name}** | `{pol}` | {ratio_str} | {n_opt} | {psnr} | {delta_psnr} | {depth_l1} | {time_ms} | {speedup} |")

    return "\n".join(lines)
