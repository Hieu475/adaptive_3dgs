"""
Primary Research Experiment — True Matched-Budget Benchmark on Real 3DGS.

Evaluates 6 Budget-Constrained Policies vs 1 Quality Upper Bound across 5 Budget Levels:
    Budgets: B ∈ {1.0, 2.0, 4.0, 8.0, 16.0} ms

Policies:
    - Full: No (Quality Upper Bound, unconstrained)
    - Random: Yes (Budget-Constrained)
    - Error-Only: Yes (Ranked strictly by E_depth + E_color)
    - Error × Influence: Yes (Strong Non-Learning Baseline: E_i × Influence_i)
    - Binary: Yes (Threshold-based stable/unstable)
    - Top-K: Yes (Continuous multi-signal importance Top-K)
    - Ours: Yes (Proposed Importance/Cost Knapsack Optimization)

Metrics:
    - PSNR (dB) ↑
    - Depth L1 (m) ↓
    - Measured Compute (ms)
    - Budget Violation Rate (%) ↓
    - Latency Jitter (ms) ↓
"""
import sys
import os
import torch
import numpy as np
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.pipeline import OnlineReconstructionPipeline
from research.matched_budget_benchmark import MatchedBudgetBenchmark, SchedulerMetrics


def create_benchmark_frames(n_frames: int = 12, H: int = 64, W: int = 80):
    """Generate structured synthetic benchmark frames."""
    fx, fy = 160.0, 160.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)
    frames = []
    
    for t in range(n_frames):
        angle = t * 0.03
        pose = torch.eye(4)
        pose[0, 0] = np.cos(angle); pose[0, 2] = np.sin(angle)
        pose[2, 0] = -np.sin(angle); pose[2, 2] = np.cos(angle)
        pose[0, 3] = 0.02 * t
        
        rgb = torch.zeros(H, W, 3)
        depth = torch.ones(H, W) * 3.0
        
        # Texture patch
        for i in range(H // 2):
            for j in range(W // 2):
                if (i // 8 + j // 8) % 2 == 0:
                    rgb[i, j] = torch.tensor([0.9, 0.2, 0.1])
                else:
                    rgb[i, j] = torch.tensor([0.1, 0.8, 0.2])
        depth[:H//2, :W//2] = 2.0
        
        # Foreground box
        box_h = slice(H // 2 + 5, H - 5)
        box_w = slice(W // 4, 3 * W // 4)
        rgb[box_h, box_w] = torch.tensor([0.7, 0.3, 0.5])
        depth[box_h, box_w] = 1.0
        
        rgb = (rgb + 0.02 * torch.randn_like(rgb)).clamp(0, 1)
        depth = depth + 0.01 * torch.randn_like(depth)
        depth[depth <= 0] = 0.1
        
        frames.append({
            'rgb': rgb,
            'depth': depth,
            'pose': pose
        })
        
    return frames, intrinsics


def real_pipeline_factory(config_overrides, device):
    """Instantiate real OnlineReconstructionPipeline with given budget overrides."""
    budget_ms = config_overrides.get('scheduler', {}).get('gpu_budget_ms', 4.0)
    policy = config_overrides.get('scheduler', {}).get('policy', 'budget_aware')
    # Derive target Gaussian count dynamically from calibrated cost model: T(M) = T0 + beta*M
    cost_per_gauss_ms = 0.0112
    target_k = max(5, int((budget_ms * 5.0) / cost_per_gauss_ms)) if budget_ms > 0 else 50
    
    base_config = {
        'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 20000, 'initial_scale': 0.02},
        'rendering': {
            'tile_size': 16,
            'image_width': 80,
            'image_height': 64,
            'use_surface_aware_depth': True,
            'attribution_top_k': 4
        },
        'scheduler': {
            'gpu_budget_ms': budget_ms,
            'policy': policy,
            'top_k': target_k if policy != 'ours' else None,  # Ours uses dynamic knapsack
        },
        'densification': {
            'max_new_per_frame': 60,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        },
    }
    pipeline = OnlineReconstructionPipeline(config=base_config, device=device)
    pipeline.importance_estimator.novelty_weight = 0.5
    pipeline.importance_estimator.use_error_prior = True
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Matched-Budget Primary Benchmark")
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--frames', type=int, default=None, help='Alias for n_frames')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    if args.frames is not None:
        args.n_frames = args.frames
    
    print("=" * 85)
    print("      R27 PRIMARY RESEARCH BENCHMARK: TRUE MATCHED-BUDGET COMPARISON")
    print("=" * 85)
    print(f"Device: {args.device} | Frames: {args.n_frames}")
    
    frames, intrinsics = create_benchmark_frames(n_frames=args.n_frames)
    
    budget_levels = [1.0, 2.0, 4.0, 8.0, 16.0]  # in ms
    benchmark = MatchedBudgetBenchmark(budget_levels_ms=budget_levels, device=args.device)
    
    print(f"Running matrix: {len(benchmark.policies)} policies x {len(budget_levels)} budget levels...")
    start_time = time.time()
    results = benchmark.run_full_matrix(real_pipeline_factory, frames, intrinsics)
    total_time = time.time() - start_time
    
    print(f"\nBenchmark matrix completed in {total_time:.1f}s")
    
    table = benchmark.format_table(results)
    print("\n" + table)
    
    # Save results
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'matched_budget')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'results_table.md')
    with open(save_path, 'w') as f:
        f.write("# R27 Primary Benchmark: True Matched-Budget Results\n\n")
        f.write(f"Evaluated on {args.n_frames} frames using measured compute time.\n\n")
        f.write(table)
        f.write("\n")
        
    json_path = os.path.join(save_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Export Pareto curve data (Quality vs Compute: x = T_opt, y = PSNR)
    pareto_csv = os.path.join(save_dir, 'pareto_quality_vs_compute.csv')
    with open(pareto_csv, 'w') as f:
        f.write("budget_ms,policy,budget_constrained,measured_compute_ms,psnr,depth_l1,jitter\n")
        for r in results:
            f.write(f"{r['budget_ms']},{r['policy_name']},{r['budget_constrained']},{r['measured_compute_ms']:.3f},{r['avg_psnr']:.3f},{r['avg_depth_l1']:.4f},{r['jitter']:.3f}\n")
            
    print(f"\nArtifacts saved to:")
    print(f"  - {save_path}")
    print(f"  - {json_path}")
    print(f"  - {pareto_csv}")


if __name__ == "__main__":
    main()
