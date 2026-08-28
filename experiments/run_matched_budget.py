"""
Real Matched-Budget Benchmark Runner.

Runs all 6 optimization policies under identical GPU compute budget constraints
using the real OnlineReconstructionPipeline.

Policies:
  1. Full: optimize 100% of Gaussians
  2. Random: random selection
  3. Error-only: select strictly by top error
  4. Binary: threshold stable vs unstable
  5. Top-K: continuous importance top-K
  6. Ours (Budget-Aware): value density (importance/cost) knapsack
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


def create_benchmark_frames(n_frames: int = 15, H: int = 64, W: int = 80):
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
        
        # Add slight noise
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
            'gpu_budget_ms': config_overrides.get('scheduler', {}).get('gpu_budget_ms', 10.0),
            'policy': config_overrides.get('scheduler', {}).get('policy', 'budget_aware'),
            'optimize_ratio': 0.5,
        },
        'densification': {
            'max_new_per_frame': 60,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        },
    }
    pipeline = OnlineReconstructionPipeline(config=base_config, device=device)
    # Enable novelty boost & error prior for importance
    pipeline.importance_estimator.novelty_weight = 0.5
    pipeline.importance_estimator.use_error_prior = True
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Matched-Budget Benchmark")
    parser.add_argument('--n_frames', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    print("=" * 72)
    print("          MATCHED-BUDGET BENCHMARK ON REAL 3DGS PIPELINE")
    print("=" * 72)
    print(f"Device: {args.device} | Frames: {args.n_frames}")
    
    frames, intrinsics = create_benchmark_frames(n_frames=args.n_frames)
    
    budget_levels = [2.0, 5.0, 10.0, 20.0]  # in ms
    benchmark = MatchedBudgetBenchmark(budget_levels_ms=budget_levels, device=args.device)
    
    print(f"Running full matrix: {len(benchmark.policies)} policies x {len(budget_levels)} budget levels...")
    start_time = time.time()
    results = benchmark.run_full_matrix(real_pipeline_factory, frames, intrinsics)
    total_time = time.time() - start_time
    
    print(f"\nMatrix execution completed in {total_time:.1f}s")
    
    table = benchmark.format_table(results)
    print("\n" + table)
    
    # Save results
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'matched_budget')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'results_table.md')
    with open(save_path, 'w') as f:
        f.write("# Matched-Budget Benchmark Results\n\n")
        f.write(f"Evaluated on {args.n_frames} frames using measured compute time.\n\n")
        f.write(table)
        f.write("\n")
        
    json_path = os.path.join(save_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to:\n- {save_path}\n- {json_path}")


if __name__ == "__main__":
    main()
