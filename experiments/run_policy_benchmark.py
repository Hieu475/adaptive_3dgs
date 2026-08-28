"""R11: Full Optimization Policy Benchmark (Quality vs Compute Pareto Frontier).

Evaluates policies across ratios and budgets with multi-seed statistical evaluation:
    - Policy 0: Full (100%)
    - Policy 1: Random (10%, 25%, 50%, 75%)
    - Policy 2: Binary (RTG-SLAM active/freeze)
    - Policy 3: Top-K Continuous Importance (10%, 25%, 50%, 75%)
    - Policy 4: Budget-Aware (2ms, 4ms, 8ms, 16ms)

Measures:
    - Quality: PSNR (dB), Depth L1
    - Latency: Mean Frame Time (ms), P50, P95, Mean FPS
    - Allocation: Optimized Gaussians, Frozen Gaussians, Optimized Ratio (%)
"""
import os
import sys
import json
import csv
import argparse
import torch
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.pipeline import OnlineReconstructionPipeline
from research.scheduler import OptimizationPolicy
from research.benchmark_policies import run_policy_experiment
from experiments.run_importance_validation import generate_synthetic_benchmark_dataset


def run_multi_seed_policy_benchmark(
    seeds: List[int],
    n_frames: int = 10,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Run full policy spectrum across multiple random seeds."""
    policy_configs = [
        {'name': 'Full (100%)', 'policy': OptimizationPolicy.FULL, 'ratio': 1.0, 'budget_ms': 16.6},
        {'name': 'Binary (RTG-SLAM)', 'policy': OptimizationPolicy.BINARY, 'ratio': 0.5, 'budget_ms': 16.6},
        {'name': 'Random (10%)', 'policy': OptimizationPolicy.RANDOM, 'ratio': 0.10, 'budget_ms': 16.6},
        {'name': 'Random (25%)', 'policy': OptimizationPolicy.RANDOM, 'ratio': 0.25, 'budget_ms': 16.6},
        {'name': 'Random (50%)', 'policy': OptimizationPolicy.RANDOM, 'ratio': 0.50, 'budget_ms': 16.6},
        {'name': 'Random (75%)', 'policy': OptimizationPolicy.RANDOM, 'ratio': 0.75, 'budget_ms': 16.6},
        {'name': 'Top-K (10%)', 'policy': OptimizationPolicy.TOP_K, 'ratio': 0.10, 'budget_ms': 16.6},
        {'name': 'Top-K (25%)', 'policy': OptimizationPolicy.TOP_K, 'ratio': 0.25, 'budget_ms': 16.6},
        {'name': 'Top-K (50%)', 'policy': OptimizationPolicy.TOP_K, 'ratio': 0.50, 'budget_ms': 16.6},
        {'name': 'Top-K (75%)', 'policy': OptimizationPolicy.TOP_K, 'ratio': 0.75, 'budget_ms': 16.6},
        {'name': 'Budget-Aware (2ms)', 'policy': OptimizationPolicy.BUDGET_AWARE, 'ratio': 0.5, 'budget_ms': 2.0},
        {'name': 'Budget-Aware (4ms)', 'policy': OptimizationPolicy.BUDGET_AWARE, 'ratio': 0.5, 'budget_ms': 4.0},
        {'name': 'Budget-Aware (8ms)', 'policy': OptimizationPolicy.BUDGET_AWARE, 'ratio': 0.5, 'budget_ms': 8.0},
        {'name': 'Budget-Aware (16ms)', 'policy': OptimizationPolicy.BUDGET_AWARE, 'ratio': 0.5, 'budget_ms': 16.0},
    ]

    all_seed_results = {cfg['name']: [] for cfg in policy_configs}

    for seed in seeds:
        print(f"\n--- Running Policy Spectrum for Seed {seed} ---")
        frames, intrinsics = generate_synthetic_benchmark_dataset(n_frames=n_frames, seed=seed)
        
        for cfg in policy_configs:
            res = run_policy_experiment(
                frames=frames,
                intrinsics=intrinsics,
                policy=cfg['policy'],
                ratio=cfg['ratio'],
                budget_ms=cfg['budget_ms'],
                device=device,
            )
            res['name'] = cfg['name']
            all_seed_results[cfg['name']].append(res)
            print(f"  {cfg['name']:<22}: PSNR={res['avg_psnr']:.2f}dB, Time={res['avg_frame_time_ms']:.1f}ms, Opt={res['avg_n_optimized']:.0f}")

    # Aggregate across seeds
    aggregated = []
    for cfg in policy_configs:
        name = cfg['name']
        runs = all_seed_results[name]
        psnrs = [r['avg_psnr'] for r in runs]
        depths = [r['avg_depth_l1'] for r in runs]
        times = [r['avg_frame_time_ms'] for r in runs]
        n_opts = [r['avg_n_optimized'] for r in runs]
        n_totals = [r['final_n_gaussians'] for r in runs]
        
        mean_psnr, std_psnr = float(np.mean(psnrs)), float(np.std(psnrs))
        mean_depth = float(np.mean(depths))
        mean_time, std_time = float(np.mean(times)), float(np.std(times))
        mean_n_opt = float(np.mean(n_opts))
        mean_n_total = float(np.mean(n_totals))
        opt_ratio = mean_n_opt / max(mean_n_total, 1.0)
        
        aggregated.append({
            'name': name,
            'policy': str(cfg['policy'].value if hasattr(cfg['policy'], 'value') else cfg['policy']),
            'mean_psnr': mean_psnr,
            'std_psnr': std_psnr,
            'mean_depth_l1': mean_depth,
            'mean_frame_time_ms': mean_time,
            'std_frame_time_ms': std_time,
            'mean_n_optimized': mean_n_opt,
            'mean_n_frozen': mean_n_total - mean_n_opt,
            'mean_n_total': mean_n_total,
            'opt_ratio': opt_ratio,
            'fps': 1000.0 / max(mean_time, 1e-4),
        })

    return {'aggregated': aggregated, 'seeds': seeds}


def main():
    parser = argparse.ArgumentParser(description="R11 Policy Benchmark")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44], help='List of random seeds')
    parser.add_argument('--output-dir', type=str, default='results/policies/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results/figures/', exist_ok=True)

    print(f"[R11] Running Policy Benchmark across seeds={args.seeds} ({args.frames} frames each)...")
    results = run_multi_seed_policy_benchmark(seeds=args.seeds, n_frames=args.frames, device=args.device)

    print("\n" + "=" * 95)
    print("              R11: OPTIMIZATION POLICY BENCHMARK (MULTI-SEED EVALUATION)")
    print("=" * 95)
    print(f"| {'Policy':<22} | {'Opt %':<8} | {'PSNR (dB)':<16} | {'Depth L1':<10} | {'Frame Time (ms)':<18} | {'FPS':<6} |")
    print("|------------------------|----------|------------------|------------|--------------------|--------|")
    
    for row in results['aggregated']:
        name = row['name']
        opt_pct = f"{row['opt_ratio']:.1%}"
        psnr_str = f"{row['mean_psnr']:.2f} ± {row['std_psnr']:.2f}"
        depth_str = f"{row['mean_depth_l1']:.4f}"
        time_str = f"{row['mean_frame_time_ms']:.1f} ± {row['std_frame_time_ms']:.1f}"
        fps_str = f"{row['fps']:.1f}"
        print(f"| {name:<22} | {opt_pct:<8} | {psnr_str:<16} | {depth_str:<10} | {time_str:<18} | {fps_str:<6} |")
        
    print("=" * 95 + "\n")

    # Save JSON results
    out_path = os.path.join(args.output_dir, 'policy_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {out_path}")

    # Save F5: Quality vs Compute CSV
    f5_path = 'results/figures/f5_quality_vs_compute.csv'
    with open(f5_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['policy_name', 'optimized_ratio_pct', 'mean_psnr', 'std_psnr', 'mean_frame_time_ms', 'mean_n_optimized', 'mean_n_frozen'])
        for row in results['aggregated']:
            writer.writerow([
                row['name'],
                f"{row['opt_ratio'] * 100.0:.2f}",
                f"{row['mean_psnr']:.4f}",
                f"{row['std_psnr']:.4f}",
                f"{row['mean_frame_time_ms']:.2f}",
                f"{row['mean_n_optimized']:.1f}",
                f"{row['mean_n_frozen']:.1f}",
            ])
    print(f"Saved F5 Pareto curve data to {f5_path}")


if __name__ == '__main__':
    main()
