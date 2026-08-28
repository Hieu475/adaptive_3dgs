"""R12: Densification Convergence & Geometric Growth Benchmark.

Evaluates hypothesis H3: Does importance-weighted densification converge faster
to high geometric and photometric fidelity than uniform / error-driven densification?

Measures:
    - PSNR(t) and DepthL1(t) trajectories per frame
    - t_90 and t_95 convergence milestones
    - Final Gaussian footprint and efficiency
"""
import os
import sys
import json
import csv
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.benchmark_densification import run_full_densification_ablation, format_densification_table
from experiments.run_importance_validation import generate_synthetic_benchmark_dataset


def compute_convergence_milestones(psnr_curve: list, target_psnr: float):
    """Compute frame indices where PSNR first reaches 90% and 95% of target."""
    t90, t95 = len(psnr_curve), len(psnr_curve)
    thresh_90 = 0.90 * target_psnr
    thresh_95 = 0.95 * target_psnr

    for idx, p in enumerate(psnr_curve):
        if p >= thresh_90 and t90 == len(psnr_curve):
            t90 = idx + 1
        if p >= thresh_95 and t95 == len(psnr_curve):
            t95 = idx + 1
    return t90, t95


def main():
    parser = argparse.ArgumentParser(description="R12 Densification Convergence")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=15, help='Number of frames to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/densification/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results/figures/', exist_ok=True)

    print(f"[R12] Generating {args.frames} frames with seed={args.seed}...")
    frames, intrinsics = generate_synthetic_benchmark_dataset(n_frames=args.frames, seed=args.seed)

    print("Running Densification Strategy Ablation & Convergence Trajectories...")
    ablation = run_full_densification_ablation(frames=frames, intrinsics=intrinsics, device=args.device)

    # Determine max PSNR target across all runs
    max_psnr_overall = max(
        max(m['psnr'] for m in exp['per_frame_records']) if exp['per_frame_records'] else 0.0
        for exp in ablation['experiments']
    )

    convergence_summary = []

    for exp in ablation['experiments']:
        psnr_curve = [m['psnr'] for m in exp['per_frame_records']]
        depth_curve = [m['depth_l1'] for m in exp['per_frame_records']]
        n_g_curve = [m['n_gaussians'] for m in exp['per_frame_records']]

        t90, t95 = compute_convergence_milestones(psnr_curve, max_psnr_overall)
        exp['t90'] = t90
        exp['t95'] = t95
        exp['psnr_trajectory'] = psnr_curve
        exp['depth_trajectory'] = depth_curve
        exp['gaussians_trajectory'] = n_g_curve

        convergence_summary.append({
            'name': exp['name'],
            'strategy': exp['strategy'],
            'adaptive_thresholds': exp['use_adaptive_thresholds'],
            'final_psnr': exp['avg_psnr'],
            'final_depth_l1': exp['avg_depth_l1'],
            'final_n_gaussians': exp['final_n_gaussians'],
            't90_frame': t90,
            't95_frame': t95,
        })

    # Save JSON results
    out_path = os.path.join(args.output_dir, 'convergence.json')
    with open(out_path, 'w') as f:
        json.dump(ablation, f, indent=4)
    print(f"Saved results to {out_path}")

    # Print summary table
    print("\n" + "=" * 85)
    print("                 R12: DENSIFICATION CONVERGENCE & GROWTH SUMMARY")
    print("=" * 85)
    print(f"| {'Strategy':<35} | {'Final PSNR':<12} | {'Depth L1':<10} | {'t_90':<6} | {'t_95':<6} | {'Gaussians':<10} |")
    print("|-------------------------------------|--------------|------------|--------|--------|------------|")
    for s in convergence_summary:
        print(f"| {s['name']:<35} | {s['final_psnr']:>10.2f}dB | {s['final_depth_l1']:>10.4f} | {s['t90_frame']:>6} | {s['t95_frame']:>6} | {s['final_n_gaussians']:>10} |")
    print("=" * 85 + "\n")

    # Save F6: Densification Convergence CSV (frame, uniform_psnr, error_psnr, imp_fixed_psnr, imp_adaptive_psnr)
    f6_path = 'results/figures/f6_densification_convergence.csv'
    with open(f6_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'uniform_psnr', 'error_psnr', 'importance_fixed_psnr', 'importance_adaptive_psnr'])
        n_pts = len(ablation['experiments'][0]['psnr_trajectory'])
        for f_idx in range(n_pts):
            row = [f_idx + 1]
            for exp in ablation['experiments']:
                row.append(f"{exp['psnr_trajectory'][f_idx]:.4f}")
            writer.writerow(row)
    print(f"Saved F6 convergence trajectory data to {f6_path}")

    # Save F8: Gaussian Count over Time CSV
    f8_path = 'results/figures/f8_gaussian_count_over_time.csv'
    with open(f8_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'uniform_gaussians', 'error_gaussians', 'importance_fixed_gaussians', 'importance_adaptive_gaussians'])
        for f_idx in range(n_pts):
            row = [f_idx + 1]
            for exp in ablation['experiments']:
                row.append(exp['gaussians_trajectory'][f_idx])
            writer.writerow(row)
    print(f"Saved F8 Gaussian count trajectory data to {f8_path}")


if __name__ == '__main__':
    main()
