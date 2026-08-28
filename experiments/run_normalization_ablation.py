"""R10: Importance Normalization Ablation.

Evaluates normalization strategies:
    - Raw components
    - Z-Score normalization (mean=0, std=1)
    - Robust MAD normalization (median=0, MAD=1)

Metrics:
    - Spearman rank correlation ρ(I, E)
    - Importance Calibration R² & Bin Error
"""
import os
import sys
import argparse
import json
import csv
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.pipeline import OnlineReconstructionPipeline
from research.importance_diagnostics import (
    spearman_rank_correlation,
    importance_calibration,
    pearson_correlation,
)
from research.attribution import normalize_importance_components
from experiments.run_importance_validation import generate_synthetic_benchmark_dataset


def main():
    parser = argparse.ArgumentParser(description="R10 Normalization Ablation")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/importance/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results/figures/', exist_ok=True)

    print(f"[R10] Generating {args.frames} frames with seed={args.seed}...")
    frames, intrinsics = generate_synthetic_benchmark_dataset(n_frames=args.frames, seed=args.seed)
    
    pipeline = OnlineReconstructionPipeline(device=args.device)
    pipeline.initialize(
        rgb=frames[0]['rgb'],
        depth=frames[0]['depth'],
        intrinsics=intrinsics,
        pose=frames[0]['pose'],
    )
    
    for i in range(1, len(frames)):
        frame = frames[i]
        pipeline.process_frame(
            rgb=frame['rgb'],
            depth=frame['depth'],
            gt_pose=frame['pose'],
        )
        
    state = pipeline.get_importance_diagnostics()
    components = state['components']
    actual_error = state['color_error'] + state['depth_error']

    methods = ['raw', 'zscore', 'robust']
    results = {}
    
    print("\n" + "=" * 65)
    print("           R10: NORMALIZATION STRATEGY ABLATION")
    print("=" * 65)
    print(f"| {'Method':<10} | {'Spearman ρ':<12} | {'Pearson r':<12} | {'Monotonicity':<15} |")
    print("|------------|--------------|--------------|-----------------|")
    
    for method in methods:
        norm_comps = normalize_importance_components(components, method=method)
        total_importance = sum(v for v in norm_comps.values())
        # Normalize to [0, 1] for calibration
        score_min = total_importance.min()
        score_max = total_importance.max()
        if score_max - score_min > 1e-8:
            imp_unit = (total_importance - score_min) / (score_max - score_min)
        else:
            imp_unit = torch.zeros_like(total_importance)
        
        rho = spearman_rank_correlation(imp_unit, actual_error)
        r_pearson = pearson_correlation(imp_unit, actual_error)
        
        calib = importance_calibration(imp_unit, actual_error, n_bins=10)
        mono = float(calib['monotonicity_score'])
        
        results[method] = {
            'spearman_rho': float(rho),
            'pearson_r': float(r_pearson),
            'monotonicity_score': mono,
            'bin_mean_importance': [float(x) for x in calib['bin_mean_importance'].cpu().numpy()],
            'bin_mean_error': [float(x) for x in calib['bin_mean_error'].cpu().numpy()],
        }
        
        print(f"| {method:<10} | {rho:>12.4f} | {r_pearson:>12.4f} | {mono:>15.2%} |")
        
    print("=" * 65 + "\n")
    
    # Save JSON results
    out_path = os.path.join(args.output_dir, 'normalization_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {out_path}")
    
    # Save F3: Calibration CSV (bin_mean_importance, raw_err, zscore_err, robust_err)
    f3_path = 'results/figures/f3_importance_calibration.csv'
    with open(f3_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bin_index', 'raw_importance', 'raw_error', 'zscore_importance', 'zscore_error', 'robust_importance', 'robust_error'])
        for b_idx in range(10):
            writer.writerow([
                b_idx,
                f"{results['raw']['bin_mean_importance'][b_idx]:.4f}",
                f"{results['raw']['bin_mean_error'][b_idx]:.6f}",
                f"{results['zscore']['bin_mean_importance'][b_idx]:.4f}",
                f"{results['zscore']['bin_mean_error'][b_idx]:.6f}",
                f"{results['robust']['bin_mean_importance'][b_idx]:.4f}",
                f"{results['robust']['bin_mean_error'][b_idx]:.6f}",
            ])
    print(f"Saved F3 calibration curve data to {f3_path}")



if __name__ == '__main__':
    main()
