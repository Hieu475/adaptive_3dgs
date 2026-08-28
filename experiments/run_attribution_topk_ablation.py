"""R15/R16: Attribution Pixel Top-K Ablation Experiment.

Investigates: How many Gaussian contributions per pixel (K ∈ {1, 4, 8, 16}) are needed
to accurately attribute reconstruction error without excessive memory/compute overhead?

Measures:
    - Spearman rank correlation ρ(I, E_comb)
    - Error Capture Coverage@10%
    - Frame Time (ms)
    - Reconstructed PSNR (dB)
"""
import os
import sys
import json
import csv
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.pipeline import OnlineReconstructionPipeline
from research.importance_diagnostics import spearman_rank_correlation
from experiments.run_importance_validation import generate_synthetic_benchmark_dataset, compute_coverage_at_k


def run_attribution_k_experiment(k_val: int, frames: list, intrinsics: torch.Tensor, device: str = 'cpu') -> dict:
    cfg = {
        'rendering': {
            'attribution_top_k': k_val,
            'use_surface_aware_depth': True,
        }
    }
    pipeline = OnlineReconstructionPipeline(config=cfg, device=device)
    pipeline.initialize(
        rgb=frames[0]['rgb'],
        depth=frames[0]['depth'],
        intrinsics=intrinsics,
        pose=frames[0]['pose'],
    )

    for i in range(1, len(frames)):
        pipeline.process_frame(
            rgb=frames[i]['rgb'],
            depth=frames[i]['depth'],
            gt_pose=frames[i]['pose'],
        )

    state = pipeline.get_importance_diagnostics()
    importance = state['importance']
    combined_error = state['color_error'] + state['depth_error']

    rho = spearman_rank_correlation(importance, combined_error)
    cov = compute_coverage_at_k(importance, combined_error, k_percentages=[10, 25, 50])
    summary = pipeline.get_metrics_summary()

    return {
        'top_k': k_val,
        'spearman_rho': float(rho),
        'coverage_at_10': float(cov[10]),
        'coverage_at_25': float(cov[25]),
        'coverage_at_50': float(cov[50]),
        'avg_psnr': summary.get('avg_psnr', 0.0),
        'avg_frame_time_ms': summary.get('avg_frame_time_ms', 0.0),
        'final_gaussians': summary.get('final_n_gaussians', 0),
    }


def main():
    parser = argparse.ArgumentParser(description="R15/R16 Attribution Top-K Ablation")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=8, help='Number of frames')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/importance/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results/figures/', exist_ok=True)

    print(f"[R15] Running Attribution Top-K Sweep across K in [1, 4, 8, 16] (seed={args.seed})...")
    frames, intrinsics = generate_synthetic_benchmark_dataset(n_frames=args.frames, seed=args.seed)

    k_values = [1, 4, 8, 16]
    ablation_records = []

    for k in k_values:
        print(f"  Evaluating Pixel Attribution Top-K = {k}...")
        res = run_attribution_k_experiment(k, frames, intrinsics, device=args.device)
        ablation_records.append(res)
        print(f"    K={k:>2}: Spearman ρ={res['spearman_rho']:.4f}, Coverage@10%={res['coverage_at_10']:.1%}, Time={res['avg_frame_time_ms']:.1f}ms")

    print("\n" + "=" * 80)
    print("           R15/R16: PIXEL ATTRIBUTION TOP-K ABLATION RESULTS")
    print("=" * 80)
    print(f"| {'Top-K':<8} | {'Spearman ρ':<12} | {'Coverage@10%':<15} | {'PSNR (dB)':<12} | {'Frame Time (ms)':<18} |")
    print("|----------|--------------|-----------------|--------------|--------------------|")
    for r in ablation_records:
        print(f"| {r['top_k']:<8} | {r['spearman_rho']:>12.4f} | {r['coverage_at_10']:>15.1%} | {r['avg_psnr']:>12.2f} | {r['avg_frame_time_ms']:>18.1f} |")
    print("=" * 80 + "\n")

    # Save JSON
    out_path = os.path.join(args.output_dir, 'topk_ablation.json')
    with open(out_path, 'w') as f:
        json.dump({'ablation': ablation_records}, f, indent=4)
    print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
