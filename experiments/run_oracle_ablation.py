#!/usr/bin/env python3
"""Systematic Research Ablation Study for Importance Formulation.

Evaluates progressive feature additions:
  - V0: Base (Error + Influence Mass)
  - V1: + Temporal Dynamics
  - V2: + Uncertainty Estimation
  - V3: + Temporal Hysteresis & Error Prior (Full Proposed System)

Evaluates:
  1. Spearman correlation with Oracle Utility: ρ(Importance, U_oracle)
  2. Top-K ranking overlap (Overlap@10%, Overlap@20%)
  3. Realized Quality Gain Ratio @20%
  4. Tier Switch Rate (switches / frame)
  5. Optimization Jitter & Mean Latency
"""
import os
import sys
import json
import time
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.pipeline import OnlineReconstructionPipeline
from research.oracle_utility import OracleUtilityExperiment, SamplingPopulation
from research.uncertainty import GaussianUncertaintyEstimator


def create_ablation_frames(n_frames: int = 10, H: int = 64, W: int = 80):
    """Generate deterministic structured frames for ablation benchmark."""
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
        
        # High texture
        for i in range(H // 2):
            for j in range(W // 2):
                if (i // 8 + j // 8) % 2 == 0:
                    rgb[i, j] = torch.tensor([0.9, 0.2, 0.1])
                else:
                    rgb[i, j] = torch.tensor([0.1, 0.8, 0.2])
        depth[:H//2, :W//2] = 2.0
        
        # Smooth flat region
        rgb[:H//2, W//2:] = torch.tensor([0.4, 0.4, 0.6])
        depth[:H//2, W//2:] = 2.5
        
        # Object edge box
        box_h = slice(H // 2 + 5, H - 5)
        box_w = slice(W // 4, 3 * W // 4)
        rgb[box_h, box_w] = torch.tensor([0.7, 0.3, 0.5])
        depth[box_h, box_w] = 1.0
        
        depth[H - 10:, :10] = 0.0
        rgb = (rgb + 0.02 * torch.randn_like(rgb)).clamp(0, 1)
        depth = depth + 0.01 * torch.randn_like(depth)
        depth[depth <= 0] = 0.0
        
        frames.append({'rgb': rgb, 'depth': depth, 'pose': pose, 'intrinsics': intrinsics})
        
    return frames, intrinsics


def run_variant(variant_name, weights, use_uncertainty, use_hysteresis, use_prior, frames, intrinsics, args):
    """Evaluate a single ablation variant."""
    print("\n" + "=" * 72)
    print(f"EVALUATING VARIANT: {variant_name}")
    print(f"Weights: {weights}")
    print(f"Flags: uncertainty={use_uncertainty}, hysteresis={use_hysteresis}, error_prior={use_prior}")
    print("=" * 72)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    config = {
        'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 20000, 'initial_scale': 0.02},
        'rendering': {
            'tile_size': 16,
            'image_width': frames[0]['rgb'].shape[1],
            'image_height': frames[0]['rgb'].shape[0],
            'use_surface_aware_depth': True,
            'attribution_top_k': 4
        },
        'scheduler': {
            'gpu_budget_ms': 50.0,
            'policy': 'budget_aware',
            'optimize_ratio': 0.6,
        },
        'densification': {
            'max_new_per_frame': 80,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        },
    }
    
    pipeline = OnlineReconstructionPipeline(config=config, device=args.device)
    pipeline.importance_estimator.weights = weights
    pipeline.importance_estimator.hysteresis_enabled = use_hysteresis
    pipeline.importance_estimator.use_error_prior = use_prior
    pipeline.importance_estimator.novelty_weight = 0.5 if use_prior else 0.0
    
    uncertainty_est = None
    if use_uncertainty:
        uncertainty_est = GaussianUncertaintyEstimator(ema_decay=0.95)
        pipeline.importance_estimator.set_uncertainty_estimator(uncertainty_est)
        pipeline.importance_estimator.uncertainty_weight = 0.3
    else:
        pipeline.importance_estimator.uncertainty_weight = 0.0
        
    f0 = frames[0]
    pipeline.initialize(f0['rgb'].to(args.device), f0['depth'].to(args.device), intrinsics.to(args.device), f0['pose'].to(args.device))
    
    opt_latencies = []
    
    for i in range(1, min(args.n_warmup + 1, len(frames))):
        f = frames[i]
        m = pipeline.process_frame(f['rgb'].to(args.device), f['depth'].to(args.device), gt_pose=f['pose'].to(args.device))
        opt_latencies.append(m['opt_time_ms'])
        
        if uncertainty_est is not None:
            diag = pipeline.get_importance_diagnostics()
            comp = diag.get('components', {})
            n = pipeline.gaussian_model.num_gaussians
            c_err = comp.get('color_error', torch.zeros(n, device=args.device))
            d_err = comp.get('depth_error', torch.zeros(n, device=args.device))
            if uncertainty_est._ema_error is not None and uncertainty_est._ema_error.shape[0] != n:
                uncertainty_est.expand_buffers(n - uncertainty_est._ema_error.shape[0], args.device)
            uncertainty_est.update(c_err[:n], d_err[:n])
            
    # Measure Hysteresis Diagnostics
    hyst_diag = pipeline.importance_estimator.get_hysteresis_diagnostics()
    switch_rate = hyst_diag.get('switch_rate_per_frame', 0.0) if use_hysteresis else 0.0
    
    # Run Oracle Evaluation
    last_frame = frames[min(args.n_warmup, len(frames) - 1)]
    experiment = OracleUtilityExperiment(
        pipeline=pipeline,
        n_samples=args.n_samples,
        n_opt_steps=args.n_opt_steps,
        w_rgb=0.7,
        w_depth=0.3,
        seed=42
    )
    
    oracle_res = experiment.run_oracle_experiment(
        last_frame['rgb'].to(args.device),
        last_frame['depth'].to(args.device),
        population_type=SamplingPopulation.IMPORTANCE_STRATIFIED
    )
    corr = experiment.compute_correlation_metrics(oracle_res)
    
    jitter = float(np.std(opt_latencies)) if len(opt_latencies) > 1 else 0.0
    mean_lat = float(np.mean(opt_latencies)) if opt_latencies else 0.0
    
    print(f"Result for {variant_name}:")
    print(f"  • ρ(Utility, Oracle): {corr.get('spearman_utility_vs_oracle', 0.0):+.4f}")
    print(f"  • Top-10% Overlap:   {corr.get('overlaps', {}).get('top_10pct', 0.0):.1%}")
    print(f"  • Top-20% Overlap:   {corr.get('overlaps', {}).get('top_20pct', 0.0):.1%}")
    print(f"  • Coverage@10%:      {corr.get('coverages', {}).get('top_10pct', 0.0):.4f}")
    print(f"  • Regret@10%:        {corr.get('regrets', {}).get('top_10pct', 0.0):.4f}")
    print(f"  • Realized Gain@20%: {corr.get('realized_gains', {}).get('top_20pct_ratio', 0.0):.4f}")
    print(f"  • Switch Rate:       {switch_rate:.2f} switches/frame")
    print(f"  • Latency Jitter:    {jitter:.2f} ms")
    
    return {
        'variant': variant_name,
        'spearman_utility_oracle': corr.get('spearman_utility_vs_oracle', 0.0),
        'spearman_p': corr.get('spearman_utility_p', 1.0),
        'overlap_10pct': corr.get('overlaps', {}).get('top_10pct', 0.0),
        'overlap_20pct': corr.get('overlaps', {}).get('top_20pct', 0.0),
        'coverage_10pct': corr.get('coverages', {}).get('top_10pct', 0.0),
        'regret_10pct': corr.get('regrets', {}).get('top_10pct', 0.0),
        'realized_gain_20pct': corr.get('realized_gains', {}).get('top_20pct_ratio', 0.0),
        'switch_rate': switch_rate,
        'jitter_ms': jitter,
        'mean_latency_ms': mean_lat,
        'final_psnr': float(m['psnr']),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Systematic Importance Ablation Suite")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--frames', type=int, default=6)
    parser.add_argument('--n_warmup', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=40)
    parser.add_argument('--n_opt_steps', type=int, default=5)
    args = parser.parse_args()
    if args.frames:
        args.n_warmup = args.frames
        
    print("=" * 95)
    print("      SYSTEMATIC IMPORTANCE ABLATION MATRIX (V0 → V1 → V2 → V3 → V4 → V5 → V6)")
    print("=" * 95)
    
    frames, intrinsics = create_ablation_frames(n_frames=args.n_warmup + 3)
    
    # 7 Progressive Variants
    variants = [
        ("V0: Error Only",
         {'depth_error': 1.0, 'color_error': 1.0, 'normal_error': 0.0, 'visibility': 0.0, 'temporal': 0.0, 'screen_space': 0.0},
         False, False, False),
        ("V1: Error + Influence",
         {'depth_error': 1.0, 'color_error': 1.0, 'normal_error': 0.0, 'visibility': 0.1, 'temporal': 0.0, 'screen_space': 0.2},
         False, False, False),
        ("V2: + Temporal Dynamics",
         {'depth_error': 1.0, 'color_error': 1.0, 'normal_error': 0.0, 'visibility': 0.1, 'temporal': 0.5, 'screen_space': 0.2},
         False, False, False),
        ("V3: + Uncertainty",
         {'depth_error': 1.0, 'color_error': 1.0, 'normal_error': 0.0, 'visibility': 0.1, 'temporal': 0.5, 'screen_space': 0.2},
         True, False, False),
        ("V4: + Projected Area",
         {'depth_error': 1.0, 'color_error': 1.0, 'normal_error': 0.0, 'visibility': 0.1, 'temporal': 0.5, 'screen_space': 0.5},
         True, False, False),
        ("V5: + Hysteresis",
         {'depth_error': 1.0, 'color_error': 1.0, 'normal_error': 0.0, 'visibility': 0.1, 'temporal': 0.5, 'screen_space': 0.5},
         True, True, False),
        ("V6: Full Utility (+ Prior)",
         {'depth_error': 1.0, 'color_error': 1.0, 'normal_error': 0.0, 'visibility': 0.1, 'temporal': 0.5, 'screen_space': 0.5},
         True, True, True),
    ]
    
    results = []
    for name, w, use_unc, use_hyst, use_prior in variants:
        res = run_variant(name, w, use_unc, use_hyst, use_prior, frames, intrinsics, args)
        results.append(res)
        
    # Print Comparison Table
    print("\n\n" + "=" * 115)
    print("                                      ABLATION STUDY SUMMARY MATRIX")
    print("=" * 115)
    print(f"{'Variant':<25} | {'ρ(Util,Oracle)':>14} | {'Ov@10%':>8} | {'Cov@10%':>8} | {'Regret@10%':>11} | {'Jitter':>8} | {'Switches':>8} | {'PSNR':>8}")
    print("-" * 115)
    for r in results:
        print(f"{r['variant']:<25} | {r['spearman_utility_oracle']:>14.4f} | {r['overlap_10pct']:>7.1%} | {r['coverage_10pct']:>8.4f} | {r['regret_10pct']:>11.4f} | {r['jitter_ms']:>6.2f}ms | {r['switch_rate']:>7.2f} | {r['final_psnr']:>6.2f}dB")
    print("=" * 115)
    
    # Save results
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'importance_ablation')
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, 'ablation_summary.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    md_file = os.path.join(save_dir, 'ablation_matrix.md')
    with open(md_file, 'w') as f:
        f.write("# R35: Systematic Utility Ablation Matrix (V0 → V6)\n\n")
        f.write("| Variant | $\\rho(U, U_{oracle})$ | Overlap@10% | Coverage@10% | Regret@10% | Jitter (ms) | Switch Rate | PSNR (dB) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results:
            f.write(f"| **{r['variant']}** | {r['spearman_utility_oracle']:.4f} | {r['overlap_10pct']*100:.1f}% | {r['coverage_10pct']:.4f} | {r['regret_10pct']:.4f} | {r['jitter_ms']:.2f} ms | {r['switch_rate']:.2f} | {r['final_psnr']:.2f} dB |\n")
        f.write("\n")
        
    print(f"\nAblation artifacts saved to:")
    print(f"  - {out_file}")
    print(f"  - {md_file}")


if __name__ == '__main__':
    main()
