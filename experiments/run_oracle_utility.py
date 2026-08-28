#!/usr/bin/env python3
"""Oracle Utility Experiment v2 — Local Quality Measurement.

Fixes from v1:
    - Measures LOCAL PSNR (at Gaussian's influence pixels) instead of global
    - Uses stratified sampling (high/mid/low importance)
    - Computes predicted_utility = importance / cost (not just importance)
    - Reports multiple correlation metrics

Usage:
    python experiments/run_oracle_utility.py [--synthetic] [--n_samples 100]

Output:
    results/oracle_utility/oracle_results.json
    results/oracle_utility/correlation_metrics.json
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
from research.oracle_utility import OracleUtilityExperiment


def create_structured_synthetic_frames(
    n_frames: int = 10, H: int = 64, W: int = 80, device: str = 'cpu'
):
    """Create synthetic frames with clear spatial structure.
    
    Scene contains:
    - Textured checkerboard region (high texture, high importance expected)
    - Smooth gradient region (flat surface, low importance expected)  
    - Depth discontinuity (object edge, high importance expected)
    - Invalid depth region (sparse depth)
    """
    print(f"[Data] Creating {n_frames} structured synthetic frames ({H}x{W})")
    
    fx, fy = 160.0, 160.0
    cx, cy = W / 2.0, H / 2.0
    intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    
    frames = []
    for t in range(n_frames):
        # Small camera motion
        angle = t * 0.03
        pose = torch.eye(4)
        pose[0, 0] = np.cos(angle); pose[0, 2] = np.sin(angle)
        pose[2, 0] = -np.sin(angle); pose[2, 2] = np.cos(angle)
        pose[0, 3] = 0.02 * t
        
        rgb = torch.zeros(H, W, 3)
        depth = torch.ones(H, W) * 3.0  # background
        
        # Region 1: Checkerboard (top-left) — high texture
        for i in range(H // 2):
            for j in range(W // 2):
                if (i // 8 + j // 8) % 2 == 0:
                    rgb[i, j] = torch.tensor([0.9, 0.2, 0.1])
                else:
                    rgb[i, j] = torch.tensor([0.1, 0.8, 0.2])
        depth[:H//2, :W//2] = 2.0
        
        # Region 2: Smooth gradient (top-right) — flat
        for j in range(W // 2, W):
            rgb[:H//2, j] = torch.tensor([0.4, 0.4, 0.6])
        depth[:H//2, W//2:] = 2.5
        
        # Region 3: Foreground box (bottom-center) — edge
        box_h = slice(H//2 + 5, H - 5)
        box_w = slice(W//4, 3*W//4)
        rgb[box_h, box_w] = torch.tensor([0.7, 0.3, 0.5])
        depth[box_h, box_w] = 1.0  # close object → depth discontinuity at edges
        
        # Region 4: Invalid depth (bottom-left corner)
        depth[H-10:, :10] = 0.0
        
        # Add per-frame noise
        rgb = (rgb + 0.03 * torch.randn_like(rgb)).clamp(0, 1)
        depth = depth + 0.01 * torch.randn_like(depth)
        depth[depth <= 0] = 0.0
        
        frames.append({'rgb': rgb, 'depth': depth, 'pose': pose, 'intrinsics': intrinsics})
    
    return frames, intrinsics


def run_experiment(args):
    device = 'cpu'
    print(f"[Config] Device: {device}")
    print(f"[Config] Warmup: {args.n_warmup} frames, Samples: {args.n_samples}, "
          f"Steps: {args.n_opt_steps}, Group: {args.group_size}")
    
    # === Load data ===
    frames, intrinsics = create_structured_synthetic_frames(
        n_frames=args.n_warmup + 3, H=args.height, W=args.width)
    
    # === Pipeline ===
    config = {
        'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 20000,
                     'initial_scale': 0.02},
        'rendering': {
            'tile_size': 16,
            'image_width': args.width,
            'image_height': args.height,
            'use_surface_aware_depth': True,
            'attribution_top_k': 4,
        },
        'scheduler': {
            'gpu_budget_ms': 100.0,
            'policy': 'budget_aware',
            'optimize_ratio': 0.7,
        },
        'densification': {
            'max_new_per_frame': 100,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        },
    }
    
    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    
    # === Warmup ===
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Warmup ({args.n_warmup} frames)")
    print(f"{'='*60}")
    
    f0 = frames[0]
    pipeline.initialize(f0['rgb'], f0['depth'], intrinsics, f0['pose'])
    print(f"  [Frame 0] Init: {pipeline.gaussian_model.num_gaussians} Gaussians")
    
    for i in range(1, min(args.n_warmup + 1, len(frames))):
        f = frames[i]
        m = pipeline.process_frame(f['rgb'], f['depth'], gt_pose=f['pose'])
        if i % 2 == 0 or i == args.n_warmup:
            print(f"  [Frame {i}] PSNR={m['psnr']:.2f} | N={m['n_gaussians']} | "
                  f"Opt={m['n_optimized']} | {m['frame_time_ms']:.0f}ms")
    
    N = pipeline.gaussian_model.num_gaussians
    print(f"\n  Warmup done. {N} Gaussians ready.")
    
    # === Oracle Experiment ===
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Oracle Utility Experiment")
    print(f"{'='*60}")
    
    last_f = frames[min(args.n_warmup, len(frames)-1)]
    oracle_rgb = last_f['rgb'].to(device)
    oracle_depth = last_f['depth'].to(device)
    
    actual_samples = min(args.n_samples, N)
    experiment = OracleUtilityExperiment(
        pipeline=pipeline,
        n_samples=actual_samples,
        n_opt_steps=args.n_opt_steps,
        seed=42,
        contribution_threshold=0.01,
        group_size=args.group_size,
    )
    
    t0 = time.time()
    results = experiment.run_oracle_experiment(oracle_rgb, oracle_depth)
    oracle_time = time.time() - t0
    print(f"\n  Completed in {oracle_time:.1f}s ({len(results)} evaluations)")
    
    # === Correlation ===
    print(f"\n{'='*60}")
    print(f"  PHASE 3: Correlation Analysis")
    print(f"{'='*60}")
    
    corr = experiment.compute_correlation_metrics(results)
    
    if 'error' in corr:
        print(f"\n  ⚠️  {corr['error']}")
    else:
        print(f"\n  Visible Gaussians: {corr['n_visible']} / {corr['n_total']}")
        print(f"\n  Spearman Correlations:")
        print(f"    ρ(predicted_utility, oracle_utility) = {corr['spearman_utility_vs_oracle']:.4f}  "
              f"(p={corr['spearman_utility_p']:.4f})")
        print(f"    ρ(importance, oracle_utility)         = {corr['spearman_importance_vs_oracle']:.4f}  "
              f"(p={corr['spearman_importance_p']:.4f})")
        print(f"    ρ(importance, ΔQ_local)               = {corr['spearman_importance_vs_deltaQ']:.4f}  "
              f"(p={corr['spearman_deltaQ_p']:.4f})")
        
        print(f"\n  Top-K Overlap (predicted vs oracle):")
        for k, v in corr['overlaps'].items():
            print(f"    {k}: {v:.0%}")
        
        print(f"\n  Realized Quality Gain Ratio:")
        for k, v in corr['realized_gains'].items():
            print(f"    {k}: {v:.4f}")
        
        dq = corr['delta_psnr_stats']
        print(f"\n  ΔQ_local distribution:")
        print(f"    mean={dq['mean']:.4f}  std={dq['std']:.4f}  "
              f"range=[{dq['min']:.4f}, {dq['max']:.4f}]  "
              f"dynamic_range={dq['range']:.4f}")
    
    # === Stats ===
    print(f"\n{'='*60}")
    print(f"  PHASE 4: Per-Sample Statistics")
    print(f"{'='*60}")
    
    visible = [r for r in results if r.get('visible', False)]
    if visible:
        pred_u = [r['predicted_utility'] for r in visible]
        oracle_u = [r['oracle_utility'] for r in visible]
        dpsnr = [r['delta_psnr_local'] for r in visible]
        pixels = [r['n_influence_pixels'] for r in visible]
        
        print(f"\n  Predicted Utility: mean={np.mean(pred_u):.4f} std={np.std(pred_u):.4f}")
        print(f"  Oracle Utility:   mean={np.mean(oracle_u):.4f} std={np.std(oracle_u):.4f}")
        print(f"  ΔQ_local (dB):    mean={np.mean(dpsnr):.4f} std={np.std(dpsnr):.4f}")
        print(f"  Influence pixels: mean={np.mean(pixels):.0f} std={np.std(pixels):.0f}")
        
        sorted_r = sorted(visible, key=lambda r: r['oracle_utility'], reverse=True)
        print(f"\n  Top 5 by Oracle Utility:")
        print(f"  {'ID':>6}  {'Pred_U':>8}  {'Oracle_U':>8}  {'ΔQ_local':>8}  {'Pixels':>6}")
        for r in sorted_r[:5]:
            print(f"  {r['gaussian_id']:>6}  {r['predicted_utility']:>8.4f}  "
                  f"{r['oracle_utility']:>8.4f}  {r['delta_psnr_local']:>8.4f}  "
                  f"{r['n_influence_pixels']:>6}")
        
        print(f"\n  Bottom 5:")
        for r in sorted_r[-5:]:
            print(f"  {r['gaussian_id']:>6}  {r['predicted_utility']:>8.4f}  "
                  f"{r['oracle_utility']:>8.4f}  {r['delta_psnr_local']:>8.4f}  "
                  f"{r['n_influence_pixels']:>6}")
    
    # === Save ===
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'oracle_utility')
    os.makedirs(save_dir, exist_ok=True)
    
    experiment.save_results(results, os.path.join(save_dir, 'oracle_results.json'))
    
    full_metrics = {
        'correlation': corr,
        'config': {
            'n_warmup': args.n_warmup, 'n_samples': actual_samples,
            'n_opt_steps': args.n_opt_steps, 'group_size': args.group_size,
            'n_gaussians': N, 'resolution': f'{args.height}x{args.width}',
        },
        'timing': {'warmup_s': None, 'oracle_s': oracle_time},
    }
    with open(os.path.join(save_dir, 'correlation_metrics.json'), 'w') as f:
        json.dump(full_metrics, f, indent=2)
    
    print(f"\n  Results saved to {save_dir}/")
    
    # === Verdict ===
    if 'error' not in corr:
        rho = corr['spearman_importance_vs_oracle']
        print(f"\n{'='*60}")
        if rho >= 0.5:
            print(f"  ✅ GOOD (ρ={rho:.3f} ≥ 0.5): Heuristic tracks oracle well")
        elif rho >= 0.2:
            print(f"  ⚠️  MODERATE (ρ={rho:.3f}): Partial signal, needs more features")
        else:
            print(f"  ❌ WEAK (ρ={rho:.3f} < 0.2): Heuristic misses key signal")
        print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Oracle Utility Experiment v2')
    parser.add_argument('--n_warmup', type=int, default=6)
    parser.add_argument('--n_samples', type=int, default=60)
    parser.add_argument('--n_opt_steps', type=int, default=10)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--synthetic', action='store_true', default=True)
    args = parser.parse_args()
    run_experiment(args)
