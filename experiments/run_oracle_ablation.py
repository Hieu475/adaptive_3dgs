#!/usr/bin/env python3
"""Oracle Utility Ablation — Compare 3 importance configurations.

Runs the oracle experiment 3 times with different importance settings:
  A: Baseline (no novelty, no error prior, no uncertainty)
  B: + Novelty boost + Error prior
  C: + Novelty + Error prior + Uncertainty

Same seed, same data, same Gaussians → clean comparison.

Output:
    results/oracle_ablation/ablation_results.json
"""
import os
import sys
import json
import time
import torch
import numpy as np
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.pipeline import OnlineReconstructionPipeline
from research.oracle_utility import OracleUtilityExperiment
from research.uncertainty import GaussianUncertaintyEstimator


def create_frames(n_frames=12, H=64, W=80):
    """Structured synthetic frames."""
    fx, fy = 160.0, 160.0
    intrinsics = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=torch.float32)
    
    frames = []
    for t in range(n_frames):
        angle = t * 0.03
        pose = torch.eye(4)
        pose[0, 0] = np.cos(angle); pose[0, 2] = np.sin(angle)
        pose[2, 0] = -np.sin(angle); pose[2, 2] = np.cos(angle)
        pose[0, 3] = 0.02 * t
        
        rgb = torch.zeros(H, W, 3)
        depth = torch.ones(H, W) * 3.0
        
        # Checkerboard (top-left)
        for i in range(H // 2):
            for j in range(W // 2):
                if (i // 8 + j // 8) % 2 == 0:
                    rgb[i, j] = torch.tensor([0.9, 0.2, 0.1])
                else:
                    rgb[i, j] = torch.tensor([0.1, 0.8, 0.2])
        depth[:H//2, :W//2] = 2.0
        
        # Smooth (top-right)
        rgb[:H//2, W//2:] = torch.tensor([0.4, 0.4, 0.6])
        depth[:H//2, W//2:] = 2.5
        
        # Box (bottom-center)
        box_h = slice(H//2 + 5, H - 5)
        box_w = slice(W//4, 3*W//4)
        rgb[box_h, box_w] = torch.tensor([0.7, 0.3, 0.5])
        depth[box_h, box_w] = 1.0
        
        depth[H-10:, :10] = 0.0
        
        rgb = (rgb + 0.03 * torch.randn_like(rgb)).clamp(0, 1)
        depth = depth + 0.01 * torch.randn_like(depth)
        depth[depth <= 0] = 0.0
        
        frames.append({'rgb': rgb, 'depth': depth, 'pose': pose, 'intrinsics': intrinsics})
    
    return frames, intrinsics


def make_pipeline(config, device='cpu'):
    return OnlineReconstructionPipeline(config=config, device=device)


def run_single_config(config_name, frames, intrinsics, n_warmup, n_samples, n_opt_steps,
                      novelty_weight, use_error_prior, use_uncertainty):
    """Run oracle experiment with a specific importance config."""
    print(f"\n{'='*60}")
    print(f"  CONFIG {config_name}")
    print(f"  novelty={novelty_weight}, error_prior={use_error_prior}, uncertainty={use_uncertainty}")
    print(f"{'='*60}")
    
    # Deterministic seeding
    torch.manual_seed(42)
    np.random.seed(42)
    
    config = {
        'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 20000,
                     'initial_scale': 0.02},
        'rendering': {'tile_size': 16,
                      'image_width': frames[0]['rgb'].shape[1],
                      'image_height': frames[0]['rgb'].shape[0],
                      'use_surface_aware_depth': True,
                      'attribution_top_k': 4},
        'scheduler': {'gpu_budget_ms': 100.0, 'policy': 'budget_aware', 'optimize_ratio': 0.7},
        'densification': {'max_new_per_frame': 100, 'strategy': 'importance',
                          'use_adaptive_thresholds': True},
    }
    
    pipeline = make_pipeline(config, device='cpu')
    
    # Configure importance estimator
    pipeline.importance_estimator.novelty_weight = novelty_weight
    pipeline.importance_estimator.use_error_prior = use_error_prior
    
    # Wire uncertainty if enabled
    uncertainty_est = None
    if use_uncertainty:
        uncertainty_est = GaussianUncertaintyEstimator(ema_decay=0.95)
        pipeline.importance_estimator.set_uncertainty_estimator(uncertainty_est)
        pipeline.importance_estimator.uncertainty_weight = 0.3
    else:
        pipeline.importance_estimator.uncertainty_weight = 0.0
    
    # Warmup
    f0 = frames[0]
    pipeline.initialize(f0['rgb'], f0['depth'], intrinsics, f0['pose'])
    print(f"  Init: {pipeline.gaussian_model.num_gaussians} Gaussians")
    
    for i in range(1, min(n_warmup + 1, len(frames))):
        f = frames[i]
        m = pipeline.process_frame(f['rgb'], f['depth'], gt_pose=f['pose'])
        
        # Update uncertainty estimator if active
        if uncertainty_est is not None:
            diag = pipeline.get_importance_diagnostics()
            components = diag.get('components', {})
            color_err = components.get('color_error', torch.zeros(pipeline.gaussian_model.num_gaussians))
            depth_err = components.get('depth_error', torch.zeros(pipeline.gaussian_model.num_gaussians))
            # Ensure sizes match
            n = pipeline.gaussian_model.num_gaussians
            if color_err.shape[0] != n:
                color_err = torch.zeros(n)
                depth_err = torch.zeros(n)
            if uncertainty_est._ema_error is not None and uncertainty_est._ema_error.shape[0] != n:
                uncertainty_est.expand_buffers(n - uncertainty_est._ema_error.shape[0], color_err.device)
            uncertainty_est.update(color_err[:n], depth_err[:n])
        
        if i == n_warmup:
            print(f"  [Frame {i}] PSNR={m['psnr']:.2f} | N={m['n_gaussians']} | Opt={m['n_optimized']}")
    
    N = pipeline.gaussian_model.num_gaussians
    print(f"  Warmup done: {N} Gaussians")
    
    # Oracle experiment
    last_f = frames[min(n_warmup, len(frames)-1)]
    
    experiment = OracleUtilityExperiment(
        pipeline=pipeline,
        n_samples=n_samples,
        n_opt_steps=n_opt_steps,
        seed=42,
        contribution_threshold=0.01,
    )
    
    t0 = time.time()
    results = experiment.run_oracle_experiment(last_f['rgb'], last_f['depth'])
    elapsed = time.time() - t0
    
    corr = experiment.compute_correlation_metrics(results)
    
    if 'error' not in corr:
        rho_imp = corr['spearman_importance_vs_oracle']
        rho_util = corr['spearman_utility_vs_oracle']
        rho_dq = corr['spearman_importance_vs_deltaQ']
        p_imp = corr['spearman_importance_p']
        n_vis = corr['n_visible']
        
        print(f"\n  ρ(importance, oracle) = {rho_imp:.4f} (p={p_imp:.4f}, n={n_vis})")
        print(f"  ρ(importance, ΔQ)     = {rho_dq:.4f}")
        
        overlaps = corr.get('overlaps', {})
        print(f"  Overlap@10% = {overlaps.get('importance_top_10pct', 0):.0%}")
        print(f"  Overlap@20% = {overlaps.get('importance_top_20pct', 0):.0%}")
        
        gains = corr.get('realized_gains', {})
        print(f"  Gain ratio@20% = {gains.get('importance_top_20pct_ratio', 0):.4f}")
    else:
        rho_imp = rho_util = rho_dq = 0.0
        p_imp = 1.0
        n_vis = 0
        print(f"  ⚠️ {corr['error']}")
    
    print(f"  Time: {elapsed:.1f}s")
    
    return {
        'config_name': config_name,
        'novelty_weight': novelty_weight,
        'use_error_prior': use_error_prior,
        'use_uncertainty': use_uncertainty,
        'n_gaussians': N,
        'n_visible': n_vis,
        'correlation': corr,
        'time_s': elapsed,
    }


def main():
    n_warmup = 6
    n_samples = 150
    n_opt_steps = 10
    
    print("Oracle Utility Ablation Study")
    print(f"n_warmup={n_warmup}, n_samples={n_samples}, n_opt_steps={n_opt_steps}")
    
    # Create shared data (same seed)
    torch.manual_seed(42)
    np.random.seed(42)
    frames, intrinsics = create_frames(n_frames=n_warmup + 3)
    
    configs = [
        ("A: Baseline",           0.0, False, False),
        ("B: +Novelty+Prior",     0.5, True,  False),
        ("C: +Nov+Prior+Uncert",  0.5, True,  True),
    ]
    
    all_results = []
    for name, nov_w, err_prior, use_unc in configs:
        result = run_single_config(
            name, frames, intrinsics, n_warmup, n_samples, n_opt_steps,
            novelty_weight=nov_w, use_error_prior=err_prior, use_uncertainty=use_unc)
        all_results.append(result)
    
    # === Summary Table ===
    print(f"\n\n{'='*70}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'ρ(imp,oracle)':>13} {'ρ(imp,ΔQ)':>10} {'p-value':>8} {'Ov@20%':>7} {'Gain@20%':>8}")
    print(f"  {'-'*25} {'-'*13} {'-'*10} {'-'*8} {'-'*7} {'-'*8}")
    
    for r in all_results:
        c = r['correlation']
        if 'error' in c:
            print(f"  {r['config_name']:<25} {'N/A':>13} {'N/A':>10} {'N/A':>8} {'N/A':>7} {'N/A':>8}")
            continue
        rho = c['spearman_importance_vs_oracle']
        rho_dq = c['spearman_importance_vs_deltaQ']
        p = c['spearman_importance_p']
        ov = c.get('overlaps', {}).get('importance_top_20pct', 0)
        gain = c.get('realized_gains', {}).get('importance_top_20pct_ratio', 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {r['config_name']:<25} {rho:>10.4f}{sig:>3} {rho_dq:>10.4f} {p:>8.4f} {ov:>6.0%} {gain:>8.4f}")
    
    print(f"{'='*70}")
    print(f"  * p<0.05  ** p<0.01  *** p<0.001")
    
    # Save
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'oracle_ablation')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'ablation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {save_dir}/ablation_results.json")


if __name__ == '__main__':
    main()
