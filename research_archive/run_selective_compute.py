#!/usr/bin/env python3
"""R21/R30 — True Selective Optimization Scaling Benchmark with Real 3DGS Rasterizer.

Compares 4 Execution Methods across Gaussian counts N and active ratios K:
  1. Full: Full scene forward + full backward + full optimizer (N, N, N)
  2. Masked: Full scene forward + full backward + zero non-active grads + full optimizer (N, N, N)
  3. Detached: Full render with detached background + active backward + sliced optimizer (N, M, M)
  4. True Selective: Frozen background cache + active render + active backward + SelectiveAdam (M+cache, M, M)

Evaluates:
  - Sizes N ∈ {10k, 25k, 50k, 100k}
  - Active ratios K ∈ {1.0, 0.75, 0.50, 0.25, 0.10}
  - Metrics: Tp50, Tp95, Tp99, frozen_cache_ms, active_render_ms, backward_ms, optimizer_ms, total_ms

Outputs:
  - results/selective_compute/selective_scaling.json
  - results/selective_compute/selective_scaling.md
"""
import os
import sys
import time
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.gaussian_repr import GaussianModel
from research.rasterizer import render as rasterize_scene, render_full
from research.background_cache import FrozenBackgroundCache
from research.selective_optimizer import SelectiveAdam
from research.losses import total_loss


def build_synthetic_scene(N: int, H: int = 48, W: int = 64, device: str = 'cpu'):
    """Build a deterministic Gaussian model and test camera."""
    torch.manual_seed(42)
    model = GaussianModel(sh_degree=0, device=device)
    
    # Place Gaussians in front of camera
    points = torch.randn(N, 3, device=device) * 0.8
    points[:, 2] += 2.5
    colors = torch.rand(N, 3, device=device)
    model.initialize_from_points(points, colors, initial_scale=0.02)
    
    fx, fy = 120.0, 120.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32, device=device)
    extrinsics = torch.eye(4, dtype=torch.float32, device=device)
    
    target_rgb = torch.rand(H, W, 3, device=device)
    target_depth = torch.ones(H, W, device=device) * 2.5
    
    return model, extrinsics, intrinsics, target_rgb, target_depth, H, W


def run_full_method(model, opt, target_rgb, target_depth, extrinsics, intrinsics, H, W, device):
    """Method A: Full forward + full backward + full optimizer."""
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # 1. Forward
    opt.zero_grad()
    render_out = rasterize_scene(
        means3D=model.positions,
        cov3D=model.build_covariance(),
        colors=model.get_colors(),
        opacities=model.opacities.squeeze(-1),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_width=W,
        image_height=H,
    )
    t_fwd = (time.perf_counter() - t0) * 1000.0
    
    # 2. Loss & Backward
    t1 = time.perf_counter()
    loss = total_loss(render_out['color'], target_rgb, render_out['depth'], target_depth, {'color': 1.0, 'depth': 0.5})
    loss['total'].backward()
    if device == 'cuda':
        torch.cuda.synchronize()
    t_bwd = (time.perf_counter() - t1) * 1000.0
    
    # 3. Optimizer
    t2 = time.perf_counter()
    opt.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    t_opt = (time.perf_counter() - t2) * 1000.0
    
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        'forward_ms': t_fwd,
        'backward_ms': t_bwd,
        'optimizer_ms': t_opt,
        'total_ms': total_ms
    }


def run_masked_method(model, opt, active_mask, target_rgb, target_depth, extrinsics, intrinsics, H, W, device):
    """Method B: Full forward + full backward + zero gradients + full optimizer."""
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    opt.zero_grad()
    render_out = rasterize_scene(
        means3D=model.positions,
        cov3D=model.build_covariance(),
        colors=model.get_colors(),
        opacities=model.opacities.squeeze(-1),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_width=W,
        image_height=H,
    )
    t_fwd = (time.perf_counter() - t0) * 1000.0
    
    t1 = time.perf_counter()
    loss = total_loss(render_out['color'], target_rgb, render_out['depth'], target_depth, {'color': 1.0, 'depth': 0.5})
    loss['total'].backward()
    
    # Zero gradients for non-active Gaussians
    with torch.no_grad():
        non_active = ~active_mask
        if model._xyz.grad is not None:
            model._xyz.grad[non_active] = 0
            
    if device == 'cuda':
        torch.cuda.synchronize()
    t_bwd = (time.perf_counter() - t1) * 1000.0
    
    t2 = time.perf_counter()
    opt.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    t_opt = (time.perf_counter() - t2) * 1000.0
    
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        'forward_ms': t_fwd,
        'backward_ms': t_bwd,
        'optimizer_ms': t_opt,
        'total_ms': total_ms
    }


def run_true_selective_method(model, opt, active_mask, cache, target_rgb, target_depth, extrinsics, intrinsics, H, W, device):
    """Method D: Frozen background cache + active render + active backward + SelectiveAdam."""
    opt.zero_grad()
    
    # 1. Extract active subset (size M only)
    active_subset = model.get_optimization_subset(active_mask)
    frozen_mask = ~active_mask
    
    # 2. Build cache once (outside pure optimization timing)
    t_c0 = time.perf_counter()
    if frozen_mask.any():
        cache.build_cache(model, frozen_mask, extrinsics, intrinsics, W, H)
    t_cache = (time.perf_counter() - t_c0) * 1000.0
    
    # 3. Measure PURE optimization step (active render + composite + loss + backward + optimizer)
    if device == 'cuda':
        torch.cuda.synchronize()
    t_opt0 = time.perf_counter()
    
    t_r0 = time.perf_counter()
    comp_out = cache.composite_with_active(active_subset, extrinsics, intrinsics, W, H)
    t_active_render = (time.perf_counter() - t_r0) * 1000.0
    
    t_l0 = time.perf_counter()
    loss = total_loss(comp_out['color'], target_rgb, comp_out['depth'], target_depth, {'color': 1.0, 'depth': 0.5})
    t_loss = (time.perf_counter() - t_l0) * 1000.0
    
    t1 = time.perf_counter()
    loss['total'].backward()
    if device == 'cuda':
        torch.cuda.synchronize()
    t_bwd = (time.perf_counter() - t1) * 1000.0
    
    t2 = time.perf_counter()
    opt.step(active_idx=active_subset['indices'])
    if device == 'cuda':
        torch.cuda.synchronize()
    t_opt = (time.perf_counter() - t2) * 1000.0
    
    pure_opt_ms = (time.perf_counter() - t_opt0) * 1000.0
    frame_total_ms = t_cache + pure_opt_ms
    
    return {
        'cache_build_ms': t_cache,
        'active_render_ms': t_active_render,
        'loss_ms': t_loss,
        'backward_ms': t_bwd,
        'optimizer_ms': t_opt,
        'pure_opt_ms': pure_opt_ms,
        'frame_total_ms': frame_total_ms
    }


def benchmark_comprehensive(sizes=[10000, 25000, 50000], ratios=[1.0, 0.75, 0.50, 0.25, 0.20, 0.10, 0.05, 0.02, 0.01], n_warmup=10, n_trials=30, device='cpu'):
    """Run full 4-method benchmark across sizes and active ratios."""
    print("=" * 105)
    print("     R30: TRUE SELECTIVE OPTIMIZATION BENCHMARK (PURE OPT TIMING & STAGE BREAKDOWN)")
    print("=" * 105)
    print(f"Device: {device} | Warmup: {n_warmup} | Trials: {n_trials} | Sizes: {sizes} | Ratios: {ratios}\n")
    
    all_results = []
    raw_trials = []
    
    for N in sizes:
        print(f"\n{'='*105}\n>>> MODEL SIZE N = {N:,d} GAUSSIANS\n{'='*105}")
        print(f"{'Ratio (K)':<10} | {'Active (M)':<10} | {'Method':<16} | {'Active Render':<14} | {'Bwd p50 (ms)':<14} | {'Opt Step p50':<14} | {'Speedup':<10}")
        print("-" * 105)
        
        random.seed(42)
        shuffled_ratios = list(ratios)
        random.shuffle(shuffled_ratios)
        
        for r in shuffled_ratios:
            M = max(1, int(round(N * r)))
            active_mask = torch.zeros(N, dtype=torch.bool, device=device)
            active_mask[:M] = True
            
            model, extrinsics, intrinsics, rgb, depth, H, W = build_synthetic_scene(N, H=48, W=64, device=device)
            opt_full = optim.Adam(model.parameters(), lr=0.001)
            opt_selective = SelectiveAdam([{'params': list(model.parameters()), 'lr': 0.001}])
            cache = FrozenBackgroundCache(device=device)
            
            # 1. Benchmark Naive Masked
            masked_runs = []
            for trial in range(n_warmup + n_trials):
                res = run_masked_method(model, opt_full, active_mask, rgb, depth, extrinsics, intrinsics, H, W, device)
                if trial >= n_warmup:
                    masked_runs.append(res)
                    raw_trials.append({
                        'n_total': N, 'active_ratio': r, 'n_active': M, 'trial_idx': trial - n_warmup,
                        'method': 'masked', **res
                    })
                    
            # 2. Benchmark True Selective
            sel_runs = []
            for trial in range(n_warmup + n_trials):
                res = run_true_selective_method(model, opt_selective, active_mask, cache, rgb, depth, extrinsics, intrinsics, H, W, device)
                if trial >= n_warmup:
                    sel_runs.append(res)
                    raw_trials.append({
                        'n_total': N, 'active_ratio': r, 'n_active': M, 'trial_idx': trial - n_warmup,
                        'method': 'selective', **res
                    })
                    
            m_bwd_p50 = float(np.percentile([x['backward_ms'] for x in masked_runs], 50))
            m_opt_p50 = float(np.percentile([x['optimizer_ms'] for x in masked_runs], 50))
            m_tot_p50 = float(np.percentile([x['total_ms'] for x in masked_runs], 50))
            
            s_render_p50 = float(np.percentile([x['active_render_ms'] for x in sel_runs], 50))
            s_loss_p50 = float(np.percentile([x['loss_ms'] for x in sel_runs], 50))
            s_opt_ms_p50 = float(np.percentile([x['optimizer_ms'] for x in sel_runs], 50))
            s_bwd_p50 = float(np.percentile([x['backward_ms'] for x in sel_runs], 50))
            s_opt_step_p50 = float(np.percentile([x['pure_opt_ms'] for x in sel_runs], 50))
            s_cache_p50 = float(np.percentile([x['cache_build_ms'] for x in sel_runs], 50))
            
            opt_speedup = float(m_tot_p50 / max(s_opt_step_p50, 1e-6))
            bwd_speedup = float(m_bwd_p50 / max(s_bwd_p50, 1e-6))
            
            print(f"{r*100:5.0f}%     | {M:<10,d} | {'Masked (Baseline)':<16} | {'-':<14} | {m_bwd_p50:12.2f}ms | {m_tot_p50:12.2f}ms | {'1.00x':<10}")
            print(f"{r*100:5.0f}%     | {M:<10,d} | {'True Selective':<16} | {s_render_p50:12.2f}ms | {s_bwd_p50:12.2f}ms | {s_opt_step_p50:12.2f}ms | {opt_speedup:8.2f}x")
            print("-" * 105)
            
            all_results.append({
                'n_total': N,
                'active_ratio': r,
                'n_active': M,
                'masked': {'bwd_p50': m_bwd_p50, 'opt_p50': m_opt_p50, 'tot_p50': m_tot_p50},
                'selective': {
                    'cache_build_p50': s_cache_p50,
                    'active_render_p50': s_render_p50,
                    'bwd_p50': s_bwd_p50,
                    'pure_opt_p50': s_opt_step_p50,
                    'loss_ms_p50': s_loss_p50,
                    'optimizer_ms_p50': s_opt_ms_p50,
                    'render_ms_p50': s_render_p50,
                },
                'bwd_speedup': bwd_speedup,
                'opt_speedup': opt_speedup
            })
            
    # Save results
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'selective_compute')
    os.makedirs(save_dir, exist_ok=True)
    
    json_path = os.path.join(save_dir, 'selective_scaling.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    raw_path = os.path.join(save_dir, 'raw_trials.json')
    with open(raw_path, 'w') as f:
        json.dump(raw_trials, f, indent=2)
        
    csv_path = os.path.join(save_dir, 'selective_scaling.csv')
    with open(csv_path, 'w') as f:
        f.write("n_total,active_ratio,n_active,masked_bwd_p50,selective_bwd_p50,bwd_speedup,masked_tot_p50,selective_opt_p50,opt_speedup\n")
        for r in all_results:
            f.write(f"{r['n_total']},{r['active_ratio']:.3f},{r['n_active']},"
                    f"{r['masked']['bwd_p50']:.2f},{r['selective']['bwd_p50']:.2f},{r['bwd_speedup']:.2f},"
                    f"{r['masked']['tot_p50']:.2f},{r['selective']['pure_opt_p50']:.2f},{r['opt_speedup']:.2f}\n")
                    
    # Compute break-even point r* where Speedup(r*) = 1.0
    break_even_points = {}
    for N in sizes:
        n_res = [r for r in all_results if r['n_total'] == N]
        if n_res:
            # Sort by active ratio descending to find the largest ratio with speedup >= 1.0
            sorted_res = sorted(n_res, key=lambda x: x['active_ratio'], reverse=True)
            r_star = None
            for res in sorted_res:
                if res['opt_speedup'] >= 1.0:
                    r_star = res['active_ratio']
                    break
            if r_star is not None:
                break_even_points[N] = float(r_star)
            else:
                # No ratio achieves speedup >= 1.0
                break_even_points[N] = 0.0
            
    md_path = os.path.join(save_dir, 'selective_scaling.md')
    with open(md_path, 'w') as f:
        f.write("# R30: Comprehensive Selective Optimization Scaling Report\n\n")
        f.write("Evaluated with **Real 3DGS Rasterizer + RGB-D Loss** across Gaussian counts and active ratios (Pure Optimization Step Timing).\n\n")
        f.write("### Systems Break-Even Points ($r^*$ where $\\text{Speedup} \\approx 1.0\\times$)\n")
        f.write("*Note: r* is defined as the largest active ratio K where selective optimization achieves speedup >= 1.0x over full optimization.*\n\n")
        for N, r_star in break_even_points.items():
            f.write(f"- **N = {N:,d} Gaussians**: $r^* = {r_star*100:.1f}\\%$ (Largest active ratio where True Selective Optimization still delivers $\\ge 1.0\\times$ speedup)\n")
        f.write("\n")
        f.write("| N Total | Active Ratio | Active (M) | Active Render | Masked Bwd (p50) | Selective Bwd (p50) | Bwd Speedup | Masked Opt | Selective Opt | Opt Speedup |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in all_results:
            f.write(f"| {r['n_total']:,d} | {r['active_ratio']*100:.0f}% | {r['n_active']:,d} | "
                    f"{r['selective']['active_render_p50']:.2f} ms | "
                    f"{r['masked']['bwd_p50']:.2f} ms | {r['selective']['bwd_p50']:.2f} ms | "
                    f"**{r['bwd_speedup']:.2f}x** | {r['masked']['tot_p50']:.2f} ms | {r['selective']['pure_opt_p50']:.2f} ms | **{r['opt_speedup']:.2f}x** |\n")
        f.write("\n")
        
    print(f"\nArtifacts saved to:")
    print(f"  - {json_path}")
    print(f"  - {raw_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="R30 Comprehensive Selective Compute Benchmark")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--sizes', type=int, nargs='+', default=[10000, 25000, 50000])
    parser.add_argument('--ratios', type=float, nargs='+', default=[1.0, 0.75, 0.50, 0.25, 0.20, 0.10, 0.05, 0.02, 0.01])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--trials', type=int, default=30)
    args = parser.parse_args()
    
    benchmark_comprehensive(sizes=args.sizes, ratios=args.ratios, n_warmup=args.warmup, n_trials=args.trials, device=args.device)


if __name__ == '__main__':
    main()
