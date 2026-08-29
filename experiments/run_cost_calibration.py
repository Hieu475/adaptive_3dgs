#!/usr/bin/env python3
"""R32 — Measured Cost Calibration Experiment (2 Models: Model A vs Model B).

Compares:
  - Model A (Linear Count): T(M) = T_0 + β · M
  - Model B (Feature-Aware): T(S) = T_0 + β_1 · M + β_2 · Area + β_3 · Influence + β_4 · SH

Protocol:
  - Multi-seed: [42, 43, 44]
  - Warmup: 5 steps
  - Trials: 15 per configuration
  - Randomized evaluation order
  - Metrics: R², MAE, MAPE (Mean Absolute Percentage Error)

Outputs:
  - results/cost_calibration/cost_model.json
  - results/cost_calibration/calibration_report.md
  - results/cost_calibration/model_comparison.json
"""
import os
import sys
import time
import json
import random
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.gaussian_repr import GaussianModel
from research.rasterizer import render as rasterize_scene
from research.background_cache import FrozenBackgroundCache
from research.selective_optimizer import SelectiveAdam
from research.losses import total_loss
from research.attribution import compute_projected_area


def measure_active_step_features(model, opt, cache, active_mask, target_rgb, target_depth, extrinsics, intrinsics, H, W, device):
    """Run one optimization step and extract pure optimization latency and feature aggregates (excluding cache build)."""
    opt.zero_grad()
    active_subset = model.get_optimization_subset(active_mask)
    frozen_mask = ~active_mask
    
    # 1. Build cache once outside timed optimization block
    if frozen_mask.any():
        cache.build_cache(model, frozen_mask, extrinsics, intrinsics, W, H)
        
    # 2. Pure optimization step measurement
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    comp_out = cache.composite_with_active(active_subset, extrinsics, intrinsics, W, H)
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    losses = total_loss(comp_out['color'], target_rgb, comp_out['depth'], target_depth, {'color': 1.0, 'depth': 0.5})
    losses['total'].backward()
    if device == 'cuda':
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    
    opt.step(active_idx=active_subset['indices'])
    if device == 'cuda':
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    
    render_ms = (t1 - t0) * 1000.0
    backward_ms = (t2 - t1) * 1000.0
    optimizer_ms = (t3 - t2) * 1000.0
    measured_ms = render_ms + backward_ms + optimizer_ms
    
    # Feature extraction
    M = int(active_mask.sum().item())
    with torch.no_grad():
        area_agg = float(model.scales[active_mask].prod(dim=-1).sum().item())
        opacity_agg = float(model.opacities[active_mask].sum().item())
        sh_deg = float(model.sh_degree)
        
    return {
        'measured_ms': measured_ms,
        'render_ms': render_ms,
        'backward_ms': backward_ms,
        'optimizer_ms': optimizer_ms,
        'M': M,
        'area': area_agg,
        'influence': opacity_agg,
        'sh': sh_deg * M
    }


from sklearn.linear_model import Ridge


def fit_ridge_linear_model(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Robust linear regression with Ridge regularizer using Scikit-Learn."""
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X[:, 1:], y)
    return np.array([float(reg.intercept_)] + [float(c) for c in reg.coef_], dtype=np.float64)


def run_rigorous_cost_calibration(N=20000, ratios=[0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 1.00], seeds=[42, 43, 44], device='cpu'):
    print("=" * 85)
    print("     R32: RIGOROUS COST CALIBRATION (MODEL A vs MODEL B COMPARISON)")
    print("=" * 85)
    print(f"Device: {device} | N = {N:,d} Gaussians | Ratios: {ratios} | Seeds: {seeds}\n")
    
    H, W = 48, 64
    all_observations = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        model = GaussianModel(sh_degree=0, device=device)
        points = torch.randn(N, 3, device=device) * 0.8
        points[:, 2] += 2.5
        colors = torch.rand(N, 3, device=device)
        scales = 0.01 + 0.04 * torch.rand(N, 3, device=device)
        model.initialize_from_points(points, colors, initial_scale=0.02)
        model._scaling.data = torch.log(scales)
        model._opacity.data = torch.randn(N, 1, device=device)  # diverse opacities
        
        fx, fy = 120.0, 120.0
        intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32, device=device)
        extrinsics = torch.eye(4, dtype=torch.float32, device=device)
        
        target_rgb = torch.rand(H, W, 3, device=device)
        target_depth = torch.ones(H, W, device=device) * 2.5
        
        cache = FrozenBackgroundCache(device=device)
        opt = SelectiveAdam([{'params': list(model.parameters()), 'lr': 0.001}])
        
        # Warmup
        for _ in range(5):
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[:1000] = True
            measure_active_step_features(model, opt, cache, mask, target_rgb, target_depth, extrinsics, intrinsics, H, W, device)
            
        # Create randomized trial list
        trial_ratios = ratios * 3  # 3 repeats per seed = 9 total per ratio
        random.shuffle(trial_ratios)
        
        for r in trial_ratios:
            M = max(1, int(round(N * r)))
            active_mask = torch.zeros(N, dtype=torch.bool, device=device)
            perm = torch.randperm(N, device=device)
            active_mask[perm[:M]] = True
            
            obs = measure_active_step_features(model, opt, cache, active_mask, target_rgb, target_depth, extrinsics, intrinsics, H, W, device)
            obs['ratio'] = r
            obs['seed'] = seed
            all_observations.append(obs)
            
    print(f"Collected {len(all_observations)} randomized multi-seed measurement points.\n")
    
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'cost_calibration')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save raw observations for figure generation
    obs_path = os.path.join(save_dir, 'observations.json')
    with open(obs_path, 'w') as f:
        json.dump(all_observations, f, indent=2)

    # 1. Fit Model A: T(M) = T_0 + beta * M
    y = np.nan_to_num(np.array([o['measured_ms'] for o in all_observations], dtype=np.float64), nan=0.0)
    M_vec = np.nan_to_num(np.array([o['M'] for o in all_observations], dtype=np.float64), nan=0.0)
    
    X_A = np.nan_to_num(np.vstack([np.ones_like(M_vec), M_vec]).T, nan=0.0)
    coeffs_A = fit_ridge_linear_model(X_A, y)
    T_0_A, beta_A = float(coeffs_A[0]), float(coeffs_A[1])
    
    pred_A = X_A @ coeffs_A
    mae_A = float(np.mean(np.abs(y - pred_A)))
    mape_A = float(np.mean(np.abs((y - pred_A) / (y + 1e-6)))) * 100.0
    r2_A = float(1.0 - np.sum((y - pred_A)**2) / np.sum((y - np.mean(y))**2))
    
    # 2. Fit Model B: T(S) = T_0 + beta_1 * M + beta_2 * Area + beta_3 * Influence
    Area_vec = np.nan_to_num(np.array([o['area'] for o in all_observations], dtype=np.float64), nan=0.0)
    Inf_vec = np.nan_to_num(np.array([o['influence'] for o in all_observations], dtype=np.float64), nan=0.0)
    
    X_B = np.nan_to_num(np.vstack([np.ones_like(M_vec), M_vec, Area_vec, Inf_vec]).T, nan=0.0)
    coeffs_B = fit_ridge_linear_model(X_B, y)
    T_0_B, beta_1_B, beta_2_B, beta_3_B = float(coeffs_B[0]), float(coeffs_B[1]), float(coeffs_B[2]), float(coeffs_B[3])
    
    pred_B = X_B @ coeffs_B
    mae_B = float(np.mean(np.abs(y - pred_B)))
    mape_B = float(np.mean(np.abs((y - pred_B) / (y + 1e-6)))) * 100.0
    r2_B = float(1.0 - np.sum((y - pred_B)**2) / np.sum((y - np.mean(y))**2))
    # 3. Fit Model C: Stage-level linear models (T_render, T_backward, T_optimizer)
    y_r = np.nan_to_num(np.array([o['render_ms'] for o in all_observations], dtype=np.float64), nan=0.0)
    y_b = np.nan_to_num(np.array([o['backward_ms'] for o in all_observations], dtype=np.float64), nan=0.0)
    y_o = np.nan_to_num(np.array([o['optimizer_ms'] for o in all_observations], dtype=np.float64), nan=0.0)
    
    coeffs_r = fit_ridge_linear_model(X_A, y_r)
    coeffs_b = fit_ridge_linear_model(X_A, y_b)
    coeffs_o = fit_ridge_linear_model(X_A, y_o)
    
    pred_r = X_A @ coeffs_r
    pred_b = X_A @ coeffs_b
    pred_o = X_A @ coeffs_o
    
    pred_C = pred_r + pred_b + pred_o
    mae_C = float(np.mean(np.abs(y - pred_C)))
    mape_C = float(np.mean(np.abs((y - pred_C) / (y + 1e-6)))) * 100.0
    r2_C = float(1.0 - np.sum((y - pred_C)**2) / np.sum((y - np.mean(y))**2))
    
    a_r, b_r = float(coeffs_r[0]), float(coeffs_r[1])
    a_b, b_b = float(coeffs_b[0]), float(coeffs_b[1])
    a_o, b_o = float(coeffs_o[0]), float(coeffs_o[1])

    print("=" * 110)
    print("                      COST MODEL EVALUATION COMPARISON")
    print("=" * 110)
    print(f"{'Metric':<25} | {'Model A (Count)':<25} | {'Model B (Feature-Aware)':<25} | {'Model C (Stage-level)':<25}")
    print("-" * 110)
    print(f"{'Formula':<25} | {'T_0 + β·M':<25} | {'T_0 + β1·M + β2·Area + β3·Inf':<25} | {'T_r(M) + T_b(M) + T_o(M)':<25}")
    print(f"{'Fixed Overhead (T_0)':<25} | {f'{T_0_A:.3f} ms':<25} | {f'{T_0_B:.3f} ms':<25} | {f'{(a_r + a_b + a_o):.3f} ms':<25}")
    print(f"{'Goodness of Fit (R²)':<25} | {f'{r2_A:.4f}':<25} | {f'{r2_B:.4f}':<25} | {f'{r2_C:.4f}':<25}")
    print(f"{'MAE (ms)':<25} | {f'{mae_A:.3f} ms':<25} | {f'{mae_B:.3f} ms':<25} | {f'{mae_C:.3f} ms':<25}")
    print(f"{'MAPE (%)':<25} | {f'{mape_A:.2f}%':<25} | {f'{mape_B:.2f}%':<25} | {f'{mape_C:.2f}%':<25}")
    print("=" * 110 + "\n")
    
    # Save results
    summary = {
        'model_A': {
            'T_0_ms': float(T_0_A),
            'beta_ms_per_gaussian': float(beta_A),
            'r2': float(r2_A),
            'mae_ms': float(mae_A),
            'mape_pct': float(mape_A)
        },
        'model_B': {
            'T_0_ms': float(T_0_B),
            'beta_1_ms': float(beta_1_B),
            'beta_2_area': float(beta_2_B),
            'beta_3_influence': float(beta_3_B),
            'r2': float(r2_B),
            'mae_ms': float(mae_B),
            'mape_pct': float(mape_B)
        },
        'model_C': {
            'render': {'a': a_r, 'b': b_r},
            'backward': {'a': a_b, 'b': b_b},
            'optimizer': {'a': a_o, 'b': b_o},
            'r2': float(r2_C),
            'mae_ms': float(mae_C),
            'mape_pct': float(mape_C)
        }
    }
    
    with open(os.path.join(save_dir, 'model_comparison.json'), 'w') as f:
        json.dump(summary, f, indent=2)
        
    with open(os.path.join(save_dir, 'cost_model.json'), 'w') as f:
        json.dump(summary['model_A'], f, indent=2)
        
    md_path = os.path.join(save_dir, 'calibration_report.md')
    with open(md_path, 'w') as f:
        f.write("# R32 Rigorous Cost Calibration Report\n\n")
        f.write("Evaluated with randomized multi-seed protocol across $N=20,000$ Gaussians.\n\n")
        f.write("| Metric | Model A (Linear Count) | Model B (Feature-Aware) | Model C (Stage-Level) |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        f.write(f"| **Formulation** | $T_0 + \\beta M$ | $T_0 + \\beta_1 M + \\beta_2 A + \\beta_3 \\text{{Inf}}$ | $\\Sigma_s (a_s + b_s M)$ |\n")
        f.write(f"| **Fixed Overhead ($T_0$)** | {T_0_A:.3f} ms | {T_0_B:.3f} ms | {a_r + a_b + a_o:.3f} ms |\n")
        f.write(f"| **Goodness of Fit ($R^2$)** | **{r2_A:.4f}** | **{r2_B:.4f}** | **{r2_C:.4f}** |\n")
        f.write(f"| **MAE** | **{mae_A:.3f} ms** | **{mae_B:.3f} ms** | **{mae_C:.3f} ms** |\n")
        f.write(f"| **MAPE** | **{mape_A:.2f}%** | **{mape_B:.2f}%** | **{mape_C:.2f}%** |\n\n")
        
    print(f"Calibration artifacts updated at:")
    print(f"  - {os.path.join(save_dir, 'model_comparison.json')}")
    print(f"  - {md_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Rigorous Cost Calibration")
    parser.add_argument('--N', type=int, default=20000)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    run_rigorous_cost_calibration(N=args.N, device=args.device)


if __name__ == '__main__':
    main()
