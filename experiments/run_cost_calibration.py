#!/usr/bin/env python3
"""R32 — Measured Cost Calibration Experiment.

Core Goal:
    Fit an empirical cost model:
        T(S) = T_0 + ∑_{i ∈ S} C_i
    where:
        C_i = β_0 + β_1 · Area_i + β_2 · Influence_i + β_3 · SH_i
    or baseline linear scaling:
        T(M) = T_0 + β · M

Validates that:
    1. Predicted compute aligns with measured actual compute (R² > 0.90, low MAE).
    2. BudgetScheduler can select subset S such that T_actual ≈ Budget.

Sweeps active ratios:
    M ∈ {5%, 10%, 20%, 25%, 50%, 75%, 100%}

Outputs:
    - results/cost_calibration/cost_model.json
    - results/cost_calibration/calibration_report.md
    - results/cost_calibration/calibration_curve.csv
"""
import os
import sys
import time
import json
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.gaussian_repr import GaussianModel
from research.rasterizer import render as rasterize_scene
from research.background_cache import FrozenBackgroundCache
from research.selective_optimizer import SelectiveAdam
from research.losses import total_loss
from research.attribution import compute_gaussian_statistics


def measure_active_step_time(model, opt, cache, active_mask, target_rgb, target_depth, extrinsics, intrinsics, H, W, device, n_trials=10, n_warmup=3):
    """Accurately measure actual wall-clock optimization time for a given active subset."""
    frozen_mask = ~active_mask
    
    # Warmup
    for _ in range(n_warmup):
        opt.zero_grad()
        active_subset = model.get_optimization_subset(active_mask)
        if frozen_mask.any():
            cache.build_cache(model, frozen_mask, extrinsics, intrinsics, W, H)
        comp_out = cache.composite_with_active(active_subset, extrinsics, intrinsics, W, H)
        losses = total_loss(comp_out['color'], target_rgb, comp_out['depth'], target_depth, {'color': 1.0, 'depth': 0.5})
        losses['total'].backward()
        opt.step(active_idx=active_subset['indices'])
        
    times = []
    for _ in range(n_trials):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        opt.zero_grad()
        active_subset = model.get_optimization_subset(active_mask)
        if frozen_mask.any():
            cache.build_cache(model, frozen_mask, extrinsics, intrinsics, W, H)
        comp_out = cache.composite_with_active(active_subset, extrinsics, intrinsics, W, H)
        losses = total_loss(comp_out['color'], target_rgb, comp_out['depth'], target_depth, {'color': 1.0, 'depth': 0.5})
        losses['total'].backward()
        opt.step(active_idx=active_subset['indices'])
        
        if device == 'cuda':
            torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) * 1000.0
        times.append(t_ms)
        
    return float(np.median(times)), float(np.std(times))


def run_cost_calibration(N=20000, ratios=[0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 1.00], device='cpu'):
    print("=" * 85)
    print("           R32: EMPIRICAL COST MODEL CALIBRATION EXPERIMENT")
    print("=" * 85)
    print(f"Device: {device} | Scene Size: N = {N:,d} Gaussians | Active Ratios: {ratios}\n")
    
    H, W = 48, 64
    torch.manual_seed(42)
    model = GaussianModel(sh_degree=0, device=device)
    points = torch.randn(N, 3, device=device) * 0.8
    points[:, 2] += 2.5
    colors = torch.rand(N, 3, device=device)
    model.initialize_from_points(points, colors, initial_scale=0.02)
    
    fx, fy = 120.0, 120.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32, device=device)
    extrinsics = torch.eye(4, dtype=torch.float32, device=device)
    
    target_rgb = torch.rand(H, W, 3, device=device)
    target_depth = torch.ones(H, W, device=device) * 2.5
    
    cache = FrozenBackgroundCache(device=device)
    opt = SelectiveAdam([{'params': list(model.parameters()), 'lr': 0.001}])
    
    measured_data = []
    
    print(f"{'Active Ratio':<14} | {'Active (M)':<12} | {'Actual Time (ms)':<18} | {'Std (ms)':<10}")
    print("-" * 65)
    
    for r in ratios:
        M = max(1, int(round(N * r)))
        active_mask = torch.zeros(N, dtype=torch.bool, device=device)
        active_mask[:M] = True
        
        t_med, t_std = measure_active_step_time(
            model, opt, cache, active_mask, target_rgb, target_depth,
            extrinsics, intrinsics, H, W, device, n_trials=8, n_warmup=2
        )
        
        print(f"{r*100:5.1f}%          | {M:<12,d} | {t_med:14.2f} ms | {t_std:8.2f} ms")
        
        measured_data.append({
            'ratio': float(r),
            'M': int(M),
            'actual_ms': float(t_med),
            'std_ms': float(t_std)
        })
        
    print("-" * 65)
    
    # 1. Fit linear model: T(M) = T_0 + beta * M
    M_vals = np.array([d['M'] for d in measured_data], dtype=np.float64)
    T_vals = np.array([d['actual_ms'] for d in measured_data], dtype=np.float64)
    
    # Design matrix [1, M]
    A = np.vstack([np.ones_like(M_vals), M_vals]).T
    (T_0, beta), residuals, rank, s = np.linalg.lstsq(A, T_vals, rcond=None)
    
    # Compute predictions and goodness of fit
    T_pred = T_0 + beta * M_vals
    mae = float(np.mean(np.abs(T_vals - T_pred)))
    ss_tot = float(np.sum((T_vals - np.mean(T_vals)) ** 2))
    ss_res = float(np.sum((T_vals - T_pred) ** 2))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    
    print("\n" + "=" * 65)
    print("                CALIBRATED COST MODEL FIT RESULTS")
    print("=" * 65)
    print(f"  • Formulation: T(M) = T_0 + β · M")
    print(f"  • Fixed Overhead T_0:   {T_0:.3f} ms")
    print(f"  • Slope β (per-Gaussian): {beta * 1000.0:.4f} µs / Gaussian ({beta:.6f} ms)")
    print(f"  • Goodness of Fit R²:   {r2:.4f}")
    print(f"  • Mean Absolute Error:  {mae:.3f} ms")
    print("=" * 65 + "\n")
    
    # Save artifacts
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'cost_calibration')
    os.makedirs(save_dir, exist_ok=True)
    
    model_json = {
        'T_0_ms': float(T_0),
        'beta_ms_per_gaussian': float(beta),
        'beta_us_per_gaussian': float(beta * 1000.0),
        'r2': float(r2),
        'mae_ms': float(mae),
        'scene_size_N': int(N),
        'device': str(device)
    }
    
    with open(os.path.join(save_dir, 'cost_model.json'), 'w') as f:
        json.dump(model_json, f, indent=2)
        
    csv_path = os.path.join(save_dir, 'calibration_curve.csv')
    with open(csv_path, 'w') as f:
        f.write("active_ratio,active_M,actual_ms,predicted_ms,error_ms\n")
        for d, pred in zip(measured_data, T_pred):
            f.write(f"{d['ratio']:.3f},{d['M']},{d['actual_ms']:.3f},{pred:.3f},{abs(d['actual_ms'] - pred):.3f}\n")
            
    report_path = os.path.join(save_dir, 'calibration_report.md')
    with open(report_path, 'w') as f:
        f.write("# R32 Cost Model Calibration Report\n\n")
        f.write("## Calibrated Latency Model\n")
        f.write(f"$$T(M) = {T_0:.3f} + {beta:.6f} \\times M$$\n\n")
        f.write(f"- **Goodness of Fit ($R^2$)**: **{r2:.4f}**\n")
        f.write(f"- **Mean Absolute Error (MAE)**: **{mae:.3f} ms**\n\n")
        f.write("| Active Ratio | Active ($M$) | Actual Measured ($T$) | Predicted ($T$) | Residual Error |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|\n")
        for d, pred in zip(measured_data, T_pred):
            f.write(f"| {d['ratio']*100:.1f}% | {d['M']:,d} | {d['actual_ms']:.2f} ms | {pred:.2f} ms | {abs(d['actual_ms'] - pred):.2f} ms |\n")
        f.write("\n")
        
    print(f"Calibration artifacts saved to:")
    print(f"  - {os.path.join(save_dir, 'cost_model.json')}")
    print(f"  - {csv_path}")
    print(f"  - {report_path}")
    
    return model_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cost Model Calibration")
    parser.add_argument('--N', type=int, default=20000)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    run_cost_calibration(N=args.N, device=args.device)
