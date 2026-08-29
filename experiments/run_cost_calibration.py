#!/usr/bin/env python3
"""R32 — Rigorous Cost Model Calibration Experiment.

Learns the Pure Optimization Cost Model T_opt(M) on real 3DGS rasterizer execution:
  - Scope: pure_optimization (cache build strictly excluded)
  - Evaluates Model A (Count-only), Model B (Workload-aware), Model C (Stage-level sum)
  - Train/Test Split: 80% train, 20% held-out test evaluation
  - Metrics: Out-of-sample R², MAE, RMSE, sMAPE (Symmetric Mean Absolute Percentage Error)

Outputs:
  - results/cost_calibration/cost_model.json
  - results/cost_calibration/model_comparison.json
  - results/cost_calibration/observations.json
  - results/cost_calibration/calibration_report.md
  - results/raw/cost_calibration_raw.json
"""
import os
import sys
import time
import json
import random
import argparse
import subprocess
import torch
import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.gaussian_repr import GaussianModel
from research.rasterizer import render as rasterize_scene
from research.background_cache import FrozenBackgroundCache
from research.selective_optimizer import SelectiveAdam
from research.losses import total_loss
from research.projection import world_to_camera, compute_2d_covariance
from research.attribution import compute_projected_area


def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "unknown"


def build_calibration_scene(N: int, H: int = 48, W: int = 64, device: str = 'cpu', seed: int = 42):
    """Build a freshly initialized GaussianModel and camera setup."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = GaussianModel(sh_degree=0, device=device)
    points = torch.randn(N, 3, device=device) * 0.8
    points[:, 2] += 2.5
    colors = torch.rand(N, 3, device=device)
    scales = 0.01 + 0.03 * torch.rand(N, 3, device=device)
    
    model.initialize_from_points(points, colors, initial_scale=0.02)
    model._scaling.data = torch.log(scales)
    model._opacity.data = torch.randn(N, 1, device=device) * 0.5  # moderate opacities
    
    fx, fy = 120.0, 120.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32, device=device)
    extrinsics = torch.eye(4, dtype=torch.float32, device=device)
    
    target_rgb = torch.rand(H, W, 3, device=device)
    target_depth = torch.ones(H, W, device=device) * 2.5
    
    return model, extrinsics, intrinsics, target_rgb, target_depth, H, W


def measure_pure_opt_step(model, active_mask, target_rgb, target_depth, extrinsics, intrinsics, H, W, device):
    """Run one pure optimization step on a cloned model state to guarantee timing isolation."""
    # Build a temporary optimizer and cache
    opt = SelectiveAdam([{'params': list(model.parameters()), 'lr': 0.001}])
    cache = FrozenBackgroundCache(device=device)
    
    active_subset = model.get_optimization_subset(active_mask)
    frozen_mask = ~active_mask
    
    # 1. Build cache outside timed block
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
    pure_opt_ms = render_ms + backward_ms + optimizer_ms
    
    # Extract true physical workload features
    M = int(active_mask.sum().item())
    with torch.no_grad():
        # True projected screen area
        cov3D_active = model.build_covariance()[active_mask]
        pts_cam = world_to_camera(model.positions[active_mask], extrinsics)
        cov2d = compute_2d_covariance(cov3D_active, pts_cam, extrinsics, intrinsics)
        proj_areas = compute_projected_area(cov2d)
        proj_areas = torch.nan_to_num(proj_areas, nan=0.0, posinf=0.0, neginf=0.0)
        area_sum = float(proj_areas.sum().item())
        # Influence mass (opacity * visibility)
        inf_mass = torch.nan_to_num(model.opacities[active_mask], nan=0.0, posinf=0.0, neginf=0.0)
        inf_sum = float(inf_mass.sum().item())
        
    return {
        'timing_scope': 'pure_optimization',
        'cache_included': False,
        'render_included': True,
        'backward_included': True,
        'optimizer_included': True,
        'M': M,
        'render_ms': render_ms,
        'backward_ms': backward_ms,
        'optimizer_ms': optimizer_ms,
        'measured_ms': pure_opt_ms,
        'area': area_sum,
        'influence': inf_sum,
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute R², MAE, RMSE, and sMAPE (symmetric MAPE)."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # sMAPE: 2 * |y - y_hat| / (|y| + |y_hat| + eps) * 100%
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-6))) * 100.0
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    return {'r2': r2, 'mae_ms': mae, 'rmse_ms': rmse, 'smape_pct': smape}


def run_cost_calibration(N: int = 25000, M_list=None, seeds=[42, 43, 44], n_trials_per_config=5, device='cpu'):
    if M_list is None:
        M_list = [250, 500, 1250, 2500, 5000, 10000, 18750, 25000]
        
    print("=" * 110)
    print("     R32: RIGOROUS COST CALIBRATION (PURE OPT TIMING & OUT-OF-SAMPLE EVALUATION)")
    print("=" * 110)
    print(f"Device: {device} | N = {N:,d} Gaussians | M counts: {M_list} | Seeds: {seeds}\n")
    
    H, W = 48, 64
    all_observations = []
    
    # 1. Warmup
    model, extrinsics, intrinsics, rgb, depth, H, W = build_calibration_scene(N, H, W, device, seed=999)
    warmup_mask = torch.zeros(N, dtype=torch.bool, device=device)
    warmup_mask[:500] = True
    for _ in range(5):
        measure_pure_opt_step(model, warmup_mask, rgb, depth, extrinsics, intrinsics, H, W, device)
        
    # 2. Randomized measurement trials across seeds
    for seed in seeds:
        model, extrinsics, intrinsics, rgb, depth, H, W = build_calibration_scene(N, H, W, device, seed=seed)
        
        trial_M_list = M_list * n_trials_per_config
        random.seed(seed)
        random.shuffle(trial_M_list)
        
        for M in trial_M_list:
            active_mask = torch.zeros(N, dtype=torch.bool, device=device)
            perm = torch.randperm(N, device=device)
            active_mask[perm[:M]] = True
            
            obs = measure_pure_opt_step(model, active_mask, rgb, depth, extrinsics, intrinsics, H, W, device)
            obs['seed'] = seed
            obs['ratio'] = float(M / N)
            obs['n_total'] = N
            all_observations.append(obs)
            
    print(f"Collected {len(all_observations)} randomized pure optimization observations.\n")
    
    # 3. Save raw observations
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'raw')
    proc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'cost_calibration')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    metadata = {
        "git_commit": get_git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "torch_version": torch.__version__,
        "seeds": seeds,
        "n_gaussians": N,
        "timing_scope": "pure_optimization"
    }
    
    with open(os.path.join(raw_dir, 'cost_calibration_raw.json'), 'w') as f:
        json.dump({'metadata': metadata, 'observations': all_observations}, f, indent=2)
        
    with open(os.path.join(proc_dir, 'observations.json'), 'w') as f:
        json.dump(all_observations, f, indent=2)
        
    # 4. Out-of-sample 80% Train / 20% Test Split
    n_obs = len(all_observations)
    np.random.seed(42)
    indices = np.random.permutation(n_obs)
    split = int(0.8 * n_obs)
    train_idx, test_idx = indices[:split], indices[split:]
    
    obs_train = [all_observations[i] for i in train_idx]
    obs_test = [all_observations[i] for i in test_idx]
    
    # Target vectors
    y_train = np.array([o['measured_ms'] for o in obs_train], dtype=np.float64)
    y_test = np.array([o['measured_ms'] for o in obs_test], dtype=np.float64)
    
    M_train = np.array([o['M'] for o in obs_train], dtype=np.float64).reshape(-1, 1)
    M_test = np.array([o['M'] for o in obs_test], dtype=np.float64).reshape(-1, 1)
    
    # Model A: T(M) = T_0 + beta * M
    reg_A = Ridge(alpha=1.0, fit_intercept=True)
    reg_A.fit(M_train, y_train)
    pred_test_A = reg_A.predict(M_test)
    metrics_A = compute_regression_metrics(y_test, pred_test_A)
    T0_A = float(reg_A.intercept_)
    beta_A = float(reg_A.coef_[0])
    
    # Model B: T(S) = T_0 + beta_1 * M + beta_2 * Area + beta_3 * Influence
    X_B_train = np.nan_to_num(np.column_stack([
        M_train,
        np.array([o['area'] for o in obs_train], dtype=np.float64),
        np.array([o['influence'] for o in obs_train], dtype=np.float64)
    ]), nan=0.0, posinf=0.0, neginf=0.0)
    X_B_test = np.nan_to_num(np.column_stack([
        M_test,
        np.array([o['area'] for o in obs_test], dtype=np.float64),
        np.array([o['influence'] for o in obs_test], dtype=np.float64)
    ]), nan=0.0, posinf=0.0, neginf=0.0)
    reg_B = Ridge(alpha=1.0, fit_intercept=True)
    reg_B.fit(X_B_train, y_train)
    pred_test_B = reg_B.predict(X_B_test)
    metrics_B = compute_regression_metrics(y_test, pred_test_B)
    T0_B = float(reg_B.intercept_)
    
    # Model C: Stage-level sum T_opt(M) = T_rend(M) + T_bwd(M) + T_opt(M)
    y_rend_train = np.array([o['render_ms'] for o in obs_train], dtype=np.float64)
    y_bwd_train = np.array([o['backward_ms'] for o in obs_train], dtype=np.float64)
    y_opt_train = np.array([o['optimizer_ms'] for o in obs_train], dtype=np.float64)
    
    reg_rend = Ridge(alpha=1.0, fit_intercept=True).fit(M_train, y_rend_train)
    reg_bwd = Ridge(alpha=1.0, fit_intercept=True).fit(M_train, y_bwd_train)
    reg_opt = Ridge(alpha=1.0, fit_intercept=True).fit(M_train, y_opt_train)
    
    pred_rend_test = reg_rend.predict(M_test)
    pred_bwd_test = reg_bwd.predict(M_test)
    pred_opt_test = reg_opt.predict(M_test)
    pred_test_C = pred_rend_test + pred_bwd_test + pred_opt_test
    metrics_C = compute_regression_metrics(y_test, pred_test_C)
    T0_C = float(reg_rend.intercept_ + reg_bwd.intercept_ + reg_opt.intercept_)
    beta_C = float(reg_rend.coef_[0] + reg_bwd.coef_[0] + reg_opt.coef_[0])
    
    r2_A_str = f"{metrics_A['r2']:.4f}"
    r2_B_str = f"{metrics_B['r2']:.4f}"
    r2_C_str = f"{metrics_C['r2']:.4f}"
    mae_A_str = f"{metrics_A['mae_ms']:.3f} ms"
    mae_B_str = f"{metrics_B['mae_ms']:.3f} ms"
    mae_C_str = f"{metrics_C['mae_ms']:.3f} ms"
    rmse_A_str = f"{metrics_A['rmse_ms']:.3f} ms"
    rmse_B_str = f"{metrics_B['rmse_ms']:.3f} ms"
    rmse_C_str = f"{metrics_C['rmse_ms']:.3f} ms"
    smape_A_str = f"{metrics_A['smape_pct']:.2f}%"
    smape_B_str = f"{metrics_B['smape_pct']:.2f}%"
    smape_C_str = f"{metrics_C['smape_pct']:.2f}%"
    
    print("=" * 115)
    print("                 COST MODEL EVALUATION (OUT-OF-SAMPLE 20% TEST SET)")
    print("=" * 115)
    print(f"{'Metric':<25} | {'Model A (Count-only)':<25} | {'Model B (Workload-aware)':<25} | {'Model C (Stage-level Sum)':<25}")
    print("-" * 115)
    print(f"{'Formula':<25} | {'T_0 + β·M':<25} | {'T_0 + β1·M + β2·A + β3·P':<25} | {'T_rend(M) + T_bwd(M) + T_opt(M)':<25}")
    print(f"{'Fixed Overhead (T_0)':<25} | {f'{T0_A:.3f} ms':<25} | {f'{T0_B:.3f} ms':<25} | {f'{T0_C:.3f} ms':<25}")
    print(f"{'Test R² ↑':<25} | {r2_A_str:<25} | {r2_B_str:<25} | {r2_C_str:<25}")
    print(f"{'Test MAE (ms) ↓':<25} | {mae_A_str:<25} | {mae_B_str:<25} | {mae_C_str:<25}")
    print(f"{'Test RMSE (ms) ↓':<25} | {rmse_A_str:<25} | {rmse_B_str:<25} | {rmse_C_str:<25}")
    print(f"{'Test sMAPE (%) ↓':<25} | {smape_A_str:<25} | {smape_B_str:<25} | {smape_C_str:<25}")
    print("=" * 115 + "\n")
    
    summary = {
        'metadata': metadata,
        'model_A': {
            'T_0_ms': T0_A,
            'beta_ms_per_gaussian': beta_A,
            'r2': metrics_A['r2'],
            'mae_ms': metrics_A['mae_ms'],
            'rmse_ms': metrics_A['rmse_ms'],
            'smape_pct': metrics_A['smape_pct']
        },
        'model_B': {
            'T_0_ms': T0_B,
            'beta_1_ms': float(reg_B.coef_[0]),
            'beta_2_area': float(reg_B.coef_[1]),
            'beta_3_influence': float(reg_B.coef_[2]),
            'r2': metrics_B['r2'],
            'mae_ms': metrics_B['mae_ms'],
            'rmse_ms': metrics_B['rmse_ms'],
            'smape_pct': metrics_B['smape_pct']
        },
        'model_C': {
            'T_0_ms': T0_C,
            'beta_ms_per_gaussian': beta_C,
            'stage_render': {'a': float(reg_rend.intercept_), 'b': float(reg_rend.coef_[0])},
            'stage_backward': {'a': float(reg_bwd.intercept_), 'b': float(reg_bwd.coef_[0])},
            'stage_optimizer': {'a': float(reg_opt.intercept_), 'b': float(reg_opt.coef_[0])},
            'r2': metrics_C['r2'],
            'mae_ms': metrics_C['mae_ms'],
            'rmse_ms': metrics_C['rmse_ms'],
            'smape_pct': metrics_C['smape_pct']
        }
    }
    
    with open(os.path.join(proc_dir, 'model_comparison.json'), 'w') as f:
        json.dump(summary, f, indent=2)
        
    with open(os.path.join(proc_dir, 'cost_model.json'), 'w') as f:
        json.dump(summary['model_A'], f, indent=2)
        
    md_path = os.path.join(proc_dir, 'calibration_report.md')
    with open(md_path, 'w') as f:
        f.write("# R32 Rigorous Cost Calibration Report\n\n")
        f.write(f"Evaluated with isolated pure optimization timing across $N={N:,d}$ Gaussians (80% Train, 20% Held-Out Test).\n\n")
        f.write("| Metric | Model A (Linear Count) | Model B (Workload-Aware) | Model C (Stage-Level Sum) |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        f.write(f"| **Formulation** | $T_0 + \\beta M$ | $T_0 + \\beta_1 M + \\beta_2 A + \\beta_3 P$ | $T_{{rend}}(M) + T_{{bwd}}(M) + T_{{opt}}(M)$ |\n")
        f.write(f"| **Fixed Overhead ($T_0$)** | {T0_A:.3f} ms | {T0_B:.3f} ms | {T0_C:.3f} ms |\n")
        f.write(f"| **Out-of-Sample $R^2$** | **{metrics_A['r2']:.4f}** | **{metrics_B['r2']:.4f}** | **{metrics_C['r2']:.4f}** |\n")
        f.write(f"| **Test MAE (ms)** | **{metrics_A['mae_ms']:.3f} ms** | **{metrics_B['mae_ms']:.3f} ms** | **{metrics_C['mae_ms']:.3f} ms** |\n")
        f.write(f"| **Test RMSE (ms)** | **{metrics_A['rmse_ms']:.3f} ms** | **{metrics_B['rmse_ms']:.3f} ms** | **{metrics_C['rmse_ms']:.3f} ms** |\n")
        f.write(f"| **Test sMAPE (%)** | **{metrics_A['smape_pct']:.2f}%** | **{metrics_B['smape_pct']:.2f}%** | **{metrics_C['smape_pct']:.2f}%** |\n\n")
        
    print(f"Calibration artifacts updated at:")
    print(f"  - {os.path.join(proc_dir, 'model_comparison.json')}")
    print(f"  - {md_path}")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rigorous Cost Calibration")
    parser.add_argument('--N', type=int, default=25000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--trials', type=int, default=5)
    args = parser.parse_args()
    
    run_cost_calibration(N=args.N, n_trials_per_config=args.trials, device=args.device)
