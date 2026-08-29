#!/usr/bin/env python3
"""Generate publication-quality research figures for Adaptive 3DGS.

Loads measured experiment results from JSON artifacts and generates:
  - Figure A: Selective Optimization Scaling & Speedup (from results/selective_compute/)
  - Figure B: Cost Model Calibration (from results/cost_calibration/)
  - Figure C: Quality vs Compute Pareto Frontier (from results/matched_budget/)

NO hard-coded numbers. NO synthetic/random data. All data must come from measured results.

Outputs:
  - results/figures/figure_a_scaling_speedup.png
  - results/figures/figure_b_cost_calibration.png
  - results/figures/figure_c_pareto_frontier.png
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
SAVE_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)


def _load_json(path: str) -> dict:
    """Load JSON file or raise clear error."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required result file not found: {path}\n"
            f"Run the corresponding experiment first to generate measured data."
        )
    with open(path, 'r') as f:
        return json.load(f)


def plot_figure_a():
    """Figure A: Selective Optimization Scaling & Speedup.
    
    Reads measured data from results/selective_compute/selective_scaling.json.
    """
    data_path = os.path.join(RESULTS_DIR, 'selective_compute', 'selective_scaling.json')
    all_results = _load_json(data_path)
    
    # Group by N
    sizes = sorted(set(r['n_total'] for r in all_results))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    plt.figure(figsize=(7, 5), dpi=300)
    for idx, N in enumerate(sizes):
        n_data = sorted(
            [r for r in all_results if r['n_total'] == N],
            key=lambda x: x['active_ratio'], reverse=True
        )
        pcts = [r['active_ratio'] * 100 for r in n_data]
        speedups = [r['opt_speedup'] for r in n_data]
        style = '-' if idx == 0 else ('--' if idx == 1 else ':')
        plt.plot(pcts, speedups, f'{markers[idx % len(markers)]}{style}',
                 color=colors[idx % len(colors)], linewidth=2.0, markersize=7,
                 label=f'N = {N:,d} Gaussians')
    
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='Break-Even (1.0x)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Active Gaussian Ratio $K = M/N$ (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Optimization Step Speedup over Baseline', fontsize=12, fontweight='bold')
    plt.title('Figure A: Selective Optimization Scaling Speedup', fontsize=13, fontweight='bold')
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    
    out_path = os.path.join(SAVE_DIR, 'figure_a_scaling_speedup.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")


def plot_figure_b():
    """Figure B: Cost Model Calibration Fit.
    
    Reads measured calibration data from results/cost_calibration/.
    Plots actual measured points and fitted model line.
    """
    # Load model comparison for fitted coefficients
    model_path = os.path.join(RESULTS_DIR, 'cost_calibration', 'model_comparison.json')
    model_data = _load_json(model_path)
    
    # Load measured observations if available
    obs_path = os.path.join(RESULTS_DIR, 'cost_calibration', 'observations.json')
    
    model_a = model_data.get('model_A', model_data.get('model_a', {}))
    T_0 = model_a.get('T_0_ms', 0.0)
    beta = model_a.get('beta_ms_per_gaussian', 0.0)
    r2 = model_a.get('r2', 0.0)
    
    plt.figure(figsize=(7, 5), dpi=300)
    
    # Plot measured observations if they exist
    if os.path.exists(obs_path):
        observations = _load_json(obs_path)
        obs_M = [o['M'] for o in observations]
        obs_T = [o['measured_ms'] for o in observations]
        plt.scatter(obs_M, obs_T, alpha=0.5, color='#d62728', edgecolors='k', s=30,
                    label='Measured Trials')
        M_range = np.linspace(min(obs_M) * 0.9, max(obs_M) * 1.1, 100)
    else:
        # If no per-observation file, still plot the model line based on available data
        print(f"  Note: Per-observation data not found at {obs_path}, plotting model line only.")
        M_range = np.linspace(100, 50000, 100)
    
    T_pred = T_0 + beta * M_range
    plt.plot(M_range, T_pred, color='#1f77b4', linewidth=2.5,
             label=f'Model A: $T_0$={T_0:.2f} + {beta:.5f}·M ($R^2$={r2:.4f})')
    
    plt.xlabel('Active Gaussian Count $M$', fontsize=12, fontweight='bold')
    plt.ylabel('Optimization Latency $T_{opt}$ (ms)', fontsize=12, fontweight='bold')
    plt.title('Figure B: Calibrated Optimization Cost Model', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    
    out_path = os.path.join(SAVE_DIR, 'figure_b_cost_calibration.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")


def plot_figure_c():
    """Figure C: Pareto Frontier of Quality vs Compute.
    
    Reads measured data from results/matched_budget/benchmark_results.json.
    """
    data_path = os.path.join(RESULTS_DIR, 'matched_budget', 'benchmark_results.json')
    results = _load_json(data_path)
    
    # Group by policy
    policy_colors = {
        'ours': '#d62728', 'budget_aware': '#d62728', 'knapsack': '#d62728',
        'random': '#7f7f7f',
        'error_only': '#ff7f0e', 'error-only': '#ff7f0e',
        'error_influence': '#9467bd', 'error-influence': '#9467bd',
        'top_k': '#1f77b4', 'top-k': '#1f77b4', 'topk': '#1f77b4',
        'binary': '#bcbd22',
        'full': '#2ca02c',
    }
    policy_markers = {
        'ours': 'o', 'budget_aware': 'o', 'knapsack': 'o',
        'random': 'x',
        'error_only': '^', 'error-only': '^',
        'error_influence': 'v', 'error-influence': 'v',
        'top_k': 's', 'top-k': 's', 'topk': 's',
        'binary': 'D',
        'full': '*',
    }
    
    # Aggregate per policy
    policy_data = {}
    for r in results:
        name = r.get('policy_name', r.get('policy', 'unknown')).lower().replace(' ', '_')
        if name not in policy_data:
            policy_data[name] = {'psnr': [], 'time': []}
        policy_data[name]['psnr'].append(r.get('avg_psnr', r.get('psnr', 0.0)))
        policy_data[name]['time'].append(r.get('measured_compute_ms', r.get('total_ms', 0.0)))
    
    plt.figure(figsize=(7, 5), dpi=300)
    for name, d in policy_data.items():
        mean_t = np.mean(d['time'])
        mean_p = np.mean(d['psnr'])
        color = policy_colors.get(name, '#333333')
        marker = policy_markers.get(name, 'o')
        display_name = name.replace('_', ' ').title()
        if 'full' in name:
            display_name += ' (Upper Bound)'
        plt.scatter(mean_t, mean_p, s=140, color=color, marker=marker,
                    label=display_name, edgecolors='black', linewidth=1.2)
    
    plt.xlabel('Measured Optimization Latency (ms) [Lower is Better]', fontsize=12, fontweight='bold')
    plt.ylabel('Reconstruction Quality (PSNR dB) [Higher is Better]', fontsize=12, fontweight='bold')
    plt.title('Figure C: Quality vs Compute Pareto Frontier', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    
    out_path = os.path.join(SAVE_DIR, 'figure_c_pareto_frontier.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")


def main():
    plot_figure_a()
    plot_figure_b()
    plot_figure_c()


if __name__ == '__main__':
    main()
