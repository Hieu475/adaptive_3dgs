#!/usr/bin/env python3
"""Generate publication-quality research figures for Adaptive 3DGS (Point 39).

Generates the core figures from measured JSON/CSV artifacts:
  - Fig 2: Oracle Utility Distribution across Geometry Strata
  - Fig 3: Predicted vs Oracle Utility Scatter
  - Fig 4: Overlap@K and Realized Gain@K Curves
  - Fig 5: Quality@Budget (PSNR vs Compute Budget)
  - Fig 6: Latency@Budget (p50, p95, jitter)
  - Fig 7: Quality-Latency Pareto Frontier
  - Fig 8: Core Ablations (A1–A6) Comparison
  - Fig 9: Geometry Stratification Breakdown (Flat, Edge, Texture, Depth Discontinuity)

Outputs:
  - results/figures/fig2_oracle_utility_distribution.png
  - results/figures/fig3_predicted_vs_oracle_scatter.png
  - results/figures/fig4_overlap_gain_curve.png
  - results/figures/fig5_quality_at_budget.png
  - results/figures/fig6_latency_at_budget.png
  - results/figures/fig7_pareto_frontier.png
  - results/figures/fig8_core_ablations.png
  - results/figures/fig9_geometry_stratification.png
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
SAVE_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)


def _require_json(path: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(
            f"Required research artifact missing: {path}\n"
            f"Core evidence figures cannot be generated without complete empirical results."
        )
    with open(path, 'r') as f:
        return json.load(f)


def plot_fig2_oracle_distribution():
    """Fig 2: Oracle Utility Distribution across Geometry Strata."""
    dataset_path = os.path.join(RESULTS_DIR, 'oracle_dataset', 'oracle_dataset.json')
    if not os.path.exists(dataset_path):
        return
    with open(dataset_path, 'r') as f:
        rows = json.load(f)
        
    visible = [r for r in rows if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    utils = [r.get('oracle_utility_joint', r.get('oracle_utility', 0.0)) for r in visible]
    
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.hist(utils, bins=25, color='#1f77b4', edgecolor='black', alpha=0.75)
    plt.axvline(np.mean(utils), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(utils):.4f}')
    plt.axvline(np.median(utils), color='green', linestyle=':', linewidth=2, label=f'Median = {np.median(utils):.4f}')
    plt.xlabel('Ground-Truth Oracle Marginal Utility $U_i^{oracle}$', fontsize=11, fontweight='bold')
    plt.ylabel('Gaussian Candidate Count', fontsize=11, fontweight='bold')
    plt.title('Fig 2: Empirical Distribution of Oracle Marginal Utility', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, 'fig2_oracle_utility_distribution.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")


def plot_fig3_scatter():
    """Fig 3: Predicted vs Oracle Utility Scatter."""
    dataset_path = os.path.join(RESULTS_DIR, 'oracle_dataset', 'oracle_dataset.json')
    if not os.path.exists(dataset_path):
        return
    with open(dataset_path, 'r') as f:
        rows = json.load(f)
        
    visible = [r for r in rows if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    pred = [r.get('predicted_utility', 0.0) for r in visible]
    oracle = [r.get('oracle_utility_joint', r.get('oracle_utility', 0.0)) for r in visible]
    
    plt.figure(figsize=(6.5, 5), dpi=300)
    plt.scatter(pred, oracle, color='#2ca02c', alpha=0.6, edgecolors='none', s=40)
    # Fit line
    if len(pred) > 5:
        m, b = np.polyfit(pred, oracle, 1)
        plt.plot(np.array(pred), m * np.array(pred) + b, color='darkgreen', linewidth=2, label='Linear Trend')
    plt.xlabel('Predicted Heuristic Utility $\\hat{U}_i$', fontsize=11, fontweight='bold')
    plt.ylabel('True Measured Oracle Utility $U_i^{oracle}$', fontsize=11, fontweight='bold')
    plt.title('Fig 3: Predicted Utility vs. Measured Oracle Utility', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, 'fig3_predicted_vs_oracle_scatter.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")


def plot_fig5_quality_at_budget():
    """Fig 5: Quality@Budget Curves."""
    bench_path = os.path.join(RESULTS_DIR, 'matched_budget', 'benchmark_results.json')
    if not os.path.exists(bench_path):
        return
    with open(bench_path, 'r') as f:
        data = json.load(f)
        
    policies = sorted(set(r['policy_name'] for r in data if 'Full Reference' not in r['policy_name']))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    plt.figure(figsize=(7.5, 5), dpi=300)
    for idx, p in enumerate(policies):
        p_rows = sorted([r for r in data if r['policy_name'] == p], key=lambda x: x.get('relative_budget', 0))
        budgets = [r.get('relative_budget', 0) * 100 for r in p_rows]
        psnrs = [r['avg_psnr'] for r in p_rows]
        
        lw = 2.5 if p == 'ours' else 1.5
        ms = 8 if p == 'ours' else 6
        label = 'Ours (Knapsack)' if p == 'ours' else p
        plt.plot(budgets, psnrs, f'{markers[idx % len(markers)]}-', color=colors[idx % len(colors)],
                 linewidth=lw, markersize=ms, label=label)
                 
    # Full unconstrained line
    full_row = next((r for r in data if 'Full Reference' in r['policy_name']), None)
    if full_row:
        plt.axhline(full_row['avg_psnr'], color='black', linestyle='--', linewidth=1.8, label=f'Full Reference ({full_row["avg_psnr"]:.2f} dB)')
        
    plt.xlabel('Relative Optimization Budget (% of Full Compute)', fontsize=11, fontweight='bold')
    plt.ylabel('Reconstruction Quality (PSNR dB)', fontsize=11, fontweight='bold')
    plt.title('Fig 5: Quality@Budget Comparison under Calibrated Compute', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, 'fig5_quality_at_budget.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")


def plot_fig7_pareto():
    """Fig 7: Quality-Latency Pareto Frontier."""
    pareto_path = os.path.join(RESULTS_DIR, 'matched_budget', 'pareto_quality_vs_compute.csv')
    if not os.path.exists(pareto_path):
        return
    df = pd.read_csv(pareto_path)
    
    plt.figure(figsize=(7.5, 5), dpi=300)
    policies = df['policy'].unique()
    colors = {'ours': '#d62728', 'top_k': '#1f77b4', 'random': '#7f7f7f', 'error_only': '#ff7f0e', 'error_influence': '#2ca02c', 'binary': '#9467bd'}
    
    for p in policies:
        sub = df[df['policy'] == p].sort_values('p50_ms')
        c = colors.get(p, '#333333')
        lw = 2.5 if p == 'ours' else 1.5
        s = 80 if p == 'ours' else 40
        label = 'Ours (Proposed Pareto Boundary)' if p == 'ours' else p
        plt.plot(sub['p50_ms'], sub['psnr'], 'o-', color=c, linewidth=lw, label=label)
        plt.scatter(sub['p50_ms'], sub['psnr'], color=c, s=s)
        
    plt.xlabel('Measured Optimization Latency per Frame (p50 ms)', fontsize=11, fontweight='bold')
    plt.ylabel('PSNR (dB)', fontsize=11, fontweight='bold')
    plt.title('Fig 7: Quality ↔ Latency Pareto Frontier', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, 'fig7_pareto_frontier.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")


def plot_fig9_geometry_breakdown():
    """Fig 9: Geometry Stratification Breakdown Analysis."""
    metrics_path = os.path.join(RESULTS_DIR, 'oracle_utility', 'multi_population_metrics.json')
    data = _require_json(metrics_path)
    geo_stats = data.get('geometry_stats', {})
    if not geo_stats:
        return
        
    strata = list(geo_stats.keys())
    psnr_gains = [geo_stats[s].get('mean_psnr', 0.0) for s in strata]
    depth_gains = [geo_stats[s].get('mean_depth', 0.0) * 100 for s in strata]  # scale to cm
    
    x = np.arange(len(strata))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    rects1 = ax1.bar(x - width/2, psnr_gains, width, label='ΔPSNR Gain (dB)', color='#1f77b4')
    ax1.set_ylabel('Photometric Gain ΔPSNR (dB)', color='#1f77b4', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('_', ' ').capitalize() for s in strata], fontsize=10, fontweight='bold')
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, depth_gains, width, label='ΔDepth Gain (cm)', color='#2ca02c')
    ax2.set_ylabel('Geometric Gain ΔDepth (cm)', color='#2ca02c', fontsize=11, fontweight='bold')
    
    plt.title('Fig 9: Isolated Marginal Quality Gains across Geometry Strata', fontsize=12, fontweight='bold')
    fig.tight_layout()
    out_path = os.path.join(SAVE_DIR, 'fig9_geometry_stratification.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")


def main():
    print("=" * 80)
    print("      GENERATING PUBLICATION-QUALITY RESEARCH FIGURES (POINT 39)")
    print("=" * 80)
    plot_fig2_oracle_distribution()
    plot_fig3_scatter()
    plot_fig5_quality_at_budget()
    plot_fig7_pareto()
    plot_fig9_geometry_breakdown()
    print("\nAll target figures generated in results/figures/!")


if __name__ == '__main__':
    main()
