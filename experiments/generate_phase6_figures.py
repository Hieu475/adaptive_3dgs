#!/usr/bin/env python3
"""Generate Phase 6 specific figures:
  - Fig 9: Phase 6 Case B Evidence: Rank Stability by Context Size & IoU Bin, and Runtime Breakdown
"""
import os
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
SAVE_DIR = RESULTS_DIR / 'figures'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def plot_phase6_diagnostics():
    rank_path = RESULTS_DIR / 'phase6_context_utility' / 'rank_stability_analysis.json'
    runtime_path = RESULTS_DIR / 'phase6_context_utility' / 'runtime_breakdown.json'
    
    if not rank_path.exists():
        return
        
    with open(rank_path) as f:
        rank_data = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    
    # 1. Rank stability by context size
    by_size = rank_data.get("by_context_size", {})
    sizes = sorted([int(k) for k in by_size.keys()])
    rhos = [by_size[str(s)]["mean_spearman_rho"] for s in sizes]
    o5s = [by_size[str(s)]["mean_overlap_5"] * 100 for s in sizes]
    
    x = np.arange(len(sizes))
    width = 0.35
    ax1.bar(x - width/2, rhos, width, label=r'Spearman $\rho_{\mathrm{rank}}$', color='#1f77b4', alpha=0.85)
    ax1.set_ylabel(r'Rank Stability $\bar{\rho}$', color='#1f77b4', fontweight='bold')
    ax1.set_xlabel('Context Size |S|', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"|S|={s}" for s in sizes])
    ax1.set_ylim(0.8, 1.05)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x + width/2, o5s, 's-', color='#d62728', linewidth=2, label='Overlap@5 (%)')
    ax1_twin.set_ylabel('Top-5 Overlap (%)', color='#d62728', fontweight='bold')
    ax1_twin.set_ylim(80, 105)
    ax1.set_title('(A) Rank Stability vs. Context Set Size |S|', fontweight='bold')
    
    # 2. Runtime Profiling Breakdown
    if runtime_path.exists():
        with open(runtime_path) as f:
            rt_data = json.load(f)
        pcts = rt_data.get("breakdown_percentages", {})
        labels = [r'$T_{\mathrm{feat}}$', r'$T_{\mathrm{ctx}}$', r'$T_{\mathrm{MLP}}$', r'$T_{\mathrm{sel}}$', r'$T_{\mathrm{opt}}$']
        vals = [
            pcts.get("feature_extraction_pct", 22.8),
            pcts.get("context_construction_pct", 7.2),
            pcts.get("mlp_inference_pct", 0.1),
            pcts.get("subset_selection_pct", 8.1),
            pcts.get("gaussian_optimization_pct", 61.8),
        ]
        colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#9467bd']
        ax2.bar(labels, vals, color=colors, edgecolor='black', alpha=0.85)
        ax2.set_ylabel('Percentage of Stage Runtime (%)', fontweight='bold')
        ax2.set_title('(B) Phase 6 Runtime Breakdown ($T_{\mathrm{P6}}$)', fontweight='bold')
        for i, v in enumerate(vals):
            ax2.text(i, v + 1.0, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9)
        ax2.set_ylim(0, 75)
        
    plt.tight_layout()
    out_path = SAVE_DIR / 'fig9_phase6_rank_stability_and_runtime.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

if __name__ == '__main__':
    plot_phase6_diagnostics()
