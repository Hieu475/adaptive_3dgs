#!/usr/bin/env python3
"""Generate all 8 publication figures for the paper.

Figure 1: End-to-End Conceptual Framework (Gaussian State -> Utility -> Budget Selection -> Online Quality)
Figure 2: Ground-Truth Marginal Utility Distribution & Heavy Tails (Gate 1)
Figure 3: Stratified Negative Utility Breakdown across Geometry Regimes (Phase 2.3)
Figure 4: Group Non-Additivity & Interaction Error Curve I(S) and R_add(S) (Phase 4.1)
Figure 5: Equal-Compute Budget-Quality Curves across 10%-80% (Phase 8.2)
Figure 6: V0-V7 State Factor Ablation Progression (Phase 6)
Figure 7: Latency vs Reconstruction Quality Pareto Frontier (Phase 8.3)
Figure 8: 50-Frame Online Reconstruction Trajectory & Per-Frame Delta Gain (Phase 10)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean aesthetic style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def generate_all_figures():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(repo_root, 'results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    print("=== GENERATING COMPLETE PUBLICATION FIGURE SUITE (FIG 1 - FIG 8) ===")
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Conceptual Framework Diagram
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4), dpi=300)
    ax.axis('off')
    
    boxes = [
        ("1. Gaussian State\nVector (s_i)", 
         "• Spatial residual & grad\n• Attribution mass\n• Temporal drift & age\n• Uncertainty & cost",
         0.05, "#E8F0FE", "#1A73E8"),
        ("2. Marginal Utility\nPrediction (U_i*)", 
         "• U_i* = ΔQ_i / c_i\n• Two-Head MLP\n• Realized quality ΔQ\n• Sub-step time cost c_i",
         0.29, "#E6F4EA", "#137333"),
        ("3. Budget-Aware\nKnapsack (S_B)", 
         "• max Σ ΔQ_i\n• s.t. Σ c_i ≤ B_t\n• O(N log N) greedy\n• Bounded approx guarantee",
         0.53, "#FEF7E0", "#B06000"),
        ("4. Online Quality\nOptimization (Q_t)", 
         "• Differentiable rasterization\n• Background cache composite\n• Constant frame budget\n• Continuous trajectory",
         0.77, "#FCE8E6", "#C5221F"),
    ]
    
    for title, desc, x, bg_col, border_col in boxes:
        rect = patches.FancyBboxPatch(
            (x, 0.15), 0.18, 0.70,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor=bg_col, edgecolor=border_col, linewidth=1.8
        )
        ax.add_patch(rect)
        ax.text(x + 0.09, 0.73, title, fontsize=10.5, fontweight='bold',
                ha='center', va='center', color='#202124')
        ax.text(x + 0.09, 0.40, desc, fontsize=8.5,
                ha='center', va='center', color='#3C4043', linespacing=1.4)
        
    # Draw interconnecting arrows
    arrow_props = dict(facecolor='#5F6368', edgecolor='none', width=1.5, headwidth=7)
    for arr_x in [0.24, 0.48, 0.72]:
        ax.annotate('', xy=(arr_x + 0.045, 0.50), xytext=(arr_x, 0.50),
                    arrowprops=arrow_props)
        
    ax.set_title("Figure 1: End-to-End Causal Framework for Budget-Constrained Gaussian Splatting", 
                 fontsize=12, fontweight='bold', pad=15)
    f1_path = os.path.join(fig_dir, 'fig1_conceptual_framework.png')
    plt.tight_layout()
    plt.savefig(f1_path)
    plt.close()
    print(f">> [Figure 1] Saved Conceptual Diagram to {f1_path}")
    
    # -------------------------------------------------------------------------
    # FIGURE 2: Marginal Utility Long-Tailed Distribution (Gate 1)
    # -------------------------------------------------------------------------
    oracle_file = os.path.join(repo_root, 'results', 'oracle_dataset', 'oracle_dataset.json')
    with open(oracle_file, 'r') as f:
        oracle_data = json.load(f)
    u_stars = np.array([float(r['oracle_utility_joint']) for r in oracle_data if r.get('visible', True)])
    
    plt.figure(figsize=(7, 4.5), dpi=300)
    # Clip extreme outliers for clean visualization
    u_plot = u_stars[np.abs(u_stars) < np.percentile(np.abs(u_stars), 98)]
    
    n_bins = 40
    counts, bins, patches_hist = plt.hist(u_plot * 1e4, bins=n_bins, color='#1A73E8', alpha=0.7, edgecolor='black', linewidth=0.6)
    
    # Highlight negative utility region
    for patch, left_edge in zip(patches_hist, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor('#D93025')
            patch.set_alpha(0.8)
            
    plt.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Neutral Utility ($U^* = 0$)')
    plt.axvline(np.mean(u_stars) * 1e4, color='#137333', linestyle='-', linewidth=1.5, label=f'Mean $U^* = +{np.mean(u_stars)*1e4:.2f}$')
    
    plt.xlabel(r'Counterfactual Marginal Utility $U_i^\star = \Delta Q_i / c_i$ ($\times 10^{-4}$)', fontsize=11, fontweight='bold')
    plt.ylabel('Gaussian Intervention Count', fontsize=11, fontweight='bold')
    plt.title('Figure 2: Long-Tailed Distribution of Empirical Marginal Utility (Gate 1)', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=9.5)
    plt.tight_layout()
    f2_path = os.path.join(fig_dir, 'fig2_marginal_utility_distribution.png')
    plt.savefig(f2_path)
    plt.close()
    print(f">> [Figure 2] Saved Marginal Utility Distribution to {f2_path}")
    
    # -------------------------------------------------------------------------
    # FIGURE 3: Negative Utility by Stratum (Phase 2.3)
    # -------------------------------------------------------------------------
    gate1_file = os.path.join(repo_root, 'results', 'gate1_headroom', 'gate1_summary.json')
    with open(gate1_file, 'r') as f:
        gate1_data = json.load(f)
    strata_rows = gate1_data['negative_utility_strata']
    
    df_st = pd.DataFrame(strata_rows)
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    
    x_pos = np.arange(len(df_st))
    strata_labels = [s.replace('_', ' ').title() for s in df_st['stratum']]
    
    # Bar for Negative %
    bars = ax1.bar(x_pos - 0.15, df_st['pct_negative'], width=0.3, color='#D93025', alpha=0.85, label='% Degraded ($U^* < 0$)')
    ax1.set_ylabel('Degradation Frequency (%)', color='#D93025', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 20)
    ax1.tick_params(axis='y', labelcolor='#D93025')
    
    # Dual axis for Mean Utility
    ax2 = ax1.twinx()
    lines = ax2.plot(x_pos + 0.15, df_st['mean_u'] * 1e4, color='#1A73E8', marker='o', linewidth=2.0, markersize=7, label=r'Mean $U^\star$ ($\times 10^{-4}$)')
    ax2.set_ylabel(r'Mean Marginal Utility $U^*$ ($\times 10^{-4}$)', color='#1A73E8', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#1A73E8')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(strata_labels, fontsize=10, fontweight='bold')
    ax1.set_title('Figure 3: Stratified Prevalence of Negative Utility (Degradation)', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', frameon=True, fontsize=9.5)
    
    plt.tight_layout()
    f3_path = os.path.join(fig_dir, 'fig3_negative_utility_by_stratum.png')
    plt.savefig(f3_path)
    plt.close()
    print(f">> [Figure 3] Saved Negative Utility Stratum Breakdown to {f3_path}")
    
    # -------------------------------------------------------------------------
    # FIGURE 4: Group Non-Additivity & Interaction Error (Phase 4.1)
    # -------------------------------------------------------------------------
    group_csv = os.path.join(repo_root, 'results', 'gate1_headroom', 'group_interaction_curve.csv')
    df_grp = pd.read_csv(group_csv)
    
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    
    # Interaction Error I(S) on left axis (log scale)
    color1 = '#B06000'
    ax1.plot(df_grp['group_size'], df_grp['interaction_error_mean'], color=color1, marker='s', linewidth=2, label=r'Interaction Error $I(S)$')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('symlog', linthresh=1.0)
    ax1.set_xlabel('Group Cardinality $|S|$', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Interaction Error $I(S) = \frac{|\Delta Q(S) - \sum \Delta Q_i|}{|\Delta Q(S)| + \epsilon}$', color=color1, fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Additivity Ratio R_add on right axis
    ax2 = ax1.twinx()
    color2 = '#137333'
    ax2.plot(df_grp['group_size'], df_grp['additivity_ratio_mean'], color=color2, marker='^', linestyle='--', linewidth=2, label=r'Additivity Ratio $R_{add}(S)$')
    ax2.set_ylabel(r'Additivity Ratio $R_{add}(S) = \frac{\Delta Q(S)}{\sum \Delta Q_i}$', color=color2, fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.05, 1.1)
    
    ax1.set_xticks(df_grp['group_size'])
    ax1.set_xticklabels(df_grp['group_size'], fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_title('Figure 4: Non-Additivity and Interaction Error vs Group Cardinality', fontsize=11, fontweight='bold')
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='center right', frameon=True, fontsize=9.5)
    
    plt.tight_layout()
    f4_path = os.path.join(fig_dir, 'fig4_group_interaction_error.png')
    plt.savefig(f4_path)
    plt.close()
    print(f">> [Figure 4] Saved Group Non-Additivity Curve to {f4_path}")
    
    # -------------------------------------------------------------------------
    # FIGURE 5: Budget-Quality Curve across 10%-80% (Phase 8.2)
    # -------------------------------------------------------------------------
    budget_file = os.path.join(repo_root, 'results', 'budget_sweep', 'phase6_budget_sweep.json')
    with open(budget_file, 'r') as f:
        b_data = json.load(f)['budget_sweep']
    df_b = pd.DataFrame(b_data)
    
    plt.figure(figsize=(8, 4.8), dpi=300)
    styles = {
        'Oracle Upper Bound': ('black', '--', 'o', 'Oracle Reference Upper Bound'),
        'Heuristic Knapsack (Ours)': ('#1A73E8', '-', '^', 'Heuristic Knapsack (Ours)'),
        'Learned Two-Head (Ours)': ('#137333', '-', 's', 'Learned Two-Head Ranking (Ours)'),
        'Error × Influence': ('#F2994A', '-.', 'v', 'Error × Influence Baseline'),
        'Error-Only Top-K': ('#D93025', ':', 'x', 'Error-Only Top-K Baseline'),
        'Random Baseline': ('#80868B', ':', 'd', 'Random Selection Baseline'),
    }
    
    for pol_name, (col, ls, marker, label) in styles.items():
        sub = df_b[df_b['policy'] == pol_name]
        if not sub.empty:
            plt.plot(sub['budget_pct'], sub['delta_quality'] * 1e4, color=col, linestyle=ls, marker=marker, linewidth=2, label=label)
            
    plt.xlabel('Compute Budget Capacity $B$ (%)', fontsize=11, fontweight='bold')
    plt.ylabel(r'Realized Joint Reconstruction Gain $\Delta Q(S_B)$ ($\times 10^{-4}$)', fontsize=11, fontweight='bold')
    plt.title('Figure 5: Reconstruction Gain vs Matched Compute Budget Capacity', fontsize=11.5, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=9.5, loc='upper left')
    plt.tight_layout()
    f5_path = os.path.join(fig_dir, 'fig5_budget_quality_curve.png')
    plt.savefig(f5_path)
    plt.close()
    print(f">> [Figure 5] Saved Budget-Quality Curve to {f5_path}")
    
    # -------------------------------------------------------------------------
    # FIGURE 6: V0-V7 State Factor Ablation Progression (Phase 6)
    # -------------------------------------------------------------------------
    ablation_file = os.path.join(repo_root, 'results', 'learned_utility', 'learned_utility_summary.json')
    with open(ablation_file, 'r') as f:
        abl_data = json.load(f)['v0_v7_ablation']
    df_abl = pd.DataFrame(abl_data)
    
    fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=300)
    x_pos = np.arange(len(df_abl))
    v_labels = [row['version'].split(':')[0] for _, row in df_abl.iterrows()]
    
    ax1.plot(x_pos, df_abl['spearman_rho'], color='#1A73E8', marker='o', linewidth=2.0, label=r'Fidelity $\rho(\hat{U}, U^*)$')
    ax1.plot(x_pos, df_abl['ndcg_20pct'], color='#137333', marker='s', linestyle='--', linewidth=2.0, label='NDCG@20%')
    ax1.set_ylabel('Ranking Metric Value', fontsize=11, fontweight='bold')
    ax1.set_ylim(min(0.0, df_abl['spearman_rho'].min() - 0.1), 1.05)
    
    ax2 = ax1.twinx()
    ax2.bar(x_pos, df_abl['ose_20pct'], alpha=0.25, color='#F2994A', width=0.4, label='OSE@20%')
    ax2.set_ylabel('Oracle Selection Efficiency (OSE)', color='#B06000', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#B06000')
    ax2.set_ylim(0, 1.0)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(v_labels, fontsize=10, fontweight='bold')
    ax1.set_xlabel('Feature Factor Ablation Version (V0 = Appearance $\\to$ V7 = Full State)', fontsize=11, fontweight='bold')
    ax1.set_title('Figure 6: Progressive State Factor Ablation (Fidelity $\\to$ Ranking $\\to$ Selection)', fontsize=11.5, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower left', frameon=True, fontsize=9.5)
    
    plt.tight_layout()
    f6_path = os.path.join(fig_dir, 'fig6_feature_ablation_progression.png')
    plt.savefig(f6_path)
    plt.close()
    print(f">> [Figure 6] Saved Feature Ablation Progression to {f6_path}")
    
    # -------------------------------------------------------------------------
    # FIGURE 7: Latency vs Quality Pareto Frontier (Phase 8.3)
    # -------------------------------------------------------------------------
    pareto_file = os.path.join(repo_root, 'results', 'budget_sweep', 'pareto_frontier.csv')
    df_par = pd.read_csv(pareto_file)
    
    plt.figure(figsize=(8, 4.8), dpi=300)
    for pol_name, (col, _, marker, label) in styles.items():
        sub = df_par[df_par['policy'] == pol_name]
        if not sub.empty:
            plt.plot(sub['latency_ms'], sub['delta_quality'] * 1e4, color=col, alpha=0.7, linestyle='--')
            plt.scatter(sub['latency_ms'], sub['delta_quality'] * 1e4, color=col, marker=marker, s=65, label=label)
            
    full_pt = df_par[df_par['policy'] == 'Full Optimization']
    if not full_pt.empty:
        plt.scatter(full_pt['latency_ms'], full_pt['delta_quality'] * 1e4, color='#7B1FA2', marker='*', s=160, label='Full Unconstrained (100% compute)', zorder=5)
        
    plt.xlabel('Optimization Latency (ms)', fontsize=11, fontweight='bold')
    plt.ylabel(r'Realized Joint Reconstruction Gain $\Delta Q$ ($\times 10^{-4}$)', fontsize=11, fontweight='bold')
    plt.title('Figure 7: Latency vs Reconstruction Quality Pareto Frontier', fontsize=11.5, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=9.5, loc='upper left')
    plt.tight_layout()
    f7_path = os.path.join(fig_dir, 'fig7_pareto_frontier.png')
    plt.savefig(f7_path)
    plt.close()
    print(f">> [Figure 7] Saved Latency vs Quality Pareto Frontier to {f7_path}")
    
    # -------------------------------------------------------------------------
    # FIGURE 8: Online Trajectory PSNR & Per-Frame Quality Delta (Phase 10)
    # -------------------------------------------------------------------------
    deltas_csv = os.path.join(repo_root, 'results', 'online_trajectory', 'per_frame_deltas.csv')
    df_tr = pd.read_csv(deltas_csv)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=300, sharex=True)
    frames_x = df_tr['frame']
    
    ax1.plot(frames_x, df_tr['psnr_full'], 'k--', label='Full Unconstrained', linewidth=1.8, alpha=0.8)
    ax1.plot(frames_x, df_tr['psnr_ours'], color='#137333', label='Ours (Selective Utility Knapsack @ 25% Budget)', linewidth=2.2)
    ax1.plot(frames_x, df_tr['psnr_error'], color='#D93025', linestyle='-.', label='Error-Only Top-K @ 25% Budget', linewidth=1.8)
    ax1.plot(frames_x, df_tr['psnr_random'], color='#80868B', linestyle=':', label='Random Baseline @ 25% Budget', linewidth=1.5)
    ax1.set_ylabel('Reconstruction PSNR (dB)', fontsize=10.5, fontweight='bold')
    ax1.set_title('(a) Online Trajectory Quality over 50 Consecutive Frames', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', frameon=True, fontsize=9)
    
    # Plot frame delta
    ax2.bar(frames_x, df_tr['delta_q_vs_random'], color='#137333', alpha=0.75, width=0.8, label=r'$\Delta Q_t = \mathrm{PSNR}_{\mathrm{ours}} - \mathrm{PSNR}_{\mathrm{random}}$')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.2, label='Baseline (ΔQ = 0)')
    mean_d = df_tr['delta_q_vs_random'].mean()
    ax2.axhline(mean_d, color='darkgreen', linestyle='-', linewidth=1.5, label=f'Mean ΔQ = {mean_d:+.2f} dB')
    ax2.set_xlabel('Online Sequence Frame Index ($t$)', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel('Quality Gain ΔQ (dB)', fontsize=10.5, fontweight='bold')
    ax2.set_title(r'(b) Frame-by-Frame Realized Reconstruction Gain over Random $\Delta Q_t$', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    f8_path = os.path.join(fig_dir, 'fig8_online_trajectory.png')
    plt.savefig(f8_path)
    plt.close()
    print(f">> [Figure 8] Saved Online Trajectory Figure to {f8_path}")
    
    print("\n[Complete] All Figures 1–8 successfully rendered and saved.")


if __name__ == '__main__':
    generate_all_figures()
