#!/usr/bin/env python3
"""Baseline Ranking Benchmark (Claim B & Step 4).

Compares ranking policies against Ground-Truth Oracle Utility U_i^{oracle}:
    1. Random
    2. Error-Only (E_rgb + E_depth)
    3. Error × Influence ((E_rgb + E_depth) · Influence_mass)
    4. Binary (RTG-SLAM stable/unstable)
    5. Heuristic Utility (Ours: Importance / Estimated Cost)
    6. Oracle (Upper Bound)

Evaluates:
    - Spearman ρ(Score, U_oracle)
    - Overlap@10%
    - Overlap@20%
    - Realized Gain Ratio@20%
    - Regret@20%

Outputs:
    - results/ranking_results.csv
    - results/figures/ranking_table.md
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_ranking_policy(
    policy_name: str,
    scores: np.ndarray,
    oracle_utils: np.ndarray,
    delta_qs: np.ndarray,
) -> Dict[str, Any]:
    """Compute rank correlation, Overlap@K, Realized Gain, and Regret for a policy."""
    n = len(scores)
    rho, p_val = spearmanr(scores, oracle_utils)
    
    score_ranks = np.argsort(-scores)
    oracle_ranks = np.argsort(-oracle_utils)
    
    overlaps = {}
    gains = {}
    regrets = {}
    
    for k_pct in [0.05, 0.10, 0.20]:
        k = max(1, int(n * k_pct))
        top_policy = set(score_ranks[:k].tolist())
        top_oracle = set(oracle_ranks[:k].tolist())
        
        ov = len(top_policy & top_oracle) / k
        overlaps[f'top_{int(k_pct*100)}pct'] = ov
        
        gain_pol = delta_qs[list(top_policy)].sum()
        gain_ora = delta_qs[list(top_oracle)].sum()
        gain_ratio = float(gain_pol / (gain_ora + 1e-8)) if gain_ora > 0 else 1.0
        gains[f'gain_{int(k_pct*100)}pct'] = gain_ratio
        regrets[f'regret_{int(k_pct*100)}pct'] = max(0.0, 1.0 - gain_ratio)
        
    return {
        'policy': policy_name,
        'spearman_rho': float(rho) if not np.isnan(rho) else 0.0,
        'p_value': float(p_val) if not np.isnan(p_val) else 1.0,
        'overlap_10pct': overlaps['top_10pct'],
        'overlap_20pct': overlaps['top_20pct'],
        'gain_ratio_20pct': gains['gain_20pct'],
        'regret_20pct': regrets['regret_20pct'],
    }


def main():
    print("=" * 80)
    print("           STEP 4: BASELINE UTILITY RANKING BENCHMARK (CLAIM B)")
    print("=" * 80)
    
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'oracle_dataset', 'oracle_dataset.json'
    )
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Oracle dataset not found at {dataset_path}. Run oracle experiment first.")
        
    with open(dataset_path, 'r') as f:
        rows = json.load(f)
        
    visible = [r for r in rows if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    print(f"Loaded {len(visible)} visible Gaussians with Ground-Truth Oracle measurements.\n")
    
    if len(visible) < 10:
        print("Insufficient samples for ranking benchmark.")
        return
        
    # Extract arrays
    n = len(visible)
    np.random.seed(42)
    
    oracle_utils = np.array([r.get('oracle_utility_joint', r.get('oracle_utility', 0.0)) for r in visible])
    delta_qs = np.array([r.get('delta_quality_local', 0.0) for r in visible])
    
    # 1. Random Policy
    random_scores = np.random.rand(n)
    
    # 2. Error-Only Policy
    error_scores = []
    for r in visible:
        feats = r.get('features', {})
        e_rgb = feats.get('rgb_error', 0.0)
        e_depth = feats.get('depth_error', 0.0)
        error_scores.append(e_rgb + e_depth)
    error_scores = np.array(error_scores)
    
    # 3. Error x Influence Policy
    error_inf_scores = []
    for r in visible:
        feats = r.get('features', {})
        e_rgb = feats.get('rgb_error', 0.0)
        e_depth = feats.get('depth_error', 0.0)
        inf = feats.get('influence_mass', r.get('influence_mass', 1.0))
        error_inf_scores.append((e_rgb + e_depth) * inf)
    error_inf_scores = np.array(error_inf_scores)
    
    # 4. Binary Policy (Tier A/B indicator)
    binary_scores = []
    for r in visible:
        tier = r.get('features', {}).get('tier', 2)
        score = 1.0 if tier in (0, 1) else 0.0
        # Add slight jitter for ranking tie-breaking
        score += 0.01 * np.random.rand()
        binary_scores.append(score)
    binary_scores = np.array(binary_scores)
    
    # 5. Heuristic Utility (Ours)
    heuristic_scores = np.array([r.get('predicted_utility', 0.0) for r in visible])
    
    # Evaluate all
    policies = [
        ('Random', random_scores),
        ('Error-Only', error_scores),
        ('Error × Influence', error_inf_scores),
        ('Binary (RTG-SLAM)', binary_scores),
        ('Heuristic Utility (Ours)', heuristic_scores),
        ('Oracle (Upper Bound)', oracle_utils),
    ]
    
    results = []
    for p_name, p_scores in policies:
        m = evaluate_ranking_policy(p_name, p_scores, oracle_utils, delta_qs)
        results.append(m)
        
    df = pd.DataFrame(results)
    
    # Save CSV
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'ranking_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Save Markdown Table
    md_lines = []
    md_lines.append("# Table 2: Utility Prediction & Ranking Fidelity (Claim B)")
    md_lines.append("")
    md_lines.append("Evaluated against Ground-Truth Oracle Utility ($U_i^{oracle} = \\Delta Q_i / \\Delta T_i$).")
    md_lines.append("")
    md_lines.append("| Method | Spearman $\\rho$ ↑ | Overlap@10% ↑ | Overlap@20% ↑ | Gain Ratio@20% ↑ | Regret@20% ↓ |")
    md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        bold = "**" if "Ours" in r['policy'] or "Oracle" in r['policy'] else ""
        md_lines.append(
            f"| {bold}{r['policy']}{bold} | "
            f"{bold}{r['spearman_rho']:+.4f}{bold} | "
            f"{r['overlap_10pct']:.1%} | "
            f"{r['overlap_20pct']:.1%} | "
            f"{r['gain_ratio_20pct']:.4f} | "
            f"{r['regret_20pct']:.4f} |"
        )
    md_lines.append("")
    
    fig_dir = os.path.join(save_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    table_path = os.path.join(fig_dir, 'ranking_table.md')
    with open(table_path, 'w') as f:
        f.write("\n".join(md_lines))
        
    print("\n" + "\n".join(md_lines))
    print(f"[Artifacts] Successfully saved ranking results to:")
    print(f"  - {csv_path}")
    print(f"  - {table_path}")


if __name__ == '__main__':
    main()
