#!/usr/bin/env python3
"""Phase 3: Validate Heuristic Utility (Points LXXI, XXXV–XXXIX).

Compares observable heuristic proxies against Ground-Truth Oracle Marginal Utility U_i^*:
    1. Random Baseline
    2. Color Error Alone (E_rgb)
    3. Depth Error Alone (E_depth)
    4. Combined Error (E_rgb + E_depth)
    5. Error × Influence ((E_rgb + E_depth) * Influence_mass)
    6. Temporal Drift Alone (Drift)
    7. Binary Tier Baseline (RTG-SLAM stable/unstable)
    8. Heuristic Utility (Ours: Pre-fusion Normalized Importance / Predicted Cost)
    9. Oracle Upper Bound (U_i^*)

Evaluates across:
    - Spearman rho vs U_joint, U_rgb, U_depth (safe spearman, no fake 1.0)
    - NDCG@10%, NDCG@20%
    - Overlap@10%, Overlap@20%
    - Oracle Selection Efficiency: OSE@20% = Delta Q(S) / Delta Q(S*)
    - Absolute Selection Regret: R_20% = Delta Q(S*) - Delta Q(S)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def safe_spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3 or np.std(x) < 1e-7 or np.std(y) < 1e-7:
        return float('nan'), float('nan')
    r, p = spearmanr(x, y)
    return (float(r) if not np.isnan(r) else float('nan')), (float(p) if not np.isnan(p) else float('nan'))


def compute_ndcg(pred_scores: np.ndarray, true_scores: np.ndarray, k: int) -> float:
    if k <= 0 or len(pred_scores) == 0:
        return 0.0
    k_eval = min(k, len(pred_scores))
    p_idx = np.argsort(-pred_scores)[:k_eval]
    i_idx = np.argsort(-true_scores)[:k_eval]
    min_val = min(0.0, float(np.min(true_scores)))
    rel = true_scores - min_val
    discounts = np.log2(np.arange(2, k_eval + 2))
    dcg = np.sum(rel[p_idx] / discounts)
    idcg = np.sum(rel[i_idx] / discounts)
    return float(dcg / (idcg + 1e-8)) if idcg > 0 else 1.0


def evaluate_ranking_policy(
    policy_name: str,
    scores: np.ndarray,
    oracle_joint: np.ndarray,
    oracle_rgb: np.ndarray,
    oracle_depth: np.ndarray,
    delta_qs: np.ndarray,
) -> Dict[str, Any]:
    """Compute rank correlations, NDCG@K, Overlap@K, OSE, and Regret for a policy."""
    n = len(scores)
    rho_joint, p_joint = safe_spearmanr(scores, oracle_joint)
    rho_rgb, p_rgb = safe_spearmanr(scores, oracle_rgb)
    rho_depth, p_depth = safe_spearmanr(scores, oracle_depth)
    
    score_ranks = np.argsort(-scores)
    oracle_ranks = np.argsort(-oracle_joint)
    
    k10 = max(1, int(n * 0.10))
    k20 = max(1, int(n * 0.20))
    
    # Overlaps
    top_pol_10 = set(score_ranks[:k10].tolist())
    top_ora_10 = set(oracle_ranks[:k10].tolist())
    ov10 = len(top_pol_10 & top_ora_10) / k10
    
    top_pol_20 = set(score_ranks[:k20].tolist())
    top_ora_20 = set(oracle_ranks[:k20].tolist())
    ov20 = len(top_pol_20 & top_ora_20) / k20
    
    # Realized Gains
    gain_pol_20 = delta_qs[list(top_pol_20)].sum()
    gain_ora_20 = delta_qs[list(top_ora_20)].sum()
    
    ose_20 = float(gain_pol_20 / (gain_ora_20 + 1e-8)) if gain_ora_20 > 0 else 1.0
    regret_20_abs = float(gain_ora_20 - gain_pol_20)
    
    ndcg_10 = compute_ndcg(scores, oracle_joint, k10)
    ndcg_20 = compute_ndcg(scores, oracle_joint, k20)
    
    return {
        'policy': policy_name,
        'spearman_joint': rho_joint,
        'p_val_joint': p_joint,
        'spearman_rgb': rho_rgb,
        'spearman_depth': rho_depth,
        'ndcg_10pct': ndcg_10,
        'ndcg_20pct': ndcg_20,
        'overlap_10pct': ov10,
        'overlap_20pct': ov20,
        'ose_20pct': ose_20,
        'regret_20pct_abs': regret_20_abs,
    }


def main():
    print("=" * 90)
    print("        PHASE 3: VALIDATE HEURISTIC UTILITY BENCHMARK (POINTS LXXI, XXXV–XXXIX)")
    print("=" * 90)
    
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'oracle_dataset', 'oracle_dataset.json'
    )
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Oracle dataset not found at {dataset_path}.")
        
    with open(dataset_path, 'r') as f:
        rows = json.load(f)
        
    visible = [r for r in rows if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    print(f">> Loaded {len(visible)} visible Gaussians with Ground-Truth Oracle measurements.\n")
    
    if len(visible) < 10:
        print("Insufficient samples for ranking benchmark.")
        return
        
    n = len(visible)
    np.random.seed(42)
    
    oracle_joint = np.array([r.get('oracle_utility_joint', r.get('oracle_utility', 0.0)) for r in visible])
    oracle_rgb = np.array([r.get('oracle_utility_rgb', 0.0) for r in visible])
    oracle_depth = np.array([r.get('oracle_utility_depth', 0.0) for r in visible])
    delta_qs = np.array([r.get('delta_quality', r.get('delta_quality_global', 0.0)) for r in visible])
    
    # 1. Random Policy
    random_scores = np.random.rand(n)
    
    # 2. Color Error Alone
    rgb_err_scores = np.array([float(r.get('features', {}).get('rgb_error', 0.0)) for r in visible])
    
    # 3. Depth Error Alone
    depth_err_scores = np.array([float(r.get('features', {}).get('depth_error', 0.0)) for r in visible])
    
    # 4. Error-Only (E_rgb + E_depth)
    error_scores = rgb_err_scores + depth_err_scores
    
    # 5. Error x Influence
    inf_mass = np.array([float(r.get('features', {}).get('influence_mass', r.get('influence_mass', 1.0))) for r in visible])
    error_inf_scores = error_scores * inf_mass
    
    # 6. Temporal Position Drift Alone
    temp_scores = np.array([float(r.get('features', {}).get('position_drift', r.get('features', {}).get('temporal_drift', 0.0))) for r in visible])
    
    # 7. Binary Tier Policy (RTG-SLAM stable/unstable)
    binary_scores = []
    for r in visible:
        tier = r.get('features', {}).get('tier', 2)
        score = 1.0 if tier in (0, 1) else 0.0
        score += 0.01 * np.random.rand()
        binary_scores.append(score)
    binary_scores = np.array(binary_scores)
    
    # 8. Heuristic Utility (Ours: Pre-fusion Norm Importance / Cost)
    heuristic_scores = np.array([float(r.get('predicted_utility', 0.0)) for r in visible])
    
    policies = [
        ('Random', random_scores),
        ('Color-Error Alone', rgb_err_scores),
        ('Depth-Error Alone', depth_err_scores),
        ('Error-Only (RGB + Depth)', error_scores),
        ('Error × Influence', error_inf_scores),
        ('Temporal Drift Alone', temp_scores),
        ('Binary (RTG-SLAM)', binary_scores),
        ('Heuristic Utility (Ours)', heuristic_scores),
        ('Oracle (Upper Bound)', oracle_joint),
    ]
    
    results = []
    for p_name, p_scores in policies:
        m = evaluate_ranking_policy(p_name, p_scores, oracle_joint, oracle_rgb, oracle_depth, delta_qs)
        results.append(m)
        
    df = pd.DataFrame(results)
    
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'ranking_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Markdown Table
    md_lines = [
        "# Phase 3: Heuristic Utility Validation Benchmark",
        "",
        "Evaluated against Ground-Truth Oracle Marginal Utility ($U_i^\\star = \\Delta Q_i / \\Delta T_i$).",
        "",
        "| Method | $\\rho(U^\\star_{joint})$ ↑ | $\\rho(U^\\star_{rgb})$ | $\\rho(U^\\star_{depth})$ | NDCG@20% ↑ | Overlap@20% ↑ | OSE@20% ↑ | Regret@20% ↓ |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    
    for r in results:
        bold = "**" if "Ours" in r['policy'] or "Oracle" in r['policy'] else ""
        rho_j_str = f"{r['spearman_joint']:+.4f}" if not np.isnan(r['spearman_joint']) else "NaN"
        rho_rgb_str = f"{r['spearman_rgb']:+.4f}" if not np.isnan(r['spearman_rgb']) else "NaN"
        rho_d_str = f"{r['spearman_depth']:+.4f}" if not np.isnan(r['spearman_depth']) else "NaN"
        
        md_lines.append(
            f"| {bold}{r['policy']}{bold} | "
            f"{bold}{rho_j_str}{bold} | "
            f"{rho_rgb_str} | "
            f"{rho_d_str} | "
            f"{r['ndcg_20pct']:.4f} | "
            f"{r['overlap_20pct']:.1%} | "
            f"{bold}{r['ose_20pct']:.4f}{bold} | "
            f"{r['regret_20pct_abs']:+.6f} |"
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
