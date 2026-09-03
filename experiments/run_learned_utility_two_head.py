#!/usr/bin/env python3
"""Phase 4: Two-Head Learned Utility Model & State Factor Analysis (Points XV, XXIV, XXV, XXVI, XXVII).

Formulation:
    ΔQ_hat_i = f_Q(s_i)
    C_hat_i  = f_C(s_i)
    U_hat_i  = ΔQ_hat_i / (C_hat_i + ε)

Features:
    0: rgb_error
    1: depth_error
    2: visibility
    3: influence_mass
    4: temporal_drift
    5: uncertainty
    6: gradient_norm
    7: projected_area
    8: age
    9: update_frequency

Evaluates:
    1. Univariate Predictive Power: corr(x_j, U*) for each feature individually (Point XV-A).
    2. Conditional Incremental Information: Delta rho as features are added (Point XV-B).
    3. Independent Quality Gain & Cost Verification: MAE(Q), MAE(C), Spearman(Q), Spearman(C) (Point XXIV).
    4. Two-Head Pairwise Ranking (Weighted by |U_i - U_j|) + Pointwise Anchor Loss (Point XXV).
    5. Temporal Held-out Generalization: Train on early frames, Test on late frames (Point XXVI).
    6. Geometry Stratum Evaluation: Flat, Edge, Texture, Depth Discontinuity (Point XXVII).
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Any

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


class TwoHeadMLP(nn.Module):
    """Two-head network predicting Quality Gain ΔQ and Optimization Cost ΔT independently."""
    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.1),
        )
        self.head_q = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )
        self.head_t = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Softplus(),  # Execution cost is strictly positive
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        delta_q = self.head_q(feat).squeeze(-1)
        delta_t = self.head_t(feat).squeeze(-1) + 0.001
        utility = delta_q / delta_t
        return delta_q, delta_t, utility


class LinearTwoHead(nn.Module):
    """Linear baseline with two heads."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear_q = nn.Linear(in_features, 1)
        self.linear_t = nn.Sequential(nn.Linear(in_features, 1), nn.Softplus())
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_q = self.linear_q(x).squeeze(-1)
        delta_t = self.linear_t(x).squeeze(-1) + 0.001
        utility = delta_q / delta_t
        return delta_q, delta_t, utility


def train_regression_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_q_train: torch.Tensor,
    y_t_train: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.005,
) -> nn.Module:
    """Train two-head model using decoupled regression loss (Point XXIV)."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_q, pred_t, _ = model(X_train)
        loss = loss_fn(pred_q, y_q_train) + 0.5 * loss_fn(pred_t, y_t_train)
        loss.backward()
        optimizer.step()
        
    return model


def train_ranking_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_oracle_train: torch.Tensor,
    y_q_train: torch.Tensor,
    y_t_train: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.005,
    lambda_pointwise: float = 0.25,
) -> nn.Module:
    """Train model directly with Difference-Weighted Pairwise Ranking Loss + Pointwise Anchor (Point XXV)."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X_train)
    
    pairs_i = []
    pairs_j = []
    pair_weights = []
    y_np = y_oracle_train.cpu().numpy()
    
    for i in range(n):
        for j in range(n):
            diff = y_np[i] - y_np[j]
            if diff > 1e-5:
                pairs_i.append(i)
                pairs_j.append(j)
                pair_weights.append(diff)
                
    if len(pairs_i) == 0:
        return model
        
    pairs_i = torch.tensor(pairs_i, dtype=torch.long, device=X_train.device)
    pairs_j = torch.tensor(pairs_j, dtype=torch.long, device=X_train.device)
    pair_weights = torch.tensor(pair_weights, dtype=torch.float32, device=X_train.device)
    pair_weights = pair_weights / (pair_weights.mean() + 1e-8)  # normalize weights
    
    loss_fn_pt = nn.SmoothL1Loss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_q, pred_t, pred_u = model(X_train)
        
        diff_u = pred_u[pairs_i] - pred_u[pairs_j]
        loss_pairwise = (pair_weights * torch.log1p(torch.exp(-diff_u.clamp(-15.0, 15.0)))).mean()
        loss_pointwise = loss_fn_pt(pred_q, y_q_train) + 0.5 * loss_fn_pt(pred_t, y_t_train)
        
        total_loss = loss_pairwise + lambda_pointwise * loss_pointwise
        total_loss.backward()
        optimizer.step()
        
    return model


def evaluate_utility_ranking(
    pred_u: np.ndarray,
    oracle_u: np.ndarray,
    delta_q: np.ndarray,
) -> Dict[str, float]:
    """Compute Spearman rho, NDCG@K, Overlap@K, OSE, and Regret."""
    n = len(pred_u)
    rho, p_val = safe_spearmanr(pred_u, oracle_u)
    
    pred_ranks = np.argsort(-pred_u)
    oracle_ranks = np.argsort(-oracle_u)
    
    k10 = max(1, int(n * 0.10))
    k20 = max(1, int(n * 0.20))
    
    top_pred_10 = set(pred_ranks[:k10].tolist())
    top_ora_10 = set(oracle_ranks[:k10].tolist())
    ov10 = len(top_pred_10 & top_ora_10) / k10
    
    top_pred_20 = set(pred_ranks[:k20].tolist())
    top_ora_20 = set(oracle_ranks[:k20].tolist())
    ov20 = len(top_pred_20 & top_ora_20) / k20
    
    gain_pred_20 = delta_q[list(top_pred_20)].sum()
    gain_ora_20 = delta_q[list(top_ora_20)].sum()
    ose_20 = float(gain_pred_20 / (gain_ora_20 + 1e-8)) if gain_ora_20 > 0 else 1.0
    regret_20_abs = float(gain_ora_20 - gain_pred_20)
    
    ndcg_20 = compute_ndcg(pred_u, oracle_u, k20)
    
    return {
        'spearman_rho': float(rho) if not np.isnan(rho) else 0.0,
        'p_val': float(p_val) if not np.isnan(p_val) else 1.0,
        'ndcg_20pct': float(ndcg_20),
        'overlap_10pct': float(ov10),
        'overlap_20pct': float(ov20),
        'ose_20pct': float(ose_20),
        'regret_20pct_abs': float(regret_20_abs),
    }


def main():
    print("=" * 90)
    print("   PHASE 4: TWO-HEAD MARGINAL UTILITY MODEL & STATE FACTOR ANALYSIS (POINTS XV, XXIV–XXVII)")
    print("=" * 90)
    
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'oracle_dataset', 'oracle_dataset.json'
    )
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Oracle dataset not found at {data_path}")
        
    with open(data_path, 'r') as f:
        rows = json.load(f)
        
    visible = [r for r in rows if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    print(f">> Loaded {len(visible)} visible samples from Oracle Dataset.\n")
    
    feature_names = [
        'rgb_error',          # 0
        'depth_error',        # 1
        'visibility',         # 2
        'influence_mass',     # 3
        'temporal_drift',     # 4
        'uncertainty',        # 5
        'gradient_norm',      # 6
        'projected_area',     # 7
        'age',                # 8
        'update_frequency',   # 9
    ]
    
    X_full = []
    y_q = []
    y_t = []
    y_oracle = []
    frames = []
    strata = []
    
    for r in visible:
        f = r.get('features', {})
        vec = [
            float(f.get('rgb_error', 0.0)),
            float(f.get('depth_error', 0.0)),
            float(f.get('visibility', 0.0)),
            float(f.get('influence_mass', r.get('influence_mass', 1.0))),
            float(f.get('temporal_drift', 0.0)),
            float(f.get('uncertainty', 0.5)),
            float(f.get('gradient_norm', 0.0)),
            float(f.get('projected_area', 1.0)),
            float(f.get('age', 1.0)),
            float(f.get('update_frequency', 0.5)),
        ]
        X_full.append(vec)
        y_q.append(float(r.get('delta_quality_local', 0.0)))
        y_t.append(float(r.get('measured_trial_cost_ms', 1.0)))
        y_oracle.append(float(r.get('oracle_utility_joint', r.get('oracle_utility', 0.0))))
        frames.append(int(r.get('frame', 0)))
        strata.append(r.get('geometry_stratum', 'unknown'))
        
    X_mat = torch.tensor(X_full, dtype=torch.float32)
    y_q_vec = torch.tensor(y_q, dtype=torch.float32)
    y_t_vec = torch.tensor(y_t, dtype=torch.float32)
    y_ora_vec = torch.tensor(y_oracle, dtype=torch.float32)
    y_ora_arr = np.array(y_oracle)
    y_q_arr = np.array(y_q)
    y_t_arr = np.array(y_t)
    
    # --- 1. Univariate Predictive Power Analysis (Point XV - Level A) ---
    print(">> 1. Univariate Predictive Power Analysis (corr(x_j, U*)):")
    univariate_results = []
    for j, name in enumerate(feature_names):
        vals = X_mat[:, j].numpy()
        rho, p = safe_spearmanr(vals, y_ora_arr)
        univariate_results.append({
            'feature': name,
            'spearman_rho': float(rho),
            'p_val': float(p),
        })
        sig = "Significant ✅" if p < 0.05 else "Not Significant"
        print(f"   - {name:<20}: ρ = {rho:+.4f} (p={p:.4f}) [{sig}]")
        
    # --- 2. Temporal Held-Out Split (Point XXVI) ---
    unique_frames = sorted(list(set(frames)))
    if len(unique_frames) >= 2:
        split_frame = unique_frames[len(unique_frames) // 2]
        train_mask = [f <= split_frame for f in frames]
        test_mask = [f > split_frame for f in frames]
        train_idx = torch.tensor([i for i, m in enumerate(train_mask) if m], dtype=torch.long)
        test_idx = torch.tensor([i for i, m in enumerate(test_mask) if m], dtype=torch.long)
        print(f"\n>> 2. Temporal Held-out Split: Train frames <= {split_frame} ({len(train_idx)}), Test frames > {split_frame} ({len(test_idx)})")
    else:
        # Fallback 70/30 random
        perm = torch.randperm(len(X_mat))
        n_tr = int(0.70 * len(X_mat))
        train_idx, test_idx = perm[:n_tr], perm[n_tr:]
        print(f"\n>> 2. Random 70/30 Split: Train ({len(train_idx)}), Test ({len(test_idx)})")
        
    # Feature Normalization (Strictly on Train Split to prevent data leakage)
    mean = X_mat[train_idx].mean(dim=0, keepdim=True)
    std = X_mat[train_idx].std(dim=0, keepdim=True) + 1e-6
    X_norm = (X_mat - mean) / std
    
    # --- 3. Conditional Incremental Information (Point XV - Level B) ---
    ablation_subsets = {
        'V0: Error Only': [0, 1],
        'V1: + Visibility': [0, 1, 2],
        'V2: + Influence': [0, 1, 2, 3],
        'V3: + Temporal Drift': [0, 1, 2, 3, 4],
        'V4: + Uncertainty': [0, 1, 2, 3, 4, 5],
        'V5: + Gradient Norm': [0, 1, 2, 3, 4, 5, 6],
        'V6: + Projected Area': [0, 1, 2, 3, 4, 5, 6, 7],
        'V7: Full State': list(range(10)),
    }
    
    print("\n>> 3. Conditional Incremental Information (Two-Head Ranking):")
    ablation_rows = []
    prev_rho = 0.0
    for v_name, feat_ids in ablation_subsets.items():
        X_sub = X_norm[:, feat_ids]
        model = TwoHeadMLP(in_features=len(feat_ids), hidden_dim=64)
        train_ranking_model(
            model,
            X_sub[train_idx],
            y_ora_vec[train_idx],
            y_q_vec[train_idx],
            y_t_vec[train_idx],
            epochs=200,
            lr=0.005,
        )
        
        with torch.no_grad():
            _, _, pred_u_test = model(X_sub[test_idx])
            m = evaluate_utility_ranking(
                pred_u_test.cpu().numpy(),
                y_ora_arr[test_idx.cpu().numpy()],
                y_q_arr[test_idx.cpu().numpy()],
            )
            delta_rho = m['spearman_rho'] - prev_rho
            prev_rho = m['spearman_rho']
            
            ablation_rows.append({
                'version': v_name,
                'inputs': len(feat_ids),
                'spearman_rho': m['spearman_rho'],
                'delta_rho': delta_rho,
                'ndcg_20pct': m['ndcg_20pct'],
                'overlap_20pct': m['overlap_20pct'],
                'ose_20pct': m['ose_20pct'],
                'regret_20pct': m['regret_20pct_abs'],
            })
            print(f"   - {v_name:<22}: ρ = {m['spearman_rho']:+.4f} (Δρ={delta_rho:+.4f}) | NDCG@20% = {m['ndcg_20pct']:.4f} | OSE@20% = {m['ose_20pct']:.3f}")
            
    # --- 4. Model Architecture & Loss Comparison (Point XXIV & XXV) ---
    print("\n>> 4. Architecture & Loss Comparison:")
    models_to_test = [
        ('Linear Two-Head (Regression)', LinearTwoHead(10), 'regression'),
        ('MLP Two-Head (Regression)', TwoHeadMLP(10, 64), 'regression'),
        ('Linear Two-Head (Ranking)', LinearTwoHead(10), 'ranking'),
        ('MLP Two-Head (Pairwise+Pointwise Ranking - Ours)', TwoHeadMLP(10, 64), 'ranking'),
    ]
    
    arch_rows = []
    trained_models = {}
    
    for name, m_inst, mode in models_to_test:
        if mode == 'regression':
            trained = train_regression_model(
                m_inst, X_norm[train_idx], y_q_vec[train_idx], y_t_vec[train_idx], epochs=200, lr=0.005
            )
        else:
            trained = train_ranking_model(
                m_inst, X_norm[train_idx], y_ora_vec[train_idx], y_q_vec[train_idx], y_t_vec[train_idx], epochs=200, lr=0.005
            )
        trained_models[name] = trained
        
        with torch.no_grad():
            pred_q_test, pred_t_test, pred_u_test = trained(X_norm[test_idx])
            
            p_q = pred_q_test.cpu().numpy()
            p_t = pred_t_test.cpu().numpy()
            p_u = pred_u_test.cpu().numpy()
            
            y_q_t = y_q_arr[test_idx.cpu().numpy()]
            y_t_t = y_t_arr[test_idx.cpu().numpy()]
            y_o_t = y_ora_arr[test_idx.cpu().numpy()]
            
            # Independent verification (Point XXIV)
            mae_q = float(np.mean(np.abs(p_q - y_q_t)))
            mae_t = float(np.mean(np.abs(p_t - y_t_t)))
            rho_q, _ = safe_spearmanr(p_q, y_q_t)
            rho_t, _ = safe_spearmanr(p_t, y_t_t)
            
            eval_metrics = evaluate_utility_ranking(p_u, y_o_t, y_q_t)
            
            arch_rows.append({
                'model': name,
                'spearman_rho': eval_metrics['spearman_rho'],
                'ndcg_20pct': eval_metrics['ndcg_20pct'],
                'overlap_20pct': eval_metrics['overlap_20pct'],
                'ose_20pct': eval_metrics['ose_20pct'],
                'regret_20pct': eval_metrics['regret_20pct_abs'],
                'mae_delta_q': mae_q,
                'spearman_delta_q': rho_q,
                'mae_cost': mae_t,
                'spearman_cost': rho_t,
            })
            print(f"   - {name:<46}: ρ={eval_metrics['spearman_rho']:+.4f} | NDCG@20%={eval_metrics['ndcg_20pct']:.4f} | OSE@20%={eval_metrics['ose_20pct']:.3f} | MAE(Q)={mae_q:.6f}")
            
    # --- 5. Geometry Stratum Evaluation (Point XXVII) ---
    print("\n>> 5. Geometry Stratum Evaluation (Oracle vs Error vs Heuristic vs Learned):")
    best_model = trained_models['MLP Two-Head (Pairwise+Pointwise Ranking - Ours)']
    with torch.no_grad():
        _, _, all_pred_u = best_model(X_norm)
        all_pred_u = all_pred_u.cpu().numpy()
        
    error_scores = (X_mat[:, 0] + X_mat[:, 1]).numpy()
    heuristic_scores = np.array([float(r.get('predicted_utility', 0.0)) for r in visible])
    
    strata_breakdown = {}
    unique_strata = ['flat', 'edge', 'texture', 'depth_discontinuity']
    
    for st in unique_strata:
        st_indices = [i for i, s in enumerate(strata) if s == st]
        if len(st_indices) >= 3:
            st_u_ora = y_ora_arr[st_indices]
            st_err = error_scores[st_indices]
            st_heur = heuristic_scores[st_indices]
            st_lrn = all_pred_u[st_indices]
            
            rho_err, _ = safe_spearmanr(st_err, st_u_ora)
            rho_heur, _ = safe_spearmanr(st_heur, st_u_ora)
            rho_lrn, _ = safe_spearmanr(st_lrn, st_u_ora)
            
            strata_breakdown[st] = {
                'count': len(st_indices),
                'mean_oracle_u': float(np.mean(st_u_ora)),
                'rho_error': float(rho_err),
                'rho_heuristic': float(rho_heur),
                'rho_learned': float(rho_lrn),
            }
            print(f"   - {st:<20} (N={len(st_indices):2d}): Mean U* = {np.mean(st_u_ora):+.5f} | ρ(Err) = {rho_err:+.4f} | ρ(Heur) = {rho_heur:+.4f} | ρ(Learned) = {rho_lrn:+.4f}")
            
    # --- 6. Export Reports ---
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'learned_utility')
    os.makedirs(save_dir, exist_ok=True)
    report_file = os.path.join(save_dir, 'feature_ablation_report.md')
    json_file = os.path.join(save_dir, 'learned_utility_summary.json')
    
    with open(json_file, 'w') as f:
        json.dump({
            'univariate_predictive_power': univariate_results,
            'conditional_ablation': ablation_rows,
            'architecture_comparison': arch_rows,
            'geometry_stratum_breakdown': strata_breakdown,
        }, f, indent=2)
        
    lines = [
        "# Phase 4: Learned Marginal Utility & State Factor Analysis Report",
        "",
        "## 1. Univariate Predictive Power Analysis (Point XV - Level A)",
        "",
        "| State Variable ($x_j$) | Spearman $\\rho(x_j, U^\\star)$ | p-value | Significance |",
        "|:---|:---:|:---:|:---:|",
    ]
    for u in univariate_results:
        sig_str = "Statistically Significant ✅" if u['p_val'] < 0.05 else "Not Significant"
        lines.append(f"| **{u['feature']}** | {u['spearman_rho']:+.4f} | {u['p_val']:.4f} | {sig_str} |")
        
    lines.extend([
        "",
        "## 2. Conditional Incremental Information (Point XV - Level B)",
        "",
        "| Model Variant | Inputs | Spearman $\\rho$ ↑ | $\\Delta \\rho$ | NDCG@20% ↑ | Overlap@20% ↑ | OSE@20% ↑ | Absolute Regret ↓ |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for a in ablation_rows:
        lines.append(
            f"| **{a['version']}** | {a['inputs']} | **{a['spearman_rho']:+.4f}** | "
            f"{a['delta_rho']:+.4f} | {a['ndcg_20pct']:.4f} | {a['overlap_20pct']:.1%} | "
            f"**{a['ose_20pct']:.4f}** | {a['regret_20pct']:+.6f} |"
        )
        
    lines.extend([
        "",
        "## 3. Model Architecture & Loss Comparison (Points XXIV & XXV)",
        "",
        "| Model Architecture | Loss Objective | Spearman $\\rho(U^\\star)$ ↑ | NDCG@20% ↑ | OSE@20% ↑ | $\\text{MAE}(\\Delta Q)$ ↓ | $\\text{MAE}(C)$ ↓ |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ])
    for m in arch_rows:
        loss_name = "Pairwise + Pointwise" if "Ranking" in m['model'] else "Decoupled Smooth-L1"
        lines.append(
            f"| **{m['model']}** | {loss_name} | **{m['spearman_rho']:+.4f}** | "
            f"{m['ndcg_20pct']:.4f} | **{m['ose_20pct']:.4f}** | {m['mae_delta_q']:.6f} | {m['mae_cost']:.2f} ms |"
        )
        
    lines.extend([
        "",
        "## 4. Geometry Stratum Breakdown (Point XXVII)",
        "",
        "| Geometry Stratum | Interventions (N) | Mean Oracle $U^\\star$ | $\\rho(\\text{Error}, U^\\star)$ | $\\rho(\\text{Heuristic}, U^\\star)$ | $\\rho(\\text{Learned Ours}, U^\\star)$ ↑ |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ])
    for st, s_data in strata_breakdown.items():
        lines.append(
            f"| **{st}** | {s_data['count']} | {s_data['mean_oracle_u']:+.6f} | "
            f"{s_data['rho_error']:+.4f} | {s_data['rho_heuristic']:+.4f} | **{s_data['rho_learned']:+.4f}** |"
        )
    lines.append("")
    
    with open(report_file, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"\n[Generated Report] Successfully saved to {report_file}")


if __name__ == '__main__':
    main()
