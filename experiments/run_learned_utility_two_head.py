#!/usr/bin/env python3
"""Two-Head Learned Utility Model with Feature Ablation and Pairwise Ranking Loss (Claims A & B, Points 16–19).

Formulation:
    ΔQ_hat_i = f_Q(s_i)
    ΔT_hat_i = f_T(s_i)
    U_hat_i = ΔQ_hat_i / (ΔT_hat_i + ε)

Feature Ablations (Point 16):
    V0: Error only (rgb_error, depth_error)
    V1: + visibility
    V2: + influence mass
    V3: + temporal drift
    V4: + uncertainty
    V5: + gradient norm
    V6: + projected area
    V7: Full (+ age, update frequency)

Model Architectures (Points 17–19):
    1. Linear Two-Head
    2. MLP-Small Two-Head (32 -> 1)
    3. MLP-Medium Two-Head (64 -> 32 -> 1)
    4. Ranking Two-Head MLP (Pairwise Ranking Loss: log(1 + exp(-(U_i - U_j))))

Outputs:
    - results/learned_utility/two_head_comparison.json
    - results/learned_utility/feature_ablation_report.md
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
            nn.Softplus(),  # Execution cost must strictly be positive
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
    epochs: int = 150,
    lr: float = 0.01,
) -> nn.Module:
    """Train two-head model using decoupled regression loss."""
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
    epochs: int = 150,
    lr: float = 0.01,
) -> nn.Module:
    """Train model directly with Pairwise Ranking Loss (Point 19)."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X_train)
    
    # Generate pairwise comparisons: pair (i, j) where U_i > U_j
    pairs_i = []
    pairs_j = []
    y_np = y_oracle_train.cpu().numpy()
    
    for i in range(n):
        for j in range(n):
            if y_np[i] > y_np[j] + 1e-4:
                pairs_i.append(i)
                pairs_j.append(j)
                
    if len(pairs_i) == 0:
        return model
        
    pairs_i = torch.tensor(pairs_i, dtype=torch.long, device=X_train.device)
    pairs_j = torch.tensor(pairs_j, dtype=torch.long, device=X_train.device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, _, pred_u = model(X_train)
        
        diff = pred_u[pairs_i] - pred_u[pairs_j]
        # Pairwise logistic ranking loss: log(1 + exp(-diff))
        loss = torch.log1p(torch.exp(-diff.clamp(-15.0, 15.0))).mean()
        loss.backward()
        optimizer.step()
        
    return model


def evaluate_utility_ranking(
    pred_u: np.ndarray,
    oracle_u: np.ndarray,
    delta_q: np.ndarray,
) -> Dict[str, float]:
    """Compute Spearman rho, Overlap@K, Realized Gain, and Regret."""
    n = len(pred_u)
    rho, p_val = spearmanr(pred_u, oracle_u)
    
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
    gain_ratio_20 = float(gain_pred_20 / (gain_ora_20 + 1e-8)) if gain_ora_20 > 0 else 1.0
    regret_20 = max(0.0, 1.0 - gain_ratio_20)
    
    return {
        'spearman_rho': float(rho) if not np.isnan(rho) else 0.0,
        'overlap_10pct': float(ov10),
        'overlap_20pct': float(ov20),
        'gain_ratio_20pct': float(gain_ratio_20),
        'regret_20pct': float(regret_20),
    }


def main():
    print("=" * 85)
    print("      STEP 7: TWO-HEAD LEARNED UTILITY MODEL & FEATURE ABLATION (POINTS 16–19)")
    print("=" * 85)
    
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'oracle_dataset', 'oracle_dataset.json'
    )
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Oracle dataset not found at {data_path}")
        
    with open(data_path, 'r') as f:
        rows = json.load(f)
        
    visible = [r for r in rows if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    print(f"Loaded {len(visible)} visible samples from Oracle Dataset.\n")
    
    if len(visible) < 15:
        print("Insufficient samples for training and ablation.")
        return
        
    # Feature matrix extraction
    feature_keys = [
        'rgb_error',          # V0
        'depth_error',        # V0
        'visibility',         # V1
        'influence_mass',     # V2
        'temporal_drift',     # V3
        'uncertainty',        # V4
        'gradient_norm',      # V5
        'projected_area',     # V6
        'age',                # V7
        'update_frequency',   # V7
    ]
    
    X_full = []
    y_q = []
    y_t = []
    y_oracle = []
    
    for r in visible:
        f = r.get('features', {})
        vec = [
            float(f.get('rgb_error', 0.0)),
            float(f.get('depth_error', 0.0)),
            float(f.get('visibility', 0.0)),
            float(f.get('influence_mass', 1.0)),
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
        
    X_mat = torch.tensor(X_full, dtype=torch.float32)
    y_q_vec = torch.tensor(y_q, dtype=torch.float32)
    y_t_vec = torch.tensor(y_t, dtype=torch.float32)
    y_ora_vec = torch.tensor(y_oracle, dtype=torch.float32)
    y_ora_arr = np.array(y_oracle)
    y_q_arr = np.array(y_q)
    
    # Train / Test split (70% train, 30% test)
    torch.manual_seed(42)
    np.random.seed(42)
    n_samples = len(X_mat)
    perm = torch.randperm(n_samples)
    n_train = int(0.70 * n_samples)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    
    # Normalize features
    mean = X_mat[train_idx].mean(dim=0, keepdim=True)
    std = X_mat[train_idx].std(dim=0, keepdim=True) + 1e-6
    X_norm = (X_mat - mean) / std
    
    # Feature Ablations (Point 16)
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
    
    ablation_results = []
    
    print(">> Evaluating Feature Ablations (V0 to V7) with Two-Head MLP...")
    for v_name, feat_indices in ablation_subsets.items():
        X_sub = X_norm[:, feat_indices]
        model = TwoHeadMLP(in_features=len(feat_indices), hidden_dim=64)
        train_regression_model(
            model,
            X_sub[train_idx],
            y_q_vec[train_idx],
            y_t_vec[train_idx],
            epochs=150,
            lr=0.01,
        )
        
        with torch.no_grad():
            _, _, pred_u_test = model(X_sub[test_idx])
            metrics = evaluate_utility_ranking(
                pred_u_test.cpu().numpy(),
                y_ora_arr[test_idx.cpu().numpy()],
                y_q_arr[test_idx.cpu().numpy()],
            )
            metrics['version'] = v_name
            metrics['n_features'] = len(feat_indices)
            ablation_results.append(metrics)
            print(f"   [{v_name:<22}] ρ={metrics['spearman_rho']:+.4f} | Ov@10%={metrics['overlap_10pct']:5.1%} | Ov@20%={metrics['overlap_20pct']:5.1%} | Gain@20%={metrics['gain_ratio_20pct']:.4f}")
            
    # Architecture Comparison (Point 17 & 19)
    print("\n>> Evaluating Architectures & Loss Formulations on V7 Full State...")
    arch_results = []
    
    # 1. Linear Two-Head
    linear_model = LinearTwoHead(in_features=10)
    train_regression_model(linear_model, X_norm[train_idx], y_q_vec[train_idx], y_t_vec[train_idx])
    with torch.no_grad():
        _, _, pred_u = linear_model(X_norm[test_idx])
        m_lin = evaluate_utility_ranking(pred_u.cpu().numpy(), y_ora_arr[test_idx.cpu().numpy()], y_q_arr[test_idx.cpu().numpy()])
        m_lin['architecture'] = 'Linear Two-Head'
        arch_results.append(m_lin)
        
    # 2. MLP-Small Two-Head
    mlp_small = TwoHeadMLP(in_features=10, hidden_dim=32)
    train_regression_model(mlp_small, X_norm[train_idx], y_q_vec[train_idx], y_t_vec[train_idx])
    with torch.no_grad():
        _, _, pred_u = mlp_small(X_norm[test_idx])
        m_sml = evaluate_utility_ranking(pred_u.cpu().numpy(), y_ora_arr[test_idx.cpu().numpy()], y_q_arr[test_idx.cpu().numpy()])
        m_sml['architecture'] = 'MLP-Small (32)'
        arch_results.append(m_sml)
        
    # 3. MLP-Medium Two-Head (Regression)
    mlp_med = TwoHeadMLP(in_features=10, hidden_dim=64)
    train_regression_model(mlp_med, X_norm[train_idx], y_q_vec[train_idx], y_t_vec[train_idx])
    with torch.no_grad():
        _, _, pred_u = mlp_med(X_norm[test_idx])
        m_med = evaluate_utility_ranking(pred_u.cpu().numpy(), y_ora_arr[test_idx.cpu().numpy()], y_q_arr[test_idx.cpu().numpy()])
        m_med['architecture'] = 'MLP-Medium (64, Regression)'
        arch_results.append(m_med)
        
    # 4. Ranking Two-Head MLP (Pairwise Ranking Loss, Point 19)
    mlp_rank = TwoHeadMLP(in_features=10, hidden_dim=64)
    train_ranking_model(mlp_rank, X_norm[train_idx], y_ora_vec[train_idx])
    with torch.no_grad():
        _, _, pred_u = mlp_rank(X_norm[test_idx])
        m_rnk = evaluate_utility_ranking(pred_u.cpu().numpy(), y_ora_arr[test_idx.cpu().numpy()], y_q_arr[test_idx.cpu().numpy()])
        m_rnk['architecture'] = 'Two-Head Ranking MLP (Pairwise Loss)'
        arch_results.append(m_rnk)
        
    for a in arch_results:
        print(f"   [{a['architecture']:<34}] ρ={a['spearman_rho']:+.4f} | Ov@10%={a['overlap_10pct']:5.1%} | Ov@20%={a['overlap_20pct']:5.1%} | Gain@20%={a['gain_ratio_20pct']:.4f}")
        
    # Save Report
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'learned_utility')
    os.makedirs(save_dir, exist_ok=True)
    report_file = os.path.join(save_dir, 'feature_ablation_report.md')
    
    lines = []
    lines.append("# Two-Head Learned Utility Model & Feature Ablation Report")
    lines.append("")
    lines.append("## 1. Feature Ablation Study (V0 to V7)")
    lines.append("")
    lines.append("| Feature Version | Inputs | Spearman $\\rho$ ↑ | Overlap@10% ↑ | Overlap@20% ↑ | Gain Ratio@20% ↑ | Regret@20% ↓ |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in ablation_results:
        lines.append(f"| **{r['version']}** | {r['n_features']} | {r['spearman_rho']:+.4f} | {r['overlap_10pct']:.1%} | {r['overlap_20pct']:.1%} | {r['gain_ratio_20pct']:.4f} | {r['regret_20pct']:.4f} |")
    lines.append("")
    
    lines.append("## 2. Architecture & Loss Formulation Comparison")
    lines.append("")
    lines.append("| Architecture | Loss Function | Spearman $\\rho$ ↑ | Overlap@10% ↑ | Overlap@20% ↑ | Gain Ratio@20% ↑ | Regret@20% ↓ |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for a in arch_results:
        loss_desc = "Pairwise Logistic" if "Ranking" in a['architecture'] else "Decoupled Smooth-L1"
        lines.append(f"| **{a['architecture']}** | {loss_desc} | {a['spearman_rho']:+.4f} | {a['overlap_10pct']:.1%} | {a['overlap_20pct']:.1%} | {a['gain_ratio_20pct']:.4f} | {a['regret_20pct']:.4f} |")
    lines.append("")
    
    with open(report_file, 'w') as f:
        f.write("\n".join(lines))
        
    json_path = os.path.join(save_dir, 'two_head_comparison.json')
    with open(json_path, 'w') as f:
        json.dump({'ablation': ablation_results, 'architectures': arch_results}, f, indent=2)
        
    print(f"\n[Artifacts] Successfully exported:")
    print(f"  - {report_file}")
    print(f"  - {json_path}")


if __name__ == '__main__':
    main()
