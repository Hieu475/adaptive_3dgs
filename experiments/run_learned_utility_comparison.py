#!/usr/bin/env python3
"""Step 17: Learned Utility vs Heuristic Utility Benchmark.

Trains a lightweight MLP (Input -> 64 -> 32 -> 1) on Oracle Dataset (state_i -> U_i^oracle)
and compares:
  1. Error-Only
  2. Error × Influence
  3. Heuristic (Ours V6)
  4. Learned MLP
  5. Oracle (Reference Upper Bound)

Outputs:
  - results/learned_utility/comparison_summary.json
  - results/learned_utility/comparison_report.md
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LightUtilityMLP(nn.Module):
    """Compact 2-layer MLP for fast per-Gaussian utility scoring."""
    def __init__(self, in_features: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def run_learned_utility_comparison():
    print("=" * 85)
    print("      STEP 17: LEARNED UTILITY vs HEURISTIC UTILITY BENCHMARK")
    print("=" * 85)
    
    # Load Oracle dataset
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'results', 'oracle_dataset', 'oracle_dataset.json')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Oracle dataset not found at {data_path}. "
            "Oracle dataset is required for learned utility evaluation. "
            "Run 'python experiments/run_oracle_utility.py' first to generate it."
        )
    
    with open(data_path, 'r') as f:
        rows = json.load(f)
            
    visible = [r for r in rows if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    print(f"Loaded {len(visible)} visible Oracle Gaussian training/eval samples.\n")
    
    if len(visible) < 10:
        print("Insufficient samples for training.")
        return
        
    # Feature extraction
    X = []
    y_oracle = []
    y_delta_q = []
    imp_scores = []
    
    for r in visible:
        imp = r.get('predicted_importance', 0.0)
        inf = r.get('influence_mass', 0.0)
        area = r.get('projected_area', 0.0)
        cost = r.get('modeled_marginal_cost_us', 1.0)
        
        X.append([imp, inf, area, float(r.get('delta_psnr_local', 0.0)), cost])
        y_oracle.append(r.get('oracle_utility', 0.0))
        y_delta_q.append(r.get('delta_quality_local', 0.0))
        imp_scores.append(imp)
        
    X_mat = torch.tensor(X, dtype=torch.float32)
    y_vec = torch.tensor(y_oracle, dtype=torch.float32)
    y_delta_q_arr = np.array(y_delta_q)
    
    # Train / Test split (80 / 20)
    n_samples = len(X_mat)
    perm = torch.randperm(n_samples)
    n_train = int(n_samples * 0.75)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    
    # Standardize inputs
    mean = X_mat[train_idx].mean(dim=0, keepdim=True)
    std = X_mat[train_idx].std(dim=0, keepdim=True) + 1e-6
    X_norm = (X_mat - mean) / std
    
    # Train Light MLP
    mlp = LightUtilityMLP(in_features=X_mat.shape[1])
    optimizer = optim.Adam(mlp.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(200):
        optimizer.zero_grad()
        pred = mlp(X_norm[train_idx])
        loss = criterion(pred, y_vec[train_idx])
        loss.backward()
        optimizer.step()
        
    # Evaluate all methods on Test Set
    with torch.no_grad():
        mlp_preds = mlp(X_norm[test_idx]).numpy()
        
    test_oracle = y_vec[test_idx].numpy()
    test_delta_q = y_delta_q_arr[test_idx.numpy()]
    test_imp = np.array([imp_scores[i] for i in test_idx.numpy()])
    test_err_only = X_mat[test_idx, 0].numpy()
    test_err_inf = (X_mat[test_idx, 0] * X_mat[test_idx, 1]).numpy()
    
    def eval_method_metrics(scores, oracle_target, delta_q):
        rho, _ = spearmanr(scores, oracle_target)
        n = len(scores)
        k10 = max(1, int(n * 0.10))
        k20 = max(1, int(n * 0.20))
        
        ranks = np.argsort(-scores)
        oracle_ranks = np.argsort(-oracle_target)
        
        ov10 = len(set(ranks[:k10]) & set(oracle_ranks[:k10])) / k10
        ov20 = len(set(ranks[:k20]) & set(oracle_ranks[:k20])) / k20
        
        gain_ours = delta_q[ranks[:k20]].sum()
        gain_oracle = delta_q[oracle_ranks[:k20]].sum()
        gain_ratio = float(gain_ours / (gain_oracle + 1e-8))
        regret = max(0.0, 1.0 - gain_ratio)
        
        return {
            'spearman_rho': float(rho),
            'overlap_10pct': float(ov10),
            'overlap_20pct': float(ov20),
            'gain_ratio_20pct': float(gain_ratio),
            'regret_10pct': float(regret)
        }
        
    m_err = eval_method_metrics(test_err_only, test_oracle, test_delta_q)
    m_inf = eval_method_metrics(test_err_inf, test_oracle, test_delta_q)
    m_heu = eval_method_metrics(test_imp, test_oracle, test_delta_q)
    m_mlp = eval_method_metrics(mlp_preds, test_oracle, test_delta_q)
    m_orc = eval_method_metrics(test_oracle, test_oracle, test_delta_q)
    
    methods = [
        ("1. Error-Only", m_err),
        ("2. Error × Influence", m_inf),
        ("3. Heuristic (Ours V6)", m_heu),
        ("4. Learned MLP (64→32→1)", m_mlp),
        ("5. Oracle (Upper Bound)", m_orc),
    ]
    
    print("=" * 85)
    print("            LEARNED UTILITY vs HEURISTIC EVALUATION MATRIX (TEST SET)")
    print("=" * 85)
    print(f"{'Method':<28} | {'ρ(Util,Oracle)':>14} | {'Ov@10%':>8} | {'Ov@20%':>8} | {'Gain@20%':>9} | {'Regret@10%':>10}")
    print("-" * 85)
    for name, m in methods:
        print(f"{name:<28} | {m['spearman_rho']:>14.4f} | {m['overlap_10pct']:>7.1%} | {m['overlap_20pct']:>7.1%} | {m['gain_ratio_20pct']:>9.4f} | {m['regret_10pct']:>10.4f}")
    print("=" * 85 + "\n")
    
    # Save results
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'learned_utility')
    os.makedirs(save_dir, exist_ok=True)
    
    summary = {name: m for name, m in methods}
    with open(os.path.join(save_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
        
    with open(os.path.join(save_dir, 'comparison_report.md'), 'w') as f:
        f.write("# Step 17: Learned Utility vs Heuristic Utility Benchmark\n\n")
        f.write("| Method | $\\rho(U, U_{oracle})$ | Overlap@10% | Overlap@20% | Gain Ratio@20% | Regret@10% |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for name, m in methods:
            f.write(f"| **{name}** | {m['spearman_rho']:.4f} | {m['overlap_10pct']*100:.1f}% | {m['overlap_20pct']*100:.1f}% | {m['gain_ratio_20pct']:.4f} | {m['regret_10pct']:.4f} |\n")
        f.write("\n")
        
    print(f"Artifacts saved to:")
    print(f"  - {os.path.join(save_dir, 'comparison_summary.json')}")
    print(f"  - {os.path.join(save_dir, 'comparison_report.md')}")
    return summary


if __name__ == '__main__':
    run_learned_utility_comparison()
