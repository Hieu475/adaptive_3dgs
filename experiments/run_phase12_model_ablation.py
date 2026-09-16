#!/usr/bin/env python3
r"""Phase 12: Controlled Model Ablation Experiment (M0 through M4).

Executes the mandatory model comparison across identical candidate pools, seeds,
budget levels, and optimizer settings:
    M0: Error Heuristic (e_rgb + e_depth)
    M1: Gradient Sensitivity (Grad-Norm ||\nabla L||)
    M2: Current TwoHeadMLP (11D Local Observable State)
    M3: Two-Stage MLP + Positive Head (11D Local + P(U* > 0))
    M4: Two-Stage MLP + Positive Head + Global Context (11D Local + 12D Global Context)

Evaluation Dimensions:
    1. Spearman Rank Correlation: \rho(score, U*)
    2. Positive Utility Detection: AUROC(P(U* > 0))
    3. Selection Quality: NDCG@20
    4. Optimal Selection Efficiency: OSE = \Delta Q_{policy} / \Delta Q_{oracle}
    5. Realized Quality Gain: \Delta Q under B = 15.0 ms
    6. Budget Efficiency: E_Q = \Delta Q / T_{opt} (gain per ms)

Outputs:
    results/phase12_paper_evidence/model_ablation/
    ├── model_ablation_results.json
    ├── checkpoints/
    │   ├── M2_seed_{seed}.pt
    │   ├── M3_seed_{seed}.pt
    │   └── M4_seed_{seed}.pt
    └── summary_ablation.md
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.phase12_protocol import (
    SEEDS,
    DEFAULT_BUDGET_MS,
    SAFETY_FACTOR,
    CANONICAL_LOCAL_FEATURES,
    CANONICAL_GLOBAL_FEATURES,
    get_phase12_output_dir,
)
from research.phase12_model import (
    TwoStageUtilityModel,
    TwoStageModelConfig,
    TwoStageModelTrainer,
    TwoStageLossConfig,
    extract_local_features_from_df,
    compute_global_features_from_df,
)
from research.utility_metrics import safe_spearmanr, compute_ndcg_at_k


def select_knapsack_subset(
    scores: np.ndarray,
    costs: np.ndarray,
    budget_ms: float = DEFAULT_BUDGET_MS,
    safety_factor: float = SAFETY_FACTOR,
    reject_nonpositive_score: bool = True,
) -> Tuple[List[int], float]:
    """Greedy knapsack selection under budgeted constraint."""
    order = np.argsort(-scores)
    selected: List[int] = []
    cur_scheduled_cost = 0.0

    for idx in order:
        sc = float(scores[idx])
        if reject_nonpositive_score and sc <= 0.0:
            continue
        c = float(costs[idx]) * safety_factor
        if cur_scheduled_cost + c <= budget_ms + 1e-7:
            selected.append(int(idx))
            cur_scheduled_cost += c

    return selected, cur_scheduled_cost


def evaluate_frame_selection(
    scores: np.ndarray,
    delta_q: np.ndarray,
    costs_ms: np.ndarray,
    oracle_u: np.ndarray,
    budget_ms: float = DEFAULT_BUDGET_MS,
    safety_factor: float = SAFETY_FACTOR,
    reject_nonpositive: bool = True,
    k: int = 20,
) -> Dict[str, float]:
    """Computes selection metrics for a single frame evaluation set."""
    N = len(scores)
    if N == 0:
        return {
            "ndcg_20": 0.0,
            "ose": 0.0,
            "realized_dq": 0.0,
            "actual_time_ms": 0.0,
            "dq_per_ms": 0.0,
            "n_selected": 0,
            "n_negative_selected": 0,
        }

    # Oracle Knapsack Selection
    oracle_selected, _ = select_knapsack_subset(
        scores=oracle_u,
        costs=costs_ms,
        budget_ms=budget_ms,
        safety_factor=safety_factor,
        reject_nonpositive_score=True,
    )
    oracle_dq = float(np.sum(delta_q[oracle_selected])) if oracle_selected else 1e-8

    # Policy Knapsack Selection
    policy_selected, _ = select_knapsack_subset(
        scores=scores,
        costs=costs_ms,
        budget_ms=budget_ms,
        safety_factor=safety_factor,
        reject_nonpositive_score=reject_nonpositive,
    )
    realized_dq = float(np.sum(delta_q[policy_selected])) if policy_selected else 0.0
    actual_t = float(np.sum(costs_ms[policy_selected])) if policy_selected else 0.0

    ose = float(realized_dq / oracle_dq) if oracle_dq > 1e-7 else 1.0
    dq_per_ms = float(realized_dq / actual_t) if actual_t > 0.0 else 0.0
    ndcg = compute_ndcg_at_k(scores, oracle_u, k=min(k, N))

    # Number of negative utility updates accidentally selected
    n_neg = int(np.sum(oracle_u[policy_selected] <= 0.0)) if policy_selected else 0

    return {
        "ndcg_20": float(ndcg),
        "ose": float(ose),
        "realized_dq": float(realized_dq),
        "actual_time_ms": float(actual_t),
        "dq_per_ms": float(dq_per_ms),
        "n_selected": len(policy_selected),
        "n_negative_selected": n_neg,
    }


def prepare_datasets(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Enriches oracle dataset with canonical local and global features, normalized by train reference."""
    # 1. Compute global features
    glob_df = compute_global_features_from_df(df)
    df_merged = df.merge(glob_df, on=["scene", "frame", "seed"])

    train_df = df_merged[df_merged["split"] == "train"].copy()
    val_df = df_merged[df_merged["split"] == "validation"].copy()
    test_df = df_merged[df_merged["split"] == "cross_scene_test"].copy()

    # Normalization statistics strictly fit on train split (zero test leakage)
    loc_cols = CANONICAL_LOCAL_FEATURES
    glob_cols = CANONICAL_GLOBAL_FEATURES

    loc_mean = train_df[loc_cols].mean().to_numpy(dtype=np.float32)
    loc_std = train_df[loc_cols].std().to_numpy(dtype=np.float32) + 1e-6

    glob_mean = train_df[glob_cols].mean().to_numpy(dtype=np.float32)
    glob_std = train_df[glob_cols].std().to_numpy(dtype=np.float32) + 1e-6

    def format_split_data(sub_df: pd.DataFrame) -> Dict[str, Any]:
        X_loc = (sub_df[loc_cols].to_numpy(dtype=np.float32) - loc_mean) / loc_std
        X_glob = (sub_df[glob_cols].to_numpy(dtype=np.float32) - glob_mean) / glob_std
        X_loc = np.nan_to_num(X_loc, nan=0.0, posinf=1.0, neginf=0.0)
        X_glob = np.nan_to_num(X_glob, nan=0.0, posinf=1.0, neginf=0.0)

        y_q = sub_df["delta_quality"].to_numpy(dtype=np.float32)
        y_t = sub_df["modeled_marginal_cost_us"].to_numpy(dtype=np.float32)
        y_u = sub_df["oracle_utility_joint"].to_numpy(dtype=np.float32)
        # Modeled optimization kernel cost C_i^{kernel} (~0.51 ms) for scheduler knapsack
        costs_ms = sub_df["modeled_marginal_cost_us"].to_numpy(dtype=np.float32)

        return {
            "df": sub_df,
            "X_loc": torch.tensor(X_loc, dtype=torch.float32),
            "X_glob": torch.tensor(X_glob, dtype=torch.float32),
            "y_q": torch.tensor(y_q, dtype=torch.float32),
            "y_t": torch.tensor(y_t, dtype=torch.float32),
            "y_u": torch.tensor(y_u, dtype=torch.float32),
            "y_u_np": y_u,
            "delta_q_np": y_q,
            "costs_ms_np": costs_ms,
        }

    return df_merged, format_split_data(train_df), format_split_data(val_df), format_split_data(test_df)


def run_model_ablation(
    oracle_csv: Path,
    output_dir: Path,
    epochs: int = 250,
    device: str = "cuda",
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   PHASE 12: CONTROLLED MODEL ABLATION EXPERIMENT (M0 - M4)")
    print("=" * 80)

    # 1. Load and prepare dataset
    assert oracle_csv.exists(), f"Oracle dataset not found at {oracle_csv}"
    df_raw = pd.read_csv(oracle_csv)
    print(f">> Loaded {len(df_raw)} audited oracle samples from {oracle_csv}")

    df_full, train_data, val_data, test_data = prepare_datasets(df_raw)
    print(f">> Splits: Train N={len(train_data['df'])}, Val N={len(val_data['df'])}, Test N={len(test_data['df'])}")

    test_df = test_data["df"]
    y_u_test = test_data["y_u_np"]
    delta_q_test = test_data["delta_q_np"]
    costs_test = test_data["costs_ms_np"]
    y_pos_test = (y_u_test > 0.0).astype(int)

    # Candidate Heuristic Proxies for Test Set
    err_test = (test_df["feat_rgb_error"] + test_df["feat_depth_error"]).to_numpy(dtype=np.float32)
    grad_test = test_df["feat_gradient_norm"].to_numpy(dtype=np.float32)

    ablation_results: Dict[str, Dict[str, Any]] = {
        "M0_Error": {"per_seed": {}, "name": "Error Heuristic (M0)"},
        "M1_GradNorm": {"per_seed": {}, "name": "Gradient Sensitivity (M1)"},
        "M2_CurrentMLP": {"per_seed": {}, "name": "Current TwoHeadMLP (M2)"},
        "M3_PositiveHead": {"per_seed": {}, "name": "MLP + Positive Head (M3)"},
        "M4_PositiveGlobal": {"per_seed": {}, "name": "MLP + Positive + Global (M4)"},
        "Policy_OraclePositive": {"per_seed": {}, "name": "Oracle-Positive + M4 Ranking"},
        "Policy_ProbThreshold": {"per_seed": {}, "name": "Probability-Thresholded M4"},
    }

    dev = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"

    # Evaluate each seed
    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ─── M0: Error Heuristic Evaluation ───
        rho_m0, _ = safe_spearmanr(err_test, y_u_test)
        auc_m0 = float(roc_auc_score(y_pos_test, err_test)) if len(np.unique(y_pos_test)) > 1 else 0.5
        sel_m0 = evaluate_frame_selection(
            scores=err_test,
            delta_q=delta_q_test,
            costs_ms=costs_test,
            oracle_u=y_u_test,
            reject_nonpositive=False,
        )
        ablation_results["M0_Error"]["per_seed"][seed] = {
            "spearman_rho": float(rho_m0),
            "auroc": auc_m0,
            **sel_m0,
        }

        # ─── M1: Gradient Sensitivity (Grad-Norm) Evaluation ───
        rho_m1, _ = safe_spearmanr(grad_test, y_u_test)
        auc_m1 = float(roc_auc_score(y_pos_test, grad_test)) if len(np.unique(y_pos_test)) > 1 else 0.5
        sel_m1 = evaluate_frame_selection(
            scores=grad_test,
            delta_q=delta_q_test,
            costs_ms=costs_test,
            oracle_u=y_u_test,
            reject_nonpositive=False,
        )
        ablation_results["M1_GradNorm"]["per_seed"][seed] = {
            "spearman_rho": float(rho_m1),
            "auroc": auc_m1,
            **sel_m1,
        }

        # ─── M2: Current TwoHeadMLP (11D Local Only, No Positive Head) ───
        cfg_m2 = TwoStageModelConfig(use_global=False, use_positive_head=False)
        model_m2 = TwoStageUtilityModel(cfg_m2)
        trainer_m2 = TwoStageModelTrainer(model_m2, device=dev)
        trainer_m2.fit(train_data, val_data, epochs=epochs)
        trainer_m2.save_checkpoint(ckpt_dir / f"M2_seed_{seed}.pt")

        model_m2.eval()
        with torch.no_grad():
            _, _, _, u_m2 = model_m2(test_data["X_loc"].to(dev))
            u_m2_np = u_m2.cpu().numpy()

        rho_m2, _ = safe_spearmanr(u_m2_np, y_u_test)
        auc_m2 = float(roc_auc_score(y_pos_test, u_m2_np)) if len(np.unique(y_pos_test)) > 1 else 0.5
        sel_m2 = evaluate_frame_selection(
            scores=u_m2_np,
            delta_q=delta_q_test,
            costs_ms=costs_test,
            oracle_u=y_u_test,
            reject_nonpositive=True,
        )
        ablation_results["M2_CurrentMLP"]["per_seed"][seed] = {
            "spearman_rho": float(rho_m2),
            "auroc": auc_m2,
            **sel_m2,
        }

        # ─── M3: Two-Stage MLP + Positive Head (11D Local Only) ───
        cfg_m3 = TwoStageModelConfig(use_global=False, use_positive_head=True)
        model_m3 = TwoStageUtilityModel(cfg_m3)
        trainer_m3 = TwoStageModelTrainer(model_m3, device=dev)
        trainer_m3.fit(train_data, val_data, epochs=epochs)
        trainer_m3.save_checkpoint(ckpt_dir / f"M3_seed_{seed}.pt")

        model_m3.eval()
        with torch.no_grad():
            p_m3, _, _, u_m3 = model_m3(test_data["X_loc"].to(dev))
            p_m3_np = p_m3.cpu().numpy()
            u_m3_np = u_m3.cpu().numpy()

        rho_m3, _ = safe_spearmanr(u_m3_np, y_u_test)
        auc_m3 = float(roc_auc_score(y_pos_test, p_m3_np)) if len(np.unique(y_pos_test)) > 1 else 0.5
        sel_m3 = evaluate_frame_selection(
            scores=u_m3_np,
            delta_q=delta_q_test,
            costs_ms=costs_test,
            oracle_u=y_u_test,
            reject_nonpositive=True,
        )
        ablation_results["M3_PositiveHead"]["per_seed"][seed] = {
            "spearman_rho": float(rho_m3),
            "auroc": auc_m3,
            **sel_m3,
        }

        # ─── M4: Full Two-Stage MLP + Positive Head + Global Context (11D + 12D) ───
        cfg_m4 = TwoStageModelConfig(use_global=True, use_positive_head=True)
        model_m4 = TwoStageUtilityModel(cfg_m4)
        trainer_m4 = TwoStageModelTrainer(model_m4, device=dev)
        trainer_m4.fit(train_data, val_data, epochs=epochs)
        trainer_m4.save_checkpoint(ckpt_dir / f"M4_seed_{seed}.pt")

        model_m4.eval()
        with torch.no_grad():
            p_m4, q_m4, t_m4, u_m4 = model_m4(
                test_data["X_loc"].to(dev),
                test_data["X_glob"].to(dev),
            )
            p_m4_np = p_m4.cpu().numpy()
            u_m4_np = u_m4.cpu().numpy()

        rho_m4, _ = safe_spearmanr(u_m4_np, y_u_test)
        auc_m4 = float(roc_auc_score(y_pos_test, p_m4_np)) if len(np.unique(y_pos_test)) > 1 else 0.5
        sel_m4 = evaluate_frame_selection(
            scores=u_m4_np,
            delta_q=delta_q_test,
            costs_ms=costs_test,
            oracle_u=y_u_test,
            reject_nonpositive=True,
        )
        ablation_results["M4_PositiveGlobal"]["per_seed"][seed] = {
            "spearman_rho": float(rho_m4),
            "auroc": auc_m4,
            **sel_m4,
        }

        # ─── Policy Variant 1: Oracle-Positive Filter + M4 Ranking ───
        # Uses oracle positive knowledge as an upper bound baseline
        oracle_filter_scores = np.where(y_u_test > 0.0, u_m4_np, -1e9)
        sel_ora_pos = evaluate_frame_selection(
            scores=oracle_filter_scores,
            delta_q=delta_q_test,
            costs_ms=costs_test,
            oracle_u=y_u_test,
            reject_nonpositive=True,
        )
        ablation_results["Policy_OraclePositive"]["per_seed"][seed] = {
            "spearman_rho": float(rho_m4),
            "auroc": 1.0,
            **sel_ora_pos,
        }

        # ─── Policy Variant 2: Probability-Thresholded Selection (p_i > 0.50) ───
        thresh_scores = np.where(p_m4_np > 0.50, u_m4_np, -1e9)
        sel_thresh = evaluate_frame_selection(
            scores=thresh_scores,
            delta_q=delta_q_test,
            costs_ms=costs_test,
            oracle_u=y_u_test,
            reject_nonpositive=True,
        )
        ablation_results["Policy_ProbThreshold"]["per_seed"][seed] = {
            "spearman_rho": float(rho_m4),
            "auroc": auc_m4,
            **sel_thresh,
        }

        print(
            f"  Seed {seed} -> M0 rho={rho_m0:.3f}, M1 rho={rho_m1:.3f}, M2 rho={rho_m2:.3f}, "
            f"M3 rho={rho_m3:.3f} (AUC={auc_m3:.3f}), M4 rho={rho_m4:.3f} (AUC={auc_m4:.3f})"
        )

    # 4. Save results JSON
    results_path = output_dir / "model_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n>> Saved full ablation results to {results_path}")

    return ablation_results


def main():
    parser = argparse.ArgumentParser(description="Phase 12: Controlled Model Ablation")
    parser.add_argument(
        "--oracle_csv",
        type=Path,
        default=REPO_ROOT / "results" / "oracle_dataset" / "oracle_dataset.csv",
        help="Path to audited oracle dataset CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=get_phase12_output_dir("model_ablation"),
        help="Output directory for ablation results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Training epochs per model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    args = parser.parse_args()

    run_model_ablation(
        oracle_csv=args.oracle_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
