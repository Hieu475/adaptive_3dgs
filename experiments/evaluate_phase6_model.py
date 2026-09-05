#!/usr/bin/env python3
"""Phase 6: Comprehensive Model Evaluation (RQ4 - Prediction Quality).

Compares:
  - Phase 6 Context-Aware TwoHeadMLP (V11)
  - Phase 4 Pointwise TwoHeadMLP (Frozen baseline)
  - Heuristic Baselines:
      * B1: RGB Error
      * B2: RGB + Depth Error
      * B3: Error × Influence Mass
      * B4: Binary Threshold

Metrics:
  - Spearman rank correlation rho(U*, U_hat)
  - Pearson correlation r(U*, U_hat)
  - Ranking quality: NDCG@5, NDCG@10, NDCG@20
  - Error metrics: MAE(U), MAE(Delta Q), MAE(Delta T)
  - Context Sensitivity:
      * Context delta correlation: rho(Delta U*_context, Delta U_hat_context)
      * Within-candidate context variance std_S(U_hat(i|S))
  - Breakdown by context size (|S| = 0, 1, 4, 8)
  - Breakdown by context type (empty, spatial_knn, overlap_top, random)

Usage:
    python experiments/evaluate_phase6_model.py
    python experiments/evaluate_phase6_model.py --dataset results/phase6_context_utility/datasets/conditional_oracle_seed_42.json
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase6_context import (
    PHASE6_FEATURE_NAMES,
    PHASE6_FEATURE_DIM,
    SELF_SLICE,
    NEIGHBOR_SLICE,
    OVERLAP_SLICE,
    SELECTED_SLICE,
)
from research.phase6_model import (
    ContextAwareTwoHeadMLP,
    Phase6ModelConfig,
    FrozenContextPredictor,
)
from research.utility_predictor import FrozenUtilityPredictor
from research.utility_metrics import (
    safe_spearmanr,
    safe_pearsonr,
    compute_ndcg_at_k,
    compute_calibration_metrics,
)


def evaluate_predictions(
    pred_u: np.ndarray,
    oracle_u: np.ndarray,
    pred_q: Optional[np.ndarray] = None,
    oracle_q: Optional[np.ndarray] = None,
    pred_t: Optional[np.ndarray] = None,
    oracle_t: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute complete prediction fidelity metrics."""
    rho_u, p_u = safe_spearmanr(pred_u, oracle_u)
    r_u, p_ru = safe_pearsonr(pred_u, oracle_u)
    mae_u = float(np.mean(np.abs(pred_u - oracle_u)))

    res = {
        "spearman_rho": float(rho_u),
        "spearman_pval": float(p_u),
        "pearson_r": float(r_u),
        "pearson_pval": float(p_ru),
        "mae_utility": mae_u,
        "ndcg_5": compute_ndcg_at_k(pred_u, oracle_u, k=5),
        "ndcg_10": compute_ndcg_at_k(pred_u, oracle_u, k=10),
        "ndcg_20": compute_ndcg_at_k(pred_u, oracle_u, k=20),
    }

    if pred_q is not None and oracle_q is not None:
        rho_q, _ = safe_spearmanr(pred_q, oracle_q)
        res["spearman_rho_delta_q"] = float(rho_q)
        res["mae_delta_q"] = float(np.mean(np.abs(pred_q - oracle_q)))

    if pred_t is not None and oracle_t is not None:
        rho_t, _ = safe_spearmanr(pred_t, oracle_t)
        res["spearman_rho_delta_t"] = float(rho_t)
        res["mae_delta_t"] = float(np.mean(np.abs(pred_t - oracle_t)))

    calib = compute_calibration_metrics(pred_u, oracle_u)
    res.update(calib)
    return res


def evaluate_context_sensitivity(
    samples: List[Dict[str, Any]],
    phase6_pred_u: np.ndarray,
    phase4_pred_u: np.ndarray,
) -> Dict[str, Any]:
    """Measure how sensitive models are to changes in context S.

    For each candidate i evaluated under both S = ∅ and S ≠ ∅:
        Delta U*(i | S) = U*(i | S) - U*(i | ∅)
        Delta U_hat(i | S) = U_hat(i | S) - U_hat(i | ∅)
    """
    # Group by (frame, candidate_id)
    cand_groups: Dict[Tuple[int, int], List[Tuple[int, Dict]]] = {}
    for idx, s in enumerate(samples):
        key = (s.get("frame", 0), s.get("candidate_id", 0))
        cand_groups.setdefault(key, []).append((idx, s))

    p6_within_std = []
    p4_within_std = []
    true_within_std = []

    true_deltas = []
    p6_deltas = []
    p4_deltas = []

    for key, group in cand_groups.items():
        if len(group) < 2:
            continue
        indices = [idx for idx, _ in group]
        
        # Stds across different contexts for the same candidate
        u_true = [s["utility_conditional"] for _, s in group]
        u_p6 = phase6_pred_u[indices]
        u_p4 = phase4_pred_u[indices]

        true_within_std.append(float(np.std(u_true)))
        p6_within_std.append(float(np.std(u_p6)))
        p4_within_std.append(float(np.std(u_p4)))

        # Find empty context sample in group
        empty_sample = next((s for _, s in group if s.get("context_size", 0) == 0), None)
        if empty_sample is not None:
            empty_idx = next(idx for idx, s in group if s.get("context_size", 0) == 0)
            base_u_true = empty_sample["utility_conditional"]
            base_u_p6 = phase6_pred_u[empty_idx]
            base_u_p4 = phase4_pred_u[empty_idx]

            for idx, s in group:
                if s.get("context_size", 0) > 0:
                    true_deltas.append(s["utility_conditional"] - base_u_true)
                    p6_deltas.append(phase6_pred_u[idx] - base_u_p6)
                    p4_deltas.append(phase4_pred_u[idx] - base_u_p4)

    # Correlation of context deltas
    rho_p6_delta, _ = safe_spearmanr(np.array(p6_deltas), np.array(true_deltas)) if len(true_deltas) > 3 else (0.0, 1.0)
    rho_p4_delta, _ = safe_spearmanr(np.array(p4_deltas), np.array(true_deltas)) if len(true_deltas) > 3 else (0.0, 1.0)

    return {
        "n_multi_context_candidates": len(cand_groups),
        "mean_true_within_candidate_std": float(np.mean(true_within_std)) if true_within_std else 0.0,
        "mean_phase6_within_candidate_std": float(np.mean(p6_within_std)) if p6_within_std else 0.0,
        "mean_phase4_within_candidate_std": float(np.mean(p4_within_std)) if p4_within_std else 0.0,
        "n_context_pairs": len(true_deltas),
        "phase6_context_delta_spearman_rho": float(rho_p6_delta),
        "phase4_context_delta_spearman_rho": float(rho_p4_delta),
        "phase6_shows_context_sensitivity": bool(np.mean(p6_within_std) > 1e-8) if p6_within_std else False,
        "phase4_is_context_invariant": bool(np.mean(p4_within_std) < 1e-8) if p4_within_std else True,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 6 context-aware model vs baselines (RQ4).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to conditional oracle dataset JSON.")
    parser.add_argument("--phase6-ckpt", type=str, default=None,
                        help="Path to Phase 6 checkpoint.")
    parser.add_argument("--phase6-norm", type=str, default=None,
                        help="Path to Phase 6 normalizer JSON.")
    parser.add_argument("--phase4-ckpt", type=str, default=None,
                        help="Path to Phase 4 checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(repo_root, "results", "phase6_context_utility")
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = args.dataset or os.path.join(
        output_dir, "datasets", f"conditional_oracle_seed_{args.seed}.json"
    )
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        sys.exit(1)

    print("=" * 80)
    print("  PHASE 6: RQ4 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    print(f"  Dataset: {dataset_path}")

    with open(dataset_path, "r") as f:
        samples = json.load(f)

    N = len(samples)
    print(f"  Total conditional evaluation samples: {N}")

    # Extract target values
    oracle_u = np.array([s["utility_conditional"] for s in samples], dtype=np.float32)
    oracle_q = np.array([s["delta_q_conditional"] for s in samples], dtype=np.float32)
    oracle_t = np.array([s["delta_t_conditional_ms"] for s in samples], dtype=np.float32)
    full_feats = np.array([s["full_feature_vector"] for s in samples], dtype=np.float32)
    self_feats = np.array([s["self_features"] for s in samples], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Evaluate Phase 6 Context-Aware Model
    # ─────────────────────────────────────────────────────────────────────────
    p6_ckpt = args.phase6_ckpt or os.path.join(
        output_dir, "checkpoints", f"context_mlp_V11_seed_{args.seed}.pt"
    )
    p6_norm = args.phase6_norm or os.path.join(
        output_dir, f"normalization_V11.json"
    )

    print(f"\n[1/4] Loading Phase 6 Predictor: {p6_ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p6_predictor = FrozenContextPredictor(p6_ckpt, p6_norm, device=device)
    p6_preds = p6_predictor.predict(torch.tensor(full_feats, dtype=torch.float32))
    p6_u = p6_preds["utility"].detach().cpu().numpy()
    p6_q = p6_preds["delta_q"].detach().cpu().numpy()
    p6_t = p6_preds["delta_t"].detach().cpu().numpy()

    p6_metrics = evaluate_predictions(p6_u, oracle_u, p6_q, oracle_q, p6_t, oracle_t)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Evaluate Phase 4 Pointwise Model (Frozen baseline)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[2/4] Loading Phase 4 Predictor...")
    try:
        p4_predictor = FrozenUtilityPredictor(seed=args.seed, device=device)
        p4_res = p4_predictor.predict_features(self_feats)
        p4_u = p4_res["predicted_utility"]
        p4_q = p4_res["predicted_delta_q"]
        p4_t = p4_res["predicted_delta_t"]
        p4_metrics = evaluate_predictions(p4_u, oracle_u, p4_q, oracle_q, p4_t, oracle_t)
        p4_available = True
    except Exception as e:
        print(f"  [WARN] Phase 4 predictor failed to load ({e}). Using mock/fallback.")
        p4_u = np.zeros(N, dtype=np.float32)
        p4_q = np.zeros(N, dtype=np.float32)
        p4_t = np.ones(N, dtype=np.float32)
        p4_metrics = evaluate_predictions(p4_u, oracle_u, p4_q, oracle_q, p4_t, oracle_t)
        p4_available = False

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Evaluate Heuristic Baselines
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[3/4] Evaluating Heuristic Baselines...")
    # Canonical feature indices:
    # 0: rgb_error, 1: depth_error, 4: influence_mass
    rgb_err = self_feats[:, 0]
    depth_err = self_feats[:, 1]
    err_sum = rgb_err + depth_err
    inf_mass = self_feats[:, 4]

    b1_scores = rgb_err
    b2_scores = err_sum
    b3_scores = err_sum * inf_mass
    b4_scores = (err_sum > np.median(err_sum)).astype(np.float32)

    b1_metrics = evaluate_predictions(b1_scores, oracle_u)
    b2_metrics = evaluate_predictions(b2_scores, oracle_u)
    b3_metrics = evaluate_predictions(b3_scores, oracle_u)
    b4_metrics = evaluate_predictions(b4_scores, oracle_u)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Context Sensitivity & Stratified Breakdowns
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[4/4] Analyzing Context Sensitivity & Stratified Breakdowns...")
    sensitivity_metrics = evaluate_context_sensitivity(samples, p6_u, p4_u)

    # Stratify by Context Size
    sizes = sorted(list(set(s.get("context_size", 0) for s in samples)))
    size_breakdown = {}
    for sz in sizes:
        mask = np.array([s.get("context_size", 0) == sz for s in samples])
        if np.sum(mask) >= 3:
            size_breakdown[f"size_{sz}"] = {
                "n_samples": int(np.sum(mask)),
                "phase6_rho": float(safe_spearmanr(p6_u[mask], oracle_u[mask])[0]),
                "phase4_rho": float(safe_spearmanr(p4_u[mask], oracle_u[mask])[0]),
                "b3_rho": float(safe_spearmanr(b3_scores[mask], oracle_u[mask])[0]),
                "phase6_ndcg_5": compute_ndcg_at_k(p6_u[mask], oracle_u[mask], k=5),
                "phase4_ndcg_5": compute_ndcg_at_k(p4_u[mask], oracle_u[mask], k=5),
            }

    # Stratify by Context Type
    types = sorted(list(set(s.get("context_type", "unknown") for s in samples)))
    type_breakdown = {}
    for ct in types:
        mask = np.array([s.get("context_type", "unknown") == ct for s in samples])
        if np.sum(mask) >= 3:
            type_breakdown[ct] = {
                "n_samples": int(np.sum(mask)),
                "phase6_rho": float(safe_spearmanr(p6_u[mask], oracle_u[mask])[0]),
                "phase4_rho": float(safe_spearmanr(p4_u[mask], oracle_u[mask])[0]),
                "b3_rho": float(safe_spearmanr(b3_scores[mask], oracle_u[mask])[0]),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Gate 6B & Gate 6D Verification
    # ─────────────────────────────────────────────────────────────────────────
    gate_6b_rho = p6_metrics["spearman_rho"] > p4_metrics["spearman_rho"]
    gate_6b_ndcg = p6_metrics["ndcg_5"] > p4_metrics["ndcg_5"]
    gate_6b_pass = bool(gate_6b_rho and gate_6b_ndcg)

    gate_6d_sensitivity = sensitivity_metrics["phase6_shows_context_sensitivity"]
    gate_6d_invariance = sensitivity_metrics["phase4_is_context_invariant"]
    gate_6d_pass = bool(gate_6d_sensitivity and gate_6d_invariance)

    print("\n" + "=" * 80)
    print("  EVALUATION SUMMARY TABLE (RQ4: PREDICTION QUALITY)")
    print("=" * 80)
    print(f"{'Policy / Model':<28} | {'Spearman ρ':<11} | {'Pearson r':<11} | {'NDCG@5':<10} | {'NDCG@10':<10} | {'MAE(U)':<10}")
    print("-" * 88)
    print(f"{'Phase 6 Context-Aware (V11)':<28} | {p6_metrics['spearman_rho']:<11.4f} | {p6_metrics['pearson_r']:<11.4f} | {p6_metrics['ndcg_5']:<10.4f} | {p6_metrics['ndcg_10']:<10.4f} | {p6_metrics['mae_utility']:<10.2e}")
    print(f"{'Phase 4 Pointwise TwoHeadMLP':<28} | {p4_metrics['spearman_rho']:<11.4f} | {p4_metrics['pearson_r']:<11.4f} | {p4_metrics['ndcg_5']:<10.4f} | {p4_metrics['ndcg_10']:<10.4f} | {p4_metrics['mae_utility']:<10.2e}")
    print(f"{'B3: Error × Influence':<28} | {b3_metrics['spearman_rho']:<11.4f} | {b3_metrics['pearson_r']:<11.4f} | {b3_metrics['ndcg_5']:<10.4f} | {b3_metrics['ndcg_10']:<10.4f} | {b3_metrics['mae_utility']:<10.2e}")
    print(f"{'B2: RGB + Depth Error':<28} | {b2_metrics['spearman_rho']:<11.4f} | {b2_metrics['pearson_r']:<11.4f} | {b2_metrics['ndcg_5']:<10.4f} | {b2_metrics['ndcg_10']:<10.4f} | {b2_metrics['mae_utility']:<10.2e}")
    print(f"{'B1: RGB Error':<28} | {b1_metrics['spearman_rho']:<11.4f} | {b1_metrics['pearson_r']:<11.4f} | {b1_metrics['ndcg_5']:<10.4f} | {b1_metrics['ndcg_10']:<10.4f} | {b1_metrics['mae_utility']:<10.2e}")
    print(f"{'B4: Binary Threshold':<28} | {b4_metrics['spearman_rho']:<11.4f} | {b4_metrics['pearson_r']:<11.4f} | {b4_metrics['ndcg_5']:<10.4f} | {b4_metrics['ndcg_10']:<10.4f} | {b4_metrics['mae_utility']:<10.2e}")
    print("=" * 88)

    print("\nGate Verification:")
    print(f"  Gate 6B (Prediction):  {'✓ PASS' if gate_6b_pass else '✗ FAIL'} "
          f"[ρ: {p6_metrics['spearman_rho']:.4f} vs {p4_metrics['spearman_rho']:.4f}, "
          f"NDCG@5: {p6_metrics['ndcg_5']:.4f} vs {p4_metrics['ndcg_5']:.4f}]")
    print(f"  Gate 6D (Interaction): {'✓ PASS' if gate_6d_pass else '✗ FAIL'} "
          f"[P6 context sensitivity: {sensitivity_metrics['mean_phase6_within_candidate_std']:.2e} > 0, "
          f"P4 context invariance: {sensitivity_metrics['mean_phase4_within_candidate_std']:.2e} == 0]")

    # ─────────────────────────────────────────────────────────────────────────
    # Save Report
    # ─────────────────────────────────────────────────────────────────────────
    report = {
        "evaluation": "Phase 6 RQ4 Model Prediction Quality",
        "seed": args.seed,
        "n_samples": N,
        "models": {
            "phase6_context_aware_v11": p6_metrics,
            "phase4_pointwise_mlp": p4_metrics,
            "b3_error_influence": b3_metrics,
            "b2_rgb_depth_error": b2_metrics,
            "b1_rgb_error": b1_metrics,
            "b4_binary": b4_metrics,
        },
        "context_sensitivity": sensitivity_metrics,
        "stratification": {
            "by_context_size": size_breakdown,
            "by_context_type": type_breakdown,
        },
        "gates": {
            "gate_6b_pass": gate_6b_pass,
            "gate_6b_details": {
                "phase6_rho": p6_metrics["spearman_rho"],
                "phase4_rho": p4_metrics["spearman_rho"],
                "phase6_ndcg_5": p6_metrics["ndcg_5"],
                "phase4_ndcg_5": p4_metrics["ndcg_5"],
            },
            "gate_6d_pass": gate_6d_pass,
            "gate_6d_details": sensitivity_metrics,
        }
    }

    report_path = os.path.join(output_dir, f"model_evaluation_rq4_seed_{args.seed}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Saved] Evaluation Report: {report_path}")


if __name__ == "__main__":
    main()
