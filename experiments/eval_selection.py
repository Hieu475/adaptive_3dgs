#!/usr/bin/env python3
"""Phase 4 RQ2: Budget-Constrained Selection Evaluation.

Evaluates selection efficiency across all protocol budgets:
  B in {10%, 20%, 40%, 60%, 80%}

Addresses:
  - top-k ranking-only selection (rank_candidates)
  - budgeted greedy selection (select_under_budget)
  - NDCG@B, Overlap@B, Regret(B), OSE(B), and Realized Delta Q(B)
  - Full baseline comparison: B0 to B7 + Oracle
  - Multi-seed aggregation (mean +/- std, 95% CI)
  - Exports results/learned_utility/rq2/budget_sweep.json and summary.json
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.protocol import (
    load_protocol,
    get_seeds,
    get_repo_root,
)
from research.utility_dataset import (
    load_canonical_oracle_dataset,
    prepare_normalized_splits,
)
from research.utility_models import (
    RandomScorer,
    RGBErrorScorer,
    RGBDepthErrorScorer,
    ErrorInfluenceScorer,
    BinaryThresholdScorer,
    LinearUtilityModel,
    TwoHeadLinear,
    TwoHeadMLP,
)
from research.utility_training import UtilityModelTrainer
from research.utility_metrics import (
    PROTOCOL_BUDGETS,
    compute_confidence_interval_95,
    rank_candidates,
    select_under_budget,
    evaluate_rq2_selection,
)


def evaluate_method_budgets(
    pred_u: np.ndarray,
    oracle_u: np.ndarray,
    delta_q: np.ndarray,
    costs: np.ndarray,
    budgets: Tuple[float, ...] = PROTOCOL_BUDGETS,
) -> Dict[str, Any]:
    """Computes detailed per-budget selection statistics for a given scoring function."""
    n = len(pred_u)
    total_cost = float(costs.sum())
    pred_ranks = rank_candidates(pred_u)
    ora_ranks = rank_candidates(oracle_u)

    budget_results = {}
    for frac in budgets:
        pct_label = f"{int(frac * 100)}pct"
        k = max(1, int(n * frac))

        # 1. Top-k ranking selection
        top_pred = set(pred_ranks[:k].tolist())
        top_ora = set(ora_ranks[:k].tolist())
        overlap = len(top_pred & top_ora) / k
        gain_pred = float(delta_q[list(top_pred)].sum())
        gain_ora = float(delta_q[list(top_ora)].sum())
        ose = float(gain_pred / (gain_ora + 1e-8)) if gain_ora > 0 else 1.0
        regret = float(gain_ora - gain_pred)

        # 2. Budget-constrained selection
        budget_val = frac * total_cost
        sel_pred, c_pred = select_under_budget(pred_u, costs, budget_val)
        sel_ora, c_ora = select_under_budget(oracle_u, costs, budget_val)
        gain_bg_pred = float(delta_q[sel_pred].sum()) if len(sel_pred) > 0 else 0.0
        gain_bg_ora = float(delta_q[sel_ora].sum()) if len(sel_ora) > 0 else 0.0
        ose_bg = float(gain_bg_pred / (gain_bg_ora + 1e-8)) if gain_bg_ora > 0 else 1.0
        regret_bg = float(gain_bg_ora - gain_bg_pred)

        budget_results[pct_label] = {
            "budget_fraction": frac,
            "k_items": k,
            "ranking_selection": {
                "overlap": float(overlap),
                "ose": float(ose),
                "regret": float(regret),
                "realized_delta_q": float(gain_pred),
                "oracle_delta_q": float(gain_ora),
            },
            "budgeted_selection": {
                "n_selected": len(sel_pred),
                "realized_cost": float(c_pred),
                "target_budget": float(budget_val),
                "ose": float(ose_bg),
                "regret": float(regret_bg),
                "realized_delta_q": float(gain_bg_pred),
                "oracle_delta_q": float(gain_bg_ora),
            },
        }

    return budget_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 RQ2 Selection under Budget.")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Directory containing model checkpoints.")
    args = parser.parse_args()

    repo_root = get_repo_root()
    protocol = load_protocol()
    seeds = get_seeds(protocol)

    ckpt_dir = args.ckpt_dir or os.path.join(repo_root, "results", "learned_utility", "checkpoints")

    print("=" * 95)
    print("   PHASE 4 RQ2: BUDGET-CONSTRAINED SELECTION SWEEP ACROSS PROTOCOL BUDGETS")
    print("=" * 95)
    print(f">> Protocol Budgets: {[f'{int(b*100)}%' for b in PROTOCOL_BUDGETS]}")

    # 1. Load canonical dataset and prepare splits
    dataset = load_canonical_oracle_dataset()
    train_ds, val_ds, test_ds, normalizer = prepare_normalized_splits(dataset=dataset)

    print(f">> Evaluated on independent test split: scene '{test_ds.metadata[0].scene}', N={len(test_ds)}")

    X_test_unnorm = dataset.get_split("cross_scene_test").X_np
    X_test_norm = test_ds.X
    y_q_test = test_ds.delta_q_np
    y_t_test = test_ds.delta_t_np
    y_u_test = test_ds.utility_np

    # 2. Heuristic baselines (B1 to B4)
    scorers = {
        "B1: RGB Error": RGBErrorScorer(rgb_idx=0),
        "B2: RGB + Depth Error": RGBDepthErrorScorer(rgb_idx=0, depth_idx=1),
        "B3: Error × Influence": ErrorInfluenceScorer(rgb_idx=0, depth_idx=1, inf_idx=4),
        "B4: Binary Threshold": BinaryThresholdScorer(rgb_idx=0, depth_idx=1),
    }

    results_by_method = {}

    for name, scorer in scorers.items():
        scores = scorer.score(X_test_unnorm)
        results_by_method[name] = evaluate_method_budgets(
            pred_u=scores,
            oracle_u=y_u_test,
            delta_q=y_q_test,
            costs=y_t_test,
        )

    # Oracle reference
    results_by_method["Oracle (Reference)"] = evaluate_method_budgets(
        pred_u=y_u_test,
        oracle_u=y_u_test,
        delta_q=y_q_test,
        costs=y_t_test,
    )

    # 3. Multi-seed methods (B0 Random, B5 Linear, B6 TwoHeadLinear, B7 TwoHeadMLP)
    seed_runs: Dict[str, List[Dict[str, Any]]] = {
        "B0: Random": [],
        "B5: Linear Utility": [],
        "B6: Two-Head Linear": [],
        "B7: Two-Head MLP (Ours)": [],
    }

    for seed in seeds:
        # B0: Random
        rnd_scorer = RandomScorer(seed=seed)
        rnd_scores = rnd_scorer.score(X_test_unnorm)
        seed_runs["B0: Random"].append(
            evaluate_method_budgets(rnd_scores, y_u_test, y_q_test, y_t_test)
        )

        # B5: Linear
        b5_path = os.path.join(ckpt_dir, f"linear_direct_seed_{seed}.pt")
        b5_model = LinearUtilityModel(in_features=len(test_ds.feature_names))
        if os.path.exists(b5_path):
            UtilityModelTrainer.load_checkpoint(b5_model, b5_path)
            b5_model.eval()
            with torch.no_grad():
                pred_u_b5 = b5_model(X_test_norm).cpu().numpy()
            seed_runs["B5: Linear Utility"].append(
                evaluate_method_budgets(pred_u_b5, y_u_test, y_q_test, y_t_test)
            )

        # B6: Two-Head Linear
        b6_path = os.path.join(ckpt_dir, f"two_head_linear_seed_{seed}.pt")
        b6_model = TwoHeadLinear(in_features=len(test_ds.feature_names))
        if os.path.exists(b6_path):
            UtilityModelTrainer.load_checkpoint(b6_model, b6_path)
            b6_model.eval()
            with torch.no_grad():
                _, _, pred_u_b6 = b6_model(X_test_norm)
            seed_runs["B6: Two-Head Linear"].append(
                evaluate_method_budgets(pred_u_b6.cpu().numpy(), y_u_test, y_q_test, y_t_test)
            )

        # B7: Two-Head MLP
        b7_path = os.path.join(ckpt_dir, f"two_head_mlp_seed_{seed}.pt")
        b7_model = TwoHeadMLP(in_features=len(test_ds.feature_names))
        if os.path.exists(b7_path):
            UtilityModelTrainer.load_checkpoint(b7_model, b7_path)
            b7_model.eval()
            with torch.no_grad():
                _, _, pred_u_b7 = b7_model(X_test_norm)
            seed_runs["B7: Two-Head MLP (Ours)"].append(
                evaluate_method_budgets(pred_u_b7.cpu().numpy(), y_u_test, y_q_test, y_t_test)
            )

    # 4. Aggregate multi-seed methods
    for name, runs in seed_runs.items():
        if not runs:
            continue
        agg = {}
        for frac in PROTOCOL_BUDGETS:
            pct_label = f"{int(frac * 100)}pct"
            ose_vals = [r[pct_label]["ranking_selection"]["ose"] for r in runs]
            dq_vals = [r[pct_label]["ranking_selection"]["realized_delta_q"] for r in runs]
            ov_vals = [r[pct_label]["ranking_selection"]["overlap"] for r in runs]
            reg_vals = [r[pct_label]["ranking_selection"]["regret"] for r in runs]

            agg[pct_label] = {
                "budget_fraction": frac,
                "ranking_selection": {
                    "ose_mean": float(np.mean(ose_vals)),
                    "ose_std": float(np.std(ose_vals)),
                    "ose_ci_95": compute_confidence_interval_95(float(np.std(ose_vals)), len(ose_vals)),
                    "realized_delta_q_mean": float(np.mean(dq_vals)),
                    "realized_delta_q_std": float(np.std(dq_vals)),
                    "overlap_mean": float(np.mean(ov_vals)),
                    "regret_mean": float(np.mean(reg_vals)),
                },
                "oracle_delta_q": runs[0][pct_label]["ranking_selection"]["oracle_delta_q"],
            }
        results_by_method[name] = agg

    # 5. Print Budget Sweep Table for Two-Head MLP vs Baselines
    print("\n" + "=" * 95)
    print(f"{'Method':<26} | {'Budget':<8} | {'OSE (Mean±Std)':<18} | {'95% CI':<10} | {'Realized ΔQ':<13} | {'Oracle ΔQ':<13}")
    print("-" * 95)

    focus_methods = ["B1: RGB Error", "B3: Error × Influence", "B7: Two-Head MLP (Ours)", "Oracle (Reference)"]
    for m_name in focus_methods:
        data = results_by_method[m_name]
        for frac in PROTOCOL_BUDGETS:
            pct_label = f"{int(frac * 100)}pct"
            b_info = data[pct_label]
            if "ose_mean" in b_info.get("ranking_selection", {}):
                ose_str = f"{b_info['ranking_selection']['ose_mean']:.3f} ±{b_info['ranking_selection']['ose_std']:.3f}"
                ci_str = f"±{b_info['ranking_selection']['ose_ci_95']:.3f}"
                dq_str = f"{b_info['ranking_selection']['realized_delta_q_mean']:+.6f}"
                ora_str = f"{b_info['oracle_delta_q']:+.6f}"
            else:
                sel = b_info["ranking_selection"]
                ose_str = f"{sel['ose']:.3f}"
                ci_str = "--"
                dq_str = f"{sel['realized_delta_q']:+.6f}"
                ora_str = f"{sel['oracle_delta_q']:+.6f}"
            print(f"{m_name:<26} | {pct_label:<8} | {ose_str:<18} | {ci_str:<10} | {dq_str:<13} | {ora_str:<13}")
        print("-" * 95)

    # 6. Save JSON Artifacts
    rq2_dir = os.path.join(repo_root, "results", "learned_utility", "rq2")
    os.makedirs(rq2_dir, exist_ok=True)

    sweep_path = os.path.join(rq2_dir, "budget_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(results_by_method, f, indent=2)

    summary_path = os.path.join(rq2_dir, "summary.json")
    summary = {
        "budgets": list(PROTOCOL_BUDGETS),
        "evaluated_methods": list(results_by_method.keys()),
        "two_head_mlp_budget_performance": results_by_method.get("B7: Two-Head MLP (Ours)", {}),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n>> Saved RQ2 budget sweep to: {sweep_path}")
    print(f">> Saved RQ2 summary to: {summary_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()
