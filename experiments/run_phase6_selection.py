#!/usr/bin/env python3
"""Phase 6: Budget-Constrained Selection Benchmark (RQ5).

Compares:
  - NO_OP: S = ∅
  - RANDOM: Random uniform permutation under budget
  - ERROR_ONLY: Rank by photometric + geometric error
  - ERROR_INFLUENCE: Rank by error × attribution mass
  - HEURISTIC: Knapsack heuristic (Importance / Cost)
  - PHASE4_LEARNED: Pointwise TwoHeadMLP U_hat = f(s_i)
  - PHASE6_STATIC: Context model with S = ∅ (static 1-pass)
  - PHASE6_ADAPTIVE (OURS): Adaptive Greedy with dynamic context S_t re-ranking
  - ORACLE_REFERENCE: Ground truth marginal reference

Fairness Contract:
  All policies face the exact same compute budget B and safety factor alpha:
      sum_{i in S_B} (alpha * C_i) <= B

Experiments:
  1. Relative Budget Sweep: [10%, 20%, 40%, 60%, 80%]
  2. Wall-Clock Budget Sweep: [10ms, 15ms, 20ms, 33.3ms]
  3. Safety Margin Ablation: alpha in [1.00, 1.05, 1.10, 1.20]
  4. Statistical Hypothesis Testing: Paired Wilcoxon test comparing Phase 6 vs Phase 4 & Heuristic

Usage:
    python experiments/run_phase6_selection.py --quick           # Quick run on prototype data
    python experiments/run_phase6_selection.py --seed 42         # Full benchmark
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from scipy.stats import wilcoxon
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase6_context import ContextConfig
from research.phase6_model import FrozenContextPredictor
from research.phase6_selection import (
    Phase6PolicyName,
    select_phase6_subset,
    SelectionResult,
)
from research.utility_predictor import FrozenUtilityPredictor


def run_budget_sweep_on_pool(
    candidates: List[Dict[str, Any]],
    positions: torch.Tensor,
    all_features: np.ndarray,
    p6_predictor: FrozenContextPredictor,
    budgets: List[float],
    budget_type: str,
    safety_factor: float = 1.0,
    reject_negative: bool = False,
    seed: int = 42,
    policies: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Evaluates all competing policies on a candidate pool across budgets."""
    if policies is None:
        policies = [
            "no_op",
            "random",
            "error_only",
            "error_influence",
            "heuristic",
            "phase4_learned",
            "phase6_static",
            "phase6_adaptive",
            "oracle_reference",
        ]

    results: List[Dict[str, Any]] = []

    for b in budgets:
        # Evaluate each policy
        for pol in policies:
            sel_res = select_phase6_subset(
                candidates=candidates,
                policy=pol,
                budget=b,
                seed=seed,
                safety_factor=safety_factor,
                reject_negative=reject_negative if pol in ("phase4_learned", "phase6_static", "phase6_adaptive", "oracle_reference") else False,
                use_predicted_cost=True,
                positions=positions,
                all_features=all_features,
                phase6_predictor=p6_predictor,
            )

            # Compute realized metrics from candidate records
            # Delta Q is computed as sum of candidate oracle deltas in the selection
            sel_idx = sel_res.selected_indices
            realized_dq = float(sum(
                float(candidates[i].get("delta_q_conditional", candidates[i].get("delta_quality_global", 0.0)))
                for i in sel_idx
            )) if sel_idx else 0.0

            realized_actual_t = float(sum(
                float(candidates[i].get("delta_t_conditional_ms", candidates[i].get("measured_trial_cost_ms", 1.0)))
                for i in sel_idx
            )) if sel_idx else 0.0

            efficiency = realized_dq / max(1e-4, sel_res.scheduled_cost) if sel_res.scheduled_cost > 0 else 0.0

            results.append({
                "budget_type": budget_type,
                "budget": float(b),
                "policy": pol,
                "safety_factor": float(safety_factor),
                "k_selected": sel_res.k_count,
                "predicted_cost": sel_res.predicted_cost,
                "scheduled_cost": sel_res.scheduled_cost,
                "actual_cost_ms": realized_actual_t,
                "budget_violation_ms": max(0.0, sel_res.scheduled_cost - b),
                "is_violation": sel_res.is_scheduled_violation,
                "realized_delta_q": realized_dq,
                "efficiency": efficiency,
                "selection_time_ms": sel_res.selection_time_ms,
                "rejected_negative": sel_res.rejected_negative_count,
            })

    return results


def compute_paired_statistics(
    results: List[Dict[str, Any]],
    baseline_policy: str = "phase4_learned",
    target_policy: str = "phase6_adaptive",
) -> Dict[str, Any]:
    """Runs paired Wilcoxon signed-rank tests across budget points."""
    # Group by budget
    budgets = sorted(list(set(r["budget"] for r in results)))
    dq_base = []
    dq_target = []

    for b in budgets:
        r_b = next((r for r in results if r["budget"] == b and r["policy"] == baseline_policy), None)
        r_t = next((r for r in results if r["budget"] == b and r["policy"] == target_policy), None)
        if r_b and r_t:
            dq_base.append(r_b["realized_delta_q"])
            dq_target.append(r_t["realized_delta_q"])

    diffs = np.array(dq_target) - np.array(dq_base)
    n = len(diffs)

    if n >= 4 and not np.all(diffs == 0):
        try:
            stat, p_val = wilcoxon(diffs, alternative="greater")
            stat_val = float(stat)
            p_val_val = float(p_val)
        except Exception:
            stat_val = 0.0
            p_val_val = 1.0
    else:
        stat_val = 0.0
        p_val_val = 1.0

    mean_diff = float(np.mean(diffs)) if n > 0 else 0.0
    win_rate = float(np.mean(diffs > 0)) if n > 0 else 0.0

    return {
        "n_budget_points": n,
        "baseline_policy": baseline_policy,
        "target_policy": target_policy,
        "mean_difference": mean_diff,
        "win_rate": win_rate,
        "wilcoxon_stat": stat_val,
        "wilcoxon_pval": p_val_val,
        "statistically_significant": bool(p_val_val < 0.05),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Budget Selection Benchmark (RQ5)")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--reject-negative", action="store_true", default=False,
                        help="Reject candidates with predicted non-positive utility.")
    parser.add_argument("--quick", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(repo_root, "results", "phase6_context_utility", "selection")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load dataset
    ds_path = args.dataset or os.path.join(
        repo_root, "results", "phase6_context_utility", "datasets", f"conditional_oracle_seed_{args.seed}.json"
    )
    print("=" * 80)
    print("  PHASE 6: RQ5 BUDGET-CONSTRAINED SELECTION BENCHMARK")
    print("=" * 80)
    print(f"  Dataset: {ds_path}")
    print(f"  Reject Negative: {args.reject_negative}")

    with open(ds_path, "r") as f:
        samples = json.load(f)

    # Filter to empty context samples to form initial candidate pool
    # Each unique candidate appears with context_size == 0
    empty_samples = [s for s in samples if s.get("context_size", 0) == 0]
    if not empty_samples:
        empty_samples = samples[:25]

    candidates = []
    for s in empty_samples:
        cand = {
            "gaussian_id": s.get("candidate_id", 0),
            "persistent_id": s.get("candidate_persistent_id", 0),
            "features": {
                "rgb_error": s["self_features"][0],
                "depth_error": s["self_features"][1],
                "influence_mass": s["self_features"][4],
            },
            "predicted_importance": float(s["self_features"][0] + s["self_features"][1]),
            "predicted_utility": float(s["utility_conditional"]),
            "predicted_delta_t": float(s.get("t_si_ms", 5.0)),
            "measured_trial_cost_ms": float(s.get("t_si_ms", 5.0)),
            "delta_q_conditional": float(s["delta_q_conditional"]),
            "delta_t_conditional_ms": float(s.get("t_si_ms", 5.0)),
            "oracle_utility_joint_global": float(s["utility_conditional"]),
            "full_feature_vector": s["full_feature_vector"],
        }
        candidates.append(cand)

    N_cand = len(candidates)
    print(f"  Candidate pool N = {N_cand}")

    # Build mock or real positions & all_features for context queries
    positions = torch.randn(max(100, N_cand + 10), 3)
    all_features = np.zeros((max(100, N_cand + 10), 11), dtype=np.float32)
    for i, c in enumerate(candidates):
        gid = c["gaussian_id"]
        if gid < len(all_features):
            all_features[gid, 0] = c["features"]["rgb_error"]
            all_features[gid, 1] = c["features"]["depth_error"]
            all_features[gid, 4] = c["features"]["influence_mass"]

    # 2. Load Phase 6 Predictor
    p6_ckpt = os.path.join(repo_root, "results", "phase6_context_utility", "checkpoints", f"context_mlp_V11_seed_{args.seed}.pt")
    p6_norm = os.path.join(repo_root, "results", "phase6_context_utility", "normalization_V11.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p6_predictor = FrozenContextPredictor(p6_ckpt, p6_norm, device=device)

    # Compute pool total cost
    total_cost = sum(c["predicted_delta_t"] for c in candidates)
    print(f"  Total candidate pool cost: {total_cost:.2f} ms")

    # 3. Experiment A: Relative Budget Sweep
    rel_fractions = [0.10, 0.20, 0.40, 0.60, 0.80] if not args.quick else [0.20, 0.50, 0.80]
    rel_budgets = [float(f * total_cost) for f in rel_fractions]
    print(f"\n[Experiment A] Running Relative Budget Sweep: {rel_fractions}...")
    rel_results = run_budget_sweep_on_pool(
        candidates=candidates,
        positions=positions,
        all_features=all_features,
        p6_predictor=p6_predictor,
        budgets=rel_budgets,
        budget_type="relative",
        safety_factor=1.10,
        reject_negative=args.reject_negative,
        seed=args.seed,
    )

    # 4. Experiment B: Wall-Clock Budget Sweep
    wall_budgets = [10.0, 15.0, 20.0, 33.3] if not args.quick else [10.0, 20.0]
    print(f"[Experiment B] Running Wall-Clock Budget Sweep: {wall_budgets} ms...")
    wall_results = run_budget_sweep_on_pool(
        candidates=candidates,
        positions=positions,
        all_features=all_features,
        p6_predictor=p6_predictor,
        budgets=wall_budgets,
        budget_type="wall_clock",
        safety_factor=1.10,
        reject_negative=args.reject_negative,
        seed=args.seed,
    )

    # 5. Experiment C: Safety Margin Ablation
    alphas = [1.00, 1.05, 1.10, 1.20]
    fixed_budget = float(0.40 * total_cost)
    print(f"[Experiment C] Running Safety Factor Ablation: {alphas} at Budget={fixed_budget:.1f}ms...")
    alpha_results = []
    for a in alphas:
        r = run_budget_sweep_on_pool(
            candidates=candidates,
            positions=positions,
            all_features=all_features,
            p6_predictor=p6_predictor,
            budgets=[fixed_budget],
            budget_type="safety_ablation",
            safety_factor=a,
            reject_negative=args.reject_negative,
            seed=args.seed,
        )
        alpha_results.extend(r)

    # 6. Statistical Hypothesis Testing
    print("\n[Analysis] Computing Statistical Hypothesis Tests...")
    test_vs_p4 = compute_paired_statistics(
        rel_results, baseline_policy="phase4_learned", target_policy="phase6_adaptive"
    )
    test_vs_heur = compute_paired_statistics(
        rel_results, baseline_policy="heuristic", target_policy="phase6_adaptive"
    )
    test_vs_err_inf = compute_paired_statistics(
        rel_results, baseline_policy="error_influence", target_policy="phase6_adaptive"
    )

    # 7. Print Summary Tables
    print("\n" + "=" * 85)
    print("  RELATIVE BUDGET SWEEP SUMMARY (Realized ΔQ × 10^5)")
    print("=" * 85)
    pols_to_show = ["no_op", "random", "error_influence", "heuristic", "phase4_learned", "phase6_static", "phase6_adaptive", "oracle_reference"]
    header = f"{'Budget':<10} | " + " | ".join(f"{p[:8]:<8}" for p in pols_to_show)
    print(header)
    print("-" * len(header))

    for idx, f in enumerate(rel_fractions):
        b = rel_budgets[idx]
        vals = []
        for p in pols_to_show:
            match = next((r for r in rel_results if abs(r["budget"] - b) < 1e-4 and r["policy"] == p), None)
            val = match["realized_delta_q"] * 1e5 if match else 0.0
            vals.append(f"{val:<8.2f}")
        print(f"{f*100:4.0f}%      | " + " | ".join(vals))
    print("=" * 85)

    print("\nGate 6C Decision Verification:")
    mean_gain_diff = test_vs_p4["mean_difference"]
    gate_6c_pass = test_vs_p4["statistically_significant"] or (test_vs_p4["win_rate"] >= 0.5 and mean_gain_diff >= 0.0)
    print(f"  Gate 6C (Decision):  {'✓ PASS' if gate_6c_pass else '✗ FAIL'}")
    print(f"    P6 Adaptive vs P4 Learned:  win_rate={test_vs_p4['win_rate']*100:.1f}%, mean_diff={mean_gain_diff:.2e}, p={test_vs_p4['wilcoxon_pval']:.4f}")
    print(f"    P6 Adaptive vs Heuristic:   win_rate={test_vs_heur['win_rate']*100:.1f}%, mean_diff={test_vs_heur['mean_difference']:.2e}, p={test_vs_heur['wilcoxon_pval']:.4f}")

    # 8. Save Artifacts
    artifacts = {
        "benchmark": "Phase 6 RQ5 Budget-Constrained Selection",
        "seed": args.seed,
        "n_candidates": N_cand,
        "relative_sweep": rel_results,
        "wall_clock_sweep": wall_results,
        "safety_ablation": alpha_results,
        "statistical_tests": {
            "vs_phase4_learned": test_vs_p4,
            "vs_heuristic": test_vs_heur,
            "vs_error_influence": test_vs_err_inf,
        },
        "gate_6c": {
            "passed": gate_6c_pass,
            "details": test_vs_p4,
        }
    }

    out_file = os.path.join(output_dir, f"selection_benchmark_seed_{args.seed}.json")
    with open(out_file, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"\n[Saved] Selection Benchmark Artifacts: {out_file}")


if __name__ == "__main__":
    main()
