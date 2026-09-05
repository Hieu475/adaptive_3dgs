"""Phase 5 Budget-Constrained Candidate Selection Policies.

Implements equal-compute subset selection under budget B:
  argmax_{S_B} sum_{i in S_B} Gain(i)  s.t.  sum_{i in S_B} C_i <= B

All policies face the exact same compute budget B and the exact same cost constraint.

Policies:
  - NO_OP: No optimization (empty selection, baseline check).
  - RANDOM: Random uniform permutation under budget B.
  - ERROR_ONLY: Rank by photometric + geometric error under budget B.
  - ERROR_INFLUENCE: Rank by error * attribution mass under budget B.
  - BINARY: High-error thresholded candidates under budget B.
  - HEURISTIC: Knapsack heuristic efficiency (Importance / Cost) under budget B.
  - LEARNED_UTILITY (OURS): TwoHeadMLP predicted utility (reject U_hat <= 0, rank U_hat) under budget B.
  - ORACLE_REFERENCE: Ground truth marginal utility reference (reject U* <= 0, rank U*) under budget B.
"""
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class PolicyName(str, Enum):
    NO_OP = "no_op"
    RANDOM = "random"
    ERROR_ONLY = "error_only"
    ERROR_INFLUENCE = "error_influence"
    BINARY = "binary"
    HEURISTIC = "heuristic"
    LEARNED_UTILITY = "learned_utility"
    ORACLE_REFERENCE = "oracle_reference"
    ORACLE = "oracle"  # Backwards compatibility alias for ORACLE_REFERENCE


@dataclass
class SelectionResult:
    policy: str
    selected_indices: List[int]       # Index relative to candidates list
    selected_gaussian_ids: List[int]  # Actual Gaussian IDs in 3DGS model
    k_count: int
    predicted_cost: float             # Sum of predicted costs (\hat{C})
    scheduled_cost: float             # Sum of safety-adjusted costs (\alpha * \hat{C} or \alpha * C)
    nominal_cost: float               # Sum of reference/measured costs (C)
    budget: float
    safety_factor: float
    selection_time_ms: float
    rejected_negative_count: int = 0
    scheduled_budget_violation: float = 0.0  # max(0, scheduled_cost - budget)
    is_scheduled_violation: bool = False


def select_budget_constrained_subset(
    candidates: List[Dict[str, Any]],
    policy: Union[str, PolicyName],
    budget: float,
    seed: int = 42,
    reject_negative: bool = True,
    use_predicted_cost: bool = True,
    safety_factor: float = 1.0,
    cost_key: str = "measured_trial_cost_ms",
    pred_cost_key: str = "predicted_delta_t",
    pred_utility_key: str = "predicted_utility",
    oracle_utility_key: str = "oracle_utility_joint_global",
) -> SelectionResult:
    """Selects candidate Gaussians subject to a hard compute budget B.

    Fairness contract:
      All policies are given the exact same budget B and candidate cost definition:
          sum_{i in S_B} (cost_i * safety_factor) <= B

    Args:
        candidates: List of candidate dictionaries.
        policy: Selection policy name.
        budget: Compute budget threshold.
        seed: Random seed for stochastic policies (RANDOM).
        reject_negative: If True, rejects candidates with non-positive utility.
        use_predicted_cost: If True, uses predicted_delta_t for all policies; if False, uses nominal trial cost.
        safety_factor: Safety multiplier alpha (e.g. 1.0, 1.05, 1.10, 1.20).
        cost_key: Key for reference/measured cost in candidate dict.
        pred_cost_key: Key for model-predicted cost in candidate dict.
        pred_utility_key: Key for model-predicted utility in candidate dict.
        oracle_utility_key: Key for ground-truth oracle utility in candidate dict.

    Returns:
        SelectionResult object containing selected indices, IDs, costs, and timings.
    """
    n = len(candidates)
    p_str = str(policy.value if hasattr(policy, "value") else policy).lower()

    if n == 0 or budget <= 0.0 or p_str in (PolicyName.NO_OP.value, "no_op"):
        return SelectionResult(
            policy=p_str,
            selected_indices=[],
            selected_gaussian_ids=[],
            k_count=0,
            predicted_cost=0.0,
            scheduled_cost=0.0,
            nominal_cost=0.0,
            budget=float(budget),
            safety_factor=float(safety_factor),
            selection_time_ms=0.0,
            rejected_negative_count=0,
            scheduled_budget_violation=0.0,
            is_scheduled_violation=False,
        )

    t0 = time.perf_counter()

    # Extract candidate IDs and costs
    cand_ids = [c.get("gaussian_id", i) for i, c in enumerate(candidates)]
    
    # Reference costs (nominal trial cost)
    nom_costs = np.array([
        float(c.get(cost_key, c.get("modeled_marginal_cost_us", 1.0)))
        for c in candidates
    ], dtype=np.float32)

    # Predicted costs
    pred_costs = np.array([
        float(c.get(pred_cost_key, nom_costs[i]))
        for i, c in enumerate(candidates)
    ], dtype=np.float32)

    # Unified packing cost for all policies
    raw_packing_costs = pred_costs if use_predicted_cost else nom_costs
    scheduled_cand_costs = raw_packing_costs * float(safety_factor)

    rejected_neg = 0
    selected_indices: List[int] = []
    cur_scheduled_cost = 0.0

    if p_str in (PolicyName.RANDOM.value, "random"):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        for idx in perm:
            c = scheduled_cand_costs[idx]
            if cur_scheduled_cost + c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_scheduled_cost += c

    elif p_str in (PolicyName.ERROR_ONLY.value, "error_only", "error"):
        # Score = rgb_error + depth_error
        err_scores = np.array([
            float(c.get("features", {}).get("rgb_error", 0.0)) +
            float(c.get("features", {}).get("depth_error", 0.0))
            for c in candidates
        ], dtype=np.float32)
        order = np.argsort(-err_scores)
        for idx in order:
            c = scheduled_cand_costs[idx]
            if cur_scheduled_cost + c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_scheduled_cost += c

    elif p_str in (PolicyName.ERROR_INFLUENCE.value, "error_influence"):
        # Score = (rgb_error + depth_error) * influence_mass
        scores = np.array([
            (float(c.get("features", {}).get("rgb_error", 0.0)) +
             float(c.get("features", {}).get("depth_error", 0.0))) *
            float(c.get("features", {}).get("influence_mass", c.get("influence_mass", 1.0)))
            for c in candidates
        ], dtype=np.float32)
        order = np.argsort(-scores)
        for idx in order:
            c = scheduled_cand_costs[idx]
            if cur_scheduled_cost + c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_scheduled_cost += c

    elif p_str in (PolicyName.BINARY.value, "binary"):
        # Threshold at median error
        err_scores = np.array([
            float(c.get("features", {}).get("rgb_error", 0.0)) +
            float(c.get("features", {}).get("depth_error", 0.0))
            for c in candidates
        ], dtype=np.float32)
        median_err = float(np.median(err_scores))
        order = np.argsort(-err_scores)
        for idx in order:
            if err_scores[idx] >= median_err:
                c = scheduled_cand_costs[idx]
                if cur_scheduled_cost + c <= budget + 1e-7:
                    selected_indices.append(int(idx))
                    cur_scheduled_cost += c

    elif p_str in (PolicyName.HEURISTIC.value, "heuristic", "knapsack"):
        # Knapsack heuristic: predicted_importance / cost
        heur_eff = np.array([
            float(c.get("predicted_importance", 1.0)) / max(1e-4, raw_packing_costs[i])
            for i, c in enumerate(candidates)
        ], dtype=np.float32)
        order = np.argsort(-heur_eff)
        for idx in order:
            c = scheduled_cand_costs[idx]
            if cur_scheduled_cost + c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_scheduled_cost += c

    elif p_str in (PolicyName.LEARNED_UTILITY.value, "learned_utility", "learned", "ours"):
        # Two-Head learned utility: \hat{U}_i = \hat{\Delta Q}_i / (\hat{\Delta T}_i + \epsilon)
        learned_u = np.array([
            float(c.get(pred_utility_key, 0.0)) for c in candidates
        ], dtype=np.float32)

        order = np.argsort(-learned_u)
        for idx in order:
            u_val = learned_u[idx]
            if reject_negative and u_val <= 0.0:
                rejected_neg += 1
                continue
            c = scheduled_cand_costs[idx]
            if cur_scheduled_cost + c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_scheduled_cost += c

    elif p_str in (PolicyName.ORACLE_REFERENCE.value, PolicyName.ORACLE.value, "oracle_reference", "oracle"):
        # Oracle Marginal-Utility Reference: U*_i = \Delta Q*_i / (\Delta T*_i + \epsilon)
        oracle_u = np.array([
            float(c.get(oracle_utility_key, c.get("oracle_utility", 0.0)))
            for c in candidates
        ], dtype=np.float32)

        order = np.argsort(-oracle_u)
        for idx in order:
            u_val = oracle_u[idx]
            if reject_negative and u_val <= 0.0:
                rejected_neg += 1
                continue
            c = scheduled_cand_costs[idx]
            if cur_scheduled_cost + c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_scheduled_cost += c

    else:
        raise ValueError(f"Unknown policy: {policy}")

    sel_time_ms = (time.perf_counter() - t0) * 1000.0

    sel_ids = [cand_ids[i] for i in selected_indices]
    tot_pred_cost = float(pred_costs[selected_indices].sum()) if selected_indices else 0.0
    tot_sched_cost = float(scheduled_cand_costs[selected_indices].sum()) if selected_indices else 0.0
    tot_nom_cost = float(nom_costs[selected_indices].sum()) if selected_indices else 0.0
    scheduled_violation = max(0.0, tot_sched_cost - budget)

    return SelectionResult(
        policy=p_str,
        selected_indices=selected_indices,
        selected_gaussian_ids=sel_ids,
        k_count=len(selected_indices),
        predicted_cost=tot_pred_cost,
        scheduled_cost=tot_sched_cost,
        nominal_cost=tot_nom_cost,
        budget=float(budget),
        safety_factor=float(safety_factor),
        selection_time_ms=float(sel_time_ms),
        rejected_negative_count=int(rejected_neg),
        scheduled_budget_violation=float(scheduled_violation),
        is_scheduled_violation=bool(scheduled_violation > 1e-5),
    )
