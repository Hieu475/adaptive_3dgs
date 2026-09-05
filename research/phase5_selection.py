"""Phase 5 Budget-Constrained Candidate Selection Policies.

Implements equal-compute subset selection under budget B:
  argmax_{S_B} sum_{i in S_B} Gain(i)  s.t.  sum_{i in S_B} C_i <= B

Policies:
  - RANDOM: Random uniform permutation under budget B.
  - ERROR_ONLY: Rank by photometric + geometric error under budget B.
  - ERROR_INFLUENCE: Rank by error * attribution mass under budget B.
  - BINARY: High-error thresholded candidates under budget B.
  - HEURISTIC: Knapsack heuristic efficiency (Importance / Cost) under budget B.
  - LEARNED_UTILITY (OURS): TwoHeadMLP predicted utility (reject U_hat <= 0, rank U_hat) under budget B.
  - ORACLE (REFERENCE): Ground truth marginal utility (reject U* <= 0, rank U*) under budget B.
"""
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class PolicyName(str, Enum):
    RANDOM = "random"
    ERROR_ONLY = "error_only"
    ERROR_INFLUENCE = "error_influence"
    BINARY = "binary"
    HEURISTIC = "heuristic"
    LEARNED_UTILITY = "learned_utility"
    ORACLE = "oracle"


@dataclass
class SelectionResult:
    policy: str
    selected_indices: List[int]       # Index relative to candidates list
    selected_gaussian_ids: List[int]  # Actual Gaussian IDs in 3DGS model
    k_count: int
    predicted_cost: float             # Sum of predicted costs (\hat{C})
    scheduled_cost: float             # Sum of safety-adjusted costs (\alpha * \hat{C})
    nominal_cost: float               # Sum of reference/measured costs (C)
    budget: float
    safety_factor: float
    selection_time_ms: float
    rejected_negative_count: int = 0
    budget_violation: float = 0.0     # max(0, nominal_cost - budget)


def select_budget_constrained_subset(
    candidates: List[Dict[str, Any]],
    policy: Union[str, PolicyName],
    budget: float,
    seed: int = 42,
    reject_negative: bool = True,
    use_predicted_cost: bool = False,
    safety_factor: float = 1.0,
    cost_key: str = "measured_trial_cost_ms",
    pred_cost_key: str = "predicted_delta_t",
    pred_utility_key: str = "predicted_utility",
    oracle_utility_key: str = "oracle_utility_joint_global",
) -> SelectionResult:
    """Selects candidate Gaussians subject to a hard compute budget B.

    Args:
        candidates: List of candidate dictionaries.
        policy: Selection policy name.
        budget: Compute budget threshold.
        seed: Random seed for stochastic policies (RANDOM).
        reject_negative: If True, rejects candidates with non-positive utility.
        cost_key: Key for reference/measured cost in candidate dict.
        pred_cost_key: Key for model-predicted cost in candidate dict.
        pred_utility_key: Key for model-predicted utility in candidate dict.
        oracle_utility_key: Key for ground-truth oracle utility in candidate dict.

    Returns:
        SelectionResult object containing selected indices, IDs, costs, and timings.
    """
    n = len(candidates)
    if n == 0 or budget <= 0.0:
        return SelectionResult(
            policy=str(policy),
            selected_indices=[],
            selected_gaussian_ids=[],
            k_count=0,
            predicted_cost=0.0,
            scheduled_cost=0.0,
            nominal_cost=0.0,
            budget=budget,
            safety_factor=float(safety_factor),
            selection_time_ms=0.0,
            rejected_negative_count=0,
            budget_violation=0.0,
        )

    p_str = str(policy.value if hasattr(policy, "value") else policy).lower()
    t0 = time.perf_counter()

    # Extract candidate IDs and costs
    cand_ids = [c.get("gaussian_id", i) for i, c in enumerate(candidates)]
    
    # Reference costs (nominal cost)
    nom_costs = np.array([
        float(c.get(cost_key, c.get("modeled_marginal_cost_us", 1.0)))
        for c in candidates
    ], dtype=np.float32)

    # Predicted costs
    pred_costs = np.array([
        float(c.get(pred_cost_key, nom_costs[i]))
        for i, c in enumerate(candidates)
    ], dtype=np.float32)

    rejected_neg = 0
    selected_indices: List[int] = []

    if p_str in (PolicyName.RANDOM.value, "random"):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        cur_cost = 0.0
        for idx in perm:
            c = nom_costs[idx]
            if cur_cost + c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_cost += c

    elif p_str in (PolicyName.ERROR_ONLY.value, "error_only", "error"):
        # Score = rgb_error + depth_error
        err_scores = np.array([
            float(c.get("features", {}).get("rgb_error", 0.0)) +
            float(c.get("features", {}).get("depth_error", 0.0))
            for c in candidates
        ], dtype=np.float32)
        order = np.argsort(-err_scores)
        cur_cost = 0.0
        for idx in order:
            c = nom_costs[idx]
            if cur_cost + c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_cost += c

    elif p_str in (PolicyName.ERROR_INFLUENCE.value, "error_influence"):
        # Score = (rgb_error + depth_error) * influence_mass
        scores = np.array([
            (float(c.get("features", {}).get("rgb_error", 0.0)) +
             float(c.get("features", {}).get("depth_error", 0.0))) *
            float(c.get("features", {}).get("influence_mass", c.get("influence_mass", 1.0)))
            for c in candidates
        ], dtype=np.float32)
        order = np.argsort(-scores)
        cur_cost = 0.0
        for idx in order:
            c = nom_costs[idx]
            if cur_cost + c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_cost += c

    elif p_str in (PolicyName.BINARY.value, "binary"):
        # Threshold at median error
        err_scores = np.array([
            float(c.get("features", {}).get("rgb_error", 0.0)) +
            float(c.get("features", {}).get("depth_error", 0.0))
            for c in candidates
        ], dtype=np.float32)
        median_err = float(np.median(err_scores))
        order = np.argsort(-err_scores)
        cur_cost = 0.0
        for idx in order:
            if err_scores[idx] >= median_err:
                c = nom_costs[idx]
                if cur_cost + c <= budget + 1e-7:
                    selected_indices.append(int(idx))
                    cur_cost += c

    elif p_str in (PolicyName.HEURISTIC.value, "heuristic", "knapsack"):
        # Knapsack heuristic: predicted_importance / cost
        heur_eff = np.array([
            float(c.get("predicted_utility",
                        float(c.get("predicted_importance", 1.0)) / max(1e-4, nom_costs[i])))
            for i, c in enumerate(candidates)
        ], dtype=np.float32)
        order = np.argsort(-heur_eff)
        cur_cost = 0.0
        for idx in order:
            c = nom_costs[idx]
            if cur_cost + c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_cost += c

    elif p_str in (PolicyName.LEARNED_UTILITY.value, "learned_utility", "learned", "ours"):
        # Two-Head learned utility: \hat{U}_i = \hat{\Delta Q}_i / \hat{\Delta T}_i
        learned_u = np.array([
            float(c.get(pred_utility_key, 0.0)) for c in candidates
        ], dtype=np.float32)

        order = np.argsort(-learned_u)
        cur_cost = 0.0
        for idx in order:
            u_val = learned_u[idx]
            if reject_negative and u_val <= 0.0:
                rejected_neg += 1
                continue
            raw_c = pred_costs[idx] if use_predicted_cost else nom_costs[idx]
            sched_c = raw_c * safety_factor
            if cur_cost + sched_c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_cost += sched_c

    elif p_str in (PolicyName.ORACLE.value, "oracle", "oracle_utility"):
        # Oracle upper bound: U*_i = \Delta Q*_i / \Delta T*_i
        oracle_u = np.array([
            float(c.get(oracle_utility_key, c.get("oracle_utility", 0.0)))
            for c in candidates
        ], dtype=np.float32)

        order = np.argsort(-oracle_u)
        cur_cost = 0.0
        for idx in order:
            u_val = oracle_u[idx]
            if reject_negative and u_val <= 0.0:
                rejected_neg += 1
                continue
            c = nom_costs[idx]
            if cur_cost + c <= budget + 1e-7:
                selected_indices.append(int(idx))
                cur_cost += c

    else:
        raise ValueError(f"Unknown policy: {policy}")

    sel_time_ms = (time.perf_counter() - t0) * 1000.0

    sel_ids = [cand_ids[i] for i in selected_indices]
    tot_pred_cost = float(pred_costs[selected_indices].sum()) if selected_indices else 0.0
    tot_sched_cost = float(tot_pred_cost * safety_factor) if use_predicted_cost else float(nom_costs[selected_indices].sum() * safety_factor if selected_indices else 0.0)
    tot_nom_cost = float(nom_costs[selected_indices].sum()) if selected_indices else 0.0
    violation = max(0.0, tot_nom_cost - budget)

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
        budget_violation=float(violation),
    )
