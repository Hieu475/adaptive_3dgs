"""Phase 6: Budget-Constrained Selection and Adaptive Greedy Selector.

Extends Phase 5 selection to context-aware and group-aware selection policies:

Policies:
  - NO_OP: S = ∅
  - RANDOM: Random permutation under budget
  - ERROR_ONLY: Rank by rgb_error + depth_error
  - ERROR_INFLUENCE: Rank by error * attribution mass
  - BINARY: High-error thresholded candidates
  - HEURISTIC: Knapsack heuristic efficiency (Importance / Cost)
  - PHASE4_LEARNED: Pointwise TwoHeadMLP predicted utility U_hat = f(s_i)
  - ORACLE_REFERENCE: Ground truth pointwise marginal utility reference
  - PHASE6_STATIC: Context model with S = ∅ (static 1-pass ranking)
  - PHASE6_ADAPTIVE (OURS): Adaptive Greedy with dynamic context S_t re-ranking
  - ORACLE_CONDITIONAL: Ground truth conditional oracle adaptive greedy

Fairness Contract:
  All policies face the exact same compute budget B and safety factor alpha:
      sum_{i in S_B} (alpha * C_i) <= B
"""
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from research.phase5_selection import (
    SelectionResult,
    PolicyName as Phase5PolicyName,
    map_candidate_to_active_index,
    select_budget_constrained_subset as select_phase5_subset,
)
from research.phase6_context import (
    ContextConfig,
    build_full_context,
    build_full_context_batch,
    build_selected_context,
    PHASE6_FEATURE_DIM,
    SELF_SLICE,
    NEIGHBOR_SLICE,
    OVERLAP_SLICE,
    SELECTED_SLICE,
)
from research.phase6_model import FrozenContextPredictor


class Phase6PolicyName(str, Enum):
    NO_OP = "no_op"
    RANDOM = "random"
    ERROR_ONLY = "error_only"
    ERROR_INFLUENCE = "error_influence"
    BINARY = "binary"
    HEURISTIC = "heuristic"
    PHASE4_LEARNED = "phase4_learned"
    LEARNED_UTILITY = "learned_utility"  # alias for phase4_learned
    ORACLE_REFERENCE = "oracle_reference"
    PHASE6_STATIC = "phase6_static"
    PHASE6_ADAPTIVE = "phase6_adaptive"
    ORACLE_CONDITIONAL = "oracle_conditional"


def adaptive_greedy_select(
    candidates: List[Dict[str, Any]],
    positions: torch.Tensor,
    all_features: np.ndarray,
    predictor: FrozenContextPredictor,
    budget: float,
    safety_factor: float = 1.0,
    reject_negative: bool = True,
    cost_key: str = "measured_trial_cost_ms",
    pred_cost_key: str = "predicted_delta_t",
    use_predicted_cost: bool = True,
    contrib_indices: Optional[torch.Tensor] = None,
    contrib_weights: Optional[torch.Tensor] = None,
    context_config: Optional[ContextConfig] = None,
) -> SelectionResult:
    """Adaptive greedy budget-constrained selection for Phase 6.

    Algorithm:
      S_0 = ∅
      remaining_pool = candidates
      while budget_remaining > 0 and remaining_pool not empty:
          for each candidate i in remaining_pool:
              build context features [self, neighbor, overlap, selected(S_t)]
              predict U_hat(i | S_t) and C_hat(i | S_t)
          i* = argmax U_hat(i | S_t)
          if U_hat(i*) <= 0 and reject_negative:
              break (all remaining have non-positive utility)
          if cost(S_t ∪ {i*}) > budget:
              remove i* from remaining_pool (does not fit)
              continue
          S_{t+1} = S_t ∪ {i*}
          remove i* from remaining_pool
      return S
    """
    t0 = time.perf_counter()
    n = len(candidates)
    policy_str = Phase6PolicyName.PHASE6_ADAPTIVE.value

    if n == 0 or budget <= 0.0:
        return SelectionResult(
            policy=policy_str,
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

    cand_ids = [c.get("gaussian_id", i) for i, c in enumerate(candidates)]
    nom_costs = np.array([
        float(c.get(cost_key, c.get("modeled_marginal_cost_us", 1.0)))
        for c in candidates
    ], dtype=np.float32)

    config = context_config or ContextConfig()

    # Pre-build static contexts (self, neighbor, overlap) for all candidates
    # This avoids recomputing KNN and attribution in every adaptive step
    active_indices = [c.get("gaussian_id", i) for i, c in enumerate(candidates)]
    
    # Pre-compute static feature slices [0:24]
    precomputed_static = []
    for i, c in enumerate(candidates):
        if "full_feature_vector" in c and len(c["full_feature_vector"]) >= 24:
            precomputed_static.append(np.asarray(c["full_feature_vector"], dtype=np.float32)[:24].copy())
        else:
            cand_idx = active_indices[i]
            ctx = build_full_context(
                positions=positions,
                candidate_idx=cand_idx,
                all_features=all_features,
                selected_indices=[],
                contrib_indices=contrib_indices,
                contrib_weights=contrib_weights,
                config=config,
            )
            # Slices: self(11) + neighbor(8) + overlap(5) = 24
            precomputed_static.append(ctx["full_vector"][:24].copy())

    selected_cand_indices: List[int] = []
    selected_gaussian_ids: List[int] = []
    remaining_pool = list(range(n))
    cur_scheduled_cost = 0.0
    cur_pred_cost = 0.0
    cur_nom_cost = 0.0
    rejected_neg = 0

    device = predictor.device

    while remaining_pool and cur_scheduled_cost < budget:
        # Build batch of 32-dim features for all remaining candidates
        batch_vectors = []
        for c_idx in remaining_pool:
            g_id = active_indices[c_idx]
            # Compute dynamic selected-set features given current S_t
            sel_ctx = build_selected_context(
                positions=positions,
                candidate_idx=g_id,
                selected_indices=selected_gaussian_ids,
                all_features=all_features,
                total_budget=budget,
                current_cost=cur_scheduled_cost,
            )
            # Combine static 24 + dynamic 8 = 32
            sel_vec = np.array([sel_ctx[name] for name in [
                "selected_count", "selected_mean_rgb_error", "selected_mean_depth_error",
                "selected_mean_influence", "selected_spatial_density",
                "candidate_selected_overlap", "candidate_selected_distance",
                "selected_budget_fraction"
            ]], dtype=np.float32)

            full_vec = np.concatenate([precomputed_static[c_idx], sel_vec])
            batch_vectors.append(full_vec)

        batch_tensor = torch.tensor(np.stack(batch_vectors), dtype=torch.float32, device=device)
        preds = predictor.predict(batch_tensor)

        pred_u = preds["utility"].detach().cpu().numpy()
        pred_t = preds["delta_t"].detach().cpu().numpy()

        # Sort remaining candidates by predicted conditional utility
        sorted_order = np.argsort(-pred_u)
        best_pos = sorted_order[0]
        best_cand_idx = remaining_pool[best_pos]
        best_u = pred_u[best_pos]
        best_t = pred_t[best_pos]

        # Negative utility rejection
        if reject_negative and best_u <= 0.0:
            # All remaining candidates have non-positive utility
            rejected_neg += len(remaining_pool)
            break

        # Cost packing check
        unit_cost = best_t if use_predicted_cost else nom_costs[best_cand_idx]
        scheduled_cost = unit_cost * float(safety_factor)

        if cur_scheduled_cost + scheduled_cost <= budget + 1e-7:
            # Fits in budget -> select it
            selected_cand_indices.append(best_cand_idx)
            selected_gaussian_ids.append(active_indices[best_cand_idx])
            cur_scheduled_cost += scheduled_cost
            cur_pred_cost += best_t
            cur_nom_cost += nom_costs[best_cand_idx]
            remaining_pool.pop(best_pos)
        else:
            # Does not fit -> remove only this candidate from pool and try next best
            remaining_pool.pop(best_pos)

    sel_time_ms = (time.perf_counter() - t0) * 1000.0
    scheduled_violation = max(0.0, cur_scheduled_cost - budget)

    return SelectionResult(
        policy=policy_str,
        selected_indices=selected_cand_indices,
        selected_gaussian_ids=selected_gaussian_ids,
        k_count=len(selected_cand_indices),
        predicted_cost=float(cur_pred_cost),
        scheduled_cost=float(cur_scheduled_cost),
        nominal_cost=float(cur_nom_cost),
        budget=float(budget),
        safety_factor=float(safety_factor),
        selection_time_ms=float(sel_time_ms),
        rejected_negative_count=int(rejected_neg),
        scheduled_budget_violation=float(scheduled_violation),
        is_scheduled_violation=bool(scheduled_violation > 1e-5),
    )


def static_context_select(
    candidates: List[Dict[str, Any]],
    positions: torch.Tensor,
    all_features: np.ndarray,
    predictor: FrozenContextPredictor,
    budget: float,
    safety_factor: float = 1.0,
    reject_negative: bool = True,
    cost_key: str = "measured_trial_cost_ms",
    use_predicted_cost: bool = True,
    contrib_indices: Optional[torch.Tensor] = None,
    contrib_weights: Optional[torch.Tensor] = None,
    context_config: Optional[ContextConfig] = None,
) -> SelectionResult:
    """Static 1-pass context selection (Phase 6 Static).

    Evaluates all candidates with initial context S = ∅, ranks them once,
    and greedily packs until budget exhausted.
    """
    t0 = time.perf_counter()
    n = len(candidates)
    policy_str = Phase6PolicyName.PHASE6_STATIC.value

    if n == 0 or budget <= 0.0:
        return SelectionResult(
            policy=policy_str,
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

    active_indices = [c.get("gaussian_id", i) for i, c in enumerate(candidates)]
    nom_costs = np.array([
        float(c.get(cost_key, c.get("modeled_marginal_cost_us", 1.0)))
        for c in candidates
    ], dtype=np.float32)

    config = context_config or ContextConfig()

    # Build S = ∅ context vectors for all candidates
    if all("full_feature_vector" in c and len(c["full_feature_vector"]) == PHASE6_FEATURE_DIM for c in candidates):
        feature_matrix = np.stack([np.asarray(c["full_feature_vector"], dtype=np.float32) for c in candidates])
    else:
        ctx_dict = build_full_context_batch(
            positions=positions,
            candidate_indices=active_indices,
            all_features=all_features,
            selected_indices=[],
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
            config=config,
        )
        feature_matrix = np.stack([ctx_dict[g_id]["full_vector"] for g_id in active_indices])

    device = predictor.device
    batch_t = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
    preds = predictor.predict(batch_t)

    pred_u = preds["utility"].detach().cpu().numpy()
    pred_t = preds["delta_t"].detach().cpu().numpy()

    packing_costs = pred_t if use_predicted_cost else nom_costs
    scheduled_cand_costs = packing_costs * float(safety_factor)

    order = np.argsort(-pred_u)
    selected_indices = []
    cur_scheduled_cost = 0.0
    rejected_neg = 0

    for idx in order:
        u_val = pred_u[idx]
        if reject_negative and u_val <= 0.0:
            rejected_neg += 1
            continue
        c = scheduled_cand_costs[idx]
        if cur_scheduled_cost + c <= budget + 1e-7:
            selected_indices.append(int(idx))
            cur_scheduled_cost += c

    sel_time_ms = (time.perf_counter() - t0) * 1000.0
    sel_ids = [active_indices[i] for i in selected_indices]
    tot_pred_cost = float(pred_t[selected_indices].sum()) if selected_indices else 0.0
    tot_sched_cost = float(scheduled_cand_costs[selected_indices].sum()) if selected_indices else 0.0
    tot_nom_cost = float(nom_costs[selected_indices].sum()) if selected_indices else 0.0
    scheduled_violation = max(0.0, tot_sched_cost - budget)

    return SelectionResult(
        policy=policy_str,
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


def select_phase6_subset(
    candidates: List[Dict[str, Any]],
    policy: Union[str, Phase6PolicyName],
    budget: float,
    seed: int = 42,
    safety_factor: float = 1.0,
    reject_negative: bool = True,
    use_predicted_cost: bool = True,
    positions: Optional[torch.Tensor] = None,
    all_features: Optional[np.ndarray] = None,
    phase6_predictor: Optional[FrozenContextPredictor] = None,
    contrib_indices: Optional[torch.Tensor] = None,
    contrib_weights: Optional[torch.Tensor] = None,
    cost_key: str = "measured_trial_cost_ms",
    pred_cost_key: str = "predicted_delta_t",
    pred_utility_key: str = "predicted_utility",
    oracle_utility_key: str = "oracle_utility_joint_global",
) -> SelectionResult:
    """Unified selection entry point for all Phase 5 and Phase 6 policies.

    Dispatches to:
      - Phase 6 Adaptive Greedy if policy is 'phase6_adaptive'
      - Phase 6 Static Context if policy is 'phase6_static'
      - Phase 5 standard selection for all baseline policies
    """
    p_str = str(policy.value if hasattr(policy, "value") else policy).lower()

    if p_str == Phase6PolicyName.PHASE6_ADAPTIVE.value:
        if positions is None or all_features is None or phase6_predictor is None:
            raise ValueError(
                "policy='phase6_adaptive' requires positions, all_features, and phase6_predictor."
            )
        return adaptive_greedy_select(
            candidates=candidates,
            positions=positions,
            all_features=all_features,
            predictor=phase6_predictor,
            budget=budget,
            safety_factor=safety_factor,
            reject_negative=reject_negative,
            cost_key=cost_key,
            pred_cost_key=pred_cost_key,
            use_predicted_cost=use_predicted_cost,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
        )

    elif p_str == Phase6PolicyName.PHASE6_STATIC.value:
        if positions is None or all_features is None or phase6_predictor is None:
            raise ValueError(
                "policy='phase6_static' requires positions, all_features, and phase6_predictor."
            )
        return static_context_select(
            candidates=candidates,
            positions=positions,
            all_features=all_features,
            predictor=phase6_predictor,
            budget=budget,
            safety_factor=safety_factor,
            reject_negative=reject_negative,
            cost_key=cost_key,
            use_predicted_cost=use_predicted_cost,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
        )

    else:
        # Map phase4_learned to learned_utility for Phase 5 selection
        p5_policy = p_str
        if p_str == "phase4_learned":
            p5_policy = "learned_utility"

        return select_phase5_subset(
            candidates=candidates,
            policy=p5_policy,
            budget=budget,
            seed=seed,
            reject_negative=reject_negative,
            use_predicted_cost=use_predicted_cost,
            safety_factor=safety_factor,
            cost_key=cost_key,
            pred_cost_key=pred_cost_key,
            pred_utility_key=pred_utility_key,
            oracle_utility_key=oracle_utility_key,
        )
