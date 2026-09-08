"""Phase 6 Evaluator (research/phase6_evaluator.py).

Measures ACTUAL joint group gain via real GPU selective optimization,
fixing the C1 critical bug where run_phase6_selection.py summed individual
delta_q values instead of running true joint optimization.

Protocol for measuring realized group gain ΔQ(S_B):
  1. snapshot_state()
  2. measure Q(∅) = quality at current model state (implicit in optimize_gaussian_group)
  3. optimize_gaussian_group(selected_ids, n_steps, rgb, depth, influence_mask)
  4. measure Q(S_B) = quality after joint optimization
  5. restore_state(snapshot)
  6. ΔQ_realized = Q(S_B) - Q(∅)  ← NOT sum of individual ΔQ_i

This module also provides evaluate_policy() which dispatches any Phase 6
policy (including Phase 5 baselines) through the same fair evaluation pipeline.
"""
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from research.phase5_selection import (
    PolicyName as Phase5PolicyName,
    SelectionResult,
    map_candidate_to_active_index,
)
from research.phase6_selection import (
    Phase6PolicyName,
    select_phase6_subset,
)
from research.phase6_model import FrozenContextPredictor
from research.utility_predictor import FrozenUtilityPredictor
from research.scheduler_metrics import (
    compute_ose,
    compute_regret,
    compute_selection_regret,
    compute_policy_efficiency,
    compute_cost_metrics,
)


def evaluate_selected_group(
    oracle_engine: Any,
    selected_gaussian_ids: List[int],
    rgb_gt: torch.Tensor,
    depth_gt: torch.Tensor,
    contrib_indices: torch.Tensor,
    contrib_weights: torch.Tensor,
    n_opt_steps: int = 5,
) -> Dict[str, float]:
    """Measures ACTUAL joint group optimization gain.

    This is the core C1 fix: instead of summing individual ΔQ_i,
    we run real selective optimization on the entire selected set S_B
    and measure the actual quality change.

    Protocol:
      1. snapshot original model state
      2. compute influence_mask for the group S_B
      3. optimize_gaussian_group(S_B, n_steps, rgb, depth, influence_mask)
      4. ΔQ_realized = result["delta_quality_global"] = Q(S_B) - Q(∅)
      5. restore original model state

    Args:
        oracle_engine: OracleUtilityExperiment instance with live pipeline
        selected_gaussian_ids: List of active Gaussian indices to optimize jointly
        rgb_gt: Ground truth RGB image (H, W, 3)
        depth_gt: Ground truth depth map (H, W)
        contrib_indices: Attribution indices (H, W, K) from render_with_attribution
        contrib_weights: Attribution weights (H, W, K) from render_with_attribution
        n_opt_steps: Number of optimization steps (default: 5, locked by protocol)

    Returns:
        Dictionary with actual joint group metrics:
        - q_baseline: Q(∅) global quality before optimization
        - q_after_joint_opt: Q(S_B) global quality after joint optimization
        - delta_q_realized: Q(S_B) - Q(∅)  ← THE CORRECT METRIC
        - delta_psnr_realized: PSNR improvement
        - delta_depth_realized: Depth L1 gain
        - actual_opt_cost_ms: Wall-clock time for joint optimization
        - n_selected: Number of Gaussians optimized jointly
        - influence_pixel_count: Number of pixels influenced by S_B
    """
    n_selected = len(selected_gaussian_ids)

    # Empty selection → NO-OP baseline
    if n_selected == 0:
        return {
            "q_baseline": 0.0,
            "q_after_joint_opt": 0.0,
            "delta_q_realized": 0.0,
            "delta_psnr_realized": 0.0,
            "delta_depth_realized": 0.0,
            "actual_opt_cost_ms": 0.0,
            "n_selected": 0,
            "influence_pixel_count": 0,
        }

    # Compute influence mask for the entire group
    influence_mask = oracle_engine._get_influence_mask(
        selected_gaussian_ids, contrib_indices, contrib_weights,
    )

    # Snapshot → optimize → measure → restore
    snapshot = oracle_engine.snapshot_state()
    try:
        device = rgb_gt.device
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt_res = oracle_engine.optimize_gaussian_group(
            indices=selected_gaussian_ids,
            n_steps=n_opt_steps,
            rgb=rgb_gt,
            depth=depth_gt,
            influence_mask=influence_mask,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        actual_opt_cost_ms = (time.perf_counter() - t0) * 1000.0

        # Extract metrics from the optimization result
        delta_q_realized = float(opt_res["delta_quality_global"])
        delta_psnr_realized = float(opt_res.get("delta_psnr_global", 0.0))

        # Depth gain: before - after (positive = improvement)
        depth_before = float(opt_res.get("depth_l1_global_before", 0.0))
        depth_after = float(opt_res.get("depth_l1_global_after", 0.0))
        delta_depth_realized = depth_before - depth_after

        # Global quality measurements
        q_baseline = float(opt_res.get("psnr_global_before", 0.0))
        q_after = float(opt_res.get("psnr_global_after", 0.0))

    finally:
        oracle_engine.restore_state(snapshot)

    return {
        "q_baseline": q_baseline,
        "q_after_joint_opt": q_after,
        "delta_q_realized": delta_q_realized,
        "delta_psnr_realized": delta_psnr_realized,
        "delta_depth_realized": delta_depth_realized,
        "actual_opt_cost_ms": actual_opt_cost_ms,
        "n_selected": n_selected,
        "influence_pixel_count": int(influence_mask.sum().item()),
    }


def evaluate_pairwise_interaction(
    oracle_engine: Any,
    idx_i: int,
    idx_j: int,
    rgb_gt: torch.Tensor,
    depth_gt: torch.Tensor,
    contrib_indices: torch.Tensor,
    contrib_weights: torch.Tensor,
    n_opt_steps: int = 5,
) -> Dict[str, float]:
    """Measures pairwise interaction between two Gaussians.

    Computes:
        ΔQ(i)     = Q({i}) - Q(∅)
        ΔQ(j)     = Q({j}) - Q(∅)
        ΔQ({i,j}) = Q({i,j}) - Q(∅)
        I(i,j)    = ΔQ({i,j}) - ΔQ(i) - ΔQ(j)

    If I(i,j) ≈ 0: additive (independent gains)
    If I(i,j) < 0: sub-additive (redundant, overlap)
    If I(i,j) > 0: super-additive (synergistic)

    Returns:
        Dictionary with all individual and joint gains plus interaction.
    """
    # Measure ΔQ(i) alone
    res_i = evaluate_selected_group(
        oracle_engine, [idx_i], rgb_gt, depth_gt,
        contrib_indices, contrib_weights, n_opt_steps,
    )

    # Measure ΔQ(j) alone
    res_j = evaluate_selected_group(
        oracle_engine, [idx_j], rgb_gt, depth_gt,
        contrib_indices, contrib_weights, n_opt_steps,
    )

    # Measure ΔQ({i,j}) jointly
    res_ij = evaluate_selected_group(
        oracle_engine, [idx_i, idx_j], rgb_gt, depth_gt,
        contrib_indices, contrib_weights, n_opt_steps,
    )

    dq_i = res_i["delta_q_realized"]
    dq_j = res_j["delta_q_realized"]
    dq_ij = res_ij["delta_q_realized"]
    interaction = dq_ij - dq_i - dq_j

    # Relative interaction: |I(i,j)| / max(|ΔQ(i)| + |ΔQ(j)|, eps)
    sum_abs = abs(dq_i) + abs(dq_j)
    relative_interaction = abs(interaction) / max(sum_abs, 1e-10)

    return {
        "idx_i": idx_i,
        "idx_j": idx_j,
        "delta_q_i": dq_i,
        "delta_q_j": dq_j,
        "delta_q_ij": dq_ij,
        "interaction": interaction,
        "relative_interaction": relative_interaction,
        "is_sub_additive": interaction < 0,
        "is_super_additive": interaction > 0,
        "cost_i_ms": res_i["actual_opt_cost_ms"],
        "cost_j_ms": res_j["actual_opt_cost_ms"],
        "cost_ij_ms": res_ij["actual_opt_cost_ms"],
    }


class Phase6Evaluator:
    """Evaluates Phase 6 budget-constrained selection policies with actual
    joint group optimization.

    Key differences from Phase 5 evaluator:
    1. Accepts context inputs (positions, all_features, contrib_indices, contrib_weights)
       required for context-aware Phase 6 policies.
    2. Uses evaluate_selected_group() for actual joint optimization
       instead of summing individual ΔQ values.
    3. Supports both Phase 5 baseline policies and Phase 6 context-aware policies.
    """

    def __init__(
        self,
        p6_predictor: Optional[FrozenContextPredictor] = None,
        p4_predictor: Optional[FrozenUtilityPredictor] = None,
        safety_factor: float = 1.0,
        use_predicted_cost: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.p6_predictor = p6_predictor
        self.p4_predictor = p4_predictor
        self.safety_factor = float(safety_factor)
        self.use_predicted_cost = use_predicted_cost

        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        elif p6_predictor is not None:
            self.device = p6_predictor.device
        elif p4_predictor is not None:
            self.device = p4_predictor.device
        else:
            self.device = torch.device("cpu")

    def sync_gpu(self):
        """Synchronize GPU for accurate timing."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def evaluate_policy(
        self,
        policy: str,
        candidates: List[Dict[str, Any]],
        budget: float,
        current_frame: int,
        oracle_engine: Any,
        rgb_gt: torch.Tensor,
        depth_gt: torch.Tensor,
        contrib_indices: torch.Tensor,
        contrib_weights: torch.Tensor,
        positions: torch.Tensor,
        all_features: np.ndarray,
        oracle_reference_gain: Optional[float] = None,
        seed: int = 42,
        reject_negative: bool = True,
        budget_type: str = "relative",
        budget_pct_str: str = "60%",
        n_opt_steps: int = 5,
        t_feat_ms: float = 0.0,
        t_pred_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluates a single policy on the given candidates with actual
        joint group optimization.

        Protocol:
          1. State leakage assertion (cand_frame == current_frame)
          2. Policy dispatch to select_phase6_subset() with full context inputs
          3. Real GPU group optimization via evaluate_selected_group()
          4. Precise cost and violation accounting

        Args:
            policy: Policy name string (any Phase 5 or Phase 6 policy)
            candidates: List of candidate dicts with features, costs, etc.
            budget: Budget value (relative fraction or wall-clock ms)
            current_frame: Current frame index for leakage assertion
            oracle_engine: OracleUtilityExperiment with live pipeline
            rgb_gt: Ground truth RGB (H, W, 3)
            depth_gt: Ground truth depth (H, W)
            contrib_indices: Attribution indices (H, W, K) — REAL, not dummy
            contrib_weights: Attribution weights (H, W, K) — REAL, not dummy
            positions: Real model.positions (N, 3) — REAL, not torch.randn
            all_features: Real canonical features (N, 11) — REAL, not np.zeros
            oracle_reference_gain: Optional oracle ΔQ for computing OSE/regret
            seed: Random seed
            reject_negative: Whether to reject candidates with negative utility
            budget_type: "relative" or "wall_clock"
            budget_pct_str: Budget percentage string for logging
            n_opt_steps: Number of optimization steps (default: 5)
            t_feat_ms: Feature extraction time (for overhead accounting)
            t_pred_ms: Prediction time (for overhead accounting)

        Returns:
            Comprehensive result dictionary with all metrics.
        """
        p_str = str(policy).lower()

        # === 1. Strict State Leakage Assertion ===
        for c in candidates:
            cand_frame = c.get("frame", current_frame)
            assert cand_frame == current_frame, (
                f"State leakage! Candidate frame {cand_frame} != "
                f"current frame {current_frame}"
            )

        # === 2. Selection Step ===
        self.sync_gpu()
        t_sel_0 = time.perf_counter()

        cond_oracle = None
        if p_str in ("oracle_conditional", "phase6_oracle_conditional"):
            from research.phase6_oracle import ConditionalOracleExperiment, ConditionalOracleConfig
            cond_oracle = ConditionalOracleExperiment(
                pipeline=oracle_engine.pipeline,
                config=ConditionalOracleConfig(n_opt_steps=n_opt_steps),
                oracle_config={'n_opt_steps': n_opt_steps}
            )

        sel_res = select_phase6_subset(
            candidates=candidates,
            policy=p_str,
            budget=budget,
            seed=seed,
            safety_factor=self.safety_factor,
            reject_negative=reject_negative,
            use_predicted_cost=self.use_predicted_cost,
            positions=positions,
            all_features=all_features,
            phase6_predictor=self.p6_predictor,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
            cond_oracle=cond_oracle,
            rgb_gt=rgb_gt,
            depth_gt=depth_gt,
            cost_key="measured_trial_cost_ms",
            pred_cost_key="predicted_delta_t",
            pred_utility_key="predicted_utility",
            oracle_utility_key="oracle_utility_joint_global",
        )

        self.sync_gpu()
        t_select_ms = (time.perf_counter() - t_sel_0) * 1000.0

        # === 3. Map candidate indices to active Gaussian model indices ===
        model = oracle_engine.pipeline.gaussian_model
        selected_cands = [candidates[i] for i in sel_res.selected_indices]
        selected_ids = []
        for c in selected_cands:
            act_idx = map_candidate_to_active_index(c, model)
            if act_idx is not None:
                selected_ids.append(act_idx)
        k_count = len(selected_ids)

        # Predicted stats for selected set
        if k_count > 0:
            pred_dq_sum = float(sum(
                float(c.get("predicted_delta_q", 0.0)) for c in selected_cands
            ))
            pred_dt_sum = float(sum(
                float(c.get("predicted_delta_t", 1.0)) for c in selected_cands
            ))
            pred_u_mean = float(np.mean([
                float(c.get("predicted_utility", 0.0)) for c in selected_cands
            ]))
        else:
            pred_dq_sum = 0.0
            pred_dt_sum = 0.0
            pred_u_mean = 0.0

        # === 4. ACTUAL Joint Group Optimization (C1 FIX) ===
        eval_res = evaluate_selected_group(
            oracle_engine=oracle_engine,
            selected_gaussian_ids=selected_ids,
            rgb_gt=rgb_gt,
            depth_gt=depth_gt,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
            n_opt_steps=n_opt_steps,
        )

        delta_q_realized = eval_res["delta_q_realized"]
        delta_psnr_realized = eval_res["delta_psnr_realized"]
        delta_depth_realized = eval_res["delta_depth_realized"]
        actual_opt_cost_ms = eval_res["actual_opt_cost_ms"]

        # === 5. Cost & Violation Accounting ===
        cost_metrics = compute_cost_metrics(
            actual_cost_ms=actual_opt_cost_ms,
            predicted_cost_ms=sel_res.predicted_cost,
            scheduled_cost_ms=sel_res.scheduled_cost,
            budget_ms=budget,
        )

        t_overhead_ms = t_feat_ms + t_pred_ms + t_select_ms
        t_total_ms = t_overhead_ms + actual_opt_cost_ms
        overhead_ratio = float(t_overhead_ms / max(1e-6, t_total_ms))

        # === 6. Comparative Metrics ===
        ose = None
        regret_abs = None
        regret_rel = None
        selection_regret = None

        if oracle_reference_gain is not None:
            ose = compute_ose(delta_q_realized, oracle_reference_gain)
            reg_dict = compute_regret(oracle_reference_gain, delta_q_realized)
            regret_abs = reg_dict["regret_abs"]
            regret_rel = reg_dict["regret_rel"]
            selection_regret = compute_selection_regret(
                oracle_reference_gain, delta_q_realized,
            )

        efficiency = compute_policy_efficiency(
            delta_q_realized, actual_opt_cost_ms,
        )

        # === 7. Diagnostic: sum of individual ΔQ for non-additivity comparison ===
        sum_individual_dq = float(sum(
            float(c.get("delta_q_conditional",
                        c.get("delta_quality_global", 0.0)))
            for c in selected_cands
        )) if k_count > 0 else 0.0

        non_additivity_gap = delta_q_realized - sum_individual_dq

        return {
            # Identity
            "seed": int(seed),
            "frame": int(current_frame),
            "budget_type": str(budget_type),
            "budget_pct_str": str(budget_pct_str),
            "budget_val": float(budget),
            "policy": p_str,

            # Selection
            "selected_ids": selected_ids,
            "k_count": int(k_count),
            "rejected_negative_count": int(sel_res.rejected_negative_count),

            # Predicted (pre-optimization)
            "predicted_delta_q": pred_dq_sum,
            "predicted_delta_t": pred_dt_sum,
            "predicted_utility": pred_u_mean,
            "predicted_total_cost_ms": sel_res.predicted_cost,
            "scheduled_cost_ms": sel_res.scheduled_cost,

            # *** ACTUAL JOINT GROUP OPTIMIZATION (C1 FIX) ***
            "actual_delta_q": delta_q_realized,
            "actual_delta_psnr": delta_psnr_realized,
            "actual_depth_gain": delta_depth_realized,
            "actual_cost_ms": actual_opt_cost_ms,

            # Non-additivity diagnostic
            "sum_individual_dq": sum_individual_dq,
            "non_additivity_gap": non_additivity_gap,
            "influence_pixel_count": eval_res["influence_pixel_count"],

            # Timing breakdown
            "feature_time_ms": float(t_feat_ms),
            "prediction_time_ms": float(t_pred_ms),
            "selection_time_ms": float(t_select_ms),
            "optimization_time_ms": float(actual_opt_cost_ms),
            "overhead_time_ms": float(t_overhead_ms),
            "total_time_ms": float(t_total_ms),
            "overhead_ratio": float(overhead_ratio),

            # Cost accounting
            "budget_violation_ms": float(cost_metrics["budget_violation_ms"]),
            "is_violation": bool(cost_metrics["is_violation"]),
            "is_wall_violation": bool(cost_metrics["is_violation"]),
            "scheduled_violation_ms": float(cost_metrics["scheduled_violation_ms"]),
            "is_scheduled_violation": bool(cost_metrics["is_scheduled_violation"]),

            # Comparative metrics
            "ose": ose,
            "regret_abs": regret_abs,
            "regret_rel": regret_rel,
            "selection_regret": selection_regret,
            "efficiency": efficiency,
        }
