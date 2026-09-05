"""Phase 5 Evaluator (research/phase5_evaluator.py).

Encapsulates single-frame and multi-frame evaluation under hard compute budgets B:
  1. Strict zero-leakage check (assert cand_frame == current_frame).
  2. Equal compute knapsack selection across all policies.
  3. Real GPU group optimization with isolated snapshot/restore.
  4. Precise accounting of predicted, scheduled, and actual costs, and actual violation tracking.
  5. Computation of OSE, absolute/relative regret, selection regret, and efficiency.
"""
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch

from research.phase5_selection import (
    PolicyName,
    SelectionResult,
    select_budget_constrained_subset,
)
from research.scheduler_metrics import (
    compute_ose,
    compute_regret,
    compute_selection_regret,
    compute_policy_efficiency,
    compute_cost_metrics,
    compute_memory_overhead,
)
from research.utility_predictor import FrozenUtilityPredictor


class Phase5Evaluator:
    """Evaluates budget-constrained Gaussian selection policies under hard compute budgets B."""

    def __init__(
        self,
        predictor: FrozenUtilityPredictor,
        device: Optional[Union[str, torch.device]] = None,
        safety_factor: float = 1.0,
        use_predicted_cost: bool = True,
    ):
        self.predictor = predictor
        if device is None:
            self.device = predictor.device
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.safety_factor = float(safety_factor)
        self.use_predicted_cost = use_predicted_cost

    def sync_gpu(self):
        """Synchronizes GPU for accurate timing measurements."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def evaluate_policy(
        self,
        policy: Union[str, PolicyName],
        candidates: List[Dict[str, Any]],
        budget: float,
        current_frame: int,
        oracle_engine: Any,
        rgb_gt: torch.Tensor,
        depth_gt: torch.Tensor,
        influence_mask: torch.Tensor,
        oracle_reference_gain: Optional[float] = None,
        seed: int = 42,
        random_repeat: Optional[int] = None,
        reject_negative: bool = True,
        budget_type: str = "relative",
        budget_pct_str: str = "60%",
        t_feat_ms: float = 0.0,
        t_pred_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluates a policy on candidate Gaussians for the given frame.
        
        Budget Concepts:
            B_sched: Scheduling budget — sum of safety-factored predicted costs.
                     Violation: V_s = max(0, scheduled_cost - budget)
            B_wall:  Wall-clock budget — actual measured execution time.
                     Violation: V_e = max(0, actual_cost_ms - budget_ms)
        """
        p_str = str(policy.value if hasattr(policy, "value") else policy).lower()

        # Strict State Leakage Assertion
        for c in candidates:
            cand_frame = c.get("frame", current_frame)
            assert cand_frame == current_frame, (
                f"State leakage violation! Candidate frame {cand_frame} != current frame {current_frame}"
            )

        # 1. Selection Step
        self.sync_gpu()
        t_sel_0 = time.perf_counter()

        actual_seed = (seed + 100 * random_repeat) if (random_repeat is not None) else seed
        sel_res = select_budget_constrained_subset(
            candidates=candidates,
            policy=policy,
            budget=budget,
            seed=actual_seed,
            reject_negative=reject_negative,
            use_predicted_cost=self.use_predicted_cost,
            safety_factor=self.safety_factor,
            cost_key="measured_trial_cost_ms",
            pred_cost_key="predicted_delta_t",
            pred_utility_key="predicted_utility",
            oracle_utility_key="oracle_utility_joint_global",
        )

        self.sync_gpu()
        t_select_ms = (time.perf_counter() - t_sel_0) * 1000.0

        num_g = oracle_engine.pipeline.gaussian_model.num_gaussians
        selected_ids = [int(i) for i in sel_res.selected_gaussian_ids if 0 <= int(i) < num_g]
        k_count = len(selected_ids)

        # Compute predicted stats for selected set
        if k_count > 0:
            sel_cands = [candidates[i] for i in sel_res.selected_indices]
            pred_dq_sum = float(sum(float(c.get("predicted_delta_q", 0.0)) for c in sel_cands))
            pred_dt_sum = float(sum(float(c.get("predicted_delta_t", 1.0)) for c in sel_cands))
            pred_u_mean = float(np.mean([float(c.get("predicted_utility", 0.0)) for c in sel_cands]))
        else:
            pred_dq_sum = 0.0
            pred_dt_sum = 0.0
            pred_u_mean = 0.0

        # 2. Group Optimization Step
        actual_opt_cost_ms = 0.0
        delta_q_realized = 0.0
        delta_psnr_realized = 0.0
        actual_depth_gain = 0.0

        if k_count > 0:
            snap = oracle_engine.snapshot_state()
            try:
                self.sync_gpu()
                t_opt_0 = time.perf_counter()

                opt_res = oracle_engine.optimize_gaussian_group(
                    indices=selected_ids,
                    n_steps=5,
                    rgb=rgb_gt,
                    depth=depth_gt,
                    influence_mask=influence_mask,
                )

                self.sync_gpu()
                actual_opt_cost_ms = (time.perf_counter() - t_opt_0) * 1000.0

                delta_q_realized = float(opt_res["delta_quality_global"])
                delta_psnr_realized = float(opt_res.get("delta_psnr_global", 0.0))
                # Compute depth L1 gain
                if "depth_l1_global_before" in opt_res and "depth_l1_global_after" in opt_res:
                    actual_depth_gain = float(opt_res["depth_l1_global_before"] - opt_res["depth_l1_global_after"])
            finally:
                oracle_engine.restore_state(snap)

        # 3. Cost & Violation Accounting
        cost_metrics = compute_cost_metrics(
            actual_cost_ms=actual_opt_cost_ms,
            predicted_cost_ms=sel_res.predicted_cost,
            scheduled_cost_ms=sel_res.scheduled_cost,
            budget_ms=budget,
        )

        t_overhead_ms = t_feat_ms + t_pred_ms + t_select_ms
        t_total_ms = t_overhead_ms + actual_opt_cost_ms
        overhead_ratio = float(t_overhead_ms / max(1e-6, t_total_ms))

        # 4. Comparative Metrics
        ose = None
        regret_abs = None
        regret_rel = None
        selection_regret = None

        if oracle_reference_gain is not None:
            ose = compute_ose(delta_q_realized, oracle_reference_gain)
            reg_dict = compute_regret(oracle_reference_gain, delta_q_realized)
            regret_abs = reg_dict["regret_abs"]
            regret_rel = reg_dict["regret_rel"]
            selection_regret = compute_selection_regret(oracle_reference_gain, delta_q_realized)

        efficiency = compute_policy_efficiency(delta_q_realized, actual_opt_cost_ms)

        return {
            "seed": int(seed),
            "random_repeat": random_repeat,
            "frame": int(current_frame),
            "budget_type": str(budget_type),
            "budget_pct_str": str(budget_pct_str),
            "budget_val": float(budget),
            "policy": p_str,
            "selected_ids": selected_ids,
            "k_count": int(k_count),
            "rejected_negative_count": int(sel_res.rejected_negative_count),
            "predicted_delta_q": pred_dq_sum,
            "predicted_delta_t": pred_dt_sum,
            "predicted_utility": pred_u_mean,
            "predicted_total_cost_ms": sel_res.predicted_cost,
            "scheduled_cost_ms": sel_res.scheduled_cost,
            "scheduled_cost": sel_res.scheduled_cost,
            "actual_cost_ms": actual_opt_cost_ms,
            "actual_delta_q": delta_q_realized,
            "actual_delta_psnr": delta_psnr_realized,
            "actual_depth_gain": actual_depth_gain,
            "feature_time_ms": float(t_feat_ms),
            "prediction_time_ms": float(t_pred_ms),
            "selection_time_ms": float(t_select_ms),
            "optimization_time_ms": float(actual_opt_cost_ms),
            "overhead_time_ms": float(t_overhead_ms),
            "total_time_ms": float(t_total_ms),
            "overhead_ratio": float(overhead_ratio),
            "budget_violation_ms": float(cost_metrics["budget_violation_ms"]),
            "is_violation": bool(cost_metrics["is_violation"]),
            "is_wall_violation": bool(cost_metrics["is_violation"]),
            "scheduled_violation_ms": float(cost_metrics["scheduled_violation_ms"]),
            "is_scheduled_violation": bool(cost_metrics["is_scheduled_violation"]),
            "ose": ose,
            "regret_abs": regret_abs,
            "regret_rel": regret_rel,
            "selection_regret": selection_regret,
            "efficiency": efficiency,
        }
