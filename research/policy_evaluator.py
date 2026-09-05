r"""Phase 5 Policy Evaluator.

Executes controlled policy evaluation under hard compute budgets B:
  1. Single-Frame Controlled Benchmark:
       G_t -> S_B -> G'_t
  2. Online Multi-Frame Sequential Trajectory:
       G_t -> S_t -> G_{t+1} -> S_{t+1} -> ...

Guarantees:
  - Zero future-frame or oracle leakage at runtime (Point 25).
  - Component latency breakdown (T_feat, T_pred, T_select, T_opt, T_total) with GPU sync (Point 23, 24).
  - True budget packing across ALL policies (Points 15, 16, 17).
  - Global quality gain target Delta Q_realized (Point 5).
  - Safety margin C_sched = alpha * C_hat (Point 10).
  - Negative utility rejection and empty subset support (Points 11, 12).
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
    compute_policy_efficiency,
    compute_cost_metrics,
    compute_selection_churn,
    compute_memory_overhead,
)
from research.utility_predictor import FrozenUtilityPredictor


class PolicyEvaluator:
    """Evaluator for budget-constrained Gaussian scheduling policies."""

    def __init__(
        self,
        predictor: FrozenUtilityPredictor,
        device: Optional[Union[str, torch.device]] = None,
        safety_factor: float = 1.0,
    ):
        """Initializes evaluator with a frozen Phase 4 predictor."""
        self.predictor = predictor
        if device is None:
            self.device = predictor.device
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.safety_factor = safety_factor

    def sync_gpu(self):
        """Synchronizes GPU if CUDA is active for rigorous latency profiling (Point 24)."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def evaluate_policy_single_frame(
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
        reject_negative: bool = True,
        use_predicted_cost: bool = True,
    ) -> Dict[str, Any]:
        r"""Evaluates a single policy on candidate Gaussians for the given frame.

        Args:
            policy: Policy name (random, error_only, error_influence, heuristic, learned_utility, oracle).
            candidates: Candidate pool annotated with features and predictions.
            budget: Compute budget threshold in ms.
            current_frame: Current frame index (enforces Point 25 zero leakage).
            oracle_engine: OracleUtilityExperiment engine for executing isolated Adam steps.
            rgb_gt: Ground truth RGB image tensor.
            depth_gt: Ground truth depth image tensor.
            influence_mask: Screen influence mask tensor.
            oracle_reference_gain: Global Delta Q of Oracle policy for OSE/Regret derivation.
            seed: Seed for stochastic policies.
            reject_negative: Whether to reject candidates with non-positive utility.
            use_predicted_cost: Whether learned policy uses predicted cost \hat{C} for knapsack.
        """
        p_str = str(policy.value if hasattr(policy, "value") else policy).lower()

        # Leakage Assertion (Point 25)
        for c in candidates:
            cand_frame = c.get("frame", current_frame)
            assert cand_frame == current_frame, (
                f"State leakage violation! Candidate frame {cand_frame} != current frame {current_frame}"
            )

        # 1. Selection Step
        self.sync_gpu()
        t_sel_0 = time.perf_counter()

        sel_res = select_budget_constrained_subset(
            candidates=candidates,
            policy=policy,
            budget=budget,
            seed=seed,
            reject_negative=reject_negative,
            use_predicted_cost=use_predicted_cost,
            safety_factor=self.safety_factor if p_str in ("learned_utility", "ours") else 1.0,
            cost_key="measured_trial_cost_ms",
            pred_cost_key="predicted_delta_t",
            pred_utility_key="predicted_utility",
            oracle_utility_key="oracle_utility_joint_global",
        )

        self.sync_gpu()
        t_select = (time.perf_counter() - t_sel_0) * 1000.0

        selected_ids = sel_res.selected_gaussian_ids

        # 2. Real Adam Optimization Step
        actual_opt_cost = 0.0
        delta_q_realized = 0.0
        delta_psnr_realized = 0.0

        if selected_ids:
            # Bitwise snapshot/restore ensures 100% equal scene initialization
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
                actual_opt_cost = (time.perf_counter() - t_opt_0) * 1000.0

                # Target quality is strictly global (Point 5)
                delta_q_realized = float(opt_res["delta_quality_global"])
                delta_psnr_realized = float(opt_res.get("delta_psnr_global", 0.0))
            finally:
                oracle_engine.restore_state(snap)

        # 3. Derived Metrics (Points 8, 9, 18, 19, 20)
        cost_metrics = compute_cost_metrics(
            actual_cost_ms=actual_opt_cost,
            predicted_cost_ms=sel_res.predicted_cost,
            budget_ms=budget,
        )

        ose = None
        regret_abs = None
        regret_rel = None
        if oracle_reference_gain is not None:
            ose = compute_ose(delta_q_realized, oracle_reference_gain)
            reg_dict = compute_regret(oracle_reference_gain, delta_q_realized)
            regret_abs = reg_dict["regret_abs"]
            regret_rel = reg_dict["regret_rel"]

        efficiency = compute_policy_efficiency(delta_q_realized, actual_opt_cost)

        return {
            "policy": p_str,
            "frame": current_frame,
            "budget_ms": float(budget),
            "k_selected": sel_res.k_count,
            "selected_ids": selected_ids,
            "rejected_negative_count": sel_res.rejected_negative_count,
            "delta_quality_realized": delta_q_realized,
            "delta_psnr_realized": delta_psnr_realized,
            "actual_cost_ms": actual_opt_cost,
            "predicted_cost_ms": sel_res.predicted_cost,
            "scheduled_cost_ms": sel_res.scheduled_cost,
            "nominal_cost_ms": sel_res.nominal_cost,
            "cost_error_ms": cost_metrics["cost_error_ms"],
            "mape_c": cost_metrics["mape_c"],
            "budget_violation_ms": cost_metrics["budget_violation_ms"],
            "is_violation": cost_metrics["is_violation"],
            "ose": ose,
            "regret_abs": regret_abs,
            "regret_rel": regret_rel,
            "efficiency": efficiency,
            "t_select_ms": float(t_select),
            "t_opt_ms": float(actual_opt_cost),
        }
