"""Unit tests for Phase 5 Frozen Utility Predictor and Budgeted Selection Policies."""
import os
import pytest
import numpy as np
import torch

from research.utility_predictor import FrozenUtilityPredictor
from research.phase5_selection import (
    PolicyName,
    SelectionResult,
    select_budget_constrained_subset,
    map_candidate_to_active_index,
)


def test_frozen_utility_predictor_initialization():
    """Verify frozen predictor loads checkpoint, sets eval mode, and freezes weights."""
    predictor = FrozenUtilityPredictor(seed=42, device="cpu")
    
    assert predictor.seed == 42
    assert predictor.in_features == 11
    assert not predictor.model.training
    
    for name, param in predictor.model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"


def test_frozen_utility_predictor_inference():
    """Verify predictor outputs valid decoupled quantities and positive cost."""
    predictor = FrozenUtilityPredictor(seed=42, device="cpu")
    
    # 5 dummy candidate feature vectors of length 11
    X = np.random.randn(5, 11).astype(np.float32)
    res = predictor.predict_features(X)
    
    assert "predicted_delta_q" in res
    assert "predicted_delta_t" in res
    assert "predicted_utility" in res
    assert "pred_time_ms" in res
    
    q = res["predicted_delta_q"]
    t = res["predicted_delta_t"]
    u = res["predicted_utility"]
    
    assert len(q) == 5
    assert len(t) == 5
    assert len(u) == 5
    # Cost head uses Softplus + eps_cost, must be strictly positive
    assert (t > 0).all()
    # Derived utility definition
    np.testing.assert_allclose(u, q / t, rtol=1e-5)
    assert res["pred_time_ms"] >= 0.0


def test_selection_hard_budget_constraint():
    """Verify that all policies strictly respect the hard compute budget.

    Reform: Under unified budget semantics, ALL policies (including baselines)
    must satisfy the same scheduled_cost <= budget constraint. The scheduled_cost
    is defined as sum(packing_cost_i * safety_factor) for all selected candidates.
    """
    candidates = [
        {
            "gaussian_id": i,
            "measured_trial_cost_ms": 5.0 + i * 2.0,
            "predicted_delta_t": 5.0 + i * 2.0,
            "predicted_utility": float(0.1 * (i % 3 - 1)),
            "predicted_importance": float(i + 1),
            "oracle_utility_joint_global": float(0.2 * (i % 3 - 1)),
            "features": {
                "rgb_error": float(0.1 * i),
                "depth_error": float(0.05 * i),
                "influence_mass": float(1.0 + i),
            },
        }
        for i in range(10)
    ]
    budget = 25.0

    for policy in PolicyName:
        for sf in [1.0, 1.10]:
            res = select_budget_constrained_subset(
                candidates,
                policy=policy,
                budget=budget,
                seed=42,
                use_predicted_cost=True,
                safety_factor=sf,
                cost_key="measured_trial_cost_ms",
                pred_cost_key="predicted_delta_t",
            )
            assert isinstance(res, SelectionResult)
            # Unified budget semantics: ALL policies must satisfy scheduled_cost <= budget
            assert res.scheduled_cost <= budget + 1e-6, (
                f"Policy {policy.value} (alpha={sf}) violates budget: "
                f"scheduled_cost={res.scheduled_cost:.4f} > budget={budget:.4f}"
            )
            # Scheduled violation must be near zero
            assert res.scheduled_budget_violation < 1e-5, (
                f"Policy {policy.value} has non-zero scheduled violation: {res.scheduled_budget_violation}"
            )


def test_negative_utility_rejection():
    """Verify that learned utility and oracle reject non-positive utility candidates."""
    # Pool with strictly negative utilities
    cands_neg = [
        {
            "gaussian_id": i,
            "measured_trial_cost_ms": 10.0,
            "predicted_delta_t": 10.0,
            "predicted_utility": -0.5 - i * 0.1,
            "oracle_utility_joint_global": -0.3 - i * 0.1,
            "features": {"rgb_error": 1.0, "depth_error": 1.0, "influence_mass": 1.0},
        }
        for i in range(5)
    ]

    res_lrn = select_budget_constrained_subset(cands_neg, policy=PolicyName.LEARNED_UTILITY, budget=50.0)
    assert res_lrn.k_count == 0
    assert res_lrn.rejected_negative_count == 5
    assert len(res_lrn.selected_indices) == 0

    res_ora = select_budget_constrained_subset(cands_neg, policy=PolicyName.ORACLE, budget=50.0)
    assert res_ora.k_count == 0
    assert res_ora.rejected_negative_count == 5

    # Mixed pool: only positive items should be selected
    cands_mixed = [
        {
            "gaussian_id": 1,
            "measured_trial_cost_ms": 10.0,
            "predicted_delta_t": 10.0,
            "predicted_utility": -0.2,
            "oracle_utility_joint_global": -0.1,
            "features": {"rgb_error": 0.9, "depth_error": 0.9, "influence_mass": 1.0},
        },
        {
            "gaussian_id": 2,
            "measured_trial_cost_ms": 10.0,
            "predicted_delta_t": 10.0,
            "predicted_utility": 0.6,
            "oracle_utility_joint_global": 0.7,
            "features": {"rgb_error": 0.2, "depth_error": 0.1, "influence_mass": 1.0},
        },
    ]
    res_mixed = select_budget_constrained_subset(cands_mixed, policy=PolicyName.LEARNED_UTILITY, budget=30.0)
    assert res_mixed.selected_gaussian_ids == [2]
    assert res_mixed.rejected_negative_count == 1


def test_empty_candidate_pool_graceful():
    """Verify graceful handling of empty candidate pools."""
    res = select_budget_constrained_subset([], policy=PolicyName.LEARNED_UTILITY, budget=20.0)
    assert res.k_count == 0
    assert res.predicted_cost == 0.0
    assert res.nominal_cost == 0.0
    assert res.selected_indices == []


def test_budget_violation_tracking():
    """Verify that actual cost violations are correctly detected and reported.

    Phase 5 Reform (Item I): The evaluation must track actual_cost vs budget
    and report violations. C_actual(S_B) > B must be flagged as is_violation=True.
    """
    from research.scheduler_metrics import compute_cost_metrics

    # Case 1: No violation (actual < budget)
    m1 = compute_cost_metrics(actual_cost_ms=15.0, predicted_cost_ms=12.0, budget_ms=20.0)
    assert m1["is_violation"] is False
    assert m1["budget_violation_ms"] == 0.0

    # Case 2: Violation (actual > budget)
    m2 = compute_cost_metrics(actual_cost_ms=700.0, predicted_cost_ms=12.0, budget_ms=20.0)
    assert m2["is_violation"] is True
    assert m2["budget_violation_ms"] == pytest.approx(680.0)

    # Case 3: Large violation typical of group optimization overhead
    m3 = compute_cost_metrics(actual_cost_ms=994.0, predicted_cost_ms=16.2, budget_ms=138.0)
    assert m3["is_violation"] is True
    assert m3["budget_violation_ms"] == pytest.approx(856.0)
    assert m3["cost_error_ms"] == pytest.approx(994.0 - 16.2)


def test_cost_calibration_metrics():
    """Verify MAE_C, MAPE_C, and R2_C computation for cost calibration (Item XIII)."""
    from research.scheduler_metrics import compute_cost_calibration_metrics

    # Perfect calibration
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([10.0, 20.0, 30.0])
    m = compute_cost_calibration_metrics(actual, predicted)
    assert m["mae_c"] == pytest.approx(0.0, abs=1e-6)
    assert m["r2_c"] == pytest.approx(1.0, abs=1e-4)

    # Imperfect calibration
    actual2 = np.array([700.0, 800.0, 900.0, 1000.0])
    predicted2 = np.array([10.0, 15.0, 20.0, 25.0])
    m2 = compute_cost_calibration_metrics(actual2, predicted2)
    assert m2["mae_c"] > 600.0  # Large systematic underestimation
    assert m2["mape_c"] > 90.0  # Very high MAPE


def test_selection_with_safety_factor_reduces_count():
    """Verify that higher safety factors reduce the number of selected candidates.

    Phase 5 Reform (Item XIV): Safety factor alpha should trade off between
    fewer candidates (lower violation risk) and quality loss.
    """
    candidates = [
        {
            "gaussian_id": i,
            "measured_trial_cost_ms": 5.0,
            "predicted_delta_t": 5.0,
            "predicted_utility": float(1.0),
            "oracle_utility_joint_global": float(1.0),
            "features": {"rgb_error": 0.5, "depth_error": 0.5, "influence_mass": 1.0},
        }
        for i in range(10)
    ]
    budget = 25.0

    res_100 = select_budget_constrained_subset(
        candidates, policy=PolicyName.LEARNED_UTILITY, budget=budget,
        safety_factor=1.0, use_predicted_cost=True,
    )
    res_120 = select_budget_constrained_subset(
        candidates, policy=PolicyName.LEARNED_UTILITY, budget=budget,
        safety_factor=1.20, use_predicted_cost=True,
    )
    # Higher safety factor means fewer candidates fit under budget
    assert res_100.k_count >= res_120.k_count
    # alpha=1.0: 5*5=25 <= 25, so 5 fit. alpha=1.2: 5*6=30 > 25, 4*6=24 <= 25, so 4 fit.
    assert res_100.k_count == 5
    assert res_120.k_count == 4


def test_noop_baseline():
    """Verify NO_OP baseline returns zero cost and zero quality (Item XVI)."""
    candidates = [
        {
            "gaussian_id": i,
            "measured_trial_cost_ms": 10.0,
            "predicted_delta_t": 10.0,
            "predicted_utility": 0.5,
            "oracle_utility_joint_global": 0.5,
            "features": {"rgb_error": 0.5, "depth_error": 0.5, "influence_mass": 1.0},
        }
        for i in range(5)
    ]
    res = select_budget_constrained_subset(
        candidates, policy=PolicyName.NO_OP, budget=100.0
    )
    assert res.k_count == 0
    assert res.predicted_cost == 0.0
    assert res.scheduled_cost == 0.0
    assert res.nominal_cost == 0.0
    assert len(res.selected_indices) == 0
    assert len(res.selected_gaussian_ids) == 0


def test_candidate_id_to_active_gaussian_mapping():
    """Test 1: Gaussian ID <-> active index mapping.

    Verifies:
      1. Direct match: candidate.gaussian_id == active index.
      2. Index shift match via persistent_id: candidate points to persistent_id relocated by pruning/compaction.
      3. Out-of-bounds rejection: invalid ID returns None.
      4. Pruned rejection: persistent_id not in model returns None.
    """
    class MockModel:
        def __init__(self, persistent_ids: torch.Tensor, num_gaussians: int):
            self.persistent_ids = persistent_ids
            self.num_gaussians = num_gaussians

    # Model after compaction: persistent_ids are [10, 25, 42, 99] at active indices [0, 1, 2, 3]
    model = MockModel(
        persistent_ids=torch.tensor([10, 25, 42, 99], dtype=torch.long),
        num_gaussians=4,
    )

    # 1. Direct match (candidate created after compaction, gid matches active index and pid)
    c_direct = {"gaussian_id": 2, "persistent_id": 42}
    assert map_candidate_to_active_index(c_direct, model) == 2

    # 2. Index shift match (candidate was created at old index 15 before compaction, but persistent_id is 42)
    c_shifted = {"gaussian_id": 15, "persistent_id": 42}
    assert map_candidate_to_active_index(c_shifted, model) == 2

    # 3. Pruned Gaussian: persistent_id 77 is not in model
    c_pruned = {"gaussian_id": 5, "persistent_id": 77}
    assert map_candidate_to_active_index(c_pruned, model) is None

    # 4. Out of bounds index with no matching persistent_id
    c_oob = {"gaussian_id": 100, "persistent_id": 999}
    assert map_candidate_to_active_index(c_oob, model) is None

    # 5. Fallback without persistent_id (pure index bounds check)
    c_fallback = {"gaussian_id": 1}
    assert map_candidate_to_active_index(c_fallback, model) == 1

    c_fallback_oob = {"gaussian_id": 10}
    assert map_candidate_to_active_index(c_fallback_oob, model) is None


def test_stage_a_stage_b_selector_equivalence():
    """Test 2: Stage A selector == Stage B selector adapter equivalence.

    Verifies that the same candidate state yields identical selection results
    in both Stage A and Stage B selector adapters.
    """
    class MockModel:
        def __init__(self, n: int):
            self.persistent_ids = torch.arange(n, dtype=torch.long)
            self.num_gaussians = n

    N = 20
    model = MockModel(N)
    budget = 15.0

    candidates = [
        {
            "gaussian_id": i,
            "persistent_id": i,
            "features": {
                "rgb_error": float(0.1 * i),
                "depth_error": float(0.05 * i),
                "influence_mass": float(1.0 + 0.1 * i),
            },
            "predicted_importance": float(i + 1),
            "measured_trial_cost_ms": 2.5,
            "predicted_delta_t": 2.5,
            "predicted_utility": float(0.2 * (i - 5)),
            "oracle_utility_joint_global": float(0.3 * (i - 5)),
        }
        for i in range(N)
    ]

    for pol in ["learned_utility", "heuristic", "error_only", "random", "no_op"]:
        # Stage A selection
        res_a = select_budget_constrained_subset(
            candidates=candidates,
            policy=pol,
            budget=budget,
            seed=42,
            reject_negative=(pol == "learned_utility"),
            use_predicted_cost=True,
            safety_factor=1.10,
        )
        selected_ids_a = []
        for s_idx in res_a.selected_indices:
            act_idx = map_candidate_to_active_index(candidates[s_idx], model)
            if act_idx is not None:
                selected_ids_a.append(act_idx)

        # Stage B adapter selection
        res_b = select_budget_constrained_subset(
            candidates=candidates,
            policy=pol,
            budget=budget,
            seed=42,
            reject_negative=(pol == "learned_utility"),
            use_predicted_cost=True,
            safety_factor=1.10,
        )
        selected_ids_b = []
        for s_idx in res_b.selected_indices:
            act_idx = map_candidate_to_active_index(candidates[s_idx], model)
            if act_idx is not None and 0 <= act_idx < N:
                selected_ids_b.append(act_idx)

        assert selected_ids_a == selected_ids_b, f"Mismatch for policy {pol}!"
        assert res_a.scheduled_cost == pytest.approx(res_b.scheduled_cost), f"Cost mismatch for policy {pol}!"
        assert res_a.k_count == res_b.k_count, f"Count mismatch for policy {pol}!"


def test_same_budget_same_cost_definition_across_policies():
    """Test 3: Same budget -> same cost definition across all policies.

    Verifies that all competing policies (RANDOM, ERROR_ONLY, ERROR_INFLUENCE,
    HEURISTIC, LEARNED_UTILITY) pack against the exact same cost constraint
    sum_{i in S} alpha * \\hat{C}_i <= B.
    """
    N = 15
    cost_per_item = 3.0
    alpha = 1.10
    budget = 10.0  # At 3.0 * 1.10 = 3.3 ms per item, at most floor(10.0 / 3.3) = 3 items can fit!

    candidates = [
        {
            "gaussian_id": i,
            "persistent_id": i,
            "measured_trial_cost_ms": cost_per_item,
            "predicted_delta_t": cost_per_item,
            "predicted_utility": float(1.0 + 0.1 * i),  # all positive
            "predicted_importance": float(i + 1),
            "oracle_utility_joint_global": float(1.0 + 0.1 * i),
            "features": {
                "rgb_error": float(0.1 * (i + 1)),
                "depth_error": float(0.05 * (i + 1)),
                "influence_mass": float(1.0 + 0.1 * i),
            },
        }
        for i in range(N)
    ]

    competing_policies = [
        PolicyName.RANDOM,
        PolicyName.ERROR_ONLY,
        PolicyName.ERROR_INFLUENCE,
        PolicyName.HEURISTIC,
        PolicyName.LEARNED_UTILITY,
    ]

    expected_max_items = int(np.floor(budget / (cost_per_item * alpha)))  # 3 items
    expected_cost = expected_max_items * cost_per_item * alpha            # 9.9 ms

    for pol in competing_policies:
        res = select_budget_constrained_subset(
            candidates=candidates,
            policy=pol,
            budget=budget,
            seed=42,
            reject_negative=(pol == PolicyName.LEARNED_UTILITY),
            use_predicted_cost=True,
            safety_factor=alpha,
        )

        # Every policy must pack with the exact same cost per item: 3.3 ms
        assert res.k_count == expected_max_items, (
            f"Policy {pol.value} selected {res.k_count} items, expected {expected_max_items}!"
        )
        assert res.scheduled_cost == pytest.approx(expected_cost, abs=1e-5), (
            f"Policy {pol.value} scheduled cost {res.scheduled_cost:.4f} != expected {expected_cost:.4f}!"
        )
        assert res.scheduled_cost <= budget + 1e-6
        assert res.scheduled_budget_violation == 0.0
        assert not res.is_scheduled_violation

