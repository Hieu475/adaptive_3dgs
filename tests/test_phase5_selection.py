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
    """Verify that all policies strictly respect the hard compute budget."""
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
        res = select_budget_constrained_subset(
            candidates,
            policy=policy,
            budget=budget,
            seed=42,
            cost_key="measured_trial_cost_ms",
            pred_cost_key="predicted_delta_t",
        )
        assert isinstance(res, SelectionResult)
        if policy == PolicyName.LEARNED_UTILITY:
            # Learned policy enforces budget on predicted cost
            assert res.predicted_cost <= budget + 1e-6
        else:
            # Baseline policies enforce budget on nominal cost
            assert res.nominal_cost <= budget + 1e-6


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
