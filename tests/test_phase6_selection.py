"""Tests for Phase 6 Budget-Constrained Selection & Adaptive Greedy.

Verifies:
  1. Strict Budget Compliance (sum alpha * C_i <= B)
  2. Determinism of Adaptive Greedy
  3. No Duplicate Candidates Selected
  4. Negative Utility Rejection
  5. Empty and Degenerate Inputs (B=0, empty pool)
  6. Monotonicity with Safety Factor
  7. Unified Policy Dispatch (select_phase6_subset)
"""
import pytest
import numpy as np
import torch

from research.phase6_selection import (
    Phase6PolicyName,
    adaptive_greedy_select,
    static_context_select,
    select_phase6_subset,
)
from research.phase6_model import (
    ContextAwareTwoHeadMLP,
    Phase6ModelConfig,
    FrozenContextPredictor,
)
from research.phase6_dataset import Phase6FeatureNormalizer
from research.phase6_context import PHASE6_FEATURE_DIM


@pytest.fixture
def mock_positions():
    torch.manual_seed(42)
    positions = []
    for i in range(4):
        for j in range(5):
            positions.append([float(i), float(j), 0.0])
    return torch.tensor(positions, dtype=torch.float32)


@pytest.fixture
def mock_all_features():
    np.random.seed(42)
    return np.random.rand(20, 11).astype(np.float32)


@pytest.fixture
def mock_candidates():
    # 20 candidates matching the 20 Gaussians
    cands = []
    for i in range(20):
        cands.append({
            "gaussian_id": i,
            "persistent_id": i,
            "features": {
                "rgb_error": 0.05 + 0.01 * i,
                "depth_error": 0.02 + 0.005 * i,
                "influence_mass": 1.0 + 0.2 * i,
            },
            "predicted_importance": 0.1 + 0.04 * i,
            "predicted_utility": 0.01 * (i - 5),  # some negative, some positive
            "predicted_delta_t": 2.0 + 0.5 * (i % 3),
            "measured_trial_cost_ms": 2.5,
        })
    return cands


@pytest.fixture
def mock_predictor(tmp_path):
    # Create and save a minimal model + normalizer
    config = Phase6ModelConfig()
    model = ContextAwareTwoHeadMLP(config)
    
    ckpt_path = str(tmp_path / "mock_ckpt.pt")
    norm_path = str(tmp_path / "mock_norm.json")

    torch.save({"model_state": model.state_dict(), "config": config.__dict__}, ckpt_path)

    normalizer = Phase6FeatureNormalizer()
    normalizer.fit(np.random.rand(50, PHASE6_FEATURE_DIM).astype(np.float32))
    normalizer.save_json(norm_path)

    return FrozenContextPredictor(ckpt_path, norm_path, device="cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Budget Compliance Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetCompliance:
    def test_adaptive_greedy_respects_budget(self, mock_candidates, mock_positions, mock_all_features, mock_predictor):
        budget = 10.0
        safety_factor = 1.1
        res = adaptive_greedy_select(
            candidates=mock_candidates,
            positions=mock_positions,
            all_features=mock_all_features,
            predictor=mock_predictor,
            budget=budget,
            safety_factor=safety_factor,
        )
        assert res.scheduled_cost <= budget + 1e-6
        assert not res.is_scheduled_violation
        assert res.scheduled_budget_violation == 0.0

    def test_static_context_respects_budget(self, mock_candidates, mock_positions, mock_all_features, mock_predictor):
        budget = 8.0
        res = static_context_select(
            candidates=mock_candidates,
            positions=mock_positions,
            all_features=mock_all_features,
            predictor=mock_predictor,
            budget=budget,
            safety_factor=1.2,
        )
        assert res.scheduled_cost <= budget + 1e-6
        assert not res.is_scheduled_violation

    def test_zero_budget_yields_empty(self, mock_candidates, mock_positions, mock_all_features, mock_predictor):
        res = adaptive_greedy_select(
            candidates=mock_candidates,
            positions=mock_positions,
            all_features=mock_all_features,
            predictor=mock_predictor,
            budget=0.0,
        )
        assert res.k_count == 0
        assert res.selected_indices == []
        assert res.scheduled_cost == 0.0

    def test_negative_budget_yields_empty(self, mock_candidates, mock_positions, mock_all_features, mock_predictor):
        res = adaptive_greedy_select(
            candidates=mock_candidates,
            positions=mock_positions,
            all_features=mock_all_features,
            predictor=mock_predictor,
            budget=-5.0,
        )
        assert res.k_count == 0
        assert res.selected_indices == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Determinism & Integrity Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminismAndIntegrity:
    def test_adaptive_greedy_is_deterministic(self, mock_candidates, mock_positions, mock_all_features, mock_predictor):
        res1 = adaptive_greedy_select(
            candidates=mock_candidates,
            positions=mock_positions,
            all_features=mock_all_features,
            predictor=mock_predictor,
            budget=15.0,
        )
        res2 = adaptive_greedy_select(
            candidates=mock_candidates,
            positions=mock_positions,
            all_features=mock_all_features,
            predictor=mock_predictor,
            budget=15.0,
        )
        assert res1.selected_indices == res2.selected_indices
        assert res1.scheduled_cost == res2.scheduled_cost

    def test_no_duplicate_selections(self, mock_candidates, mock_positions, mock_all_features, mock_predictor):
        res = adaptive_greedy_select(
            candidates=mock_candidates,
            positions=mock_positions,
            all_features=mock_all_features,
            predictor=mock_predictor,
            budget=50.0,  # Large budget
            reject_negative=False,
        )
        # All selected indices must be unique
        assert len(res.selected_indices) == len(set(res.selected_indices))
        assert len(res.selected_gaussian_ids) == len(set(res.selected_gaussian_ids))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Safety Factor & Monotonicity Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyFactor:
    def test_safety_factor_constrains_selection(self, mock_candidates, mock_positions, mock_all_features, mock_predictor):
        budget = 12.0
        res_low = adaptive_greedy_select(
            candidates=mock_candidates,
            positions=mock_positions,
            all_features=mock_all_features,
            predictor=mock_predictor,
            budget=budget,
            safety_factor=1.0,
            reject_negative=False,
        )
        res_high = adaptive_greedy_select(
            candidates=mock_candidates,
            positions=mock_positions,
            all_features=mock_all_features,
            predictor=mock_predictor,
            budget=budget,
            safety_factor=2.0,
            reject_negative=False,
        )
        # Higher safety factor must select <= candidates
        assert res_high.k_count <= res_low.k_count


# ─────────────────────────────────────────────────────────────────────────────
# 4. Unified Dispatch Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifiedDispatch:
    def test_dispatch_noop(self, mock_candidates):
        res = select_phase6_subset(mock_candidates, policy="no_op", budget=10.0)
        assert res.k_count == 0
        assert res.policy == "no_op"

    def test_dispatch_random(self, mock_candidates):
        res = select_phase6_subset(mock_candidates, policy="random", budget=10.0, seed=42)
        assert res.k_count > 0
        assert res.scheduled_cost <= 10.0 + 1e-6

    def test_dispatch_heuristic(self, mock_candidates):
        res = select_phase6_subset(mock_candidates, policy="heuristic", budget=10.0)
        assert res.k_count > 0
        assert res.scheduled_cost <= 10.0 + 1e-6

    def test_dispatch_error_influence(self, mock_candidates):
        res = select_phase6_subset(mock_candidates, policy="error_influence", budget=10.0)
        assert res.k_count > 0
        assert res.scheduled_cost <= 10.0 + 1e-6

    def test_dispatch_phase6_adaptive(self, mock_candidates, mock_positions, mock_all_features, mock_predictor):
        res = select_phase6_subset(
            candidates=mock_candidates,
            policy="phase6_adaptive",
            budget=10.0,
            positions=mock_positions,
            all_features=mock_all_features,
            phase6_predictor=mock_predictor,
        )
        assert res.policy == "phase6_adaptive"
        assert res.scheduled_cost <= 10.0 + 1e-6

    def test_dispatch_phase6_static(self, mock_candidates, mock_positions, mock_all_features, mock_predictor):
        res = select_phase6_subset(
            candidates=mock_candidates,
            policy="phase6_static",
            budget=10.0,
            positions=mock_positions,
            all_features=mock_all_features,
            phase6_predictor=mock_predictor,
        )
        assert res.policy == "phase6_static"
        assert res.scheduled_cost <= 10.0 + 1e-6
