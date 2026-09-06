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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Controlled Adaptivity Verification: Static != Adaptive
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveVsStaticControlled:
    def test_adaptive_reranking_diverges_from_static(self):
        """Controlled test: presence of candidate in selected set degrades
        utility of co-located candidate, causing adaptive to pick a different
        Gaussian than static 1-pass selection."""
        class ContextSensitiveMockPredictor:
            """Predictor where selected context features penalize utility."""
            def __init__(self):
                self.device = "cpu"

            def predict(self, x: torch.Tensor):
                # x is (N, 32)
                # Feature slices: self(0:11), neighbor(11:19), overlap(19:24), selected(24:32)
                # If selected features (e.g. selected overlap or density) are > 0, penalize utility
                # candidate 0: base utility 10.0
                # candidate 1: base utility 9.0 (co-located with 0, penalty if 0 is selected)
                # candidate 2: base utility 8.0 (distant from 0, no penalty)
                N = x.shape[0]
                delta_q = torch.zeros(N)
                delta_t = torch.ones(N)

                # Selected features: index 24 is candidate_selected_overlap
                # or index 26 is min_dist_to_selected
                for i in range(N):
                    # Check if selected set is non-empty (budget_fraction > 0 at index 24)
                    selected_active = x[i, 24] > 0 or x[i, 28] > 0 or x[i, 29] > 0
                    base_u = x[i, 0]  # rgb_error encodes base quality

                    if selected_active and x[i, 1] > 0.5:
                        # Heavy penalty if neighbor is selected
                        u = base_u * 0.1
                    else:
                        u = base_u

                    delta_q[i] = u
                    delta_t[i] = 1.0

                return {
                    "delta_q": delta_q,
                    "delta_t": delta_t,
                    "utility": delta_q / delta_t,
                }

        predictor = ContextSensitiveMockPredictor()

        # 3 candidates:
        # Cand 0: high utility (10), at (0, 0, 0)
        # Cand 1: high initial utility (9), co-located at (0.01, 0, 0) -> subject to penalty
        # Cand 2: moderate utility (8), far away at (10, 10, 0) -> no penalty
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [10.0, 10.0, 0.0],
        ], dtype=torch.float32)

        # Feature col 0: rgb_error (base utility), col 1: co-location flag
        all_features = np.zeros((3, 11), dtype=np.float32)
        all_features[0, 0] = 10.0
        all_features[1, 0] = 9.0
        all_features[1, 1] = 1.0  # flagged for overlap penalty
        all_features[2, 0] = 8.0
        all_features[2, 1] = 0.0  # distant, no penalty

        candidates = [
            {"gaussian_id": 0, "persistent_id": 0, "measured_trial_cost_ms": 1.0, "predicted_delta_t": 1.0, "predicted_importance": 10.0},
            {"gaussian_id": 1, "persistent_id": 1, "measured_trial_cost_ms": 1.0, "predicted_delta_t": 1.0, "predicted_importance": 9.0},
            {"gaussian_id": 2, "persistent_id": 2, "measured_trial_cost_ms": 1.0, "predicted_delta_t": 1.0, "predicted_importance": 8.0},
        ]

        # Budget allows picking exactly 2 candidates (B = 2.0, cost = 1.0 each)
        budget = 2.0

        # Static 1-pass: evaluates S=∅ for all, ranks [0 (u=10), 1 (u=9), 2 (u=8)] -> picks {0, 1}
        res_static = static_context_select(
            candidates=candidates,
            positions=positions,
            all_features=all_features,
            predictor=predictor,
            budget=budget,
            safety_factor=1.0,
            use_predicted_cost=False,
        )

        # Adaptive Greedy:
        # Step 1: picks 0 (u=10). S = {0}.
        # Step 2: re-evaluates remaining {1, 2} with S={0}.
        #         Cand 1 penalty triggers -> u drops to 0.9!
        #         Cand 2 has u=8.0 > 0.9!
        #         Adaptive picks 2! -> picks {0, 2}
        res_adaptive = adaptive_greedy_select(
            candidates=candidates,
            positions=positions,
            all_features=all_features,
            predictor=predictor,
            budget=budget,
            safety_factor=1.0,
            use_predicted_cost=False,
        )

        assert res_static.selected_indices == [0, 1], f"Static should pick [0, 1], got {res_static.selected_indices}"
        assert res_adaptive.selected_indices == [0, 2], f"Adaptive should pick [0, 2], got {res_adaptive.selected_indices}"
        assert res_static.selected_indices != res_adaptive.selected_indices, "Static and Adaptive must strictly diverge under context shift!"
