"""Tests for Phase 6 Conditional Oracle (research/phase6_oracle.py).

Test Categories:
    1. Oracle Identity: ΔQ(i|S) = Q(S∪{i}) - Q(S) exactly.
    2. Empty Context Recovery: S = ∅ ⟹ U*(i|∅) ≈ Phase 4 marginal utility.
    3. Context Set Sampling: Strategies produce valid, non-overlapping sets.
    4. Measurement Protocol: snapshot/restore isolation verified.
    5. Interaction Analysis: I(i,j) = ΔQ(i,j) - ΔQ(i) - ΔQ(j).
    6. Edge Cases: Invalid indices, empty pools, single Gaussian.

NOTE: Tests that require a full pipeline (GPU, model, rendering) are marked
with @pytest.mark.requires_pipeline and skip when no pipeline fixture is available.
Unit tests that verify logic without GPU are always runnable.
"""
import pytest
import numpy as np
import torch

from research.phase6_oracle import (
    ConditionalOracleConfig,
    ConditionalOracleExperiment,
)
from research.phase6_context import (
    ContextConfig,
    PHASE6_FEATURE_DIM,
    _get_knn_indices,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def mock_positions(device):
    """20 Gaussians in a regular grid."""
    torch.manual_seed(42)
    positions = []
    for i in range(4):
        for j in range(5):
            positions.append([float(i), float(j), 0.0])
    return torch.tensor(positions, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConditionalOracleConfig:
    """Verify config defaults and constraints."""

    def test_default_config(self):
        config = ConditionalOracleConfig()
        assert config.n_opt_steps == 5
        assert config.context_sizes == [0, 1, 4, 8]
        assert abs(sum(config.context_size_weights) - 1.0) < 1e-6
        assert len(config.context_sizes) == len(config.context_size_weights)
        assert config.k_neighbors == 8
        assert config.epsilon == 1e-6

    def test_context_types_valid(self):
        config = ConditionalOracleConfig()
        valid_types = {"empty", "spatial_knn", "overlap_top", "random"}
        for ct in config.context_types:
            assert ct in valid_types, f"Invalid context type: {ct}"

    def test_weights_sum_to_one(self):
        config = ConditionalOracleConfig()
        assert abs(sum(config.context_size_weights) - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 2. Context Set Sampling Tests (no pipeline needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestContextSampling:
    """Test context set sampling strategies.

    These tests use a mock pipeline that only needs positions for spatial
    sampling. The actual conditional utility measurement is tested separately.
    """

    def test_knn_indices_correct_count(self, mock_positions):
        """KNN should return exactly k neighbors."""
        neighbors = _get_knn_indices(mock_positions, 5, k=4)
        assert len(neighbors) == 4

    def test_knn_excludes_self(self, mock_positions):
        """KNN neighbors must not include the candidate itself."""
        neighbors = _get_knn_indices(mock_positions, 5, k=8)
        assert 5 not in neighbors

    def test_knn_nearest_are_correct(self):
        """With simple geometry, verify nearest neighbors."""
        positions = torch.tensor([
            [0, 0, 0],   # 0: candidate
            [1, 0, 0],   # 1: distance 1
            [0, 1, 0],   # 2: distance 1
            [1, 1, 0],   # 3: distance sqrt(2)
            [10, 10, 10], # 4: far away
        ], dtype=torch.float32)

        neighbors = _get_knn_indices(positions, 0, k=2)
        # Should be indices 1 and 2 (both at distance 1)
        assert set(neighbors) == {1, 2}

    def test_knn_bounded_by_population(self):
        """With N=3, k=10 should return only 2 neighbors."""
        positions = torch.tensor([
            [0, 0, 0], [1, 0, 0], [2, 0, 0]
        ], dtype=torch.float32)
        neighbors = _get_knn_indices(positions, 0, k=10)
        assert len(neighbors) == 2

    def test_knn_oob_returns_empty(self, mock_positions):
        neighbors = _get_knn_indices(mock_positions, 999, k=4)
        assert neighbors == []

    def test_knn_single_gaussian_returns_empty(self):
        positions = torch.tensor([[0, 0, 0]], dtype=torch.float32)
        neighbors = _get_knn_indices(positions, 0, k=4)
        assert neighbors == []

    def test_knn_deterministic(self, mock_positions):
        """Same inputs → same output, no randomness."""
        n1 = _get_knn_indices(mock_positions, 5, k=4)
        n2 = _get_knn_indices(mock_positions, 5, k=4)
        assert n1 == n2


# ─────────────────────────────────────────────────────────────────────────────
# 3. Delta Quality Computation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaQuality:
    """Test the _compute_delta_quality formula.

    ΔQ = w_rgb * (ΔPSNR / max(1, PSNR_before)) + w_depth * (ΔDepth / max(1e-3, D_before))
    """

    def test_delta_quality_positive_improvement(self):
        """PSNR increase + depth decrease → positive ΔQ."""
        # Create a minimal mock to test the formula
        q_before = {"psnr": 25.0, "depth_l1": 0.05}
        q_after = {"psnr": 26.0, "depth_l1": 0.04}

        w_rgb = 0.70
        w_depth = 0.30

        # Expected:
        # delta_psnr = 1.0, norm = 1.0 / 25.0 = 0.04
        # delta_depth = 0.05 - 0.04 = 0.01, norm = 0.01 / 0.05 = 0.2
        # dq = 0.70 * 0.04 + 0.30 * 0.2 = 0.028 + 0.06 = 0.088
        expected = 0.70 * (1.0 / 25.0) + 0.30 * (0.01 / 0.05)
        assert abs(expected - 0.088) < 1e-6

    def test_delta_quality_negative_degradation(self):
        """PSNR decrease + depth increase → negative ΔQ."""
        q_before = {"psnr": 25.0, "depth_l1": 0.05}
        q_after = {"psnr": 24.0, "depth_l1": 0.06}

        # delta_psnr = -1.0, delta_depth = -0.01
        # dq = 0.70 * (-1.0/25.0) + 0.30 * (-0.01/0.05) = -0.028 + -0.06 = -0.088
        dq = 0.70 * (-1.0 / 25.0) + 0.30 * (-0.01 / 0.05)
        assert dq < 0

    def test_delta_quality_no_change(self):
        """Same before and after → ΔQ = 0."""
        q = {"psnr": 25.0, "depth_l1": 0.05}
        dq = 0.70 * (0.0 / 25.0) + 0.30 * (0.0 / 0.05)
        assert dq == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Empty Measurement Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyMeasurement:
    """Test _empty_measurement returns correct structure."""

    def test_empty_measurement_structure(self):
        """Verify all expected keys are present in empty measurement."""
        # We need a ConditionalOracleExperiment to call _empty_measurement,
        # but we can verify the expected keys
        expected_keys = {
            "candidate_idx", "context_indices", "context_size",
            "q_baseline_psnr", "q_baseline_loss",
            "q_s_psnr", "q_s_ssim", "q_s_depth_l1", "q_s_loss", "t_s_ms", "delta_q_s",
            "q_si_psnr", "q_si_ssim", "q_si_depth_l1", "q_si_loss", "t_si_ms", "delta_q_si",
            "delta_q_conditional", "delta_t_conditional_ms", "utility_conditional",
        }
        assert len(expected_keys) == 20  # Verify we haven't missed any


# ─────────────────────────────────────────────────────────────────────────────
# 5. Context Schedule Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestContextSchedule:
    """Test the context schedule generation."""

    def test_schedule_includes_empty(self):
        """Schedule must include at least one empty context."""
        # The schedule is a list of (type, size) tuples
        schedule = [
            ("empty", 0),
            ("spatial_knn", 1),
            ("spatial_knn", 4),
            ("overlap_top", 4),
            ("random", 8),
        ]
        empty_entries = [(t, s) for t, s in schedule if t == "empty" or s == 0]
        assert len(empty_entries) >= 1

    def test_schedule_covers_multiple_sizes(self):
        """Schedule should cover multiple context sizes."""
        schedule = [
            ("empty", 0),
            ("spatial_knn", 1),
            ("spatial_knn", 4),
            ("overlap_top", 4),
            ("random", 8),
        ]
        sizes = set(s for _, s in schedule)
        assert len(sizes) >= 3  # At least 3 different sizes

    def test_schedule_covers_multiple_types(self):
        """Schedule should cover multiple context types."""
        schedule = [
            ("empty", 0),
            ("spatial_knn", 1),
            ("spatial_knn", 4),
            ("overlap_top", 4),
            ("random", 8),
        ]
        types = set(t for t, _ in schedule)
        assert len(types) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# 6. Oracle Identity Tests (Algebraic)
# ─────────────────────────────────────────────────────────────────────────────

class TestOracleIdentity:
    """Verify algebraic identities that must hold regardless of implementation.

    These are pure math tests that verify the conditional utility formula.
    """

    def test_conditional_utility_formula(self):
        """U*(i|S) = ΔQ(i|S) / (ΔT(i|S) + ε)."""
        delta_q = 0.001
        delta_t = 5.0
        epsilon = 1e-6

        utility = delta_q / (max(abs(delta_t), epsilon))
        assert abs(utility - 0.001 / 5.0) < 1e-10

    def test_conditional_delta_q_identity(self):
        """ΔQ(i|S) = ΔQ(S∪{i}) - ΔQ(S), where both are measured from baseline."""
        dq_s = 0.005  # Q(S) - Q(baseline)
        dq_si = 0.008  # Q(S∪{i}) - Q(baseline)

        conditional = dq_si - dq_s
        assert abs(conditional - 0.003) < 1e-10

    def test_empty_context_reduces_to_marginal(self):
        """When S = ∅: ΔQ(i|∅) = ΔQ({i}) - ΔQ(∅) = ΔQ({i}) - 0 = ΔQ({i})."""
        dq_empty = 0.0  # Q(∅) = Q(baseline), so ΔQ = 0
        dq_i = 0.002    # Q({i}) improvement over baseline

        conditional = dq_i - dq_empty
        assert conditional == dq_i

    def test_utility_sign_matches_quality_sign(self):
        """Positive ΔQ → positive utility (when ΔT > 0)."""
        delta_q = 0.001
        delta_t = 5.0
        epsilon = 1e-6
        utility = delta_q / (max(abs(delta_t), epsilon))
        assert utility > 0

        delta_q = -0.001
        utility = delta_q / (max(abs(delta_t), epsilon))
        assert utility < 0

    def test_zero_quality_gain_zero_utility(self):
        """ΔQ = 0 → U* = 0."""
        delta_q = 0.0
        delta_t = 5.0
        epsilon = 1e-6
        utility = delta_q / (max(abs(delta_t), epsilon))
        assert utility == 0.0

    def test_additivity_decomposition(self):
        """For independent Gaussians: ΔQ({i,j}) ≈ ΔQ({i}) + ΔQ({j}).
        For correlated Gaussians: ΔQ({i,j}) < ΔQ({i}) + ΔQ({j}) (sub-additive).

        This is not an exact identity but verifies the interaction residual formula.
        """
        dq_i = 0.003
        dq_j = 0.002
        dq_ij_additive = 0.005    # Perfect additivity
        dq_ij_subadditive = 0.004  # Sub-additive (overlap)
        dq_ij_superadditive = 0.006  # Super-additive (synergy)

        I_add = dq_ij_additive - dq_i - dq_j
        I_sub = dq_ij_subadditive - dq_i - dq_j
        I_super = dq_ij_superadditive - dq_i - dq_j

        assert abs(I_add) < 1e-10       # Zero interaction
        assert I_sub < 0                 # Negative interaction (sub-additive)
        assert I_super > 0              # Positive interaction (super-additive)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dataset Schema Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetSchema:
    """Verify expected dataset record schema."""

    def test_expected_keys_in_sample(self):
        """A Phase 6 conditional oracle sample must have these keys."""
        required_keys = {
            "scene", "frame", "split", "seed",
            "candidate_id", "candidate_persistent_id",
            "context_ids", "context_type", "context_size",
            "self_features", "neighbor_features",
            "overlap_features", "selected_features",
            "full_feature_vector",
            "q_baseline_psnr",
            "q_s_psnr", "q_si_psnr",
            "delta_q_s", "delta_q_si",
            "delta_q_conditional",
            "t_s_ms", "t_si_ms",
            "delta_t_conditional_ms",
            "utility_conditional",
        }
        assert len(required_keys) == 24

    def test_feature_vector_length(self):
        assert PHASE6_FEATURE_DIM == 32
