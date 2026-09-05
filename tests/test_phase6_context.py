"""Tests for Phase 6 Context Representation (research/phase6_context.py).

Test Categories:
    1. Determinism: Same inputs → same outputs (no randomness in context extraction).
    2. Sensitivity: Different S_t → different context for same candidate.
    3. No-leakage: No future-frame features leak into context.
    4. Shape/schema: Feature vector has correct dimensions and feature names.
    5. Edge cases: Empty model, single Gaussian, out-of-bounds index.
    6. Batch consistency: Batch output matches single-candidate output.
"""
import pytest
import numpy as np
import torch

from research.phase6_context import (
    build_neighbor_context,
    build_neighbor_context_batch,
    build_overlap_context,
    build_overlap_context_batch,
    build_selected_context,
    build_full_context,
    build_full_context_batch,
    ContextConfig,
    PHASE6_FEATURE_NAMES,
    PHASE6_FEATURE_DIM,
    NEIGHBOR_FEATURE_NAMES,
    OVERLAP_FEATURE_NAMES,
    SELECTED_FEATURE_NAMES,
    SELF_SLICE,
    NEIGHBOR_SLICE,
    OVERLAP_SLICE,
    SELECTED_SLICE,
    _empty_neighbor_context,
    _empty_overlap_context,
    _get_pixel_mask,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    """Test device (CPU for deterministic tests)."""
    return torch.device('cpu')


@pytest.fixture
def mock_positions(device):
    """Create 20 Gaussians in a regular grid for predictable KNN."""
    torch.manual_seed(42)
    # 4x5 grid in XY plane, Z = 0
    positions = []
    for i in range(4):
        for j in range(5):
            positions.append([float(i), float(j), 0.0])
    return torch.tensor(positions, dtype=torch.float32, device=device)


@pytest.fixture
def mock_features():
    """Create (20, 11) canonical features with known values."""
    np.random.seed(42)
    feats = np.random.rand(20, 11).astype(np.float32)
    # Make feature 0 (rgb_error) linearly increasing for easy verification
    feats[:, 0] = np.arange(20, dtype=np.float32) / 20.0
    return feats


@pytest.fixture
def mock_attribution(device):
    """Create simple attribution maps (H=10, W=10, K_top=4).

    Gaussian 0 influences pixels (0:5, 0:5) — top-left quadrant.
    Gaussian 1 influences pixels (3:8, 3:8) — overlapping center region.
    Gaussian 5 influences pixels (5:10, 5:10) — bottom-right (no overlap with 0).
    """
    H, W, K = 10, 10, 4
    contrib_indices = torch.full((H, W, K), -1, dtype=torch.long, device=device)
    contrib_weights = torch.zeros(H, W, K, dtype=torch.float32, device=device)

    # Gaussian 0: top-left
    contrib_indices[0:5, 0:5, 0] = 0
    contrib_weights[0:5, 0:5, 0] = 0.5

    # Gaussian 1: center (overlaps with 0 at (3:5, 3:5))
    contrib_indices[3:8, 3:8, 1] = 1
    contrib_weights[3:8, 3:8, 1] = 0.4

    # Gaussian 5: bottom-right (no overlap with 0)
    contrib_indices[5:10, 5:10, 2] = 5
    contrib_weights[5:10, 5:10, 2] = 0.3

    # Gaussian 2: small footprint at (0:2, 8:10)
    contrib_indices[0:2, 8:10, 3] = 2
    contrib_weights[0:2, 8:10, 3] = 0.2

    return contrib_indices, contrib_weights


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature Schema Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureSchema:
    """Verify feature names, dimensions, and slicing."""

    def test_feature_dim_is_32(self):
        assert PHASE6_FEATURE_DIM == 32

    def test_feature_names_count(self):
        assert len(PHASE6_FEATURE_NAMES) == 32

    def test_feature_names_unique(self):
        assert len(set(PHASE6_FEATURE_NAMES)) == 32

    def test_self_features_are_canonical(self):
        from research.utility_features import CANONICAL_FEATURE_NAMES
        assert PHASE6_FEATURE_NAMES[:11] == CANONICAL_FEATURE_NAMES

    def test_neighbor_feature_names(self):
        assert len(NEIGHBOR_FEATURE_NAMES) == 8

    def test_overlap_feature_names(self):
        assert len(OVERLAP_FEATURE_NAMES) == 5

    def test_selected_feature_names(self):
        assert len(SELECTED_FEATURE_NAMES) == 8

    def test_slices_cover_full_vector(self):
        assert SELF_SLICE == slice(0, 11)
        assert NEIGHBOR_SLICE == slice(11, 19)
        assert OVERLAP_SLICE == slice(19, 24)
        assert SELECTED_SLICE == slice(24, 32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Determinism Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """Context extraction must be deterministic: same inputs → same outputs."""

    def test_neighbor_context_deterministic(self, mock_positions, mock_features):
        ctx1 = build_neighbor_context(mock_positions, 5, mock_features, k=8)
        ctx2 = build_neighbor_context(mock_positions, 5, mock_features, k=8)

        for key in NEIGHBOR_FEATURE_NAMES:
            assert ctx1[key] == ctx2[key], f"Non-deterministic: {key}"

    def test_full_context_deterministic(self, mock_positions, mock_features, mock_attribution):
        contrib_indices, contrib_weights = mock_attribution

        ctx1 = build_full_context(
            mock_positions, 0, mock_features,
            selected_indices=[1, 2],
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
        )
        ctx2 = build_full_context(
            mock_positions, 0, mock_features,
            selected_indices=[1, 2],
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
        )

        np.testing.assert_array_equal(ctx1["full_vector"], ctx2["full_vector"])

    def test_overlap_context_deterministic(self, mock_attribution):
        contrib_indices, contrib_weights = mock_attribution

        ctx1 = build_overlap_context(0, [1, 5], contrib_indices, contrib_weights)
        ctx2 = build_overlap_context(0, [1, 5], contrib_indices, contrib_weights)

        for key in OVERLAP_FEATURE_NAMES:
            assert ctx1[key] == ctx2[key], f"Non-deterministic: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Context Sensitivity Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestContextSensitivity:
    """Different S_t → different context for the same candidate."""

    def test_different_selected_set_changes_context(self, mock_positions, mock_features):
        """Same candidate, different selected sets → different selected_features."""
        ctx_empty = build_selected_context(
            mock_positions, 5, selected_indices=[],
            all_features=mock_features,
        )
        ctx_with_1 = build_selected_context(
            mock_positions, 5, selected_indices=[1],
            all_features=mock_features,
        )
        ctx_with_many = build_selected_context(
            mock_positions, 5, selected_indices=[1, 2, 3, 10],
            all_features=mock_features,
        )

        # selected_count should differ
        assert ctx_empty["selected_count"] == 0.0
        assert ctx_with_1["selected_count"] == 1.0
        assert ctx_with_many["selected_count"] == 4.0

        # Distance should differ (different selected positions)
        assert ctx_with_1["candidate_selected_distance"] != ctx_with_many["candidate_selected_distance"]

    def test_full_vector_changes_with_selected_set(self, mock_positions, mock_features):
        """Full 32-dim vector should change when S_t changes."""
        ctx1 = build_full_context(
            mock_positions, 5, mock_features, selected_indices=[],
        )
        ctx2 = build_full_context(
            mock_positions, 5, mock_features, selected_indices=[1, 2, 3],
        )

        # Self and neighbor parts should be identical
        np.testing.assert_array_equal(
            ctx1["full_vector"][SELF_SLICE],
            ctx2["full_vector"][SELF_SLICE],
        )
        np.testing.assert_array_equal(
            ctx1["full_vector"][NEIGHBOR_SLICE],
            ctx2["full_vector"][NEIGHBOR_SLICE],
        )

        # Selected part should differ
        assert not np.array_equal(
            ctx1["full_vector"][SELECTED_SLICE],
            ctx2["full_vector"][SELECTED_SLICE],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Neighbor Context Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNeighborContext:
    """Tests for build_neighbor_context and batch variant."""

    def test_neighbor_count_bounded_by_k(self, mock_positions, mock_features):
        ctx = build_neighbor_context(mock_positions, 5, mock_features, k=8)
        assert ctx["neighbor_count"] == 8.0

    def test_neighbor_count_bounded_by_population(self, device):
        """With only 3 Gaussians, k=8 should return 2 neighbors."""
        positions = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                                 dtype=torch.float32, device=device)
        features = np.random.rand(3, 11).astype(np.float32)
        ctx = build_neighbor_context(positions, 0, features, k=8)
        assert ctx["neighbor_count"] == 2.0

    def test_single_gaussian_returns_empty(self, device):
        positions = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)
        features = np.random.rand(1, 11).astype(np.float32)
        ctx = build_neighbor_context(positions, 0, features, k=8)
        assert ctx == _empty_neighbor_context()

    def test_neighbor_mean_correctness(self, device):
        """Verify mean computation with known values."""
        positions = torch.tensor([
            [0, 0, 0],  # candidate
            [1, 0, 0],  # neighbor 1 (distance = 1)
            [0, 1, 0],  # neighbor 2 (distance = 1)
            [10, 10, 10],  # far away, should not be neighbor with k=2
        ], dtype=torch.float32, device=device)

        features = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],  # candidate
            [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],  # neighbor 1
            [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4],  # neighbor 2
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],  # far
        ], dtype=np.float32)

        ctx = build_neighbor_context(positions, 0, features, k=2)

        # Mean of neighbor rgb_error (idx 0): (0.2 + 0.4) / 2 = 0.3
        assert abs(ctx["neighbor_mean_rgb_error"] - 0.3) < 1e-5
        # Mean of neighbor depth_error (idx 1): (0.4 + 0.6) / 2 = 0.5
        assert abs(ctx["neighbor_mean_depth_error"] - 0.5) < 1e-5
        assert ctx["neighbor_count"] == 2.0

    def test_batch_matches_single(self, mock_positions, mock_features):
        """Batch neighbor context must match individual calls."""
        candidates = [0, 5, 10, 15]
        batch_result = build_neighbor_context_batch(
            mock_positions, candidates, mock_features, k=4
        )

        for idx in candidates:
            single = build_neighbor_context(mock_positions, idx, mock_features, k=4)
            for key in NEIGHBOR_FEATURE_NAMES:
                assert abs(batch_result[idx][key] - single[key]) < 1e-5, (
                    f"Batch mismatch for candidate {idx}, key {key}: "
                    f"{batch_result[idx][key]} != {single[key]}"
                )

    def test_oob_index_returns_empty(self, mock_positions, mock_features):
        ctx = build_neighbor_context(mock_positions, 999, mock_features, k=8)
        assert ctx == _empty_neighbor_context()

    def test_negative_index_returns_empty(self, mock_positions, mock_features):
        ctx = build_neighbor_context(mock_positions, -1, mock_features, k=8)
        assert ctx == _empty_neighbor_context()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Overlap Context Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlapContext:
    """Tests for build_overlap_context using tile-level attribution."""

    def test_overlap_with_overlapping_gaussian(self, mock_attribution):
        """Gaussian 0 and 1 share pixels at (3:5, 3:5) → non-zero overlap."""
        contrib_indices, contrib_weights = mock_attribution
        ctx = build_overlap_context(0, [1], contrib_indices, contrib_weights)

        assert ctx["mean_overlap"] > 0.0
        assert ctx["max_overlap"] > 0.0

    def test_no_overlap_with_distant_gaussian(self, mock_attribution):
        """Gaussian 0 (top-left) and 5 (bottom-right) → zero overlap."""
        contrib_indices, contrib_weights = mock_attribution
        ctx = build_overlap_context(0, [5], contrib_indices, contrib_weights)

        assert ctx["mean_overlap"] == 0.0
        assert ctx["max_overlap"] == 0.0
        assert ctx["high_overlap_count"] == 0.0

    def test_overlap_area_fraction_bounded(self, mock_attribution):
        """Overlap area fraction must be in [0, 1]."""
        contrib_indices, contrib_weights = mock_attribution
        ctx = build_overlap_context(0, [1, 5], contrib_indices, contrib_weights)

        assert 0.0 <= ctx["overlap_area_fraction"] <= 1.0

    def test_empty_neighbors_returns_zero(self, mock_attribution):
        contrib_indices, contrib_weights = mock_attribution
        ctx = build_overlap_context(0, [], contrib_indices, contrib_weights)
        assert ctx == _empty_overlap_context()

    def test_high_overlap_count_threshold(self, mock_attribution):
        """Gaussian 0 overlaps with 1 (IoU > 0.1) but not 5 (IoU = 0)."""
        contrib_indices, contrib_weights = mock_attribution
        ctx = build_overlap_context(
            0, [1, 5], contrib_indices, contrib_weights,
            overlap_threshold=0.1,
        )
        # Only 1 neighbor (Gaussian 1) should have overlap > 0.1
        assert ctx["high_overlap_count"] <= 1.0

    def test_batch_overlap_matches_single(self, mock_attribution):
        """Batch overlap must match individual calls."""
        contrib_indices, contrib_weights = mock_attribution
        candidates = [0, 1]
        neighbor_map = {0: [1, 5], 1: [0, 5]}

        batch = build_overlap_context_batch(
            candidates, neighbor_map,
            contrib_indices, contrib_weights,
        )

        for cand_idx in candidates:
            single = build_overlap_context(
                cand_idx, neighbor_map[cand_idx],
                contrib_indices, contrib_weights,
            )
            for key in OVERLAP_FEATURE_NAMES:
                assert abs(batch[cand_idx][key] - single[key]) < 1e-5, (
                    f"Batch mismatch for {cand_idx}, {key}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Selected-Set Context Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectedContext:
    """Tests for build_selected_context."""

    def test_empty_selected_returns_zeros(self, mock_positions, mock_features):
        ctx = build_selected_context(
            mock_positions, 5, selected_indices=[],
            all_features=mock_features,
        )
        assert ctx["selected_count"] == 0.0
        assert ctx["selected_mean_rgb_error"] == 0.0
        assert ctx["selected_budget_fraction"] == 0.0

    def test_budget_fraction_computation(self, mock_positions, mock_features):
        ctx = build_selected_context(
            mock_positions, 5, selected_indices=[1, 2],
            all_features=mock_features,
            total_budget=100.0,
            current_cost=50.0,
        )
        assert abs(ctx["selected_budget_fraction"] - 0.5) < 1e-5

    def test_budget_fraction_clipped_to_1(self, mock_positions, mock_features):
        ctx = build_selected_context(
            mock_positions, 5, selected_indices=[1],
            all_features=mock_features,
            total_budget=100.0,
            current_cost=150.0,
        )
        assert ctx["selected_budget_fraction"] == 1.0

    def test_selected_distance_positive(self, mock_positions, mock_features):
        """Distance from candidate to selected set should be positive."""
        ctx = build_selected_context(
            mock_positions, 0, selected_indices=[15, 19],
            all_features=mock_features,
        )
        assert ctx["candidate_selected_distance"] > 0.0

    def test_selected_density_with_multiple(self, mock_positions, mock_features):
        """Spatial density should be mean pairwise distance among selected."""
        ctx = build_selected_context(
            mock_positions, 0, selected_indices=[1, 2, 3],
            all_features=mock_features,
        )
        assert ctx["selected_spatial_density"] > 0.0

    def test_selected_density_single_is_zero(self, mock_positions, mock_features):
        """With only 1 selected, pairwise distance is undefined → 0."""
        ctx = build_selected_context(
            mock_positions, 0, selected_indices=[5],
            all_features=mock_features,
        )
        assert ctx["selected_spatial_density"] == 0.0

    def test_invalid_selected_indices_filtered(self, mock_positions, mock_features):
        """OOB selected indices should be filtered out."""
        ctx = build_selected_context(
            mock_positions, 0, selected_indices=[1, 999, -5],
            all_features=mock_features,
        )
        assert ctx["selected_count"] == 1.0  # only index 1 is valid


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full Context Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFullContext:
    """Tests for build_full_context and batch variant."""

    def test_output_shape(self, mock_positions, mock_features):
        ctx = build_full_context(mock_positions, 5, mock_features)
        assert ctx["full_vector"].shape == (32,)
        assert ctx["full_vector"].dtype == np.float32

    def test_self_features_match_input(self, mock_positions, mock_features):
        ctx = build_full_context(mock_positions, 5, mock_features)
        np.testing.assert_array_equal(
            ctx["self_features"], mock_features[5]
        )

    def test_feature_names_included(self, mock_positions, mock_features):
        ctx = build_full_context(mock_positions, 5, mock_features)
        assert ctx["feature_names"] == PHASE6_FEATURE_NAMES

    def test_no_nans_in_output(self, mock_positions, mock_features, mock_attribution):
        contrib_indices, contrib_weights = mock_attribution
        ctx = build_full_context(
            mock_positions, 0, mock_features,
            selected_indices=[1, 2],
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
        )
        assert not np.any(np.isnan(ctx["full_vector"]))

    def test_batch_matches_single(self, mock_positions, mock_features, mock_attribution):
        contrib_indices, contrib_weights = mock_attribution
        candidates = [0, 1, 5]
        selected = [2, 3]

        batch = build_full_context_batch(
            mock_positions, candidates, mock_features,
            selected_indices=selected,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
        )

        for idx in candidates:
            single = build_full_context(
                mock_positions, idx, mock_features,
                selected_indices=selected,
                contrib_indices=contrib_indices,
                contrib_weights=contrib_weights,
            )
            np.testing.assert_allclose(
                batch[idx]["full_vector"],
                single["full_vector"],
                rtol=1e-5,
                err_msg=f"Batch/single mismatch for candidate {idx}",
            )

    def test_oob_candidate_returns_zeros_self(self, mock_positions, mock_features):
        """OOB candidate should have zero self features."""
        ctx = build_full_context(mock_positions, 999, mock_features)
        np.testing.assert_array_equal(
            ctx["self_features"], np.zeros(11, dtype=np.float32)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. No-Leakage Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLeakage:
    """Verify that context extraction does not leak future information."""

    def test_context_uses_only_current_state(self, mock_positions, mock_features):
        """Context should only depend on current positions and features,
        not on any future frame data."""
        # Modify features for "future" frame
        future_features = mock_features.copy()
        future_features[:, 0] = 99.0  # Change rgb_error to huge value

        # Current context should NOT include future values
        ctx_current = build_full_context(
            mock_positions, 5, mock_features, selected_indices=[1]
        )
        ctx_future = build_full_context(
            mock_positions, 5, future_features, selected_indices=[1]
        )

        # Self features should differ (using different feature arrays)
        assert not np.array_equal(
            ctx_current["self_features"],
            ctx_future["self_features"],
        )

        # But both should be valid (no NaN)
        assert not np.any(np.isnan(ctx_current["full_vector"]))
        assert not np.any(np.isnan(ctx_future["full_vector"]))

    def test_neighbor_context_independent_of_selected(self, mock_positions, mock_features):
        """Neighbor context should NOT change when S_t changes.
        Only selected_features should change."""
        ctx_empty_s = build_full_context(
            mock_positions, 5, mock_features, selected_indices=[],
        )
        ctx_full_s = build_full_context(
            mock_positions, 5, mock_features, selected_indices=[1, 2, 3],
        )

        # Neighbor features (part of context that depends only on spatial neighborhood)
        np.testing.assert_array_equal(
            ctx_empty_s["full_vector"][NEIGHBOR_SLICE],
            ctx_full_s["full_vector"][NEIGHBOR_SLICE],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Pixel Mask Helper Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPixelMask:
    """Tests for _get_pixel_mask helper."""

    def test_mask_shape(self, mock_attribution):
        contrib_indices, contrib_weights = mock_attribution
        mask = _get_pixel_mask(0, contrib_indices, contrib_weights, 0.01)
        assert mask.shape == (10, 10)
        assert mask.dtype == torch.bool

    def test_mask_correct_pixels(self, mock_attribution):
        """Gaussian 0 influences (0:5, 0:5) with weight 0.5 > threshold 0.01."""
        contrib_indices, contrib_weights = mock_attribution
        mask = _get_pixel_mask(0, contrib_indices, contrib_weights, 0.01)

        # Top-left quadrant should be True
        assert mask[0, 0].item() == True
        assert mask[4, 4].item() == True

        # Bottom-right should be False (Gaussian 0 not there)
        assert mask[9, 9].item() == False

    def test_mask_threshold_filtering(self, mock_attribution):
        """High threshold should filter out low-weight pixels."""
        contrib_indices, contrib_weights = mock_attribution
        mask_low = _get_pixel_mask(0, contrib_indices, contrib_weights, 0.01)
        mask_high = _get_pixel_mask(0, contrib_indices, contrib_weights, 0.9)

        assert mask_low.sum() >= mask_high.sum()

    def test_absent_gaussian_empty_mask(self, mock_attribution):
        """Gaussian 99 doesn't exist in attribution → empty mask."""
        contrib_indices, contrib_weights = mock_attribution
        mask = _get_pixel_mask(99, contrib_indices, contrib_weights, 0.01)
        assert mask.sum().item() == 0
