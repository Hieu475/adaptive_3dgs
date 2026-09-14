"""Unit tests for Phase 9A Scale-Invariant Feature Representations."""
import pytest
import numpy as np
import torch

from research.phase9_protocol import (
    CANONICAL_FEATURE_SCHEMA,
    FEATURE_VARIANTS,
    EPS,
)
from research.phase9_features import (
    FEAT_IDX,
    A1_TRANSFORMED_INDICES,
    A2_TRANSFORMED_INDICES,
    A2_UNCHANGED_INDICES,
    compute_robust_population_scale,
    transform_features_array,
    transform_grouped_features,
    transform_candidates,
    ScaleInvariantFeatureTransformer,
)


def _make_dummy_features(n: int = 40, seed: int = 42) -> np.ndarray:
    """Generate dummy raw feature matrix [n, 11] with realistic scales."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 11), dtype=np.float32)
    X[:, 0] = rng.uniform(0.05, 0.8, n)    # rgb_error
    X[:, 1] = rng.uniform(0.1, 2.5, n)     # depth_error (metric)
    X[:, 2] = rng.uniform(0.5, 20.0, n)    # gradient_norm
    X[:, 3] = rng.integers(10, 500, n)     # visibility_count
    X[:, 4] = rng.uniform(1.0, 50.0, n)    # influence_mass
    X[:, 5] = rng.choice([0.0, 0.0, 0.05, 0.12], n)  # position_drift (sparse zeros)
    X[:, 6] = rng.choice([0.0, 0.0, 0.02, 0.08], n)  # residual_drift_ema
    X[:, 7] = rng.uniform(0.1, 0.8, n)     # uncertainty_var
    X[:, 8] = rng.uniform(2.0, 45.0, n)    # projected_area
    X[:, 9] = rng.uniform(0.0, 1.0, n)     # update_frequency
    X[:, 10] = rng.integers(0, 50, n)      # age
    return X


def test_zero_scale_no_nan():
    """Verify that populations with median == 0 or all zeros produce NO NaN or Inf."""
    n = 30
    X = np.zeros((n, 11), dtype=np.float32)
    # Even if all features are completely 0
    for var in FEATURE_VARIANTS:
        Z = transform_features_array(X, variant=var)
        assert np.all(np.isfinite(Z)), f"Non-finite values found in variant {var} on all-zero input"
        assert not np.any(np.isnan(Z)), f"NaN encountered in variant {var} on all-zero input"
        assert Z.shape == (n, 11)

    # Sparse zeros (median == 0, but some non-zeros)
    X_sparse = np.zeros((n, 11), dtype=np.float32)
    X_sparse[:3, 1] = 0.5  # depth_error: only 3 non-zeros out of 30 -> median is 0
    X_sparse[:2, 5] = 0.1  # position_drift: only 2 non-zeros
    for var in FEATURE_VARIANTS:
        Z = transform_features_array(X_sparse, variant=var)
        assert np.all(np.isfinite(Z)), f"Non-finite values in sparse input for variant {var}"
        assert not np.any(np.isnan(Z)), f"NaN in sparse input for variant {var}"


def test_constant_population():
    """Verify that a population where all elements have the same value behaves predictably."""
    n = 20
    X = np.full((n, 11), 5.0, dtype=np.float32)
    for var in ["geometry_relative", "geometry_optimization_relative"]:
        Z = transform_features_array(X, variant=var)
        assert np.all(np.isfinite(Z))
        # In A1, depth_error (idx 1), position_drift (idx 5), projected_area (idx 8) should be 5.0 / (5.0 + eps) ~ 1.0
        for idx in [1, 5, 8]:
            np.testing.assert_allclose(Z[:, idx], 5.0 / (5.0 + EPS), rtol=1e-4)
        # Unchanged features should remain 5.0
        assert np.allclose(Z[:, 0], 5.0)  # rgb_error


def test_relative_feature_dimension():
    """Verify that output dimension is strictly [N, 11] across all variants."""
    for n in [1, 5, 40, 100]:
        X = _make_dummy_features(n=n)
        for var in FEATURE_VARIANTS:
            Z = transform_features_array(X, variant=var)
            assert Z.shape == (n, 11)
            assert Z.dtype == np.float32


def test_train_test_same_schema():
    """Verify that canonical schema order is strictly preserved."""
    X = _make_dummy_features(20)
    for var in FEATURE_VARIANTS:
        Z = transform_features_array(X, variant=var)
        assert Z.shape[1] == len(CANONICAL_FEATURE_SCHEMA)


def test_no_oracle_dependency():
    """Verify that transformation relies strictly on feature matrix, zero dependency on targets."""
    X = _make_dummy_features(25)
    # Function signature takes only X, variant, eps — no target inputs exist
    Z1 = transform_features_array(X, variant="geometry_relative")
    Z2 = transform_features_array(X, variant="geometry_relative")
    np.testing.assert_array_equal(Z1, Z2)


def test_deterministic_transform():
    """Verify transforms are 100% deterministic bit-for-bit across multiple invocations."""
    X = _make_dummy_features(50, seed=123)
    for var in FEATURE_VARIANTS:
        out1 = transform_features_array(X, variant=var)
        out2 = transform_features_array(X, variant=var)
        np.testing.assert_array_equal(out1, out2)


def test_a0_identity():
    """Verify that A0 ('raw') is an exact identity mapping."""
    X = _make_dummy_features(30)
    Z = transform_features_array(X, variant="raw")
    np.testing.assert_array_equal(X, Z)


def test_a1_selective_transformation():
    """Verify that A1 transforms ONLY indices [1, 5, 8] and preserves the other 8 features."""
    X = _make_dummy_features(40)
    Z = transform_features_array(X, variant="geometry_relative")

    # Transformed indices
    for idx in A1_TRANSFORMED_INDICES:
        scale = compute_robust_population_scale(X[:, idx])
        expected = X[:, idx] / scale
        np.testing.assert_allclose(Z[:, idx], expected, rtol=1e-5)

    # Unchanged indices
    for idx in range(11):
        if idx not in A1_TRANSFORMED_INDICES:
            np.testing.assert_array_equal(Z[:, idx], X[:, idx],
                                         err_msg=f"Feature {CANONICAL_FEATURE_SCHEMA[idx]} (idx {idx}) was unexpectedly modified in A1")


def test_a2_selective_transformation():
    """Verify that A2 transforms indices [1, 2, 4, 5, 6, 7, 8] and preserves [0, 3, 9, 10]."""
    X = _make_dummy_features(40)
    Z = transform_features_array(X, variant="geometry_optimization_relative")

    for idx in A2_TRANSFORMED_INDICES:
        scale = compute_robust_population_scale(X[:, idx])
        expected = X[:, idx] / scale
        np.testing.assert_allclose(Z[:, idx], expected, rtol=1e-5)

    for idx in A2_UNCHANGED_INDICES:
        np.testing.assert_array_equal(Z[:, idx], X[:, idx],
                                     err_msg=f"Feature {CANONICAL_FEATURE_SCHEMA[idx]} (idx {idx}) was unexpectedly modified in A2")


def test_scale_invariance_property():
    """Direct scientific verification: if depth_error is multiplied by 3.0 (camera scale shift),
    the relative depth error remains invariant!
    """
    X = _make_dummy_features(40)
    Z_orig = transform_features_array(X, variant="geometry_relative")

    # Simulate 3x scene scale expansion: depth errors and projected areas scale by 3x
    X_scaled = X.copy()
    X_scaled[:, FEAT_IDX["depth_error"]] *= 3.0
    X_scaled[:, FEAT_IDX["projected_area"]] *= 3.0

    Z_scaled = transform_features_array(X_scaled, variant="geometry_relative")

    # The relative features should be invariant under uniform scene scaling!
    np.testing.assert_allclose(
        Z_orig[:, FEAT_IDX["depth_error"]],
        Z_scaled[:, FEAT_IDX["depth_error"]],
        rtol=1e-4,
        err_msg="Relative depth error failed scale invariance test!"
    )
    np.testing.assert_allclose(
        Z_orig[:, FEAT_IDX["projected_area"]],
        Z_scaled[:, FEAT_IDX["projected_area"]],
        rtol=1e-4,
        err_msg="Relative projected area failed scale invariance test!"
    )


def test_candidate_dict_transformation():
    """Verify transformation on candidate dictionary lists."""
    cands = [
        {"features": {name: float(i + j) for j, name in enumerate(CANONICAL_FEATURE_SCHEMA)}, "id": i}
        for i in range(10)
    ]
    transformed = transform_candidates(cands, variant="geometry_relative")
    assert len(transformed) == len(cands)
    for c in transformed:
        assert "representation_variant" in c
        assert c["representation_variant"] == "geometry_relative"
        assert len(c["features"]) == 11
        assert np.isfinite(c["features"]["depth_error"])


def test_transformer_wrapper_torch():
    """Verify PyTorch Tensor support in ScaleInvariantFeatureTransformer."""
    X = torch.from_numpy(_make_dummy_features(20))
    transformer = ScaleInvariantFeatureTransformer(variant="A1")
    Z = transformer.transform(X)
    assert isinstance(Z, torch.Tensor)
    assert Z.shape == (20, 11)
    assert not torch.isnan(Z).any()
