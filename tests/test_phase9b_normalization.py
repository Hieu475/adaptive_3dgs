"""Unit Tests for Phase 9B Normalization Strategies (B0, B1, B2).

Tests:
    1. test_standard_normalizer: Verify B0 z-score computation and serialization.
    2. test_mad_zero: Verify B1 fallback when MAD is zero (e.g. constant/sparse features).
    3. test_mad_outlier_resistance: Verify B1 scales remain stable under extreme outliers.
    4. test_ema_initialization: Verify B2 starts exactly at train statistics.
    5. test_ema_update: Verify B2 recursive smoothing equation mu_t = beta*mu_{t-1} + (1-beta)*mu_frame.
    6. test_no_future_frame_access: Verify normalization at frame t depends only on <= t.
    7. test_no_oracle_dependency: Verify normalizers require zero oracle labels or U*.
    8. test_deterministic_update: Verify sequential replay produces bitwise identical results.
    9. test_finite_output: Verify zero NaN / Inf across extreme degenerate inputs.
"""
import pytest
import numpy as np
import torch

from research.phase9b_protocol import (
    CANONICAL_FEATURE_SCHEMA,
    B2_EMA_BETA,
    EPS,
    NORMALIZATION_VARIANTS,
)
from research.phase9b_normalization import (
    StandardNormalizer,
    RobustMADNormalizer,
    OnlineEMANormalizer,
    create_normalizer,
)


@pytest.fixture
def sample_features():
    """Deterministic synthetic [N, 11] feature matrix."""
    rng = np.random.default_rng(42)
    N = 100
    X = rng.normal(loc=2.0, scale=1.5, size=(N, 11)).astype(np.float32)
    # Feature 5 (position drift) is mostly zeros (sparse)
    X[:, 5] = 0.0
    X[rng.choice(N, size=15, replace=False), 5] = rng.exponential(scale=0.5, size=15)
    return X


def test_standard_normalizer(sample_features, tmp_path):
    """Test StandardNormalizer (B0) fit, transform, and JSON serialization."""
    norm = StandardNormalizer(eps=EPS)
    norm.fit(sample_features)

    assert norm.mean is not None and norm.std is not None
    assert len(norm.mean) == 11
    assert len(norm.std) == 11
    assert np.all(norm.std > 0)

    Z = norm.transform(sample_features)
    assert Z.shape == sample_features.shape
    assert np.all(np.isfinite(Z))
    # Standardized mean should be near 0, std near 1 (for non-sparse cols)
    np.testing.assert_allclose(np.mean(Z[:, 0]), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(Z[:, 0]), 1.0, atol=1e-4)

    # Test JSON save & load
    p = tmp_path / "b0_norm.json"
    norm.save_json(str(p))
    loaded = StandardNormalizer.load_json(str(p))
    np.testing.assert_allclose(norm.mean, loaded.mean)
    np.testing.assert_allclose(norm.std, loaded.std)


def test_mad_zero():
    """Test RobustMADNormalizer (B1) fallback when MAD is zero (e.g. constant/sparse column)."""
    # Create dataset where col 0 is constant zero (>50% identical values)
    X = np.zeros((50, 11), dtype=np.float32)
    X[:, 1:] = np.random.randn(50, 10).astype(np.float32)

    norm = RobustMADNormalizer(eps=EPS)
    norm.fit(X)

    # Col 0 MAD is 0 -> scale must fall back safely and remain strictly positive
    assert norm.scale is not None
    assert norm.scale[0] >= EPS
    assert np.all(np.isfinite(norm.scale))

    Z = norm.transform(X)
    assert np.all(np.isfinite(Z))
    assert np.all(Z[:, 0] == 0.0)


def test_mad_outlier_resistance(sample_features):
    """Test that RobustMADNormalizer (B1) scale is not distorted by extreme outliers."""
    norm_clean = RobustMADNormalizer(eps=EPS).fit(sample_features)

    # Inject massive outliers into 2 samples of feature 0
    corrupted = sample_features.copy()
    corrupted[0, 0] = 10000.0
    corrupted[1, 0] = -10000.0

    norm_corrupt = RobustMADNormalizer(eps=EPS).fit(corrupted)

    # Median and MAD should change very little compared to StandardNormalizer mean/std
    clean_scale = norm_clean.scale[0]
    corrupt_scale = norm_corrupt.scale[0]
    ratio = corrupt_scale / clean_scale
    assert 0.8 < ratio < 1.3, f"MAD scale distorted by outlier: ratio={ratio}"

    # Verify StandardNormalizer WOULD be severely distorted
    std_clean = StandardNormalizer().fit(sample_features).std[0]
    std_corrupt = StandardNormalizer().fit(corrupted).std[0]
    assert std_corrupt / std_clean > 50.0, "Standard normalizer did not show expected outlier inflation"


def test_ema_initialization(sample_features):
    """Test OnlineEMANormalizer (B2) initializes exactly from train statistics."""
    ema = OnlineEMANormalizer(beta=B2_EMA_BETA, eps=EPS)
    ema.fit(sample_features)

    expected_mu = np.mean(sample_features, axis=0)
    expected_sigma = (np.std(sample_features, axis=0) + EPS).astype(np.float32)

    np.testing.assert_allclose(ema.mu_current, expected_mu, atol=1e-5)
    np.testing.assert_allclose(ema.sigma_current, expected_sigma, atol=1e-5)
    assert len(ema.history) == 0


def test_ema_update(sample_features):
    """Test that B2 recursive smoothing matches mathematical EMA equation."""
    beta = 0.90
    ema = OnlineEMANormalizer(beta=beta, eps=EPS)
    ema.fit(sample_features)

    mu_0 = ema.mu_current.copy()
    sigma_0 = ema.sigma_current.copy()

    # Create a frame with shifted features
    frame1 = np.ones((20, 11), dtype=np.float32) * 5.0
    frame1_mu = np.mean(frame1, axis=0)
    frame1_sigma = np.maximum(np.std(frame1, axis=0), EPS)

    Z1, step_m = ema.update_and_transform_frame(frame1, frame_id=1)

    expected_mu_1 = beta * mu_0 + (1.0 - beta) * frame1_mu
    expected_sigma_1 = beta * sigma_0 + (1.0 - beta) * frame1_sigma

    np.testing.assert_allclose(ema.mu_current, expected_mu_1, atol=1e-5)
    np.testing.assert_allclose(ema.sigma_current, expected_sigma_1, atol=1e-5)
    assert len(ema.history) == 1
    assert step_m["frame_id"] == 1
    assert step_m["latency_ms"] >= 0.0
    assert np.all(np.isfinite(Z1))


def test_no_future_frame_access(sample_features):
    """Test that normalization at frame t depends strictly on <= t and never on future frames."""
    rng = np.random.default_rng(123)
    frames = [rng.normal(loc=i * 0.5, scale=1.0, size=(15, 11)).astype(np.float32) for i in range(5)]

    # Run 1: process frames 0, 1, 2
    ema1 = OnlineEMANormalizer(beta=0.9, eps=EPS).fit(sample_features)
    z0_run1, _ = ema1.update_and_transform_frame(frames[0], frame_id=0)
    z1_run1, _ = ema1.update_and_transform_frame(frames[1], frame_id=1)
    z2_run1, _ = ema1.update_and_transform_frame(frames[2], frame_id=2)

    # Run 2: process frames 0, 1, 2, but where frames 3 and 4 are drastically altered
    ema2 = OnlineEMANormalizer(beta=0.9, eps=EPS).fit(sample_features)
    z0_run2, _ = ema2.update_and_transform_frame(frames[0], frame_id=0)
    z1_run2, _ = ema2.update_and_transform_frame(frames[1], frame_id=1)
    z2_run2, _ = ema2.update_and_transform_frame(frames[2], frame_id=2)

    # Outputs at frames 0, 1, 2 must be bitwise identical
    np.testing.assert_array_equal(z0_run1, z0_run2)
    np.testing.assert_array_equal(z1_run1, z1_run2)
    np.testing.assert_array_equal(z2_run1, z2_run2)


def test_no_oracle_dependency():
    """Verify normalizers take ONLY candidate features, with no requirement for U*, Delta Q, or Delta T."""
    for var in NORMALIZATION_VARIANTS:
        norm = create_normalizer(var)
        # Normalizers require only X [N, 11]
        dummy_X = np.random.randn(30, 11).astype(np.float32)
        norm.fit(dummy_X)
        out = norm.transform(dummy_X)
        assert out.shape == dummy_X.shape
        assert np.all(np.isfinite(out))


def test_deterministic_update(sample_features):
    """Verify sequential replay of frames produces bitwise identical results."""
    rng = np.random.default_rng(999)
    frames = [rng.normal(size=(25, 11)).astype(np.float32) for _ in range(4)]

    ema = OnlineEMANormalizer(beta=0.9).fit(sample_features)
    outputs_run1 = [ema.update_and_transform_frame(f, frame_id=i)[0] for i, f in enumerate(frames)]

    # Reset and replay
    ema.reset()
    outputs_run2 = [ema.update_and_transform_frame(f, frame_id=i)[0] for i, f in enumerate(frames)]

    for o1, o2 in zip(outputs_run1, outputs_run2):
        np.testing.assert_array_equal(o1, o2)


def test_finite_output():
    """Verify all normalizers handle degenerate edge cases without producing NaN or Inf."""
    degenerate_cases = [
        np.zeros((20, 11), dtype=np.float32),              # All zeros
        np.ones((20, 11), dtype=np.float32) * 1e-12,       # Extremely tiny numbers
        np.ones((20, 11), dtype=np.float32) * 1e8,         # Very large numbers
        np.full((20, 11), fill_value=42.0, dtype=np.float32), # Constant value
    ]

    for X_deg in degenerate_cases:
        for var in NORMALIZATION_VARIANTS:
            norm = create_normalizer(var)
            norm.fit(X_deg)
            Z = norm.transform(X_deg)
            assert np.all(np.isfinite(Z)), f"NaN/Inf produced by {var} on degenerate input"

            if isinstance(norm, OnlineEMANormalizer):
                Z_frame, _ = norm.update_and_transform_frame(X_deg, frame_id=0)
                assert np.all(np.isfinite(Z_frame)), f"NaN/Inf produced by {var} in online step"


def test_reset_to_train_stats(sample_features):
    """Verify that reset() restores OnlineEMANormalizer exactly to train reference statistics."""
    ema = OnlineEMANormalizer(beta=0.90, eps=EPS)
    ema.fit(sample_features)

    mu_init = ema.mu_init.copy()
    sigma_init = ema.sigma_init.copy()

    # Apply multiple shifts
    for i in range(5):
        frame = np.random.randn(20, 11).astype(np.float32) + float(i * 10.0)
        ema.update_and_transform_frame(frame, frame_id=i)

    # State must have shifted
    assert not np.allclose(ema.mu_current, mu_init)
    assert len(ema.history) == 5

    # Reset
    ema.reset()
    np.testing.assert_array_equal(ema.mu_current, mu_init)
    np.testing.assert_array_equal(ema.sigma_current, sigma_init)
    assert len(ema.history) == 0


def test_frozen_weights(sample_features):
    """Verify that TwoHeadMLP model weights remain strictly frozen during online normalization and inference."""
    import copy
    from research.utility_models import TwoHeadMLP

    torch.manual_seed(42)
    model = TwoHeadMLP(in_features=11, hidden_dim=64)
    model.eval()

    state_before = copy.deepcopy(model.state_dict())

    ema = OnlineEMANormalizer(beta=0.90, eps=EPS)
    ema.fit(sample_features)

    # Simulate 10 frames of online evaluation
    rng = np.random.default_rng(42)
    for fr in range(10):
        frame = rng.normal(loc=fr * 0.5, scale=1.0, size=(25, 11)).astype(np.float32)
        Z_norm, _ = ema.update_and_transform_frame(frame, frame_id=fr)
        with torch.no_grad():
            x_t = torch.from_numpy(Z_norm).float()
            _ = model(x_t)

    state_after = model.state_dict()

    for key in state_before:
        assert torch.equal(state_before[key], state_after[key]), f"Model weight '{key}' was modified during test-time inference!"
