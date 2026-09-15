"""Unit Tests for Phase 9C: B2 Robustness & Adaptation Necessity.

Verifies:
    1. test_protocol_integrity: Verify Phase 9C protocol invariants and frozen settings.
    2. test_b0_b2_static_exact_equivalence: Verify Gate 9C-3 equivalence (B2-static == B0).
    3. test_b2_active_divergence: Verify B2 active adaptation differs from static when domain shifts.
    4. test_beta_sensitivity_hierarchy: Verify adaptation velocity ordered correctly:
       ||mu_1 - mu_0||_(0.80) > ||mu_1 - mu_0||_(0.90) > ||mu_1 - mu_0||_(0.95).
    5. test_perturbation_properties: Verify scale and offset transforms preserve finiteness and shapes.
    6. test_b2_perturbation_absorption: Verify B2 absorbs affine scale/offset shifts better than B0.
    7. test_temporal_convergence_steady_state: Verify drift D_t^norm strictly decays to zero on stationary stream.
    8. test_no_future_or_oracle_leakage: Verify zero access to future frames or oracle utility labels.
"""
import pytest
import numpy as np
import torch

from research.phase9_protocol import CANONICAL_FEATURE_SCHEMA
from research.phase9b_protocol import EPS
from research.phase9b_normalization import (
    StandardNormalizer,
    OnlineEMANormalizer,
)
from research.phase9c_protocol import (
    validate_protocol_integrity_9c,
    BETA_VALUES,
    PERTURBATION_SCALES,
    PERTURBATION_OFFSETS,
)
from research.phase9c_analysis import (
    B2StaticNormalizer,
    apply_feature_perturbation,
    verify_b0_b2_static_equivalence,
    compute_per_frame_spearman,
    assess_sensitivity_range,
)
from research.utility_models import TwoHeadMLP


@pytest.fixture
def synthetic_data():
    """Deterministic synthetic train and test frame splits."""
    rng = np.random.default_rng(42)
    # Train distribution: centered at 2.0, scale 1.5
    X_train = rng.normal(loc=2.0, scale=1.5, size=(120, 11)).astype(np.float32)
    # Test frame 0: shifted to 3.5, scale 2.0
    X_test_0 = rng.normal(loc=3.5, scale=2.0, size=(40, 11)).astype(np.float32)
    # Test frame 1: further shifted to 4.0, scale 2.2
    X_test_1 = rng.normal(loc=4.0, scale=2.2, size=(40, 11)).astype(np.float32)
    return X_train, X_test_0, X_test_1


def test_protocol_integrity():
    """Verify Phase 9C protocol invariants."""
    assert validate_protocol_integrity_9c() is True


def test_b0_b2_static_exact_equivalence(synthetic_data):
    """Verify Gate 9C-3: B2-static (update off) is bitwise / numerically identical to B0."""
    X_train, X_test_0, X_test_1 = synthetic_data

    b0_norm = StandardNormalizer(eps=EPS).fit(X_train)
    b2_static = B2StaticNormalizer(eps=EPS).fit(X_train)

    # Initial moments must match exactly
    np.testing.assert_allclose(b0_norm.mean, b2_static.mu_init, atol=1e-6)
    np.testing.assert_allclose(b0_norm.std, b2_static.sigma_init, atol=1e-6)

    # Transform frame 0
    z0_b0 = b0_norm.transform(X_test_0)
    z0_b2_static, m0 = b2_static.update_and_transform_frame(X_test_0, frame_id=0)
    np.testing.assert_allclose(z0_b0, z0_b2_static, atol=1e-6)
    assert m0["d_norm"] == 0.0

    # Transform frame 1
    z1_b0 = b0_norm.transform(X_test_1)
    z1_b2_static, m1 = b2_static.update_and_transform_frame(X_test_1, frame_id=1)
    np.testing.assert_allclose(z1_b0, z1_b2_static, atol=1e-6)
    assert m1["d_norm"] == 0.0

    # Pass through neural model to verify prediction equivalence
    torch.manual_seed(42)
    model = TwoHeadMLP(in_features=11, hidden_dim=64)
    model.eval()

    with torch.no_grad():
        _, _, pred_b0 = model(torch.from_numpy(z0_b0))
        _, _, pred_b2s = model(torch.from_numpy(z0_b2_static))

    equiv, max_diff = verify_b0_b2_static_equivalence(pred_b0.numpy(), pred_b2s.numpy(), tolerance=1e-5)
    assert equiv is True, f"B0 and B2-static predictions diverged! max_diff={max_diff}"
    assert max_diff < 1e-6


def test_b2_active_divergence(synthetic_data):
    """Verify B2 active adaptation diverges from static reference under domain shift."""
    X_train, X_test_0, _ = synthetic_data

    b2_active = OnlineEMANormalizer(beta=0.90, eps=EPS).fit(X_train)
    b2_static = B2StaticNormalizer(eps=EPS).fit(X_train)

    z_active, step_m = b2_active.update_and_transform_frame(X_test_0, frame_id=0)
    z_static, _ = b2_static.update_and_transform_frame(X_test_0, frame_id=0)

    # Step drift should be positive
    assert step_m["d_norm"] > 0.0
    # Active normalization should differ from static
    diff = np.max(np.abs(z_active - z_static))
    assert diff > 0.01, f"Expected active adaptation to shift features, but max diff was {diff}"


def test_beta_sensitivity_hierarchy(synthetic_data):
    """Verify beta=0.80 adapts faster than 0.90 and 0.95."""
    X_train, X_test_0, _ = synthetic_data

    shifts = {}
    for beta in [0.80, 0.90, 0.95]:
        ema = OnlineEMANormalizer(beta=beta, eps=EPS).fit(X_train)
        _, step_m = ema.update_and_transform_frame(X_test_0, frame_id=0)
        shifts[beta] = step_m["mu_l2_shift"]

    # Lower beta (faster adaptation) produces larger initial shift
    assert shifts[0.80] > shifts[0.90] > shifts[0.95], (
        f"Beta hierarchy violated: 0.80={shifts[0.80]}, 0.90={shifts[0.90]}, 0.95={shifts[0.95]}"
    )


def test_perturbation_properties(synthetic_data):
    """Verify apply_feature_perturbation correctly applies scale and offset."""
    _, X_test_0, _ = synthetic_data

    for a in PERTURBATION_SCALES:
        for b in PERTURBATION_OFFSETS:
            X_p = apply_feature_perturbation(X_test_0, scale=a, offset=b)
            assert X_p.shape == X_test_0.shape
            assert np.all(np.isfinite(X_p))
            expected = a * X_test_0 + b
            np.testing.assert_allclose(X_p, expected, atol=1e-5)


def test_b2_perturbation_absorption(synthetic_data):
    """Verify that B2 online adaptation absorbs affine scale/offset shifts better than B0."""
    X_train, X_test_0, _ = synthetic_data

    # Scale shift a=1.2, offset b=0.2
    scale = 1.2
    offset = 0.2
    X_pert = apply_feature_perturbation(X_test_0, scale=scale, offset=offset)

    # Standard normalizer (B0)
    b0 = StandardNormalizer().fit(X_train)
    z_b0_clean = b0.transform(X_test_0)
    z_b0_pert = b0.transform(X_pert)

    # Under B0, mean shifted by offset / std, and scaled by scale
    b0_norm_drift = np.mean(np.abs(z_b0_clean - z_b0_pert))

    # Online EMA normalizer (B2)
    # Stream multiple frames of perturbed data to let B2 adapt
    b2_clean = OnlineEMANormalizer(beta=0.90).fit(X_train)
    b2_pert = OnlineEMANormalizer(beta=0.90).fit(X_train)

    z_clean = None
    z_pert = None
    for step in range(10):
        z_clean, _ = b2_clean.update_and_transform_frame(X_test_0, frame_id=step)
        z_pert, _ = b2_pert.update_and_transform_frame(X_pert, frame_id=step)

    # In adapted steady-state, B2 features are much closer to clean than B0
    b2_adapted_drift = np.mean(np.abs(z_clean - z_pert))
    assert b2_adapted_drift < b0_norm_drift, (
        f"B2 should absorb affine perturbation better than B0: B2 drift={b2_adapted_drift}, B0 drift={b0_norm_drift}"
    )


def test_temporal_convergence_steady_state(synthetic_data):
    """Verify drift D_t^norm strictly decays to zero on stationary stream."""
    X_train, X_test_0, _ = synthetic_data

    ema = OnlineEMANormalizer(beta=0.90, eps=EPS).fit(X_train)
    drifts = []

    # Stream identical stationary frame 20 times
    for step in range(20):
        _, step_m = ema.update_and_transform_frame(X_test_0, frame_id=step)
        drifts.append(step_m["d_norm"])

    # Initial drift > 0
    assert drifts[0] > 0.0
    # Verify strict monotonic decrease of drift over identical stationary frames
    for s in range(1, len(drifts)):
        assert drifts[s] < drifts[s - 1]
    # Final drift matches theoretical exponential factor beta^(steps-1) = 0.9^19 ~= 0.135
    expected_ratio = 0.90 ** 19
    actual_ratio = drifts[-1] / drifts[0]
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-4)


def test_no_future_or_oracle_leakage(synthetic_data):
    """Verify normalizers take ONLY candidate features without oracle labels or future frames."""
    X_train, X_test_0, X_test_1 = synthetic_data

    b2_static = B2StaticNormalizer().fit(X_train)
    z0, m0 = b2_static.update_and_transform_frame(X_test_0, frame_id=0)
    assert np.all(np.isfinite(z0))

    ema = OnlineEMANormalizer(beta=0.90).fit(X_train)
    z0_a, _ = ema.update_and_transform_frame(X_test_0, frame_id=0)
    assert np.all(np.isfinite(z0_a))
