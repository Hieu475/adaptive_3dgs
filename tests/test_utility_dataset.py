"""Unit tests for Phase 4 UtilityDataset, canonical features, and train-only normalizer."""
import pytest
import numpy as np
import torch

from research.utility_features import (
    CANONICAL_FEATURE_NAMES,
    CANONICAL_FEATURE_SPECS,
    extract_feature_vector,
    ABLATION_SUBSETS,
)
from research.utility_dataset import (
    FeatureNormalizer,
    UtilityDataset,
    SampleMetadata,
    load_canonical_oracle_dataset,
    prepare_normalized_splits,
)


def test_canonical_features_count_and_schema():
    assert len(CANONICAL_FEATURE_NAMES) == 11
    assert len(CANONICAL_FEATURE_SPECS) == 11
    assert "temporal_drift" not in CANONICAL_FEATURE_NAMES
    assert "position_drift" in CANONICAL_FEATURE_NAMES
    assert "residual_drift_ema" in CANONICAL_FEATURE_NAMES
    assert "update_frequency" in CANONICAL_FEATURE_NAMES
    assert "age" in CANONICAL_FEATURE_NAMES
    for spec in CANONICAL_FEATURE_SPECS:
        assert hasattr(spec, "normalization")
        assert len(spec.normalization) > 0


def test_extract_feature_vector():
    sample_row = {
        "features": {
            "rgb_error": 0.5,
            "depth_error": 1.2,
            "gradient_norm": 3.4,
            "visibility_count": 42.0,
            "influence_mass": 10.0,
            "position_drift": 0.05,
            "residual_drift_ema": 0.12,
            "uncertainty_var": 0.08,
            "projected_area": 15.0,
            "update_frequency": 0.8,
            "age": 5,
        }
    }
    vec = extract_feature_vector(sample_row)
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 11
    assert np.isclose(vec[0], 0.5)
    assert np.isclose(vec[3], 42.0)
    assert np.isclose(vec[10], 5.0)


def test_normalizer_train_only_fit():
    X_train = np.array([[1.0, 10.0], [3.0, 30.0]], dtype=np.float32)
    X_test = np.array([[2.0, 20.0]], dtype=np.float32)

    norm = FeatureNormalizer()
    norm.fit(X_train)

    assert np.allclose(norm.mean, [2.0, 20.0])
    assert np.allclose(norm.std, [1.0 + 1e-6, 10.0 + 1e-6])

    X_train_norm = norm.transform(X_train)
    assert np.allclose(np.mean(X_train_norm, axis=0), [0.0, 0.0], atol=1e-5)

    X_test_norm = norm.transform(X_test)
    assert np.allclose(X_test_norm, [[0.0, 0.0]], atol=1e-5)


def test_dataset_loading_and_splitting():
    dataset = load_canonical_oracle_dataset()
    assert len(dataset) > 0
    assert dataset.features.shape[1] == 11
    assert len(dataset.delta_q) == len(dataset)
    assert len(dataset.delta_t) == len(dataset)
    assert len(dataset.utility) == len(dataset)
    assert len(dataset.seeds) == len(dataset)
    assert len(dataset.frames) == len(dataset)
    assert len(dataset.scenes) == len(dataset)
    assert len(dataset.geometry_strata) == len(dataset)
    assert len(dataset.splits) == len(dataset)

    train_ds = dataset.get_split("train")
    val_ds = dataset.get_split("validation")
    test_ds = dataset.get_split("cross_scene_test")

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    assert len(train_ds) + len(val_ds) + len(test_ds) == len(dataset)


def test_prepare_normalized_splits_train_only_leakage_check():
    dataset = load_canonical_oracle_dataset()
    train_norm, val_norm, test_norm, normalizer = prepare_normalized_splits(dataset=dataset)

    # Train mean should be approximately 0.0 and std 1.0
    train_mean = train_norm.features.mean(dim=0).numpy()
    assert np.allclose(train_mean, 0.0, atol=1e-4)

    # Normalizer should record n_samples_fit equal to train length
    assert normalizer.n_samples_fit == len(train_norm)
