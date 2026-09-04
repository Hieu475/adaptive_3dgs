"""Unit tests for Phase 4 Canonical Feature Schema and Extraction."""
import pytest
import numpy as np

from research.utility_features import (
    CANONICAL_FEATURE_NAMES,
    CANONICAL_FEATURE_SPECS,
    UTILITY_FEATURES,
    ABLATION_SUBSETS,
    SchemaError,
    DatasetSchemaError,
    extract_feature_vector,
    generate_feature_schema_table,
    get_canonical_feature_names,
)


def test_canonical_features_names_and_constant():
    assert len(UTILITY_FEATURES) == 11
    assert isinstance(UTILITY_FEATURES, tuple)
    assert UTILITY_FEATURES == (
        "rgb_error",
        "depth_error",
        "gradient_norm",
        "visibility_count",
        "influence_mass",
        "position_drift",
        "residual_drift_ema",
        "uncertainty_var",
        "projected_area",
        "update_frequency",
        "age",
    )
    assert get_canonical_feature_names() == list(UTILITY_FEATURES)
    assert "temporal_drift" not in UTILITY_FEATURES


def test_feature_ablation_subsets_v0_v7():
    assert len(ABLATION_SUBSETS) == 8
    keys = list(ABLATION_SUBSETS.keys())
    assert keys[0].startswith("V0:")
    assert keys[1].startswith("V1:")
    assert keys[2].startswith("V2:")
    assert keys[3].startswith("V3:")
    assert keys[4].startswith("V4:")
    assert keys[5].startswith("V5:")
    assert "Temporal State" in keys[5]
    assert keys[6].startswith("V6:")
    assert keys[7].startswith("V7:")

    # V5 must include position_drift and residual_drift_ema
    v5_feats = ABLATION_SUBSETS[keys[5]]
    assert "position_drift" in v5_feats
    assert "residual_drift_ema" in v5_feats
    assert "temporal_drift" not in v5_feats

    # V7 must include all 11 features
    v7_feats = ABLATION_SUBSETS[keys[7]]
    assert len(v7_feats) == 11
    assert set(v7_feats) == set(UTILITY_FEATURES)


def test_strict_extraction_passes_when_all_features_present():
    valid_row = {
        "gaussian_id": 101,
        "features": {name: float(i * 0.1) for i, name in enumerate(UTILITY_FEATURES)},
    }
    vec = extract_feature_vector(valid_row, strict=True)
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert len(vec) == 11
    for i in range(11):
        assert np.isclose(vec[i], i * 0.1)


def test_strict_extraction_raises_on_missing_feature():
    # Missing residual_drift_ema
    features = {name: 1.0 for name in UTILITY_FEATURES if name != "residual_drift_ema"}
    bad_row = {"gaussian_id": 202, "features": features}

    with pytest.raises((DatasetSchemaError, SchemaError)) as excinfo:
        extract_feature_vector(bad_row, strict=True)
    assert "residual_drift_ema" in str(excinfo.value)


def test_strict_extraction_raises_when_features_dict_missing():
    bad_row = {"gaussian_id": 303}
    with pytest.raises((DatasetSchemaError, SchemaError)):
        extract_feature_vector(bad_row, strict=True)


def test_generate_feature_schema_table():
    table = generate_feature_schema_table()
    assert isinstance(table, str)
    assert "| Index | Feature Name |" in table
    for name in UTILITY_FEATURES:
        assert f"`{name}`" in table
