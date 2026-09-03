"""Tests for Scientific Protocol v1 and configuration getters."""
import pytest
from research.protocol import (
    load_protocol,
    get_seeds,
    get_resolution,
    get_splits,
    get_dataset_config,
    get_oracle_config,
    get_budget_config,
    get_statistics_config,
    get_state_factor_config,
    get_densification_policy,
)


def test_protocol_resolution():
    H, W = get_resolution()
    assert H == 240
    assert W == 320


def test_protocol_resolution_fr2():
    H, W = get_resolution("tum_fr2_xyz")
    assert H == 240
    assert W == 320


def test_protocol_seeds():
    assert get_seeds() == [42, 43, 44, 45, 46]


def test_protocol_splits():
    splits = get_splits()
    assert splits["train_frames"] == list(range(41))
    assert splits["val_frames"] == list(range(41, 61))
    assert splits["test_scene"] == "tum_fr2_xyz"


def test_protocol_required_keys():
    protocol = load_protocol()

    assert "reproducibility" in protocol
    assert "datasets" in protocol
    assert "splits" in protocol
    assert "oracle_specification" in protocol
    assert "budget_levels" in protocol
    assert "statistical_testing" in protocol


def test_protocol_oracle_config():
    oracle_cfg = get_oracle_config()
    assert "n_opt_steps" in oracle_cfg
    assert "w_rgb" in oracle_cfg
    assert "w_depth" in oracle_cfg
    assert "min_influence_pixels" in oracle_cfg
    assert oracle_cfg["n_opt_steps"] == 5
    assert oracle_cfg["w_rgb"] == 0.70
    assert oracle_cfg["w_depth"] == 0.30
    assert oracle_cfg["min_influence_pixels"] == 25


def test_protocol_budget_config():
    budget_cfg = get_budget_config()
    assert "optimization_relative" in budget_cfg
    assert "wall_clock_ms" in budget_cfg
    assert 0.20 in budget_cfg["optimization_relative"]
    assert len(budget_cfg["wall_clock_ms"]) > 0


def test_protocol_statistics_config():
    stats_cfg = get_statistics_config()
    assert stats_cfg["confidence_interval_level"] == 0.95
    assert stats_cfg["bootstrap_resamples"] == 1000
    assert stats_cfg["paired_tests"] == "wilcoxon"
    assert stats_cfg["effect_size_metric"] == "cohens_d"


def test_protocol_dataset_config():
    cfg = get_dataset_config("tum_fr1_desk")
    assert "full_path" in cfg
    assert cfg["image_width"] == 320
    assert cfg["image_height"] == 240
    assert cfg["depth_scale"] == 5000.0


def test_protocol_state_factors():
    factors = get_state_factor_config()
    assert "appearance" in factors
    assert "geometry" in factors
    assert "uncertainty" in factors


def test_protocol_densification_policy():
    policy = get_densification_policy()
    assert policy in ["fresh", "inherit"]
    assert policy == "fresh"

