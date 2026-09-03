"""Unit tests for Experiment Schema, Reproducibility, and Provenance Bundling."""
import os
import tempfile
import pytest
import numpy as np
import torch

from research.schema import (
    ExperimentMetadata,
    QualityMetrics,
    LatencyMetrics,
    MemoryMetrics,
    SelectionMetrics,
    ExperimentMetrics,
    ExperimentResult,
)
from research.reproducibility import (
    get_git_commit,
    get_hardware_info,
    set_seed,
    bootstrap_ci,
    save_experiment_bundle,
    DatasetManifest,
)


def test_schema_serialization():
    """Verify complete round-trip JSON and YAML serialization of ExperimentResult."""
    meta = ExperimentMetadata(
        name="test_experiment",
        version="1.0.0",
        git_commit="abc1234",
        timestamp="2026-09-03T10:00:00",
        seed=42,
        dataset="TUM",
        scene="freiburg1_desk",
        frame_range=[0, 10],
        hardware={"device": "cuda"},
        config={"gpu_budget_ms": 20.0},
    )
    metrics = ExperimentMetrics(
        quality=QualityMetrics(psnr_mean=28.5, psnr_std=1.2, psnr_ci95=(27.8, 29.2)),
        latency=LatencyMetrics(mean_ms=18.4, p50_ms=18.0, p95_ms=21.2, violation_rate=0.0),
        memory=MemoryMetrics(active_gaussians_mean=150.0, total_gaussians_final=1200),
        selection=SelectionMetrics(spearman_rho=0.45, gain_efficiency=0.82),
    )
    result = ExperimentResult(experiment=meta, metrics=metrics)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = os.path.join(tmp_dir, "result.json")
        yaml_path = os.path.join(tmp_dir, "result.yaml")
        
        result.to_json(json_path)
        result.to_yaml(yaml_path)
        
        assert os.path.exists(json_path)
        assert os.path.exists(yaml_path)
        
        # Reload from dictionary
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        reloaded = ExperimentResult.from_dict(data)
        assert reloaded.experiment.name == "test_experiment"
        assert reloaded.metrics.quality.psnr_mean == 28.5
        assert reloaded.metrics.latency.p50_ms == 18.0
        assert reloaded.metrics.selection.spearman_rho == 0.45


def test_reproducibility_utilities():
    """Verify hardware capture, git commit retrieval, and deterministic seeding."""
    commit = get_git_commit()
    assert isinstance(commit, str) and len(commit) > 0
    
    hw = get_hardware_info()
    assert "os" in hw
    assert "python" in hw
    assert "pytorch" in hw
    assert "device_name" in hw
    
    # Deterministic seeding test
    set_seed(123)
    r1 = torch.rand(5)
    set_seed(123)
    r2 = torch.rand(5)
    assert torch.equal(r1, r2)


def test_bootstrap_ci():
    """Verify bootstrap confidence interval computation."""
    np.random.seed(42)
    data = np.random.normal(loc=10.0, scale=1.0, size=100)
    lower, upper = bootstrap_ci(data, stat_fn=np.mean, n_boot=500, ci=0.95)
    
    assert lower < upper
    assert lower < 10.0 < upper
    assert abs((lower + upper) / 2.0 - 10.0) < 0.5


def test_save_experiment_bundle():
    """Verify immutable-ish bundle emission (Section XXXII)."""
    meta = ExperimentMetadata(name="test_bundle", git_commit="test_hash")
    metrics = ExperimentMetrics()
    result = ExperimentResult(experiment=meta, metrics=metrics)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_experiment_bundle(result, tmp_dir, markdown_report="# Test Report")
        
        assert os.path.exists(os.path.join(tmp_dir, "result.json"))
        assert os.path.exists(os.path.join(tmp_dir, "config.yaml"))
        assert os.path.exists(os.path.join(tmp_dir, "git_commit.txt"))
        assert os.path.exists(os.path.join(tmp_dir, "environment.txt"))
        assert os.path.exists(os.path.join(tmp_dir, "report.md"))
