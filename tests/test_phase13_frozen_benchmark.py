"""Tests for Phase 13 Frozen Benchmark Protocol."""
import os
import sys
import pytest
import torch
import numpy as np

# Repository root setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_phase13_frozen_benchmark import (
    build_pipeline_config,
    run_policy_trajectory,
    aggregate_policy_runs,
    generate_markdown_summary,
    POLICIES_CANONICAL,
)


def test_build_pipeline_config():
    """Verify configuration generator sets appropriate policy flags."""
    cfg_ours = build_pipeline_config("ours", seed=42, budget_ms=15.0, device="cpu")
    assert cfg_ours["training"]["use_adaptive_k"] is True
    assert cfg_ours["scheduler"]["policy"] == "ours"
    assert cfg_ours["scheduler"]["gpu_budget_ms"] == 15.0

    cfg_err = build_pipeline_config("error_only", seed=42, budget_ms=15.0, device="cpu")
    assert cfg_err["training"]["use_adaptive_k"] is False
    assert cfg_err["scheduler"]["policy"] == "error_only"

    cfg_noop = build_pipeline_config("no_op", seed=42, budget_ms=15.0, device="cpu")
    assert cfg_noop["training"]["use_adaptive_k"] is False
    assert cfg_noop["scheduler"]["policy"] == "no_op"


def test_trajectory_execution_smoke():
    """Verify running a smoke trajectory (3 frames) on CPU works cleanly."""
    device = "cpu"
    H, W = 32, 32
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics[0, 0] = 30.0
    intrinsics[1, 1] = 30.0
    intrinsics[0, 2] = 16.0
    intrinsics[1, 2] = 16.0

    frames = []
    for i in range(3):
        frames.append({
            "rgb": torch.rand(H, W, 3, dtype=torch.float32, device=device),
            "depth": torch.ones(H, W, dtype=torch.float32, device=device) * 1.5,
            "pose": torch.eye(4, dtype=torch.float32, device=device),
        })

    summary, logs = run_policy_trajectory(
        policy="no_op",
        seed=42,
        frames=frames,
        intrinsics=intrinsics,
        budget_ms=10.0,
        device=device,
        W=W,
        H=H,
    )

    assert summary["policy"] == "no_op"
    assert summary["n_frames"] == 2
    assert summary["mean_n_optimized"] == 0
    assert summary["mean_k"] == 0.0
    assert len(logs) == 2


def test_aggregate_policy_runs():
    """Verify multi-seed statistical aggregation and bootstrap CI."""
    fake_runs = [
        {
            "policy": "ours",
            "seed": 42 + i,
            "mean_psnr": 20.0 + i * 0.5,
            "final_psnr": 21.0 + i * 0.5,
            "mean_ssim": 0.80 + i * 0.01,
            "final_ssim": 0.82 + i * 0.01,
            "mean_depth_l1": 0.10 - i * 0.01,
            "final_depth_l1": 0.09 - i * 0.01,
            "N_final": 5000 + i * 100,
            "mean_new_per_frame": 50.0,
            "mean_n_optimized": 300.0,
            "mean_k": 3.5,
            "mean_opt_time_ms": 25.0,
            "mean_frame_time_ms": 35.0,
            "fps": 28.5,
            "peak_vram_mb": 150.0,
            "catastrophic_failures": 0,
        }
        for i in range(5)
    ]

    agg = aggregate_policy_runs(fake_runs)
    assert agg["policy"] == "ours"
    assert agg["n_seeds"] == 5
    assert len(agg["mean_psnr"]["ci_95"]) == 2
    assert agg["mean_psnr"]["mean"] == pytest.approx(21.0)
    assert agg["catastrophic_failures"] == 0


def test_generate_markdown_summary():
    """Verify markdown generator produces valid markdown with expected sections."""
    manifest = {
        "timestamp": "2026-09-19T03:00:00",
        "git_sha": "abc1234",
        "git_branch": "phase13-cuda-acceleration",
        "dataset_name": "tum_fr2_xyz",
        "resolution": "320x240",
        "renderer_backend": "gsplat",
        "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "vram_gb": 6.0,
        "cuda_version": "12.8",
    }
    agg_runs = [
        {
            "policy": pol,
            "mean_psnr": {"mean": 20.0, "std": 0.2, "ci_95": [19.8, 20.2]},
            "final_psnr": {"mean": 20.5, "std": 0.3},
            "mean_ssim": {"mean": 0.85, "std": 0.01},
            "mean_depth_l1": {"mean": 0.05, "std": 0.002},
            "mean_n_optimized": {"mean": 250.0},
            "mean_k": {"mean": 4.0},
            "mean_opt_time_ms": {"mean": 30.0},
            "fps": {"mean": 25.0},
            "peak_vram_mb": {"mean": 200.0},
            "catastrophic_failures": 0,
        }
        for pol in POLICIES_CANONICAL
    ]
    md = generate_markdown_summary(agg_runs, manifest, n_frames=150, budget_ms=15.0)
    assert "# Phase 13 Authoritative Frozen Benchmark Report" in md
    assert "Substrate Headroom" in md
    assert "Provenance & Reproducibility Guarantees" in md
