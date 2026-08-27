"""Tests for policy benchmark harness (Milestone R4)."""
import pytest
import torch
from research.scheduler import OptimizationPolicy
from research.benchmark_policies import (
    run_policy_experiment,
    run_full_policy_ablation_matrix,
    format_benchmark_table,
)


class TestBenchmarkHarness:
    """Test policy benchmarking and ablation evaluation."""

    @pytest.fixture
    def test_frames(self):
        torch.manual_seed(42)
        H, W = 32, 32
        intrinsics = torch.tensor([
            [30.0, 0.0, 16.0],
            [0.0, 30.0, 16.0],
            [0.0, 0.0, 1.0],
        ])
        frames = []
        for i in range(2):
            rgb = torch.rand(H, W, 3) * 0.8 + 0.1
            depth = torch.full((H, W), 2.0) + torch.randn(H, W) * 0.01
            pose = torch.eye(4)
            pose[0, 3] = i * 0.02
            frames.append({'rgb': rgb, 'depth': depth, 'pose': pose})
        return frames, intrinsics

    def test_single_policy_run(self, test_frames):
        frames, intrinsics = test_frames
        res = run_policy_experiment(
            frames, intrinsics, policy=OptimizationPolicy.TOP_K, ratio=0.5, device='cpu'
        )
        assert 'avg_psnr' in res
        assert 'avg_depth_l1' in res
        assert 'avg_n_optimized' in res
        assert res['avg_psnr'] > 0
        assert res['avg_n_optimized'] > 0

    def test_full_ablation_matrix_and_table(self, test_frames):
        frames, intrinsics = test_frames
        ablation = run_full_policy_ablation_matrix(
            frames, intrinsics, ratios=[0.25, 0.50], device='cpu'
        )
        assert 'full_psnr' in ablation
        assert len(ablation['experiments']) > 4

        table = format_benchmark_table(ablation)
        assert isinstance(table, str)
        assert "Full (100%)" in table
        assert "Random" in table
        assert "Top-K Imp" in table
        assert "Binary" in table
