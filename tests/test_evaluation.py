"""Tests for research evaluation and paper artifact generation (Milestone R7)."""
import pytest
import torch
from research.evaluation import (
    generate_table_1_main_benchmark,
    generate_table_2_ablation_study,
    generate_ascii_pareto_curve,
    generate_tier_distribution_chart,
    generate_hypothesis_verification_summary,
)


class TestResearchEvaluation:
    """Test evaluation report and table generation."""

    @pytest.fixture
    def test_frames(self):
        torch.manual_seed(42)
        H, W = 32, 32
        intrinsics = torch.tensor([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]])
        frames = []
        for i in range(2):
            rgb = torch.rand(H, W, 3) * 0.8 + 0.1
            depth = torch.full((H, W), 2.0)
            pose = torch.eye(4)
            pose[0, 3] = i * 0.02
            frames.append({'rgb': rgb, 'depth': depth, 'pose': pose})
        return frames, intrinsics

    def test_table_1_generation(self):
        metrics = {'avg_psnr': 31.45, 'avg_depth_l1': 0.0142, 'avg_fps': 34.2, 'final_n_gaussians': 182400}
        t1 = generate_table_1_main_benchmark(metrics)
        assert "Table 1" in t1
        assert "Original 3DGS" in t1
        assert "SplaTAM" in t1
        assert "RTG-SLAM" in t1
        assert "31.45" in t1

    def test_table_2_ablation_study(self, test_frames):
        frames, intrinsics = test_frames
        t2, data = generate_table_2_ablation_study(frames, intrinsics, device='cpu')
        assert "Table 2" in t2
        assert "Full System (Ours)" in t2
        assert "w/o Surface-Aware Depth" in t2
        assert "w/o Importance Densification" in t2
        assert len(data['variants']) == 4

    def test_figures_and_hypotheses_generation(self):
        f1 = generate_ascii_pareto_curve({})
        assert "Figure 1" in f1
        assert "Pareto" in f1

        f2 = generate_tier_distribution_chart()
        assert "Figure 2" in f2
        assert "Tier A" in f2

        h_sum = generate_hypothesis_verification_summary(results=None)
        assert "Hypothesis" in h_sum
        assert "H1" in h_sum
        assert "H2" in h_sum
        assert "H3" in h_sum
        assert "H4" in h_sum
        assert "PENDING" in h_sum
