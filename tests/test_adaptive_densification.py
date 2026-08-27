"""Tests for Adaptive Importance-Driven Densification (Milestone R5)."""
import pytest
import torch
from research.densification import (
    compute_error_masks,
    compute_sampling_probability_map,
    sample_candidates,
    create_gaussians_from_candidates,
)
from research.scheduler import BudgetScheduler
from research.pipeline import OnlineReconstructionPipeline


class TestSamplingProbabilityMap:
    """Test candidate creation probability distribution."""

    def test_probability_sums_to_one(self):
        H, W = 32, 32
        color_err = torch.rand(H, W)
        depth_err = torch.rand(H, W)
        transmission = torch.rand(H, W)
        mask = torch.rand(H, W) > 0.5
        
        prob_map = compute_sampling_probability_map(
            color_err, depth_err, transmission, mask=mask
        )
        
        assert abs(prob_map.sum().item() - 1.0) < 1e-5
        # Pixels outside mask must have 0 probability
        assert (prob_map[~mask] == 0.0).all()

    def test_high_error_receives_higher_probability(self):
        H, W = 16, 16
        color_err = torch.zeros(H, W)
        depth_err = torch.zeros(H, W)
        transmission = torch.zeros(H, W)
        
        # Region A has high color error, Region B has zero error
        color_err[:8, :] = 1.0
        mask = torch.ones(H, W, dtype=torch.bool)
        
        prob_map = compute_sampling_probability_map(
            color_err, depth_err, transmission, mask=mask
        )
        
        assert prob_map[:8, :].sum() > prob_map[8:, :].sum()


class TestSampleCandidatesStrategies:
    """Test candidate pixel sampling strategies."""

    @pytest.fixture
    def error_maps(self):
        torch.manual_seed(42)
        H, W = 32, 32
        color_err = torch.zeros(H, W)
        depth_err = torch.zeros(H, W)
        transmission = torch.zeros(H, W)
        
        # Quadrant 0 (top-left) has high error
        color_err[:16, :16] = 0.8
        depth_err[:16, :16] = 0.5
        transmission[:16, :16] = 0.6
        
        # Quadrant 3 (bottom-right) has very low error
        color_err[16:, 16:] = 0.01
        
        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[:16, :16] = True
        mask[16:, 16:] = True
        
        return mask, color_err, depth_err, transmission

    def test_uniform_sampling(self, error_maps):
        mask, color_err, depth_err, transmission = error_maps
        samples = sample_candidates(mask, num_samples=50, strategy='uniform')
        assert samples.shape == (50, 2)

    def test_importance_sampling_concentrates_on_high_error(self, error_maps):
        mask, color_err, depth_err, transmission = error_maps
        torch.manual_seed(42)
        
        samples_imp = sample_candidates(
            mask, num_samples=100, strategy='importance',
            color_err=color_err, depth_err=depth_err, transmission=transmission
        )
        
        # Check coordinates (u, v): top-left quadrant has u < 16 and v < 16
        in_top_left = (samples_imp[:, 0] < 16) & (samples_imp[:, 1] < 16)
        top_left_count = in_top_left.sum().item()
        
        # Over 90% of samples should be drawn from the high-error top-left quadrant
        assert top_left_count > 80, f"Expected >80% samples in high-error region, got {top_left_count}%"

    def test_empty_mask_handling(self):
        empty_mask = torch.zeros(16, 16, dtype=torch.bool)
        samples = sample_candidates(empty_mask, num_samples=10, strategy='importance')
        assert samples.shape == (0, 2)


class TestAdaptiveThresholds:
    """Test dynamic error threshold adaptation."""

    def test_scene_complexity_adaptation(self):
        scheduler = BudgetScheduler()
        
        # Easy scene: low variance
        easy_depth = torch.full((100,), 1.0) + torch.randn(100) * 0.001
        easy_color = torch.full((100,), 0.5) + torch.randn(100) * 0.001
        d_th_easy, c_th_easy = scheduler.adaptive_threshold(easy_depth, easy_color, k=2.0)
        
        # Hard scene: high variance
        scheduler_hard = BudgetScheduler()
        hard_depth = torch.randn(100) * 0.2 + 1.0
        hard_color = torch.randn(100) * 0.15 + 0.5
        d_th_hard, c_th_hard = scheduler_hard.adaptive_threshold(hard_depth, hard_color, k=2.0)
        
        # Complex scenes should have higher thresholds to focus only on major errors
        assert d_th_hard > d_th_easy
        assert c_th_hard > c_th_easy


class TestDensificationPipelineIntegration:
    """Test densification strategies inside the full pipeline."""

    def test_pipeline_with_importance_densification(self):
        torch.manual_seed(42)
        H, W = 32, 32
        intrinsics = torch.tensor([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]])
        
        cfg = {
            'densification': {
                'strategy': 'importance',
                'use_adaptive_thresholds': True,
                'max_new_per_frame': 50,
            },
            'rendering': {
                'image_width': W,
                'image_height': H,
                'tile_size': 16,
            }
        }
        
        pipeline = OnlineReconstructionPipeline(config=cfg, device='cpu')
        
        rgb0 = torch.rand(H, W, 3) * 0.8 + 0.1
        depth0 = torch.full((H, W), 2.0)
        pipeline.initialize(rgb0, depth0, intrinsics)
        n_init = pipeline.gaussian_model.num_gaussians
        
        # Frame 1 with higher error to trigger densification
        rgb1 = torch.rand(H, W, 3) * 0.8 + 0.1
        depth1 = torch.full((H, W), 2.0)
        
        pipeline.process_frame(rgb1, depth1)
        n_after = pipeline.gaussian_model.num_gaussians
        
        assert n_after >= n_init, "Densification should add new Gaussians"
