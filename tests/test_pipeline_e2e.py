"""End-to-end pipeline integration tests for R1, R2, R3 milestones."""
import pytest
import torch
from research.pipeline import OnlineReconstructionPipeline
from research.importance_diagnostics import compute_full_diagnostics, format_diagnostics_report


class TestPipelineEndToEnd:
    """Test full pipeline workflow across multiple frames."""

    @pytest.fixture
    def synthetic_rgbd_data(self):
        """Generate synthetic RGB-D frames for testing."""
        torch.manual_seed(42)
        H, W = 64, 64
        intrinsics = torch.tensor([
            [50.0, 0.0, 32.0],
            [0.0, 50.0, 32.0],
            [0.0, 0.0, 1.0],
        ])
        
        frames = []
        for i in range(3):
            rgb = torch.rand(H, W, 3) * 0.8 + 0.1
            depth = torch.full((H, W), 2.0) + torch.randn(H, W) * 0.02
            pose = torch.eye(4)
            pose[0, 3] = i * 0.05
            frames.append({'rgb': rgb, 'depth': depth, 'pose': pose})
            
        return frames, intrinsics

    def test_pipeline_with_surface_aware_depth(self, synthetic_rgbd_data):
        """Pipeline should run with surface-aware depth enabled (Milestone R2)."""
        frames, intrinsics = synthetic_rgbd_data
        
        config = {
            'rendering': {
                'tile_size': 16,
                'image_width': 64,
                'image_height': 64,
                'use_surface_aware_depth': True,
                'depth_threshold_opaque': 0.3,
            },
            'densification': {
                'max_new_per_frame': 50,
            },
            'scheduler': {
                'gpu_budget_ms': 100.0,
            }
        }
        
        pipeline = OnlineReconstructionPipeline(config=config, device='cpu')
        
        pipeline.initialize(
            rgb=frames[0]['rgb'],
            depth=frames[0]['depth'],
            intrinsics=intrinsics,
            pose=frames[0]['pose'],
        )
        assert pipeline.gaussian_model.num_gaussians > 0
        
        metrics_1 = pipeline.process_frame(
            rgb=frames[1]['rgb'],
            depth=frames[1]['depth'],
            gt_pose=frames[1]['pose'],
        )
        
        assert 'psnr' in metrics_1
        assert 'depth_l1' in metrics_1
        assert 'n_visible' in metrics_1
        assert metrics_1['n_visible'] > 0
        assert metrics_1['n_optimized'] > 0
        
        metrics_2 = pipeline.process_frame(
            rgb=frames[2]['rgb'],
            depth=frames[2]['depth'],
            gt_pose=frames[2]['pose'],
        )
        
        summary = pipeline.get_metrics_summary()
        assert summary['total_frames'] == 2

    def test_importance_diagnostics_integration(self, synthetic_rgbd_data):
        """Test computing full diagnostics on pipeline state (Milestone R3)."""
        frames, intrinsics = synthetic_rgbd_data
        
        pipeline = OnlineReconstructionPipeline(device='cpu')
        pipeline.initialize(
            rgb=frames[0]['rgb'],
            depth=frames[0]['depth'],
            intrinsics=intrinsics,
            pose=frames[0]['pose'],
        )
        
        pipeline.process_frame(
            rgb=frames[1]['rgb'],
            depth=frames[1]['depth'],
            gt_pose=frames[1]['pose'],
        )
        
        N = pipeline.gaussian_model.num_gaussians
        importance = pipeline.importance_estimator.compute_importance()
        tiers = pipeline.importance_estimator.classify_tier(importance)
        
        screen_area = getattr(pipeline.importance_estimator, '_screen_areas', None)
        if screen_area is None or screen_area.shape[0] != N:
            screen_area = torch.ones(N)
            
        stats = {
            'color_error': pipeline.importance_estimator._running_color_error[:N],
            'depth_error': pipeline.importance_estimator._running_depth_error[:N],
            'visibility': pipeline.importance_estimator._visibility_count[:N],
            'screen_area': screen_area,
            'visibility_mask': pipeline.importance_estimator._visibility_count[:N] > 0,
        }
        
        diagnostics = compute_full_diagnostics(importance, tiers, stats)
        assert 'basic_stats' in diagnostics
        assert 'importance_error_correlation' in diagnostics
        assert 'calibration' in diagnostics
        assert 'tier_quality' in diagnostics
        
        report = format_diagnostics_report(diagnostics)
        assert len(report) > 100
