"""Unit tests for pipeline correctness (Phase A).

Validates:
- A1: adaptive_threshold() return ordering (depth_thresh vs color_thresh)
- A2: update_confidence() integration and binary policy usage
- A3: temporal persistence pruning (Tier C freeze vs Tier D prune)
- A4: get_importance_diagnostics() API correctness and error handling
"""
import pytest
import torch
from research.pipeline import OnlineReconstructionPipeline
from research.scheduler import BudgetScheduler, OptimizationPolicy
from research.densification import prune_low_value
from research.gaussian_repr import GaussianModel, GaussianState
from research.importance import Tier


class TestAdaptiveThresholdOrdering:
    """Test A1: adaptive_threshold() ordering."""

    def test_adaptive_threshold_matches_statistics(self):
        scheduler = BudgetScheduler()
        # High depth error (mean 5.0, std ~ 2.0), low color error (mean 0.1, std ~ 0.02)
        depth_errs = torch.tensor([3.0, 5.0, 7.0])
        color_errs = torch.tensor([0.08, 0.10, 0.12])

        k = 1.5
        depth_thresh, color_thresh = scheduler.adaptive_threshold(
            depth_errors=depth_errs,
            color_errors=color_errs,
            k=k,
        )

        expected_depth_std = scheduler._depth_error_stats.std()
        expected_color_std = scheduler._color_error_stats.std()

        assert depth_thresh >= color_thresh
        assert depth_thresh == pytest.approx(min(0.5, max(0.01, k * expected_depth_std)), rel=1e-3)
        assert color_thresh == pytest.approx(min(0.3, max(0.01, k * expected_color_std)), rel=1e-3)


class TestConfidenceIntegration:
    """Test A2: Confidence updates in pipeline."""

    def test_pipeline_updates_confidence(self):
        pipeline = OnlineReconstructionPipeline(device='cpu')
        H, W = 16, 16
        rgb = torch.rand(H, W, 3) * 0.8 + 0.1
        depth = torch.full((H, W), 2.0)
        intrinsics = torch.tensor([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]])

        pipeline.initialize(rgb, depth, intrinsics)
        initial_conf = pipeline.gaussian_model._confidence.clone()
        N_init = initial_conf.shape[0]

        # Process a frame to trigger importance and confidence update
        metrics = pipeline.process_frame(rgb, depth)
        updated_conf = pipeline.gaussian_model._confidence[:N_init]

        assert not torch.allclose(initial_conf, updated_conf), "Confidence buffer should be updated by pipeline"


class TestPruningTemporalPersistence:
    """Test A3: Pruning criteria (Tier C freeze vs Tier D prune)."""

    def test_low_importance_not_pruned_immediately(self):
        model = GaussianModel(sh_degree=0, device='cpu')
        pts = torch.randn(10, 3)
        model.initialize_from_points(pts, initial_opacity=0.8)

        # Low importance but zero_contrib_frames is 0
        importance = torch.full((10,), 0.005)
        zero_contrib = torch.zeros(10, dtype=torch.long)

        prune_low_value(
            model,
            importance,
            opacity_threshold=0.005,
            zero_contrib_frames=zero_contrib,
            prune_patience=50,
        )
        model.compact()

        assert model.num_gaussians == 10, "Low importance alone should NOT prune Gaussians"

    def test_persistent_zero_contribution_pruned(self):
        model = GaussianModel(sh_degree=0, device='cpu')
        pts = torch.randn(10, 3)
        model.initialize_from_points(pts, initial_opacity=0.8)

        importance = torch.full((10,), 0.005)
        zero_contrib = torch.tensor([60, 10, 0, 0, 0, 0, 0, 0, 0, 0])  # first Gaussian exceeded patience

        prune_low_value(
            model,
            importance,
            opacity_threshold=0.005,
            zero_contrib_frames=zero_contrib,
            prune_patience=50,
        )
        model.compact()

        assert model.num_gaussians == 9, "Persistent zero-contribution Gaussian should be pruned"

    def test_opacity_failure_pruned(self):
        model = GaussianModel(sh_degree=0, device='cpu')
        pts = torch.randn(5, 3)
        model.initialize_from_points(pts, initial_opacity=0.001)  # below opacity threshold

        importance = torch.full((5,), 0.8)
        prune_low_value(
            model,
            importance,
            opacity_threshold=0.005,
            zero_contrib_frames=None,
        )
        model.compact()

        assert model.num_gaussians == 0, "Transparent Gaussians should be pruned"


class TestImportanceDiagnosticsAPI:
    """Test A4 & Phase C: get_importance_diagnostics() API."""

    def test_uninitialized_raises_runtime_error(self):
        pipeline = OnlineReconstructionPipeline(device='cpu')
        with pytest.raises(RuntimeError):
            pipeline.get_importance_diagnostics()

    def test_returns_complete_research_state(self):
        pipeline = OnlineReconstructionPipeline(device='cpu')
        H, W = 16, 16
        rgb = torch.rand(H, W, 3) * 0.8 + 0.1
        depth = torch.full((H, W), 2.0)
        intrinsics = torch.tensor([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]])

        pipeline.initialize(rgb, depth, intrinsics)
        pipeline.process_frame(rgb, depth)

        state = pipeline.get_importance_diagnostics()

        expected_keys = [
            'importance', 'color_error', 'depth_error', 'visibility',
            'screen_area', 'temporal_change', 'tiers', 'confidence', 'components'
        ]
        for k in expected_keys:
            assert k in state, f"Missing key {k} in get_importance_diagnostics"

        N = pipeline.gaussian_model.num_gaussians
        assert state['importance'].shape == (N,)
        assert state['color_error'].shape == (N,)
        assert state['depth_error'].shape == (N,)
        assert state['visibility'].shape == (N,)
        assert state['tiers'].shape == (N,)
        assert state['confidence'].shape == (N,)
        assert 'color' in state['components']
        assert 'depth' in state['components']
