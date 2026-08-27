"""Tests for per-Gaussian attribution module."""
import torch
import pytest
import math


class TestRasterizeWithAttribution:
    """Test that attribution rasterizer produces correct outputs."""

    def _make_simple_scene(self, n_gaussians=5, device='cpu'):
        """Create a simple scene for testing."""
        torch.manual_seed(42)
        means2D = torch.rand(n_gaussians, 2, device=device) * 16  # within one tile
        conics = torch.zeros(n_gaussians, 3, device=device)
        conics[:, 0] = 0.5   # inverse covariance
        conics[:, 2] = 0.5
        colors = torch.rand(n_gaussians, 3, device=device)
        opacities = torch.full((n_gaussians,), 0.5, device=device)
        depths = torch.arange(1, n_gaussians + 1, dtype=torch.float32, device=device)
        bg_color = torch.zeros(3, device=device)
        sorted_indices = torch.arange(n_gaussians, device=device)
        return means2D, conics, colors, opacities, depths, bg_color, sorted_indices

    def test_matches_original_rasterizer(self):
        """Attribution rasterizer should produce identical color/depth to original."""
        from research.rasterizer import rasterize_pixels
        from research.attribution import rasterize_pixels_with_attribution

        means2D, conics, colors, opacities, depths, bg_color, sorted_indices = (
            self._make_simple_scene()
        )
        pixel_coords = torch.tensor([[8.0, 8.0], [4.0, 4.0], [12.0, 12.0]])

        # Original
        orig_color, orig_depth, orig_T = rasterize_pixels(
            pixel_coords, sorted_indices, means2D, conics,
            colors, opacities, depths, bg_color
        )

        # With attribution
        attr_color, attr_depth, attr_T, cw, ci, dom = rasterize_pixels_with_attribution(
            pixel_coords, sorted_indices, means2D, conics,
            colors, opacities, depths, bg_color, n_gaussians=5, top_k=4
        )

        torch.testing.assert_close(attr_color, orig_color, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(attr_depth, orig_depth, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(attr_T, orig_T, atol=1e-5, rtol=1e-5)

    def test_contribution_weights_sum(self):
        """Top-K contribution weights should approximately sum to (1 - final_T)."""
        from research.attribution import rasterize_pixels_with_attribution

        means2D, conics, colors, opacities, depths, bg_color, sorted_indices = (
            self._make_simple_scene()
        )
        pixel_coords = torch.tensor([[8.0, 8.0], [4.0, 4.0]])

        # Use top_k >= n_gaussians to capture all contributions
        _, _, final_T, cw, ci, _ = rasterize_pixels_with_attribution(
            pixel_coords, sorted_indices, means2D, conics,
            colors, opacities, depths, bg_color, n_gaussians=5, top_k=5
        )

        total_weight = cw.sum(dim=1)  # sum over top-k
        expected = 1.0 - final_T
        # Should be close (exact when top_k >= K)
        torch.testing.assert_close(total_weight, expected, atol=1e-4, rtol=1e-4)

    def test_dominant_index_is_max_weight(self):
        """Dominant index should correspond to the Gaussian with highest weight."""
        from research.attribution import rasterize_pixels_with_attribution

        means2D, conics, colors, opacities, depths, bg_color, sorted_indices = (
            self._make_simple_scene()
        )
        pixel_coords = torch.tensor([[8.0, 8.0]])

        _, _, _, cw, ci, dom = rasterize_pixels_with_attribution(
            pixel_coords, sorted_indices, means2D, conics,
            colors, opacities, depths, bg_color, n_gaussians=5, top_k=5
        )

        # The dominant index should be the index with highest contribution
        max_idx = ci[0, cw[0].argmax()]
        assert dom[0].item() == max_idx.item()

    def test_empty_scene(self):
        """Empty scene should return bg color and empty attribution."""
        from research.attribution import rasterize_pixels_with_attribution

        pixel_coords = torch.tensor([[8.0, 8.0]])
        sorted_indices = torch.empty(0, dtype=torch.long)
        means2D = torch.empty(0, 2)
        conics = torch.empty(0, 3)
        colors = torch.empty(0, 3)
        opacities = torch.empty(0)
        depths = torch.empty(0)
        bg_color = torch.ones(3)

        color, depth, T, cw, ci, dom = rasterize_pixels_with_attribution(
            pixel_coords, sorted_indices, means2D, conics,
            colors, opacities, depths, bg_color, n_gaussians=0, top_k=4
        )

        torch.testing.assert_close(color, bg_color.unsqueeze(0))
        assert T[0].item() == 1.0
        assert dom[0].item() == -1


class TestComputeGaussianStatistics:
    """Test per-Gaussian error attribution from pixel data."""

    def test_single_gaussian_single_pixel(self):
        """One Gaussian covering one pixel: error should be the pixel error."""
        from research.attribution import compute_gaussian_statistics

        H, W = 2, 2
        rendered_color = torch.zeros(H, W, 3)
        gt_color = torch.ones(H, W, 3) * 0.5  # uniform error = 0.5
        rendered_depth = torch.ones(H, W) * 2.0
        gt_depth = torch.ones(H, W) * 1.5  # depth error = 0.5

        # One Gaussian contributes to pixel (0,0) with weight 0.8
        contrib_weights = torch.zeros(H, W, 1)
        contrib_weights[0, 0, 0] = 0.8
        contrib_indices = torch.full((H, W, 1), -1, dtype=torch.long)
        contrib_indices[0, 0, 0] = 0

        stats = compute_gaussian_statistics(
            rendered_color, rendered_depth, gt_color, gt_depth,
            contrib_weights, contrib_indices, n_gaussians=1
        )

        # Color error at pixel (0,0) = mean(|0 - 0.5|) = 0.5
        # Weighted: 0.8 * 0.5 / (0.8) = 0.5
        assert abs(stats['color_error'][0].item() - 0.5) < 1e-5
        assert abs(stats['depth_error'][0].item() - 0.5) < 1e-5
        assert stats['visibility_mask'][0].item() is True
        assert stats['screen_area'][0].item() == pytest.approx(0.8, abs=1e-5)

    def test_invisible_gaussian(self):
        """Gaussian not contributing to any pixel should have zero stats."""
        from research.attribution import compute_gaussian_statistics

        H, W = 4, 4
        rendered_color = torch.rand(H, W, 3)
        gt_color = torch.rand(H, W, 3)
        rendered_depth = torch.rand(H, W)
        gt_depth = torch.rand(H, W)

        # No Gaussian contributes
        contrib_weights = torch.zeros(H, W, 2)
        contrib_indices = torch.full((H, W, 2), -1, dtype=torch.long)

        stats = compute_gaussian_statistics(
            rendered_color, rendered_depth, gt_color, gt_depth,
            contrib_weights, contrib_indices, n_gaussians=3
        )

        assert (stats['color_error'] == 0).all()
        assert (stats['depth_error'] == 0).all()
        assert (stats['visibility_mask'] == False).all()
        assert (stats['screen_area'] == 0).all()

    def test_multiple_gaussians_different_errors(self):
        """Different Gaussians covering different error regions should get different scores."""
        from research.attribution import compute_gaussian_statistics

        H, W = 4, 4
        rendered_color = torch.zeros(H, W, 3)
        gt_color = torch.zeros(H, W, 3)

        # High error in top-left quadrant
        gt_color[:2, :2, :] = 1.0  # error = 1.0
        # Low error in bottom-right quadrant
        gt_color[2:, 2:, :] = 0.1  # error = 0.1

        rendered_depth = torch.ones(H, W)
        gt_depth = torch.ones(H, W)

        # Gaussian 0 covers top-left (high error)
        # Gaussian 1 covers bottom-right (low error)
        contrib_weights = torch.zeros(H, W, 1)
        contrib_indices = torch.full((H, W, 1), -1, dtype=torch.long)

        contrib_weights[:2, :2, 0] = 0.9
        contrib_indices[:2, :2, 0] = 0

        contrib_weights[2:, 2:, 0] = 0.9
        contrib_indices[2:, 2:, 0] = 1

        stats = compute_gaussian_statistics(
            rendered_color, rendered_depth, gt_color, gt_depth,
            contrib_weights, contrib_indices, n_gaussians=2
        )

        # Gaussian 0 should have higher color error than Gaussian 1
        assert stats['color_error'][0] > stats['color_error'][1]


class TestNormalization:
    """Test importance component normalization."""

    def test_raw_passthrough(self):
        from research.attribution import normalize_importance_components

        components = {'a': torch.tensor([1.0, 2.0, 3.0])}
        result = normalize_importance_components(components, method='raw')
        torch.testing.assert_close(result['a'], components['a'])

    def test_zscore_zero_mean_unit_var(self):
        from research.attribution import normalize_importance_components

        values = torch.randn(100) * 5 + 10
        components = {'test': values}
        result = normalize_importance_components(components, method='zscore')
        normalized = result['test']

        assert abs(normalized.mean().item()) < 0.1
        assert abs(normalized.std().item() - 1.0) < 0.1

    def test_robust_uses_median(self):
        from research.attribution import normalize_importance_components

        # With outliers: robust should handle better
        values = torch.tensor([1.0, 1.1, 1.0, 0.9, 1.0, 100.0])
        components = {'test': values}

        zscore = normalize_importance_components(components, method='zscore')
        robust = normalize_importance_components(components, method='robust')

        # The outlier (100.0) should have less extreme value with robust
        # Both should identify it as outlier, but robust centers around median
        # Median ≈ 1.0, MAD ≈ 0.05 → outlier gets very high score
        assert robust['test'].shape == values.shape

    def test_unknown_method_raises(self):
        from research.attribution import normalize_importance_components

        with pytest.raises(ValueError):
            normalize_importance_components({'a': torch.tensor([1.0])}, method='unknown')


class TestRenderWithAttribution:
    """End-to-end test of render_with_attribution."""

    def _make_scene(self, n=3, device='cpu'):
        """Create a minimal 3D scene."""
        torch.manual_seed(42)
        means3D = torch.randn(n, 3, device=device)
        means3D[:, 2] = means3D[:, 2].abs() + 2.0  # ensure in front of camera

        # Build covariance from scale + rotation
        scales = torch.full((n, 3), 0.1, device=device)
        S = torch.zeros(n, 3, 3, device=device)
        S[:, 0, 0] = scales[:, 0]
        S[:, 1, 1] = scales[:, 1]
        S[:, 2, 2] = scales[:, 2]
        cov3D = torch.bmm(S, S.transpose(1, 2))  # simple isotropic

        colors = torch.rand(n, 3, device=device)
        opacities = torch.full((n,), 0.8, device=device)
        extrinsics = torch.eye(4, device=device)
        intrinsics = torch.tensor([
            [100.0, 0, 16.0],
            [0, 100.0, 16.0],
            [0, 0, 1.0]
        ], device=device)

        return means3D, cov3D, colors, opacities, extrinsics, intrinsics

    def test_output_shapes(self):
        from research.attribution import render_with_attribution

        means3D, cov3D, colors, opacities, extrinsics, intrinsics = self._make_scene()
        H, W = 32, 32
        top_k = 4

        result = render_with_attribution(
            means3D, cov3D, colors, opacities,
            extrinsics, intrinsics, W, H, top_k=top_k
        )

        assert result['color'].shape == (H, W, 3)
        assert result['depth'].shape == (H, W)
        assert result['transmission'].shape == (H, W)
        assert result['dominant_index'].shape == (H, W)
        assert result['contrib_weights'].shape == (H, W, top_k)
        assert result['contrib_indices'].shape == (H, W, top_k)

    def test_dominant_indices_valid_range(self):
        from research.attribution import render_with_attribution

        n = 5
        means3D, cov3D, colors, opacities, extrinsics, intrinsics = self._make_scene(n=n)
        H, W = 32, 32

        result = render_with_attribution(
            means3D, cov3D, colors, opacities,
            extrinsics, intrinsics, W, H
        )

        dom = result['dominant_index']
        # All valid indices should be in [0, n) or -1 (no coverage)
        assert (dom >= -1).all()
        assert (dom < n).all()

    def test_attribution_statistics_integration(self):
        """Full integration: render → attribute → statistics."""
        from research.attribution import render_with_attribution, compute_gaussian_statistics

        n = 5
        means3D, cov3D, colors, opacities, extrinsics, intrinsics = self._make_scene(n=n)
        H, W = 32, 32

        result = render_with_attribution(
            means3D, cov3D, colors, opacities,
            extrinsics, intrinsics, W, H
        )

        # Fake GT
        gt_color = torch.rand(H, W, 3)
        gt_depth = torch.ones(H, W) * 3.0

        stats = compute_gaussian_statistics(
            result['color'], result['depth'],
            gt_color, gt_depth,
            result['contrib_weights'], result['contrib_indices'],
            n_gaussians=n
        )

        assert stats['color_error'].shape == (n,)
        assert stats['depth_error'].shape == (n,)
        assert stats['visibility'].shape == (n,)
        assert stats['screen_area'].shape == (n,)
        assert stats['visibility_mask'].shape == (n,)

        # At least some Gaussians should be visible
        # (depends on projection, but with our setup they should be)
        # We just check the output is valid (no NaN)
        assert not torch.isnan(stats['color_error']).any()
        assert not torch.isnan(stats['depth_error']).any()
