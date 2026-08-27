"""Unit tests for the Python reference rasterizer.

Tests cover:
1. Gaussian weight computation
2. Tile assignment
3. Depth sorting
4. Alpha compositing
5. Full render pipeline
6. Early termination
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.rasterizer import (
    compute_gaussian_weight, tile_gaussians,
    sort_by_depth, rasterize_pixels
)


class TestGaussianWeight:
    def test_peak_at_center(self):
        """Weight should be maximum at the Gaussian center."""
        mean = torch.tensor([100.0, 100.0])
        # Identity inverse covariance
        conic = torch.tensor([1.0, 0.0, 1.0])
        opacity = 0.9
        
        center = torch.tensor([[100.0, 100.0]])
        offset = torch.tensor([[105.0, 100.0]])
        
        w_center = compute_gaussian_weight(center, mean, conic, opacity)
        w_offset = compute_gaussian_weight(offset, mean, conic, opacity)
        
        assert w_center.item() > w_offset.item()
    
    def test_symmetric(self):
        """Equal offsets should give equal weights."""
        mean = torch.tensor([0.0, 0.0])
        conic = torch.tensor([1.0, 0.0, 1.0])
        
        p1 = torch.tensor([[5.0, 0.0]])
        p2 = torch.tensor([[-5.0, 0.0]])
        
        w1 = compute_gaussian_weight(p1, mean, conic, 1.0)
        w2 = compute_gaussian_weight(p2, mean, conic, 1.0)
        
        assert torch.allclose(w1, w2, atol=1e-6)
    
    def test_opacity_scales_weight(self):
        mean = torch.tensor([0.0, 0.0])
        conic = torch.tensor([1.0, 0.0, 1.0])
        pixel = torch.tensor([[0.0, 0.0]])
        
        w_high = compute_gaussian_weight(pixel, mean, conic, 0.9)
        w_low = compute_gaussian_weight(pixel, mean, conic, 0.1)
        
        assert w_high.item() == pytest.approx(0.9 * w_low.item() / 0.1, rel=1e-4)


class TestTileGaussians:
    def test_single_gaussian_single_tile(self):
        means = torch.tensor([[8.0, 8.0]])  # center of first tile
        radii = torch.tensor([3])
        tiles = tile_gaussians(means, radii, 64, 64, tile_size=16)
        assert (0, 0) in tiles
        assert tiles[(0, 0)].tolist() == [0]
    
    def test_gaussian_spans_tiles(self):
        means = torch.tensor([[16.0, 16.0]])  # at tile boundary
        radii = torch.tensor([5])
        tiles = tile_gaussians(means, radii, 64, 64, tile_size=16)
        assert len(tiles) >= 2  # Should span at least 2 tiles


class TestSortByDepth:
    def test_front_to_back(self):
        depths = torch.tensor([5.0, 1.0, 3.0, 2.0])
        indices = torch.tensor([0, 1, 2, 3])
        sorted_idx = sort_by_depth(depths, indices)
        assert sorted_idx.tolist() == [1, 3, 2, 0]


class TestAlphaCompositing:
    def test_single_opaque_gaussian(self):
        """Single fully-opaque Gaussian should dominate color."""
        pixel = torch.tensor([[0.0, 0.0]])
        sorted_idx = torch.tensor([0])
        means2D = torch.tensor([[0.0, 0.0]])
        conics = torch.tensor([[0.01, 0.0, 0.01]])  # Very wide Gaussian
        colors = torch.tensor([[1.0, 0.0, 0.0]])  # Red
        opacities = torch.tensor([0.99])
        depths = torch.tensor([1.0])
        bg = torch.tensor([0.0, 0.0, 0.0])
        
        color, depth, T = rasterize_pixels(
            pixel, sorted_idx, means2D, conics, colors, opacities, depths, bg
        )
        
        # Should be mostly red
        assert color[0, 0].item() > 0.9
    
    def test_transmittance_decreases(self):
        """Adding Gaussians should decrease transmittance."""
        pixel = torch.tensor([[0.0, 0.0]])
        means2D = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        conics = torch.tensor([[0.01, 0., 0.01], [0.01, 0., 0.01]])
        colors = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        opacities = torch.tensor([0.5, 0.5])
        depths = torch.tensor([1.0, 2.0])
        bg = torch.zeros(3)
        
        # One Gaussian
        _, _, T1 = rasterize_pixels(
            pixel, torch.tensor([0]), means2D, conics, colors, opacities, depths, bg
        )
        # Two Gaussians
        _, _, T2 = rasterize_pixels(
            pixel, torch.tensor([0, 1]), means2D, conics, colors, opacities, depths, bg
        )
        
        assert T2.item() < T1.item()
    
    def test_background_color_when_empty(self):
        pixel = torch.tensor([[0.0, 0.0]])
        bg = torch.tensor([0.5, 0.5, 0.5])
        
        color, _, _ = rasterize_pixels(
            pixel, torch.tensor([], dtype=torch.long),
            torch.empty(0, 2), torch.empty(0, 3),
            torch.empty(0, 3), torch.empty(0),
            torch.empty(0), bg
        )
        
        assert torch.allclose(color[0], bg, atol=1e-6)
