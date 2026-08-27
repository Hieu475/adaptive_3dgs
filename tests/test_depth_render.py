"""Unit tests for RTG-SLAM style depth rendering.

Tests cover:
1. Ray-plane intersection math
2. Gradient flow from depth to Gaussian position and normal
3. Surface-aware depth vs naive alpha-blended depth
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.depth_render import ray_plane_intersection


class TestRayPlaneIntersection:
    def test_perpendicular_ray(self):
        """Ray along z-axis hitting xy-plane at z=5."""
        ray_origin = torch.tensor([[0., 0., 0.]])
        ray_dir = torch.tensor([[0., 0., 1.]])
        plane_point = torch.tensor([[0., 0., 5.]])
        plane_normal = torch.tensor([[0., 0., 1.]])
        
        p_hit, depth, valid = ray_plane_intersection(
            ray_origin, ray_dir, plane_point, plane_normal
        )
        
        assert valid[0].item() == True
        assert depth[0].item() == pytest.approx(5.0, abs=1e-4)
        assert torch.allclose(p_hit[0], torch.tensor([0., 0., 5.]), atol=1e-4)
    
    def test_angled_ray(self):
        """Ray at 45° hitting plane at z=10."""
        ray_origin = torch.tensor([[0., 0., 0.]])
        ray_dir = torch.tensor([[1., 0., 1.]]) / (2**0.5)  # normalize
        plane_point = torch.tensor([[0., 0., 10.]])
        plane_normal = torch.tensor([[0., 0., 1.]])
        
        p_hit, depth, valid = ray_plane_intersection(
            ray_origin, ray_dir, plane_point, plane_normal
        )
        
        assert valid[0].item() == True
        assert p_hit[0, 2].item() == pytest.approx(10.0, abs=1e-4)
    
    def test_parallel_ray_invalid(self):
        """Ray parallel to plane should be invalid."""
        ray_origin = torch.tensor([[0., 0., 0.]])
        ray_dir = torch.tensor([[1., 0., 0.]])  # parallel to z=5 plane
        plane_point = torch.tensor([[0., 0., 5.]])
        plane_normal = torch.tensor([[0., 0., 1.]])
        
        _, _, valid = ray_plane_intersection(
            ray_origin, ray_dir, plane_point, plane_normal
        )
        assert valid[0].item() == False
    
    def test_ray_behind_plane_invalid(self):
        """Ray pointing away from plane should be invalid."""
        ray_origin = torch.tensor([[0., 0., 0.]])
        ray_dir = torch.tensor([[0., 0., -1.]])  # pointing backward
        plane_point = torch.tensor([[0., 0., 5.]])
        plane_normal = torch.tensor([[0., 0., 1.]])
        
        _, _, valid = ray_plane_intersection(
            ray_origin, ray_dir, plane_point, plane_normal
        )
        assert valid[0].item() == False
    
    def test_gradient_to_plane_point(self):
        """∂L_depth/∂p_G should be non-zero."""
        ray_origin = torch.tensor([[0., 0., 0.]])
        ray_dir = torch.tensor([[0., 0., 1.]])
        plane_point = torch.tensor([[0., 0., 5.]], requires_grad=True)
        plane_normal = torch.tensor([[0., 0., 1.]])
        
        _, depth, valid = ray_plane_intersection(
            ray_origin, ray_dir, plane_point, plane_normal
        )
        
        loss = depth.sum()
        loss.backward()
        
        assert plane_point.grad is not None
        assert plane_point.grad.abs().sum() > 0, "No gradient to plane_point"
    
    def test_gradient_to_normal(self):
        """∂L_depth/∂n_G should be non-zero for angled rays."""
        ray_origin = torch.tensor([[0., 0., 0.]])
        ray_dir = torch.tensor([[0.1, 0., 1.]]) / torch.tensor([[0.1, 0., 1.]]).norm()
        plane_point = torch.tensor([[0., 0., 5.]])
        plane_normal = torch.tensor([[0., 0., 1.]], requires_grad=True)
        
        _, depth, valid = ray_plane_intersection(
            ray_origin, ray_dir, plane_point, plane_normal
        )
        
        loss = depth.sum()
        loss.backward()
        
        assert plane_normal.grad is not None
        assert plane_normal.grad.abs().sum() > 0, "No gradient to plane_normal"
    
    def test_batch_processing(self):
        """Should handle batched inputs."""
        N = 50
        ray_origins = torch.zeros(N, 3)
        ray_dirs = torch.randn(N, 3)
        ray_dirs[:, 2] = ray_dirs[:, 2].abs() + 0.1
        ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
        
        plane_points = torch.randn(N, 3)
        plane_points[:, 2] = plane_points[:, 2].abs() + 1.0
        plane_normals = torch.randn(N, 3)
        plane_normals = plane_normals / plane_normals.norm(dim=-1, keepdim=True)
        
        p_hit, depth, valid = ray_plane_intersection(
            ray_origins, ray_dirs, plane_points, plane_normals
        )
        
        assert p_hit.shape == (N, 3)
        assert depth.shape == (N,)
        assert valid.shape == (N,)


class TestRenderDepthSurfaceAware:
    """Tests for full render_depth_surface_aware pipeline."""

    def _make_scene(self, n=5, device='cpu'):
        torch.manual_seed(42)
        means3D = torch.randn(n, 3, device=device, requires_grad=True)
        with torch.no_grad():
            means3D[:, 2] = means3D[:, 2].abs() + 2.0  # in front of camera
        
        normals = torch.randn(n, 3, device=device, requires_grad=True)
        with torch.no_grad():
            normals[:, 2] = -1.0  # facing camera
            normals.data = normals.data / normals.data.norm(dim=-1, keepdim=True)
        
        scales = torch.full((n, 3), 0.2, device=device)
        S = torch.zeros(n, 3, 3, device=device)
        S[:, 0, 0] = scales[:, 0]
        S[:, 1, 1] = scales[:, 1]
        S[:, 2, 2] = scales[:, 2]
        cov3D = torch.bmm(S, S.transpose(1, 2))
        
        opacities = torch.full((n,), 0.9, device=device)
        extrinsics = torch.eye(4, device=device)
        intrinsics = torch.tensor([
            [100.0, 0, 16.0],
            [0, 100.0, 16.0],
            [0, 0, 1.0]
        ], device=device)
        
        return means3D, normals, opacities, cov3D, extrinsics, intrinsics

    def test_output_shapes_and_types(self):
        from research.depth_render import render_depth_surface_aware
        means3D, normals, opacities, cov3D, extrinsics, intrinsics = self._make_scene()
        H, W = 32, 32
        
        res = render_depth_surface_aware(
            means3D, normals, opacities, cov3D,
            extrinsics, intrinsics, W, H,
            opacity_threshold=0.3
        )
        
        assert res['depth'].shape == (H, W)
        assert res['hit_mask'].shape == (H, W)
        assert res['gaussian_index'].shape == (H, W)
        assert res['hit_mask'].dtype == torch.bool
        assert res['gaussian_index'].dtype == torch.long

    def test_gradient_flow_to_positions_and_normals(self):
        from research.depth_render import render_depth_surface_aware
        means3D, normals, opacities, cov3D, extrinsics, intrinsics = self._make_scene()
        H, W = 32, 32
        
        res = render_depth_surface_aware(
            means3D, normals, opacities, cov3D,
            extrinsics, intrinsics, W, H,
            opacity_threshold=0.3
        )
        
        hit_mask = res['hit_mask']
        assert hit_mask.any(), "At least some pixels should hit"
        
        loss = res['depth'][hit_mask].mean()
        loss.backward()
        
        assert means3D.grad is not None, "Gradients should flow to positions"
        assert normals.grad is not None, "Gradients should flow to surface normals"
        assert means3D.grad.abs().sum() > 0
        assert normals.grad.abs().sum() > 0
