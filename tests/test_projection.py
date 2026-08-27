"""Unit tests for projection math (research/projection.py).

Tests cover:
1. World-to-camera transform correctness
2. Perspective projection math
3. Jacobian computation and non-singularity near near-plane
4. 2D covariance from 3D covariance
5. Conic (inverse covariance) computation
6. Gradient flow through the entire projection pipeline
"""
import pytest
import torch
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.projection import (
    world_to_camera, project_to_screen, compute_projection_jacobian,
    compute_2d_covariance, cov2d_to_conic, compute_bounding_boxes,
    compute_radii
)


def make_intrinsics(fx=500., fy=500., cx=320., cy=240.):
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=torch.float32)


def make_identity_extrinsics():
    return torch.eye(4, dtype=torch.float32)


class TestWorldToCamera:
    def test_identity_transform(self):
        """Identity extrinsics should not change points."""
        points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ext = make_identity_extrinsics()
        result = world_to_camera(points, ext)
        assert torch.allclose(result, points, atol=1e-6)
    
    def test_translation(self):
        """Pure translation."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        ext = torch.eye(4)
        ext[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
        result = world_to_camera(points, ext)
        expected = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_batch(self):
        points = torch.randn(100, 3)
        ext = make_identity_extrinsics()
        result = world_to_camera(points, ext)
        assert result.shape == (100, 3)


class TestProjectToScreen:
    def test_on_optical_axis(self):
        """Point on optical axis projects to principal point."""
        intrinsics = make_intrinsics()
        points_cam = torch.tensor([[0.0, 0.0, 5.0]])
        pts2d, depths = project_to_screen(points_cam, intrinsics)
        assert torch.allclose(pts2d[0], torch.tensor([320., 240.]), atol=1e-4)
        assert torch.allclose(depths[0], torch.tensor(5.0), atol=1e-4)
    
    def test_depth_is_z(self):
        """Depth should equal z-coordinate."""
        intrinsics = make_intrinsics()
        points_cam = torch.tensor([[1.0, 2.0, 7.0], [-1.0, 0.5, 3.0]])
        _, depths = project_to_screen(points_cam, intrinsics)
        assert torch.allclose(depths, torch.tensor([7.0, 3.0]), atol=1e-4)
    
    def test_near_plane_clamp(self):
        """Points very close to or behind camera should not cause NaN/Inf."""
        intrinsics = make_intrinsics()
        points_cam = torch.tensor([[1.0, 1.0, 0.0001], [0.0, 0.0, -1.0]])
        pts2d, depths = project_to_screen(points_cam, intrinsics)
        assert torch.isfinite(pts2d).all(), "NaN/Inf in projected points"
        assert torch.isfinite(depths).all(), "NaN/Inf in depths"


class TestJacobian:
    def test_jacobian_shape(self):
        intrinsics = make_intrinsics()
        points_cam = torch.randn(20, 3).abs() + 0.5  # Ensure z > 0
        points_cam[:, 2] = points_cam[:, 2].abs() + 1.0
        J = compute_projection_jacobian(points_cam, intrinsics)
        assert J.shape == (20, 2, 3)
    
    def test_jacobian_no_singularity_near_plane(self):
        """Jacobian J should not be singular when z ≤ z_near (clamped).
        
        The z-clamp at 1e-6 prevents singularity.
        """
        intrinsics = make_intrinsics()
        # Points very close to near plane
        points_cam = torch.tensor([
            [1.0, 1.0, 0.001],   # very close to camera
            [1.0, 1.0, 0.0001],  # extremely close
            [0.0, 0.0, 1e-7],    # practically at camera
        ])
        J = compute_projection_jacobian(points_cam, intrinsics)
        assert torch.isfinite(J).all(), "Jacobian has NaN/Inf near near-plane"
        
        # Jacobian should still have full rank (rank 2) for each point
        for i in range(J.shape[0]):
            rank = torch.linalg.matrix_rank(J[i].float())
            assert rank == 2, f"Jacobian singular at point {i}, rank={rank}"
    
    def test_jacobian_numerical_correctness(self):
        """Compare analytic Jacobian with numerical finite differences."""
        intrinsics = make_intrinsics()
        points = torch.tensor([[2.0, 1.0, 5.0]], requires_grad=True)
        
        # Analytic Jacobian
        J_analytic = compute_projection_jacobian(points, intrinsics)[0]  # (2, 3)
        
        # Numerical Jacobian via central finite differences
        eps = 1e-3
        J_numeric = torch.zeros(2, 3)
        for j in range(3):
            p_plus = points.clone().detach()
            p_plus[0, j] += eps
            p_minus = points.clone().detach()
            p_minus[0, j] -= eps
            proj_plus, _ = project_to_screen(p_plus, intrinsics)
            proj_minus, _ = project_to_screen(p_minus, intrinsics)
            J_numeric[:, j] = (proj_plus[0] - proj_minus[0]) / (2 * eps)
        
        # Tolerance accounts for finite-difference approximation error
        assert torch.allclose(J_analytic, J_numeric, atol=0.5), \
            f"Analytic Jacobian doesn't match numerical:\n{J_analytic}\nvs\n{J_numeric}"


class TestCov2D:
    def test_2d_cov_symmetric(self):
        """2D covariance should be symmetric."""
        torch.manual_seed(42)
        N = 10
        # Build a valid 3D covariance
        L = torch.randn(N, 3, 3)
        cov3D = torch.bmm(L, L.transpose(1, 2))  # PSD
        
        points_cam = torch.randn(N, 3)
        points_cam[:, 2] = points_cam[:, 2].abs() + 2.0
        
        ext = make_identity_extrinsics()
        intr = make_intrinsics()
        
        cov2D = compute_2d_covariance(cov3D, points_cam, ext, intr)
        assert torch.allclose(cov2D, cov2D.transpose(1, 2), atol=1e-5)
    
    def test_2d_cov_psd(self):
        """2D covariance should be PSD."""
        torch.manual_seed(42)
        N = 20
        L = torch.randn(N, 3, 3)
        cov3D = torch.bmm(L, L.transpose(1, 2))
        
        points_cam = torch.randn(N, 3)
        points_cam[:, 2] = points_cam[:, 2].abs() + 2.0
        
        cov2D = compute_2d_covariance(cov3D, points_cam, make_identity_extrinsics(), make_intrinsics())
        
        for i in range(N):
            eigvals = torch.linalg.eigvalsh(cov2D[i])
            assert (eigvals >= -1e-5).all(), f"2D cov not PSD at {i}: {eigvals}"
    
    def test_gradient_through_cov2d(self):
        """Gradient should flow from 2D covariance to 3D covariance."""
        L = torch.randn(5, 3, 3, requires_grad=True)
        cov3D = torch.bmm(L, L.transpose(1, 2))
        
        points_cam = torch.randn(5, 3)
        points_cam[:, 2] = points_cam[:, 2].abs() + 2.0
        
        cov2D = compute_2d_covariance(cov3D, points_cam, make_identity_extrinsics(), make_intrinsics())
        loss = cov2D.sum()
        loss.backward()
        
        assert L.grad is not None
        assert L.grad.abs().sum() > 0


class TestConicAndBoundingBox:
    def test_conic_inverse(self):
        """cov2d @ conic_matrix should give identity."""
        cov2D = torch.tensor([[[4.0, 1.0], [1.0, 3.0]]])
        conics = cov2d_to_conic(cov2D)  # (1, 3)
        
        # Reconstruct inverse from conic
        a, b, c = conics[0]
        inv_matrix = torch.tensor([[a, b], [b, c]])
        
        product = cov2D[0] @ inv_matrix
        assert torch.allclose(product, torch.eye(2), atol=1e-4)
    
    def test_bounding_box_contains_mean(self):
        means2D = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
        cov2D = torch.tensor([[[10., 0.], [0., 10.]], [[5., 1.], [1., 5.]]])
        bb_min, bb_max = compute_bounding_boxes(means2D, cov2D)
        
        for i in range(2):
            assert (bb_min[i] <= means2D[i]).all()
            assert (bb_max[i] >= means2D[i]).all()
    
    def test_radii_positive(self):
        cov2D = torch.tensor([[[10., 0.], [0., 10.]], [[5., 1.], [1., 5.]]])
        radii = compute_radii(cov2D)
        assert (radii > 0).all()
