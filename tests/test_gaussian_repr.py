"""Unit tests for Gaussian representation (research/gaussian_repr.py).

Tests cover:
1. Covariance Σ = R·S·Sᵀ·Rᵀ is positive semi-definite
2. Quaternion → rotation matrix correctness
3. Parameter initialization shapes
4. Add/prune/compact operations
5. Gradient flow through covariance computation
"""
import pytest
import torch
import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.gaussian_repr import (
    GaussianModel, GaussianState,
    quaternion_to_rotation_matrix, build_scaling_matrix
)


class TestQuaternionToRotation:
    """Test quaternion to rotation matrix conversion."""
    
    def test_identity_quaternion(self):
        """q = (1, 0, 0, 0) should give identity rotation."""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        R = quaternion_to_rotation_matrix(q)
        assert torch.allclose(R[0], torch.eye(3), atol=1e-6)
    
    def test_rotation_is_orthogonal(self):
        """R·Rᵀ = I for any valid quaternion."""
        torch.manual_seed(42)
        q = torch.randn(10, 4)
        R = quaternion_to_rotation_matrix(q)
        for i in range(10):
            RRT = R[i] @ R[i].T
            assert torch.allclose(RRT, torch.eye(3), atol=1e-5), \
                f"R·Rᵀ not identity for sample {i}"
    
    def test_determinant_is_one(self):
        """det(R) = 1 for proper rotation (not reflection)."""
        torch.manual_seed(42)
        q = torch.randn(10, 4)
        R = quaternion_to_rotation_matrix(q)
        for i in range(10):
            det = torch.det(R[i])
            assert torch.allclose(det, torch.tensor(1.0), atol=1e-5), \
                f"det(R) = {det.item()}, expected 1.0"
    
    def test_90_degree_rotation_z(self):
        """90° rotation around z-axis: q = (cos(45°), 0, 0, sin(45°))."""
        angle = math.pi / 2
        q = torch.tensor([[math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)]])
        R = quaternion_to_rotation_matrix(q)
        expected = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        assert torch.allclose(R[0], expected, atol=1e-5)
    
    def test_batch_processing(self):
        """Should handle batched quaternions."""
        q = torch.randn(100, 4)
        R = quaternion_to_rotation_matrix(q)
        assert R.shape == (100, 3, 3)


class TestCovarianceMatrix:
    """Test Σ = R·S·Sᵀ·Rᵀ properties."""
    
    def test_positive_semi_definite(self):
        """Covariance must be PSD: all eigenvalues ≥ 0."""
        model = GaussianModel(sh_degree=0)
        torch.manual_seed(42)
        points = torch.randn(50, 3)
        model.initialize_from_points(points)
        
        # Randomize scale and rotation
        model._scaling = torch.nn.Parameter(torch.randn(50, 3))
        model._rotation = torch.nn.Parameter(torch.randn(50, 4))
        
        cov = model.build_covariance()  # (50, 3, 3)
        
        for i in range(50):
            eigenvalues = torch.linalg.eigvalsh(cov[i])
            assert (eigenvalues >= -1e-6).all(), \
                f"Gaussian {i}: eigenvalues {eigenvalues.tolist()} has negative value"
    
    def test_symmetric(self):
        """Covariance must be symmetric: Σ = Σᵀ."""
        model = GaussianModel(sh_degree=0)
        torch.manual_seed(42)
        points = torch.randn(20, 3)
        model.initialize_from_points(points)
        model._scaling = torch.nn.Parameter(torch.randn(20, 3))
        model._rotation = torch.nn.Parameter(torch.randn(20, 4))
        
        cov = model.build_covariance()
        assert torch.allclose(cov, cov.transpose(1, 2), atol=1e-6), \
            "Covariance is not symmetric"
    
    def test_identity_rotation_gives_diagonal(self):
        """With identity rotation, covariance should be diagonal."""
        model = GaussianModel(sh_degree=0)
        points = torch.zeros(5, 3)
        model.initialize_from_points(points)
        # Identity quaternion already set by initialize_from_points
        
        cov = model.build_covariance()
        scales = model.scales  # (5, 3)
        
        for i in range(5):
            expected_diag = scales[i] ** 2
            actual_diag = torch.diagonal(cov[i])
            assert torch.allclose(actual_diag, expected_diag, atol=1e-5)
            # Off-diagonal should be ~0
            off_diag = cov[i] - torch.diag(actual_diag)
            assert torch.allclose(off_diag, torch.zeros(3, 3), atol=1e-6)
    
    def test_gradient_flows_through_covariance(self):
        """Gradient must flow from covariance back to scale and rotation params.
        
        Note: At identity quaternion with isotropic scale, d(sum(Σ))/dq = 0
        because the covariance is rotationally invariant. We use non-identity
        rotation and anisotropic scale to break this symmetry.
        """
        model = GaussianModel(sh_degree=0)
        torch.manual_seed(42)
        points = torch.randn(10, 3)
        model.initialize_from_points(points)
        
        # Use non-identity rotation and anisotropic scale to ensure gradient flows
        model._rotation = torch.nn.Parameter(torch.randn(10, 4))
        model._scaling = torch.nn.Parameter(torch.tensor(
            [[0.1, 0.5, 1.0]] * 10  # anisotropic scale
        ))
        
        cov = model.build_covariance()
        # Use off-diagonal element sum to ensure rotation has nonzero gradient
        loss = cov[:, 0, 1].sum() + cov[:, 0, 2].sum() + cov.sum()
        loss.backward()
        
        assert model._scaling.grad is not None, "No gradient for scaling"
        assert model._rotation.grad is not None, "No gradient for rotation"
        assert model._scaling.grad.abs().sum() > 0, "Zero gradient for scaling"
        assert model._rotation.grad.abs().sum() > 0, "Zero gradient for rotation"


class TestGaussianModelOperations:
    """Test model initialization, add, prune, compact."""
    
    def test_initialize_from_points(self):
        model = GaussianModel(sh_degree=3)
        points = torch.randn(100, 3)
        colors = torch.rand(100, 3)
        model.initialize_from_points(points, colors=colors)
        
        assert model.num_gaussians == 100
        assert model._xyz.shape == (100, 3)
        assert model._scaling.shape == (100, 3)
        assert model._rotation.shape == (100, 4)
        assert model._opacity.shape == (100, 1)
        assert model._normals.shape == (100, 3)
        assert model._state.shape == (100,)
    
    def test_opacity_in_unit_range(self):
        model = GaussianModel(sh_degree=0)
        model.initialize_from_points(torch.randn(50, 3), initial_opacity=0.5)
        opacities = model.opacities
        assert (opacities >= 0).all() and (opacities <= 1).all()
    
    def test_scales_positive(self):
        model = GaussianModel(sh_degree=0)
        model.initialize_from_points(torch.randn(50, 3))
        assert (model.scales > 0).all()
    
    def test_add_gaussians(self):
        model = GaussianModel(sh_degree=0)
        model.initialize_from_points(torch.randn(10, 3))
        assert model.num_gaussians == 10
        
        model.add_gaussians({'xyz': torch.randn(5, 3)})
        assert model.num_gaussians == 15
    
    def test_prune_and_compact(self):
        model = GaussianModel(sh_degree=0)
        model.initialize_from_points(torch.randn(20, 3))
        
        # Prune first 5
        mask = torch.zeros(20, dtype=torch.bool)
        mask[:5] = True
        model.prune_gaussians(mask)
        
        assert model.num_gaussians == 20  # Still in memory
        assert (model._state[:5] == GaussianState.PRUNED).all()
        
        model.compact()
        assert model.num_gaussians == 15  # Removed from memory
    
    def test_active_mask(self):
        model = GaussianModel(sh_degree=0)
        model.initialize_from_points(torch.randn(10, 3))
        mask = torch.tensor([True, False, True, False, False, False, False, False, False, False])
        model.prune_gaussians(mask)
        active = model.get_active_mask()
        assert active.sum() == 8
