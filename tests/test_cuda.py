"""Unit tests for CUDA kernels and backend execution correctness.

Verifies:
1. Numerical correctness of CUDA preprocess vs Python reference projection.
2. Key-value permutation preservation in GPU radix sort.
3. Explicit acknowledgment of custom CUDA rasterizer stub status.
4. Production gsplat backend numerical consistency vs reference rasterizer.
5. Strict fail-fast backend resolution with zero silent fallbacks.
"""
import os
import pytest
import torch
from research.projection import world_to_camera, project_to_screen
from research.rasterizer import render, get_renderer_backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAKernels:
    """Test suite for CUDA kernels and backend execution."""
    
    @classmethod
    def setup_class(cls):
        import adaptive_3dgs._C as _C
        cls._C = _C
        cls.device = 'cuda'
    
    def test_preprocess_gaussians_numerical_correctness(self):
        """Test preprocess_gaussians CUDA kernel matches Python reference projection."""
        N = 100
        positions = torch.randn(N, 3, device=self.device, dtype=torch.float32)
        positions[:, 2] = torch.abs(positions[:, 2]) + 2.0  # Ensure all in front of camera
        scales = torch.zeros(N, 3, device=self.device, dtype=torch.float32)
        rotations = torch.zeros(N, 4, device=self.device, dtype=torch.float32)
        rotations[:, 0] = 1.0  # Identity quaternion (w=1, x=0, y=0, z=0)
        
        view_matrix = torch.eye(4, device=self.device, dtype=torch.float32)
        # Translation offset
        view_matrix[0, 3] = 0.5
        view_matrix[1, 3] = -0.2
        view_matrix[2, 3] = 0.1
        
        proj_matrix = torch.tensor([
            [500.0, 0.0, 320.0, 0.0],
            [0.0, 500.0, 240.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], device=self.device, dtype=torch.float32)
        
        # 1. CUDA kernel execution
        means2d_cuda = self._C.preprocess_gaussians(
            positions, scales, rotations, view_matrix, proj_matrix
        )
        
        assert means2d_cuda.shape == (N, 2)
        assert means2d_cuda.device.type == 'cuda'
        assert not torch.isnan(means2d_cuda).any()
        
        # 2. Python reference calculation
        means_cam = world_to_camera(positions, view_matrix)
        means2d_ref, depths_ref = project_to_screen(means_cam, proj_matrix[:3, :3])
        
        # 3. Numerical verification
        torch.testing.assert_close(
            means2d_cuda,
            means2d_ref,
            rtol=1e-4,
            atol=1e-4,
            msg="CUDA preprocess_gaussians 2D coordinates diverge from reference projection"
        )
    
    def test_radix_sort_permutation_and_numerical_correctness(self):
        """Test radix sort key ordering and key-value permutation preservation."""
        N = 100
        keys = torch.randint(0, 5000, (N,), device=self.device)
        values = torch.arange(N, device=self.device)
        
        sorted_indices = self._C.radix_sort(keys, values)
        assert sorted_indices.shape == (N,)
        
        sorted_keys = keys[sorted_indices]
        sorted_values = values[sorted_indices]
        
        # 1. Keys must be monotonically non-decreasing
        diffs = sorted_keys[1:] - sorted_keys[:-1]
        assert (diffs >= 0).all(), "radix_sort keys are not non-decreasing"
        
        # 2. Sorted keys must match reference ground truth
        ref_sorted_keys, ref_order = torch.sort(keys, stable=True)
        torch.testing.assert_close(sorted_keys, ref_sorted_keys)
        
        # 3. Key-Value permutation must be strictly preserved: keys[sorted_values] == sorted_keys
        assert (keys[sorted_values] == sorted_keys).all(), (
            "radix_sort corrupted key-value association: permutation does not map to original keys"
        )
    
    def test_rasterize_forward_cuda_is_prototype_stub(self):
        """Verify custom CUDA rasterizer stub behavior.
        
        Note: The project uses 'gsplat' for production CUDA rendering in Phase 13.
        cuda/rasterize.cu is currently a prototype kernel stub returning zeros.
        """
        W, H = 64, 48
        N = 20
        means2d = torch.rand(N, 2, device=self.device) * 50.0
        conics = torch.ones(N, 3, device=self.device)
        colors = torch.rand(N, 3, device=self.device)
        opacities = torch.ones(N, device=self.device) * 0.8
        depths = torch.rand(N, device=self.device) * 3.0 + 1.0
        
        out = self._C.rasterize_forward(
            W, H, means2d, conics, colors, opacities, depths
        )
        assert out.shape == (H, W, 3)
        assert out.device.type == 'cuda'
        # Explicit test acknowledging that the custom C++ binding is a stub
        assert (out == 0).all(), "Notice: custom CUDA kernel returned non-zero, check if implemented"
    
    def test_production_backend_gsplat_correctness(self):
        """Test that production gsplat backend renders non-trivial image consistent with reference."""
        W, H = 64, 48
        means3D = torch.tensor([[0.0, 0.0, 2.0]], device=self.device, dtype=torch.float32)
        cov3D = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0) * 0.01
        colors = torch.tensor([[1.0, 0.5, 0.2]], device=self.device, dtype=torch.float32)
        opacities = torch.tensor([0.8], device=self.device, dtype=torch.float32)
        extrinsics = torch.eye(4, device=self.device, dtype=torch.float32)
        intrinsics = torch.tensor([
            [50.0, 0.0, 32.0],
            [0.0, 50.0, 24.0],
            [0.0, 0.0, 1.0]
        ], device=self.device, dtype=torch.float32)
        
        res_gsplat = render(
            means3D, cov3D, colors, opacities, extrinsics, intrinsics,
            W, H, backend='gsplat'
        )
        res_ref = render(
            means3D, cov3D, colors, opacities, extrinsics, intrinsics,
            W, H, backend='reference'
        )
        
        # Check non-empty rendering
        assert res_gsplat['color'].sum() > 10.0
        assert (res_gsplat['depth'] > 0).any()
        assert (res_gsplat['transmission'] < 1.0).any()
        
        # Compare total energy between gsplat and reference within 2% margin
        sum_gsplat = res_gsplat['color'].sum()
        sum_ref = res_ref['color'].sum()
        rel_diff = torch.abs(sum_gsplat - sum_ref) / sum_ref
        assert rel_diff.item() < 0.02, f"gsplat and reference energy diverge: rel_diff={rel_diff.item():.4f}"
    
    def test_renderer_backend_failfast_no_silent_fallback(self):
        """Verify strict backend fail-fast behavior with no silent fallbacks."""
        # 1. Invalid backend name must raise ValueError
        with pytest.raises(ValueError, match="Invalid renderer backend"):
            get_renderer_backend("unknown_backend_foo")
            
        # 2. 'custom_cuda' must raise NotImplementedError (explicit prototype marker)
        W, H = 64, 48
        means3D = torch.tensor([[0.0, 0.0, 2.0]], device=self.device)
        cov3D = torch.eye(3, device=self.device).unsqueeze(0) * 0.01
        colors = torch.tensor([[1.0, 0.5, 0.2]], device=self.device)
        opacities = torch.tensor([0.8], device=self.device)
        extrinsics = torch.eye(4, device=self.device)
        intrinsics = torch.eye(3, device=self.device)
        
        with pytest.raises(NotImplementedError, match="prototype under development"):
            render(means3D, cov3D, colors, opacities, extrinsics, intrinsics, W, H, backend='custom_cuda')
            
        # 3. 'gsplat' on CPU must raise RuntimeError
        means_cpu = means3D.cpu()
        cov_cpu = cov3D.cpu()
        col_cpu = colors.cpu()
        op_cpu = opacities.cpu()
        ext_cpu = extrinsics.cpu()
        int_cpu = intrinsics.cpu()
        with pytest.raises(RuntimeError, match="requires CUDA device"):
            render(means_cpu, cov_cpu, col_cpu, op_cpu, ext_cpu, int_cpu, W, H, backend='gsplat')
