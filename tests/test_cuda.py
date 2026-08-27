"""Unit tests for CUDA kernels."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAKernels:
    """Test suite for CUDA kernels via C++ extension."""
    
    @classmethod
    def setup_class(cls):
        import adaptive_3dgs._C as _C
        cls._C = _C
        cls.device = 'cuda'
    
    def test_preprocess_gaussians_cuda(self):
        """Test preprocess_gaussians CUDA kernel execution and output shape."""
        N = 100
        positions = torch.randn(N, 3, device=self.device, dtype=torch.float32)
        positions[:, 2] = torch.abs(positions[:, 2]) + 2.0
        scales = torch.zeros(N, 3, device=self.device, dtype=torch.float32)
        rotations = torch.zeros(N, 4, device=self.device, dtype=torch.float32)
        rotations[:, 0] = 1.0
        
        view_matrix = torch.eye(4, device=self.device, dtype=torch.float32)
        proj_matrix = torch.tensor([
            [500.0, 0.0, 320.0, 0.0],
            [0.0, 500.0, 240.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], device=self.device, dtype=torch.float32)
        
        means2d = self._C.preprocess_gaussians(
            positions, scales, rotations, view_matrix, proj_matrix
        )
        
        assert means2d.shape == (N, 2)
        assert means2d.device.type == 'cuda'
        assert not torch.isnan(means2d).any()
    
    def test_radix_sort_cuda(self):
        """Test radix sort CUDA wrapper."""
        N = 50
        keys = torch.randint(0, 1000, (N,), device=self.device)
        values = torch.arange(N, device=self.device)
        
        sorted_indices = self._C.radix_sort(keys, values)
        assert sorted_indices.shape == (N,)
        
        sorted_keys = keys[sorted_indices]
        diffs = sorted_keys[1:] - sorted_keys[:-1]
        assert (diffs >= 0).all()
    
    def test_rasterize_forward_cuda(self):
        """Test rasterize forward CUDA wrapper."""
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
