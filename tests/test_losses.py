"""Unit tests for loss functions (research/losses.py).

Tests cover:
1. L1 loss correctness
2. Depth loss with valid mask
3. Normal consistency loss
4. Robust losses (Charbonnier, Huber)
5. Compact loss (entropy-based)
6. Temporal loss
7. Total loss combining all components
8. Gradient flow through all losses
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.losses import (
    color_loss, depth_loss, normal_consistency_loss,
    robust_loss, compact_loss, temporal_loss, total_loss
)


class TestColorLoss:
    def test_identical_images_zero_loss(self):
        img = torch.rand(64, 64, 3)
        assert color_loss(img, img).item() == pytest.approx(0.0, abs=1e-6)
    
    def test_known_l1_value(self):
        pred = torch.ones(2, 2, 3)
        gt = torch.zeros(2, 2, 3)
        assert color_loss(pred, gt).item() == pytest.approx(1.0, abs=1e-6)
    
    def test_with_mask(self):
        pred = torch.ones(4, 4, 3)
        gt = torch.zeros(4, 4, 3)
        mask = torch.zeros(4, 4, 3, dtype=torch.bool)
        mask[0, 0] = True  # Only one pixel
        loss = color_loss(pred, gt, mask)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)
    
    def test_gradient(self):
        pred = torch.rand(8, 8, 3, requires_grad=True)
        gt = torch.rand(8, 8, 3)
        loss = color_loss(pred, gt)
        loss.backward()
        assert pred.grad is not None


class TestDepthLoss:
    def test_identical_zero(self):
        d = torch.rand(64, 64) + 0.1
        assert depth_loss(d, d).item() == pytest.approx(0.0, abs=1e-6)
    
    def test_valid_mask(self):
        pred = torch.ones(4, 4)
        gt = torch.zeros(4, 4)
        gt[0, 0] = 2.0  # Only this pixel is valid (>0)
        loss = depth_loss(pred, gt)
        # Only pixel (0,0) is valid: |1 - 2| = 1
        assert loss.item() == pytest.approx(1.0, abs=1e-6)
    
    def test_gradient(self):
        pred = torch.rand(8, 8) + 0.1
        pred = pred.detach().requires_grad_(True)  # Ensure leaf tensor
        gt = torch.rand(8, 8) + 0.1
        loss = depth_loss(pred, gt)
        loss.backward()
        assert pred.grad is not None


class TestNormalConsistencyLoss:
    def test_identical_normals_zero_loss(self):
        n = torch.randn(10, 3)
        n = n / n.norm(dim=-1, keepdim=True)
        loss = normal_consistency_loss(n, n)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)
    
    def test_opposite_normals_max_loss(self):
        n1 = torch.tensor([[0., 0., 1.0]])
        n2 = torch.tensor([[0., 0., -1.0]])
        loss = normal_consistency_loss(n1, n2)
        assert loss.item() == pytest.approx(2.0, abs=1e-5)  # 1 - (-1) = 2
    
    def test_perpendicular_normals(self):
        n1 = torch.tensor([[1., 0., 0.]])
        n2 = torch.tensor([[0., 1., 0.]])
        loss = normal_consistency_loss(n1, n2)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)  # 1 - 0 = 1


class TestRobustLoss:
    def test_charbonnier_at_zero(self):
        r = torch.tensor([0.0])
        loss = robust_loss(r, 'charbonnier', epsilon=0.01)
        assert loss.item() == pytest.approx(0.0, abs=0.01)
    
    def test_charbonnier_less_than_l2(self):
        """Charbonnier should be less than L2 for large residuals (robust)."""
        r = torch.tensor([5.0])
        l_char = robust_loss(r, 'charbonnier', epsilon=0.01)
        l_l2 = r ** 2
        assert l_char.item() < l_l2.item()
    
    def test_huber_at_zero(self):
        r = torch.tensor([0.0])
        loss = robust_loss(r, 'huber', epsilon=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
    
    def test_huber_gradient(self):
        r = torch.tensor([2.0], requires_grad=True)
        loss = robust_loss(r, 'huber', epsilon=1.0)
        loss.backward()
        assert r.grad is not None


class TestCompactLoss:
    def test_binary_opacities_low_loss(self):
        """Opacities at 0 or 1 should have low entropy."""
        alpha_binary = torch.tensor([[0.99], [0.01], [0.99], [0.01]])
        loss_binary = compact_loss(alpha_binary)
        
        alpha_mid = torch.tensor([[0.5], [0.5], [0.5], [0.5]])
        loss_mid = compact_loss(alpha_mid)
        
        assert loss_binary.item() < loss_mid.item(), \
            "Binary opacities should have lower entropy than 0.5"
    
    def test_gradient(self):
        alpha = torch.tensor([[0.5]], requires_grad=True)
        loss = compact_loss(alpha)
        loss.backward()
        assert alpha.grad is not None


class TestTemporalLoss:
    def test_no_movement_zero_loss(self):
        pos = torch.randn(10, 3)
        assert temporal_loss(pos, pos).item() == pytest.approx(0.0, abs=1e-6)
    
    def test_known_movement(self):
        prev = torch.zeros(1, 3)
        curr = torch.tensor([[1.0, 0.0, 0.0]])
        loss = temporal_loss(prev, curr)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)  # ||[1,0,0]||² = 1


class TestTotalLoss:
    def test_returns_dict(self):
        pred_c = torch.rand(4, 4, 3)
        gt_c = torch.rand(4, 4, 3)
        pred_d = torch.rand(4, 4) + 0.1
        gt_d = torch.rand(4, 4) + 0.1
        
        weights = {'color': 1.0, 'depth': 0.5}
        result = total_loss(pred_c, gt_c, pred_d, gt_d, weights)
        
        assert 'total' in result
        assert 'color' in result
        assert 'depth' in result
    
    def test_gradient_through_total_loss(self):
        pred_c = torch.rand(4, 4, 3, requires_grad=True)
        gt_c = torch.rand(4, 4, 3)
        pred_d = (torch.rand(4, 4) + 0.1).detach().requires_grad_(True)
        gt_d = torch.rand(4, 4) + 0.1
        
        weights = {'color': 1.0, 'depth': 0.5}
        result = total_loss(pred_c, gt_c, pred_d, gt_d, weights)
        result['total'].backward()
        
        assert pred_c.grad is not None
        assert pred_d.grad is not None
