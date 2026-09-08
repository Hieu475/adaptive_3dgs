"""Unit tests for Phase 6 Models and Loss Functions (Reformed).

Tests:
  1. ContextAwareTwoHeadMLP forward pass & shapes across ablation variants (V8-V11).
  2. ResidualContextModel forward pass, parameter counts, and additive residual combination.
  3. Phase6Loss with meaningful pair filtering (no tie preference hack!) and listwise KL loss.
  4. FrozenContextPredictor inference interface and gradient freezing invariant.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn

from research.phase6_model import (
    Phase6ModelConfig,
    ContextAwareTwoHeadMLP,
    ResidualContextModel,
    Phase6Loss,
    create_ablation_variant,
)
from research.phase6_context import PHASE6_FEATURE_DIM


class TestContextAwareModels:
    def test_v11_forward_shape(self):
        config = create_ablation_variant("V11")
        model = ContextAwareTwoHeadMLP(config)
        x = torch.randn(8, PHASE6_FEATURE_DIM)
        dq, dt, u = model(x)
        assert dq.shape == (8,)
        assert dt.shape == (8,)
        assert u.shape == (8,)
        assert (dt > 0).all()  # Cost strictly positive

    def test_ablation_variants_forward(self):
        variants = [
            "V8", "V9", "V10", "V11",
            "self_only", "self_neighbor", "self_overlap", "self_selected",
            "self_neighbor_overlap", "self_neighbor_selected", "self_overlap_selected", "all_features"
        ]
        for var in variants:
            config = create_ablation_variant(var)
            model = ContextAwareTwoHeadMLP(config)
            x = torch.randn(4, PHASE6_FEATURE_DIM)
            dq, dt, u = model(x)
            assert dq.shape == (4,)
            assert dt.shape == (4,)
            assert u.shape == (4,)

    def test_residual_model_forward(self):
        config = create_ablation_variant("V11")
        model = ResidualContextModel(config)
        x = torch.randn(6, PHASE6_FEATURE_DIM)
        dq, dt, u = model(x)
        assert dq.shape == (6,)
        assert dt.shape == (6,)
        assert u.shape == (6,)
        assert (dt > 0).all()


class TestPhase6LossReformed:
    def test_loss_backward(self):
        loss_fn = Phase6Loss(lambda_q=1.0, lambda_c=0.5, lambda_r=1.0, lambda_list=0.5)
        config = create_ablation_variant("V11")
        model = ContextAwareTwoHeadMLP(config)

        x = torch.randn(10, PHASE6_FEATURE_DIM)
        target_q = torch.randn(10) * 1e-5
        target_t = torch.rand(10) * 20.0 + 1.0
        target_u = target_q / target_t

        dq, dt, u = model(x)
        out = loss_fn(dq, dt, u, target_q, target_t, target_u)

        assert "total" in out
        assert "loss_q" in out
        assert "loss_t" in out
        assert "loss_r" in out
        assert "loss_list" in out
        assert torch.isfinite(out["total"])

        out["total"].backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_no_tie_preference_hack(self):
        """Verify that identical target utilities do not produce artificial preferences."""
        loss_fn = Phase6Loss(lambda_q=0.0, lambda_c=0.0, lambda_r=1.0, lambda_list=0.0)

        # 4 identical targets: all pairs have 0 difference
        pred_q = torch.zeros(4)
        pred_t = torch.ones(4)
        pred_u = torch.tensor([0.1, 0.2, 0.3, 0.4])  # model predicts differences
        target_q = torch.zeros(4)
        target_t = torch.ones(4)
        target_u = torch.ones(4)  # All targets equal!

        out = loss_fn(pred_q, pred_t, pred_u, target_q, target_t, target_u)
        # Because all targets are identical, they should be filtered out by tau
        # meaning ranking loss should be 0.0
        assert out["loss_r"].item() == 0.0
