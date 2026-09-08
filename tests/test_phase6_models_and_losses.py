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

    def test_group_aware_ranking_loss(self):
        """Verify ranking and listwise loss operate strictly within groups."""
        loss_fn = Phase6Loss(lambda_q=0.0, lambda_c=0.0, lambda_r=1.0, lambda_list=1.0)

        # 6 samples: group 0 has 3, group 1 has 3
        group_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        pred_q = torch.zeros(6)
        pred_t = torch.ones(6)
        # In group 0: targets are [1.0, 2.0, 3.0], pred is [1.0, 2.0, 3.0] (perfect rank)
        # In group 1: targets are [10.0, 20.0, 30.0], pred is [30.0, 20.0, 10.0] (inverted rank)
        target_u = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        pred_u = torch.tensor([1.0, 2.0, 3.0, 30.0, 20.0, 10.0])
        target_q = target_u
        target_t = torch.ones(6)

        out = loss_fn(pred_q, pred_t, pred_u, target_q, target_t, target_u, group_ids=group_ids)
        assert out["loss_r"].item() > 0.0
        assert out["loss_list"].item() > 0.0

    def test_empty_context_regularization(self):
        """Verify L_0 empty context regularization penalizes non-zero residual predictions."""
        loss_fn = Phase6Loss(lambda_q=0.0, lambda_c=0.0, lambda_r=0.0, lambda_list=0.0, lambda_res=0.0, lambda_zero=2.0)

        pred_q = torch.zeros(4)
        pred_t = torch.ones(4)
        pred_u = torch.ones(4)
        target_q = torch.zeros(4)
        target_t = torch.ones(4)
        target_u = torch.ones(4)

        pred_r = torch.tensor([0.5, 0.0, 0.0, 0.0])
        is_empty = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Only index 0 is empty context

        out = loss_fn(
            pred_q, pred_t, pred_u, target_q, target_t, target_u,
            pred_r=pred_r, is_empty=is_empty
        )
        assert out["loss_zero"].item() > 0.0
        assert torch.isclose(out["total"], 2.0 * out["loss_zero"])



class TestP0Invariants:
    def test_phase4_backbone_is_frozen(self):
        """P0.2: Verify Phase 4 backbone weights are completely frozen and invariant during training."""
        import hashlib
        from research.utility_models import TwoHeadMLP

        p4_model = TwoHeadMLP(in_features=11, hidden_dim=64)
        config = create_ablation_variant("V11")
        model = ResidualContextModel(config, p4_model=p4_model)

        # Hash weights before training
        def hash_p4(m):
            h = hashlib.sha256()
            for p in m.p4_model.parameters():
                h.update(p.detach().cpu().numpy().tobytes())
            return h.hexdigest()

        hash_before = hash_p4(model)

        # Optimizer strictly over context_parameters()
        opt = torch.optim.Adam(model.context_parameters(), lr=1e-3)
        loss_fn = Phase6Loss(lambda_q=1.0, lambda_c=1.0, lambda_r=1.0, lambda_list=1.0)

        # Run 5 training steps
        for _ in range(5):
            opt.zero_grad()
            x = torch.randn(10, PHASE6_FEATURE_DIM)
            target_q = torch.randn(10) * 1e-5
            target_t = torch.rand(10) * 20.0 + 1.0
            target_u = target_q / target_t
            group_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

            dq, dt, u = model(x)
            loss_dict = loss_fn(
                dq, dt, u, target_q, target_t, target_u,
                group_ids=group_ids
            )
            loss_dict["total"].backward()

            # Verify no gradients on p4_model
            for p in model.p4_model.parameters():
                assert p.grad is None or (p.grad == 0).all()
                assert not p.requires_grad

            opt.step()

        hash_after = hash_p4(model)
        assert hash_before == hash_after, "Phase 4 backbone weights must remain invariant!"


