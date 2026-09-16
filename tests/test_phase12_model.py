#!/usr/bin/env python3
"""Unit Tests for Phase 12 Two-Stage Model and Global Context Interface."""
import unittest
import torch
import numpy as np

from research.phase12_protocol import CANONICAL_LOCAL_FEATURES, CANONICAL_GLOBAL_FEATURES, SAFETY_FACTOR
from research.phase12_model import (
    LocalEncoder,
    GlobalContextEncoder,
    PositiveUtilityHead,
    QualityHead,
    CostHead,
    TwoStageUtilityModel,
    TwoStageModelConfig,
    TwoStageLoss,
    TwoStageLossConfig,
)
from research.pipeline import OnlineReconstructionPipeline
from research.phase10_runtime import extract_global_context, Phase10ModelBundle, Phase10Selector


class TestPhase12Model(unittest.TestCase):
    """Test suite for Phase 12 model architecture and loss."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_local_encoder(self):
        enc = LocalEncoder(in_dim=11, hidden_dim=64).to(self.device)
        x = torch.randn(16, 11, device=self.device)
        h = enc(x)
        self.assertEqual(h.shape, (16, 64))
        self.assertTrue(torch.all(torch.isfinite(h)))

    def test_global_context_encoder(self):
        enc = GlobalContextEncoder(in_dim=12, hidden_dim=32).to(self.device)
        c = torch.randn(16, 12, device=self.device)
        h = enc(c)
        self.assertEqual(h.shape, (16, 32))
        self.assertTrue(torch.all(torch.isfinite(h)))

    def test_positive_utility_head(self):
        head = PositiveUtilityHead(in_dim=64, hidden_dim=32).to(self.device)
        h = torch.randn(16, 64, device=self.device)
        p = head(h)
        self.assertEqual(p.shape, (16,))
        self.assertTrue(torch.all(p >= 0.0))
        self.assertTrue(torch.all(p <= 1.0))

    def test_cost_head_strict_positivity(self):
        head = CostHead(in_dim=64, hidden_dim=32, eps_cost=0.001).to(self.device)
        # Test with extreme negative inputs
        h = torch.full((16, 64), -100.0, device=self.device)
        t = head(h)
        self.assertEqual(t.shape, (16,))
        self.assertTrue(torch.all(t >= 0.001))

    def test_twostage_model_variants(self):
        # M2 variant: local only, no positive head
        cfg_m2 = TwoStageModelConfig(use_global=False, use_positive_head=False)
        m2 = TwoStageUtilityModel(cfg_m2).to(self.device)
        x_loc = torch.randn(10, 11, device=self.device)
        p2, q2, t2, u2 = m2(x_loc)
        self.assertEqual(u2.shape, (10,))
        self.assertTrue(torch.all(p2 == 1.0))
        self.assertTrue(torch.allclose(u2, q2 / t2))

        # M3 variant: local only + positive head
        cfg_m3 = TwoStageModelConfig(use_global=False, use_positive_head=True)
        m3 = TwoStageUtilityModel(cfg_m3).to(self.device)
        p3, q3, t3, u3 = m3(x_loc)
        self.assertEqual(u3.shape, (10,))
        self.assertTrue(torch.all(p3 >= 0.0) and torch.all(p3 <= 1.0))
        self.assertTrue(torch.allclose(u3, p3 * (q3 / t3)))

        # M4 variant: local + global + positive head
        cfg_m4 = TwoStageModelConfig(use_global=True, use_positive_head=True)
        m4 = TwoStageUtilityModel(cfg_m4).to(self.device)
        c_glob = torch.randn(12, device=self.device)
        p4, q4, t4, u4 = m4(x_loc, c_glob)
        self.assertEqual(u4.shape, (10,))
        self.assertTrue(torch.all(p4 >= 0.0) and torch.all(p4 <= 1.0))
        self.assertTrue(torch.all(t4 > 0.0))

    def test_thresholded_utility(self):
        cfg = TwoStageModelConfig(use_global=False, use_positive_head=True, decision_threshold=0.50)
        m = TwoStageUtilityModel(cfg).to(self.device)
        x_loc = torch.randn(20, 11, device=self.device)
        u_thresh = m.predict_thresholded_utility(x_loc, tau=0.50)
        self.assertEqual(u_thresh.shape, (20,))
        # Dropped candidates should have large negative sentinel
        p, _, _, _ = m(x_loc)
        dropped = p <= 0.50
        if dropped.any():
            self.assertTrue(torch.all(u_thresh[dropped] < -1e8))

    def test_twostage_loss(self):
        loss_fn = TwoStageLoss(TwoStageLossConfig())
        p = torch.sigmoid(torch.randn(10))
        q = torch.randn(10) * 1e-4
        t = torch.relu(torch.randn(10)) + 0.5
        u = p * (q / t)

        target_q = torch.randn(10) * 1e-4
        target_t = torch.full((10,), 0.5)
        target_u = target_q / target_t

        loss, loss_dict = loss_fn(
            pred_p=p,
            pred_q=q,
            pred_t=t,
            pred_u=u,
            target_q=target_q,
            target_t=target_t,
            target_u=target_u,
            use_positive_head=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_pos", loss_dict)
        self.assertIn("loss_q", loss_dict)
        self.assertIn("loss_t", loss_dict)
        self.assertIn("loss_rank", loss_dict)


class TestPipelineGlobalContextInterface(unittest.TestCase):
    """Test suite for pipeline and runtime global context extraction."""

    def test_pipeline_get_global_context(self):
        pipe = OnlineReconstructionPipeline(device="cpu")
        ctx = pipe.get_global_context(budget_ms=15.0)
        self.assertEqual(ctx.shape, (12,))
        self.assertTrue(np.all(np.isfinite(ctx)))
        self.assertEqual(len(CANONICAL_GLOBAL_FEATURES), 12)

    def test_runtime_extract_global_context(self):
        pipe = OnlineReconstructionPipeline(device="cpu")
        ctx = extract_global_context(pipe, budget_ms=15.0)
        self.assertEqual(ctx.shape, (12,))
        self.assertTrue(np.all(np.isfinite(ctx)))


if __name__ == "__main__":
    unittest.main()
