"""Unit and integration tests for Phase 10 End-to-End Runtime.

Verifies:
    1. Protocol integrity and frozen invariants.
    2. Checkpoint and normalizer loading (frozen TwoHeadMLP + OnlineEMANormalizer).
    3. Observable feature extraction (canonical 11-D schema, finiteness).
    4. Model forward pass (A1 transform + B2 online update + TwoHeadMLP).
    5. Phase10Selector policy execution (NO_OP, ERROR_ONLY, OURS, FULL).
    6. Budget enforcement and dual accounting (predicted vs scheduled vs actual).
    7. StateStore continuous update without state reset (S_t -> S_{t+1}).
"""
import os
import sys
import unittest
import numpy as np
import torch

from research.phase10_protocol import (
    validate_protocol_integrity_10,
    to_dict as protocol_to_dict,
    get_checkpoint_path_for_seed,
    get_normalizer_path_for_seed,
    CANONICAL_FEATURE_SCHEMA,
    DEFAULT_BUDGET_MS,
    SAFETY_FACTOR,
    B2_BETA,
)
from research.phase10_runtime import (
    Phase10ModelBundle,
    Phase10Selector,
    extract_online_features,
    update_statestore_closed_loop,
    get_pipeline_config,
)
from research.pipeline import OnlineReconstructionPipeline
from research.gaussian_repr import GaussianModel


class TestPhase10Runtime(unittest.TestCase):
    """Test suite for Phase 10 integration and runtime engine."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42

    def test_01_protocol_integrity(self):
        """Verify protocol invariants match Phase 10 specifications."""
        self.assertTrue(validate_protocol_integrity_10())
        proto = protocol_to_dict()
        self.assertEqual(proto["phase"], "Phase 10: End-to-End Adaptive 3DGS Integration")
        self.assertEqual(proto["invariants"]["b2_beta"], 0.90)
        self.assertEqual(proto["invariants"]["base_representation"], "geometry_relative")
        self.assertEqual(proto["budget_ms"], 15.0)

    def test_02_checkpoint_and_normalizer_loading(self):
        """Verify frozen model checkpoint and normalizer JSON load properly."""
        ckpt_p = get_checkpoint_path_for_seed(self.seed)
        norm_p = get_normalizer_path_for_seed(self.seed)
        self.assertTrue(os.path.exists(ckpt_p), f"Missing checkpoint {ckpt_p}")
        self.assertTrue(os.path.exists(norm_p), f"Missing normalizer {norm_p}")

        bundle = Phase10ModelBundle(seed=self.seed, device=self.device)
        self.assertEqual(bundle.normalizer.beta, 0.90)
        self.assertFalse(bundle.model.training)
        # Check all parameters are frozen (requires_grad = False)
        for param in bundle.model.parameters():
            self.assertFalse(param.requires_grad)

    def test_03_inference_pipeline(self):
        """Verify A1 transform + B2 online update + TwoHeadMLP forward pass."""
        bundle = Phase10ModelBundle(seed=self.seed, device=self.device)
        bundle.reset_normalizer()

        # Synthetic [N=100, 11] feature matrix
        rng = np.random.default_rng(self.seed)
        X_fake = rng.exponential(scale=1.0, size=(100, 11)).astype(np.float32)

        res = bundle.predict(X_fake, frame_id=1, update_normalizer=True)
        self.assertIn("predicted_utility", res)
        self.assertIn("predicted_cost", res)
        self.assertIn("predicted_quality", res)
        self.assertIn("norm_metrics", res)

        self.assertEqual(len(res["predicted_utility"]), 100)
        self.assertEqual(len(res["predicted_cost"]), 100)
        self.assertEqual(len(res["predicted_quality"]), 100)

        # All values must be finite
        self.assertTrue(np.all(np.isfinite(res["predicted_utility"])))
        self.assertTrue(np.all(np.isfinite(res["predicted_cost"])))
        self.assertTrue(np.all(np.isfinite(res["predicted_quality"])))

        # Costs must be strictly positive
        self.assertTrue(np.all(res["predicted_cost"] > 0))

        # Normalizer should have updated history
        self.assertEqual(len(bundle.normalizer.history), 1)

    def test_04_feature_extraction_shape_and_finiteness(self):
        """Verify extract_online_features matches canonical schema and produces finite values."""
        cfg = get_pipeline_config(policy="ours", budget_ms=15.0, seed=self.seed)
        pipe = OnlineReconstructionPipeline(config=cfg, device=self.device)

        # Initialize pipeline with dummy frame
        H, W = 240, 320
        rgb = torch.rand(H, W, 3, device=self.device)
        depth = torch.ones(H, W, device=self.device)
        intrinsics = torch.eye(3, device=self.device)
        pose = torch.eye(4, device=self.device)

        pipe.initialize(rgb=rgb, depth=depth, intrinsics=intrinsics, pose=pose)
        N = pipe.gaussian_model.num_gaussians
        self.assertGreater(N, 0)

        feats = extract_online_features(pipe, N)
        self.assertEqual(feats.shape, (N, 11))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_05_selector_policies_and_budget_enforcement(self):
        """Verify Phase10Selector adheres to budget and executes all policies."""
        bundle = Phase10ModelBundle(seed=self.seed, device=self.device)
        selector = Phase10Selector(
            model_bundle=bundle,
            budget_ms=15.0,
            safety_factor=1.10,
            device=self.device,
        )

        cfg = get_pipeline_config(policy="ours", budget_ms=15.0, seed=self.seed)
        pipe = OnlineReconstructionPipeline(config=cfg, device=self.device)
        H, W = 240, 320
        rgb = torch.rand(H, W, 3, device=self.device)
        depth = torch.ones(H, W, device=self.device)
        intrinsics = torch.eye(3, device=self.device)
        pose = torch.eye(4, device=self.device)
        pipe.initialize(rgb=rgb, depth=depth, intrinsics=intrinsics, pose=pose)
        N = pipe.gaussian_model.num_gaussians

        # 1. NO_OP: exactly 0 selected
        mask_noop, diag_noop = selector.select(pipe, policy="no_op", frame_idx=1)
        self.assertEqual(mask_noop.sum().item(), 0)
        self.assertEqual(diag_noop["n_selected"], 0)
        self.assertEqual(diag_noop["scheduled_cost"], 0.0)

        # 2. ERROR_ONLY: budget scheduled_cost <= budget_ms
        mask_err, diag_err = selector.select(pipe, policy="error_only", frame_idx=1)
        self.assertLessEqual(diag_err["scheduled_cost"], 15.0 + 1e-5)
        self.assertEqual(mask_err.sum().item(), diag_err["n_selected"])

        # 3. OURS (B2): budget scheduled_cost <= budget_ms
        mask_ours, diag_ours = selector.select(pipe, policy="ours", frame_idx=1)
        self.assertLessEqual(diag_ours["scheduled_cost"], 15.0 + 1e-5)
        self.assertEqual(mask_ours.sum().item(), diag_ours["n_selected"])

        # 4. FULL: all Gaussians selected
        mask_full, diag_full = selector.select(pipe, policy="full", frame_idx=1)
        self.assertEqual(mask_full.sum().item(), N)
        self.assertEqual(diag_full["n_selected"], N)

    def test_06_statestore_continuity_across_frames(self):
        """Verify S_t -> S_{t+1} state continuity without state reset."""
        cfg = get_pipeline_config(policy="ours", budget_ms=15.0, seed=self.seed)
        pipe = OnlineReconstructionPipeline(config=cfg, device=self.device)
        H, W = 240, 320
        rgb = torch.rand(H, W, 3, device=self.device)
        depth = torch.ones(H, W, device=self.device)
        intrinsics = torch.eye(3, device=self.device)
        pose = torch.eye(4, device=self.device)
        pipe.initialize(rgb=rgb, depth=depth, intrinsics=intrinsics, pose=pose)

        store = pipe.gaussian_model.state_store
        N = pipe.gaussian_model.num_gaussians
        self.assertEqual(store.num_gaussians, N)

        # Frame 1 update
        opt_mask1 = torch.zeros(N, dtype=torch.bool, device=self.device)
        opt_mask1[:10] = True
        update_statestore_closed_loop(pipe, frame_idx=1, optimize_mask=opt_mask1)

        # Check update counts and ages
        self.assertEqual(store.update_counts[:10].sum().item(), 10)
        self.assertEqual(store.update_counts[10:].sum().item(), 0)
        self.assertTrue(torch.all(store.ages >= 0))

        # Frame 2 update (same 10 Gaussians optimized again)
        update_statestore_closed_loop(pipe, frame_idx=2, optimize_mask=opt_mask1)
        self.assertEqual(store.update_counts[:10].sum().item(), 20)
        self.assertEqual(store.update_counts[10:].sum().item(), 0)

        # Persistence verified across frames
        self.assertEqual(store.num_gaussians, N)


if __name__ == "__main__":
    unittest.main()
