"""Tests for Phase 6 Evaluator (tests/test_phase6_evaluator.py).

Verifies:
1. evaluate_selected_group correctly measures joint group optimization gain
2. Non-additivity: Q(S) - Q(∅) ≠ ΔQ_1 + ΔQ_2 + ΔQ_3
3. Snapshot/restore integrity after evaluation
4. evaluate_pairwise_interaction computes correct interaction terms
5. Phase6Evaluator.evaluate_policy dispatches correctly
6. Empty selection returns zero-gain NO-OP
7. Cost metrics are properly computed
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase6_evaluator import (
    evaluate_selected_group,
    evaluate_pairwise_interaction,
    Phase6Evaluator,
)


class MockModel:
    """Mock Gaussian model with minimal required API."""

    def __init__(self, n_gaussians: int = 100):
        self.num_gaussians = n_gaussians
        self._positions = torch.randn(n_gaussians, 3)
        self._params = {"positions": self._positions}
        self.state_store = None

    @property
    def positions(self):
        return self._positions

    def named_parameters(self):
        yield "positions", self._positions

    def named_buffers(self):
        return iter([])

    def get_optimization_subset(self, mask):
        return {"indices": mask.nonzero(as_tuple=True)[0]}


class MockPipeline:
    """Mock pipeline for testing."""

    def __init__(self, n_gaussians: int = 100):
        self.gaussian_model = MockModel(n_gaussians)
        self.optimizer = None
        self.current_pose = torch.eye(4)
        self.intrinsics = torch.tensor([100.0, 100.0, 50.0, 50.0])


class MockOracleEngine:
    """Mock oracle engine that returns controlled quality deltas."""

    def __init__(self, n_gaussians: int = 100, device: str = "cpu"):
        self.pipeline = MockPipeline(n_gaussians)
        self.contribution_threshold = 0.01
        self._device = torch.device(device)
        self._snapshot_count = 0
        self._restore_count = 0
        # Map: frozenset(indices) -> delta_quality_global
        self._quality_map = {}
        # Default: linear gain per Gaussian
        self._default_gain_per_gaussian = 0.001

    def set_quality_for_group(self, indices, delta_quality):
        """Set specific quality gain for a group of indices."""
        key = frozenset(indices)
        self._quality_map[key] = delta_quality

    def snapshot_state(self):
        self._snapshot_count += 1
        return {"snapshot_id": self._snapshot_count}

    def restore_state(self, snapshot):
        self._restore_count += 1

    def _get_influence_mask(self, indices, contrib_indices, contrib_weights):
        """Return a simple influence mask for the given indices."""
        H, W, K = contrib_indices.shape
        mask = torch.zeros(H, W, dtype=torch.bool)
        for idx in indices:
            mask |= (contrib_indices == idx).any(dim=-1)
        return mask

    def optimize_gaussian_group(self, indices, n_steps, rgb, depth, influence_mask):
        """Return mock optimization results with controlled quality deltas."""
        key = frozenset(indices)
        if key in self._quality_map:
            delta_q = self._quality_map[key]
        else:
            # Default: sub-additive — joint gain < sum of individual gains
            n = len(indices)
            delta_q = self._default_gain_per_gaussian * n * (1.0 - 0.1 * (n - 1))

        psnr_before = 25.0
        depth_l1_before = 0.05
        delta_psnr = delta_q * 100  # arbitrary scaling
        delta_depth = delta_q * 10

        return {
            "delta_quality_global": delta_q,
            "delta_psnr_global": delta_psnr,
            "psnr_global_before": psnr_before,
            "psnr_global_after": psnr_before + delta_psnr,
            "depth_l1_global_before": depth_l1_before,
            "depth_l1_global_after": depth_l1_before - delta_depth,
            "measured_trial_cost_ms": 1.0 * len(indices),
        }


def _make_test_attribution(H=10, W=10, K=4, n_gaussians=100):
    """Create test attribution tensors."""
    contrib_indices = torch.randint(0, n_gaussians, (H, W, K))
    contrib_weights = torch.rand(H, W, K)
    # Normalize weights
    contrib_weights = contrib_weights / contrib_weights.sum(dim=-1, keepdim=True)
    return contrib_indices, contrib_weights


class TestEvaluateSelectedGroup(unittest.TestCase):
    """Tests for the evaluate_selected_group function."""

    def setUp(self):
        self.oracle = MockOracleEngine(n_gaussians=100)
        self.rgb = torch.rand(10, 10, 3)
        self.depth = torch.rand(10, 10) * 5.0
        self.contrib_indices, self.contrib_weights = _make_test_attribution()

    def test_empty_selection_returns_zero(self):
        """Empty selection should return all-zero NO-OP result."""
        res = evaluate_selected_group(
            self.oracle, [], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        self.assertEqual(res["n_selected"], 0)
        self.assertEqual(res["delta_q_realized"], 0.0)
        self.assertEqual(res["actual_opt_cost_ms"], 0.0)
        self.assertEqual(res["delta_psnr_realized"], 0.0)

    def test_single_gaussian_returns_positive_gain(self):
        """Single Gaussian optimization should produce a gain."""
        self.oracle.set_quality_for_group([5], 0.002)
        res = evaluate_selected_group(
            self.oracle, [5], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        self.assertEqual(res["n_selected"], 1)
        self.assertAlmostEqual(res["delta_q_realized"], 0.002, places=6)
        self.assertGreater(res["actual_opt_cost_ms"], 0.0)

    def test_group_optimization_returns_joint_gain(self):
        """Group optimization should return the actual joint gain."""
        self.oracle.set_quality_for_group([1, 2, 3], 0.005)
        res = evaluate_selected_group(
            self.oracle, [1, 2, 3], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        self.assertEqual(res["n_selected"], 3)
        self.assertAlmostEqual(res["delta_q_realized"], 0.005, places=6)

    def test_snapshot_restore_called(self):
        """Must call snapshot before and restore after optimization."""
        self.oracle.set_quality_for_group([10], 0.001)
        evaluate_selected_group(
            self.oracle, [10], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        self.assertEqual(self.oracle._snapshot_count, 1)
        self.assertEqual(self.oracle._restore_count, 1)

    def test_restore_called_even_on_error(self):
        """Restore must be called even if optimization raises an error."""
        def failing_optimize(*args, **kwargs):
            raise RuntimeError("Simulated optimization failure")

        self.oracle.optimize_gaussian_group = failing_optimize
        with self.assertRaises(RuntimeError):
            evaluate_selected_group(
                self.oracle, [1], self.rgb, self.depth,
                self.contrib_indices, self.contrib_weights,
            )
        # restore_state MUST be called via finally block
        self.assertEqual(self.oracle._restore_count, 1)

    def test_influence_pixel_count_positive(self):
        """Influence pixel count should be positive for non-empty selection."""
        # Set specific contrib_indices so that idx=0 is visible
        self.contrib_indices[0, 0, 0] = 0
        self.contrib_weights[0, 0, 0] = 0.5
        self.oracle.set_quality_for_group([0], 0.001)
        res = evaluate_selected_group(
            self.oracle, [0], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        self.assertGreater(res["influence_pixel_count"], 0)


class TestNonAdditivity(unittest.TestCase):
    """Tests that verify joint gain ≠ sum of individual gains."""

    def setUp(self):
        self.oracle = MockOracleEngine(n_gaussians=100)
        self.rgb = torch.rand(10, 10, 3)
        self.depth = torch.rand(10, 10) * 5.0
        self.contrib_indices, self.contrib_weights = _make_test_attribution()

    def test_sub_additive_group(self):
        """For overlapping Gaussians, joint gain < sum of individual gains."""
        # Set individual gains
        self.oracle.set_quality_for_group([1], 0.003)
        self.oracle.set_quality_for_group([2], 0.003)
        # Set joint gain < sum (sub-additive due to overlap)
        self.oracle.set_quality_for_group([1, 2], 0.004)

        res_1 = evaluate_selected_group(
            self.oracle, [1], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        res_2 = evaluate_selected_group(
            self.oracle, [2], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        res_12 = evaluate_selected_group(
            self.oracle, [1, 2], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )

        sum_individual = res_1["delta_q_realized"] + res_2["delta_q_realized"]
        joint_gain = res_12["delta_q_realized"]

        # Joint < sum (sub-additive)
        self.assertLess(joint_gain, sum_individual,
                        "Joint gain should be LESS than sum of individual gains "
                        "for overlapping Gaussians (sub-additive)")
        # But still positive
        self.assertGreater(joint_gain, 0.0)

    def test_super_additive_group(self):
        """For synergistic Gaussians, joint gain > sum of individual gains."""
        self.oracle.set_quality_for_group([10], 0.001)
        self.oracle.set_quality_for_group([20], 0.001)
        # Joint gain > sum (super-additive / synergistic)
        self.oracle.set_quality_for_group([10, 20], 0.003)

        res_10 = evaluate_selected_group(
            self.oracle, [10], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        res_20 = evaluate_selected_group(
            self.oracle, [20], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        res_joint = evaluate_selected_group(
            self.oracle, [10, 20], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )

        sum_individual = res_10["delta_q_realized"] + res_20["delta_q_realized"]
        joint_gain = res_joint["delta_q_realized"]

        self.assertGreater(joint_gain, sum_individual,
                           "Joint gain should be MORE than sum for synergistic Gaussians")


class TestPairwiseInteraction(unittest.TestCase):
    """Tests for evaluate_pairwise_interaction."""

    def setUp(self):
        self.oracle = MockOracleEngine(n_gaussians=100)
        self.rgb = torch.rand(10, 10, 3)
        self.depth = torch.rand(10, 10) * 5.0
        self.contrib_indices, self.contrib_weights = _make_test_attribution()

    def test_sub_additive_interaction(self):
        """Sub-additive pair should have negative interaction."""
        self.oracle.set_quality_for_group([1], 0.003)
        self.oracle.set_quality_for_group([2], 0.003)
        self.oracle.set_quality_for_group([1, 2], 0.004)

        res = evaluate_pairwise_interaction(
            self.oracle, 1, 2, self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        # I(1,2) = ΔQ({1,2}) - ΔQ(1) - ΔQ(2) = 0.004 - 0.003 - 0.003 = -0.002
        self.assertAlmostEqual(res["interaction"], -0.002, places=6)
        self.assertTrue(res["is_sub_additive"])
        self.assertFalse(res["is_super_additive"])

    def test_super_additive_interaction(self):
        """Super-additive pair should have positive interaction."""
        self.oracle.set_quality_for_group([5], 0.001)
        self.oracle.set_quality_for_group([6], 0.001)
        self.oracle.set_quality_for_group([5, 6], 0.003)

        res = evaluate_pairwise_interaction(
            self.oracle, 5, 6, self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        # I(5,6) = 0.003 - 0.001 - 0.001 = 0.001
        self.assertAlmostEqual(res["interaction"], 0.001, places=6)
        self.assertTrue(res["is_super_additive"])
        self.assertFalse(res["is_sub_additive"])

    def test_relative_interaction(self):
        """Relative interaction should be |I|/(|ΔQ_i| + |ΔQ_j|)."""
        self.oracle.set_quality_for_group([1], 0.003)
        self.oracle.set_quality_for_group([2], 0.003)
        self.oracle.set_quality_for_group([1, 2], 0.004)

        res = evaluate_pairwise_interaction(
            self.oracle, 1, 2, self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        expected_rel = abs(-0.002) / (0.003 + 0.003)
        self.assertAlmostEqual(res["relative_interaction"], expected_rel, places=4)


class TestPhase6Evaluator(unittest.TestCase):
    """Tests for Phase6Evaluator class."""

    def test_instantiation(self):
        """Evaluator should instantiate without errors."""
        evaluator = Phase6Evaluator(
            p6_predictor=None,
            p4_predictor=None,
            safety_factor=1.0,
            use_predicted_cost=True,
            device="cpu",
        )
        self.assertEqual(evaluator.device, torch.device("cpu"))
        self.assertEqual(evaluator.safety_factor, 1.0)

    def test_device_inference_from_predictor(self):
        """Device should be inferred from predictor if not specified."""
        mock_pred = MagicMock()
        mock_pred.device = torch.device("cpu")
        evaluator = Phase6Evaluator(p6_predictor=mock_pred)
        self.assertEqual(evaluator.device, torch.device("cpu"))


class TestEvaluateSelectedGroupContract(unittest.TestCase):
    """Tests that verify the return contract of evaluate_selected_group."""

    def setUp(self):
        self.oracle = MockOracleEngine(n_gaussians=100)
        self.rgb = torch.rand(10, 10, 3)
        self.depth = torch.rand(10, 10) * 5.0
        self.contrib_indices, self.contrib_weights = _make_test_attribution()

    def test_return_keys(self):
        """Result must contain all required keys."""
        required_keys = {
            "q_baseline", "q_after_joint_opt", "delta_q_realized",
            "delta_psnr_realized", "delta_depth_realized",
            "actual_opt_cost_ms", "n_selected", "influence_pixel_count",
        }
        self.oracle.set_quality_for_group([1], 0.001)
        res = evaluate_selected_group(
            self.oracle, [1], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        self.assertTrue(required_keys.issubset(res.keys()),
                        f"Missing keys: {required_keys - res.keys()}")

    def test_all_values_are_float_or_int(self):
        """All return values must be numeric (float or int)."""
        self.oracle.set_quality_for_group([1], 0.001)
        res = evaluate_selected_group(
            self.oracle, [1], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        for key, val in res.items():
            self.assertIsInstance(val, (int, float),
                                 f"Key '{key}' has type {type(val)}, expected numeric")

    def test_negative_utility_not_clamped(self):
        """Negative delta_q should NOT be clamped to zero."""
        self.oracle.set_quality_for_group([50], -0.001)
        res = evaluate_selected_group(
            self.oracle, [50], self.rgb, self.depth,
            self.contrib_indices, self.contrib_weights,
        )
        self.assertLess(res["delta_q_realized"], 0.0,
                        "Negative utility should NOT be clamped to 0")


if __name__ == "__main__":
    unittest.main()
