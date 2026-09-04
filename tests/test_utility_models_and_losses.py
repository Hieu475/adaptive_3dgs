"""Unit tests for Phase 4 utility models, baseline scorers, and loss functions."""
import pytest
import numpy as np
import torch

from research.utility_models import (
    RandomScorer,
    RGBErrorScorer,
    RGBDepthErrorScorer,
    ErrorInfluenceScorer,
    BinaryThresholdScorer,
    LinearUtilityModel,
    TwoHeadLinear,
    TwoHeadMLP,
)
from research.utility_losses import (
    PointwiseTwoHeadLoss,
    PairwiseRankingLoss,
    JointRankingAndPointwiseLoss,
    DirectUtilityRegressionLoss,
)


def test_baseline_scorers():
    X = np.array([
        [0.1, 0.5, 1.0, 10.0, 2.0],
        [0.8, 0.2, 2.0, 20.0, 5.0],
    ], dtype=np.float32)

    s_rgb = RGBErrorScorer(rgb_idx=0).score(X)
    assert np.allclose(s_rgb, [0.1, 0.8])

    s_comb = RGBDepthErrorScorer(rgb_idx=0, depth_idx=1, w_rgb=0.7, w_depth=0.3).score(X)
    expected = [0.7 * 0.1 + 0.3 * 0.5, 0.7 * 0.8 + 0.3 * 0.2]
    assert np.allclose(s_comb, expected)

    s_inf = ErrorInfluenceScorer(rgb_idx=0, depth_idx=1, inf_idx=4).score(X)
    assert np.allclose(s_inf, [(0.1 + 0.5) * 2.0, (0.8 + 0.2) * 5.0])

    s_bin = BinaryThresholdScorer(rgb_idx=0, depth_idx=1).score(X)
    assert s_bin.shape == (2,)


def test_two_head_mlp_architecture_and_forward():
    model = TwoHeadMLP(in_features=11, hidden_dim=32)
    x = torch.randn(8, 11)
    delta_q, delta_t, utility = model(x)

    assert delta_q.shape == (8,)
    assert delta_t.shape == (8,)
    assert utility.shape == (8,)
    # delta_t must be strictly positive due to Softplus + epsilon
    assert (delta_t > 0).all()
    # utility must equal delta_q / delta_t
    assert torch.allclose(utility, delta_q / delta_t)


def test_two_head_linear_architecture():
    model = TwoHeadLinear(in_features=11)
    x = torch.randn(5, 11)
    delta_q, delta_t, utility = model(x)

    assert delta_q.shape == (5,)
    assert (delta_t > 0).all()
    assert torch.allclose(utility, delta_q / delta_t)


def test_losses_joint_and_pairwise():
    target_u = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    pred_u = torch.tensor([1.1, 1.9, 3.1, 3.8], dtype=torch.float32)
    target_q = torch.randn(4)
    target_t = torch.abs(torch.randn(4)) + 1.0
    pred_q = target_q + 0.01
    pred_t = target_t + 0.01

    ranking_loss_module = PairwiseRankingLoss()
    pairs_i, pairs_j, pair_weights = ranking_loss_module.find_pairs(target_u)
    assert len(pairs_i) > 0

    joint_loss_module = JointRankingAndPointwiseLoss()
    total_loss, metrics = joint_loss_module(
        pred_q=pred_q,
        pred_t=pred_t,
        pred_u=pred_u,
        target_q=target_q,
        target_t=target_t,
        pairs_i=pairs_i,
        pairs_j=pairs_j,
        pair_weights=pair_weights,
    )
    assert total_loss.item() > 0.0
    assert "loss_total" in metrics
    assert "loss_pairwise" in metrics
    assert "loss_pointwise" in metrics


def test_two_head_utility_loss_config():
    from research.utility_losses import LossConfig, TwoHeadUtilityLoss
    cfg = LossConfig(lambda_rank=2.0, lambda_q=0.5, lambda_t=0.25)
    loss_fn = TwoHeadUtilityLoss(config=cfg)
    assert loss_fn.config.lambda_rank == 2.0
    assert loss_fn.config.lambda_q == 0.5
    assert loss_fn.config.lambda_t == 0.25


def test_selection_metrics_and_budget():
    from research.utility_metrics import (
        PROTOCOL_BUDGETS,
        rank_candidates,
        select_under_budget,
        compute_confidence_interval_95,
    )
    assert len(PROTOCOL_BUDGETS) == 5
    assert PROTOCOL_BUDGETS == (0.10, 0.20, 0.40, 0.60, 0.80)

    scores = np.array([0.1, 0.9, 0.4, 0.7])
    ranked = rank_candidates(scores)
    assert list(ranked) == [1, 3, 2, 0]

    costs = np.array([10.0, 20.0, 15.0, 25.0])
    selected, real_cost = select_under_budget(scores, costs, budget=45.0)
    # Ranked order: index 1 (cost 20) -> index 3 (cost 25) -> sum = 45.0
    assert selected == [1, 3]
    assert real_cost == 45.0

    ci = compute_confidence_interval_95(std=0.1, n=5)
    assert np.isclose(ci, 1.96 * 0.1 / np.sqrt(5))
