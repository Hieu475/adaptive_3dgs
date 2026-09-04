"""Unit tests for Phase 4 Utility Models, Baselines, and Losses."""
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
    LinearTwoHead,
    TwoHeadMLP,
)
from research.utility_losses import (
    quality_loss,
    cost_loss,
    pairwise_utility_loss,
    two_head_loss,
    TwoHeadUtilityLoss,
    LossConfig,
    PairwiseRankingLoss,
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
    model = TwoHeadMLP(in_features=11, hidden_dim=64)
    x = torch.randn(12, 11)
    delta_q, delta_t, utility = model(x)

    assert delta_q.shape == (12,)
    assert delta_t.shape == (12,)
    assert utility.shape == (12,)
    assert (delta_t > 0).all()
    assert torch.allclose(utility, delta_q / delta_t)

    pred_u = model.predict_utility(x)
    assert torch.allclose(pred_u, utility)


def test_two_head_linear_alias_and_forward():
    assert LinearTwoHead is TwoHeadLinear
    model = LinearTwoHead(in_features=11)
    x = torch.randn(6, 11)
    delta_q, delta_t, utility = model(x)

    assert delta_q.shape == (6,)
    assert (delta_t > 0).all()
    assert torch.allclose(utility, delta_q / delta_t)


def test_linear_utility_model_forward():
    model = LinearUtilityModel(in_features=11)
    x = torch.randn(5, 11)
    u = model(x)
    assert u.shape == (5,)
    assert torch.allclose(model.predict_utility(x), u)


def test_quality_and_cost_losses():
    pred_q = torch.tensor([1.0, 2.0])
    target_q = torch.tensor([1.1, 1.9])
    l_q = quality_loss(pred_q, target_q)
    assert l_q.item() >= 0.0

    pred_t = torch.tensor([50.0, 70.0])
    target_t = torch.tensor([48.0, 72.0])
    l_t = cost_loss(pred_t, target_t)
    assert l_t.item() >= 0.0


def test_two_head_loss_functional_and_module():
    target_u = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    pred_u = torch.tensor([1.1, 1.9, 3.1, 3.8], dtype=torch.float32)
    target_q = torch.randn(4)
    target_t = torch.abs(torch.randn(4)) + 1.0
    pred_q = target_q + 0.01
    pred_t = target_t + 0.01

    miner = PairwiseRankingLoss()
    p_i, p_j, w = miner.find_pairs(target_u)
    assert len(p_i) > 0

    # Functional loss
    tot_loss, m_dict = two_head_loss(
        pred_q=pred_q,
        pred_t=pred_t,
        pred_u=pred_u,
        target_q=target_q,
        target_t=target_t,
        pairs_i=p_i,
        pairs_j=p_j,
        pair_weights=w,
        lambda_rank=1.0,
        lambda_q=0.25,
        lambda_t=0.125,
    )
    assert tot_loss.item() > 0.0
    assert "loss_total" in m_dict
    assert "loss_rank" in m_dict
    assert "loss_quality" in m_dict
    assert "loss_cost" in m_dict

    # Module loss
    loss_module = TwoHeadUtilityLoss(LossConfig(lambda_rank=1.0, lambda_q=0.25, lambda_t=0.125))
    mod_loss, mod_m = loss_module(
        pred_q=pred_q,
        pred_t=pred_t,
        pred_u=pred_u,
        target_q=target_q,
        target_t=target_t,
        pairs_i=p_i,
        pairs_j=p_j,
        pair_weights=w,
    )
    assert np.isclose(tot_loss.item(), mod_loss.item(), atol=1e-5)
