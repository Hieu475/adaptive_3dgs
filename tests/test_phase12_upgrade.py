"""Unit and regression tests for Phase 12 Upgrade (Phases 12-I, 12-J, 12-K, 12-L/M)."""
import pytest
import numpy as np
import torch

from research.interaction_audit import (
    InteractionAuditExperiment,
    InteractionAuditConfig,
)
from research.conditional_dataset import (
    ConditionalDatasetGenerator,
    ConditionalDatasetConfig,
    H_S_DIM,
)
from research.interaction_model import (
    InteractionAwareModel,
    InteractionAwareLoss,
    ConditionalUtilityDataset,
)
from research.sequential_scheduler import (
    SequentialInteractionScheduler,
    SequentialSchedulerConfig,
)


class DummyPipeline:
    """Mock pipeline for unit testing without full CUDA rasterizer."""
    def __init__(self, n_gaussians=100):
        self.gaussian_model = type("MockModel", (), {
            "num_gaussians": n_gaussians,
            "positions": torch.randn(n_gaussians, 3),
        })()


def test_interaction_audit_summary_and_go_no_go():
    """Test interaction audit metric calculation and GO/NO-GO logic."""
    pipeline = DummyPipeline()
    config = InteractionAuditConfig()
    audit = InteractionAuditExperiment(pipeline, config)

    # Synthetic pair results
    pair_results = [
        {
            'pair_info': {'overlap_bin': 'high', 'depth_bin': 'near'},
            'interaction_metrics': {
                'interaction_residual': -0.005,
                'additivity_ratio': 0.45,
            }
        },
        {
            'pair_info': {'overlap_bin': 'low', 'depth_bin': 'far'},
            'interaction_metrics': {
                'interaction_residual': 0.0001,
                'additivity_ratio': 0.99,
            }
        }
    ]

    # Synthetic conditional results
    conditional_results = [
        {
            'candidate_idx': 0,
            'measurements': [
                {'context_size': 0, 'metrics': {'delta_q_conditional': 0.010}},
                {'context_size': 2, 'overlap_bin': 'high', 'metrics': {'delta_q_conditional': -0.002}},  # Sign flip!
            ]
        },
        {
            'candidate_idx': 1,
            'measurements': [
                {'context_size': 0, 'metrics': {'delta_q_conditional': 0.005}},
                {'context_size': 2, 'overlap_bin': 'low', 'metrics': {'delta_q_conditional': 0.004}},
            ]
        }
    ]

    summary = audit.compute_audit_summary(pair_results, conditional_results)
    assert 'R_flip' in summary
    assert summary['R_flip'] == 0.5  # 1 flip out of 2 evals
    assert summary['n_sign_flips'] == 1
    assert 'I_ij_mean' in summary

    decision = audit.assess_go_no_go(summary)
    assert decision['is_go'] is True
    assert decision['decision'] == 'GO'


def test_compute_ndcg_at_k():
    """Test NDCG calculation."""
    pipeline = DummyPipeline()
    audit = InteractionAuditExperiment(pipeline)

    true_scores = np.array([3.0, 2.0, 1.0, 0.0])
    pred_scores = np.array([3.0, 2.0, 1.0, 0.0])  # Perfect ranking
    ndcg = audit.compute_ndcg_at_k(true_scores, pred_scores, k=3)
    assert np.isclose(ndcg, 1.0)

    reversed_scores = np.array([0.0, 1.0, 2.0, 3.0])
    ndcg_rev = audit.compute_ndcg_at_k(true_scores, reversed_scores, k=3)
    assert ndcg_rev < 1.0


def test_h_S_dimension_and_empty_context():
    """Test h_S permutation-invariant representation computation."""
    pipeline = DummyPipeline(n_gaussians=50)
    gen = ConditionalDatasetGenerator(pipeline)

    N = 50
    all_features = np.random.randn(N, 11).astype(np.float32)
    positions = torch.randn(N, 3)
    contrib_indices = torch.randint(0, N, (10, 10, 4))
    contrib_weights = torch.rand(10, 10, 4)

    # Empty context
    h_S_empty = gen.compute_h_S(
        candidate_idx=5,
        context_indices=[],
        all_features=all_features,
        positions=positions,
        contrib_indices=contrib_indices,
        contrib_weights=contrib_weights,
        h=10,
        w=10,
    )
    assert h_S_empty.shape == (H_S_DIM,)
    assert np.all(h_S_empty == 0.0)

    # Non-empty context
    h_S_active = gen.compute_h_S(
        candidate_idx=5,
        context_indices=[1, 2, 3],
        all_features=all_features,
        positions=positions,
        contrib_indices=contrib_indices,
        contrib_weights=contrib_weights,
        h=10,
        w=10,
    )
    assert h_S_active.shape == (H_S_DIM,)
    assert not np.all(h_S_active == 0.0)


def test_interaction_aware_model_forward_and_loss():
    """Test forward pass and loss computation of InteractionAwareModel."""
    batch_size = 16
    local_dim = 11
    context_dim = 51

    model = InteractionAwareModel(local_dim=local_dim, context_dim=context_dim)
    criterion = InteractionAwareLoss()

    s_i = torch.randn(batch_size, local_dim)
    h_S = torch.randn(batch_size, context_dim)

    preds = model(s_i, h_S)
    assert 'p_i' in preds
    assert 'delta_q' in preds
    assert 'delta_c' in preds
    assert 'utility' in preds

    assert preds['p_i'].shape == (batch_size, 1)
    assert (preds['p_i'] >= 0.0).all() and (preds['p_i'] <= 1.0).all()
    assert (preds['delta_c'] >= 0.0).all()

    targets = {
        'delta_q': torch.randn(batch_size, 1),
        'delta_c': torch.abs(torch.randn(batch_size, 1)) + 0.5,
        'utility': torch.randn(batch_size, 1),
    }

    loss = criterion(preds, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_sequential_scheduler_execution():
    """Test two-stage sequential greedy scheduler."""
    N = 100
    all_features = torch.randn(N, 11)
    positions = torch.randn(N, 3)
    error_scores = torch.rand(N)
    grad_norms = torch.rand(N)

    model = InteractionAwareModel(local_dim=11, context_dim=51)
    config = SequentialSchedulerConfig(
        k_prime=20,
        max_k=10,
        gpu_budget_us=2000.0,
        kappa_risk=0.0,
    )
    scheduler = SequentialInteractionScheduler(model=model, config=config)

    res = scheduler.select_sequential(
        all_features=all_features,
        positions=positions,
        error_scores=error_scores,
        grad_norms=grad_norms,
        budget_us=2000.0,
    )

    assert 'selected_indices' in res
    assert 'selected_mask' in res
    assert 'scheduler_latency_ms' in res
    assert len(res['selected_indices']) <= config.max_k
    assert res['total_pred_cost_us'] <= 2000.0 or len(res['selected_indices']) <= 1


def test_risk_aware_abstention():
    """Test that high risk penalty (kappa_risk) correctly triggers abstention or smaller sets."""
    N = 50
    all_features = torch.randn(N, 11) * 0.1
    positions = torch.randn(N, 3)

    model = InteractionAwareModel(local_dim=11, context_dim=51)

    # Neutral scheduler
    config_neutral = SequentialSchedulerConfig(k_prime=20, max_k=10, kappa_risk=0.0)
    sched_neutral = SequentialInteractionScheduler(model=model, config=config_neutral)
    res_neutral = sched_neutral.select_sequential(all_features=all_features, positions=positions)

    # Risk-averse scheduler (very high kappa)
    config_risk = SequentialSchedulerConfig(k_prime=20, max_k=10, kappa_risk=100.0)
    sched_risk = SequentialInteractionScheduler(model=model, config=config_risk)
    res_risk = sched_risk.select_sequential(all_features=all_features, positions=positions)

    # Risk-averse should pick fewer or equal Gaussians compared to neutral
    assert len(res_risk['selected_indices']) <= len(res_neutral['selected_indices'])
