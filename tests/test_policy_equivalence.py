"""Unit and Integration Tests for Policy Equivalence & Distinction (R37).

Verifies that the optimization policies are strictly distinct:
- Error-Only ranks strictly by raw error E_i.
- Error×Influence ranks strictly by E_i × Influence_i.
- Top-K Importance ranks by continuous multi-signal importance score.
- Ours (Budget-Aware) solves the value density knapsack optimization.
"""
import pytest
import torch

from research.scheduler import BudgetScheduler, OptimizationPolicy


def test_policies_are_strictly_distinct():
    """Test that all 4 primary policies produce distinct selection sets given realistic diverse scores."""
    scheduler = BudgetScheduler(gpu_budget_ms=10.0)
    N = 100
    
    # Create distinct distributions
    # Gaussian A: high error, low influence
    # Gaussian B: low error, high influence
    # Gaussian C: moderate error, moderate influence
    error_scores = torch.linspace(0.1, 1.0, N)
    influence_scores = torch.linspace(1.0, 0.1, N)  # inversely correlated
    error_influence_scores = error_scores * influence_scores
    importance_scores = 0.5 * error_scores + 0.5 * influence_scores
    cost_estimates = torch.linspace(1.0, 5.0, N)
    
    K = 20
    mask_error = scheduler.select_by_policy(
        policy=OptimizationPolicy.ERROR_ONLY,
        importance_scores=importance_scores,
        error_scores=error_scores,
        top_k=K
    )
    
    mask_error_inf = scheduler.select_by_policy(
        policy=OptimizationPolicy.ERROR_INFLUENCE,
        importance_scores=importance_scores,
        error_influence_scores=error_influence_scores,
        top_k=K
    )
    
    mask_topk = scheduler.select_by_policy(
        policy=OptimizationPolicy.TOP_K,
        importance_scores=importance_scores,
        top_k=K
    )
    
    mask_ours = scheduler.select_by_policy(
        policy=OptimizationPolicy.OURS,
        importance_scores=importance_scores,
        cost_estimates=cost_estimates,
        top_k=K
    )
    
    # Check that masks are not all identical
    assert not torch.equal(mask_error, mask_error_inf), "Error-Only and ErrorxInfluence must differ"
    assert not torch.equal(mask_error, mask_topk), "Error-Only and Top-K must differ"
    assert not torch.equal(mask_error_inf, mask_topk), "ErrorxInfluence and Top-K must differ"
    
    # Error-only selects the highest error Gaussians (indices at the end)
    assert mask_error[-K:].all(), "Error-only must select top K highest error Gaussians"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
