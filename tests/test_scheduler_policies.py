"""Tests for scheduler optimization policies and cost modeling (Milestone R4)."""
import pytest
import torch
from research.scheduler import (
    BudgetScheduler,
    OptimizationPolicy,
    estimate_gaussian_costs,
)


class TestCostModel:
    """Test per-Gaussian compute cost estimation."""

    def test_cost_scaling_with_screen_area(self):
        screen_areas = torch.tensor([10.0, 100.0, 1000.0])
        costs = estimate_gaussian_costs(screen_areas=screen_areas, base_cost_us=0.5, area_cost_factor=0.002)
        
        # Larger footprint should have strictly higher cost
        assert costs[0] < costs[1] < costs[2]
        assert costs[0].item() == pytest.approx(0.5 + 0.002 * 10.0, abs=1e-5)
        assert costs[2].item() == pytest.approx(0.5 + 0.002 * 1000.0, abs=1e-5)

    def test_cost_scaling_with_sh_degree(self):
        screen_areas = torch.tensor([100.0, 100.0])
        cost_sh0 = estimate_gaussian_costs(screen_areas=screen_areas, sh_degree=0)
        cost_sh3 = estimate_gaussian_costs(screen_areas=screen_areas, sh_degree=3)
        
        assert (cost_sh3 > cost_sh0).all()

    def test_cost_fallback_uniform(self):
        costs = estimate_gaussian_costs(n_gaussians=50, base_cost_us=1.0)
        assert costs.shape == (50,)
        assert (costs == 1.0).all()


class TestOptimizationPolicies:
    """Test policy-based Gaussian selection."""

    @pytest.fixture
    def scheduler_setup(self):
        torch.manual_seed(42)
        N = 100
        importance = torch.rand(N)
        # Sort so we have known ordering
        importance, _ = torch.sort(importance, descending=True)
        tiers = torch.full((N,), 2, dtype=torch.long)  # default C
        tiers[:20] = 0   # top 20 Tier A
        tiers[20:60] = 1 # next 40 Tier B
        confidence = 1.0 - importance # high importance = low confidence
        
        screen_areas = torch.rand(N) * 500.0
        costs = estimate_gaussian_costs(screen_areas=screen_areas)
        
        scheduler = BudgetScheduler(gpu_budget_ms=3.0)
        return scheduler, importance, tiers, confidence, costs, N

    def test_policy_0_full(self, scheduler_setup):
        scheduler, importance, tiers, confidence, costs, N = scheduler_setup
        mask = scheduler.select_by_policy(
            OptimizationPolicy.FULL, importance, tiers=tiers, confidence=confidence
        )
        assert mask.shape == (N,)
        assert mask.all(), "Policy 0 (FULL) must select 100% of Gaussians"

    def test_policy_1_random_ratios(self, scheduler_setup):
        scheduler, importance, tiers, confidence, costs, N = scheduler_setup
        for r in [0.1, 0.25, 0.5, 0.75]:
            mask = scheduler.select_by_policy(
                OptimizationPolicy.RANDOM, importance, ratio=r
            )
            expected_count = int(round(N * r))
            assert mask.sum().item() == expected_count

    def test_policy_2_binary(self, scheduler_setup):
        scheduler, importance, tiers, confidence, costs, N = scheduler_setup
        mask_conf = scheduler.select_by_policy(
            OptimizationPolicy.BINARY, importance, confidence=confidence, binary_threshold=0.5
        )
        # Low confidence (< 0.5) should be selected
        assert (mask_conf == (confidence < 0.5)).all()

        mask_tiers = scheduler.select_by_policy(
            OptimizationPolicy.BINARY, importance, tiers=tiers
        )
        # Tiers A & B (0 and 1) should be selected
        assert (mask_tiers == ((tiers == 0) | (tiers == 1))).all()

    def test_policy_3_top_k(self, scheduler_setup):
        scheduler, importance, tiers, confidence, costs, N = scheduler_setup
        ratio = 0.30
        expected_k = int(round(N * ratio))
        mask = scheduler.select_by_policy(
            OptimizationPolicy.TOP_K, importance, ratio=ratio
        )
        assert mask.sum().item() == expected_k
        # Since importance was sorted descending, the first expected_k must be True
        assert mask[:expected_k].all()
        assert not mask[expected_k:].any()

    def test_top_k_selects_more_importance_than_random(self, scheduler_setup):
        scheduler, importance, tiers, confidence, costs, N = scheduler_setup
        ratio = 0.50
        mask_topk = scheduler.select_by_policy(
            OptimizationPolicy.TOP_K, importance, ratio=ratio
        )
        mask_rand = scheduler.select_by_policy(
            OptimizationPolicy.RANDOM, importance, ratio=ratio
        )
        
        total_imp_topk = importance[mask_topk].sum().item()
        total_imp_rand = importance[mask_rand].sum().item()
        assert total_imp_topk >= total_imp_rand, "Top-K must capture >= total importance than Random"

    def test_policy_4_budget_aware(self, scheduler_setup):
        scheduler, importance, tiers, confidence, costs, N = scheduler_setup
        mask = scheduler.select_by_policy(
            OptimizationPolicy.BUDGET_AWARE, importance, tiers=tiers, cost_estimates=costs
        )
        assert mask.shape == (N,)
        assert mask.any(), "Budget-aware policy should select high value density Gaussians"
        # Total cost of selected Gaussians must not exceed optimize budget
        budget_us = scheduler.gpu_budget_ms * 1000 * scheduler.budget_allocation['optimize']
        selected_cost = costs[mask].sum().item()
        assert selected_cost <= budget_us + max(costs).item()
