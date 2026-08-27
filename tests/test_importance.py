"""Unit tests for importance scoring and budget scheduler."""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.importance import GaussianImportanceEstimator, Tier
from research.scheduler import BudgetScheduler, RunningStats


class TestImportanceEstimator:
    def test_compute_importance_shape(self):
        est = GaussianImportanceEstimator()
        N = 100
        est.update_statistics(
            depth_errors=torch.rand(N),
            color_errors=torch.rand(N),
            normal_errors=torch.rand(N),
            visibility_mask=torch.ones(N, dtype=torch.bool),
            positions=torch.randn(N, 3),
        )
        importance = est.compute_importance()
        assert importance.shape == (N,)
        assert (importance >= 0).all() and (importance <= 1).all()
    
    def test_high_error_gives_high_importance(self):
        est = GaussianImportanceEstimator()
        N = 10
        depth_err = torch.zeros(N)
        color_err = torch.zeros(N)
        depth_err[0] = 10.0  # One Gaussian with very high error
        color_err[0] = 10.0
        
        est.update_statistics(
            depth_errors=depth_err,
            color_errors=color_err,
            normal_errors=torch.zeros(N),
            visibility_mask=torch.ones(N, dtype=torch.bool),
            positions=torch.randn(N, 3),
        )
        importance = est.compute_importance()
        assert importance[0] > importance[1:].max()
    
    def test_tier_classification(self):
        est = GaussianImportanceEstimator(tau_high=0.7, tau_low=0.3)
        importance = torch.tensor([0.9, 0.5, 0.1, 0.0])
        tiers = est.classify_tier(importance)
        assert tiers[0] == Tier.A  # high
        assert tiers[1] == Tier.B  # medium
        assert tiers[2] == Tier.C  # low
        assert tiers[3] == Tier.C  # very low (but not long-term zero)
    
    def test_confidence_update(self):
        est = GaussianImportanceEstimator()
        conf = torch.tensor([[0.5], [0.5]])
        imp = torch.tensor([1.0, 0.0])
        updated = est.update_confidence(conf, imp, learning_rate=0.5)
        assert updated[0].item() > 0.5
        assert updated[1].item() < 0.5


class TestBudgetScheduler:
    def test_select_respects_budget(self):
        sched = BudgetScheduler(gpu_budget_ms=1.0, cost_per_gaussian_us=10.0)
        N = 100
        importance = torch.rand(N)
        tiers = torch.zeros(N, dtype=torch.long)  # All Tier A
        
        mask = sched.select_for_optimization(importance, tiers)
        n_selected = mask.sum().item()
        
        # Budget = 1.0ms * 0.5 (optimize fraction) = 500us
        # Cost per Gaussian = 10us → max 50 Gaussians
        assert n_selected <= 50
    
    def test_tier_a_always_eligible(self):
        sched = BudgetScheduler(gpu_budget_ms=100.0)  # large budget
        N = 10
        importance = torch.ones(N)
        tiers = torch.zeros(N, dtype=torch.long)  # All Tier A
        mask = sched.select_for_optimization(importance, tiers, frame_idx=0)
        assert mask.all()
    
    def test_tier_c_never_optimized(self):
        sched = BudgetScheduler(gpu_budget_ms=100.0)
        N = 10
        importance = torch.ones(N)
        tiers = torch.full((N,), 2, dtype=torch.long)  # All Tier C
        mask = sched.select_for_optimization(importance, tiers, frame_idx=0)
        assert not mask.any()
    
    def test_allocate_budget_sums_to_total(self):
        sched = BudgetScheduler(gpu_budget_ms=10.0)
        alloc = sched.allocate_budget()
        total = sum(alloc.values())
        assert total == pytest.approx(10.0, abs=1e-6)
    
    def test_adaptive_threshold(self):
        sched = BudgetScheduler()
        depth_errs = torch.randn(100).abs()
        color_errs = torch.randn(100).abs()
        dt, ct = sched.adaptive_threshold(depth_errs, color_errs)
        assert dt > 0
        assert ct > 0
    
    def test_lod_scorer(self):
        sched = BudgetScheduler()
        areas = torch.tensor([100., 10., 1.])
        errors = torch.tensor([0.5, 0.5, 0.5])
        complexity = torch.tensor([1.0, 1.0, 1.0])
        scores = sched.lod_scorer(areas, errors, complexity)
        assert scores[0] > scores[1] > scores[2]


class TestRunningStats:
    def test_mean(self):
        stats = RunningStats()
        stats.update(torch.tensor([1., 2., 3., 4., 5.]))
        assert stats.mean() == pytest.approx(3.0, abs=1e-4)
    
    def test_std(self):
        stats = RunningStats()
        data = torch.tensor([1., 2., 3., 4., 5.])
        stats.update(data)
        expected_std = data.std().item()
        assert abs(stats.std() - expected_std) < 0.5  # Approximate
