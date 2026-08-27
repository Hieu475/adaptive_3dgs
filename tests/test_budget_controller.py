"""Tests for Closed-Loop Budget Controller & Latency Benchmarks (Milestone R6)."""
import pytest
import torch
from research.scheduler import BudgetScheduler, OptimizationPolicy
from research.benchmark_budgets import (
    run_budget_experiment,
    run_full_budget_matrix,
    format_budget_table,
)


class TestBudgetFeedbackController:
    """Test closed-loop feedback adaptation."""

    def test_over_budget_increases_cost_estimate(self):
        scheduler = BudgetScheduler(gpu_budget_ms=10.0, cost_per_gaussian_us=1.0)
        initial_cost = scheduler.cost_per_gaussian_us
        
        # Simulate 5 frames running at 20ms (2x over budget)
        for _ in range(5):
            scheduler.adjust_budget_from_profiling(actual_frame_ms=20.0)
            
        assert scheduler.cost_per_gaussian_us > initial_cost, "Cost estimate should scale up when over budget"

    def test_under_budget_decreases_cost_estimate(self):
        scheduler = BudgetScheduler(gpu_budget_ms=20.0, cost_per_gaussian_us=1.0)
        initial_cost = scheduler.cost_per_gaussian_us
        
        # Simulate 5 frames running at 5ms (well under budget)
        for _ in range(5):
            scheduler.adjust_budget_from_profiling(actual_frame_ms=5.0)
            
        assert scheduler.cost_per_gaussian_us < initial_cost, "Cost estimate should scale down when under budget"

    def test_latency_statistics_calculation(self):
        scheduler = BudgetScheduler(gpu_budget_ms=15.0)
        for t in [10.0, 12.0, 14.0, 16.0, 20.0]:
            scheduler.adjust_budget_from_profiling(actual_frame_ms=t)
            
        stats = scheduler.get_latency_statistics()
        assert stats['mean_frame_time_ms'] == pytest.approx(14.4, abs=0.1)
        assert stats['budget_violation_rate'] == pytest.approx(0.4, abs=0.05)  # 2 of 5 frames violated
        assert stats['avg_fps'] > 0
        assert stats['p95_frame_time_ms'] > stats['mean_frame_time_ms']


class TestBudgetBenchmarkHarness:
    """Test budget evaluation runner."""

    @pytest.fixture
    def test_frames(self):
        torch.manual_seed(42)
        H, W = 32, 32
        intrinsics = torch.tensor([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]])
        frames = []
        for i in range(2):
            rgb = torch.rand(H, W, 3) * 0.8 + 0.1
            depth = torch.full((H, W), 2.0)
            pose = torch.eye(4)
            pose[0, 3] = i * 0.02
            frames.append({'rgb': rgb, 'depth': depth, 'pose': pose})
        return frames, intrinsics

    def test_budget_matrix_and_table(self, test_frames):
        frames, intrinsics = test_frames
        ablation = run_full_budget_matrix(
            frames, intrinsics, budgets=[4.0, 16.0, None], device='cpu'
        )
        assert len(ablation['experiments']) == 3
        
        table = format_budget_table(ablation)
        assert isinstance(table, str)
        assert "4ms" in table
        assert "16ms" in table
        assert "Unconstrained" in table
