"""Budget-Aware Gaussian Scheduler.

Research novelty: Formulates Gaussian optimization as a constrained
optimization problem under GPU compute budget.

Knapsack formulation:
    max  Σ_{i ∈ S} I_i  (total importance)
    s.t. Σ_{i ∈ S} Cost_i ≤ Budget_GPU  (e.g., 3.0 ms)

Budget allocation across tasks:
    - Gaussian optimization (gradient updates)
    - Densification (adding new Gaussians)
    - Rendering quality / LOD
    - Memory management (compaction, eviction)

Adaptive thresholds:
    δ_depth(t) = k · σ_depth(t)
    δ_color(t) = k · σ_color(t)
    threshold = f(scene_complexity, uncertainty, GPU_budget)
"""
import torch
from typing import Dict, Tuple, Optional, List
import time


class BudgetScheduler:
    """GPU budget-aware scheduler for Gaussian optimization.
    
    Decides which Gaussians to optimize each frame, how many new Gaussians
    to add, and at what quality level to render, all under a strict GPU
    time budget.
    """
    
    def __init__(
        self,
        gpu_budget_ms: float = 3.0,
        budget_allocation: Optional[Dict[str, float]] = None,
        tier_schedule: Optional[Dict[str, int]] = None,
        cost_per_gaussian_us: float = 0.5,
        cost_densify_us: float = 2.0,
    ):
        """Initialize budget scheduler.
        
        Args:
            gpu_budget_ms: total GPU budget per frame in milliseconds
            budget_allocation: fraction of budget for each task
                {optimize, densify, render, memory}
            tier_schedule: how often to update each tier
                {A: every N frames, B: every M frames}
            cost_per_gaussian_us: estimated cost per Gaussian optimization step (μs)
            cost_densify_us: estimated cost per new Gaussian creation (μs)
        """
        self.gpu_budget_ms = gpu_budget_ms
        self.budget_allocation = budget_allocation or {
            'optimize': 0.50,
            'densify': 0.20,
            'render': 0.20,
            'memory': 0.10,
        }
        self.tier_schedule = tier_schedule or {
            'A': 1,   # every frame
            'B': 5,   # every 5 frames
            'C': -1,  # never (frozen)
            'D': -1,  # never (prune)
        }
        self.cost_per_gaussian_us = cost_per_gaussian_us
        self.cost_densify_us = cost_densify_us
        
        # Adaptive threshold state
        self._depth_error_stats = RunningStats()
        self._color_error_stats = RunningStats()
        self._frame_count = 0
        
        # Performance tracking
        self._actual_times: List[float] = []
    
    def select_for_optimization(
        self,
        importance_scores: torch.Tensor,
        tiers: torch.Tensor,
        cost_estimates: Optional[torch.Tensor] = None,
        frame_idx: int = 0,
    ) -> torch.Tensor:
        """Select Gaussians for optimization under budget constraint.
        
        Solves approximate knapsack:
            max  Σ_{i ∈ S} I_i
            s.t. Σ_{i ∈ S} Cost_i ≤ Budget_optimize
        
        Uses greedy approximation: sort by importance/cost ratio, pick greedily.
        
        Args:
            importance_scores: (N,) per-Gaussian importance
            tiers: (N,) tier classification (0=A, 1=B, 2=C, 3=D)
            cost_estimates: (N,) per-Gaussian optimization cost estimate.
                If None, uses uniform cost.
            frame_idx: current frame index (for periodic scheduling)
        
        Returns:
            optimize_mask: (N,) boolean mask of Gaussians to optimize
        """
        N = importance_scores.shape[0]
        device = importance_scores.device
        
        # Budget for optimization in microseconds
        budget_us = self.gpu_budget_ms * 1000 * self.budget_allocation['optimize']
        
        # Determine which Gaussians are eligible based on tier schedule
        eligible = torch.zeros(N, dtype=torch.bool, device=device)
        
        # Tier A: every frame
        eligible[tiers == 0] = True
        
        # Tier B: every N frames
        b_interval = self.tier_schedule['B']
        if b_interval > 0 and frame_idx % b_interval == 0:
            eligible[tiers == 1] = True
        
        # Tier C & D: never optimize
        # (already False)
        
        if not eligible.any():
            return torch.zeros(N, dtype=torch.bool, device=device)
        
        # Cost estimates
        if cost_estimates is None:
            cost_estimates = torch.full((N,), self.cost_per_gaussian_us, device=device)
        
        # Greedy knapsack: sort eligible by importance/cost ratio (descending)
        eligible_indices = torch.where(eligible)[0]
        eligible_importance = importance_scores[eligible_indices]
        eligible_cost = cost_estimates[eligible_indices]
        
        # Value density = importance / cost
        value_density = eligible_importance / (eligible_cost + 1e-8)
        sorted_order = torch.argsort(value_density, descending=True)
        
        # Greedily select
        optimize_mask = torch.zeros(N, dtype=torch.bool, device=device)
        remaining_budget = budget_us
        
        for i in sorted_order:
            idx = eligible_indices[i]
            cost = eligible_cost[i].item()
            if cost <= remaining_budget:
                optimize_mask[idx] = True
                remaining_budget -= cost
            if remaining_budget <= 0:
                break
        
        return optimize_mask
    
    def compute_max_new_gaussians(self) -> int:
        """Compute maximum number of new Gaussians allowed this frame.
        
        Returns:
            max_new: maximum number of new Gaussians
        """
        budget_us = self.gpu_budget_ms * 1000 * self.budget_allocation['densify']
        return max(1, int(budget_us / self.cost_densify_us))
    
    def allocate_budget(self) -> Dict[str, float]:
        """Compute actual budget allocation in milliseconds.
        
        Returns:
            Dict mapping task name to allocated milliseconds
        """
        return {
            task: self.gpu_budget_ms * frac
            for task, frac in self.budget_allocation.items()
        }
    
    def adaptive_threshold(
        self,
        depth_errors: torch.Tensor,
        color_errors: torch.Tensor,
        k: float = 2.0,
    ) -> Tuple[float, float]:
        """Compute adaptive thresholds based on error statistics.
        
        δ_depth(t) = k · σ_depth(t)
        δ_color(t) = k · σ_color(t)
        
        Thresholds adapt to current scene difficulty: harder scenes
        (higher variance) get more lenient thresholds.
        
        Args:
            depth_errors: (M,) recent depth errors
            color_errors: (M,) recent color errors
            k: multiplier for standard deviation
        
        Returns:
            depth_threshold, color_threshold
        """
        self._depth_error_stats.update(depth_errors)
        self._color_error_stats.update(color_errors)
        
        depth_threshold = k * self._depth_error_stats.std()
        color_threshold = k * self._color_error_stats.std()
        
        # Clamp to reasonable range
        depth_threshold = max(0.01, min(depth_threshold, 0.5))
        color_threshold = max(0.01, min(color_threshold, 0.3))
        
        return depth_threshold, color_threshold
    
    def lod_scorer(
        self,
        screen_areas: torch.Tensor,
        photometric_errors: torch.Tensor,
        geometric_complexity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute error-driven LOD score.
        
        LOD_score_i = ScreenSpaceArea_i · PhotometricError_i · GeometricComplexity_i
        
        Higher score = more important to render at full detail.
        A Gaussian at an object edge or complex texture region should have
        higher score than one on a flat wall at the same distance.
        
        Args:
            screen_areas: (N,) screen-space area of each Gaussian
            photometric_errors: (N,) per-Gaussian photometric error
            geometric_complexity: (N,) local geometric complexity
        
        Returns:
            lod_scores: (N,) LOD importance scores
        """
        return screen_areas * photometric_errors * geometric_complexity
    
    def adjust_budget_from_profiling(self, actual_ms: float):
        """Feedback loop: adjust cost estimates based on actual timing.
        
        If we consistently under/over-estimate costs, adapt.
        
        Args:
            actual_ms: actual GPU time consumed this frame
        """
        self._actual_times.append(actual_ms)
        if len(self._actual_times) > 10:
            self._actual_times = self._actual_times[-10:]
        
        avg_actual = sum(self._actual_times) / len(self._actual_times)
        ratio = avg_actual / self.gpu_budget_ms if self.gpu_budget_ms > 0 else 1.0
        
        # Adjust cost estimate
        if ratio > 1.1:  # Consistently over budget
            self.cost_per_gaussian_us *= 1.05
        elif ratio < 0.8:  # Under budget, can afford more
            self.cost_per_gaussian_us *= 0.95
        
        self._frame_count += 1


class RunningStats:
    """Online computation of running mean and variance (Welford's algorithm)."""
    
    def __init__(self):
        self.n = 0
        self.mean_val = 0.0
        self.M2 = 0.0
    
    def update(self, values: torch.Tensor):
        """Update with a batch of values."""
        batch_mean = values.mean().item()
        batch_var = values.var().item() if values.numel() > 1 else 0.0
        batch_n = values.numel()
        
        if self.n == 0:
            self.mean_val = batch_mean
            self.M2 = batch_var * batch_n
            self.n = batch_n
        else:
            total_n = self.n + batch_n
            delta = batch_mean - self.mean_val
            self.mean_val = (self.n * self.mean_val + batch_n * batch_mean) / total_n
            self.M2 += batch_var * batch_n + delta**2 * self.n * batch_n / total_n
            self.n = total_n
    
    def std(self) -> float:
        """Return current standard deviation."""
        if self.n < 2:
            return 0.0
        return (self.M2 / self.n) ** 0.5
    
    def mean(self) -> float:
        return self.mean_val
