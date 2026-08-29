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
from enum import Enum
import torch
from typing import Dict, Tuple, Optional, List, Union
import time


class OptimizationPolicy(str, Enum):
    """Optimization selection policies for research benchmarking."""
    FULL = "full"                         # Policy 0: Optimize 100% of Gaussians (unconstrained upper bound)
    RANDOM = "random"                     # Policy 1: Random selection budget-scaled
    ERROR_ONLY = "error_only"             # Policy: Optimize top-K ranked strictly by raw photometric/depth error
    ERROR_INFLUENCE = "error_influence"   # Policy: Strong non-learning baseline (Error × Contribution Mass)
    BINARY = "binary"                     # Policy 2: Binary stable/unstable (RTG-SLAM threshold)
    TOP_K = "top_k"                       # Policy 3: Continuous importance rank top-K / ratio r
    BUDGET_AWARE = "budget_aware"         # Policy 4: Importance/Cost knapsack optimization
    OURS = "ours"                         # Alias for BUDGET_AWARE


def estimate_gaussian_costs(
    screen_areas: Optional[torch.Tensor] = None,
    projected_areas: Optional[torch.Tensor] = None,
    n_gaussians: Optional[int] = None,
    base_cost_us: float = 0.5,
    area_cost_factor: float = 0.002,
    sh_degree: int = 0,
    cost_coeffs: Optional[Tuple[float, float, float, float]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Cost model: C_i = β₀ + β₁·Area_i + β₂·Influence_i + β₃·SH_degree.
    
    Calibrated against measured isolated optimization trials from oracle dataset.
    
    Args:
        screen_areas: (N,) contribution mass or screen areas
        projected_areas: (N,) true geometric projected screen area in px²
        n_gaussians: fallback number of Gaussians
        base_cost_us: base gradient cost per Gaussian in microseconds (β₀)
        area_cost_factor: footprint scaling factor (β₁)
        sh_degree: spherical harmonics degree
        cost_coeffs: optional calibrated (beta_0, beta_1, beta_2, beta_3) tuple
        device: torch device
        
    Returns:
        costs: (N,) estimated microsecond compute costs
    """
    if cost_coeffs is not None and (projected_areas is not None or screen_areas is not None):
        b0, b1, b2, b3 = cost_coeffs
        ref = projected_areas if projected_areas is not None else screen_areas
        dev = ref.device
        N = ref.shape[0]
        area = projected_areas if projected_areas is not None else torch.zeros(N, device=dev)
        inf = screen_areas if screen_areas is not None else torch.zeros(N, device=dev)
        sh_val = float(sh_degree)
        costs = b0 + b1 * area + b2 * inf + b3 * sh_val
        return torch.clamp(costs, min=0.1)
        
    if screen_areas is not None:
        device = screen_areas.device
        sh_multiplier = 1.0 + 0.1 * sh_degree
        return (base_cost_us + area_cost_factor * screen_areas) * sh_multiplier
    
    if n_gaussians is None:
        raise ValueError("Either screen_areas or n_gaussians must be provided")
    return torch.full((n_gaussians,), base_cost_us, device=device or torch.device('cpu'))


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
        self._actual_times: List[float] = []
        self._actual_opt_times: List[float] = []
        self._violation_history: List[float] = []
        
        # 2-Phase Adaptive Budget Feedback Controller
        self.budget_scale_factor: float = 1.0
        self.feedback_lambda: float = 0.3
        self.last_budget_state: Dict[str, Any] = {}
        self.budget_history: List[Dict[str, Any]] = []
    
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
        
        # Budget for optimization in microseconds (scaled dynamically by 2-phase controller)
        budget_us = self.gpu_budget_ms * 1000 * self.budget_allocation['optimize'] * self.budget_scale_factor
        
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
    
    def select_by_policy(
        self,
        policy: Union[str, OptimizationPolicy],
        importance_scores: torch.Tensor,
        tiers: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        cost_estimates: Optional[torch.Tensor] = None,
        error_scores: Optional[torch.Tensor] = None,
        error_influence_scores: Optional[torch.Tensor] = None,
        ratio: float = 0.5,
        top_k: Optional[int] = None,
        frame_idx: int = 0,
        binary_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Select Gaussians for optimization according to specified policy.
        
        Policies (Milestone R4, R19, R23):
            - FULL (Policy 0): 100% of Gaussians selected.
            - RANDOM (Policy 1): Uniform random sample of ratio r (or top_k).
            - ERROR_ONLY: Ranked strictly by raw photometric + depth error (E_i).
            - ERROR_INFLUENCE: Strong non-learning baseline (E_i × Influence_i).
            - BINARY (Policy 2): Binary stable/unstable (e.g. confidence < threshold or tier in {A, B}).
            - TOP_K (Policy 3): Top-K highest continuous importance scores.
            - BUDGET_AWARE (Policy 4): Importance/Cost knapsack selection.
        """
        N = importance_scores.shape[0]
        device = importance_scores.device
        policy_str = str(policy).lower()
        if hasattr(policy, "value"):
            policy_str = policy.value
        
        if N == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        
        # Determine target count K
        if top_k is not None:
            K = min(N, max(1, top_k))
        else:
            K = min(N, max(1, int(round(N * ratio))))
            
        if policy_str in ("full", OptimizationPolicy.FULL.value):
            return torch.ones(N, dtype=torch.bool, device=device)
            
        elif policy_str in ("random", OptimizationPolicy.RANDOM.value):
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            perm = torch.randperm(N, device=device)[:K]
            mask[perm] = True
            return mask
            
        elif policy_str in ("error_only", OptimizationPolicy.ERROR_ONLY.value):
            score_tensor = error_scores if error_scores is not None else importance_scores
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            _, top_indices = torch.topk(score_tensor[:N], min(K, score_tensor.shape[0]))
            mask[top_indices] = True
            return mask
            
        elif policy_str in ("error_influence", OptimizationPolicy.ERROR_INFLUENCE.value):
            score_tensor = error_influence_scores if error_influence_scores is not None else importance_scores
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            _, top_indices = torch.topk(score_tensor[:N], min(K, score_tensor.shape[0]))
            mask[top_indices] = True
            return mask
            
        elif policy_str in ("binary", OptimizationPolicy.BINARY.value):
            if confidence is not None:
                conf = confidence.squeeze(-1) if confidence.ndim > 1 else confidence
                return conf < binary_threshold
            elif tiers is not None:
                return (tiers == 0) | (tiers == 1)
            else:
                return importance_scores >= binary_threshold
                
        elif policy_str in ("top_k", OptimizationPolicy.TOP_K.value):
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            _, top_indices = torch.topk(importance_scores, K)
            mask[top_indices] = True
            return mask
            
        elif policy_str in ("budget_aware", "ours", OptimizationPolicy.BUDGET_AWARE.value, OptimizationPolicy.OURS.value):
            if tiers is None:
                tiers = torch.full((N,), 2, dtype=torch.long, device=device)
                tiers[importance_scores > 0.8] = 0
                tiers[(importance_scores >= 0.2) & (importance_scores <= 0.8)] = 1
            return self.select_for_optimization(
                importance_scores, tiers, cost_estimates=cost_estimates, frame_idx=frame_idx
            )
            
        else:
            raise ValueError(f"Unknown optimization policy: {policy}")
    
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
    
    def adjust_budget_from_profiling(
        self,
        actual_frame_ms: float,
        actual_opt_ms: Optional[float] = None,
        n_optimized: Optional[int] = None,
    ):
        """Closed-loop feedback controller: adapt cost model and budget allocations.
        
        Controls budget compliance by dynamically tuning:
        1. Effective cost per Gaussian (α · Cost_est)
        2. Task budget shares (optimize vs densify vs memory)
        
        Args:
            actual_frame_ms: total measured frame time in ms
            actual_opt_ms: measured optimization time in ms
            n_optimized: number of Gaussians optimized this frame
        """
        self._actual_times.append(actual_frame_ms)
        if len(self._actual_times) > 50:
            self._actual_times = self._actual_times[-50:]
        
        # Track budget violation
        if self.gpu_budget_ms > 0:
            violated = actual_frame_ms > self.gpu_budget_ms
            if not hasattr(self, '_violation_history'):
                self._violation_history = []
            self._violation_history.append(float(violated))
            if len(self._violation_history) > 50:
                self._violation_history = self._violation_history[-50:]
        
        # Update empirical per-Gaussian optimization cost if opt timing provided
        if actual_opt_ms is not None and n_optimized is not None and n_optimized > 0:
            empirical_cost_us = (actual_opt_ms * 1000.0) / n_optimized
            # Smoothly update cost estimate with EMA (α = 0.2)
            self.cost_per_gaussian_us = 0.8 * self.cost_per_gaussian_us + 0.2 * empirical_cost_us
        else:
            # Latency ratio feedback
            recent_times = self._actual_times[-10:]
            avg_actual = sum(recent_times) / len(recent_times)
            ratio = avg_actual / self.gpu_budget_ms if self.gpu_budget_ms > 0 else 1.0
            
            if ratio > 1.05:  # Over budget -> increase cost estimate to select fewer
                scale = min(1.25, 1.0 + 0.5 * (ratio - 1.0))
                self.cost_per_gaussian_us *= scale
            elif ratio < 0.85:  # Under budget -> decrease cost estimate to select more
                scale = max(0.80, 1.0 - 0.3 * (1.0 - ratio))
                self.cost_per_gaussian_us *= scale
                
        # Clamp cost estimate to realistic bounds (0.01 μs to 100 μs)
        self.cost_per_gaussian_us = max(0.01, min(self.cost_per_gaussian_us, 100.0))
        self._frame_count += 1
        
    def get_latency_statistics(self) -> Dict[str, float]:
        """Compute latency distribution and budget violation statistics.
        
        Returns:
            Dict with:
                'mean_frame_time_ms': mean frame latency
                'std_frame_time_ms': latency jitter / standard deviation
                'p95_frame_time_ms': 95th percentile latency
                'p99_frame_time_ms': 99th percentile latency
                'budget_violation_rate': fraction of frames over budget
                'avg_fps': mean throughput
                'min_fps': 5th percentile throughput
        """
        import numpy as np
        if not self._actual_times:
            return {
                'mean_frame_time_ms': 0.0,
                'std_frame_time_ms': 0.0,
                'p95_frame_time_ms': 0.0,
                'p99_frame_time_ms': 0.0,
                'budget_violation_rate': 0.0,
                'avg_fps': 0.0,
                'min_fps': 0.0,
            }
            
        times = np.array(self._actual_times)
        mean_t = float(np.mean(times))
        std_t = float(np.std(times)) if len(times) > 1 else 0.0
        p95_t = float(np.percentile(times, 95)) if len(times) >= 5 else mean_t
        p99_t = float(np.percentile(times, 99)) if len(times) >= 10 else p95_t
        
        violations = getattr(self, '_violation_history', [])
        violation_rate = float(np.mean(violations)) if violations else 0.0
        
        avg_fps = 1000.0 / max(mean_t, 1e-4)
        min_fps = 1000.0 / max(p95_t, 1e-4)
        
        return {
            'mean_frame_time_ms': mean_t,
            'std_frame_time_ms': std_t,
            'p95_frame_time_ms': p95_t,
            'p99_frame_time_ms': p99_t,
            'budget_violation_rate': violation_rate,
            'avg_fps': avg_fps,
            'min_fps': min_fps,
        }


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
