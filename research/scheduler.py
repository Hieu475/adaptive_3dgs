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
    NO_OP = "no_op"                                         # Policy: 0% optimized (tracking/rendering baseline)
    FULL = "full"                                           # Policy 0: Optimize 100% of Gaussians (unconstrained upper bound)
    RANDOM = "random"                                       # Policy 1: Random selection budget-scaled
    ERROR_ONLY = "error_only"                               # Policy: Optimize top-K ranked strictly by raw photometric/depth error
    ERROR_INFLUENCE = "error_influence"                     # Policy: Instantaneous Error × Contribution Mass baseline
    ERROR_INFLUENCE_TEMPORAL = "error_influence_temporal"   # Policy: Temporal filtered Error × Contribution Mass baseline
    BINARY = "binary"                                       # Policy 2: Binary stable/unstable (RTG-SLAM threshold)
    TOP_K = "top_k"                                         # Policy 3: Continuous importance rank top-K / ratio r
    BUDGET_AWARE = "budget_aware"                           # Policy 4: Importance/Cost knapsack optimization
    OURS = "ours"                                           # Alias for BUDGET_AWARE
    LEARNED_UTILITY = "learned_utility"                     # Policy 5: Two-Head Learned Marginal Utility Knapsack
    ORACLE = "oracle"                                       # Policy 6: Ground truth marginal utility upper bound


def estimate_gaussian_cost_components(
    screen_areas: Optional[torch.Tensor] = None,
    projected_areas: Optional[torch.Tensor] = None,
    n_gaussians: Optional[int] = None,
    base_cost_us: float = 0.5,
    area_cost_factor: float = 0.002,
    sh_degree: int = 0,
    cost_coeffs: Optional[Tuple[float, float, float, float]] = None,
    backend: str = "gsplat",
    cost_backward_per_step_us: float = 0.35,
    cost_optimizer_per_step_us: float = 0.15,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose compute cost into decoupled base and incremental step components.

    Exact Formulation:
        C_i(K) = 0                                if K == 0
        C_i(K) = C_i_base + K * C_i_step          if K >= 1

    where:
        C_i_base = C_overhead + C_render(backend, A_i, SH)
        C_i_step = C_backward + C_optimizer

    Returns:
        (base_costs, step_costs): tuple of (N,) tensors in microseconds
    """
    backend_mult = 1.0
    if backend == "reference":
        backend_mult = 5.0
    elif backend == "custom_cuda":
        backend_mult = 1.0

    step_unit = cost_backward_per_step_us + cost_optimizer_per_step_us

    if cost_coeffs is not None and (projected_areas is not None or screen_areas is not None):
        b0, b1, b2, b3 = cost_coeffs
        ref = projected_areas if projected_areas is not None else screen_areas
        dev = ref.device
        N = ref.shape[0]
        area = projected_areas if projected_areas is not None else torch.zeros(N, device=dev)
        inf = screen_areas if screen_areas is not None else torch.zeros(N, device=dev)
        sh_val = float(sh_degree)
        base = torch.clamp((b0 * backend_mult + b1 * area + b2 * inf + b3 * sh_val), min=0.1)
        step = torch.full((N,), step_unit, device=dev)
        return base, step

    if screen_areas is not None:
        dev = screen_areas.device
        N = screen_areas.shape[0]
        sh_multiplier = 1.0 + 0.1 * sh_degree
        base = (base_cost_us * backend_mult + area_cost_factor * screen_areas) * sh_multiplier
        step = torch.full((N,), step_unit, device=dev)
        return base, step

    if n_gaussians is None:
        raise ValueError("Either screen_areas or n_gaussians must be provided")
    dev = device or torch.device('cpu')
    base = torch.full((n_gaussians,), base_cost_us * backend_mult, device=dev)
    step = torch.full((n_gaussians,), step_unit, device=dev)
    return base, step


def estimate_gaussian_costs(
    screen_areas: Optional[torch.Tensor] = None,
    projected_areas: Optional[torch.Tensor] = None,
    n_gaussians: Optional[int] = None,
    base_cost_us: float = 0.5,
    area_cost_factor: float = 0.002,
    sh_degree: int = 0,
    cost_coeffs: Optional[Tuple[float, float, float, float]] = None,
    n_micro_steps: Optional[int] = None,
    backend: str = "gsplat",
    cost_backward_per_step_us: float = 0.35,
    cost_optimizer_per_step_us: float = 0.15,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Cost model: C_i = C_base_i + K * C_step_i."""
    base_costs, step_costs = estimate_gaussian_cost_components(
        screen_areas=screen_areas,
        projected_areas=projected_areas,
        n_gaussians=n_gaussians,
        base_cost_us=base_cost_us,
        area_cost_factor=area_cost_factor,
        sh_degree=sh_degree,
        cost_coeffs=cost_coeffs,
        backend=backend,
        cost_backward_per_step_us=cost_backward_per_step_us,
        cost_optimizer_per_step_us=cost_optimizer_per_step_us,
        device=device,
    )
    if n_micro_steps is not None:
        return base_costs + float(n_micro_steps) * step_costs
    return base_costs


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
        error_influence_temporal_scores: Optional[torch.Tensor] = None,
        ratio: Optional[float] = None,
        top_k: Optional[int] = None,
        frame_idx: int = 0,
        binary_threshold: float = 0.5,
        utility_scores: Optional[torch.Tensor] = None,
        oracle_scores: Optional[torch.Tensor] = None,
        safety_factor: float = 1.0,
        budget_override_us: Optional[float] = None,
        reject_negative: bool = True,
        seed: int = 42,
        update_counts: Optional[torch.Tensor] = None,
        visibility_mask: Optional[torch.Tensor] = None,
        warmup_steps: int = 3,
        warmup_budget_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Select Gaussians for optimization according to specified policy.
        
        Fairness and Budget Constraint Contract (Phase 5):
          All policies obey the exact same hard budget constraint:
              sum_{i in S_B} C_i <= B
          Random: random ordering -> budget packing
          Error: error ordering -> budget packing
          Error x Influence: instantaneous (error * influence) ordering -> budget packing
          Error x Influence Temporal: temporal (error * influence) ordering -> budget packing
          Heuristic: value density (importance / cost) -> budget packing
          Learned Utility: predicted U_hat ordering -> reject U_hat <= 0 -> safety-aware budget packing
          Oracle: oracle U* ordering -> reject U* <= 0 -> budget packing
        """
        N = importance_scores.shape[0]
        device = importance_scores.device
        policy_str = str(policy).lower()
        if hasattr(policy, "value"):
            policy_str = policy.value
        
        if N == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        
        # Effective budget in microseconds
        if budget_override_us is not None:
            budget_us = float(budget_override_us)
        else:
            budget_us = float(self.gpu_budget_ms * 1000.0 * self.budget_allocation['optimize'] * self.budget_scale_factor)
            
        cost_estimates_provided = cost_estimates is not None
        if cost_estimates is None:
            cost_estimates = torch.full((N,), self.cost_per_gaussian_us, device=device)

        if policy_str in ("no_op", OptimizationPolicy.NO_OP.value):
            return torch.zeros(N, dtype=torch.bool, device=device)

        if policy_str in ("full", OptimizationPolicy.FULL.value):
            return torch.ones(N, dtype=torch.bool, device=device)

        def _pack_by_scores(
            scores: torch.Tensor,
            costs: torch.Tensor,
            max_budget: float,
            reject_neg: bool = False,
            safety: float = 1.0,
            max_k: Optional[int] = None,
        ) -> torch.Tensor:
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            if reject_neg:
                valid_idx = torch.where(scores > 0)[0]
                if len(valid_idx) == 0:
                    return mask  # Empty selection: avoids wasting compute on harmful optimization
                sub_scores = scores[valid_idx]
                sub_costs = costs[valid_idx] * safety
                sub_order = torch.argsort(sub_scores, descending=True)
                cum_costs = torch.cumsum(sub_costs[sub_order], dim=0)
                selected_sub = sub_order[cum_costs <= max_budget + 1e-7]
                if max_k is not None:
                    selected_sub = selected_sub[:max_k]
                mask[valid_idx[selected_sub]] = True
                return mask
            else:
                sub_costs = costs * safety
                order = torch.argsort(scores, descending=True)
                cum_costs = torch.cumsum(sub_costs[order], dim=0)
                selected = order[cum_costs <= max_budget + 1e-7]
                if max_k is not None:
                    selected = selected[:max_k]
                mask[selected] = True
                return mask

        if policy_str in ("random", OptimizationPolicy.RANDOM.value):
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            try:
                g = torch.Generator(device=device)
                g.manual_seed(seed + frame_idx * 17)
                perm = torch.randperm(N, generator=g, device=device)
            except Exception:
                perm = torch.randperm(N, device=device)

            if ratio is not None and not cost_estimates_provided and budget_override_us is None:
                k = int(round(N * ratio))
                mask[perm[:k]] = True
                return mask
            if top_k is not None and not cost_estimates_provided and budget_override_us is None:
                mask[perm[:min(top_k, N)]] = True
                return mask

            cum_cost = torch.cumsum(cost_estimates[perm], dim=0)
            selected = perm[cum_cost <= budget_us + 1e-7]
            if top_k is not None:
                selected = selected[:top_k]
            mask[selected] = True
            return mask

        elif policy_str in ("error_only", OptimizationPolicy.ERROR_ONLY.value):
            score_tensor = error_scores if error_scores is not None else importance_scores
            return _pack_by_scores(score_tensor, cost_estimates, budget_us, max_k=top_k)

        elif policy_str in ("error_influence", OptimizationPolicy.ERROR_INFLUENCE.value):
            score_tensor = error_influence_scores if error_influence_scores is not None else (
                (error_scores if error_scores is not None else importance_scores) * importance_scores
            )
            return _pack_by_scores(score_tensor, cost_estimates, budget_us, max_k=top_k)

        elif policy_str in ("error_influence_temporal", OptimizationPolicy.ERROR_INFLUENCE_TEMPORAL.value):
            score_tensor = error_influence_temporal_scores if error_influence_temporal_scores is not None else (
                error_influence_scores if error_influence_scores is not None else (
                    (error_scores if error_scores is not None else importance_scores) * importance_scores
                )
            )
            return _pack_by_scores(score_tensor, cost_estimates, budget_us, max_k=top_k)

        elif policy_str in ("binary", OptimizationPolicy.BINARY.value):
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            if confidence is not None:
                conf = confidence.squeeze(-1) if confidence.ndim > 1 else confidence
                eligible = conf < binary_threshold
            elif tiers is not None:
                eligible = (tiers == 0) | (tiers == 1)
            else:
                eligible = importance_scores >= binary_threshold
            if not eligible.any():
                return mask
            elig_idx = torch.where(eligible)[0]
            elig_scores = (error_scores[elig_idx] if error_scores is not None else importance_scores[elig_idx])
            elig_costs = cost_estimates[elig_idx]
            order = torch.argsort(elig_scores, descending=True)
            cum_cost = torch.cumsum(elig_costs[order], dim=0)
            selected = order[cum_cost <= budget_us + 1e-7]
            if top_k is not None:
                selected = selected[:top_k]
            mask[elig_idx[selected]] = True
            return mask

        elif policy_str in ("top_k", OptimizationPolicy.TOP_K.value):
            effective_k = top_k
            if effective_k is None and ratio is not None and (not cost_estimates_provided or budget_override_us is None):
                effective_k = int(round(N * ratio))
            if effective_k is not None and budget_override_us is None:
                mask = torch.zeros(N, dtype=torch.bool, device=device)
                _, top_indices = torch.topk(importance_scores, min(effective_k, N))
                mask[top_indices] = True
                return mask
            return _pack_by_scores(importance_scores, cost_estimates, budget_us, max_k=top_k)

        elif policy_str in ("budget_aware", "ours", "heuristic", OptimizationPolicy.BUDGET_AWARE.value, OptimizationPolicy.OURS.value):
            # Knapsack heuristic value density: (Temporal Error × Influence Mass) / cost
            base_score = error_influence_temporal_scores if error_influence_temporal_scores is not None else (
                error_influence_scores if error_influence_scores is not None else importance_scores
            )
            density = base_score / (cost_estimates + 1e-6)

            use_warmup = (
                update_counts is not None and
                warmup_budget_ratio > 0.0 and
                warmup_steps > 0 and
                (update_counts[:N] < warmup_steps).any()
            )
            if not use_warmup:
                return _pack_by_scores(density, cost_estimates, budget_us, max_k=top_k)

            # Two-tier warmup knapsack: guarantee newly spawned primitives get optimized
            vis = visibility_mask[:N] if visibility_mask is not None else torch.ones(N, dtype=torch.bool, device=device)
            uc = update_counts[:N]
            warmup_mask = (uc < warmup_steps) & vis
            warmup_idx = torch.where(warmup_mask)[0]

            if len(warmup_idx) == 0:
                return _pack_by_scores(density, cost_estimates, budget_us, max_k=top_k)

            warmup_budget = budget_us * warmup_budget_ratio
            w_uc = uc[warmup_idx]
            w_density = density[warmup_idx]
            # Prioritize candidates closest to graduating (w_uc == 2 first, then 1, then 0) to prevent backlog, breaking ties with density
            w_score = w_uc.float() * 1e4 + w_density
            w_order = warmup_idx[torch.argsort(w_score, descending=True)]
            w_cum_cost = torch.cumsum(cost_estimates[w_order], dim=0)
            w_selected_local = w_cum_cost <= warmup_budget + 1e-7
            selected_warmup = w_order[w_selected_local]
            # Mature candidates packing with decoupled remaining budget without CPU sync
            rem_budget = budget_us * (1.0 - warmup_budget_ratio)
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[selected_warmup] = True

            # Open-market candidates (not yet selected)
            mature_idx = torch.where(~mask & (density > 0))[0]
            if len(mature_idx) > 0 and rem_budget > 0:
                m_density = density[mature_idx]
                m_costs = cost_estimates[mature_idx]
                m_order = torch.argsort(m_density, descending=True)
                m_cum_cost = torch.cumsum(m_costs[m_order], dim=0)
                m_selected = m_order[m_cum_cost <= rem_budget + 1e-7]
                if top_k is not None:
                    rem_k = max(0, top_k - len(selected_warmup))
                    m_selected = m_selected[:rem_k]
                mask[mature_idx[m_selected]] = True

            return mask

        elif policy_str in ("learned_utility", OptimizationPolicy.LEARNED_UTILITY.value) or (policy_str in ("budget_aware", "ours") and utility_scores is not None):
            if utility_scores is not None:
                if utility_scores.shape[0] < N:
                    eff = torch.cat([utility_scores, torch.zeros(N - utility_scores.shape[0], device=device)])
                else:
                    eff = utility_scores[:N]
            else:
                eff = (importance_scores / (cost_estimates + 1e-6))
            return _pack_by_scores(
                scores=eff,
                costs=cost_estimates,
                max_budget=budget_us,
                reject_neg=reject_negative,
                safety=safety_factor,
                max_k=top_k,
            )

        elif policy_str in ("oracle", OptimizationPolicy.ORACLE.value):
            eff = oracle_scores if oracle_scores is not None else (
                utility_scores if utility_scores is not None else importance_scores
            )
            return _pack_by_scores(
                scores=eff,
                costs=cost_estimates,
                max_budget=budget_us,
                reject_neg=reject_negative,
                safety=1.0,
                max_k=top_k,
            )

        else:
            raise ValueError(f"Unknown optimization policy: {policy}")

    def allocate_adaptive_micro_steps(
        self,
        importance_scores: torch.Tensor,
        base_costs: torch.Tensor,
        step_costs: Optional[Union[torch.Tensor, float]] = None,
        budget_override_us: Optional[float] = None,
        max_k: int = 3,
        update_counts: Optional[torch.Tensor] = None,
        visibility_mask: Optional[torch.Tensor] = None,
        warmup_steps: int = 3,
        warmup_budget_ratio: float = 0.20,
        warmup_k: int = 2,
    ) -> torch.Tensor:
        """Solve Generalized Multi-Choice Knapsack for adaptive micro-steps K_i.
        
        Exact Linear Formulation:
            C_i(K) = 0                                if K == 0
            C_i(K) = base_costs[i] + K * step_costs[i] if K >= 1
            
        Subject to:
            sum_i (base_costs[i] + K_i * step_costs[i]) * 1[K_i > 0] <= Budget
        
        When update_counts is provided (Age-Aware Guaranteed Warm-up):
            Newly spawned Gaussians (update_count < warmup_steps) receive a dedicated
            slice of compute (up to warmup_budget_ratio * Budget) with guaranteed
            warmup_k micro-steps before mature primitives compete for remaining budget.
        
        Args:
            importance_scores: (N,) ranking score (e.g. value density or utility)
            base_costs: (N,) base overhead and rendering cost per Gaussian (K=0 -> 0 cost)
            step_costs: (N,) or float incremental cost per micro-step
            budget_override_us: optional budget override in microseconds
            max_k: maximum micro-steps for top candidates (e.g. 3 or 5)
            update_counts: optional (N,) tensor of prior update counts per Gaussian
            visibility_mask: optional (N,) boolean tensor of visible Gaussians
            warmup_steps: minimum updates before graduating to open knapsack market
            warmup_budget_ratio: fraction of budget reserved for warm-up candidates
            warmup_k: guaranteed micro-steps for warm-up candidates
            
        Returns:
            k_steps: (N,) int64 tensor containing micro-step count for each Gaussian
        """
        N = importance_scores.shape[0]
        device = importance_scores.device
        if N == 0:
            return torch.zeros(0, dtype=torch.long, device=device)
            
        if budget_override_us is not None:
            budget_us = float(budget_override_us)
        else:
            budget_us = float(self.gpu_budget_ms * 1000.0 * self.budget_allocation['optimize'] * self.budget_scale_factor)

        k_steps = torch.zeros(N, dtype=torch.long, device=device)
        step_c = step_costs if isinstance(step_costs, torch.Tensor) else float(step_costs if step_costs is not None else 0.50)

        # Check for warm-up candidates
        use_warmup = (
            update_counts is not None and 
            warmup_budget_ratio > 0.0 and 
            warmup_steps > 0 and 
            (update_counts[:N] < warmup_steps).any()
        )

        spent_budget = 0.0

        if use_warmup:
            uc = update_counts[:N]
            vis = visibility_mask[:N] if visibility_mask is not None else torch.ones(N, dtype=torch.bool, device=device)
            warmup_candidates = torch.where((uc < warmup_steps) & vis)[0]
            
            if len(warmup_candidates) > 0:
                warmup_budget = budget_us * warmup_budget_ratio
                w_uc = uc[warmup_candidates]
                w_imp = importance_scores[warmup_candidates]
                # Prioritize candidates closest to graduating (w_uc == 2 first, then 1, then 0) to prevent backlog, breaking ties with importance
                w_score = w_uc.float() * 1e4 + w_imp
                w_order = warmup_candidates[torch.argsort(w_score, descending=True)]
                
                target_w_k = min(warmup_k, max_k)
                w_base = base_costs[w_order]
                w_step = step_c[w_order] if isinstance(step_c, torch.Tensor) else step_c
                w_costs = w_base + float(target_w_k) * w_step
                
                cum_w_costs = torch.cumsum(w_costs, dim=0)
                within_w_budget = cum_w_costs <= warmup_budget + 1e-7
                selected_w = w_order[within_w_budget]
                
                k_steps[selected_w] = target_w_k
                rem_budget = budget_us * (1.0 - warmup_budget_ratio)
            else:
                rem_budget = budget_us
        else:
            rem_budget = budget_us

        if rem_budget <= 0:
            return k_steps

        # Candidates not yet allocated
        unallocated_mask = (k_steps == 0) & (importance_scores > 0)
        if visibility_mask is not None:
            unallocated_mask = unallocated_mask & visibility_mask[:N]
            
        unallocated_indices = torch.where(unallocated_mask)[0]
        n_eligible = len(unallocated_indices)
        if n_eligible == 0:
            return k_steps

        # Sort remaining candidates by importance_scores
        sub_imp = importance_scores[unallocated_indices]
        order_local = torch.argsort(sub_imp, descending=True)
        ordered_indices = unallocated_indices[order_local]

        cutoff_high = max(1, int(0.15 * n_eligible))
        cutoff_med = max(1, int(0.40 * n_eligible))

        target_k = torch.ones(n_eligible, dtype=torch.long, device=device)
        target_k[:cutoff_high] = max_k
        target_k[cutoff_high:cutoff_med] = min(2, max_k)

        item_base = base_costs[ordered_indices]
        item_step = step_c[ordered_indices] if isinstance(step_c, torch.Tensor) else step_c
        item_costs = item_base + target_k.float() * item_step

        cum_costs = torch.cumsum(item_costs, dim=0)
        within_budget = cum_costs <= rem_budget + 1e-7

        k_steps[ordered_indices[within_budget]] = target_k[within_budget]
        return k_steps
    
    def compute_max_new_gaussians(
        self,
        n_error_pixels: Optional[int] = None,
        current_coverage: Optional[float] = None,
        n_warmup: Optional[int] = None,
        max_warmup_queue: int = 500,
    ) -> int:
        """Compute maximum number of new Gaussians allowed this frame.

        Args:
            n_error_pixels: number of pixels currently flagged for
                densification this frame (e.g. `combined_mask.sum()`). If
                None, falls back to the pure compute-budget-derived cap
                (legacy behavior).
            current_coverage: current frame's scene coverage fraction [0, 1].
                If coverage >= 0.90: throttles densification by 80% (factor 0.20)
                to transition from scene exploration to map refinement.
                If coverage <= 0.80: maintains full exploration budget cap.
                Between 0.80 and 0.90: linearly ramps down.
            n_warmup: current number of Gaussians in the warm-up backlog (update_count < warmup_steps).
                If n_warmup > max_warmup_queue: throttles densification to prevent queue overflow.
            max_warmup_queue: maximum allowable warm-up queue length before backpressure throttling.

        Returns:
            max_new: maximum number of new Gaussians to create this frame
        """
        budget_us = self.gpu_budget_ms * 1000 * self.budget_allocation['densify']
        budget_cap = max(1, int(budget_us / self.cost_densify_us))

        if current_coverage is not None:
            if current_coverage >= 0.90:
                throttle_factor = 0.20
            elif current_coverage <= 0.80:
                throttle_factor = 1.0
            else:
                throttle_factor = 1.0 - 0.80 * ((current_coverage - 0.80) / 0.10)
            budget_cap = max(1, int(budget_cap * throttle_factor))

        if n_warmup is not None and n_warmup > max_warmup_queue:
            backlog_factor = max(0.10, float(max_warmup_queue) / float(n_warmup))
            budget_cap = max(10, int(budget_cap * backlog_factor))

        if n_error_pixels is None:
            return budget_cap
        return max(1, min(budget_cap, int(n_error_pixels)))
    
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
