"""Phase 12-L & 12-M: Sequential Adaptive Greedy Scheduler with Risk-Aware Abstention.

Implements Sections 16, 17, and 18 of the Adaptive 3DGS Research Upgrade Plan:
- Section 16: Two-Stage Candidate Screening (N -> K' -> K)
- Section 17: Sequential Adaptive Greedy Selection under GPU budget
- Section 18: Risk-Aware Extension (LCB scoring and abstention)
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interaction_model import InteractionAwareModel
from .conditional_dataset import H_S_DIM


@dataclass
class SequentialSchedulerConfig:
    """Configuration for the Sequential Interaction-Aware Scheduler."""
    k_prime: int = 32                   # Stage 1 coarse screening pool size
    max_k: int = 16                     # Max number of Gaussians to select in S
    gpu_budget_us: float = 3000.0       # Optimization budget in microseconds (e.g. 3.0 ms)
    min_positive_prob: float = 0.5      # Minimum probability p_i to be considered safe
    kappa_risk: float = 0.0             # LCB risk penalty factor (0 = neutral, >0 = risk-averse)
    stage1_criterion: str = "error"     # "error", "grad_norm", "pointwise_utility", "error_influence"
    epsilon: float = 1e-6               # Numerical stability epsilon
    device: str = "cpu"


class SequentialInteractionScheduler:
    r"""Sequential Adaptive Greedy Scheduler (Phase 12-L / 12-M).
    
    Replaces the static 'score once, sort once, pack' knapsack with a dynamic
    conditional selection loop:
        S = {}
        B_rem = B
        while feasible candidates remain:
            evaluate V(i | S, B_rem) for i in K' \ S
            filter out unsafe candidates (p_i < thresh, LCB < 0, cost > B_rem)
            select i* = argmax V(i | S, B_rem)
            S <- S U {i*}
            B_rem <- B_rem - Delta_C(i* | S)
    """

    def __init__(
        self,
        model: Optional[InteractionAwareModel] = None,
        config: Optional[SequentialSchedulerConfig] = None,
    ):
        self.config = config or SequentialSchedulerConfig()
        self.model = model
        
        # Determine active device
        if model is not None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def screen_stage1(
        self,
        n_total: int,
        error_scores: Optional[torch.Tensor] = None,
        grad_norms: Optional[torch.Tensor] = None,
        pointwise_utilities: Optional[torch.Tensor] = None,
        influence_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stage 1: Coarse screening N -> K' using cheap heuristics.
        
        Complexity: O(N) ranking.
        """
        k_prime = min(self.config.k_prime, n_total)
        device = self.device

        criterion = self.config.stage1_criterion
        if criterion == "error" and error_scores is not None:
            scores = error_scores.to(device)
        elif criterion == "grad_norm" and grad_norms is not None:
            scores = grad_norms.to(device)
        elif criterion == "pointwise_utility" and pointwise_utilities is not None:
            scores = pointwise_utilities.to(device)
        elif criterion == "error_influence" and error_scores is not None and influence_scores is not None:
            scores = (error_scores * influence_scores).to(device)
        else:
            # Fallback to whatever is available or random
            if error_scores is not None:
                scores = error_scores.to(device)
            elif pointwise_utilities is not None:
                scores = pointwise_utilities.to(device)
            else:
                return torch.randperm(n_total, device=device)[:k_prime]

        # Top-K' candidates
        _, top_indices = torch.topk(scores, k_prime, largest=True)
        return top_indices

    def compute_context_representation(
        self,
        candidate_indices: torch.Tensor,
        selected_indices: List[int],
        all_features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        total_budget_us: float = 3000.0,
    ) -> torch.Tensor:
        r"""Vectorized computation of context representation h_S for a batch of candidates.
        
        Args:
            candidate_indices: (M,) tensor of candidate Gaussian indices in K' \ S
            selected_indices: list of currently selected indices in S
            all_features: (N, 11) canonical feature matrix
            positions: (N, 3) 3D Gaussian positions
            total_budget_us: total budget in microseconds for normalization
            
        Returns:
            h_S_batch: (M, 51) tensor of context representations
        """
        M = candidate_indices.shape[0]
        device = candidate_indices.device

        if len(selected_indices) == 0:
            return torch.zeros((M, H_S_DIM), dtype=torch.float32, device=device)

        # 1. Aggregate statistics of S (44 dims: mean, std, max, min of 11 features)
        S_idx_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
        S_feats = all_features[S_idx_tensor]  # (|S|, 11)
        
        mean_feat = torch.mean(S_feats, dim=0)
        std_feat = torch.std(S_feats, dim=0) if len(selected_indices) > 1 else torch.zeros_like(mean_feat)
        max_feat = torch.max(S_feats, dim=0).values
        min_feat = torch.min(S_feats, dim=0).values
        
        agg_stats = torch.cat([mean_feat, std_feat, max_feat, min_feat], dim=0)  # (44,)
        agg_stats_expanded = agg_stats.unsqueeze(0).expand(M, -1)  # (M, 44)

        # 2. Interaction descriptors (5 dims)
        # Fast spatial approximation: 3D distance and depth conflict
        cand_pos = positions[candidate_indices] if positions is not None else torch.zeros((M, 3), device=device)
        S_pos = positions[S_idx_tensor] if positions is not None else torch.zeros((len(selected_indices), 3), device=device)

        if positions is not None:
            # Pairwise depth differences: (M, |S|)
            depth_diff = torch.abs(cand_pos[:, 2].unsqueeze(1) - S_pos[:, 2].unsqueeze(0))
            mean_depth_conflict = torch.mean(depth_diff, dim=1, keepdim=True)

            # Pairwise 3D Euclidean distances
            dist_3d = torch.cdist(cand_pos, S_pos)  # (M, |S|)
            # Approximate screen IoU inversely related to 3D distance
            approx_iou = torch.clamp(1.0 / (1.0 + 5.0 * dist_3d), 0.0, 1.0)
            mean_iou = torch.mean(approx_iou, dim=1, keepdim=True)
            max_iou = torch.max(approx_iou, dim=1, keepdim=True).values
            alpha_comp = mean_iou
            attr_redundancy = torch.clamp(torch.sum(approx_iou, dim=1, keepdim=True) / max(len(selected_indices), 1), 0.0, 1.0)
        else:
            mean_depth_conflict = torch.zeros((M, 1), device=device)
            mean_iou = torch.zeros((M, 1), device=device)
            max_iou = torch.zeros((M, 1), device=device)
            alpha_comp = torch.zeros((M, 1), device=device)
            attr_redundancy = torch.zeros((M, 1), device=device)

        inter_desc = torch.cat([mean_iou, max_iou, mean_depth_conflict, alpha_comp, attr_redundancy], dim=1)  # (M, 5)

        # 3. Set metadata (2 dims: |S|, budget fraction)
        set_size_val = float(len(selected_indices))
        budget_frac_val = float(len(selected_indices)) / max(self.config.max_k, 1)
        meta = torch.tensor([[set_size_val, budget_frac_val]], dtype=torch.float32, device=device).expand(M, -1)  # (M, 2)

        h_S_batch = torch.cat([agg_stats_expanded, inter_desc, meta], dim=1)  # (M, 51)
        return h_S_batch

    def select_sequential(
        self,
        all_features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        error_scores: Optional[torch.Tensor] = None,
        grad_norms: Optional[torch.Tensor] = None,
        pointwise_utilities: Optional[torch.Tensor] = None,
        influence_scores: Optional[torch.Tensor] = None,
        budget_us: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute the full Two-Stage Sequential Adaptive Greedy Selection loop.
        
        Args:
            all_features: (N, 11) canonical features
            positions: (N, 3) 3D positions
            error_scores: (N,) photometric/depth error
            grad_norms: (N,) gradient norm
            pointwise_utilities: (N,) precomputed pointwise utilities
            influence_scores: (N,) screen footprint / influence
            budget_us: total budget in microseconds (overrides config if provided)
            
        Returns:
            Dict containing selected indices, execution stats, and step history.
        """
        start_time = time.perf_counter()
        N = all_features.shape[0]
        device = all_features.device
        self.device = device
        if self.model is not None:
            self.model.to(device)
        total_budget = budget_us if budget_us is not None else self.config.gpu_budget_us
        remaining_budget = total_budget

        # ── Stage 1: Coarse Screening N -> K' ───────────────────────────
        k_prime_indices = self.screen_stage1(
            n_total=N,
            error_scores=error_scores,
            grad_norms=grad_norms,
            pointwise_utilities=pointwise_utilities,
            influence_scores=influence_scores,
        )

        candidate_set: Set[int] = set(k_prime_indices.cpu().tolist())
        selected_set: List[int] = []
        step_history: List[Dict[str, Any]] = []

        total_pred_quality = 0.0
        total_pred_cost = 0.0

        # ── Stage 2: Sequential Adaptive Greedy Selection ─────────────────
        while len(candidate_set) > 0 and len(selected_set) < self.config.max_k and remaining_budget > 0:
            remaining_cand_list = list(candidate_set)
            cand_tensor = torch.tensor(remaining_cand_list, dtype=torch.long, device=device)
            M = len(remaining_cand_list)

            # 1. Candidate local features s_i (M, 11)
            s_i_batch = all_features[cand_tensor]

            # 2. Compute context representation h_S for current S (M, 51)
            h_S_batch = self.compute_context_representation(
                candidate_indices=cand_tensor,
                selected_indices=selected_set,
                all_features=all_features,
                positions=positions,
                total_budget_us=total_budget,
            )

            # 3. Model forward pass
            if self.model is not None:
                with torch.no_grad():
                    preds = self.model(s_i_batch, h_S_batch)
                    p_i = preds['p_i'].squeeze(1)          # (M,)
                    delta_q = preds['delta_q'].squeeze(1)  # (M,)
                    delta_c = preds['delta_c'].squeeze(1)  # (M,) in ms or normalized
            else:
                # Fallback: heuristics if no model loaded
                p_i = torch.ones(M, device=device)
                delta_q = s_i_batch[:, 0] if s_i_batch.shape[1] > 0 else torch.ones(M, device=device)
                delta_c = torch.full((M,), 0.5, device=device)

            # Convert cost estimate to microseconds
            cost_us = delta_c * 1000.0 if torch.max(delta_c) < 50.0 else delta_c

            # 4. Risk-Aware Scoring (Section 18, Phase 12-M)
            if self.config.kappa_risk > 0.0:
                # Epistemic uncertainty proxy based on distance from training distribution
                sigma_proxy = torch.clamp(torch.std(s_i_batch, dim=1) * 0.2, min=0.01)
                lcb_q = delta_q - self.config.kappa_risk * sigma_proxy
                effective_q = torch.clamp(lcb_q, min=0.0)
                safe_mask = (lcb_q > 0.0) & (p_i >= self.config.min_positive_prob)
            else:
                effective_q = delta_q
                safe_mask = (delta_q > 0.0) & (p_i >= self.config.min_positive_prob)

            # Budget feasibility
            feasible_mask = safe_mask & (cost_us <= remaining_budget)

            if not feasible_mask.any():
                # No safe, feasible candidate remaining -> abstain
                break

            # Utility score V(i | S, B_rem)
            utility_scores = (p_i * effective_q) / (cost_us + self.config.epsilon)
            utility_scores[~feasible_mask] = -1e9

            # 5. Greedy selection
            best_idx_in_batch = torch.argmax(utility_scores).item()
            chosen_global_idx = remaining_cand_list[best_idx_in_batch]
            chosen_cost_us = cost_us[best_idx_in_batch].item()
            chosen_q = delta_q[best_idx_in_batch].item()
            chosen_p = p_i[best_idx_in_batch].item()

            # 6. Update state
            selected_set.append(chosen_global_idx)
            candidate_set.remove(chosen_global_idx)
            remaining_budget -= chosen_cost_us

            total_pred_quality += chosen_q
            total_pred_cost += chosen_cost_us

            step_history.append({
                'step': len(selected_set),
                'chosen_idx': chosen_global_idx,
                'p_i': chosen_p,
                'delta_q': chosen_q,
                'cost_us': chosen_cost_us,
                'utility': utility_scores[best_idx_in_batch].item(),
                'remaining_budget_us': remaining_budget,
            })

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Build boolean mask for all N Gaussians
        selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
        if len(selected_set) > 0:
            selected_mask[torch.tensor(selected_set, dtype=torch.long, device=device)] = True

        return {
            'selected_mask': selected_mask,
            'selected_indices': selected_set,
            'n_selected': len(selected_set),
            'total_pred_quality': total_pred_quality,
            'total_pred_cost_us': total_pred_cost,
            'remaining_budget_us': remaining_budget,
            'scheduler_latency_ms': elapsed_ms,
            'step_history': step_history,
            'abstained': len(selected_set) == 0,
        }
