"""Phase 12-I: Interaction Audit for Adaptive 3DGS.

Core research module: systematically measures whether Gaussian utility
depends on the already-selected context set S, or is truly pointwise.

Mathematical Formulations:

    Pair Interaction (Section 7.1):
        I_{ij} = ΔQ({i,j}) - ΔQ_i - ΔQ_j

    Pair Interaction Ratio (Section 7.2):
        R_{pair} = ΔQ({i,j}) / (ΔQ_i + ΔQ_j + ε)

    Conditional Utility Deviation (Section 7.3):
        D_Q(i,S) = ΔQ(i|S) - ΔQ(i|∅)

    Conditional Utility Ratio (Section 7.3):
        R_Q(i,S) = ΔQ(i|S) / (ΔQ(i|∅) + ε)

    Sign-Flip Rate (Section 8):
        R_{flip} = P[sign(U*(i|S)) ≠ sign(U*(i|∅))]

    Rank Instability (Section 9):
        Spearman ρ, Kendall τ, NDCG@K, Top-K overlap
        between rank(U*(i|∅)) and rank(U*(i|S))

Sampling Design (Section 6):
    Context sizes: |S| ∈ {0, 1, 2, 4}
    Overlap bins: Low (IoU < 0.1), Medium (0.1 ≤ IoU < 0.5), High (IoU ≥ 0.5)
    Depth conflict bins: near, medium, far

GO/NO-GO Checkpoint (Section 11):
    GO if R_flip > 0.05, or R_Q varies strongly with overlap, or |I_ij| is significant.
    NO-GO if U(i|S) ≈ U(i|∅) and rank(i|S) ≈ rank(i|∅) for most candidates.
"""
import os
import json
import time
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import spearmanr, kendalltau
from copy import deepcopy

from .oracle_utility import OracleUtilityExperiment
from .phase6_oracle import ConditionalOracleExperiment, ConditionalOracleConfig
from .phase6_context import (
    build_full_context,
    _get_pixel_mask,
    _get_knn_indices,
    ContextConfig,
    PHASE6_FEATURE_NAMES,
)
from .utility_features import CANONICAL_FEATURE_NAMES
from .phase12_protocol import PHASE12_OUTPUT_DIR, SEEDS


@dataclass
class InteractionAuditConfig:
    """Configuration for Phase 12-I Interaction Audit."""
    n_candidates: int = 30           # candidates per frame to evaluate
    n_pairs: int = 50                # pairs to test per frame
    context_sizes: List[int] = field(default_factory=lambda: [0, 1, 2, 4])
    overlap_bins: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'low': (0.0, 0.1),
        'medium': (0.1, 0.5),
        'high': (0.5, 1.0),
    })
    depth_conflict_quantiles: List[float] = field(default_factory=lambda: [0.33, 0.67])
    n_opt_steps: int = 5
    min_influence_pixels: int = 25
    contribution_threshold: float = 0.01
    epsilon: float = 1e-6
    seed: int = 42


class InteractionAuditExperiment:
    """Phase 12-I: Interaction Audit Engine.
    
    Answers the 5 key questions:
    1. What % of interventions change utility when S changes?
    2. What % of candidates flip utility sign?
    3. Does interaction depend on screen-space overlap?
    4. Does interaction depend on depth conflict / attribution?
    5. Is interaction strong enough to change top-K selection?
    """
    
    def __init__(self, pipeline, config: Optional[InteractionAuditConfig] = None):
        self.config = config or InteractionAuditConfig()
        self.pipeline = pipeline
        
        # Initialize underlying oracle engines
        cond_config = ConditionalOracleConfig(
            n_opt_steps=self.config.n_opt_steps,
            epsilon=self.config.epsilon,
            contribution_threshold=self.config.contribution_threshold,
        )
        self.cond_oracle = ConditionalOracleExperiment(
            pipeline=pipeline,
            config=cond_config,
        )
        self.oracle_engine = self.cond_oracle.oracle_engine
    
    def compute_pairwise_iou(self, idx_i: int, idx_j: int, contrib_indices: torch.Tensor, contrib_weights: torch.Tensor) -> float:
        """Compute screen-space IoU between two Gaussians.
        
        Args:
            idx_i: Gaussian index i
            idx_j: Gaussian index j
            contrib_indices: (H, W, K) tensor of top-K contributing Gaussian indices per pixel
            contrib_weights: (H, W, K) tensor of weights for the top-K Gaussians
            
        Returns:
            IoU score [0, 1]
        """
        mask_i = _get_pixel_mask(idx_i, contrib_indices, contrib_weights, self.config.contribution_threshold)
        mask_j = _get_pixel_mask(idx_j, contrib_indices, contrib_weights, self.config.contribution_threshold)
        
        intersection = torch.logical_and(mask_i, mask_j).sum().item()
        union = torch.logical_or(mask_i, mask_j).sum().item()
        
        if union == 0:
            return 0.0
        return intersection / union
    
    def compute_depth_conflict(self, idx_i: int, idx_j: int, positions: torch.Tensor) -> Dict[str, float]:
        """Compute depth conflict and distance metrics between two Gaussians.
        
        Args:
            idx_i: Gaussian index i
            idx_j: Gaussian index j
            positions: (N, 3) tensor of Gaussian positions
            
        Returns:
            Dictionary with distance and depth conflict metrics.
        """
        pos_i = positions[idx_i]
        pos_j = positions[idx_j]
        
        dist_3d = torch.norm(pos_i - pos_j).item()
        depth_diff = abs(pos_i[2].item() - pos_j[2].item())
        
        return {
            'dist_3d': dist_3d,
            'depth_diff': depth_diff,
        }
    
    def sample_stratified_pairs(self, candidate_pool: List[int], positions: torch.Tensor, contrib_indices: torch.Tensor, contrib_weights: torch.Tensor) -> List[Dict]:
        """Sample pairs stratified by overlap and depth conflict bins.
        
        Args:
            candidate_pool: List of candidate Gaussian indices
            positions: (N, 3) tensor of Gaussian positions
            contrib_indices: (H, W, K) indices tensor
            contrib_weights: (H, W, K) weights tensor
            
        Returns:
            List of dictionary containing pair configurations.
        """
        import random
        random.seed(self.config.seed)
        
        pairs = []
        n_pool = len(candidate_pool)
        
        attempts = 0
        max_attempts = self.config.n_pairs * 10
        
        sampled = set()
        
        while len(pairs) < self.config.n_pairs and attempts < max_attempts:
            attempts += 1
            if len(candidate_pool) < 2:
                break
            i, j = random.sample(candidate_pool, 2)
            pair_key = tuple(sorted([i, j]))
            
            if pair_key in sampled:
                continue
                
            iou = self.compute_pairwise_iou(i, j, contrib_indices, contrib_weights)
            conflict_info = self.compute_depth_conflict(i, j, positions)
            
            overlap_bin = 'low'
            for bin_name, (low, high) in self.config.overlap_bins.items():
                if low <= iou <= high:
                    overlap_bin = bin_name
                    break
                    
            pairs.append({
                'idx_i': i,
                'idx_j': j,
                'iou': iou,
                'overlap_bin': overlap_bin,
                'dist_3d': conflict_info['dist_3d'],
                'depth_diff': conflict_info['depth_diff']
            })
            sampled.add(pair_key)
            
        return pairs
    
    def sample_context_by_overlap(
        self, candidate_idx: int, overlap_bin: str, candidate_pool: List[int], positions: torch.Tensor, 
        contrib_indices: torch.Tensor, contrib_weights: torch.Tensor, context_size: int
    ) -> List[int]:
        """Sample context set S with specific overlap characteristics."""
        import random
        
        low, high = self.config.overlap_bins.get(overlap_bin, (0.0, 1.0))
        
        valid_context = []
        for c_idx in candidate_pool:
            if c_idx == candidate_idx:
                continue
            iou = self.compute_pairwise_iou(candidate_idx, c_idx, contrib_indices, contrib_weights)
            if low <= iou <= high:
                valid_context.append(c_idx)
                
        if len(valid_context) < context_size:
            available = [c for c in candidate_pool if c != candidate_idx and c not in valid_context]
            needed = context_size - len(valid_context)
            if len(available) >= needed:
                valid_context.extend(random.sample(available, needed))
            else:
                valid_context.extend(available)
                
        if not valid_context:
            return []
            
        return random.sample(valid_context, min(context_size, len(valid_context)))
    
    def run_pair_interaction_audit(
        self, rgb_gt: torch.Tensor, depth_gt: torch.Tensor, contrib_indices: torch.Tensor, contrib_weights: torch.Tensor,
        candidate_pool: List[int], positions: torch.Tensor, scene_name: str, frame_idx: int
    ) -> List[Dict]:
        """Run pairwise interaction measurements.
        For each sampled pair (i,j):
        1. Measure ΔQ_i, ΔQ_j, ΔQ({i,j})
        2. Compute I_ij, R_pair
        3. Record IoU, depth conflict, 3D distance
        """
        pairs = self.sample_stratified_pairs(candidate_pool, positions, contrib_indices, contrib_weights)
        
        results = []
        for pair in pairs:
            idx_i = pair['idx_i']
            idx_j = pair['idx_j']
            
            interaction_metrics = self.cond_oracle.measure_pairwise_interaction(
                idx_i=idx_i,
                idx_j=idx_j,
                rgb_gt=rgb_gt,
                depth_gt=depth_gt,
                contrib_indices=contrib_indices,
                contrib_weights=contrib_weights,
            )
            
            results.append({
                'pair_info': pair,
                'interaction_metrics': interaction_metrics
            })
            
        return results
    
    def run_conditional_utility_audit(
        self, rgb_gt: torch.Tensor, depth_gt: torch.Tensor, contrib_indices: torch.Tensor, contrib_weights: torch.Tensor,
        candidate_pool: List[int], positions: torch.Tensor, scene_name: str, frame_idx: int
    ) -> List[Dict]:
        """Run conditional utility measurements.
        For each candidate i, for each context size |S|:
        1. Sample context S (stratified by overlap)
        2. Measure ΔQ(i|∅) and ΔQ(i|S)
        3. Compute D_Q, R_Q
        4. Check for sign flips
        """
        import random
        
        candidates_to_test = random.sample(candidate_pool, min(self.config.n_candidates, len(candidate_pool)))
        
        results = []
        
        for idx in candidates_to_test:
            candidate_results = {
                'candidate_idx': idx,
                'measurements': []
            }
            
            # Baseline: Empty context (U*(i|∅))
            base_metrics = self.cond_oracle.measure_conditional_utility(
                candidate_idx=idx,
                context_indices=[],
                rgb_gt=rgb_gt,
                depth_gt=depth_gt,
                contrib_indices=contrib_indices,
                contrib_weights=contrib_weights,
            )
            
            candidate_results['measurements'].append({
                'context_size': 0,
                'overlap_bin': 'none',
                'context_set': [],
                'metrics': base_metrics
            })
            
            # Contextual measurements
            for size in self.config.context_sizes:
                if size == 0:
                    continue
                    
                for overlap_bin in self.config.overlap_bins.keys():
                    context_set = self.sample_context_by_overlap(
                        candidate_idx=idx,
                        overlap_bin=overlap_bin,
                        candidate_pool=candidate_pool,
                        positions=positions,
                        contrib_indices=contrib_indices,
                        contrib_weights=contrib_weights,
                        context_size=size
                    )
                    
                    if not context_set:
                        continue
                        
                    cond_metrics = self.cond_oracle.measure_conditional_utility(
                        candidate_idx=idx,
                        context_indices=context_set,
                        rgb_gt=rgb_gt,
                        depth_gt=depth_gt,
                        contrib_indices=contrib_indices,
                        contrib_weights=contrib_weights,
                    )
                    
                    candidate_results['measurements'].append({
                        'context_size': size,
                        'overlap_bin': overlap_bin,
                        'context_set': context_set,
                        'metrics': cond_metrics
                    })
                    
            results.append(candidate_results)
            
        return results
    
    def run_full_frame_audit(
        self, rgb_gt: torch.Tensor, depth_gt: torch.Tensor, scene_name: str, frame_idx: int, seed: int = 42
    ) -> Dict[str, Any]:
        """Run complete interaction audit for one frame.
        
        Protocol:
            1. Render with attribution to get per-pixel Gaussian contributions
            2. Build candidate pool from influential visible Gaussians
            3. Run pair interaction audit (Section 7.1-7.2)
            4. Run conditional utility audit (Section 7.3, 8, 9)
            5. Compute summary statistics and GO/NO-GO assessment
        
        Args:
            rgb_gt: Ground-truth RGB image (H, W, 3).
            depth_gt: Ground-truth depth map (H, W).
            scene_name: Scene identifier.
            frame_idx: Frame number.
            seed: Random seed for reproducibility.
            
        Returns:
            Dict with pair_results, conditional_results, summary, and go_no_go.
        """
        H, W = rgb_gt.shape[:2]
        model = self.pipeline.gaussian_model
        
        # 1. Render with attribution to get candidates
        with torch.no_grad():
            render_dict = self.oracle_engine._render_with_attribution(H, W)
            contrib_indices = render_dict['contrib_indices']
            contrib_weights = render_dict['contrib_weights']
            
            # Build candidate pool: visible Gaussians with sufficient influence
            flat_indices = contrib_indices.reshape(-1, contrib_indices.shape[-1])
            flat_weights = contrib_weights.reshape(-1, contrib_weights.shape[-1])
            
            valid_mask = flat_weights > self.config.contribution_threshold
            influential = flat_indices[valid_mask].unique().tolist()
            
            # Filter to valid range
            N = model.num_gaussians
            candidate_pool = [idx for idx in influential if 0 <= idx < N]
            positions = model.positions
            
        # 3. Run pair interaction audit
        pair_results = self.run_pair_interaction_audit(
            rgb_gt, depth_gt, contrib_indices, contrib_weights,
            candidate_pool, positions, scene_name, frame_idx
        )
        
        # 4. Run conditional utility audit
        conditional_results = self.run_conditional_utility_audit(
            rgb_gt, depth_gt, contrib_indices, contrib_weights,
            candidate_pool, positions, scene_name, frame_idx
        )
        
        # 5. Compute summary
        summary = self.compute_audit_summary(pair_results, conditional_results)
        
        return {
            'scene': scene_name,
            'frame': frame_idx,
            'n_candidates': len(candidate_pool),
            'n_pairs': len(pair_results),
            'pair_results': pair_results,
            'conditional_results': conditional_results,
            'summary': summary,
            'go_no_go': self.assess_go_no_go(summary)
        }
    
    def compute_audit_summary(self, pair_results: List[Dict], conditional_results: List[Dict]) -> Dict[str, Any]:
        """Compute all summary statistics from Phase 12-I audit data.

        Computes:
            - R_flip: sign-flip rate, overall and by overlap bin
            - D_Q: conditional utility deviation distribution
            - R_Q: conditional utility ratio distribution
            - I_ij: pairwise interaction distribution by overlap/depth bins
            - Rank instability: Spearman, Kendall-tau, NDCG, Top-K overlap
        """
        summary = {}
        eps = self.config.epsilon

        # ─── Pairwise interaction statistics ────────────────────────────────
        pair_interactions = []
        pair_ratios = []
        pair_by_overlap = {'low': [], 'medium': [], 'high': []}

        for p in pair_results:
            m = p.get('interaction_metrics', {})
            i_ij = m.get('interaction_residual', 0.0)
            r_pair = m.get('additivity_ratio', 1.0)
            pair_interactions.append(i_ij)
            pair_ratios.append(r_pair)

            overlap_bin = p.get('pair_info', {}).get('overlap_bin', 'low')
            if overlap_bin in pair_by_overlap:
                pair_by_overlap[overlap_bin].append(i_ij)

        if pair_interactions:
            arr = np.array(pair_interactions)
            summary['I_ij_mean'] = float(np.mean(arr))
            summary['I_ij_std'] = float(np.std(arr))
            summary['I_ij_median'] = float(np.median(arr))
            summary['mean_abs_I_ij'] = float(np.mean(np.abs(arr)))
            summary['pct_sub_additive'] = float(np.mean(arr < -eps) * 100)
            summary['pct_super_additive'] = float(np.mean(arr > eps) * 100)
            summary['R_pair_mean'] = float(np.mean(pair_ratios))
            summary['R_pair_median'] = float(np.median(pair_ratios))

            for bin_name, vals in pair_by_overlap.items():
                if vals:
                    v = np.array(vals)
                    summary[f'I_ij_mean_{bin_name}'] = float(np.mean(v))
                    summary[f'pct_sub_additive_{bin_name}'] = float(np.mean(v < -eps) * 100)
        else:
            summary['I_ij_mean'] = 0.0
            summary['mean_abs_I_ij'] = 0.0

        # ─── Conditional utility statistics ─────────────────────────────────
        flips = 0
        total_conds = 0
        deviations = []          # D_Q values
        ratios = []              # R_Q values
        flips_by_overlap = {'low': [0, 0], 'medium': [0, 0], 'high': [0, 0]}

        unconditional_utils = {}  # candidate_idx -> U*(i|∅)
        conditional_utils_by_size = {}  # context_size -> {candidate_idx: U*(i|S)}

        for cand in conditional_results:
            c_idx = cand['candidate_idx']
            base_dq = None

            for m in cand['measurements']:
                if m['context_size'] == 0:
                    base_dq = m['metrics'].get('delta_q_conditional', 0.0)
                    unconditional_utils[c_idx] = base_dq
                    break

            if base_dq is None:
                continue

            for m in cand['measurements']:
                if m['context_size'] > 0:
                    cond_dq = m['metrics'].get('delta_q_conditional', 0.0)

                    # D_Q(i,S) = ΔQ(i|S) - ΔQ(i|∅)
                    d_q = cond_dq - base_dq
                    deviations.append(d_q)

                    # R_Q(i,S) = ΔQ(i|S) / (ΔQ(i|∅) + ε)
                    r_q = cond_dq / (abs(base_dq) + eps)
                    ratios.append(r_q)

                    # Sign flip
                    is_flip = (base_dq > 0 and cond_dq < 0) or (base_dq < 0 and cond_dq > 0)
                    if is_flip:
                        flips += 1
                    total_conds += 1

                    # Sign flip by overlap bin
                    ob = m.get('overlap_bin', 'low')
                    if ob in flips_by_overlap:
                        flips_by_overlap[ob][1] += 1
                        if is_flip:
                            flips_by_overlap[ob][0] += 1

                    # Collect for rank instability
                    sz = m['context_size']
                    if sz not in conditional_utils_by_size:
                        conditional_utils_by_size[sz] = {}
                    conditional_utils_by_size[sz][c_idx] = cond_dq

        summary['R_flip'] = flips / max(1, total_conds)
        summary['n_sign_flips'] = flips
        summary['n_conditional_evals'] = total_conds

        for ob, (n_flip, n_total) in flips_by_overlap.items():
            summary[f'R_flip_{ob}'] = n_flip / max(1, n_total)

        if deviations:
            d_arr = np.array(deviations)
            summary['D_Q_mean'] = float(np.mean(d_arr))
            summary['D_Q_std'] = float(np.std(d_arr))
            summary['D_Q_abs_mean'] = float(np.mean(np.abs(d_arr)))

        if ratios:
            r_arr = np.array(ratios)
            summary['R_Q_mean'] = float(np.mean(r_arr))
            summary['R_Q_std'] = float(np.std(r_arr))
            summary['R_Q_median'] = float(np.median(r_arr))

        # ─── Rank instability ───────────────────────────────────────────────
        rank_stats = self.compute_rank_instability(unconditional_utils, conditional_utils_by_size)
        summary['rank_instability'] = rank_stats

        return summary

    def compute_rank_instability(
        self,
        unconditional_utilities: Dict[int, float],
        conditional_utilities_by_size: Dict[int, Dict[int, float]],
    ) -> Dict[str, Any]:
        """Compare candidate rankings under different context conditions.

        For each context size |S|, computes:
            - Spearman ρ between rank(U*(i|∅)) and rank(U*(i|S))
            - Kendall τ
            - NDCG@K for K in {5, 10, 20}
            - Top-K overlap for K in {5, 10}
        """
        results = {}

        for ctx_sz, cond_map in conditional_utilities_by_size.items():
            # Find common candidates
            common = sorted(set(unconditional_utilities.keys()) & set(cond_map.keys()))
            if len(common) < 3:
                continue

            u_uncond = np.array([unconditional_utilities[c] for c in common])
            u_cond = np.array([cond_map[c] for c in common])

            rho, p_rho = spearmanr(u_uncond, u_cond)
            tau, p_tau = kendalltau(u_uncond, u_cond)

            # Top-K overlap
            rank_uncond = np.argsort(-u_uncond)
            rank_cond = np.argsort(-u_cond)

            top_k_results = {}
            for k in [5, 10, 20]:
                if k > len(common):
                    continue
                top_uncond = set(rank_uncond[:k])
                top_cond = set(rank_cond[:k])
                overlap = len(top_uncond & top_cond) / k
                top_k_results[f'top_{k}_overlap'] = overlap

            # NDCG@K
            ndcg_results = {}
            for k in [5, 10, 20]:
                if k > len(common):
                    continue
                ndcg_results[f'ndcg_{k}'] = self.compute_ndcg_at_k(u_uncond, u_cond, k)

            results[f'|S|={ctx_sz}'] = {
                'spearman_rho': float(rho) if not np.isnan(rho) else 0.0,
                'spearman_p': float(p_rho) if not np.isnan(p_rho) else 1.0,
                'kendall_tau': float(tau) if not np.isnan(tau) else 0.0,
                'kendall_p': float(p_tau) if not np.isnan(p_tau) else 1.0,
                'n_common': len(common),
                **top_k_results,
                **ndcg_results,
            }

        return results

    def compute_ndcg_at_k(self, true_scores: np.ndarray, pred_scores: np.ndarray, k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain at K.

        Uses true_scores as relevance and pred_scores for ranking.

        Args:
            true_scores: Ground-truth relevance scores.
            pred_scores: Predicted scores used for ranking.
            k: Cutoff position.

        Returns:
            NDCG@K in [0, 1].
        """
        # Rank by predicted scores (descending)
        pred_order = np.argsort(-pred_scores)[:k]
        ideal_order = np.argsort(-true_scores)[:k]

        # DCG
        discounts = 1.0 / np.log2(np.arange(1, k + 1) + 1)
        dcg = np.sum(true_scores[pred_order] * discounts)
        idcg = np.sum(true_scores[ideal_order] * discounts)

        if idcg < 1e-10:
            return 1.0
        return float(np.clip(dcg / idcg, 0.0, 1.0))

    def assess_go_no_go(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Determine GO/NO-GO based on audit results (Section 11).

        GO criteria (any one sufficient):
            1. R_flip > 0.05 (5% of candidates flip utility sign)
            2. R_Q varies strongly with overlap (|R_Q_mean - 1.0| > 0.15)
            3. |I_ij| has significant magnitude (mean_abs_I_ij > 1e-4)
            4. Rank instability: Spearman ρ < 0.85 for any |S|

        NO-GO:
            - U(i|S) ≈ U(i|∅) for most candidates
            - rank(i|S) ≈ rank(i|∅)
        """
        is_go = False
        reasons = []

        # Criterion 1: Sign-flip rate
        r_flip = summary.get('R_flip', 0.0)
        if r_flip > 0.05:
            is_go = True
            reasons.append(f"Sign-flip rate R_flip={r_flip:.3f} > 0.05 threshold")

        # Criterion 2: R_Q deviation from 1.0
        r_q_mean = summary.get('R_Q_mean', 1.0)
        if abs(r_q_mean - 1.0) > 0.15:
            is_go = True
            reasons.append(f"Utility ratio R_Q={r_q_mean:.3f} deviates significantly from 1.0")

        # Criterion 3: Pairwise interaction magnitude
        mean_abs_I = summary.get('mean_abs_I_ij', 0.0)
        if mean_abs_I > 1e-4:
            is_go = True
            reasons.append(f"Significant pairwise interaction |I_ij|={mean_abs_I:.6f}")

        # Criterion 4: Rank instability
        rank_info = summary.get('rank_instability', {})
        for ctx_key, ctx_stats in rank_info.items():
            rho = ctx_stats.get('spearman_rho', 1.0)
            if rho < 0.85:
                is_go = True
                reasons.append(f"Rank instability at {ctx_key}: ρ={rho:.3f} < 0.85")

        # Overlap-specific sign flips
        r_flip_high = summary.get('R_flip_high', 0.0)
        if r_flip_high > 0.10:
            is_go = True
            reasons.append(f"High-overlap sign-flip rate R_flip_high={r_flip_high:.3f} > 0.10")

        if not is_go:
            reasons.append("Interaction effects are weak across all criteria.")
            reasons.append("Pointwise utility appears sufficient for subset selection.")
            reasons.append("Recommend: Do NOT build interaction-aware model.")

        return {
            'decision': 'GO' if is_go else 'NO-GO',
            'is_go': is_go,
            'reasons': reasons,
            'key_metrics': {
                'R_flip': r_flip,
                'R_flip_high_overlap': r_flip_high,
                'R_Q_mean': r_q_mean,
                'mean_abs_I_ij': mean_abs_I,
            }
        }
