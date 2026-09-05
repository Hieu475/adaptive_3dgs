"""Phase 6: Conditional Oracle for Context-Aware Marginal Utility.

This module extends OracleUtilityExperiment to measure conditional marginal utility:

    U*(i | S) = (Q(S ∪ {i}) - Q(S)) / (T(S ∪ {i}) - T(S) + ε)

where S is a context set of already-optimized Gaussians.

Key Differences from Phase 3/4 Oracle:
    Phase 3/4: U*(i) = ΔQ(i) / ΔT(i)        (pointwise, S = ∅)
    Phase 6:   U*(i|S) = ΔQ(i|S) / ΔT(i|S)  (conditional, S ≠ ∅ possible)

When S = ∅, this degenerates to Phase 3/4 pointwise marginal utility.

Measurement Protocol:
    1. snapshot_state()
    2. If |S| > 0: optimize_gaussian_group(S) → measure Q(S), T(S)
    3. snapshot after S optimization (intermediate state)
    4. optimize_gaussian_group(S ∪ {i}) from ORIGINAL state → measure Q(S∪{i}), T(S∪{i})
    5. restore_state() to original
    6. Compute: ΔQ(i|S) = Q(S∪{i}) - Q(S), ΔT(i|S) = T(S∪{i}) - T(S)

IMPORTANT: Steps 2 and 4 both start from the SAME original snapshot.
    - Step 2 optimizes ONLY S from original state → Q(S)
    - Step 4 optimizes S∪{i} from original state → Q(S∪{i})
    This ensures ΔQ(i|S) captures the true conditional marginal gain.

Context Set Sampling Strategies:
    - "empty": S = ∅ (recovers Phase 4 marginal utility)
    - "spatial_knn": K nearest 3D neighbors
    - "overlap_top": Highest projected-overlap neighbors
    - "random": Uniformly random from candidate pool

Context Size Distribution (for dataset generation):
    30% S = ∅, 25% |S| = 1, 25% |S| = 4, 20% |S| = 8

Invariants:
    - All measurements use snapshot/restore isolation (non-destructive).
    - Quality Q(·) uses global metrics (full-frame PSNR), not local.
    - Cost T(·) uses wall-clock measured_trial_cost_ms.
    - Oracle reference name: "Oracle Conditional Marginal Reference" (NOT "optimal").
"""
import time
import math
import numpy as np
import torch
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .oracle_utility import OracleUtilityExperiment
from .phase6_context import (
    build_full_context,
    build_full_context_batch,
    ContextConfig,
    PHASE6_FEATURE_NAMES,
    _get_knn_indices,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConditionalOracleConfig:
    """Configuration for conditional oracle measurements.

    Attributes:
        n_opt_steps: Gradient steps per optimization trial (locked at 5 by protocol).
        context_sizes: List of context set sizes to evaluate.
        context_size_weights: Sampling weights for each context size
            (must sum to 1.0 and align with context_sizes).
        context_types: List of context sampling strategies.
        k_neighbors: KNN neighborhood size for spatial context.
        epsilon: Small constant to prevent division by zero in utility.
        contribution_threshold: Min weight for pixel influence.
        overlap_threshold: IoU threshold for "high overlap" neighbors.
    """
    n_opt_steps: int = 5
    context_sizes: List[int] = field(default_factory=lambda: [0, 1, 4, 8])
    context_size_weights: List[float] = field(
        default_factory=lambda: [0.30, 0.25, 0.25, 0.20]
    )
    context_types: List[str] = field(default_factory=lambda: [
        "empty", "spatial_knn", "overlap_top", "random"
    ])
    k_neighbors: int = 8
    epsilon: float = 1e-6
    contribution_threshold: float = 0.01
    overlap_threshold: float = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Oracle Experiment
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalOracleExperiment:
    """Extends OracleUtilityExperiment for conditional marginal utility measurement.

    Core method: measure_conditional_utility(candidate_idx, context_indices, ...)
    returns the full set of conditional measurements:
        Q(S), Q(S∪{i}), ΔQ(i|S), ΔT(i|S), U*(i|S)
    """

    def __init__(
        self,
        pipeline,
        config: Optional[ConditionalOracleConfig] = None,
        oracle_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize conditional oracle with underlying pipeline.

        Args:
            pipeline: OnlineReconstructionPipeline instance (initialized, with model loaded).
            config: ConditionalOracleConfig for Phase 6 parameters.
            oracle_config: Optional kwargs forwarded to OracleUtilityExperiment.
        """
        self.config = config or ConditionalOracleConfig()
        self.pipeline = pipeline

        # Initialize the base oracle engine
        oracle_kwargs = oracle_config or {}
        oracle_kwargs.setdefault('n_opt_steps', self.config.n_opt_steps)
        self.oracle_engine = OracleUtilityExperiment(pipeline, **oracle_kwargs)

        # Context config for feature extraction
        self.context_config = ContextConfig(
            k_neighbors=self.config.k_neighbors,
            overlap_threshold=self.config.overlap_threshold,
            contribution_threshold=self.config.contribution_threshold,
        )

    # ─── Core Measurement ────────────────────────────────────────────────

    def measure_quality_at_state(
        self,
        rgb_gt: torch.Tensor,
        depth_gt: torch.Tensor,
    ) -> Dict[str, float]:
        """Measure current global rendering quality without any optimization.

        Uses full-frame mask for global PSNR/SSIM/DepthL1.

        Args:
            rgb_gt: Ground-truth RGB image (H, W, 3).
            depth_gt: Ground-truth depth map (H, W).

        Returns:
            Dict with keys: "psnr", "ssim", "depth_l1", "loss", "mse".
        """
        H, W = rgb_gt.shape[:2]
        full_mask = torch.ones(H, W, dtype=torch.bool, device=rgb_gt.device)

        with torch.no_grad():
            render_out = self.oracle_engine._render(H, W)
            rendered_color = render_out['color']
            rendered_depth = render_out['depth']

            psnr = self.oracle_engine._compute_local_psnr(rendered_color, rgb_gt, full_mask)
            ssim = self.oracle_engine._compute_local_ssim(rendered_color, rgb_gt, full_mask)
            depth_l1 = self.oracle_engine._compute_local_depth_l1(
                rendered_depth, depth_gt, full_mask
            )
            loss = self.oracle_engine._compute_local_loss(
                rendered_color, rendered_depth, rgb_gt, depth_gt, full_mask
            )

            mse_val = ((rendered_color[full_mask] - rgb_gt[full_mask]) ** 2).mean().item()

        return {
            "psnr": psnr,
            "ssim": ssim,
            "depth_l1": depth_l1,
            "loss": loss,
            "mse": mse_val,
        }

    def measure_conditional_utility(
        self,
        candidate_idx: int,
        context_indices: List[int],
        rgb_gt: torch.Tensor,
        depth_gt: torch.Tensor,
        contrib_indices: torch.Tensor,
        contrib_weights: torch.Tensor,
    ) -> Dict[str, Any]:
        """Core measurement: conditional marginal utility U*(i | S).

        Protocol:
            1. snapshot_state() (original state)
            2. Measure Q_baseline (quality at original state)
            3. If |S| > 0:
                a. optimize_gaussian_group(S) from original state
                b. Measure Q(S), T(S)
                c. restore_state() to original
            4. optimize_gaussian_group(S ∪ {i}) from original state
            5. Measure Q(S∪{i}), T(S∪{i})
            6. restore_state() to original
            7. Compute ΔQ(i|S), ΔT(i|S), U*(i|S)

        Args:
            candidate_idx: Active tensor index of candidate Gaussian i.
            context_indices: Active tensor indices of context set S.
            rgb_gt: Ground-truth RGB image (H, W, 3).
            depth_gt: Ground-truth depth map (H, W).
            contrib_indices: (H, W, K_top) per-pixel Gaussian contributor IDs.
            contrib_weights: (H, W, K_top) per-pixel blend weights.

        Returns:
            Dict containing all measurement quantities (see docstring below).
        """
        H, W = rgb_gt.shape[:2]
        model = self.pipeline.gaussian_model
        N = model.num_gaussians
        eps = self.config.epsilon

        # Validate indices
        valid_context = [i for i in context_indices if 0 <= i < N and i != candidate_idx]
        if candidate_idx < 0 or candidate_idx >= N:
            return self._empty_measurement(candidate_idx, context_indices)

        # ─── 1. Snapshot original state ───
        snapshot = self.oracle_engine.snapshot_state()

        try:
            # ─── 2. Measure baseline quality Q_0 ───
            q_baseline = self.measure_quality_at_state(rgb_gt, depth_gt)

            # ─── 3. Measure Q(S), T(S) if |S| > 0 ───
            if len(valid_context) > 0:
                # Get influence mask for context set
                influence_mask_s = self.oracle_engine._get_influence_mask(
                    valid_context, contrib_indices, contrib_weights
                )

                result_s = self.oracle_engine.optimize_gaussian_group(
                    indices=valid_context,
                    n_steps=self.config.n_opt_steps,
                    rgb=rgb_gt,
                    depth=depth_gt,
                    influence_mask=influence_mask_s,
                )

                q_s = {
                    "psnr": result_s['psnr_global_after'],
                    "ssim": result_s['ssim_global_after'],
                    "depth_l1": result_s['depth_l1_global_after'],
                    "loss": result_s['loss_global_after'],
                }
                t_s_ms = result_s['measured_trial_cost_ms']

                # Restore to original state before next measurement
                self.oracle_engine.restore_state(snapshot)
            else:
                q_s = q_baseline.copy()
                t_s_ms = 0.0

            # ─── 4. Measure Q(S ∪ {i}), T(S ∪ {i}) from original state ───
            group_s_plus_i = valid_context + [candidate_idx]
            influence_mask_si = self.oracle_engine._get_influence_mask(
                group_s_plus_i, contrib_indices, contrib_weights
            )

            result_si = self.oracle_engine.optimize_gaussian_group(
                indices=group_s_plus_i,
                n_steps=self.config.n_opt_steps,
                rgb=rgb_gt,
                depth=depth_gt,
                influence_mask=influence_mask_si,
            )

            q_si = {
                "psnr": result_si['psnr_global_after'],
                "ssim": result_si['ssim_global_after'],
                "depth_l1": result_si['depth_l1_global_after'],
                "loss": result_si['loss_global_after'],
            }
            t_si_ms = result_si['measured_trial_cost_ms']

        finally:
            # ─── 5. ALWAYS restore to original state ───
            self.oracle_engine.restore_state(snapshot)

        # ─── 6. Compute conditional deltas ───
        # ΔQ uses the normalized delta_quality formulation from oracle_utility.py
        # Q(S) improvement over baseline
        dq_s = self._compute_delta_quality(q_baseline, q_s)
        # Q(S∪{i}) improvement over baseline
        dq_si = self._compute_delta_quality(q_baseline, q_si)

        # Conditional marginal gain
        delta_q_conditional = dq_si - dq_s
        delta_t_conditional_ms = t_si_ms - t_s_ms

        # Conditional marginal utility
        utility_conditional = delta_q_conditional / (
            max(abs(delta_t_conditional_ms), eps)
        )

        return {
            # Identification
            "candidate_idx": candidate_idx,
            "context_indices": list(valid_context),
            "context_size": len(valid_context),

            # Baseline (original state)
            "q_baseline_psnr": q_baseline["psnr"],
            "q_baseline_loss": q_baseline["loss"],

            # Q(S) — quality after optimizing only context set
            "q_s_psnr": q_s["psnr"],
            "q_s_ssim": q_s["ssim"],
            "q_s_depth_l1": q_s["depth_l1"],
            "q_s_loss": q_s["loss"],
            "t_s_ms": t_s_ms,
            "delta_q_s": dq_s,

            # Q(S ∪ {i}) — quality after optimizing context + candidate
            "q_si_psnr": q_si["psnr"],
            "q_si_ssim": q_si["ssim"],
            "q_si_depth_l1": q_si["depth_l1"],
            "q_si_loss": q_si["loss"],
            "t_si_ms": t_si_ms,
            "delta_q_si": dq_si,

            # Conditional marginal quantities (PRIMARY OUTPUTS)
            "delta_q_conditional": delta_q_conditional,
            "delta_t_conditional_ms": delta_t_conditional_ms,
            "utility_conditional": utility_conditional,
        }

    def _compute_delta_quality(
        self,
        q_before: Dict[str, float],
        q_after: Dict[str, float],
    ) -> float:
        """Compute normalized delta quality (same formula as oracle_utility.py).

        ΔQ = w_rgb * (ΔPSNR / max(1, PSNR_before)) + w_depth * (ΔDepth / max(1e-3, Depth_before))

        Args:
            q_before: Quality dict with "psnr", "depth_l1" keys.
            q_after: Quality dict with "psnr", "depth_l1" keys.

        Returns:
            Scalar normalized quality improvement (positive = better).
        """
        w_rgb = self.oracle_engine.w_rgb
        w_depth = self.oracle_engine.w_depth

        delta_psnr = q_after["psnr"] - q_before["psnr"]
        delta_depth = q_before["depth_l1"] - q_after["depth_l1"]  # positive = depth error reduced

        norm_psnr = delta_psnr / max(1.0, q_before["psnr"])
        norm_depth = delta_depth / max(1e-3, q_before["depth_l1"])

        return w_rgb * norm_psnr + w_depth * norm_depth

    def _empty_measurement(
        self,
        candidate_idx: int,
        context_indices: List[int],
    ) -> Dict[str, Any]:
        """Return zero-filled measurement for invalid candidates."""
        return {
            "candidate_idx": candidate_idx,
            "context_indices": list(context_indices),
            "context_size": len(context_indices),
            "q_baseline_psnr": 0.0,
            "q_baseline_loss": 0.0,
            "q_s_psnr": 0.0,
            "q_s_ssim": 0.0,
            "q_s_depth_l1": 0.0,
            "q_s_loss": 0.0,
            "t_s_ms": 0.0,
            "delta_q_s": 0.0,
            "q_si_psnr": 0.0,
            "q_si_ssim": 0.0,
            "q_si_depth_l1": 0.0,
            "q_si_loss": 0.0,
            "t_si_ms": 0.0,
            "delta_q_si": 0.0,
            "delta_q_conditional": 0.0,
            "delta_t_conditional_ms": 0.0,
            "utility_conditional": 0.0,
        }

    # ─── Context Set Sampling ────────────────────────────────────────────

    def sample_context_set(
        self,
        candidate_idx: int,
        context_type: str,
        context_size: int,
        candidate_pool: List[int],
        positions: torch.Tensor,
        contrib_indices: Optional[torch.Tensor] = None,
        contrib_weights: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> List[int]:
        """Sample a context set S for candidate i.

        Args:
            candidate_idx: Active tensor index of the candidate.
            context_type: One of "empty", "spatial_knn", "overlap_top", "random".
            context_size: Target size |S|.
            candidate_pool: List of all candidate indices (to sample from for "random").
            positions: (N, 3) tensor of all Gaussian positions.
            contrib_indices: Optional (H,W,K_top) for overlap-based context.
            contrib_weights: Optional (H,W,K_top) for overlap-based context.
            seed: Random seed for reproducibility.

        Returns:
            List of active tensor indices forming the context set S.
        """
        if context_size <= 0 or context_type == "empty":
            return []

        N = positions.shape[0]
        if candidate_idx < 0 or candidate_idx >= N:
            return []

        rng = np.random.default_rng(seed)

        if context_type == "spatial_knn":
            neighbors = _get_knn_indices(positions, candidate_idx, k=context_size)
            return neighbors[:context_size]

        elif context_type == "overlap_top":
            if (contrib_indices is not None and contrib_weights is not None):
                return self._sample_by_overlap(
                    candidate_idx, context_size, positions,
                    contrib_indices, contrib_weights
                )
            else:
                # Fallback to spatial KNN
                neighbors = _get_knn_indices(positions, candidate_idx, k=context_size)
                return neighbors[:context_size]

        elif context_type == "random":
            pool = [i for i in candidate_pool if i != candidate_idx and 0 <= i < N]
            if len(pool) == 0:
                return []
            n_sample = min(context_size, len(pool))
            chosen = rng.choice(pool, size=n_sample, replace=False)
            return chosen.tolist()

        else:
            raise ValueError(f"Unknown context_type: {context_type}")

    def _sample_by_overlap(
        self,
        candidate_idx: int,
        context_size: int,
        positions: torch.Tensor,
        contrib_indices: torch.Tensor,
        contrib_weights: torch.Tensor,
    ) -> List[int]:
        """Sample context set by highest projected pixel overlap with candidate.

        Args:
            candidate_idx: Active tensor index of the candidate.
            context_size: Number of context Gaussians to select.
            positions: (N, 3) tensor of all positions.
            contrib_indices: (H, W, K_top) per-pixel Gaussian IDs.
            contrib_weights: (H, W, K_top) per-pixel blend weights.

        Returns:
            List of indices sorted by descending overlap with candidate.
        """
        from .phase6_context import _get_pixel_mask

        threshold = self.config.contribution_threshold

        # Get candidate pixel mask
        cand_mask = _get_pixel_mask(candidate_idx, contrib_indices, contrib_weights, threshold)
        if cand_mask.sum().item() == 0:
            # Fallback to spatial KNN
            return _get_knn_indices(positions, candidate_idx, k=context_size)

        # Get spatial neighbors as overlap candidates
        n_candidates = min(context_size * 4, positions.shape[0] - 1)
        neighbors = _get_knn_indices(positions, candidate_idx, k=n_candidates)

        if not neighbors:
            return []

        # Compute IoU with each neighbor
        overlaps = []
        for n_idx in neighbors:
            n_mask = _get_pixel_mask(n_idx, contrib_indices, contrib_weights, threshold)
            intersection = (cand_mask & n_mask).sum().item()
            union = (cand_mask | n_mask).sum().item()
            iou = intersection / max(union, 1)
            overlaps.append((n_idx, iou))

        # Sort by descending IoU
        overlaps.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in overlaps[:context_size]]

    # ─── Dataset Generation ──────────────────────────────────────────────

    def generate_conditional_dataset(
        self,
        candidate_pool: List[int],
        rgb_gt: torch.Tensor,
        depth_gt: torch.Tensor,
        contrib_indices: torch.Tensor,
        contrib_weights: torch.Tensor,
        all_features: np.ndarray,
        scene_name: str = "scene",
        frame_idx: int = 0,
        split: str = "test",
        seed: int = 42,
        n_samples: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate conditional oracle dataset for a single frame.

        For each candidate, samples multiple context sets (varying type and size)
        and measures the conditional marginal utility for each.

        Distribution of context sizes (default):
            30% S = ∅  (recovers Phase 4 marginal utility)
            25% |S| = 1
            25% |S| = 4
            20% |S| = 8

        Args:
            candidate_pool: List of candidate active tensor indices.
            rgb_gt: Ground-truth RGB (H, W, 3).
            depth_gt: Ground-truth depth (H, W).
            contrib_indices: (H, W, K_top) per-pixel Gaussian IDs.
            contrib_weights: (H, W, K_top) per-pixel blend weights.
            all_features: (N, 11) canonical features for ALL Gaussians.
            scene_name: Scene identifier.
            frame_idx: Frame number.
            split: Dataset split ("train", "validation", "cross_scene_test").
            seed: Random seed.
            n_samples: Max total samples to generate (None = all).
            max_candidates: Max candidates to evaluate (None = all).

        Returns:
            List of sample dicts, each containing measurements + context features.
        """
        rng = np.random.default_rng(seed)
        model = self.pipeline.gaussian_model
        positions = model.positions

        # Subsample candidates if needed
        pool = [i for i in candidate_pool if 0 <= i < model.num_gaussians]
        if max_candidates is not None and len(pool) > max_candidates:
            pool = rng.choice(pool, size=max_candidates, replace=False).tolist()

        # Build context size/type schedule
        schedule = self._build_context_schedule(len(pool), rng)

        results: List[Dict[str, Any]] = []
        sample_idx = 0

        for cand_idx in pool:
            # Get persistent ID
            p_id = cand_idx
            if hasattr(model, 'persistent_ids') and cand_idx < len(model.persistent_ids):
                p_id = int(model.persistent_ids[cand_idx].item())

            for ctx_type, ctx_size in schedule:
                if n_samples is not None and sample_idx >= n_samples:
                    return results

                # Sample context set
                context_ids = self.sample_context_set(
                    candidate_idx=cand_idx,
                    context_type=ctx_type,
                    context_size=ctx_size,
                    candidate_pool=pool,
                    positions=positions,
                    contrib_indices=contrib_indices,
                    contrib_weights=contrib_weights,
                    seed=seed + sample_idx,
                )

                # Measure conditional utility
                measurement = self.measure_conditional_utility(
                    candidate_idx=cand_idx,
                    context_indices=context_ids,
                    rgb_gt=rgb_gt,
                    depth_gt=depth_gt,
                    contrib_indices=contrib_indices,
                    contrib_weights=contrib_weights,
                )

                # Build context features
                ctx_feats = build_full_context(
                    positions=positions,
                    candidate_idx=cand_idx,
                    all_features=all_features,
                    selected_indices=context_ids,
                    contrib_indices=contrib_indices,
                    contrib_weights=contrib_weights,
                    config=self.context_config,
                )

                # Assemble sample record
                sample = {
                    # Identification
                    "scene": scene_name,
                    "frame": frame_idx,
                    "split": split,
                    "seed": seed,
                    "candidate_id": cand_idx,
                    "candidate_persistent_id": p_id,
                    "context_ids": context_ids,
                    "context_type": ctx_type,
                    "context_size": len(context_ids),

                    # Features (serializable)
                    "self_features": ctx_feats["self_features"].tolist(),
                    "neighbor_features": ctx_feats["neighbor_features"],
                    "overlap_features": ctx_feats["overlap_features"],
                    "selected_features": ctx_feats["selected_features"],
                    "full_feature_vector": ctx_feats["full_vector"].tolist(),

                    # Oracle measurements
                    "q_baseline_psnr": measurement["q_baseline_psnr"],
                    "q_s_psnr": measurement["q_s_psnr"],
                    "q_si_psnr": measurement["q_si_psnr"],
                    "delta_q_s": measurement["delta_q_s"],
                    "delta_q_si": measurement["delta_q_si"],
                    "delta_q_conditional": measurement["delta_q_conditional"],
                    "t_s_ms": measurement["t_s_ms"],
                    "t_si_ms": measurement["t_si_ms"],
                    "delta_t_conditional_ms": measurement["delta_t_conditional_ms"],
                    "utility_conditional": measurement["utility_conditional"],
                }

                results.append(sample)
                sample_idx += 1

        return results

    def _build_context_schedule(
        self,
        n_candidates: int,
        rng: np.random.Generator,
    ) -> List[Tuple[str, int]]:
        """Build the context (type, size) schedule for one candidate.

        Returns a list of (context_type, context_size) tuples representing
        the different context conditions each candidate will be evaluated under.

        Default schedule per candidate:
            1× empty (S = ∅)
            1× spatial_knn, |S| = 1
            1× spatial_knn, |S| = 4
            1× overlap_top, |S| = 4
            1× random, |S| = 8

        This gives ~5 samples per candidate, covering all context types and sizes.
        """
        schedule = [
            ("empty", 0),
            ("spatial_knn", 1),
            ("spatial_knn", 4),
            ("overlap_top", 4),
            ("random", 8),
        ]
        return schedule

    # ─── Interaction Analysis ────────────────────────────────────────────

    def measure_pairwise_interaction(
        self,
        idx_i: int,
        idx_j: int,
        rgb_gt: torch.Tensor,
        depth_gt: torch.Tensor,
        contrib_indices: torch.Tensor,
        contrib_weights: torch.Tensor,
    ) -> Dict[str, Any]:
        """Measure the interaction residual I(i,j) = ΔQ({i,j}) - ΔQ({i}) - ΔQ({j}).

        This quantifies the non-additivity between two Gaussians:
            I(i,j) > 0: super-additive (synergy)
            I(i,j) < 0: sub-additive (competition/redundancy)
            I(i,j) ≈ 0: independent

        Args:
            idx_i: Active tensor index of Gaussian i.
            idx_j: Active tensor index of Gaussian j.
            rgb_gt: Ground-truth RGB (H, W, 3).
            depth_gt: Ground-truth depth (H, W).
            contrib_indices: (H, W, K_top).
            contrib_weights: (H, W, K_top).

        Returns:
            Dict with individual deltas, joint delta, interaction residual, and overlap IoU.
        """
        model = self.pipeline.gaussian_model
        N = model.num_gaussians
        snapshot = self.oracle_engine.snapshot_state()

        try:
            # Baseline
            q_base = self.measure_quality_at_state(rgb_gt, depth_gt)

            # ΔQ({i}) — optimize only i
            mask_i = self.oracle_engine._get_influence_mask([idx_i], contrib_indices, contrib_weights)
            result_i = self.oracle_engine.optimize_gaussian_group(
                [idx_i], self.config.n_opt_steps, rgb_gt, depth_gt, mask_i
            )
            dq_i = self._compute_delta_quality(
                q_base,
                {"psnr": result_i["psnr_global_after"], "depth_l1": result_i["depth_l1_global_after"]}
            )
            self.oracle_engine.restore_state(snapshot)

            # ΔQ({j}) — optimize only j
            mask_j = self.oracle_engine._get_influence_mask([idx_j], contrib_indices, contrib_weights)
            result_j = self.oracle_engine.optimize_gaussian_group(
                [idx_j], self.config.n_opt_steps, rgb_gt, depth_gt, mask_j
            )
            dq_j = self._compute_delta_quality(
                q_base,
                {"psnr": result_j["psnr_global_after"], "depth_l1": result_j["depth_l1_global_after"]}
            )
            self.oracle_engine.restore_state(snapshot)

            # ΔQ({i,j}) — optimize both together
            mask_ij = self.oracle_engine._get_influence_mask(
                [idx_i, idx_j], contrib_indices, contrib_weights
            )
            result_ij = self.oracle_engine.optimize_gaussian_group(
                [idx_i, idx_j], self.config.n_opt_steps, rgb_gt, depth_gt, mask_ij
            )
            dq_ij = self._compute_delta_quality(
                q_base,
                {"psnr": result_ij["psnr_global_after"], "depth_l1": result_ij["depth_l1_global_after"]}
            )

        finally:
            self.oracle_engine.restore_state(snapshot)

        # Interaction residual
        interaction = dq_ij - dq_i - dq_j

        # Compute pixel overlap IoU
        from .phase6_context import _get_pixel_mask
        mask_pixels_i = _get_pixel_mask(
            idx_i, contrib_indices, contrib_weights, self.config.contribution_threshold
        )
        mask_pixels_j = _get_pixel_mask(
            idx_j, contrib_indices, contrib_weights, self.config.contribution_threshold
        )
        intersection = (mask_pixels_i & mask_pixels_j).sum().item()
        union = (mask_pixels_i | mask_pixels_j).sum().item()
        overlap_iou = intersection / max(union, 1)

        # 3D distance
        dist_3d = float(torch.dist(
            model.positions[idx_i], model.positions[idx_j]
        ).item())

        return {
            "idx_i": idx_i,
            "idx_j": idx_j,
            "delta_q_i": dq_i,
            "delta_q_j": dq_j,
            "delta_q_ij": dq_ij,
            "interaction_residual": interaction,
            "is_sub_additive": interaction < 0,
            "is_super_additive": interaction > 0,
            "overlap_iou": overlap_iou,
            "distance_3d": dist_3d,
            "additivity_ratio": dq_ij / (dq_i + dq_j + self.config.epsilon),
        }
