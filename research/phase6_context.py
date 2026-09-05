"""Phase 6: Context Representation for Conditional Utility Estimation.

This module builds contextual feature representations for each candidate Gaussian,
capturing spatial neighborhood, co-visibility overlap, and selected-set information.

Phase 6 Thesis:
    Replace pointwise utility U_hat_i = f(s_i) with conditional utility:
        U_hat_i = f(s_i, N_i, G_t, S_t)
    where:
        s_i     = 11-dim canonical self features (Phase 4, frozen)
        N_i     = neighborhood aggregate statistics (KNN in 3D)
        G_t     = overlap / co-visibility features
        S_t     = already-selected set aggregate features

Feature Vector Layout (32 dimensions):
    [0:11]   Self features (canonical Phase 4)
    [11:19]  Neighborhood features (8-dim)
    [19:24]  Overlap features (5-dim)
    [24:32]  Selected-set features (8-dim)

Invariants:
    - This module ONLY builds feature representations. No training code.
    - Self features are the 11 canonical features from utility_features.py.
    - Neighborhood uses KDTree / torch.cdist on model.positions (3D Euclidean).
    - Overlap uses tile-level attribution masks from render_with_attribution.
    - Selected-set features depend on S_t and must be recomputed when S changes.
    - All computations use pre-intervention state only (no post-optimization leakage).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch

from .utility_features import CANONICAL_FEATURE_NAMES, extract_feature_vector


# ─────────────────────────────────────────────────────────────────────────────
# Feature Name Constants (deterministic ordering)
# ─────────────────────────────────────────────────────────────────────────────

NEIGHBOR_FEATURE_NAMES: List[str] = [
    "neighbor_mean_rgb_error",
    "neighbor_mean_depth_error",
    "neighbor_mean_gradient_norm",
    "neighbor_mean_influence_mass",
    "neighbor_mean_uncertainty_var",
    "neighbor_mean_projected_area",
    "neighbor_std_rgb_error",
    "neighbor_count",
]

OVERLAP_FEATURE_NAMES: List[str] = [
    "mean_overlap",
    "max_overlap",
    "high_overlap_count",
    "weighted_overlap",
    "overlap_area_fraction",
]

SELECTED_FEATURE_NAMES: List[str] = [
    "selected_count",
    "selected_mean_rgb_error",
    "selected_mean_depth_error",
    "selected_mean_influence",
    "selected_spatial_density",
    "candidate_selected_overlap",
    "candidate_selected_distance",
    "selected_budget_fraction",
]

PHASE6_FEATURE_NAMES: List[str] = (
    list(CANONICAL_FEATURE_NAMES)
    + NEIGHBOR_FEATURE_NAMES
    + OVERLAP_FEATURE_NAMES
    + SELECTED_FEATURE_NAMES
)

PHASE6_FEATURE_DIM: int = len(PHASE6_FEATURE_NAMES)  # 32

# Feature group slicing
SELF_SLICE = slice(0, 11)
NEIGHBOR_SLICE = slice(11, 19)
OVERLAP_SLICE = slice(19, 24)
SELECTED_SLICE = slice(24, 32)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextConfig:
    """Configuration for Phase 6 context feature extraction.

    Attributes:
        k_neighbors: Number of spatial KNN neighbors for neighborhood context.
        overlap_threshold: IoU threshold to count as "high overlap" neighbor.
        contribution_threshold: Minimum attribution weight for pixel influence.
        use_projected_overlap: If True, compute overlap via projected tile attribution.
                               If False, use 3D distance-based proxy.
    """
    k_neighbors: int = 8
    overlap_threshold: float = 0.1
    contribution_threshold: float = 0.01
    use_projected_overlap: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Neighborhood Context (KNN in 3D space)
# ─────────────────────────────────────────────────────────────────────────────

def build_neighbor_context(
    positions: torch.Tensor,
    candidate_idx: int,
    all_features: np.ndarray,
    k: int = 8,
) -> Dict[str, float]:
    """Build neighborhood aggregate statistics for candidate i.

    Uses spatial KNN on 3D positions (Euclidean distance) to find the k nearest
    neighbors, then aggregates their canonical features.

    Args:
        positions: (N, 3) tensor of all Gaussian 3D positions.
        candidate_idx: Index of the candidate Gaussian in [0, N-1].
        all_features: (N, 11) array of canonical features for ALL Gaussians.
        k: Number of nearest neighbors (excluding self).

    Returns:
        Dict with keys matching NEIGHBOR_FEATURE_NAMES (8 values).
    """
    N = positions.shape[0]
    if N <= 1 or candidate_idx < 0 or candidate_idx >= N:
        return _empty_neighbor_context()

    k_actual = min(k, N - 1)
    if k_actual <= 0:
        return _empty_neighbor_context()

    # Compute distances from candidate to all Gaussians
    cand_pos = positions[candidate_idx].unsqueeze(0)  # (1, 3)
    dists = torch.cdist(cand_pos, positions).squeeze(0)  # (N,)

    # Get k+1 nearest (includes self at distance 0), then exclude self
    topk_dists, topk_idx = torch.topk(dists, k=k_actual + 1, largest=False)

    # Filter out self (distance ~ 0)
    non_self_mask = topk_idx != candidate_idx
    neighbor_idx = topk_idx[non_self_mask][:k_actual]

    n_neighbors = len(neighbor_idx)
    if n_neighbors == 0:
        return _empty_neighbor_context()

    # Gather neighbor features
    neighbor_idx_np = neighbor_idx.cpu().numpy()
    neighbor_feats = all_features[neighbor_idx_np]  # (n_neighbors, 11)

    # Feature indices in CANONICAL_FEATURE_NAMES:
    # 0: rgb_error, 1: depth_error, 2: gradient_norm,
    # 4: influence_mass, 7: uncertainty_var, 8: projected_area
    idx_rgb_err = 0
    idx_depth_err = 1
    idx_grad_norm = 2
    idx_influence = 4
    idx_uncertainty = 7
    idx_proj_area = 8

    return {
        "neighbor_mean_rgb_error": float(np.mean(neighbor_feats[:, idx_rgb_err])),
        "neighbor_mean_depth_error": float(np.mean(neighbor_feats[:, idx_depth_err])),
        "neighbor_mean_gradient_norm": float(np.mean(neighbor_feats[:, idx_grad_norm])),
        "neighbor_mean_influence_mass": float(np.mean(neighbor_feats[:, idx_influence])),
        "neighbor_mean_uncertainty_var": float(np.mean(neighbor_feats[:, idx_uncertainty])),
        "neighbor_mean_projected_area": float(np.mean(neighbor_feats[:, idx_proj_area])),
        "neighbor_std_rgb_error": float(np.std(neighbor_feats[:, idx_rgb_err])),
        "neighbor_count": float(n_neighbors),
    }


def _empty_neighbor_context() -> Dict[str, float]:
    """Return zero-filled neighbor context for degenerate cases."""
    return {name: 0.0 for name in NEIGHBOR_FEATURE_NAMES}


# ─────────────────────────────────────────────────────────────────────────────
# Batch Neighborhood Context (vectorized for efficiency)
# ─────────────────────────────────────────────────────────────────────────────

def build_neighbor_context_batch(
    positions: torch.Tensor,
    candidate_indices: List[int],
    all_features: np.ndarray,
    k: int = 8,
) -> Dict[int, Dict[str, float]]:
    """Vectorized neighbor context for multiple candidates.

    Computes all KNN distances in a single torch.cdist call, then aggregates
    features for each candidate. Much faster than calling build_neighbor_context
    in a loop for large candidate pools.

    Args:
        positions: (N, 3) tensor of all Gaussian 3D positions.
        candidate_indices: List of candidate indices in [0, N-1].
        all_features: (N, 11) array of canonical features for ALL Gaussians.
        k: Number of nearest neighbors (excluding self).

    Returns:
        Dict mapping candidate_idx -> neighbor context dict.
    """
    N = positions.shape[0]
    result: Dict[int, Dict[str, float]] = {}

    if N <= 1 or len(candidate_indices) == 0:
        for idx in candidate_indices:
            result[idx] = _empty_neighbor_context()
        return result

    k_actual = min(k, N - 1)
    if k_actual <= 0:
        for idx in candidate_indices:
            result[idx] = _empty_neighbor_context()
        return result

    # Batch distance computation
    valid_indices = [i for i in candidate_indices if 0 <= i < N]
    if not valid_indices:
        for idx in candidate_indices:
            result[idx] = _empty_neighbor_context()
        return result

    cand_t = torch.tensor(valid_indices, dtype=torch.long, device=positions.device)
    cand_pos = positions[cand_t]  # (M, 3)
    dists = torch.cdist(cand_pos, positions)  # (M, N)

    # Get k+1 nearest for each candidate
    topk_dists, topk_idx = torch.topk(dists, k=k_actual + 1, largest=False, dim=-1)

    # Feature extraction indices
    idx_rgb_err = 0
    idx_depth_err = 1
    idx_grad_norm = 2
    idx_influence = 4
    idx_uncertainty = 7
    idx_proj_area = 8

    topk_idx_np = topk_idx.cpu().numpy()

    for ci, cand_idx in enumerate(valid_indices):
        # Filter out self from neighbors
        neighbor_raw = topk_idx_np[ci]
        non_self = neighbor_raw[neighbor_raw != cand_idx][:k_actual]

        n_neighbors = len(non_self)
        if n_neighbors == 0:
            result[cand_idx] = _empty_neighbor_context()
            continue

        nf = all_features[non_self]  # (n_neighbors, 11)
        result[cand_idx] = {
            "neighbor_mean_rgb_error": float(np.mean(nf[:, idx_rgb_err])),
            "neighbor_mean_depth_error": float(np.mean(nf[:, idx_depth_err])),
            "neighbor_mean_gradient_norm": float(np.mean(nf[:, idx_grad_norm])),
            "neighbor_mean_influence_mass": float(np.mean(nf[:, idx_influence])),
            "neighbor_mean_uncertainty_var": float(np.mean(nf[:, idx_uncertainty])),
            "neighbor_mean_projected_area": float(np.mean(nf[:, idx_proj_area])),
            "neighbor_std_rgb_error": float(np.std(nf[:, idx_rgb_err])),
            "neighbor_count": float(n_neighbors),
        }

    # Handle invalid indices
    for idx in candidate_indices:
        if idx not in result:
            result[idx] = _empty_neighbor_context()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Overlap Context (co-visibility via tile-level attribution)
# ─────────────────────────────────────────────────────────────────────────────

def build_overlap_context(
    candidate_idx: int,
    neighbor_indices: List[int],
    contrib_indices: torch.Tensor,
    contrib_weights: torch.Tensor,
    contribution_threshold: float = 0.01,
    overlap_threshold: float = 0.1,
) -> Dict[str, float]:
    """Build co-visibility overlap statistics between candidate and neighbors.

    Uses per-pixel attribution maps (contrib_indices, contrib_weights) from
    render_with_attribution to compute pixel-level overlap IoU between the
    candidate Gaussian and each of its spatial neighbors.

    Args:
        candidate_idx: Active tensor index of the candidate.
        neighbor_indices: List of active tensor indices of neighbors.
        contrib_indices: (H, W, K_top) int tensor of per-pixel Gaussian IDs.
        contrib_weights: (H, W, K_top) float tensor of per-pixel blend weights.
        contribution_threshold: Minimum weight to consider a pixel "influenced".
        overlap_threshold: IoU threshold to count a neighbor as "high overlap".

    Returns:
        Dict with keys matching OVERLAP_FEATURE_NAMES (5 values).
    """
    if len(neighbor_indices) == 0:
        return _empty_overlap_context()

    # Build candidate pixel mask: pixels where candidate has significant contribution
    cand_mask = _get_pixel_mask(candidate_idx, contrib_indices, contrib_weights,
                                contribution_threshold)
    cand_pixel_count = int(cand_mask.sum().item())

    if cand_pixel_count == 0:
        return _empty_overlap_context()

    ious = []
    for n_idx in neighbor_indices:
        n_mask = _get_pixel_mask(n_idx, contrib_indices, contrib_weights,
                                 contribution_threshold)
        intersection = (cand_mask & n_mask).sum().item()
        union = (cand_mask | n_mask).sum().item()
        iou = intersection / max(union, 1)
        ious.append(iou)

    ious_arr = np.array(ious, dtype=np.float32)

    # Compute weighted overlap (IoU weighted by neighbor's pixel count)
    weighted_sum = 0.0
    weight_total = 0.0
    for ni, n_idx in enumerate(neighbor_indices):
        n_mask = _get_pixel_mask(n_idx, contrib_indices, contrib_weights,
                                 contribution_threshold)
        n_count = float(n_mask.sum().item())
        weighted_sum += ious_arr[ni] * n_count
        weight_total += n_count
    weighted_overlap = weighted_sum / max(weight_total, 1.0)

    total_pixels = contrib_indices.shape[0] * contrib_indices.shape[1]

    return {
        "mean_overlap": float(np.mean(ious_arr)),
        "max_overlap": float(np.max(ious_arr)),
        "high_overlap_count": float(np.sum(ious_arr >= overlap_threshold)),
        "weighted_overlap": float(weighted_overlap),
        "overlap_area_fraction": float(cand_pixel_count / max(total_pixels, 1)),
    }


def _get_pixel_mask(
    gaussian_idx: int,
    contrib_indices: torch.Tensor,
    contrib_weights: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Get boolean mask of pixels significantly influenced by a Gaussian.

    Args:
        gaussian_idx: Active tensor index of the Gaussian.
        contrib_indices: (H, W, K_top) per-pixel contributor IDs.
        contrib_weights: (H, W, K_top) per-pixel blend weights.
        threshold: Minimum weight threshold.

    Returns:
        (H, W) boolean mask.
    """
    idx_match = (contrib_indices == gaussian_idx)
    weight_ok = (contrib_weights > threshold)
    significant = (idx_match & weight_ok).any(dim=-1)  # (H, W)
    return significant


def _empty_overlap_context() -> Dict[str, float]:
    """Return zero-filled overlap context for degenerate cases."""
    return {name: 0.0 for name in OVERLAP_FEATURE_NAMES}


def build_overlap_context_batch(
    candidate_indices: List[int],
    neighbor_map: Dict[int, List[int]],
    contrib_indices: torch.Tensor,
    contrib_weights: torch.Tensor,
    contribution_threshold: float = 0.01,
    overlap_threshold: float = 0.1,
) -> Dict[int, Dict[str, float]]:
    """Batch overlap context for multiple candidates.

    Pre-computes pixel masks for all relevant Gaussians once, then computes
    pairwise IoU efficiently via cached masks.

    Args:
        candidate_indices: List of candidate active indices.
        neighbor_map: Dict mapping candidate_idx -> list of neighbor indices.
        contrib_indices: (H, W, K_top) per-pixel contributor IDs.
        contrib_weights: (H, W, K_top) per-pixel blend weights.
        contribution_threshold: Minimum weight threshold.
        overlap_threshold: IoU threshold for "high overlap".

    Returns:
        Dict mapping candidate_idx -> overlap context dict.
    """
    # Collect all unique Gaussian indices we need masks for
    all_gaussians = set(candidate_indices)
    for neighbors in neighbor_map.values():
        all_gaussians.update(neighbors)

    # Pre-compute all pixel masks
    pixel_masks: Dict[int, torch.Tensor] = {}
    for g_idx in all_gaussians:
        pixel_masks[g_idx] = _get_pixel_mask(
            g_idx, contrib_indices, contrib_weights, contribution_threshold
        )

    total_pixels = contrib_indices.shape[0] * contrib_indices.shape[1]
    result: Dict[int, Dict[str, float]] = {}

    for cand_idx in candidate_indices:
        neighbors = neighbor_map.get(cand_idx, [])
        if not neighbors:
            result[cand_idx] = _empty_overlap_context()
            continue

        cand_mask = pixel_masks.get(cand_idx)
        if cand_mask is None or cand_mask.sum().item() == 0:
            result[cand_idx] = _empty_overlap_context()
            continue

        cand_pixel_count = int(cand_mask.sum().item())
        ious = []
        n_pixel_counts = []

        for n_idx in neighbors:
            n_mask = pixel_masks.get(n_idx)
            if n_mask is None:
                ious.append(0.0)
                n_pixel_counts.append(0.0)
                continue
            intersection = (cand_mask & n_mask).sum().item()
            union = (cand_mask | n_mask).sum().item()
            iou = intersection / max(union, 1)
            ious.append(iou)
            n_pixel_counts.append(float(n_mask.sum().item()))

        ious_arr = np.array(ious, dtype=np.float32)
        n_counts_arr = np.array(n_pixel_counts, dtype=np.float32)

        weight_total = float(n_counts_arr.sum())
        weighted_overlap = (
            float(np.dot(ious_arr, n_counts_arr)) / max(weight_total, 1.0)
        )

        result[cand_idx] = {
            "mean_overlap": float(np.mean(ious_arr)),
            "max_overlap": float(np.max(ious_arr)),
            "high_overlap_count": float(np.sum(ious_arr >= overlap_threshold)),
            "weighted_overlap": weighted_overlap,
            "overlap_area_fraction": float(cand_pixel_count / max(total_pixels, 1)),
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Selected-Set Context (features of already-selected Gaussians S_t)
# ─────────────────────────────────────────────────────────────────────────────

def build_selected_context(
    positions: torch.Tensor,
    candidate_idx: int,
    selected_indices: List[int],
    all_features: np.ndarray,
    total_budget: float = 1.0,
    current_cost: float = 0.0,
    contrib_indices: Optional[torch.Tensor] = None,
    contrib_weights: Optional[torch.Tensor] = None,
    contribution_threshold: float = 0.01,
) -> Dict[str, float]:
    """Build aggregate statistics of the already-selected set S_t.

    Captures what has already been selected, enabling the model to predict
    conditional marginal utility U*(i | S_t) rather than independent U*(i).

    Args:
        positions: (N, 3) tensor of all Gaussian 3D positions.
        candidate_idx: Active tensor index of the candidate.
        selected_indices: List of active tensor indices already in S_t.
        all_features: (N, 11) array of canonical features for ALL Gaussians.
        total_budget: Total optimization budget (for computing budget fraction).
        current_cost: Current accumulated cost of S_t.
        contrib_indices: Optional (H,W,K_top) for overlap computation.
        contrib_weights: Optional (H,W,K_top) for overlap computation.
        contribution_threshold: Minimum attribution weight.

    Returns:
        Dict with keys matching SELECTED_FEATURE_NAMES (8 values).
    """
    N = positions.shape[0]
    n_selected = len(selected_indices)

    if n_selected == 0 or candidate_idx < 0 or candidate_idx >= N:
        return _empty_selected_context(total_budget, current_cost)

    # Filter valid selected indices
    valid_selected = [i for i in selected_indices if 0 <= i < N]
    n_valid = len(valid_selected)

    if n_valid == 0:
        return _empty_selected_context(total_budget, current_cost)

    # Gather selected features
    selected_feats = all_features[valid_selected]  # (n_valid, 11)

    # Feature indices
    idx_rgb_err = 0
    idx_depth_err = 1
    idx_influence = 4

    # Mean features of selected set
    mean_rgb_err = float(np.mean(selected_feats[:, idx_rgb_err]))
    mean_depth_err = float(np.mean(selected_feats[:, idx_depth_err]))
    mean_influence = float(np.mean(selected_feats[:, idx_influence]))

    # Spatial density: mean pairwise distance among selected
    selected_t = torch.tensor(valid_selected, dtype=torch.long, device=positions.device)
    selected_pos = positions[selected_t]  # (n_valid, 3)

    if n_valid >= 2:
        pairwise_dists = torch.cdist(selected_pos, selected_pos)
        # Mean of upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(n_valid, n_valid, device=positions.device), diagonal=1).bool()
        spatial_density = float(pairwise_dists[mask].mean().item()) if mask.sum() > 0 else 0.0
    else:
        spatial_density = 0.0

    # Candidate-to-selected overlap (pixel-level if attribution available)
    cand_selected_overlap = 0.0
    if contrib_indices is not None and contrib_weights is not None:
        cand_mask = _get_pixel_mask(candidate_idx, contrib_indices, contrib_weights,
                                     contribution_threshold)
        if cand_mask.sum().item() > 0:
            # Union of all selected pixel masks
            selected_union = torch.zeros_like(cand_mask)
            for s_idx in valid_selected:
                s_mask = _get_pixel_mask(s_idx, contrib_indices, contrib_weights,
                                         contribution_threshold)
                selected_union |= s_mask

            intersection = (cand_mask & selected_union).sum().item()
            union = (cand_mask | selected_union).sum().item()
            cand_selected_overlap = float(intersection / max(union, 1))

    # Candidate-to-selected mean 3D distance
    cand_pos = positions[candidate_idx].unsqueeze(0)  # (1, 3)
    dists_to_selected = torch.cdist(cand_pos, selected_pos).squeeze(0)  # (n_valid,)
    cand_selected_distance = float(dists_to_selected.mean().item())

    # Budget fraction consumed
    budget_fraction = current_cost / max(total_budget, 1e-8)

    return {
        "selected_count": float(n_valid),
        "selected_mean_rgb_error": mean_rgb_err,
        "selected_mean_depth_error": mean_depth_err,
        "selected_mean_influence": mean_influence,
        "selected_spatial_density": spatial_density,
        "candidate_selected_overlap": cand_selected_overlap,
        "candidate_selected_distance": cand_selected_distance,
        "selected_budget_fraction": float(np.clip(budget_fraction, 0.0, 1.0)),
    }


def _empty_selected_context(
    total_budget: float = 1.0,
    current_cost: float = 0.0,
) -> Dict[str, float]:
    """Return zero-filled selected context for S_t = ∅."""
    return {
        "selected_count": 0.0,
        "selected_mean_rgb_error": 0.0,
        "selected_mean_depth_error": 0.0,
        "selected_mean_influence": 0.0,
        "selected_spatial_density": 0.0,
        "candidate_selected_overlap": 0.0,
        "candidate_selected_distance": 0.0,
        "selected_budget_fraction": float(np.clip(
            current_cost / max(total_budget, 1e-8), 0.0, 1.0
        )),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full Context Builder (combines all context types)
# ─────────────────────────────────────────────────────────────────────────────

def build_full_context(
    positions: torch.Tensor,
    candidate_idx: int,
    all_features: np.ndarray,
    selected_indices: Optional[List[int]] = None,
    contrib_indices: Optional[torch.Tensor] = None,
    contrib_weights: Optional[torch.Tensor] = None,
    total_budget: float = 1.0,
    current_cost: float = 0.0,
    config: Optional[ContextConfig] = None,
) -> Dict[str, Any]:
    """Build full Phase 6 context vector: self + neighbor + overlap + selected.

    This is the main entry point for Phase 6 feature extraction. Returns both
    individual context dicts and the concatenated 32-dim feature vector.

    Args:
        positions: (N, 3) tensor of all Gaussian 3D positions.
        candidate_idx: Active tensor index of the candidate.
        all_features: (N, 11) array of canonical features for ALL Gaussians.
        selected_indices: List of active indices already selected (S_t).
        contrib_indices: Optional (H,W,K_top) from render_with_attribution.
        contrib_weights: Optional (H,W,K_top) from render_with_attribution.
        total_budget: Total optimization budget.
        current_cost: Current accumulated cost of S_t.
        config: Context configuration.

    Returns:
        Dict containing:
            "self_features": np.ndarray (11,)
            "neighbor_features": Dict[str, float] (8 values)
            "overlap_features": Dict[str, float] (5 values)
            "selected_features": Dict[str, float] (8 values)
            "full_vector": np.ndarray (32,)
            "feature_names": List[str] (32 names)
    """
    if config is None:
        config = ContextConfig()

    if selected_indices is None:
        selected_indices = []

    N = positions.shape[0]

    # ─── Self features (11-dim) ───
    if 0 <= candidate_idx < N:
        self_feats = all_features[candidate_idx].copy()  # (11,)
    else:
        self_feats = np.zeros(11, dtype=np.float32)

    # ─── Neighbor context (8-dim) ───
    neighbor_ctx = build_neighbor_context(
        positions=positions,
        candidate_idx=candidate_idx,
        all_features=all_features,
        k=config.k_neighbors,
    )

    # ─── Overlap context (5-dim) ───
    if (config.use_projected_overlap
            and contrib_indices is not None
            and contrib_weights is not None):
        # Get neighbor indices for overlap computation
        neighbor_indices = _get_knn_indices(
            positions, candidate_idx, k=config.k_neighbors
        )
        overlap_ctx = build_overlap_context(
            candidate_idx=candidate_idx,
            neighbor_indices=neighbor_indices,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
            contribution_threshold=config.contribution_threshold,
            overlap_threshold=config.overlap_threshold,
        )
    else:
        overlap_ctx = _empty_overlap_context()

    # ─── Selected-set context (8-dim) ───
    selected_ctx = build_selected_context(
        positions=positions,
        candidate_idx=candidate_idx,
        selected_indices=selected_indices,
        all_features=all_features,
        total_budget=total_budget,
        current_cost=current_cost,
        contrib_indices=contrib_indices,
        contrib_weights=contrib_weights,
        contribution_threshold=config.contribution_threshold,
    )

    # ─── Concatenate into 32-dim vector ───
    neighbor_vec = np.array(
        [neighbor_ctx[name] for name in NEIGHBOR_FEATURE_NAMES],
        dtype=np.float32,
    )
    overlap_vec = np.array(
        [overlap_ctx[name] for name in OVERLAP_FEATURE_NAMES],
        dtype=np.float32,
    )
    selected_vec = np.array(
        [selected_ctx[name] for name in SELECTED_FEATURE_NAMES],
        dtype=np.float32,
    )

    full_vector = np.concatenate([self_feats, neighbor_vec, overlap_vec, selected_vec])
    assert full_vector.shape == (PHASE6_FEATURE_DIM,), (
        f"Expected {PHASE6_FEATURE_DIM}-dim vector, got {full_vector.shape}"
    )

    return {
        "self_features": self_feats,
        "neighbor_features": neighbor_ctx,
        "overlap_features": overlap_ctx,
        "selected_features": selected_ctx,
        "full_vector": full_vector,
        "feature_names": PHASE6_FEATURE_NAMES,
    }


def build_full_context_batch(
    positions: torch.Tensor,
    candidate_indices: List[int],
    all_features: np.ndarray,
    selected_indices: Optional[List[int]] = None,
    contrib_indices: Optional[torch.Tensor] = None,
    contrib_weights: Optional[torch.Tensor] = None,
    total_budget: float = 1.0,
    current_cost: float = 0.0,
    config: Optional[ContextConfig] = None,
) -> Dict[int, Dict[str, Any]]:
    """Batch full context extraction for multiple candidates.

    Vectorizes KNN and overlap computations for efficiency.

    Args:
        positions: (N, 3) tensor of all Gaussian 3D positions.
        candidate_indices: List of candidate active indices.
        all_features: (N, 11) array of canonical features for ALL Gaussians.
        selected_indices: List of active indices already selected (S_t).
        contrib_indices: Optional (H,W,K_top) from render_with_attribution.
        contrib_weights: Optional (H,W,K_top) from render_with_attribution.
        total_budget: Total optimization budget.
        current_cost: Current accumulated cost of S_t.
        config: Context configuration.

    Returns:
        Dict mapping candidate_idx -> full context dict.
    """
    if config is None:
        config = ContextConfig()

    if selected_indices is None:
        selected_indices = []

    N = positions.shape[0]

    # ─── Batch neighbor context ───
    neighbor_ctxs = build_neighbor_context_batch(
        positions=positions,
        candidate_indices=candidate_indices,
        all_features=all_features,
        k=config.k_neighbors,
    )

    # ─── Batch KNN indices for overlap ───
    neighbor_map: Dict[int, List[int]] = {}
    if (config.use_projected_overlap
            and contrib_indices is not None
            and contrib_weights is not None):
        neighbor_map = _get_knn_indices_batch(
            positions, candidate_indices, k=config.k_neighbors
        )

        # Batch overlap context
        overlap_ctxs = build_overlap_context_batch(
            candidate_indices=candidate_indices,
            neighbor_map=neighbor_map,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
            contribution_threshold=config.contribution_threshold,
            overlap_threshold=config.overlap_threshold,
        )
    else:
        overlap_ctxs = {idx: _empty_overlap_context() for idx in candidate_indices}

    # ─── Per-candidate assembly ───
    results: Dict[int, Dict[str, Any]] = {}

    for cand_idx in candidate_indices:
        # Self features
        if 0 <= cand_idx < N:
            self_feats = all_features[cand_idx].copy()
        else:
            self_feats = np.zeros(11, dtype=np.float32)

        neighbor_ctx = neighbor_ctxs.get(cand_idx, _empty_neighbor_context())
        overlap_ctx = overlap_ctxs.get(cand_idx, _empty_overlap_context())

        # Selected context (computed per-candidate due to spatial dependency)
        selected_ctx = build_selected_context(
            positions=positions,
            candidate_idx=cand_idx,
            selected_indices=selected_indices,
            all_features=all_features,
            total_budget=total_budget,
            current_cost=current_cost,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
            contribution_threshold=config.contribution_threshold,
        )

        # Concatenate
        neighbor_vec = np.array(
            [neighbor_ctx[name] for name in NEIGHBOR_FEATURE_NAMES],
            dtype=np.float32,
        )
        overlap_vec = np.array(
            [overlap_ctx[name] for name in OVERLAP_FEATURE_NAMES],
            dtype=np.float32,
        )
        selected_vec = np.array(
            [selected_ctx[name] for name in SELECTED_FEATURE_NAMES],
            dtype=np.float32,
        )
        full_vector = np.concatenate([self_feats, neighbor_vec, overlap_vec, selected_vec])

        results[cand_idx] = {
            "self_features": self_feats,
            "neighbor_features": neighbor_ctx,
            "overlap_features": overlap_ctx,
            "selected_features": selected_ctx,
            "full_vector": full_vector,
            "feature_names": PHASE6_FEATURE_NAMES,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_knn_indices(
    positions: torch.Tensor,
    candidate_idx: int,
    k: int = 8,
) -> List[int]:
    """Get k-nearest neighbor indices for a single candidate.

    Args:
        positions: (N, 3) tensor.
        candidate_idx: Index of the candidate.
        k: Number of neighbors.

    Returns:
        List of neighbor indices (excluding self).
    """
    N = positions.shape[0]
    k_actual = min(k, N - 1)
    if k_actual <= 0 or candidate_idx < 0 or candidate_idx >= N:
        return []

    cand_pos = positions[candidate_idx].unsqueeze(0)
    dists = torch.cdist(cand_pos, positions).squeeze(0)
    _, topk_idx = torch.topk(dists, k=k_actual + 1, largest=False)
    topk_list = topk_idx.cpu().tolist()
    return [i for i in topk_list if i != candidate_idx][:k_actual]


def _get_knn_indices_batch(
    positions: torch.Tensor,
    candidate_indices: List[int],
    k: int = 8,
) -> Dict[int, List[int]]:
    """Get KNN indices for multiple candidates in a single torch.cdist call.

    Args:
        positions: (N, 3) tensor.
        candidate_indices: List of candidate indices.
        k: Number of neighbors.

    Returns:
        Dict mapping candidate_idx -> list of neighbor indices.
    """
    N = positions.shape[0]
    k_actual = min(k, N - 1)
    result: Dict[int, List[int]] = {}

    if k_actual <= 0 or len(candidate_indices) == 0:
        return {idx: [] for idx in candidate_indices}

    valid = [i for i in candidate_indices if 0 <= i < N]
    if not valid:
        return {idx: [] for idx in candidate_indices}

    cand_t = torch.tensor(valid, dtype=torch.long, device=positions.device)
    cand_pos = positions[cand_t]
    dists = torch.cdist(cand_pos, positions)
    _, topk_idx = torch.topk(dists, k=k_actual + 1, largest=False, dim=-1)
    topk_np = topk_idx.cpu().numpy()

    for ci, cand_idx in enumerate(valid):
        neighbors = [int(i) for i in topk_np[ci] if i != cand_idx][:k_actual]
        result[cand_idx] = neighbors

    for idx in candidate_indices:
        if idx not in result:
            result[idx] = []

    return result
