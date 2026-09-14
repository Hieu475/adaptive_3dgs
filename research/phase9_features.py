"""Phase 9A: Scale-Invariant Feature Representations.

Transforms raw canonical 11-dimensional Gaussian state vectors into
scale-relative representations to mitigate cross-scene distribution shift.

Variants:
    A0: "raw"
        Identity mapping: phi_A0(x) = x in R^11.
    A1: "geometry_relative"
        Scale-relative geometric factors:
            - depth_error    <- e_depth / (median(e_depth_V) + eps)
            - position_drift <- d_pos   / (median(d_pos_V) + eps)
            - projected_area <- A_proj  / (median(A_proj_V) + eps)
        Remaining 8 features unchanged.
    A2: "geometry_optimization_relative"
        A1 transformations PLUS optimization and attribution relatives:
            - gradient_norm      <- g     / (median(g_V) + eps)
            - influence_mass     <- I     / (median(I_V) + eps)
            - uncertainty_var    <- u_var / (median(u_var_V) + eps)
            - residual_drift_ema <- r_ema / (median(r_ema_V) + eps)
        Kept strictly unchanged:
            - rgb_error
            - visibility_count
            - update_frequency
            - age

Invariants & Constraints:
    - Never uses oracle utility U*, delta_q, delta_t, or test labels.
    - Operates frame-by-frame or within current candidate population V.
    - Strictly deterministic, finite (no NaN, no Inf, handles zero median).
    - Preserves input dimension R^11.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch

from research.phase9_protocol import (
    CANONICAL_FEATURE_SCHEMA,
    FEATURE_VARIANTS,
    VARIANT_ALIASES,
    EPS,
)

# Feature index mapping for quick lookup
FEAT_IDX: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_FEATURE_SCHEMA)}

# Indices transformed in A1
A1_TRANSFORMED_INDICES: List[int] = [
    FEAT_IDX["depth_error"],      # 1
    FEAT_IDX["position_drift"],   # 5
    FEAT_IDX["projected_area"],   # 8
]

# Indices transformed in A2 (A1 + additional)
A2_ADDITIONAL_INDICES: List[int] = [
    FEAT_IDX["gradient_norm"],       # 2
    FEAT_IDX["influence_mass"],      # 4
    FEAT_IDX["residual_drift_ema"],  # 6
    FEAT_IDX["uncertainty_var"],     # 7
]

A2_TRANSFORMED_INDICES: List[int] = sorted(A1_TRANSFORMED_INDICES + A2_ADDITIONAL_INDICES)

# Unchanged indices in A2
A2_UNCHANGED_INDICES: List[int] = [
    FEAT_IDX["rgb_error"],          # 0: absolute photometric error
    FEAT_IDX["visibility_count"],   # 3: dimensionless count
    FEAT_IDX["update_frequency"],   # 9: dimensionless ratio in [0, 1]
    FEAT_IDX["age"],                # 10: dimensionless frame count
]


def compute_robust_population_scale(
    values: np.ndarray,
    eps: float = EPS,
) -> float:
    """Compute robust central scale factor from a 1D population array.
    
    Guarantees:
        - Scale is strictly positive (>= eps)
        - If median is 0 (e.g. >50% zeros like position drift), uses median of
          non-zero values if available, or mean, or falls back to 1.0.
        - Never returns NaN or Inf.
    """
    if len(values) == 0:
        return 1.0
    
    clean_vals = values[np.isfinite(values)]
    if len(clean_vals) == 0:
        return 1.0

    med = float(np.median(clean_vals))
    if med > eps:
        return med + eps

    # Median is near zero or zero (common in sparse features like drift)
    pos = clean_vals[clean_vals > eps]
    if len(pos) > 0:
        pos_med = float(np.median(pos))
        return pos_med + eps

    # Population is all zeros or non-positive
    mean_val = float(np.mean(np.abs(clean_vals)))
    if mean_val > eps:
        return mean_val + eps

    return 1.0


def transform_features_array(
    X: np.ndarray,
    variant: str = "geometry_relative",
    eps: float = EPS,
) -> np.ndarray:
    """Transform feature matrix X [N, 11] using current population statistics.

    Args:
        X: [N, 11] raw canonical feature matrix
        variant: "raw" (A0), "geometry_relative" (A1), or "geometry_optimization_relative" (A2)
        eps: safety epsilon

    Returns:
        Z: [N, 11] transformed feature matrix (copy, original untouched)
    """
    variant_norm = VARIANT_ALIASES.get(variant, variant)
    if variant_norm not in FEATURE_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Expected one of {FEATURE_VARIANTS}")

    if X.ndim != 2 or X.shape[1] != len(CANONICAL_FEATURE_SCHEMA):
        raise ValueError(f"Expected X with shape [N, {len(CANONICAL_FEATURE_SCHEMA)}], got {X.shape}")

    if variant_norm == "raw":
        return X.copy().astype(np.float32)

    Z = X.copy().astype(np.float32)
    indices_to_transform = (
        A1_TRANSFORMED_INDICES if variant_norm == "geometry_relative"
        else A2_TRANSFORMED_INDICES
    )

    for idx in indices_to_transform:
        feat_col = X[:, idx]
        scale = compute_robust_population_scale(feat_col, eps=eps)
        Z[:, idx] = (feat_col / scale).astype(np.float32)

    # Sanity check: no NaN or Inf
    if not np.all(np.isfinite(Z)):
        raise FloatingPointError(f"NaN or Inf encountered after transform for variant '{variant}'")

    return Z


def transform_grouped_features(
    X: np.ndarray,
    group_keys: List[Any],
    variant: str = "geometry_relative",
    eps: float = EPS,
) -> np.ndarray:
    """Transform features grouped by frame/context (e.g. (scene, frame)).
    
    Computes population statistics separately for each unique group key,
    ensuring frame-local contextual normalization without cross-frame leakage.
    """
    variant_norm = VARIANT_ALIASES.get(variant, variant)
    if variant_norm == "raw" or len(group_keys) != len(X):
        return transform_features_array(X, variant=variant, eps=eps)

    unique_groups = sorted(list(set(group_keys)), key=lambda k: str(k))
    Z = np.zeros_like(X, dtype=np.float32)

    for g in unique_groups:
        mask = [k == g for k in group_keys]
        idx_arr = np.where(mask)[0]
        if len(idx_arr) > 0:
            Z[idx_arr] = transform_features_array(X[idx_arr], variant=variant_norm, eps=eps)

    return Z


def transform_candidates(
    candidates: List[Dict[str, Any]],
    variant: str = "geometry_relative",
    eps: float = EPS,
) -> List[Dict[str, Any]]:
    """Transform candidate feature dicts in-place or return modified copies.
    
    Each candidate dictionary contains a 'features' dict with the 11 keys.
    Returns new list of candidate dicts with transformed features.
    """
    variant_norm = VARIANT_ALIASES.get(variant, variant)
    if variant_norm == "raw":
        return candidates

    import copy
    output_candidates = copy.deepcopy(candidates)

    if len(output_candidates) == 0:
        return output_candidates

    # Extract [N, 11] matrix
    X = np.zeros((len(output_candidates), len(CANONICAL_FEATURE_SCHEMA)), dtype=np.float32)
    for i, cand in enumerate(output_candidates):
        feats = cand.get("features", {})
        for j, name in enumerate(CANONICAL_FEATURE_SCHEMA):
            X[i, j] = float(feats.get(name, 0.0))

    Z = transform_features_array(X, variant=variant_norm, eps=eps)

    for i, cand in enumerate(output_candidates):
        feats = cand.get("features", {})
        for j, name in enumerate(CANONICAL_FEATURE_SCHEMA):
            feats[name] = float(Z[i, j])
        cand["features"] = feats
        cand["representation_variant"] = variant_norm

    return output_candidates


class ScaleInvariantFeatureTransformer:
    """Stateful transformer wrapper adhering to scikit-learn / FeatureNormalizer interface."""

    def __init__(self, variant: str = "geometry_relative", eps: float = EPS):
        self.variant = VARIANT_ALIASES.get(variant, variant)
        self.eps = eps

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        group_keys: Optional[List[Any]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            device = X.device
            dtype = X.dtype
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X

        if group_keys is not None:
            Z_np = transform_grouped_features(X_np, group_keys, variant=self.variant, eps=self.eps)
        else:
            Z_np = transform_features_array(X_np, variant=self.variant, eps=self.eps)

        if is_torch:
            return torch.tensor(Z_np, device=device, dtype=dtype)
        return Z_np
