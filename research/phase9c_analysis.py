"""Phase 9C: Robustness and Adaptation Analysis Utilities.

Provides implementation and analysis engines for:
    1. 9C-1: Sensitivity grid of OnlineEMANormalizer across beta in {0.80, 0.90, 0.95}.
    2. 9C-2: Adaptation ablation normalizers:
             - B0: Standard train normalizer
             - B2: Online EMA normalizer (beta=0.90)
             - B2-static: Online normalizer with train initialization but test updates OFF.
    3. 9C-3: Temporal dynamics tracking (drift, scale drift, selection overlap, per-frame rho_t).
    4. 9C-4: Controlled covariate perturbation transforms (scale a, offset b).
"""
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
from scipy.stats import spearmanr

from research.phase9_protocol import CANONICAL_FEATURE_SCHEMA
from research.phase9b_protocol import EPS
from research.phase9b_normalization import (
    StandardNormalizer,
    OnlineEMANormalizer,
)
from research.phase9c_protocol import (
    BETA_VALUES,
    PERTURBATION_SCALES,
    PERTURBATION_OFFSETS,
)


class B2StaticNormalizer(OnlineEMANormalizer):
    """B2 Online Adaptive Normalizer with adaptation updates disabled (B2-static ablation).
    
    Initialized from training statistics (mu_0 = mu_train, sigma_0 = sigma_train),
    but never updates its running moments during test-time frame processing.
    
    Serves as the diagnostic control for Gate 9C-3:
        If test-time adaptation is correctly implemented, B2_static MUST produce
        predictions bitwise / numerically identical to B0 (StandardNormalizer),
        while active B2 (with updates ON) diverges due to empirical covariate shifts.
    """

    def __init__(self, eps: float = EPS):
        super().__init__(beta=1.0, eps=eps)
        self.normalizer_type = "B2StaticNormalizer"

    def update_and_transform_frame(
        self,
        X_frame: Union[np.ndarray, torch.Tensor],
        frame_id: Optional[int] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, float]]:
        """Process frame using frozen initial train statistics without online update."""
        if self.mu_current is None or self.sigma_current is None:
            raise RuntimeError("B2StaticNormalizer must be fit before calling update_and_transform_frame.")

        t0 = time.perf_counter()
        is_torch = isinstance(X_frame, torch.Tensor)
        if is_torch:
            X_np = X_frame.detach().cpu().numpy().astype(np.float32)
        else:
            X_np = np.asarray(X_frame, dtype=np.float32)

        # In B2-static, mu_current and sigma_current remain strictly frozen at mu_init, sigma_init
        if is_torch:
            device = X_frame.device
            mu_t = torch.tensor(self.mu_init, device=device, dtype=X_frame.dtype)
            sig_t = torch.tensor(self.sigma_init, device=device, dtype=X_frame.dtype)
            Z_norm = (X_frame - mu_t) / sig_t
        else:
            Z_norm = ((X_np - self.mu_init) / self.sigma_init).astype(np.float32)

        lat_ms = (time.perf_counter() - t0) * 1000.0

        step_metrics = {
            "frame_id": frame_id if frame_id is not None else len(self.history),
            "latency_ms": lat_ms,
            "d_norm": 0.0,
            "mu_l2_shift": 0.0,
            "sigma_l2_shift": 0.0,
            "mean_mu": float(np.mean(self.mu_init)),
            "mean_sigma": float(np.mean(self.sigma_init)),
        }
        self.history.append(step_metrics)
        return Z_norm, step_metrics


def apply_feature_perturbation(
    X: np.ndarray,
    scale: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Applies controlled synthetic scale and offset perturbations to raw features.
    
    Formula:
        X'_{i, j} = scale * X_{i, j} + offset
        
    Args:
        X: [N, 11] raw or representation-transformed feature matrix
        scale: multiplicative factor a in {0.8, 1.0, 1.2}
        offset: additive shift b in {-0.2, 0.0, +0.2}
        
    Returns:
        X_perturbed: [N, 11] perturbed feature matrix with guaranteed finite values
    """
    X_pert = scale * np.asarray(X, dtype=np.float32) + offset
    if not np.all(np.isfinite(X_pert)):
        raise FloatingPointError(f"NaN/Inf generated during feature perturbation (scale={scale}, offset={offset})")
    return X_pert.astype(np.float32)


def compute_per_frame_spearman(
    pred_u: np.ndarray,
    oracle_u: np.ndarray,
    frame_indices: List[int],
) -> float:
    """Computes Spearman rank correlation rho_t strictly within a single frame's candidate subset."""
    if len(frame_indices) < 3:
        return float("nan")
    p_frame = pred_u[frame_indices]
    o_frame = oracle_u[frame_indices]
    if np.std(p_frame) < 1e-7 or np.std(o_frame) < 1e-7:
        return 0.0
    res = spearmanr(p_frame, o_frame)
    return float(res.correlation) if np.isfinite(res.correlation) else 0.0


def verify_b0_b2_static_equivalence(
    b0_predictions: np.ndarray,
    b2_static_predictions: np.ndarray,
    tolerance: float = 1e-5,
) -> Tuple[bool, float]:
    """Verifies that B0 and B2-static produce identical utility predictions (Gate 9C-3).
    
    Returns:
        is_equivalent: True if max absolute difference <= tolerance
        max_diff: float maximum absolute difference across all candidates
    """
    diff = np.max(np.abs(b0_predictions - b2_static_predictions))
    is_equivalent = bool(diff <= tolerance)
    return is_equivalent, float(diff)


def assess_sensitivity_range(
    metrics_by_beta: Dict[float, float],
    relative_threshold: float = 0.15,
) -> Tuple[str, float]:
    """Evaluates whether performance across beta in {0.80, 0.90, 0.95} is stable or sensitive.
    
    Args:
        metrics_by_beta: mapping of beta -> metric (e.g. Spearman rho or OSE@20)
        relative_threshold: maximum relative dispersion (range / mean) for 'stable'
        
    Returns:
        assessment: 'stable' or 'sensitive'
        rel_range: relative range value
    """
    vals = list(metrics_by_beta.values())
    val_range = max(vals) - min(vals)
    val_mean = abs(float(np.mean(vals)))
    rel_range = val_range / max(val_mean, 1e-4)
    assessment = "stable" if rel_range <= relative_threshold else "sensitive"
    return assessment, float(rel_range)
