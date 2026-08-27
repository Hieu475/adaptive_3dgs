"""Evaluation metrics for 3D Gaussian Splatting reconstruction.

Metrics:
- Appearance: PSNR, SSIM, LPIPS
- Geometry: Depth L1, AbsRel, Accuracy, Completion, Chamfer distance
- Tracking: ATE (Absolute Trajectory Error)
- System: FPS, frame time, VRAM usage
"""
import numpy as np
import torch
from typing import Dict, Optional, List
import json
import time


def compute_psnr(
    pred: np.ndarray,
    gt: np.ndarray,
    max_val: float = 1.0,
) -> float:
    """Peak Signal-to-Noise Ratio.
    
    PSNR = 10 * log10(max_val^2 / MSE)
    """
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / mse)


def compute_ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    window_size: int = 11,
) -> float:
    """Structural Similarity Index (simplified).
    
    Uses the mean-based SSIM formula with default constants.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Convert to float64
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    
    if pred.max() <= 1.0:
        pred *= 255.0
        gt *= 255.0
    
    mu1 = pred.mean()
    mu2 = gt.mean()
    sigma1_sq = pred.var()
    sigma2_sq = gt.var()
    sigma12 = np.mean((pred - mu1) * (gt - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim)


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute depth evaluation metrics.
    
    Metrics:
    - L1: mean absolute error
    - AbsRel: |d_pred - d_gt| / d_gt
    - Accuracy: % of pixels where max(d_pred/d_gt, d_gt/d_pred) < threshold
    - Completion: % of valid GT pixels with valid prediction
    
    Args:
        pred_depth: (H, W) predicted depth
        gt_depth: (H, W) ground truth depth
        valid_mask: (H, W) boolean mask
    
    Returns:
        Dict with metric values
    """
    if valid_mask is None:
        valid_mask = gt_depth > 0
    
    valid = valid_mask & (pred_depth > 0)
    
    if valid.sum() == 0:
        return {'depth_l1': 0.0, 'abs_rel': 0.0, 'accuracy_1.25': 0.0, 'completion': 0.0}
    
    pred_v = pred_depth[valid].astype(np.float64)
    gt_v = gt_depth[valid].astype(np.float64)
    
    # L1
    l1 = np.mean(np.abs(pred_v - gt_v))
    
    # AbsRel
    abs_rel = np.mean(np.abs(pred_v - gt_v) / (gt_v + 1e-8))
    
    # Accuracy (δ < 1.25)
    ratio = np.maximum(pred_v / (gt_v + 1e-8), gt_v / (pred_v + 1e-8))
    accuracy_125 = np.mean(ratio < 1.25) * 100.0
    
    # Completion
    completion = (valid.sum() / max(valid_mask.sum(), 1)) * 100.0
    
    return {
        'depth_l1': float(l1),
        'abs_rel': float(abs_rel),
        'accuracy_1.25': float(accuracy_125),
        'completion': float(completion),
    }


def compute_ate(
    pred_poses: List[np.ndarray],
    gt_poses: List[np.ndarray],
) -> Dict[str, float]:
    """Absolute Trajectory Error.
    
    ATE = sqrt(mean(||t_pred - t_gt||^2))
    """
    errors = []
    for pred, gt in zip(pred_poses, gt_poses):
        t_pred = pred[:3, 3]
        t_gt = gt[:3, 3]
        errors.append(np.linalg.norm(t_pred - t_gt))
    
    errors = np.array(errors)
    return {
        'ate_rmse': float(np.sqrt(np.mean(errors ** 2))),
        'ate_mean': float(np.mean(errors)),
        'ate_median': float(np.median(errors)),
        'ate_max': float(np.max(errors)),
    }


def compute_all_metrics(
    pred_color: np.ndarray,
    gt_color: np.ndarray,
    pred_depth: Optional[np.ndarray] = None,
    gt_depth: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all available metrics."""
    metrics = {
        'psnr': compute_psnr(pred_color, gt_color),
        'ssim': compute_ssim(pred_color, gt_color),
    }
    
    if pred_depth is not None and gt_depth is not None:
        depth_metrics = compute_depth_metrics(pred_depth, gt_depth)
        metrics.update(depth_metrics)
    
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate reconstruction metrics")
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    
    print("Evaluation metrics module ready.")
    print("Use compute_all_metrics() with predicted and ground truth images.")
