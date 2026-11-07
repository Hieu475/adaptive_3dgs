"""Evaluation metrics for RGB-D mapping audits.

Depth metrics distinguish missing map support from depth error where a map
prediction exists. RGB SSIM uses the spatial, Gaussian-window implementation
from torchmetrics; the old image-wide statistic remains available by name.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def _as_hwc_rgb(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"RGB image must have 3 dimensions, got {tuple(image.shape)}")
    if image.shape[-1] in (1, 3):
        return image
    if image.shape[0] in (1, 3):
        return image.permute(1, 2, 0)
    raise ValueError(f"Cannot infer RGB channel dimension from {tuple(image.shape)}")


def ssim_legacy_global(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> float:
    """Reproduce the former global-image-statistics SSIM-like score."""
    pred = _as_hwc_rgb(pred_rgb).float()
    target = _as_hwc_rgb(target_rgb).to(device=pred.device, dtype=pred.dtype)
    if pred.shape != target.shape:
        raise ValueError(f"RGB shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    c1, c2 = 0.01**2, 0.03**2
    mu1, mu2 = pred.mean((0, 1)), target.mean((0, 1))
    var1 = ((pred - mu1) ** 2).mean((0, 1))
    var2 = ((target - mu2) ** 2).mean((0, 1))
    covariance = ((pred - mu1) * (target - mu2)).mean((0, 1))
    score = ((2 * mu1 * mu2 + c1) * (2 * covariance + c2)) / (
        (mu1.square() + mu2.square() + c1) * (var1 + var2 + c2)
    )
    return float(score.mean().item())


def ssim_standard(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    data_range: float = 1.0,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute spatial-window SSIM with torchmetrics on RGB in [0, data_range].

    When ``valid_mask`` is supplied, the local SSIM map is averaged only at
    valid GT-depth pixels. Local windows still use neighboring RGB pixels.
    """
    pred = _as_hwc_rgb(pred_rgb).float()
    target = _as_hwc_rgb(target_rgb).to(device=pred.device, dtype=pred.dtype)
    if pred.shape != target.shape:
        raise ValueError(f"RGB shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    h, w, _ = pred.shape
    if min(h, w) < 3:
        raise ValueError("SSIM requires image dimensions of at least 3x3")
    pred_bchw = pred.permute(2, 0, 1).unsqueeze(0).contiguous()
    target_bchw = target.permute(2, 0, 1).unsqueeze(0).contiguous()
    kernel_size = min(11, h if h % 2 else h - 1, w if w % 2 else w - 1)
    score, score_map = structural_similarity_index_measure(
        pred_bchw,
        target_bchw,
        gaussian_kernel=True,
        sigma=1.5,
        kernel_size=kernel_size,
        data_range=data_range,
        return_full_image=True,
    )
    if valid_mask is None:
        return float(score.item())
    mask = valid_mask.to(device=score_map.device, dtype=torch.bool)
    if mask.shape != (h, w):
        raise ValueError(f"valid_mask must have shape {(h, w)}, got {tuple(mask.shape)}")
    if not mask.any():
        return float("nan")
    return float(score_map[0, :, mask].mean().item())


def depth_metric_bundle(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    *,
    valid_gt_mask: Optional[torch.Tensor] = None,
    pred_valid_mask: Optional[torch.Tensor] = None,
    near: Optional[float] = None,
    far: Optional[float] = None,
) -> Dict[str, float | int]:
    """Compute GT-valid and prediction-overlap depth metrics in input units.

    A finite positive GT sample is valid unless a protocol mask or range is
    supplied. Missing/invalid predictions are assigned zero for the
    GT-valid metric, so uncovered valid pixels are penalized rather than
    silently discarded. The overlap metric is diagnostic only.
    """
    pred = pred_depth.squeeze().float()
    gt = gt_depth.squeeze().to(device=pred.device, dtype=pred.dtype)
    if pred.shape != gt.shape:
        raise ValueError(f"Depth shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    gt_valid = torch.isfinite(gt) & (gt > 0)
    if valid_gt_mask is not None:
        mask = valid_gt_mask.to(device=gt.device, dtype=torch.bool).squeeze()
        if mask.shape != gt.shape:
            raise ValueError(f"valid_gt_mask must have shape {tuple(gt.shape)}")
        gt_valid &= mask
    if near is not None:
        gt_valid &= gt >= float(near)
    if far is not None:
        gt_valid &= gt <= float(far)
    pred_valid = torch.isfinite(pred) & (pred > 0)
    if pred_valid_mask is not None:
        pred_mask = pred_valid_mask.to(device=pred.device, dtype=torch.bool).squeeze()
        if pred_mask.shape != pred.shape:
            raise ValueError(f"pred_valid_mask must have shape {tuple(pred.shape)}")
        pred_valid &= pred_mask
    safe_pred = torch.where(pred_valid, pred, torch.zeros_like(pred))
    overlap = gt_valid & pred_valid
    gt_count = int(gt_valid.sum().item())
    overlap_count = int(overlap.sum().item())
    total_gt_pixels = int(gt.numel())
    if gt_count:
        gt_errors = (safe_pred[gt_valid] - gt[gt_valid]).abs()
        gt_l1 = float(gt_errors.mean().item())
        gt_rmse = float(torch.sqrt((safe_pred[gt_valid] - gt[gt_valid]).square().mean()).item())
        coverage = overlap_count / gt_count
    else:
        gt_l1 = float("nan")
        gt_rmse = float("nan")
        coverage = float("nan")
    if overlap_count:
        overlap_errors = (pred[overlap] - gt[overlap]).abs()
        overlap_l1 = float(overlap_errors.mean().item())
        overlap_rmse = float(torch.sqrt((pred[overlap] - gt[overlap]).square().mean()).item())
    else:
        overlap_l1 = float("nan")
        overlap_rmse = float("nan")
    return {
        "depth_l1_gt_valid": gt_l1,
        "depth_rmse_gt_valid": gt_rmse,
        "depth_l1_overlap": overlap_l1,
        "depth_rmse_overlap": overlap_rmse,
        "pred_depth_coverage": float(coverage),
        "valid_gt_pixel_count": gt_count,
        "overlap_pixel_count": overlap_count,
        "total_pixel_count": total_gt_pixels,
    }


def rgb_metric_bundle(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    *,
    valid_gt_mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
) -> Dict[str, float | int]:
    """Return masked PSNR, spatial SSIM, and explicitly named legacy score."""
    pred = _as_hwc_rgb(pred_rgb).float()
    target = _as_hwc_rgb(gt_rgb).to(device=pred.device, dtype=pred.dtype)
    if pred.shape != target.shape:
        raise ValueError(f"RGB shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    if valid_gt_mask is not None:
        mask = valid_gt_mask.to(device=pred.device, dtype=torch.bool).squeeze()
        if mask.shape != pred.shape[:2]:
            raise ValueError(f"valid_gt_mask must have shape {tuple(pred.shape[:2])}")
        finite &= mask
    count = int(finite.sum().item())
    if count:
        mse = (pred[finite] - target[finite]).square().mean()
        psnr_valid = float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item())
    else:
        psnr_valid = float("nan")
    return {
        "psnr_valid_gt": psnr_valid,
        "ssim_standard": ssim_standard(pred, target, data_range=data_range),
        "ssim_standard_valid_gt": ssim_standard(
            pred, target, data_range=data_range, valid_mask=finite),
        "ssim_legacy": ssim_legacy_global(pred, target),
        "valid_rgb_pixel_count": count,
    }
