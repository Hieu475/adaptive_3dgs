import pytest
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from research.mechanism.metrics import (
    depth_metric_bundle,
    rgb_metric_bundle,
    ssim_legacy_global,
    ssim_standard,
)


def test_spatial_ssim_matches_torchmetrics_reference():
    torch.manual_seed(4)
    pred = torch.rand(40, 48, 3)
    target = torch.rand(40, 48, 3)
    expected = structural_similarity_index_measure(
        pred.permute(2, 0, 1).unsqueeze(0),
        target.permute(2, 0, 1).unsqueeze(0),
        data_range=1.0,
        gaussian_kernel=True,
        sigma=1.5,
        kernel_size=11,
    )
    assert ssim_standard(pred, target) == pytest.approx(float(expected), abs=1e-7)


def test_spatial_ssim_identity_brightness_shift_translation_and_noise():
    torch.manual_seed(2)
    base = torch.zeros(32, 32, 3)
    base[8:24, 8:24] = 0.8
    identical = ssim_standard(base, base)
    brighter = ssim_standard((base + 0.1).clamp(0, 1), base)
    shifted = ssim_standard(torch.roll(base, shifts=3, dims=1), base)
    noisy = ssim_standard((base + 0.15 * torch.randn_like(base)).clamp(0, 1), base)
    assert identical == pytest.approx(1.0, abs=1e-7)
    assert brighter < identical
    assert shifted < identical
    assert noisy < identical
    assert ssim_legacy_global(base, base) == pytest.approx(1.0, abs=1e-7)


def test_depth_gt_valid_penalizes_missing_prediction_and_reports_overlap():
    gt = torch.tensor([[1.0, 2.0], [0.0, 3.0]])
    pred = torch.tensor([[1.0, 0.0], [4.0, 2.0]])
    metrics = depth_metric_bundle(pred, gt)
    # GT-valid L1 includes the missing depth at GT=2 as a zero prediction.
    assert metrics["depth_l1_gt_valid"] == pytest.approx(1.0)
    assert metrics["depth_l1_overlap"] == pytest.approx(0.5)
    assert metrics["pred_depth_coverage"] == pytest.approx(2 / 3)
    assert metrics["valid_gt_pixel_count"] == 3
    assert metrics["overlap_pixel_count"] == 2


def test_depth_metric_respects_protocol_mask_and_range():
    gt = torch.tensor([[0.3, 1.0], [2.0, 5.0]])
    pred = torch.tensor([[0.3, 1.5], [2.5, 5.0]])
    protocol_mask = torch.tensor([[True, True], [False, True]])
    metrics = depth_metric_bundle(
        pred, gt, valid_gt_mask=protocol_mask, near=0.4, far=4.0)
    assert metrics["valid_gt_pixel_count"] == 1
    assert metrics["depth_l1_gt_valid"] == pytest.approx(0.5)
    assert metrics["pred_depth_coverage"] == pytest.approx(1.0)


def test_rgb_bundle_keeps_standard_and_legacy_names_separate():
    rgb = torch.rand(20, 24, 3)
    result = rgb_metric_bundle(rgb, rgb, valid_gt_mask=torch.ones(20, 24, dtype=torch.bool))
    assert result["psnr_valid_gt"] > 100
    assert result["ssim_standard"] == pytest.approx(1.0, abs=1e-7)
    assert result["ssim_legacy"] == pytest.approx(1.0, abs=1e-7)
    assert result["valid_rgb_pixel_count"] == 20 * 24
