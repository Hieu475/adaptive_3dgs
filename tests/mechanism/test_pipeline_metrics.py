import torch

from research.pipeline import OnlineReconstructionPipeline


def _tiny_scene():
    h = w = 16
    rgb = torch.full((h, w, 3), 0.45)
    depth = torch.full((h, w), 2.0)
    intrinsics = torch.tensor([[20.0, 0.0, 8.0],
                               [0.0, 20.0, 8.0],
                               [0.0, 0.0, 1.0]])
    return rgb, depth, intrinsics


def test_eval_frame_reports_depth_coverage_and_image_panels():
    rgb, depth, intrinsics = _tiny_scene()
    pipeline = OnlineReconstructionPipeline(device="cpu")
    pipeline.initialize(rgb, depth, intrinsics)
    result = pipeline.eval_frame(rgb, depth, torch.eye(4), return_images=True)
    assert result["ssim"] == result["ssim_standard"]
    assert "ssim_legacy" in result
    assert result["valid_gt_pixel_count"] == depth.numel()
    assert 0.0 <= result["pred_depth_coverage"] <= 1.0
    assert result["overlap_pixel_count"] <= result["valid_gt_pixel_count"]
    assert set(result["images"]) >= {
        "rgb_prediction", "rgb_ground_truth", "depth_prediction",
        "depth_ground_truth", "depth_absolute_error_gt_valid",
        "pred_depth_coverage_mask", "gt_valid_mask",
    }
    pipeline.cleanup()


def test_process_frame_separates_optimization_budget_from_frame_deadline():
    rgb, depth, intrinsics = _tiny_scene()
    pipeline = OnlineReconstructionPipeline(device="cpu")
    pipeline.initialize(rgb, depth, intrinsics)
    result = pipeline.process_frame(rgb, depth)
    assert result["budget_opt_modeled_ms"] == 16.6
    assert result["time_frame_wall_ms"] >= 0
    assert result["time_opt_cuda_ms"] is None
    assert result["violation_opt"] is None
    assert result["violation_frame"] is None
    assert "gain_train_psnr_db" in result
    assert result["gain_holdout_psnr_db"] is None
    pipeline.cleanup()
