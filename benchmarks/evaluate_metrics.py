"""
Compute PSNR, SSIM, LPIPS, depth metrics (L1, AbsRel, Accuracy, Completion).
"""
import numpy as np

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes PSNR between two images."""
    # TODO: Implement PSNR
    return 0.0

def compute_depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray):
    """Computes depth metrics."""
    # TODO: Implement L1, AbsRel, etc.
    return {}

if __name__ == "__main__":
    print("Evaluating metrics...")
