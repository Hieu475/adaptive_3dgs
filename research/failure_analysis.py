"""Failure Analysis System for Adaptive 3D Gaussian Splatting.

Research Motivation:
    In adaptive 3DGS, continuous Gaussian importance estimation drives selective
    optimization and densification under strict compute budgets. However, various
    geometric, photometric, and viewpoint phenomena can challenge importance-driven
    Gaussian selection:

    1. FLAT_SURFACE:
       Large uniform regions where gradient magnitudes and per-pixel errors are low,
       leading to uniformly low importance scores. While individual pixel errors are small,
       accumulated low-frequency geometric/photometric drift can degrade overall quality.

    2. OBJECT_EDGE:
       Depth discontinuities and geometric boundaries between foreground and background
       objects. High-frequency occlusion boundaries require high Gaussian importance and
       dense coverage to prevent boundary blurring or bleeding artifacts.

    3. HIGH_TEXTURE:
       Complex photometric texture regions requiring high spatial frequency representation
       and numerous Gaussians. Underestimation of importance here leads to loss of fine detail.

    4. SPARSE_DEPTH:
       Regions with invalid, missing (depth == 0), or noisy depth sensor measurements
       (e.g., specular reflections, dark surfaces, sensor dropouts), where geometric error
       signals are absent or uninformative.

    5. VIEWPOINT_CHANGE:
       Rapid or large camera viewpoint shifts exposing previously occluded or unobserved
       geometry, causing transient drops in reconstruction quality before adaptation.

This module provides diagnostic tools to identify, categorize, and quantify these
failure cases in real-time or offline evaluation.
"""
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn.functional as F


class FailureType(str, Enum):
    """Enumeration of recognized failure case categories."""
    FLAT_SURFACE = "FLAT_SURFACE"
    OBJECT_EDGE = "OBJECT_EDGE"
    HIGH_TEXTURE = "HIGH_TEXTURE"
    SPARSE_DEPTH = "SPARSE_DEPTH"
    VIEWPOINT_CHANGE = "VIEWPOINT_CHANGE"


def sobel_filter_2d(image: torch.Tensor) -> torch.Tensor:
    """Apply Sobel edge detection to compute 2D gradient magnitude.

    Uses standard 3x3 Sobel convolution kernels:
        S_x = [[-1,  0,  1],
               [-2,  0,  2],
               [-1,  0,  1]]

        S_y = [[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]]

    Gradient magnitude:
        G = sqrt(G_x^2 + G_y^2)

    All operations are PyTorch-native with replicate padding to avoid boundary artifacts.

    Args:
        image: (H, W) 2D single-channel tensor, (H, W, C) multi-channel tensor,
               or (C, H, W) / (1, C, H, W) tensor.

    Returns:
        gradient_magnitude: (H, W) float tensor of spatial gradient magnitudes.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")

    orig_ndim = image.ndim
    device = image.device
    dtype = torch.float32 if image.dtype not in (torch.float32, torch.float64, torch.float16) else image.dtype

    # Reshape input to (1, C, H, W)
    if orig_ndim == 2:
        # (H, W) -> (1, 1, H, W)
        x = image.unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        C = 1
        H, W = image.shape
    elif orig_ndim == 3:
        if image.shape[2] in (1, 3, 4) and image.shape[0] not in (1, 3, 4):
            # (H, W, C) -> (1, C, H, W)
            H, W, C = image.shape
            x = image.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
        elif image.shape[0] in (1, 3, 4):
            # (C, H, W) -> (1, C, H, W)
            C, H, W = image.shape
            x = image.unsqueeze(0).to(dtype=dtype)
        else:
            # Ambiguous 3D tensor: assume (H, W, C)
            H, W, C = image.shape
            x = image.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
    elif orig_ndim == 4:
        # (B, C, H, W) -> take first batch
        B, C, H, W = image.shape
        x = image[:1].to(dtype=dtype)
    else:
        raise ValueError(f"Unsupported image shape for Sobel filter: {image.shape}")

    if H < 2 or W < 2:
        return torch.zeros((H, W), dtype=dtype, device=device)

    # Sanitize NaNs and Infs for filtering
    x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Define Sobel kernels
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  2.0,  1.0]],
        dtype=dtype,
        device=device,
    )

    weight_x = kx.view(1, 1, 3, 3).expand(C, 1, 3, 3)
    weight_y = ky.view(1, 1, 3, 3).expand(C, 1, 3, 3)

    # Replicate padding (1 pixel around border)
    x_pad = F.pad(x_clean, (1, 1, 1, 1), mode='replicate')

    gx = F.conv2d(x_pad, weight_x, groups=C)
    gy = F.conv2d(x_pad, weight_y, groups=C)

    if C == 1:
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12).squeeze(0).squeeze(0)
    else:
        # Aggregate across channels via RMS
        grad_mag = torch.sqrt((gx ** 2 + gy ** 2).sum(dim=1) + 1e-12).squeeze(0) / math.sqrt(C)

    return grad_mag


class FailureCaseAnalyzer:
    """Diagnostic system for failure case identification and severity analysis.

    Categorizes reconstruction failure modes into five canonical types:
    1. FLAT_SURFACE: Uniform regions with low gradient where quality may degrade silently.
    2. OBJECT_EDGE: Depth discontinuities where Gaussian coverage is critical.
    3. HIGH_TEXTURE: Intricate color patterns requiring high spatial frequency representation.
    4. SPARSE_DEPTH: Unobserved or invalid depth sensor measurements.
    5. VIEWPOINT_CHANGE: Large camera displacement inducing transient quality drops.
    """

    def __init__(
        self,
        flat_color_threshold: float = 0.08,
        flat_depth_threshold: float = 0.05,
        edge_std_multiplier: float = 2.0,
        texture_window_size: int = 5,
        texture_var_threshold: Optional[float] = None,
        viewpoint_threshold: float = 0.1,
    ):
        """Initialize the failure case analyzer.

        Args:
            flat_color_threshold: Maximum color gradient magnitude for flat surface detection.
            flat_depth_threshold: Maximum depth gradient magnitude for flat surface detection.
            edge_std_multiplier: Multiplier k for edge detection threshold (mean + k * std).
            texture_window_size: Window size for local gradient variance estimation.
            texture_var_threshold: Optional fixed variance threshold for high-texture regions.
            viewpoint_threshold: Threshold for camera motion (translation + rotation).
        """
        self.flat_color_threshold = flat_color_threshold
        self.flat_depth_threshold = flat_depth_threshold
        self.edge_std_multiplier = edge_std_multiplier
        self.texture_window_size = texture_window_size
        self.texture_var_threshold = texture_var_threshold
        self.viewpoint_threshold = viewpoint_threshold

    def detect_flat_surfaces(
        self,
        rgb: Optional[torch.Tensor],
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Detect flat / uniform surface regions.

        Identifies regions with low spatial gradient magnitude in both color and depth.

        Args:
            rgb: Optional (H, W, 3) or (H, W) color image in [0, 1].
            depth: (H, W) depth map.

        Returns:
            flat_mask: (H, W) boolean tensor, True for flat surface pixels.
        """
        H, W = depth.shape[:2]
        valid_depth = (depth > 0) & (~torch.isnan(depth)) & (~torch.isinf(depth))

        if not valid_depth.any():
            return torch.zeros((H, W), dtype=torch.bool, device=depth.device)

        grad_depth = sobel_filter_2d(depth)
        depth_flat = (grad_depth < self.flat_depth_threshold) & valid_depth

        if rgb is not None:
            grad_rgb = sobel_filter_2d(rgb)
            color_flat = grad_rgb < self.flat_color_threshold
            flat_mask = depth_flat & color_flat
        else:
            flat_mask = depth_flat

        return flat_mask

    def detect_object_edges(
        self,
        depth: torch.Tensor,
        rendered_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Detect object boundary edges via depth discontinuities.

        Computes depth gradient using Sobel filter and thresholds at:
            Threshold = mean(grad_depth) + k * std(grad_depth)
        where k is edge_std_multiplier (default 2.0).

        Args:
            depth: (H, W) ground-truth depth map.
            rendered_depth: Optional (H, W) rendered depth map.

        Returns:
            edge_mask: (H, W) boolean tensor, True for object boundary pixels.
        """
        H, W = depth.shape[:2]
        valid_depth = (depth > 0) & (~torch.isnan(depth)) & (~torch.isinf(depth))

        if valid_depth.sum() < 2:
            return torch.zeros((H, W), dtype=torch.bool, device=depth.device)

        grad_depth = sobel_filter_2d(depth)
        valid_grads = grad_depth[valid_depth]

        mean_val = valid_grads.mean()
        std_val = valid_grads.std()
        thresh = mean_val + self.edge_std_multiplier * std_val

        edge_mask = (grad_depth > thresh) & valid_depth

        if rendered_depth is not None:
            valid_rend = (rendered_depth > 0) & (~torch.isnan(rendered_depth)) & (~torch.isinf(rendered_depth))
            if valid_rend.sum() >= 2:
                grad_rend = sobel_filter_2d(rendered_depth)
                rend_grads = grad_rend[valid_rend]
                thresh_rend = rend_grads.mean() + self.edge_std_multiplier * rend_grads.std()
                edge_rend = (grad_rend > thresh_rend) & valid_rend
                edge_mask = edge_mask | edge_rend

        return edge_mask

    def detect_high_texture(
        self,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Detect complex high-texture regions.

        Uses local color gradient variance as a proxy for spatial frequency and entropy:
            Var_local(G) = E[G^2] - (E[G])^2

        Args:
            rgb: (H, W, 3) or (H, W) color image.

        Returns:
            high_texture_mask: (H, W) boolean tensor, True for high-texture pixels.
        """
        H, W = rgb.shape[:2]
        if H < 2 or W < 2:
            return torch.zeros((H, W), dtype=torch.bool, device=rgb.device)

        grad_rgb = sobel_filter_2d(rgb)  # (H, W)

        k = self.texture_window_size
        pad = k // 2
        grad_4d = grad_rgb.unsqueeze(0).unsqueeze(0)
        grad_pad = F.pad(grad_4d, (pad, pad, pad, pad), mode='replicate')

        local_mean = F.avg_pool2d(grad_pad, kernel_size=k, stride=1)
        local_sq_mean = F.avg_pool2d(grad_pad ** 2, kernel_size=k, stride=1)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0.0).squeeze(0).squeeze(0)

        if self.texture_var_threshold is not None:
            thresh_var = self.texture_var_threshold
            thresh_grad = 0.0
        else:
            thresh_var = local_var.mean() + 1.0 * local_var.std()
            thresh_grad = grad_rgb.mean() + 0.5 * grad_rgb.std()

        high_texture_mask = (local_var > thresh_var) & (grad_rgb > thresh_grad)
        return high_texture_mask

    def detect_sparse_depth(
        self,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Detect regions with invalid, missing, or sparse depth measurements.

        Identifies:
        - Exact invalid depth pixels (depth <= 0, NaN, Inf)
        - Depth hole regions with low local valid measurement density (< 30% valid in 7x7 patch).

        Args:
            depth: (H, W) depth map.

        Returns:
            sparse_mask: (H, W) boolean tensor, True for sparse/invalid depth pixels.
        """
        H, W = depth.shape[:2]
        invalid_pixels = (depth <= 0) | torch.isnan(depth) | torch.isinf(depth)

        if H < 7 or W < 7:
            return invalid_pixels

        valid_float = (~invalid_pixels).float().unsqueeze(0).unsqueeze(0)
        valid_pad = F.pad(valid_float, (3, 3, 3, 3), mode='replicate')
        valid_density = F.avg_pool2d(valid_pad, kernel_size=7, stride=1).squeeze(0).squeeze(0)

        sparse_mask = invalid_pixels | (valid_density < 0.3)
        return sparse_mask

    def detect_viewpoint_change(
        self,
        current_pose: Optional[torch.Tensor],
        prev_pose: Optional[torch.Tensor],
        threshold: Optional[float] = None,
    ) -> bool:
        """Detect significant camera viewpoint shift.

        Computes the SE(3) transformation delta:
            ||Δt||_2 + Δθ_geodesic > threshold

        where Δθ_geodesic = arccos((Trace(R_rel) - 1) / 2).

        Args:
            current_pose: (4, 4) current camera-to-world or world-to-camera matrix.
            prev_pose: (4, 4) previous camera matrix.
            threshold: Motion threshold (defaults to self.viewpoint_threshold).

        Returns:
            viewpoint_changed: bool, True if camera displacement exceeds threshold.
        """
        if current_pose is None or prev_pose is None:
            return False

        thresh = threshold if threshold is not None else self.viewpoint_threshold

        # Extract rotation and translation
        R_curr = current_pose[:3, :3]
        t_curr = current_pose[:3, 3]

        R_prev = prev_pose[:3, :3]
        t_prev = prev_pose[:3, 3]

        delta_t = torch.norm(t_curr - t_prev).item()

        # Geodesic rotation difference
        R_rel = torch.mm(R_curr, R_prev.transpose(0, 1))
        trace = torch.clamp((torch.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        delta_theta = math.acos(trace.item())

        total_motion = delta_t + delta_theta
        return total_motion > thresh

    def compute_region_quality(
        self,
        region_mask: torch.Tensor,
        rendered_color: torch.Tensor,
        gt_color: torch.Tensor,
    ) -> float:
        """Compute reconstruction quality (PSNR in dB) within a masked region.

        PSNR = -10 * log10(MSE + 1e-10)

        Args:
            region_mask: (H, W) boolean mask.
            rendered_color: (H, W, 3) rendered RGB image in [0, 1].
            gt_color: (H, W, 3) ground truth RGB image in [0, 1].

        Returns:
            psnr_db: float, PSNR in dB. Returns 0.0 if region is empty.
        """
        if region_mask.sum() == 0:
            return 0.0

        rendered_masked = rendered_color[region_mask]
        gt_masked = gt_color[region_mask]

        mse = ((rendered_masked - gt_masked) ** 2).mean().item()
        if mse < 1e-10:
            return 50.0

        psnr = -10.0 * math.log10(mse)
        return float(psnr)

    def compute_region_importance(
        self,
        region_mask: torch.Tensor,
        importance: torch.Tensor,
        contrib_indices: torch.Tensor,
    ) -> float:
        """Compute mean importance of Gaussians contributing to a masked image region.

        Args:
            region_mask: (H, W) boolean region mask.
            importance: (N,) per-Gaussian continuous importance scores.
            contrib_indices: (H, W, top_k) Gaussian index tensor (-1 for empty).

        Returns:
            mean_importance: float, mean importance score. Returns 0.0 if no Gaussians contribute.
        """
        if region_mask.sum() == 0 or importance.numel() == 0:
            return 0.0

        masked_indices = contrib_indices[region_mask]  # (M, top_k)
        valid = (masked_indices >= 0) & (masked_indices < importance.shape[0])

        if not valid.any():
            return 0.0

        unique_gaussians = torch.unique(masked_indices[valid])
        if unique_gaussians.numel() == 0:
            return 0.0

        mean_imp = importance[unique_gaussians].mean().item()
        return float(mean_imp)

    def _extract_contributing_gaussians(
        self,
        region_mask: torch.Tensor,
        contrib_indices: torch.Tensor,
        n_gaussians: int,
    ) -> List[int]:
        """Helper to extract unique valid Gaussian indices for a masked region."""
        if region_mask.sum() == 0 or n_gaussians == 0:
            return []

        masked = contrib_indices[region_mask]
        valid = (masked >= 0) & (masked < n_gaussians)
        if not valid.any():
            return []

        unique_idx = torch.unique(masked[valid]).tolist()
        return [int(idx) for idx in unique_idx]

    def get_edge_importance_analysis(
        self,
        depth: torch.Tensor,
        importance: torch.Tensor,
        contrib_indices: torch.Tensor,
        contrib_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Perform differential edge-vs-flat importance analysis.

        This analysis validates the core hypothesis of importance-driven 3DGS:
        Gaussians contributing to object depth discontinuities (edges) should carry
        systematically higher importance than Gaussians on uniform flat surfaces.

        Args:
            depth: (H, W) ground-truth depth map.
            importance: (N,) continuous Gaussian importance scores.
            contrib_indices: (H, W, top_k) contributing Gaussian indices.
            contrib_weights: Optional (H, W, top_k) contribution alpha weights.

        Returns:
            Dict containing:
                - 'edge_pixels': int count of edge pixels
                - 'flat_pixels': int count of flat pixels
                - 'edge_mean_importance': float mean importance on edges
                - 'flat_mean_importance': float mean importance on flat regions
                - 'importance_ratio': float ratio (edge_mean / flat_mean)
                - 'edge_gaussians_count': int count of unique Gaussians at edges
                - 'flat_gaussians_count': int count of unique Gaussians on flats
                - 'edge_weighted_importance': float alpha-weighted edge importance
                - 'flat_weighted_importance': float alpha-weighted flat importance
        """
        N = importance.shape[0]
        edge_mask = self.detect_object_edges(depth)
        flat_mask = self.detect_flat_surfaces(rgb=None, depth=depth)

        edge_pixels = int(edge_mask.sum().item())
        flat_pixels = int(flat_mask.sum().item())

        edge_mean_imp = self.compute_region_importance(edge_mask, importance, contrib_indices)
        flat_mean_imp = self.compute_region_importance(flat_mask, importance, contrib_indices)

        edge_gaussians = self._extract_contributing_gaussians(edge_mask, contrib_indices, N)
        flat_gaussians = self._extract_contributing_gaussians(flat_mask, contrib_indices, N)

        # Compute alpha-weighted importance if weights provided
        if contrib_weights is not None:
            # Edge weighted importance
            if edge_pixels > 0:
                cw_edge = contrib_weights[edge_mask]
                ci_edge = contrib_indices[edge_mask]
                v_edge = (ci_edge >= 0) & (ci_edge < N) & (cw_edge > 0)
                if v_edge.any():
                    imp_vals = importance[ci_edge[v_edge]]
                    w_vals = cw_edge[v_edge]
                    edge_weighted_imp = ((imp_vals * w_vals).sum() / (w_vals.sum() + 1e-8)).item()
                else:
                    edge_weighted_imp = 0.0
            else:
                edge_weighted_imp = 0.0

            # Flat weighted importance
            if flat_pixels > 0:
                cw_flat = contrib_weights[flat_mask]
                ci_flat = contrib_indices[flat_mask]
                v_flat = (ci_flat >= 0) & (ci_flat < N) & (cw_flat > 0)
                if v_flat.any():
                    imp_vals = importance[ci_flat[v_flat]]
                    w_vals = cw_flat[v_flat]
                    flat_weighted_imp = ((imp_vals * w_vals).sum() / (w_vals.sum() + 1e-8)).item()
                else:
                    flat_weighted_imp = 0.0
            else:
                flat_weighted_imp = 0.0
        else:
            edge_weighted_imp = edge_mean_imp
            flat_weighted_imp = flat_mean_imp

        importance_ratio = (edge_mean_imp + 1e-6) / (flat_mean_imp + 1e-6)

        return {
            'edge_pixels': edge_pixels,
            'flat_pixels': flat_pixels,
            'edge_mean_importance': float(edge_mean_imp),
            'flat_mean_importance': float(flat_mean_imp),
            'importance_ratio': float(importance_ratio),
            'edge_gaussians_count': len(edge_gaussians),
            'flat_gaussians_count': len(flat_gaussians),
            'edge_weighted_importance': float(edge_weighted_imp),
            'flat_weighted_importance': float(flat_weighted_imp),
        }

    def analyze_frame(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        rendered_color: torch.Tensor,
        rendered_depth: torch.Tensor,
        importance: torch.Tensor,
        visibility_mask: Optional[torch.Tensor],
        contrib_indices: torch.Tensor,
        contrib_weights: torch.Tensor,
        n_gaussians: int,
        current_pose: Optional[torch.Tensor] = None,
        prev_pose: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Execute full failure analysis for an RGB-D frame.

        Identifies active failure modes across all 5 canonical failure types:
        1. FLAT_SURFACE
        2. OBJECT_EDGE
        3. HIGH_TEXTURE
        4. SPARSE_DEPTH
        5. VIEWPOINT_CHANGE

        Args:
            rgb: (H, W, 3) Ground truth color image in [0, 1].
            depth: (H, W) Ground truth depth map.
            rendered_color: (H, W, 3) Rendered color image in [0, 1].
            rendered_depth: (H, W) Rendered depth map.
            importance: (N,) Per-Gaussian importance scores.
            visibility_mask: (N,) Boolean visibility tensor (optional).
            contrib_indices: (H, W, top_k) Contributing Gaussian indices.
            contrib_weights: (H, W, top_k) Contributing alpha weights.
            n_gaussians: Total Gaussian count N.
            current_pose: Optional (4, 4) current camera pose.
            prev_pose: Optional (4, 4) previous camera pose.

        Returns:
            Dict containing:
                Per failure type:
                    - affected_pixels: int
                    - affected_gaussians: List[int]
                    - quality_in_region: float (PSNR in dB)
                    - importance_in_region: float (mean importance)
                    - failure_severity: float in [0, 1]
                Global summary metrics:
                    - overall_psnr: float
                    - edge_importance_analysis: Dict
                    - detected_failure_types: List[str]
        """
        H, W = depth.shape[:2]
        total_pixels = H * W

        # 1. Detect Failure Regions
        mask_flat = self.detect_flat_surfaces(rgb, depth)
        mask_edge = self.detect_object_edges(depth, rendered_depth)
        mask_texture = self.detect_high_texture(rgb)
        mask_sparse = self.detect_sparse_depth(depth)
        viewpoint_changed = self.detect_viewpoint_change(current_pose, prev_pose)

        results: Dict[str, Any] = {}

        # --- 1. FLAT_SURFACE ---
        pixels_flat = int(mask_flat.sum().item())
        g_flat = self._extract_contributing_gaussians(mask_flat, contrib_indices, n_gaussians)
        quality_flat = self.compute_region_quality(mask_flat, rendered_color, rgb)
        imp_flat = self.compute_region_importance(mask_flat, importance, contrib_indices)
        if pixels_flat > 0:
            # Severity rises if quality is degraded (< 35 dB) and importance is very low (< 0.2)
            quality_penalty = max(0.0, 1.0 - quality_flat / 35.0)
            imp_deficit = max(0.0, 0.5 - imp_flat)
            area_weight = min(1.0, pixels_flat / (0.2 * total_pixels + 1e-6))
            sev_flat = float(torch.clamp(torch.tensor(quality_penalty * (1.0 + imp_deficit) * area_weight), 0.0, 1.0).item())
        else:
            sev_flat = 0.0

        results[FailureType.FLAT_SURFACE] = {
            'affected_pixels': pixels_flat,
            'affected_gaussians': g_flat,
            'quality_in_region': float(quality_flat),
            'importance_in_region': float(imp_flat),
            'failure_severity': sev_flat,
        }

        # --- 2. OBJECT_EDGE ---
        pixels_edge = int(mask_edge.sum().item())
        g_edge = self._extract_contributing_gaussians(mask_edge, contrib_indices, n_gaussians)
        quality_edge = self.compute_region_quality(mask_edge, rendered_color, rgb)
        imp_edge = self.compute_region_importance(mask_edge, importance, contrib_indices)
        if pixels_edge > 0:
            # Edges are high-frequency; quality drops below 30 dB indicate edge bleeding/blur
            quality_penalty = max(0.0, 1.0 - quality_edge / 30.0)
            sev_edge = float(torch.clamp(torch.tensor(quality_penalty), 0.0, 1.0).item())
        else:
            sev_edge = 0.0

        results[FailureType.OBJECT_EDGE] = {
            'affected_pixels': pixels_edge,
            'affected_gaussians': g_edge,
            'quality_in_region': float(quality_edge),
            'importance_in_region': float(imp_edge),
            'failure_severity': sev_edge,
        }

        # --- 3. HIGH_TEXTURE ---
        pixels_tex = int(mask_texture.sum().item())
        g_tex = self._extract_contributing_gaussians(mask_texture, contrib_indices, n_gaussians)
        quality_tex = self.compute_region_quality(mask_texture, rendered_color, rgb)
        imp_tex = self.compute_region_importance(mask_texture, importance, contrib_indices)
        if pixels_tex > 0:
            # Texture quality drops below 32 dB indicate under-densification/blur
            quality_penalty = max(0.0, 1.0 - quality_tex / 32.0)
            sev_tex = float(torch.clamp(torch.tensor(quality_penalty), 0.0, 1.0).item())
        else:
            sev_tex = 0.0

        results[FailureType.HIGH_TEXTURE] = {
            'affected_pixels': pixels_tex,
            'affected_gaussians': g_tex,
            'quality_in_region': float(quality_tex),
            'importance_in_region': float(imp_tex),
            'failure_severity': sev_tex,
        }

        # --- 4. SPARSE_DEPTH ---
        pixels_sparse = int(mask_sparse.sum().item())
        g_sparse = self._extract_contributing_gaussians(mask_sparse, contrib_indices, n_gaussians)
        quality_sparse = self.compute_region_quality(mask_sparse, rendered_color, rgb)
        imp_sparse = self.compute_region_importance(mask_sparse, importance, contrib_indices)
        if pixels_sparse > 0:
            sparse_frac = min(1.0, pixels_sparse / total_pixels)
            quality_penalty = max(0.0, 1.0 - quality_sparse / 30.0) if quality_sparse > 0 else 0.5
            sev_sparse = float(torch.clamp(torch.tensor(sparse_frac * (0.5 + 0.5 * quality_penalty)), 0.0, 1.0).item())
        else:
            sev_sparse = 0.0

        results[FailureType.SPARSE_DEPTH] = {
            'affected_pixels': pixels_sparse,
            'affected_gaussians': g_sparse,
            'quality_in_region': float(quality_sparse),
            'importance_in_region': float(imp_sparse),
            'failure_severity': sev_sparse,
        }

        # --- 5. VIEWPOINT_CHANGE ---
        if viewpoint_changed:
            mask_all = torch.ones((H, W), dtype=torch.bool, device=depth.device)
            quality_all = self.compute_region_quality(mask_all, rendered_color, rgb)
            if visibility_mask is not None:
                vis_indices = torch.where(visibility_mask)[0].tolist()
                imp_view = float(importance[visibility_mask].mean().item()) if visibility_mask.any() else 0.0
            else:
                vis_indices = list(range(n_gaussians))
                imp_view = float(importance.mean().item()) if importance.numel() > 0 else 0.0

            # Compute motion magnitude
            t_curr = current_pose[:3, 3]
            t_prev = prev_pose[:3, 3]
            R_rel = torch.mm(current_pose[:3, :3], prev_pose[:3, :3].transpose(0, 1))
            trace = torch.clamp((torch.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
            motion = (torch.norm(t_curr - t_prev) + math.acos(trace.item())).item()
            sev_view = float(torch.clamp(torch.tensor(motion / (self.viewpoint_threshold * 4.0)), 0.0, 1.0).item())

            results[FailureType.VIEWPOINT_CHANGE] = {
                'affected_pixels': total_pixels,
                'affected_gaussians': [int(i) for i in vis_indices],
                'quality_in_region': float(quality_all),
                'importance_in_region': float(imp_view),
                'failure_severity': sev_view,
            }
        else:
            results[FailureType.VIEWPOINT_CHANGE] = {
                'affected_pixels': 0,
                'affected_gaussians': [],
                'quality_in_region': 0.0,
                'importance_in_region': 0.0,
                'failure_severity': 0.0,
            }

        # Global metrics
        mask_full = torch.ones((H, W), dtype=torch.bool, device=depth.device)
        overall_psnr = self.compute_region_quality(mask_full, rendered_color, rgb)
        edge_analysis = self.get_edge_importance_analysis(depth, importance, contrib_indices, contrib_weights)

        # Detect active failure modes with non-zero severity
        detected = [
            ft.value for ft in FailureType
            if results[ft]['failure_severity'] > 0.1 or results[ft]['affected_pixels'] > (0.05 * total_pixels)
        ]

        results['overall_psnr'] = float(overall_psnr)
        results['total_gaussians'] = int(n_gaussians)
        results['visible_gaussians_count'] = int(visibility_mask.sum().item()) if visibility_mask is not None else n_gaussians
        results['edge_importance_analysis'] = edge_analysis
        results['detected_failure_types'] = detected

        return results


def format_failure_analysis_report(analysis: Dict[str, Any]) -> str:
    """Generate a clean ASCII failure analysis report for terminal and logging.

    Args:
        analysis: Dictionary returned by FailureCaseAnalyzer.analyze_frame.

    Returns:
        Formatted multi-line report string.
    """
    lines = [
        "=" * 72,
        "             ADAPTIVE 3DGS FAILURE ANALYSIS DIAGNOSTIC REPORT",
        "=" * 72,
        f"Overall Frame Quality (PSNR): {analysis.get('overall_psnr', 0.0):.2f} dB",
        f"Active Failure Modes Detected: {', '.join(analysis.get('detected_failure_types', [])) or 'None'}",
        "-" * 72,
        f"{'Failure Category':<18} | {'Pixels':<8} | {'PSNR (dB)':<10} | {'Mean Imp':<9} | {'Severity':<8}",
        "-" * 72,
    ]

    for ft in FailureType:
        res = analysis.get(ft.value, analysis.get(ft, {}))
        pixels = res.get('affected_pixels', 0)
        psnr = res.get('quality_in_region', 0.0)
        imp = res.get('importance_in_region', 0.0)
        sev = res.get('failure_severity', 0.0)
        psnr_str = f"{psnr:.2f}" if pixels > 0 else "N/A"
        lines.append(
            f"{ft.value:<18} | {pixels:<8d} | {psnr_str:<10} | {imp:<9.4f} | {sev:<8.2f}"
        )

    lines.append("-" * 72)
    edge_info = analysis.get('edge_importance_analysis', {})
    if edge_info:
        lines.extend([
            "Edge vs Flat Differential Importance Analysis:",
            f"  - Edge Pixels: {edge_info.get('edge_pixels', 0):,d}  |  Flat Pixels: {edge_info.get('flat_pixels', 0):,d}",
            f"  - Edge Mean Importance: {edge_info.get('edge_mean_importance', 0.0):.4f}",
            f"  - Flat Mean Importance: {edge_info.get('flat_mean_importance', 0.0):.4f}",
            f"  - Edge/Flat Importance Ratio: {edge_info.get('importance_ratio', 0.0):.2f}x (Higher is better)",
            f"  - Edge Gaussians: {edge_info.get('edge_gaussians_count', 0)}  |  Flat Gaussians: {edge_info.get('flat_gaussians_count', 0)}",
        ])
    lines.append("=" * 72)

    return "\n".join(lines)
