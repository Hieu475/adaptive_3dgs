"""Adaptive Gaussian densification and pruning.

Handles the online addition of new Gaussians based on error masks
and the removal of low-value Gaussians.

Key mechanisms from RTG-SLAM:
- Error masks: color error + depth error + transmission identify missing geometry
- Selective sampling: only add Gaussians where errors are high
- Importance-driven: use importance scores to guide addition/removal
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def compute_error_masks(
    color_err: torch.Tensor,
    depth_err: torch.Tensor,
    transmission: torch.Tensor,
    color_threshold: float = 0.1,
    depth_threshold: float = 0.05,
    transmission_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Compute error masks for densification.
    
    Areas with high error or high transmission (see-through) need new Gaussians.
    
    Args:
        color_err: (H, W) per-pixel color error
        depth_err: (H, W) per-pixel depth error
        transmission: (H, W) remaining light transmittance
        color_threshold: threshold for color error mask
        depth_threshold: threshold for depth error mask
        transmission_threshold: threshold for transmission mask
    
    Returns:
        Dict with 'color_mask', 'depth_mask', 'transmission_mask', 'combined_mask'
    """
    color_mask = color_err > color_threshold
    depth_mask = depth_err > depth_threshold
    transmission_mask = transmission > transmission_threshold
    
    # Combined: any condition triggers densification
    combined = color_mask | depth_mask | transmission_mask
    
    return {
        'color_mask': color_mask,
        'depth_mask': depth_mask,
        'transmission_mask': transmission_mask,
        'combined_mask': combined,
    }


def sample_candidates(
    error_mask: torch.Tensor,
    num_samples: int,
    strategy: str = 'random',
) -> torch.Tensor:
    """Sample pixel locations from error mask for new Gaussian creation.
    
    Args:
        error_mask: (H, W) boolean mask of high-error regions
        num_samples: maximum number of samples
        strategy: 'random' or 'importance' (weight by error magnitude)
    
    Returns:
        candidates_uv: (K, 2) pixel coordinates (u, v) where K <= num_samples
    """
    # Get all valid pixel locations
    ys, xs = torch.where(error_mask)
    
    if len(ys) == 0:
        return torch.empty(0, 2, dtype=torch.long, device=error_mask.device)
    
    # Subsample if too many
    K = min(num_samples, len(ys))
    
    if strategy == 'random':
        indices = torch.randperm(len(ys), device=error_mask.device)[:K]
    else:
        indices = torch.randperm(len(ys), device=error_mask.device)[:K]
    
    return torch.stack([xs[indices], ys[indices]], dim=-1)  # (K, 2) as (u, v)


def unproject_pixels(
    uv: torch.Tensor,
    depth_map: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    """Unproject pixel coordinates to 3D world points using depth.
    
    Args:
        uv: (K, 2) pixel coordinates (u, v)
        depth_map: (H, W) depth map
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (4, 4) world-to-camera transform
    
    Returns:
        points_world: (K, 3) 3D points in world space
    """
    intrinsics = intrinsics.to(uv.device)
    extrinsics = extrinsics.to(uv.device)
    
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    u = uv[:, 0].float()
    v = uv[:, 1].float()
    
    # Get depth at pixel locations
    d = depth_map[v.long(), u.long()]
    
    # Unproject to camera space
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d
    
    points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (K, 3)
    
    # Transform to world space: p_world = R^T @ (p_cam - t)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_world = (points_cam - t.unsqueeze(0)) @ R  # R^T = R since R is orthogonal -> use R
    # Actually: camera-to-world is inverse of extrinsics
    # p_cam = R @ p_world + t  =>  p_world = R^T @ (p_cam - t)
    points_world = (points_cam - t.unsqueeze(0)) @ R.T.T  # R^{-1} = R^T
    # Correction: extrinsics maps world->cam, so R_cw, t_cw
    # p_cam = R_cw @ p_world + t_cw
    # p_world = R_cw^T @ (p_cam - t_cw)
    R_cw = extrinsics[:3, :3]
    t_cw = extrinsics[:3, 3]
    points_world = (points_cam - t_cw.unsqueeze(0)) @ R_cw  # This is R_cw^T since (A @ B)^T = B^T @ A^T... 
    # Let me be precise:
    # p_world = R_cw^{-1} @ (p_cam - t_cw) = R_cw^T @ (p_cam - t_cw)
    # In batch: points_world = (points_cam - t_cw) @ R_cw  (since (R^T @ v)^T = v^T @ R)
    points_world = torch.mm(points_cam - t_cw.unsqueeze(0), R_cw.T.T)
    # R_cw.T.T = R_cw... that's wrong.
    # Let's do it properly:
    # points_world[i] = R_cw^T @ (points_cam[i] - t_cw)
    # = (points_cam[i] - t_cw) @ R_cw  -- NO
    # If we write it as row vectors: p_world^T = (p_cam - t)^T @ (R^T)^T = (p_cam - t)^T @ R
    # Wait: p_world = R^T @ (p_cam - t)
    # As row vectors: p_world^T = (p_cam - t)^T @ R^{T^T} = (p_cam - t)^T @ R -- NO
    # p^T = ((R^T)(p_cam - t))^T = (p_cam - t)^T @ R
    # YES! So points_world = (points_cam - t_cw) @ R_cw
    
    points_world = (points_cam - t_cw.unsqueeze(0)) @ R_cw
    
    return points_world


def create_gaussians_from_candidates(
    candidates_uv: torch.Tensor,
    rgb_image: torch.Tensor,
    depth_map: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    initial_scale: float = 0.01,
    initial_opacity: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Initialize new Gaussians from RGB-D data at candidate pixel locations.
    
    Args:
        candidates_uv: (K, 2) pixel coordinates
        rgb_image: (H, W, 3) RGB image
        depth_map: (H, W) depth map
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (4, 4) world-to-camera transform
        initial_scale: initial Gaussian scale
        initial_opacity: initial opacity value
    
    Returns:
        Dict with 'xyz', 'scaling', 'rotation', 'opacity', 'features_dc', 'normals'
    """
    import math
    
    if candidates_uv.shape[0] == 0:
        device = rgb_image.device
        return {
            'xyz': torch.empty(0, 3, device=device),
            'scaling': torch.empty(0, 3, device=device),
            'rotation': torch.empty(0, 4, device=device),
            'opacity': torch.empty(0, 1, device=device),
            'features_dc': torch.empty(0, 1, 3, device=device),
            'normals': torch.empty(0, 3, device=device),
        }
    
    device = rgb_image.device
    K = candidates_uv.shape[0]
    
    # Filter out candidates with invalid depth
    u = candidates_uv[:, 0].long()
    v = candidates_uv[:, 1].long()
    depths = depth_map[v, u]
    valid = depths > 0
    candidates_uv = candidates_uv[valid]
    u = u[valid]
    v = v[valid]
    K = candidates_uv.shape[0]
    
    if K == 0:
        return {
            'xyz': torch.empty(0, 3, device=device),
            'scaling': torch.empty(0, 3, device=device),
            'rotation': torch.empty(0, 4, device=device),
            'opacity': torch.empty(0, 1, device=device),
            'features_dc': torch.empty(0, 1, 3, device=device),
            'normals': torch.empty(0, 3, device=device),
        }
    
    # Unproject to 3D
    xyz = unproject_pixels(candidates_uv, depth_map, intrinsics, extrinsics)
    
    # Get colors
    colors = rgb_image[v, u]  # (K, 3)
    colors_clamped = colors.clamp(1e-4, 1.0 - 1e-4)
    features_dc = torch.log(colors_clamped / (1.0 - colors_clamped)).unsqueeze(1)  # (K, 1, 3)
    
    # Default parameters
    scaling = torch.full((K, 3), math.log(initial_scale), device=device)
    rotation = torch.zeros(K, 4, device=device)
    rotation[:, 0] = 1.0  # identity quaternion
    inv_sig = math.log(initial_opacity / (1.0 - initial_opacity + 1e-8))
    opacity = torch.full((K, 1), inv_sig, device=device)
    normals = torch.zeros(K, 3, device=device)
    normals[:, 2] = 1.0  # default normal along z
    
    return {
        'xyz': xyz,
        'scaling': scaling,
        'rotation': rotation,
        'opacity': opacity,
        'features_dc': features_dc,
        'normals': normals,
    }


def importance_driven_densification(
    model,
    importance_scores: torch.Tensor,
    high_importance_threshold: float = 0.8,
    gradient_threshold: float = 0.0002,
):
    """Split/clone high-importance Gaussians with large position gradients.
    
    Args:
        model: GaussianModel instance
        importance_scores: (N,) per-Gaussian importance
        high_importance_threshold: only densify above this importance
        gradient_threshold: minimum position gradient magnitude for densification
    """
    if model._xyz.grad is None:
        return
    
    grad_norms = model._xyz.grad.norm(dim=-1)  # (N,)
    candidates = (importance_scores > high_importance_threshold) & (grad_norms > gradient_threshold)
    
    if not candidates.any():
        return
    
    # Clone candidates (simple strategy)
    idx = torch.where(candidates)[0]
    new_params = {
        'xyz': model._xyz.data[idx] + torch.randn_like(model._xyz.data[idx]) * 0.001,
        'scaling': model._scaling.data[idx],
        'rotation': model._rotation.data[idx],
        'opacity': model._opacity.data[idx],
        'features_dc': model._features_dc.data[idx],
        'features_rest': model._features_rest.data[idx],
        'normals': model._normals.data[idx],
    }
    model.add_gaussians(new_params)


def prune_low_value(
    model,
    importance_scores: torch.Tensor,
    opacity_threshold: float = 0.005,
    importance_threshold: float = 0.01,
):
    """Remove low-contribution Gaussians.
    
    Args:
        model: GaussianModel instance
        importance_scores: (N,) per-Gaussian importance
        opacity_threshold: prune if opacity below this
        importance_threshold: prune if importance below this for extended period
    """
    opacities = model.opacities.squeeze(-1)  # (N,)
    
    prune_mask = (opacities < opacity_threshold) | (importance_scores < importance_threshold)
    
    if prune_mask.any():
        model.prune_gaussians(prune_mask)
