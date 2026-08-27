"""Python reference rasterizer for 3D Gaussian Splatting.

This is a pure-Python/PyTorch implementation for correctness verification.
The production pipeline uses CUDA kernels (cuda/rasterize.cu).

Rendering equation (front-to-back alpha compositing):
    C(u) = Σᵢ cᵢ · fᵢ(u) · ∏ⱼ<ᵢ (1 - fⱼ(u))
where:
    fᵢ(u) = αᵢ · exp(-0.5 · (u-μᵢ)ᵀ · Σ₂D,ᵢ⁻¹ · (u-μᵢ))
"""
import torch
from typing import Dict, Tuple, Optional
import math


def compute_gaussian_weight(
    pixel_coords: torch.Tensor,
    mean2D: torch.Tensor,
    conic: torch.Tensor,
    opacity: float,
) -> torch.Tensor:
    """Compute the Gaussian contribution at pixel locations.
    
    f(u) = α · exp(-0.5 · (u-μ)ᵀ · Σ⁻¹ · (u-μ))
    
    where conic = (Σ⁻¹₀₀, Σ⁻¹₀₁, Σ⁻¹₁₁) is the upper-triangle of the inverse cov.
    
    Args:
        pixel_coords: (P, 2) pixel positions
        mean2D: (2,) Gaussian center
        conic: (3,) inverse covariance (a, b, c) where Σ⁻¹ = [[a,b],[b,c]]
        opacity: scalar opacity α
    
    Returns:
        weights: (P,) Gaussian weights at each pixel
    """
    d = pixel_coords - mean2D.unsqueeze(0)  # (P, 2)
    a, b, c = conic[0], conic[1], conic[2]
    
    # Mahalanobis distance: dᵀ Σ⁻¹ d
    power = -0.5 * (a * d[:, 0]**2 + 2 * b * d[:, 0] * d[:, 1] + c * d[:, 1]**2)
    
    # Clamp to avoid numerical overflow
    power = power.clamp(max=0.0)  # Should be ≤ 0 for valid Gaussian
    
    return opacity * torch.exp(power)


def tile_gaussians(
    means2D: torch.Tensor,
    radii: torch.Tensor,
    image_width: int,
    image_height: int,
    tile_size: int = 16,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Assign Gaussians to screen tiles based on their bounding boxes.
    
    Args:
        means2D: (N, 2) screen-space centers
        radii: (N,) pixel radii
        image_width, image_height: image dimensions
        tile_size: tile size in pixels
    
    Returns:
        Dict mapping (tile_y, tile_x) -> tensor of Gaussian indices
    """
    N = means2D.shape[0]
    if N == 0:
        return {}
        
    n_tiles_x = (image_width + tile_size - 1) // tile_size
    n_tiles_y = (image_height + tile_size - 1) // tile_size
    
    cx = means2D[:, 0]
    cy = means2D[:, 1]
    r = radii.float()
    
    tx_min = ((cx - r) / tile_size).floor().long().clamp(0, n_tiles_x - 1)
    tx_max = ((cx + r) / tile_size).floor().long().clamp(0, n_tiles_x - 1)
    ty_min = ((cy - r) / tile_size).floor().long().clamp(0, n_tiles_y - 1)
    ty_max = ((cy + r) / tile_size).floor().long().clamp(0, n_tiles_y - 1)
    
    tiles = {}
    for i in range(N):
        y_start = ty_min[i].item()
        y_end = ty_max[i].item()
        x_start = tx_min[i].item()
        x_end = tx_max[i].item()
        for ty in range(y_start, y_end + 1):
            for tx in range(x_start, x_end + 1):
                key = (ty, tx)
                if key not in tiles:
                    tiles[key] = []
                tiles[key].append(i)
    
    return {k: torch.tensor(v, dtype=torch.long, device=means2D.device) for k, v in tiles.items()}


def sort_by_depth(
    depths: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Sort Gaussian indices by depth (front-to-back) within a tile.
    
    Args:
        depths: (N,) depth values for all Gaussians
        indices: (K,) indices of Gaussians in this tile
    
    Returns:
        sorted_indices: (K,) indices sorted by increasing depth
    """
    tile_depths = depths[indices]
    sorted_order = torch.argsort(tile_depths)
    return indices[sorted_order]


def rasterize_pixels(
    pixel_coords: torch.Tensor,
    sorted_indices: torch.Tensor,
    means2D: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    bg_color: torch.Tensor,
    early_termination_threshold: float = 0.99,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alpha-composite Gaussians at given pixel coordinates.
    
    Front-to-back compositing:
        C(u) = Σᵢ cᵢ · αᵢ(u) · Tᵢ
        Tᵢ = ∏ⱼ<ᵢ (1 - αⱼ(u))
    
    Args:
        pixel_coords: (P, 2) pixel positions
        sorted_indices: (K,) depth-sorted Gaussian indices
        means2D: (N, 2) all Gaussian centers
        conics: (N, 3) all conics (inverse covariance)
        colors: (N, 3) all Gaussian colors
        opacities: (N,) all opacities
        depths: (N,) all depths
        bg_color: (3,) background color
        early_termination_threshold: stop when transmittance < (1 - threshold)
    
    Returns:
        rendered_color: (P, 3)
        rendered_depth: (P,)
        transmittance: (P,) final transmittance
    """
    P = pixel_coords.shape[0]
    device = pixel_coords.device
    K = sorted_indices.shape[0]
    
    if K == 0 or P == 0:
        rendered_color = bg_color.unsqueeze(0).expand(P, 3)
        rendered_depth = torch.zeros(P, device=device)
        T = torch.ones(P, device=device)
        return rendered_color, rendered_depth, T
    
    sub_means = means2D[sorted_indices]       # (K, 2)
    sub_conics = conics[sorted_indices]       # (K, 3)
    sub_colors = colors[sorted_indices]       # (K, 3)
    sub_opacities = opacities[sorted_indices] # (K,)
    sub_depths = depths[sorted_indices]       # (K,)
    
    d = pixel_coords.unsqueeze(1) - sub_means.unsqueeze(0)  # (P, K, 2)
    a = sub_conics[:, 0].unsqueeze(0)  # (1, K)
    b = sub_conics[:, 1].unsqueeze(0)
    c = sub_conics[:, 2].unsqueeze(0)
    
    dx = d[:, :, 0]
    dy = d[:, :, 1]
    
    power = -0.5 * (a * dx**2 + 2.0 * b * dx * dy + c * dy**2)
    power = power.clamp(max=0.0)
    
    alpha = (sub_opacities.unsqueeze(0) * torch.exp(power)).clamp(max=0.99)  # (P, K)
    
    # Front-to-back transmittance
    one_minus_alpha = 1.0 - alpha
    cumprod = torch.cumprod(one_minus_alpha, dim=1)  # (P, K)
    T = torch.cat([torch.ones(P, 1, device=device), cumprod[:, :-1]], dim=1)  # (P, K)
    weights = alpha * T  # (P, K)
    
    final_T = cumprod[:, -1]  # (P,)
    rendered_color = (weights.unsqueeze(-1) * sub_colors.unsqueeze(0)).sum(dim=1) + final_T.unsqueeze(-1) * bg_color.unsqueeze(0)
    rendered_depth = (weights * sub_depths.unsqueeze(0)).sum(dim=1)
    
    return rendered_color, rendered_depth, final_T


def render(
    means3D: torch.Tensor,
    cov3D: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_width: int,
    image_height: int,
    bg_color: Optional[torch.Tensor] = None,
    tile_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """Full rendering pipeline: project → tile → sort → rasterize.
    
    Args:
        means3D: (N, 3) Gaussian centers in world space
        cov3D: (N, 3, 3) 3D covariance matrices
        colors: (N, 3) Gaussian colors
        opacities: (N,) Gaussian opacities
        extrinsics: (4, 4) world-to-camera transform
        intrinsics: (3, 3) camera intrinsic matrix
        image_width, image_height: output dimensions
        bg_color: (3,) background color, default black
        tile_size: tile size for binning
    
    Returns:
        Dict with 'color' (H,W,3), 'depth' (H,W), 'transmission' (H,W)
    """
    from .projection import (
        world_to_camera, project_to_screen,
        compute_2d_covariance, cov2d_to_conic, compute_radii
    )
    
    if bg_color is None:
        bg_color = torch.zeros(3, device=means3D.device)
    
    N = means3D.shape[0]
    device = means3D.device
    
    # 1. Transform to camera space
    means_cam = world_to_camera(means3D, extrinsics)  # (N, 3)
    
    # 2. Frustum culling: keep only points in front of camera
    valid = means_cam[:, 2] > 0.1  # near plane
    if not valid.any():
        return {
            'color': bg_color.unsqueeze(0).unsqueeze(0).expand(image_height, image_width, 3),
            'depth': torch.zeros(image_height, image_width, device=device),
            'transmission': torch.ones(image_height, image_width, device=device),
        }
    
    # 3. Project to screen
    means2D, depths = project_to_screen(means_cam, intrinsics)  # (N,2), (N,)
    
    # 4. Compute 2D covariance
    cov2D = compute_2d_covariance(cov3D, means_cam, extrinsics, intrinsics)  # (N,2,2)
    conics = cov2d_to_conic(cov2D)  # (N, 3)
    radii = compute_radii(cov2D)  # (N,)
    
    # 5. Screen-space culling
    in_screen = (
        (means2D[:, 0] + radii.float() > 0) &
        (means2D[:, 0] - radii.float() < image_width) &
        (means2D[:, 1] + radii.float() > 0) &
        (means2D[:, 1] - radii.float() < image_height) &
        valid
    )
    
    # 6. Tile assignment
    tiles = tile_gaussians(means2D[in_screen], radii[in_screen], 
                          image_width, image_height, tile_size)
    
    # Map local indices back to global
    global_indices = torch.where(in_screen)[0]
    
    # 7. Render each tile
    color_image = torch.zeros(image_height, image_width, 3, device=device)
    depth_image = torch.zeros(image_height, image_width, device=device)
    trans_image = torch.ones(image_height, image_width, device=device)
    
    n_tiles_x = (image_width + tile_size - 1) // tile_size
    n_tiles_y = (image_height + tile_size - 1) // tile_size
    
    for (ty, tx), local_indices in tiles.items():
        # Map to global indices
        tile_global_idx = global_indices[local_indices]
        
        # Sort by depth
        sorted_idx = sort_by_depth(depths, tile_global_idx)
        
        # Pixel coordinates in this tile
        py_start = ty * tile_size
        px_start = tx * tile_size
        py_end = min(py_start + tile_size, image_height)
        px_end = min(px_start + tile_size, image_width)
        
        ys = torch.arange(py_start, py_end, device=device, dtype=torch.float32)
        xs = torch.arange(px_start, px_end, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        pixel_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (P, 2)
        
        # Rasterize
        tile_color, tile_depth, tile_T = rasterize_pixels(
            pixel_coords, sorted_idx, means2D, conics, colors,
            opacities, depths, bg_color
        )
        
        # Write back to image
        h = py_end - py_start
        w = px_end - px_start
        color_image[py_start:py_end, px_start:px_end] = tile_color.reshape(h, w, 3)
        depth_image[py_start:py_end, px_start:px_end] = tile_depth.reshape(h, w)
        trans_image[py_start:py_end, px_start:px_end] = tile_T.reshape(h, w)
    
    return {
        'color': color_image,
        'depth': depth_image,
        'transmission': trans_image,
    }
