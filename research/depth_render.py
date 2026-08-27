"""RTG-SLAM style surface-aware depth rendering.

Instead of alpha-compositing depth (which causes floaters), this module
finds the frontmost opaque Gaussian per pixel and computes ray-plane
intersection with its dominant surface plane.

Math:
    θ = (p_G - t_cam) · n_G / (d_ray · n_G)
    p_hit = t_cam + θ · d_ray
    D(u) = ||p_hit - t_cam||₂

Gradient backprop: ∂L_depth/∂n_G and ∂L_depth/∂p_G flow through θ.
"""
import torch
from typing import Tuple, Optional, Dict


def generate_ray_directions(
    image_width: int,
    image_height: int,
    intrinsics: torch.Tensor,
    device: str = 'cpu',
) -> torch.Tensor:
    """Generate per-pixel ray directions in camera space.
    
    Args:
        image_width, image_height: image dimensions
        intrinsics: (3, 3) camera intrinsic matrix
        device: torch device
    
    Returns:
        ray_dirs: (H, W, 3) normalized ray directions in camera space
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    u = torch.arange(image_width, device=device, dtype=torch.float32)
    v = torch.arange(image_height, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    
    # Ray direction = K^{-1} @ [u, v, 1]^T
    dirs_x = (uu - cx) / fx
    dirs_y = (vv - cy) / fy
    dirs_z = torch.ones_like(dirs_x)
    
    ray_dirs = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
    ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)  # normalize
    
    return ray_dirs


def find_frontmost_opaque(
    sorted_indices: torch.Tensor,
    opacities: torch.Tensor,
    means2D: torch.Tensor,
    conics: torch.Tensor,
    pixel_coords: torch.Tensor,
    opacity_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the first Gaussian above opacity threshold for each pixel.
    
    Iterates through depth-sorted Gaussians and finds the first one whose
    evaluated opacity at the pixel location exceeds the threshold.
    
    Args:
        sorted_indices: (K,) depth-sorted Gaussian indices
        opacities: (N,) per-Gaussian base opacities
        means2D: (N, 2) screen-space centers
        conics: (N, 3) inverse covariance (a, b, c)
        pixel_coords: (P, 2) pixel positions
        opacity_threshold: threshold for "opaque" classification
    
    Returns:
        dominant_idx: (P,) index of dominant Gaussian per pixel (-1 if none)
        dominant_mask: (P,) boolean mask of pixels with valid dominant Gaussian
    """
    P = pixel_coords.shape[0]
    device = pixel_coords.device
    
    dominant_idx = torch.full((P,), -1, dtype=torch.long, device=device)
    dominant_mask = torch.zeros(P, dtype=torch.bool, device=device)
    T = torch.ones(P, device=device)  # transmittance
    
    for idx in sorted_indices:
        # Compute Gaussian weight at each pixel
        d = pixel_coords - means2D[idx].unsqueeze(0)  # (P, 2)
        a, b, c = conics[idx]
        power = -0.5 * (a * d[:, 0]**2 + 2 * b * d[:, 0] * d[:, 1] + c * d[:, 1]**2)
        power = power.clamp(max=0.0)
        
        alpha = (opacities[idx] * torch.exp(power)).clamp(max=0.99)
        effective_opacity = T * alpha
        
        # Mark pixels where this Gaussian is the first opaque one
        is_opaque = (effective_opacity > opacity_threshold) & (~dominant_mask)
        dominant_idx[is_opaque] = idx
        dominant_mask = dominant_mask | is_opaque
        
        # Update transmittance
        T = T * (1.0 - alpha)
        
        # Early exit if all pixels found their dominant Gaussian
        if dominant_mask.all():
            break
    
    return dominant_idx, dominant_mask


def ray_plane_intersection(
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    plane_point: torch.Tensor,
    plane_normal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ray-plane intersection (differentiable).
    
    θ = (p_G - t_cam) · n_G / (d_ray · n_G)
    p_hit = t_cam + θ · d_ray
    D(u) = ||p_hit - t_cam||₂ = |θ| · ||d_ray||₂
    
    Gradients flow through θ to both p_G (Gaussian center) and n_G (normal),
    enabling depth loss to supervise geometry.
    
    Args:
        ray_origin: (..., 3) camera position (t_cam)
        ray_dir: (..., 3) ray direction (d_ray), should be normalized
        plane_point: (..., 3) point on the plane (p_G = Gaussian center)
        plane_normal: (..., 3) plane normal (n_G = Gaussian normal)
    
    Returns:
        p_hit: (..., 3) intersection points
        depth: (...,) depth values ||p_hit - t_cam||₂  
        valid: (...,) boolean mask for valid intersections (ray not parallel to plane)
    """
    # Numerator: (p_G - t_cam) · n_G
    diff = plane_point - ray_origin  # (..., 3)
    numerator = (diff * plane_normal).sum(dim=-1)  # (...)
    
    # Denominator: d_ray · n_G
    denominator = (ray_dir * plane_normal).sum(dim=-1)  # (...)
    
    # Valid intersection: denominator not too close to zero (ray not parallel)
    valid = denominator.abs() > 1e-6
    
    # Safe division
    safe_denom = torch.where(valid, denominator, torch.ones_like(denominator))
    theta = numerator / safe_denom  # (...)
    
    # Intersection point
    p_hit = ray_origin + theta.unsqueeze(-1) * ray_dir  # (..., 3)
    
    # Depth = distance from camera to hit point
    depth = (p_hit - ray_origin).norm(dim=-1)  # (...)
    
    # Only valid if intersection is in front of camera (theta > 0)
    valid = valid & (theta > 0)
    
    return p_hit, depth, valid


def render_depth_surface_aware(
    means3D: torch.Tensor,
    normals: torch.Tensor,
    opacities: torch.Tensor,
    cov3D: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_width: int,
    image_height: int,
    opacity_threshold: float = 0.5,
    tile_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """Full RTG-SLAM style surface-aware depth rendering.
    
    Pipeline:
    1. Project Gaussians to screen space
    2. For each pixel, find frontmost opaque Gaussian
    3. Compute ray-plane intersection with that Gaussian's surface plane
    4. Return depth map
    
    The resulting depth map is differentiable w.r.t. Gaussian positions and normals.
    
    Args:
        means3D: (N, 3) Gaussian centers in world space
        normals: (N, 3) Gaussian surface normals in world space
        opacities: (N,) opacity values in [0, 1]
        cov3D: (N, 3, 3) 3D covariance matrices
        extrinsics: (4, 4) world-to-camera transform
        intrinsics: (3, 3) camera intrinsic matrix
        image_width, image_height: image dimensions
        opacity_threshold: threshold for opaque classification
        tile_size: tile size for binning
    
    Returns:
        Dict with:
            'depth': (H, W) depth map
            'hit_mask': (H, W) boolean mask of valid depth pixels
            'gaussian_index': (H, W) index of dominant Gaussian per pixel
    """
    from .projection import (
        world_to_camera, project_to_screen,
        compute_2d_covariance, cov2d_to_conic, compute_radii
    )
    from .rasterizer import tile_gaussians, sort_by_depth
    
    device = means3D.device
    N = means3D.shape[0]
    
    # 1. Project to screen
    means_cam = world_to_camera(means3D, extrinsics)  # (N, 3)
    valid_z = means_cam[:, 2] > 0.1
    
    means2D, depths = project_to_screen(means_cam, intrinsics)
    cov2D = compute_2d_covariance(cov3D, means_cam, extrinsics, intrinsics)
    conics = cov2d_to_conic(cov2D)
    radii = compute_radii(cov2D)
    
    # Screen-space culling
    in_screen = (
        (means2D[:, 0] + radii.float() > 0) &
        (means2D[:, 0] - radii.float() < image_width) &
        (means2D[:, 1] + radii.float() > 0) &
        (means2D[:, 1] - radii.float() < image_height) &
        valid_z
    )
    
    # 2. Generate ray directions
    ray_dirs = generate_ray_directions(image_width, image_height, intrinsics, device)
    
    # Camera position in world space
    R_cam = extrinsics[:3, :3]  # world-to-camera rotation
    t_cam_in_cam = extrinsics[:3, 3]
    cam_pos_world = -R_cam.T @ t_cam_in_cam  # camera center in world coordinates
    
    # Transform ray directions to world space
    ray_dirs_world = torch.einsum('ij,...j->...i', R_cam.T, ray_dirs)  # (H, W, 3)
    ray_dirs_world = ray_dirs_world / ray_dirs_world.norm(dim=-1, keepdim=True)
    
    # 3. Tile assignment and depth sorting
    tiles = tile_gaussians(means2D[in_screen], radii[in_screen],
                          image_width, image_height, tile_size)
    global_indices = torch.where(in_screen)[0]
    
    # 4. Per-tile: find dominant Gaussian and compute ray-plane intersection
    depth_map = torch.zeros(image_height, image_width, device=device)
    hit_mask = torch.zeros(image_height, image_width, dtype=torch.bool, device=device)
    index_map = torch.full((image_height, image_width), -1, dtype=torch.long, device=device)
    
    for (ty, tx), local_indices in tiles.items():
        tile_global_idx = global_indices[local_indices]
        sorted_idx = sort_by_depth(depths, tile_global_idx)
        
        # Pixel grid for this tile
        py_start = ty * tile_size
        px_start = tx * tile_size
        py_end = min(py_start + tile_size, image_height)
        px_end = min(px_start + tile_size, image_width)
        
        ys = torch.arange(py_start, py_end, device=device, dtype=torch.float32)
        xs = torch.arange(px_start, px_end, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        pixel_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        P = pixel_coords.shape[0]
        
        # Find dominant opaque Gaussian
        dom_idx, dom_mask = find_frontmost_opaque(
            sorted_idx, opacities, means2D, conics, pixel_coords, opacity_threshold
        )
        
        if not dom_mask.any():
            continue
        
        # Ray-plane intersection for pixels with valid dominant Gaussian
        valid_pixels = torch.where(dom_mask)[0]
        valid_gaussian_idx = dom_idx[valid_pixels]
        
        # Get pixel ray directions in world space
        pixel_y = (pixel_coords[valid_pixels, 1]).long()
        pixel_x = (pixel_coords[valid_pixels, 0]).long()
        pixel_ray_dirs = ray_dirs_world[pixel_y, pixel_x]  # (V, 3)
        
        # Ray-plane intersection
        ray_origins = cam_pos_world.unsqueeze(0).expand(valid_pixels.shape[0], -1)
        plane_points = means3D[valid_gaussian_idx]  # (V, 3)
        plane_normals = normals[valid_gaussian_idx]  # (V, 3)
        
        _, pixel_depths, valid_intersect = ray_plane_intersection(
            ray_origins, pixel_ray_dirs, plane_points, plane_normals
        )
        
        # Write to output maps
        h = py_end - py_start
        w = px_end - px_start
        
        tile_depth = torch.zeros(P, device=device)
        tile_hit = torch.zeros(P, dtype=torch.bool, device=device)
        tile_index = torch.full((P,), -1, dtype=torch.long, device=device)
        
        valid_final = dom_mask.clone()
        valid_final[valid_pixels[~valid_intersect]] = False
        
        tile_depth[valid_pixels[valid_intersect]] = pixel_depths[valid_intersect]
        tile_hit[valid_pixels[valid_intersect]] = True
        tile_index[valid_pixels[valid_intersect]] = valid_gaussian_idx[valid_intersect]
        
        depth_map[py_start:py_end, px_start:px_end] = tile_depth.reshape(h, w)
        hit_mask[py_start:py_end, px_start:px_end] = tile_hit.reshape(h, w)
        index_map[py_start:py_end, px_start:px_end] = tile_index.reshape(h, w)
    
    return {
        'depth': depth_map,
        'hit_mask': hit_mask,
        'gaussian_index': index_map,
    }
