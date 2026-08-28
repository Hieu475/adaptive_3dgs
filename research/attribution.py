"""Per-Gaussian Attribution: Pixel Error → Gaussian Responsibility.

This module solves the fundamental attribution problem:
    Given pixel-wise RGB-D errors, which Gaussians are responsible?

Key equations:
    Contribution weight:  w_{u,i} = α_i(u) · T_i(u)
    where T_i(u) = ∏_{j<i} (1 - α_j(u))

    Per-Gaussian color error:
        E^color_i = Σ_u w_{u,i} · e_c(u) / (Σ_u w_{u,i} + ε)

    Per-Gaussian depth error:
        E^depth_i = Σ_u w_{u,i} · e_d(u) / (Σ_u w_{u,i} + ε)

    Visibility:
        V_i = (1 / N_pixels) · Σ_u 1[w_{u,i} > ε]

    Contribution mass:
        C_i = Σ_u w_{u,i}

This replaces the naive global-mean approach:
    per_gaussian_err.fill_(global_mean)  ← WRONG
with true per-Gaussian error attribution.
"""
import math
import torch
from typing import Dict, Optional, Tuple


def rasterize_pixels_with_attribution(
    pixel_coords: torch.Tensor,
    sorted_indices: torch.Tensor,
    means2D: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    bg_color: torch.Tensor,
    n_gaussians: int,
    top_k: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alpha-composite Gaussians AND record per-Gaussian contributions.

    This extends rasterize_pixels() to also output:
    - contribution_weights: sparse (P, top_k) contribution weights
    - contribution_indices: sparse (P, top_k) global Gaussian indices
    - dominant_index: (P,) index of the Gaussian with highest contribution

    We track the top-K contributors per pixel to keep memory bounded
    (full N×P attribution is prohibitive).

    Args:
        pixel_coords: (P, 2) pixel positions
        sorted_indices: (K,) depth-sorted Gaussian indices (global)
        means2D: (N, 2) all Gaussian centers
        conics: (N, 3) all conics (inverse covariance)
        colors: (N, 3) all Gaussian colors
        opacities: (N,) all opacities
        depths: (N,) all depths
        bg_color: (3,) background color
        n_gaussians: total number of Gaussians in the scene
        top_k: number of top contributors to track per pixel

    Returns:
        rendered_color: (P, 3)
        rendered_depth: (P,)
        transmittance: (P,) final transmittance
        contrib_weights: (P, top_k) contribution weights w_{u,i}
        contrib_indices: (P, top_k) global Gaussian indices (-1 = unused)
        dominant_index: (P,) index of Gaussian with max contribution
    """
    P = pixel_coords.shape[0]
    device = pixel_coords.device
    K = sorted_indices.shape[0]

    # Initialize outputs
    contrib_weights = torch.zeros(P, top_k, device=device)
    contrib_indices = torch.full((P, top_k), -1, dtype=torch.long, device=device)
    dominant_index = torch.full((P,), -1, dtype=torch.long, device=device)

    if K == 0 or P == 0:
        rendered_color = bg_color.unsqueeze(0).expand(P, 3)
        rendered_depth = torch.zeros(P, device=device)
        T = torch.ones(P, device=device)
        return rendered_color, rendered_depth, T, contrib_weights, contrib_indices, dominant_index

    sub_means = means2D[sorted_indices]       # (K, 2)
    sub_conics = conics[sorted_indices]       # (K, 3)
    sub_colors = colors[sorted_indices]       # (K, 3)
    sub_opacities = opacities[sorted_indices] # (K,)
    sub_depths = depths[sorted_indices]       # (K,)

    # Compute all alpha values: (P, K)
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
    weights = alpha * T  # (P, K) — these are w_{u,i}

    final_T = cumprod[:, -1]  # (P,)

    # Standard rendering outputs
    rendered_color = (weights.unsqueeze(-1) * sub_colors.unsqueeze(0)).sum(dim=1) \
                     + final_T.unsqueeze(-1) * bg_color.unsqueeze(0)
    rendered_depth = (weights * sub_depths.unsqueeze(0)).sum(dim=1)

    # === Attribution: track top-K contributors per pixel ===
    actual_k = min(top_k, K)
    if actual_k > 0:
        topk_weights, topk_local_idx = weights.topk(actual_k, dim=1)  # (P, actual_k)
        topk_global_idx = sorted_indices[topk_local_idx]  # (P, actual_k) → global indices

        contrib_weights[:, :actual_k] = topk_weights
        contrib_indices[:, :actual_k] = topk_global_idx

        # Dominant = highest weight contributor
        dominant_index = topk_global_idx[:, 0]  # already sorted by topk descending

    return (rendered_color, rendered_depth, final_T,
            contrib_weights, contrib_indices, dominant_index)


def render_with_attribution(
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
    top_k: int = 8,
) -> Dict[str, torch.Tensor]:
    """Full rendering pipeline with per-Gaussian attribution.

    Extends the standard render() to also produce:
    - 'dominant_index': (H, W) index of dominant Gaussian per pixel
    - 'contrib_weights': (H, W, top_k) per-pixel contribution weights
    - 'contrib_indices': (H, W, top_k) per-pixel Gaussian indices

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
        top_k: number of top contributors to track per pixel

    Returns:
        Dict with:
            'color': (H, W, 3)
            'depth': (H, W)
            'transmission': (H, W)
            'dominant_index': (H, W) int64
            'contrib_weights': (H, W, top_k)
            'contrib_indices': (H, W, top_k) int64
    """
    from .projection import (
        world_to_camera, project_to_screen,
        compute_2d_covariance, cov2d_to_conic, compute_radii
    )
    from .rasterizer import tile_gaussians, sort_by_depth

    if bg_color is None:
        bg_color = torch.zeros(3, device=means3D.device)

    N = means3D.shape[0]
    device = means3D.device

    # 1. Transform to camera space
    means_cam = world_to_camera(means3D, extrinsics)

    # 2. Frustum culling
    valid = means_cam[:, 2] > 0.1
    if not valid.any():
        return {
            'color': bg_color.unsqueeze(0).unsqueeze(0).expand(image_height, image_width, 3),
            'depth': torch.zeros(image_height, image_width, device=device),
            'transmission': torch.ones(image_height, image_width, device=device),
            'dominant_index': torch.full((image_height, image_width), -1,
                                         dtype=torch.long, device=device),
            'contrib_weights': torch.zeros(image_height, image_width, top_k, device=device),
            'contrib_indices': torch.full((image_height, image_width, top_k), -1,
                                          dtype=torch.long, device=device),
        }

    # 3. Project to screen
    means2D, depths = project_to_screen(means_cam, intrinsics)

    # 4. Compute 2D covariance
    cov2D = compute_2d_covariance(cov3D, means_cam, extrinsics, intrinsics)
    conics = cov2d_to_conic(cov2D)
    radii = compute_radii(cov2D)

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
    global_indices = torch.where(in_screen)[0]

    # 7. Render each tile with attribution
    color_image = torch.zeros(image_height, image_width, 3, device=device)
    depth_image = torch.zeros(image_height, image_width, device=device)
    trans_image = torch.ones(image_height, image_width, device=device)
    dominant_image = torch.full((image_height, image_width), -1,
                                dtype=torch.long, device=device)
    cw_image = torch.zeros(image_height, image_width, top_k, device=device)
    ci_image = torch.full((image_height, image_width, top_k), -1,
                           dtype=torch.long, device=device)

    for (ty, tx), local_indices in tiles.items():
        tile_global_idx = global_indices[local_indices]
        sorted_idx = sort_by_depth(depths, tile_global_idx)

        py_start = ty * tile_size
        px_start = tx * tile_size
        py_end = min(py_start + tile_size, image_height)
        px_end = min(px_start + tile_size, image_width)

        ys = torch.arange(py_start, py_end, device=device, dtype=torch.float32)
        xs = torch.arange(px_start, px_end, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        pixel_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        (tile_color, tile_depth, tile_T,
         tile_cw, tile_ci, tile_dom) = rasterize_pixels_with_attribution(
            pixel_coords, sorted_idx, means2D, conics, colors,
            opacities, depths, bg_color, N, top_k=top_k
        )

        h = py_end - py_start
        w = px_end - px_start
        color_image[py_start:py_end, px_start:px_end] = tile_color.reshape(h, w, 3)
        depth_image[py_start:py_end, px_start:px_end] = tile_depth.reshape(h, w)
        trans_image[py_start:py_end, px_start:px_end] = tile_T.reshape(h, w)
        dominant_image[py_start:py_end, px_start:px_end] = tile_dom.reshape(h, w)
        cw_image[py_start:py_end, px_start:px_end] = tile_cw.reshape(h, w, top_k)
        ci_image[py_start:py_end, px_start:px_end] = tile_ci.reshape(h, w, top_k)

    return {
        'color': color_image,
        'depth': depth_image,
        'transmission': trans_image,
        'dominant_index': dominant_image,
        'contrib_weights': cw_image,
        'contrib_indices': ci_image,
    }


def compute_gaussian_statistics(
    rendered_color: torch.Tensor,
    rendered_depth: torch.Tensor,
    gt_color: torch.Tensor,
    gt_depth: torch.Tensor,
    contrib_weights: torch.Tensor,
    contrib_indices: torch.Tensor,
    n_gaussians: int,
    cov2D: Optional[torch.Tensor] = None,
    epsilon: float = 1e-7,
) -> Dict[str, torch.Tensor]:
    """Compute per-Gaussian statistics from pixel-level attribution.

    This is the core attribution function that converts pixel errors
    to per-Gaussian error metrics.

    Math:
        E^color_i = Σ_u w_{u,i} · e_c(u) / (Σ_u w_{u,i} + ε)
        E^depth_i = Σ_u w_{u,i} · e_d(u) / (Σ_u w_{u,i} + ε)
        V_i = (1/N_pixels) · Σ_u 1[w_{u,i} > ε]
        Influence_i = Σ_u w_{u,i}  (alpha contribution mass)
        Area_i = π · √(det(cov2D_i))  (geometric projected screen area)

    Args:
        rendered_color: (H, W, 3) rendered color image
        rendered_depth: (H, W) rendered depth
        gt_color: (H, W, 3) ground truth color
        gt_depth: (H, W) ground truth depth
        contrib_weights: (H, W, top_k) contribution weights
        contrib_indices: (H, W, top_k) Gaussian indices
        n_gaussians: total number of Gaussians
        cov2D: optional (N, 2, 2) projected 2D covariance matrices
        epsilon: small constant for numerical stability

    Returns:
        Dict with:
            'color_error': (N,) per-Gaussian weighted color error
            'depth_error': (N,) per-Gaussian weighted depth error
            'visibility': (N,) fraction of pixels each Gaussian contributes to
            'influence_mass': (N,) total alpha-weighted contribution weight
            'projected_area': (N,) true geometric projected area in pixels²
            'contribution_mass': (N,) alias for influence_mass
            'screen_area': (N,) backward compatibility alias
            'visibility_mask': (N,) bool, True if Gaussian was visible
    """
    H, W = rendered_depth.shape
    device = rendered_color.device
    top_k = contrib_weights.shape[-1]
    total_pixels = H * W

    # Compute pixel-level errors
    color_err = (rendered_color - gt_color).abs().mean(dim=-1)  # (H, W)
    depth_err = (rendered_depth - gt_depth).abs()  # (H, W)
    depth_valid = (gt_depth > 0) & (~torch.isnan(gt_depth))
    depth_err = depth_err * depth_valid.float()

    # Flatten for efficient scatter operations
    flat_indices = contrib_indices.reshape(-1)  # (H*W*top_k,)
    flat_weights = contrib_weights.reshape(-1)  # (H*W*top_k,)

    # Expand pixel errors to match flattened indices
    flat_color_err = color_err.unsqueeze(-1).expand(H, W, top_k).reshape(-1)
    flat_depth_err = depth_err.unsqueeze(-1).expand(H, W, top_k).reshape(-1)

    # Filter out invalid indices (-1 from padding)
    valid_mask = (flat_indices >= 0) & (flat_indices < n_gaussians)
    valid_indices = flat_indices[valid_mask]
    valid_weights = flat_weights[valid_mask]
    valid_color_err = flat_color_err[valid_mask]
    valid_depth_err = flat_depth_err[valid_mask]

    # Total contribution weight per Gaussian: W_i = Σ_u w_{u,i}
    total_weight = torch.zeros(n_gaussians, device=device)
    total_weight.scatter_add_(0, valid_indices, valid_weights)

    # Weighted color error: Σ_u w_{u,i} · e_c(u)
    weighted_color_err = torch.zeros(n_gaussians, device=device)
    weighted_color_err.scatter_add_(0, valid_indices, valid_weights * valid_color_err)

    # Weighted depth error: Σ_u w_{u,i} · e_d(u)
    weighted_depth_err = torch.zeros(n_gaussians, device=device)
    weighted_depth_err.scatter_add_(0, valid_indices, valid_weights * valid_depth_err)

    # Pixel count per Gaussian (for visibility)
    pixel_count = torch.zeros(n_gaussians, device=device)
    contributing = (valid_weights > epsilon).float()
    pixel_count.scatter_add_(0, valid_indices, contributing)

    # Normalize: E_i = weighted_err / (total_weight + ε)
    per_gaussian_color_err = weighted_color_err / (total_weight + epsilon)
    per_gaussian_depth_err = weighted_depth_err / (total_weight + epsilon)

    # Visibility: fraction of total pixels
    visibility = pixel_count / (total_pixels + epsilon)

    # Influence mass: total alpha-weighted contribution Σ_u w_{u,i}
    influence_mass = total_weight

    # Projected Area: True geometric screen-space area
    if cov2D is not None and cov2D.shape[0] == n_gaussians:
        projected_area = compute_projected_area(cov2D)
    else:
        projected_area = influence_mass

    # Visibility mask: Gaussian contributed to at least one pixel
    visibility_mask = pixel_count > 0

    return {
        'color_error': per_gaussian_color_err,
        'depth_error': per_gaussian_depth_err,
        'visibility': visibility,
        'influence_mass': influence_mass,
        'projected_area': projected_area,
        'contribution_mass': influence_mass,
        'screen_area': influence_mass,  # backward compat alias
        'visibility_mask': visibility_mask,
        'pixel_count': pixel_count,
        'total_weight': total_weight,
    }


def compute_projected_area(
    cov2D: torch.Tensor,
) -> torch.Tensor:
    """Compute true geometric projected screen-space area of each Gaussian.

    Uses the ellipse area formula:
        A_i = π · r_x · r_y
    where r_x, r_y are the semi-axes (square roots of eigenvalues of cov2D).

    For a 2x2 symmetric matrix [[a, b], [b, c]]:
        λ₁,₂ = (a+c)/2 ± √(((a-c)/2)² + b²)
        r_x = √λ₁,  r_y = √λ₂
        A = π · r_x · r_y = π · √(det(cov2D))

    Args:
        cov2D: (N, 2, 2) projected 2D covariance matrices

    Returns:
        areas: (N,) projected area in pixels²
    """
    import math

    a = cov2D[:, 0, 0]
    b = cov2D[:, 0, 1]
    c = cov2D[:, 1, 1]

    trace = a + c
    det = a * c - b * b

    discriminant = (trace * trace / 4 - det).clamp(min=0)
    sqrt_disc = torch.sqrt(discriminant)

    lambda1 = trace / 2 + sqrt_disc
    lambda2 = (trace / 2 - sqrt_disc).clamp(min=0)

    r_x = torch.sqrt(lambda1.clamp(min=1e-8))
    r_y = torch.sqrt(lambda2.clamp(min=1e-8))

    return math.pi * r_x * r_y


def normalize_importance_components(
    components: Dict[str, torch.Tensor],
    method: str = 'zscore',
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Normalize importance components to comparable scales.

    Three methods (for ablation study):
        Version A (raw): No normalization, raw values
        Version B (zscore): Per-feature z-score normalization
            x̂ = (x - μ) / (σ + ε)
        Version C (robust): Robust normalization using median/MAD
            x̂ = (x - median(x)) / (MAD(x) + ε)

    Args:
        components: dict mapping component name → (N,) tensor
        method: 'raw', 'zscore', or 'robust'
        epsilon: numerical stability constant

    Returns:
        Normalized components dict (same keys)
    """
    normalized = {}

    for name, values in components.items():
        if method == 'raw':
            normalized[name] = values
        elif method == 'zscore':
            mu = values.mean()
            sigma = values.std()
            normalized[name] = (values - mu) / (sigma + epsilon)
        elif method == 'robust':
            median = values.median()
            mad = (values - median).abs().median()
            normalized[name] = (values - median) / (mad + epsilon)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    return normalized
