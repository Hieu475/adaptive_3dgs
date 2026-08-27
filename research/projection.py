"""Differentiable projection from 3D Gaussians to 2D screen space.

Key math:
- World to camera: x_c = W @ x (4x4 extrinsic matrix)
- Perspective projection: p = π(x_c) = K @ x_c (3x3 intrinsic matrix)
- 2D covariance via Jacobian: Σ₂D = J @ W @ Σ @ Wᵀ @ Jᵀ
  where J is the Jacobian of the perspective projection.
"""
import torch
from typing import Tuple, Optional


def world_to_camera(
    points: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    """Transform points from world space to camera space.
    
    Args:
        points: (N, 3) world-space positions
        extrinsics: (4, 4) world-to-camera transform [R|t]
    
    Returns:
        (N, 3) camera-space positions
    """
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]   # (3,)
    return points @ R.T + t.unsqueeze(0)  # (N, 3)


def project_to_screen(
    points_cam: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perspective projection from camera space to 2D pixel coordinates.
    
    Args:
        points_cam: (N, 3) camera-space positions (x_c, y_c, z_c)
        intrinsics: (3, 3) camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
    
    Returns:
        points_2d: (N, 2) pixel coordinates (u, v)
        depths: (N,) depth values (z_c)
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    z = points_cam[:, 2].clamp(min=1e-6)  # Avoid division by zero
    u = fx * points_cam[:, 0] / z + cx
    v = fy * points_cam[:, 1] / z + cy
    
    return torch.stack([u, v], dim=-1), z


def compute_projection_jacobian(
    points_cam: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Compute the Jacobian of perspective projection.
    
    J = ∂π/∂x_c = [[fx/z, 0, -fx*x/z²],
                    [0, fy/z, -fy*y/z²]]
    
    Args:
        points_cam: (N, 3) camera-space positions
        intrinsics: (3, 3) intrinsic matrix
    
    Returns:
        J: (N, 2, 3) Jacobian matrices
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2].clamp(min=1e-6)
    z2 = z * z
    
    J = torch.zeros(points_cam.shape[0], 2, 3, 
                    device=points_cam.device, dtype=points_cam.dtype)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x / z2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y / z2
    
    return J


def compute_2d_covariance(
    cov3D: torch.Tensor,
    points_cam: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Compute 2D screen-space covariance from 3D covariance.
    
    Σ₂D = J @ W[:3,:3] @ Σ₃D @ W[:3,:3]ᵀ @ Jᵀ
    
    where:
    - J is the Jacobian of perspective projection (2x3)
    - W[:3,:3] is the rotation part of the extrinsic matrix (3x3)
    - Σ₃D is the 3D covariance in world space (3x3)
    
    Args:
        cov3D: (N, 3, 3) world-space 3D covariance matrices
        points_cam: (N, 3) camera-space positions (for Jacobian computation)
        extrinsics: (4, 4) world-to-camera transform
        intrinsics: (3, 3) camera intrinsic matrix
    
    Returns:
        cov2D: (N, 2, 2) screen-space 2D covariance matrices
    """
    W = extrinsics[:3, :3]  # (3, 3)
    J = compute_projection_jacobian(points_cam, intrinsics)  # (N, 2, 3)
    
    # T = J @ W  -> (N, 2, 3)
    T = torch.bmm(J, W.unsqueeze(0).expand(cov3D.shape[0], -1, -1))  # (N, 2, 3)
    
    # Σ₂D = T @ Σ₃D @ Tᵀ  -> (N, 2, 2)
    cov2D = torch.bmm(torch.bmm(T, cov3D), T.transpose(1, 2))
    
    # Add small regularization for numerical stability
    cov2D[:, 0, 0] = cov2D[:, 0, 0] + 0.3
    cov2D[:, 1, 1] = cov2D[:, 1, 1] + 0.3
    
    return cov2D


def cov2d_to_conic(cov2D: torch.Tensor) -> torch.Tensor:
    """Convert 2D covariance matrix to conic (inverse) representation.
    
    For a 2x2 symmetric matrix [[a, b], [b, c]], the inverse is:
    [[c, -b], [-b, a]] / det  where det = a*c - b*b
    
    We store as (a_inv, b_inv, c_inv) = (c/det, -b/det, a/det)
    
    Args:
        cov2D: (N, 2, 2) covariance matrices
    
    Returns:
        conics: (N, 3) conic parameters
    """
    a = cov2D[:, 0, 0]
    b = cov2D[:, 0, 1]
    c = cov2D[:, 1, 1]
    
    det = a * c - b * b
    det = det.clamp(min=1e-8)
    
    return torch.stack([c / det, -b / det, a / det], dim=-1)


def compute_bounding_boxes(
    means2D: torch.Tensor,
    cov2D: torch.Tensor,
    n_sigma: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute axis-aligned bounding boxes from 2D covariance.
    
    Uses eigenvalue decomposition to find the extent of the ellipse,
    then creates an AABB that contains the n_sigma ellipse.
    
    Args:
        means2D: (N, 2) screen-space centers
        cov2D: (N, 2, 2) screen-space covariance
        n_sigma: Number of standard deviations for bounding box
    
    Returns:
        bb_min: (N, 2) top-left corners
        bb_max: (N, 2) bottom-right corners
    """
    # Eigenvalues of 2x2 symmetric matrix
    a = cov2D[:, 0, 0]
    b = cov2D[:, 0, 1]
    c = cov2D[:, 1, 1]
    
    # λ = (a+c)/2 ± sqrt(((a-c)/2)² + b²)
    mid = 0.5 * (a + c)
    disc = torch.sqrt(((a - c) * 0.5) ** 2 + b ** 2 + 1e-8)
    lambda_max = mid + disc  # Larger eigenvalue
    
    # Radius = n_sigma * sqrt(lambda_max)
    radius = n_sigma * torch.sqrt(lambda_max.clamp(min=1e-8))
    
    bb_min = means2D - radius.unsqueeze(-1)
    bb_max = means2D + radius.unsqueeze(-1)
    
    return bb_min, bb_max


def compute_radii(cov2D: torch.Tensor, n_sigma: float = 3.0) -> torch.Tensor:
    """Compute screen-space radii from 2D covariance for tile culling.
    
    Args:
        cov2D: (N, 2, 2) covariance matrices
        n_sigma: number of std devs
    
    Returns:
        radii: (N,) pixel radii
    """
    a = cov2D[:, 0, 0]
    b = cov2D[:, 0, 1]
    c = cov2D[:, 1, 1]
    
    mid = 0.5 * (a + c)
    disc = torch.sqrt(((a - c) * 0.5) ** 2 + b ** 2 + 1e-8)
    lambda_max = mid + disc
    
    return (n_sigma * torch.sqrt(lambda_max.clamp(min=1e-8))).ceil().int()
