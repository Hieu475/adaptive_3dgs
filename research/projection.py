import torch
from typing import Tuple

def world_to_camera(points: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Transforms points from world space to camera space.
    """
    # TODO: p_cam = R * p_world + t
    pass

def project_to_screen(points_cam: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Perspective projection from camera space to 2D screen space.
    """
    # TODO: project points using camera matrix
    pass

def compute_2d_covariance(
    means3D: torch.Tensor, 
    cov3D: torch.Tensor, 
    view_matrix: torch.Tensor, 
    proj_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Computes 2D covariance using Jacobian approximation: Σ₂D = J·W·Σ·Wᵀ·Jᵀ
    """
    # TODO: Implement Jacobian approximation
    pass

def compute_bounding_boxes(means2D: torch.Tensor, cov2D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes screen-space bounding boxes for ellipses.
    """
    # TODO: Compute based on eigenvalues of cov2D
    pass
