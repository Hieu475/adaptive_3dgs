import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class GaussianModel(nn.Module):
    """
    Gaussian representation for 3D Gaussian Splatting.
    
    Stores positions (μ ∈ R³), scales, rotations (quaternions), opacities (α), 
    SH coefficients, normals, and confidence scores.
    """
    
    # State tracking constants
    UNSTABLE = 0
    STABLE = 1
    FROZEN = 2
    PRUNED = 3

    def __init__(self, sh_degree: int = 3):
        super().__init__()
        self.sh_degree = sh_degree
        
        # Tensors storing properties
        # TODO: Initialize properly
        self._xyz = torch.empty(0, 3)
        self._scaling = torch.empty(0, 3)
        self._rotation = torch.empty(0, 4)
        self._opacity = torch.empty(0, 1)
        self._features_dc = torch.empty(0, 1, 3)
        self._features_rest = torch.empty(0, 15, 3)
        self._normals = torch.empty(0, 3)
        self._confidence = torch.empty(0, 1)
        self._state = torch.empty(0, 1, dtype=torch.long)
        
    def build_covariance(self) -> torch.Tensor:
        """
        Builds the 3D covariance matrix using R·S·Sᵀ·Rᵀ decomposition.
        """
        # TODO: implement R*S*S^T*R^T
        pass
        
    def get_active_gaussians(self) -> torch.Tensor:
        """
        Returns mask or indices of active (non-pruned) Gaussians.
        """
        return self._state != self.PRUNED

    def add_gaussians(self, new_xyz: torch.Tensor, new_features: Dict[str, torch.Tensor]):
        """
        Adds new Gaussians.
        """
        # TODO: concatenate tensors
        pass

    def prune_gaussians(self, mask: torch.Tensor):
        """
        Prunes Gaussians based on boolean mask.
        """
        # TODO: set state to PRUNED or remove from memory
        pass
