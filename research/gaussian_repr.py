"""3D Gaussian Splatting representation.

Each Gaussian G_i = (μ_i, Σ_i, α_i, SH_i) where:
- μ ∈ R³: position
- Σ = R·S·S^T·R^T: covariance (parameterized by scale s ∈ R³ and rotation q ∈ R⁴ quaternion)
- α ∈ [0,1]: opacity  
- SH: spherical harmonics coefficients for view-dependent color
- n ∈ R³: surface normal
- confidence ∈ [0,1]: importance score
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math


class GaussianState:
    """State tracking constants for Gaussian lifecycle."""
    UNSTABLE = 0
    STABLE = 1
    FROZEN = 2
    PRUNED = 3


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
    
    Args:
        q: Quaternions of shape (..., 4), format (w, x, y, z)
    
    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    # Normalize quaternion
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    
    return R


def build_scaling_matrix(scales: torch.Tensor) -> torch.Tensor:
    """Build diagonal scaling matrix from scale vector.
    
    Args:
        scales: Scale vectors of shape (..., 3)
    
    Returns:
        Diagonal matrices of shape (..., 3, 3)
    """
    S = torch.zeros(*scales.shape[:-1], 3, 3, device=scales.device, dtype=scales.dtype)
    S[..., 0, 0] = scales[..., 0]
    S[..., 1, 1] = scales[..., 1]
    S[..., 2, 2] = scales[..., 2]
    return S


class GaussianModel(nn.Module):
    """3D Gaussian Splatting scene representation.
    
    Manages a collection of 3D Gaussians with differentiable parameters
    for position, shape (via scale+rotation), opacity, appearance (SH),
    normals, and confidence scores.
    """
    
    def __init__(self, sh_degree: int = 3, device: str = 'cpu'):
        super().__init__()
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.device = device
        self._num_gaussians = 0
        
        # Learnable parameters (registered as nn.Parameter for gradient tracking)
        self._xyz = nn.Parameter(torch.empty(0, 3, device=device))
        self._scaling = nn.Parameter(torch.empty(0, 3, device=device))  # log-scale
        self._rotation = nn.Parameter(torch.empty(0, 4, device=device))  # quaternion (w,x,y,z)
        self._opacity = nn.Parameter(torch.empty(0, 1, device=device))  # logit-space opacity
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3, device=device))  # DC component
        self._features_rest = nn.Parameter(torch.empty(0, self.num_sh_coeffs - 1, 3, device=device))
        self._normals = nn.Parameter(torch.empty(0, 3, device=device))
        
        # Non-differentiable state
        self.register_buffer('_confidence', torch.empty(0, 1, device=device))
        self.register_buffer('_state', torch.empty(0, dtype=torch.long, device=device))
    
    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]
    
    @property
    def positions(self) -> torch.Tensor:
        return self._xyz
    
    @property
    def opacities(self) -> torch.Tensor:
        """Sigmoid-activated opacities in [0, 1]."""
        return torch.sigmoid(self._opacity)
    
    @property
    def scales(self) -> torch.Tensor:
        """Exp-activated scales (always positive)."""
        return torch.exp(self._scaling)
    
    @property
    def rotations(self) -> torch.Tensor:
        """Normalized quaternions."""
        return self._rotation / (self._rotation.norm(dim=-1, keepdim=True) + 1e-8)
    
    def get_colors(self, directions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get colors. If directions provided, evaluate SH; otherwise return DC.
        
        Args:
            directions: Optional viewing directions (..., 3)
        Returns:
            Colors (..., 3) in [0, 1]
        """
        if directions is None or self.sh_degree == 0:
            return torch.sigmoid(self._features_dc[:, 0, :])  # (N, 3)
        # For higher SH degrees, a full SH evaluation would go here
        # For now, return DC component
        return torch.sigmoid(self._features_dc[:, 0, :])
    
    def build_covariance(self) -> torch.Tensor:
        """Build 3D covariance matrices: Σ = R·S·S^T·R^T.
        
        The decomposition ensures Σ is always symmetric positive semi-definite.
        Scale is stored in log-space and exponentiated to ensure positivity.
        Rotation is stored as quaternion and converted to rotation matrix.
        
        Returns:
            Covariance matrices of shape (N, 3, 3)
        """
        R = quaternion_to_rotation_matrix(self._rotation)  # (N, 3, 3)
        s = self.scales  # (N, 3) - always positive via exp
        S = build_scaling_matrix(s)  # (N, 3, 3)
        
        # Σ = R @ S @ S^T @ R^T
        RS = torch.bmm(R, S)  # (N, 3, 3)
        cov = torch.bmm(RS, RS.transpose(1, 2))  # (N, 3, 3)
        return cov
    
    def get_optimization_subset(
        self,
        optimize_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract optimization subset containing only active Gaussians (M <= N).
        
        Eliminates O(N) full-size tensor copies. Active tensors are strictly of size (M, ...).
        
        Args:
            optimize_mask: (N,) boolean tensor of Gaussians selected for optimization.
            
        Returns:
            Dict containing active indices, active sliced parameters, and active 3D attributes.
        """
        N = self.num_gaussians
        active_idx = torch.where(optimize_mask[:N])[0]
        M = len(active_idx)
        
        if M == 0:
            return {
                'indices': active_idx,
                'xyz': torch.empty(0, 3, device=self.device),
                'scaling': torch.empty(0, 3, device=self.device),
                'rotation': torch.empty(0, 4, device=self.device),
                'opacity': torch.empty(0, 1, device=self.device),
                'features_dc': torch.empty(0, 1, 3, device=self.device),
                'features_rest': torch.empty(0, self.num_sh_coeffs - 1, 3, device=self.device),
                'normals': torch.empty(0, 3, device=self.device),
                'means3D': torch.empty(0, 3, device=self.device),
                'cov3D': torch.empty(0, 3, 3, device=self.device),
                'colors': torch.empty(0, 3, device=self.device),
                'opacities': torch.empty(0, device=self.device),
            }
            
        # Sliced active parameters
        xyz = self._xyz[active_idx]
        scaling = self._scaling[active_idx]
        rotation = self._rotation[active_idx]
        opacity = self._opacity[active_idx]
        features_dc = self._features_dc[active_idx]
        features_rest = self._features_rest[active_idx] if self._features_rest.numel() > 0 else torch.empty(0, device=self.device)
        normals = self._normals[active_idx] if self._normals.numel() > 0 else torch.empty(0, 3, device=self.device)
        
        # Build active-only covariance (M, 3, 3)
        R_act = quaternion_to_rotation_matrix(rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-8))
        S_act = build_scaling_matrix(torch.exp(scaling))
        M_act = torch.bmm(R_act, S_act)
        cov3D = torch.bmm(M_act, M_act.transpose(1, 2))
        
        colors = torch.sigmoid(features_dc[:, 0, :])
        opacities = torch.sigmoid(opacity).squeeze(-1)
        
        return {
            'indices': active_idx,
            'xyz': xyz,
            'scaling': scaling,
            'rotation': rotation,
            'opacity': opacity,
            'features_dc': features_dc,
            'features_rest': features_rest,
            'normals': normals,
            'means3D': xyz,
            'cov3D': cov3D,
            'colors': colors,
            'opacities': opacities,
        }

    def get_active_mask(self) -> torch.Tensor:
        """Returns boolean mask of active (non-pruned) Gaussians."""
        return self._state != GaussianState.PRUNED
    
    def get_active_gaussians(self) -> Dict[str, torch.Tensor]:
        """Returns dict of properties for all active Gaussians."""
        mask = self.get_active_mask()
        return {
            'xyz': self._xyz[mask],
            'scaling': self._scaling[mask],
            'rotation': self._rotation[mask],
            'opacity': self._opacity[mask],
            'features_dc': self._features_dc[mask],
            'features_rest': self._features_rest[mask],
            'normals': self._normals[mask],
            'confidence': self._confidence[mask],
            'state': self._state[mask],
        }
    
    def initialize_from_points(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        initial_scale: float = 0.01,
        initial_opacity: float = 0.1,
        initial_confidence: float = 0.5,
    ):
        """Initialize Gaussians from a 3D point cloud.
        
        Args:
            points: (N, 3) positions
            colors: (N, 3) RGB colors in [0,1], optional
            normals: (N, 3) surface normals, optional
            initial_scale: initial isotropic scale
            initial_opacity: initial opacity (before sigmoid)
            initial_confidence: initial confidence score
        """
        N = points.shape[0]
        device = points.device
        
        self._xyz = nn.Parameter(points.clone())
        self._scaling = nn.Parameter(torch.full((N, 3), math.log(initial_scale), device=device))
        # Identity quaternion (w=1, x=0, y=0, z=0)
        rot = torch.zeros(N, 4, device=device)
        rot[:, 0] = 1.0
        self._rotation = nn.Parameter(rot)
        # Inverse sigmoid for initial opacity
        inv_sig = math.log(initial_opacity / (1.0 - initial_opacity + 1e-8))
        self._opacity = nn.Parameter(torch.full((N, 1), inv_sig, device=device))
        
        if colors is not None:
            # Store DC SH component (inverse sigmoid)
            colors_clamped = colors.clamp(1e-4, 1.0 - 1e-4)
            dc = torch.log(colors_clamped / (1.0 - colors_clamped)).unsqueeze(1)  # (N,1,3)
        else:
            dc = torch.zeros(N, 1, 3, device=device)
        self._features_dc = nn.Parameter(dc)
        self._features_rest = nn.Parameter(torch.zeros(N, self.num_sh_coeffs - 1, 3, device=device))
        
        if normals is not None:
            self._normals = nn.Parameter(normals.clone())
        else:
            self._normals = nn.Parameter(torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(N, -1).clone())
        
        self._confidence = torch.full((N, 1), initial_confidence, device=device)
        self._state = torch.full((N,), GaussianState.UNSTABLE, dtype=torch.long, device=device)
        self._num_gaussians = N
    
    @torch.no_grad()
    def add_gaussians(self, new_params: Dict[str, torch.Tensor]):
        """Add new Gaussians to the model.
        
        Args:
            new_params: Dict with keys matching parameter names.
                Required: 'xyz'. Optional: 'scaling', 'rotation', 'opacity',
                'features_dc', 'features_rest', 'normals', 'confidence'
        """
        new_xyz = new_params['xyz']
        N_new = new_xyz.shape[0]
        device = new_xyz.device
        
        def _cat(old, new):
            return nn.Parameter(torch.cat([old.data, new], dim=0))
        
        self._xyz = _cat(self._xyz, new_xyz)
        self._scaling = _cat(self._scaling, new_params.get(
            'scaling', torch.full((N_new, 3), math.log(0.01), device=device)))
        self._rotation = _cat(self._rotation, new_params.get(
            'rotation', torch.tensor([[1.0, 0, 0, 0]], device=device).expand(N_new, -1).clone()))
        inv_sig = math.log(0.1 / 0.9)
        self._opacity = _cat(self._opacity, new_params.get(
            'opacity', torch.full((N_new, 1), inv_sig, device=device)))
        self._features_dc = _cat(self._features_dc, new_params.get(
            'features_dc', torch.zeros(N_new, 1, 3, device=device)))
        self._features_rest = _cat(self._features_rest, new_params.get(
            'features_rest', torch.zeros(N_new, self.num_sh_coeffs - 1, 3, device=device)))
        self._normals = _cat(self._normals, new_params.get(
            'normals', torch.tensor([[0., 0., 1.]], device=device).expand(N_new, -1).clone()))
        
        self._confidence = torch.cat([
            self._confidence, torch.full((N_new, 1), 0.5, device=device)], dim=0)
        self._state = torch.cat([
            self._state, torch.full((N_new,), GaussianState.UNSTABLE, dtype=torch.long, device=device)], dim=0)
        self._num_gaussians = self._xyz.shape[0]
    
    @torch.no_grad()
    def prune_gaussians(self, mask: torch.Tensor):
        """Mark Gaussians for pruning (soft delete via state flag).
        
        Args:
            mask: Boolean tensor (N,) - True = prune this Gaussian
        """
        self._state[mask] = GaussianState.PRUNED
    
    @torch.no_grad()
    def compact(self):
        """Remove pruned Gaussians from memory (hard delete)."""
        keep = self._state != GaussianState.PRUNED
        
        self._xyz = nn.Parameter(self._xyz.data[keep])
        self._scaling = nn.Parameter(self._scaling.data[keep])
        self._rotation = nn.Parameter(self._rotation.data[keep])
        self._opacity = nn.Parameter(self._opacity.data[keep])
        self._features_dc = nn.Parameter(self._features_dc.data[keep])
        self._features_rest = nn.Parameter(self._features_rest.data[keep])
        self._normals = nn.Parameter(self._normals.data[keep])
        self._confidence = self._confidence[keep]
        self._state = self._state[keep]
        self._num_gaussians = self._xyz.shape[0]
