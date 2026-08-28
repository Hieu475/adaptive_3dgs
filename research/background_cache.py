"""Frozen Background Cache for True Selective Optimization (R21/R29).

Architecture:
    1. Render Frozen Gaussians (N - M elements) once per keyframe / schedule update:
       Frozen Set → C_f (color), D_f (depth), T_f (transmittance), A_f (accumulated opacity).
    2. Render Active Gaussians (M elements, M << N) with autograd gradient tracking.
    3. Exact Depth-Aware Alpha Compositing:
       Merges active and frozen contributions in strict front-to-back depth order:
           C(u) = C_active(u) + T_active(u) · C_frozen(u)
           D(u) = D_active(u) + T_active(u) · D_frozen(u)
       where frozen Gaussians are completely detached (zero autograd graph tape)
       and only active Gaussians compute gradients.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any


class FrozenBackgroundCache:
    """Cache holding rendered representations of non-active (frozen) Gaussians."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.is_valid = False
        self.color: Optional[torch.Tensor] = None
        self.depth: Optional[torch.Tensor] = None
        self.transmission: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None
        self.frozen_indices: Optional[torch.Tensor] = None
        self.n_frozen: int = 0
        
    def invalidate(self):
        """Invalidate the cache when camera moves or active selection changes."""
        self.is_valid = False
        self.color = None
        self.depth = None
        self.transmission = None
        self.alpha = None
        self.frozen_indices = None
        self.n_frozen = 0

    @torch.no_grad()
    def build_cache(
        self,
        model,
        frozen_mask: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
        tile_size: int = 16,
        bg_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Render and cache the frozen Gaussian subset (no gradient tracking)."""
        if bg_color is None:
            bg_color = torch.zeros(3, device=self.device)
            
        N = model.num_gaussians
        frozen_idx = torch.where(frozen_mask[:N])[0]
        self.n_frozen = len(frozen_idx)
        self.frozen_indices = frozen_idx
        
        if self.n_frozen == 0:
            self.color = bg_color.unsqueeze(0).unsqueeze(0).expand(image_height, image_width, 3).clone()
            self.depth = torch.zeros((image_height, image_width), device=self.device)
            self.transmission = torch.ones((image_height, image_width), device=self.device)
            self.alpha = torch.zeros((image_height, image_width), device=self.device)
            self.is_valid = True
            return {
                'color': self.color,
                'depth': self.depth,
                'transmission': self.transmission,
                'alpha': self.alpha
            }
            
        # Extract frozen parameters (detached)
        frozen_pos = model.positions[frozen_idx].detach()
        frozen_cov = model.build_covariance()[frozen_idx].detach()
        frozen_col = model.get_colors()[frozen_idx].detach()
        frozen_op = model.opacities.squeeze(-1)[frozen_idx].detach()
        
        from .rasterizer import render as rasterize_scene
        frozen_render = rasterize_scene(
            means3D=frozen_pos,
            cov3D=frozen_cov,
            colors=frozen_col,
            opacities=frozen_op,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_width=image_width,
            image_height=image_height,
            bg_color=bg_color,
            tile_size=tile_size
        )
        
        self.color = frozen_render['color']
        self.depth = frozen_render['depth']
        self.transmission = frozen_render['transmission']
        self.alpha = 1.0 - self.transmission
        self.is_valid = True
        
        return {
            'color': self.color,
            'depth': self.depth,
            'transmission': self.transmission,
            'alpha': self.alpha
        }

    def composite_with_active(
        self,
        active_subset: Dict[str, torch.Tensor],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
        tile_size: int = 16,
        bg_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Composite active Gaussians with the cached frozen background.
        
        Differentiable pass:
        - Only active parameters in `active_subset` generate gradient nodes.
        - Blends through the frozen background transmission channel:
            C_final(u) = C_active(u) + T_active(u) · C_frozen(u)
            D_final(u) = D_active(u) + T_active(u) · D_frozen(u)
        """
        if bg_color is None:
            bg_color = torch.zeros(3, device=self.device)
            
        M = len(active_subset.get('indices', []))
        if M == 0 and self.is_valid and self.color is not None:
            return {
                'color': self.color,
                'depth': self.depth,
                'transmission': self.transmission,
                'alpha': self.alpha
            }
            
        # Render active subset with gradient tracking
        from .rasterizer import render as rasterize_scene
        active_render = rasterize_scene(
            means3D=active_subset['means3D'],
            cov3D=active_subset['cov3D'],
            colors=active_subset['colors'],
            opacities=active_subset['opacities'],
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_width=image_width,
            image_height=image_height,
            bg_color=bg_color,
            tile_size=tile_size
        )
        
        if not self.is_valid or self.color is None:
            return active_render
            
        # Alpha Compositing with background cache:
        c_active = active_render['color']
        t_active = active_render['transmission'].unsqueeze(-1)
        d_active = active_render['depth']
        
        composite_color = c_active + t_active * self.color
        composite_depth = d_active + active_render['transmission'] * self.depth
        composite_trans = active_render['transmission'] * self.transmission
        
        return {
            'color': composite_color,
            'depth': composite_depth,
            'transmission': composite_trans,
            'alpha': 1.0 - composite_trans,
            'active_color': c_active,
            'active_depth': d_active
        }
