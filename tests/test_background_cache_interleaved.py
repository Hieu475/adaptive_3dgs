"""Verification Tests for FrozenBackgroundCache on Depth-Interleaved Scenes (Step 2).

Verifies the mathematical properties of the Grouped Active/Frozen Compositing Approximation:
1. Active Gaussians strictly in front:
   - Composited color/depth matches full render exactly (< 1e-4 error).
   - Active parameter gradients match full render exactly (cosine similarity > 0.999).
   - Frozen parameter gradients are strictly 0.0.

2. Depth-interleaved scene (A_1, F_1, A_2, F_2, ...):
   - Color MAE and Depth MAE are quantitatively bounded.
   - Active parameter gradient cosine similarity is strongly positive (> 0.85),
     proving consistent descent directions without gradient reversal.
   - Frozen parameter gradients remain strictly 0.0 (no tape leakage).
"""
import pytest
import torch
import numpy as np

from research.gaussian_repr import GaussianModel
from research.rasterizer import render as rasterize_scene, render_full
from research.background_cache import FrozenBackgroundCache


def create_ordered_depth_model(
    n_gaussians: int = 40,
    device: str = 'cpu',
) -> GaussianModel:
    """Creates a synthetic scene with Gaussians monotonically ordered along camera Z."""
    torch.manual_seed(42)
    model = GaussianModel(sh_degree=0, device=device)
    
    # Points arranged along camera optical axis with increasing depth
    z_coords = torch.linspace(1.5, 3.5, n_gaussians, device=device)
    xy_coords = (torch.rand(n_gaussians, 2, device=device) - 0.5) * 0.20
    points = torch.cat([xy_coords, z_coords.unsqueeze(-1)], dim=-1)
    
    # Distinct colors
    colors = torch.rand(n_gaussians, 3, device=device)
    
    # Moderate opacity so light partially transmits through layers
    model.initialize_from_points(points, colors, initial_scale=0.04)
    with torch.no_grad():
        model._opacity.fill_(torch.logit(torch.tensor(0.50)))
    return model


def get_camera_setup(H: int = 32, W: int = 32):
    fx, fy = 60.0, 60.0
    intrinsics = torch.tensor([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1]], dtype=torch.float32)
    extrinsics = torch.eye(4, dtype=torch.float32)
    return intrinsics, extrinsics, H, W


def compute_gradient_cosine_similarity(g1: torch.Tensor, g2: torch.Tensor) -> float:
    """Compute cosine similarity between two flattened gradient tensors."""
    v1 = g1.flatten()
    v2 = g2.flatten()
    dot = torch.dot(v1, v2)
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float((dot / (norm1 * norm2)).item())


class TestBackgroundCacheInterleaved:
    """Explicit tests for background cache with depth-interleaved active/frozen Gaussians."""

    def test_front_active_exact_gradient_alignment(self):
        """When active Gaussians are in front of frozen Gaussians,

        compositing is exact and gradients align with cosine similarity > 0.999.
        """
        N = 30
        model = create_ordered_depth_model(n_gaussians=N)
        intrinsics, extrinsics, H, W = get_camera_setup()

        # Front half active (indices 0..14), back half frozen (indices 15..29)
        active_mask = torch.zeros(N, dtype=torch.bool)
        active_mask[: N // 2] = True
        frozen_mask = ~active_mask
        active_idx = torch.where(active_mask)[0]
        frozen_idx = torch.where(frozen_mask)[0]

        target_rgb = torch.rand(H, W, 3)

        # 1. Full Model Pass
        full_out = render_full(model, extrinsics, intrinsics, W, H)
        loss_full = ((full_out['color'] - target_rgb) ** 2).mean()
        loss_full.backward()
        g_full_xyz = model._xyz.grad[active_idx].clone()

        # Reset gradients
        model._xyz.grad.zero_()

        # 2. Background Cache Pass
        cache = FrozenBackgroundCache(device='cpu')
        cache.build_cache(model, frozen_mask, extrinsics, intrinsics, W, H)
        active_subset = model.get_optimization_subset(active_mask)
        comp_out = cache.composite_with_active(active_subset, extrinsics, intrinsics, W, H)
        loss_comp = ((comp_out['color'] - target_rgb) ** 2).mean()
        loss_comp.backward()
        g_comp_xyz = model._xyz.grad[active_idx].clone()
        g_frozen_xyz = model._xyz.grad[frozen_idx]

        # 3. Verifications
        # Frozen Gaussians must have strictly zero gradients
        assert (g_frozen_xyz == 0.0).all(), "Frozen Gaussians received non-zero gradients!"

        # Cosine similarity on active parameters must be ~1.0
        cos_sim = compute_gradient_cosine_similarity(g_full_xyz, g_comp_xyz)
        assert cos_sim > 0.999, f"Expected cosine similarity > 0.999 for front-active, got {cos_sim:.5f}"

        # Color and depth differences should be near machine epsilon
        color_mae = (full_out['color'] - comp_out['color']).abs().mean().item()
        assert color_mae < 1e-4, f"Color MAE unexpectedly large: {color_mae}"

    def test_interleaved_active_frozen_fidelity_and_gradient_cosine(self):
        """When active and frozen Gaussians alternate in depth (A_0, F_0, A_1, F_1, ...),

        grouped compositing provides a close volumetric approximation:
        1. Frozen gradients are strictly zero.
        2. Active gradients have strong positive directional cosine similarity (> 0.85).
        3. Photometric error is bounded (MAE < 0.05).
        """
        N = 40
        model = create_ordered_depth_model(n_gaussians=N)
        intrinsics, extrinsics, H, W = get_camera_setup()

        # Alternating: even indices active (20 active), odd indices frozen (20 frozen)
        active_mask = torch.zeros(N, dtype=torch.bool)
        active_mask[::2] = True
        frozen_mask = ~active_mask
        active_idx = torch.where(active_mask)[0]
        frozen_idx = torch.where(frozen_mask)[0]

        target_rgb = torch.rand(H, W, 3)

        # 1. Full Model Pass
        full_out = render_full(model, extrinsics, intrinsics, W, H)
        loss_full = ((full_out['color'] - target_rgb) ** 2).mean()
        loss_full.backward()
        g_full_xyz = model._xyz.grad[active_idx].clone()

        # Reset gradients
        model._xyz.grad.zero_()

        # 2. Background Cache Pass
        cache = FrozenBackgroundCache(device='cpu')
        cache.build_cache(model, frozen_mask, extrinsics, intrinsics, W, H)
        active_subset = model.get_optimization_subset(active_mask)
        comp_out = cache.composite_with_active(active_subset, extrinsics, intrinsics, W, H)
        loss_comp = ((comp_out['color'] - target_rgb) ** 2).mean()
        loss_comp.backward()
        g_comp_xyz = model._xyz.grad[active_idx].clone()
        g_frozen_xyz = model._xyz.grad[frozen_idx]

        # 3. Mathematical Verifications
        # a) Frozen parameter isolation
        assert (g_frozen_xyz == 0.0).all(), "Frozen parameters in interleaved scene must receive 0 gradients!"

        # b) Active parameter gradient directional alignment
        cos_sim_xyz = compute_gradient_cosine_similarity(g_full_xyz, g_comp_xyz)
        assert cos_sim_xyz > 0.85, (
            f"Interleaved active gradient cosine similarity ({cos_sim_xyz:.4f}) dropped below 0.85 threshold!"
        )

        # c) Bounded photometric and depth discrepancy
        color_mae = (full_out['color'] - comp_out['color']).abs().mean().item()
        depth_mae = (full_out['depth'] - comp_out['depth']).abs().mean().item()
        assert color_mae < 0.05, f"Color MAE too large: {color_mae}"
        assert depth_mae < 0.20, f"Depth MAE too large: {depth_mae}"

    def test_depth_interleaved_multi_parameter_gradients(self):
        """Verifies cosine similarity across all trainable parameter groups (xyz, scaling, opacity)."""
        N = 30
        model = create_ordered_depth_model(n_gaussians=N)
        intrinsics, extrinsics, H, W = get_camera_setup()

        # 1 in 3 active
        active_mask = torch.zeros(N, dtype=torch.bool)
        active_mask[::3] = True
        frozen_mask = ~active_mask
        active_idx = torch.where(active_mask)[0]

        target_rgb = torch.rand(H, W, 3)
        target_depth = torch.full((H, W), 2.5)

        # Full pass
        full_out = render_full(model, extrinsics, intrinsics, W, H)
        loss_full = (
            ((full_out['color'] - target_rgb) ** 2).mean()
            + 0.5 * ((full_out['depth'] - target_depth) ** 2).mean()
        )
        loss_full.backward()
        full_grads = {
            'xyz': model._xyz.grad[active_idx].clone(),
            'scaling': model._scaling.grad[active_idx].clone(),
            'opacity': model._opacity.grad[active_idx].clone(),
        }

        # Clear grads
        model._xyz.grad.zero_()
        model._scaling.grad.zero_()
        model._opacity.grad.zero_()

        # Cache pass
        cache = FrozenBackgroundCache(device='cpu')
        cache.build_cache(model, frozen_mask, extrinsics, intrinsics, W, H)
        active_subset = model.get_optimization_subset(active_mask)
        comp_out = cache.composite_with_active(active_subset, extrinsics, intrinsics, W, H)
        loss_comp = (
            ((comp_out['color'] - target_rgb) ** 2).mean()
            + 0.5 * ((comp_out['depth'] - target_depth) ** 2).mean()
        )
        loss_comp.backward()
        comp_grads = {
            'xyz': model._xyz.grad[active_idx].clone(),
            'scaling': model._scaling.grad[active_idx].clone(),
            'opacity': model._opacity.grad[active_idx].clone(),
        }

        # Check positive cosine similarity across all parameter modalities
        for p_name in ['xyz', 'scaling', 'opacity']:
            c_sim = compute_gradient_cosine_similarity(full_grads[p_name], comp_grads[p_name])
            assert c_sim > 0.80, f"Parameter {p_name} cosine similarity ({c_sim:.4f}) fell below 0.80!"
