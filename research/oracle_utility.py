"""Oracle Utility Ground-Truth Experiment Engine for Adaptive 3DGS.

Core research module: measures the TRUE marginal utility of optimizing
each Gaussian by isolating its individual or group contribution.

Mathematical Formulations:
    Ground-Truth Oracle Marginal Utility:
        U_i^{oracle} = (Q(G^{+i}) - Q(G)) / (ΔT_i + ε)

    Where raw measurements are decoupled and logged:
        ΔPSNR_{local, i} = PSNR(G^{+i}) - PSNR(G)
        ΔSSIM_{local, i} = SSIM(G^{+i}) - SSIM(G)
        ΔDepth_{local, i} = DepthL1(G) - DepthL1(G^{+i})
        ΔLoss_{local, i} = Loss(G) - Loss(G^{+i})
        ΔT_i = measured_trial_cost_ms

    Multi-Component Oracle Utilities:
        U_i^{rgb} = ΔPSNR_i / (ΔT_i + ε)
        U_i^{depth} = ΔDepth_i / (ΔT_i + ε)
        U_i^{loss} = ΔLoss_i / (ΔT_i + ε)
        U_i^{joint} = (w_rgb · norm(ΔPSNR_i) + w_depth · norm(ΔDepth_i)) / (ΔT_i + ε)

Sampling Populations (Points 3 & 25):
    - RANDOM_VISIBLE: Uniform random sample across visible Gaussians
    - IMPORTANCE_STRATIFIED: Stratified terciles (Low / Mid / High predicted importance)
    - UNIFORM_VISIBLE: Spatially distributed sample across depth z-bins
    - GEOMETRY_STRATIFIED: Stratified into 4 distinct physical regions:
        1. Flat surfaces: Low depth gradient, low color gradient
        2. Object edges: High depth discontinuity / depth gradient
        3. High texture: High spatial color variance / color gradient
        4. Depth discontinuity: Missing or invalid depth boundary

Group Scaling & Non-Additivity (Point 5):
    Evaluates group_size ∈ {1, 4, 16} to report:
        R_{add}(S) = ΔQ(S) / (∑_{i ∈ S} ΔQ_i + ε)

Oracle Repeat Stability (Point 6):
    Runs n = 3–5 repeat trials to measure μ_U, σ_U, and CV = σ_U / (|μ_U| + ε).
"""
import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import spearmanr
from copy import deepcopy

from .rasterizer import render as rasterize_scene
from .attribution import render_with_attribution, compute_gaussian_statistics, compute_projected_area
from .scheduler import estimate_gaussian_costs
from .background_cache import FrozenBackgroundCache
from .selective_optimizer import SelectiveAdam


class SamplingPopulation(str, Enum):
    """Sampling strategies for unbiased oracle validation."""
    IMPORTANCE_STRATIFIED = "importance_stratified"
    RANDOM_VISIBLE = "random_visible"
    UNIFORM_VISIBLE = "uniform_visible"
    GEOMETRY_STRATIFIED = "geometry_stratified"


class OracleUtilityExperiment:
    """Oracle Utility Experiment — Ground truth marginal utility engine."""
    
    def __init__(
        self,
        pipeline,
        n_samples: int = 150,
        n_opt_steps: int = 5,
        w_rgb: float = 0.70,
        w_depth: float = 0.30,
        seed: int = 42,
        contribution_threshold: float = 0.01,
        group_size: int = 1,
        min_influence_pixels: int = 25,
        protocol: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            pipeline: OnlineReconstructionPipeline instance (initialized)
            n_samples: number of Gaussians to evaluate per population
            n_opt_steps: gradient steps per Gaussian/group (default: 5, locked by protocol)
            w_rgb: weight of photometric improvement in joint ΔQ (default: 0.70)
            w_depth: weight of geometric depth improvement in joint ΔQ (default: 0.30)
            seed: random seed for reproducibility
            contribution_threshold: minimum w_{u,i} to include pixel in local region
            group_size: Gaussians per optimization group (1, 4, 16)
            min_influence_pixels: minimum pixel count for robust local utility estimation (default: 25)
            protocol: optional loaded protocol dictionary
        """
        self.pipeline = pipeline
        self.protocol = protocol
        
        # Override defaults with protocol if provided
        if protocol is not None:
            try:
                from research.protocol import get_oracle_config
                ocfg = get_oracle_config(protocol)
                n_opt_steps = int(ocfg.get('n_opt_steps', n_opt_steps))
                w_rgb = float(ocfg.get('w_rgb', w_rgb))
                w_depth = float(ocfg.get('w_depth', w_depth))
                min_influence_pixels = int(ocfg.get('min_influence_pixels', min_influence_pixels))
            except Exception:
                pass
                
        self.n_samples = n_samples
        self.n_opt_steps = n_opt_steps
        self.w_rgb = w_rgb
        self.w_depth = w_depth
        self.seed = seed
        self.contribution_threshold = contribution_threshold
        self.group_size = group_size
        self.min_influence_pixels = min_influence_pixels
        torch.manual_seed(seed)
        np.random.seed(seed)

        
    def snapshot_state(self) -> Dict:
        """Save all Gaussian parameters and optimizer state (deep copy)."""
        model = self.pipeline.gaussian_model
        snapshot = {}
        for name, param in model.named_parameters():
            snapshot[f'param_{name}'] = param.data.clone()
        for name, buf in model.named_buffers():
            snapshot[f'buffer_{name}'] = buf.clone()
        if self.pipeline.optimizer is not None:
            snapshot['optimizer_state'] = deepcopy(self.pipeline.optimizer.state_dict())
        else:
            snapshot['optimizer_state'] = None
        if hasattr(model, 'state_store') and model.state_store is not None:
            snapshot['state_store'] = model.state_store.state_dict()
        return snapshot
        
    def restore_state(self, snapshot: Dict):
        """Restore all parameters, optimizer, and persistent state from snapshot."""
        model = self.pipeline.gaussian_model
        for name, param in model.named_parameters():
            key = f'param_{name}'
            if key in snapshot:
                param.data.copy_(snapshot[key])
        for name, buf in model.named_buffers():
            key = f'buffer_{name}'
            if key in snapshot:
                buf.copy_(snapshot[key])
        if snapshot['optimizer_state'] is not None and self.pipeline.optimizer is not None:
            self.pipeline.optimizer.load_state_dict(snapshot['optimizer_state'])
        if 'state_store' in snapshot and hasattr(model, 'state_store') and model.state_store is not None:
            model.state_store.load_state_dict(snapshot['state_store'])

    def _render(self, H: int, W: int) -> Dict:
        """Render current scene state."""
        model = self.pipeline.gaussian_model
        return rasterize_scene(
            means3D=model.positions,
            cov3D=model.build_covariance(),
            colors=model.get_colors(),
            opacities=model.opacities.squeeze(-1),
            extrinsics=self.pipeline.current_pose,
            intrinsics=self.pipeline.intrinsics,
            image_width=W,
            image_height=H,
            tile_size=self.pipeline.config.get('rendering', {}).get('tile_size', 16),
        )

    def _render_with_attribution(self, H: int, W: int) -> Dict:
        """Render with per-Gaussian contribution tracking."""
        model = self.pipeline.gaussian_model
        return render_with_attribution(
            means3D=model.positions,
            cov3D=model.build_covariance(),
            colors=model.get_colors(),
            opacities=model.opacities.squeeze(-1),
            extrinsics=self.pipeline.current_pose,
            intrinsics=self.pipeline.intrinsics,
            image_width=W,
            image_height=H,
            tile_size=self.pipeline.config.get('rendering', {}).get('tile_size', 16),
            top_k=self.pipeline.config.get('rendering', {}).get('attribution_top_k', 8),
        )

    def _compute_local_psnr(
        self,
        rendered: torch.Tensor,
        gt: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> float:
        """Compute PSNR in dB only within masked region."""
        if pixel_mask.sum() == 0:
            return 0.0
        mse = ((rendered[pixel_mask] - gt[pixel_mask]) ** 2).mean().item()
        if mse < 1e-10:
            return 50.0
        return -10.0 * math.log10(mse)

    def _compute_local_ssim(
        self,
        rendered: torch.Tensor,
        gt: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> float:
        """Compute structural similarity within masked region."""
        if pixel_mask.sum() < 4:
            return 1.0
        r_pix = rendered[pixel_mask]
        g_pix = gt[pixel_mask]
        mu1 = r_pix.mean().item()
        mu2 = g_pix.mean().item()
        sig1_sq = r_pix.var().item() if r_pix.numel() > 1 else 0.0
        sig2_sq = g_pix.var().item() if g_pix.numel() > 1 else 0.0
        sig12 = ((r_pix - mu1) * (g_pix - mu2)).mean().item()
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2.0 * mu1 * mu2 + c1) * (2.0 * sig12 + c2)) / (
            (mu1**2 + mu2**2 + c1) * (sig1_sq + sig2_sq + c2) + 1e-8
        )
        return float(np.clip(ssim, 0.0, 1.0))

    def _compute_local_depth_l1(
        self,
        rendered_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> float:
        """Compute mean absolute depth error in meters within masked region."""
        valid = pixel_mask & (gt_depth > 0) & (~torch.isnan(gt_depth))
        if valid.sum() == 0:
            return 0.0
        return (rendered_depth[valid] - gt_depth[valid]).abs().mean().item()

    def _compute_local_loss(
        self,
        rendered_color: torch.Tensor,
        rendered_depth: torch.Tensor,
        gt_color: torch.Tensor,
        gt_depth: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> float:
        """Compute composite weighted RGB-D loss within masked region."""
        if pixel_mask.sum() == 0:
            return 0.0
        loss_rgb = ((rendered_color[pixel_mask] - gt_color[pixel_mask]) ** 2).mean().item()
        valid_d = pixel_mask & (gt_depth > 0) & (~torch.isnan(gt_depth))
        loss_d = (rendered_depth[valid_d] - gt_depth[valid_d]).abs().mean().item() if valid_d.any() else 0.0
        return self.w_rgb * loss_rgb + self.w_depth * loss_d

    def _get_influence_mask(
        self,
        gaussian_indices: List[int],
        contrib_indices: torch.Tensor,
        contrib_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Get boolean mask of pixels influenced by given Gaussians."""
        H, W, K = contrib_indices.shape
        mask = torch.zeros(H, W, dtype=torch.bool, device=contrib_indices.device)
        for idx in gaussian_indices:
            idx_match = (contrib_indices == idx)
            weight_ok = (contrib_weights > self.contribution_threshold)
            significant = (idx_match & weight_ok).any(dim=-1)
            mask |= significant
        return mask

    def optimize_gaussian_group(
        self,
        indices: List[int],
        n_steps: int,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        influence_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Optimize a group of Gaussians and measure raw RGB-D local quality improvements."""
        model = self.pipeline.gaussian_model
        H, W = rgb.shape[:2]
        device = rgb.device
        
        n_influence_pixels = int(influence_mask.sum().item())
        is_small_region = (n_influence_pixels < self.min_influence_pixels)
        full_mask = torch.ones(H, W, dtype=torch.bool, device=device)
        
        # === 1. Pre-optimization measurements (Local and Global) ===
        with torch.no_grad():
            before_out = self._render(H, W)
            before_color = before_out['color']
            before_depth = before_out['depth']
            
            # Local metrics
            psnr_local_before = self._compute_local_psnr(before_color, rgb, influence_mask)
            ssim_local_before = self._compute_local_ssim(before_color, rgb, influence_mask)
            depth_l1_before = self._compute_local_depth_l1(before_depth, depth, influence_mask)
            loss_local_before = self._compute_local_loss(before_color, before_depth, rgb, depth, influence_mask)
            
            # Global metrics
            psnr_global_before = self._compute_local_psnr(before_color, rgb, full_mask)
            ssim_global_before = self._compute_local_ssim(before_color, rgb, full_mask)
            depth_l1_global_before = self._compute_local_depth_l1(before_depth, depth, full_mask)
            loss_global_before = self._compute_local_loss(before_color, before_depth, rgb, depth, full_mask)
        
        # === 2. Isolated True Selective Optimization Trial ===
        opt_mask = torch.zeros(model.num_gaussians, dtype=torch.bool, device=device)
        opt_mask[indices] = True
        
        trial_opt = SelectiveAdam([{'params': list(model.parameters()), 'lr': 0.001}])
        trial_cache = FrozenBackgroundCache(device=device)
        frozen_mask = ~opt_mask
        
        # Pre-cache frozen background once for the trial
        if frozen_mask.any():
            trial_cache.build_cache(
                model=model,
                frozen_mask=frozen_mask,
                extrinsics=self.pipeline.current_pose,
                intrinsics=self.pipeline.intrinsics,
                image_width=W,
                image_height=H,
            )
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for step in range(n_steps):
            trial_opt.zero_grad()
            active_subset = model.get_optimization_subset(opt_mask)
            
            comp_out = trial_cache.composite_with_active(
                active_subset=active_subset,
                extrinsics=self.pipeline.current_pose,
                intrinsics=self.pipeline.intrinsics,
                image_width=W,
                image_height=H,
            )
            
            # Loss formulation
            if n_influence_pixels > 0 and n_influence_pixels < H * W * 0.8:
                loss_rgb = ((comp_out['color'][influence_mask] - rgb[influence_mask]) ** 2).mean()
                valid_d = influence_mask & (depth > 0)
                loss_depth = (comp_out['depth'][valid_d] - depth[valid_d]).abs().mean() if valid_d.any() else 0.0
            else:
                loss_rgb = ((comp_out['color'] - rgb) ** 2).mean()
                valid_d = depth > 0
                loss_depth = (comp_out['depth'][valid_d] - depth[valid_d]).abs().mean() if valid_d.any() else 0.0
                
            total_loss = self.w_rgb * loss_rgb + self.w_depth * loss_depth
            total_loss.backward()
            
            # True Selective Optimizer step (O(M))
            trial_opt.step(active_idx=active_subset['indices'])
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
        measured_trial_cost_ms = (time.perf_counter() - start_time) * 1000.0
        
        # === 3. Post-optimization measurements (Local and Global) ===
        with torch.no_grad():
            after_out = self._render(H, W)
            after_color = after_out['color']
            after_depth = after_out['depth']
            
            # Local
            psnr_local_after = self._compute_local_psnr(after_color, rgb, influence_mask)
            ssim_local_after = self._compute_local_ssim(after_color, rgb, influence_mask)
            depth_l1_after = self._compute_local_depth_l1(after_depth, depth, influence_mask)
            loss_local_after = self._compute_local_loss(after_color, after_depth, rgb, depth, influence_mask)
            
            # Global
            psnr_global_after = self._compute_local_psnr(after_color, rgb, full_mask)
            ssim_global_after = self._compute_local_ssim(after_color, rgb, full_mask)
            depth_l1_global_after = self._compute_local_depth_l1(after_depth, depth, full_mask)
            loss_global_after = self._compute_local_loss(after_color, after_depth, rgb, depth, full_mask)
            
        # Unclamped deltas (positive = quality improvement, negative = degradation)
        delta_psnr_local = psnr_local_after - psnr_local_before
        delta_ssim_local = ssim_local_after - ssim_local_before
        delta_depth_gain_local = depth_l1_before - depth_l1_after  # Unclamped: positive means error reduced
        delta_loss_local = loss_local_before - loss_local_after    # Unclamped: positive means loss reduced
        
        # Dimension-free normalized relative gains (unclamped)
        norm_delta_psnr = delta_psnr_local / max(1.0, psnr_local_before)
        norm_delta_depth = delta_depth_gain_local / max(1e-3, depth_l1_before)
        delta_quality_local = self.w_rgb * norm_delta_psnr + self.w_depth * norm_delta_depth
        
        # Global deltas (unclamped)
        delta_psnr_global = psnr_global_after - psnr_global_before
        delta_ssim_global = ssim_global_after - ssim_global_before
        delta_depth_gain_global = depth_l1_global_before - depth_l1_global_after
        delta_loss_global = loss_global_before - loss_global_after
        norm_delta_psnr_global = delta_psnr_global / max(1.0, psnr_global_before)
        norm_delta_depth_global = delta_depth_gain_global / max(1e-3, depth_l1_global_before)
        delta_quality_global = self.w_rgb * norm_delta_psnr_global + self.w_depth * norm_delta_depth_global
        
        # Actual trial cost strictly separating ΔQ and ΔT (Point 7)
        actual_cost_ms = max(0.001, measured_trial_cost_ms)
        
        # Local utilities (secondary diagnostic)
        oracle_util_rgb_local = delta_psnr_local / actual_cost_ms
        oracle_util_depth_local = delta_depth_gain_local / actual_cost_ms
        oracle_util_loss_local = delta_loss_local / actual_cost_ms
        oracle_util_joint_local = delta_quality_local / actual_cost_ms
        
        # Global utilities (PRIMARY SCIENTIFIC ESTIMAND: 3-FIX-1)
        oracle_util_rgb_global = delta_psnr_global / actual_cost_ms
        oracle_util_depth_global = delta_depth_gain_global / actual_cost_ms
        oracle_util_loss_global = delta_loss_global / actual_cost_ms
        oracle_util_joint_global = delta_quality_global / actual_cost_ms
        
        return {
            # Local metrics (secondary diagnostics)
            'psnr_local_before': psnr_local_before,
            'psnr_local_after': psnr_local_after,
            'delta_psnr_local': delta_psnr_local,
            'ssim_local_before': ssim_local_before,
            'ssim_local_after': ssim_local_after,
            'delta_ssim_local': delta_ssim_local,
            'depth_l1_before': depth_l1_before,
            'depth_l1_after': depth_l1_after,
            'delta_depth_gain_local': delta_depth_gain_local,
            'loss_local_before': loss_local_before,
            'loss_local_after': loss_local_after,
            'delta_loss_local': delta_loss_local,
            'delta_quality_local': delta_quality_local,
            'oracle_utility_rgb_local': oracle_util_rgb_local,
            'oracle_utility_depth_local': oracle_util_depth_local,
            'oracle_utility_loss_local': oracle_util_loss_local,
            'oracle_utility_joint_local': oracle_util_joint_local,
            
            # Global metrics (primary scientific estimand)
            'psnr_global_before': psnr_global_before,
            'psnr_global_after': psnr_global_after,
            'delta_psnr_global': delta_psnr_global,
            'ssim_global_before': ssim_global_before,
            'ssim_global_after': ssim_global_after,
            'delta_ssim_global': delta_ssim_global,
            'depth_l1_global_before': depth_l1_global_before,
            'depth_l1_global_after': depth_l1_global_after,
            'delta_depth_gain_global': delta_depth_gain_global,
            'loss_global_before': loss_global_before,
            'loss_global_after': loss_global_after,
            'delta_loss_global': delta_loss_global,
            'delta_quality_global': delta_quality_global,
            'oracle_utility_rgb_global': oracle_util_rgb_global,
            'oracle_utility_depth_global': oracle_util_depth_global,
            'oracle_utility_loss_global': oracle_util_loss_global,
            'oracle_utility_joint_global': oracle_util_joint_global,
            
            # Canonical primary aliases
            'delta_quality': delta_quality_global,
            'delta_psnr': delta_psnr_global,
            'delta_ssim': delta_ssim_global,
            'delta_depth': delta_depth_gain_global,
            'delta_loss': delta_loss_global,
            'oracle_utility_rgb': oracle_util_rgb_global,
            'oracle_utility_depth': oracle_util_depth_global,
            'oracle_utility_loss': oracle_util_loss_global,
            'oracle_utility_joint': oracle_util_joint_global,
            'oracle_utility': oracle_util_joint_global,
            
            'measured_trial_cost_ms': measured_trial_cost_ms,
            'n_influence_pixels': n_influence_pixels,
            'is_small_region': is_small_region,
        }


    def sample_geometry_stratified(
        self,
        visible_indices: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        contrib_indices: torch.Tensor,
        contrib_weights: torch.Tensor,
        n_samples: int,
    ) -> Tuple[List[int], Dict[int, str]]:
        """Sample candidates across 4 geometry strata (Points 3 & 25).
        
        Strata:
            - 'edge': High depth gradient / boundary
            - 'texture': High color gradient, moderate/low depth gradient
            - 'depth_discontinuity': High missing/invalid depth ratio or boundary
            - 'flat': Low depth gradient, low color gradient
            
        Returns:
            sampled_indices: List of candidate Gaussian indices
            strata_map: Dict mapping idx -> stratum name
        """
        device = rgb.device
        n_vis = len(visible_indices)
        if n_vis == 0:
            return [], {}
            
        # Compute 2D image gradients
        d_rgb_x = torch.zeros_like(rgb)
        d_rgb_y = torch.zeros_like(rgb)
        d_rgb_x[:, 1:] = (rgb[:, 1:] - rgb[:, :-1]).abs()
        d_rgb_y[1:, :] = (rgb[1:, :] - rgb[:-1, :]).abs()
        rgb_grad = (d_rgb_x + d_rgb_y).mean(dim=-1)  # (H, W)
        
        d_depth_x = torch.zeros_like(depth)
        d_depth_y = torch.zeros_like(depth)
        d_depth_x[:, 1:] = (depth[:, 1:] - depth[:, :-1]).abs()
        d_depth_y[1:, :] = (depth[1:, :] - depth[:-1, :]).abs()
        depth_grad = d_depth_x + d_depth_y  # (H, W)
        depth_missing = (depth <= 0) | torch.isnan(depth)
        
        # Calculate per-Gaussian geometric stats across their influence masks
        g_depth_vals = []
        g_rgb_vals = []
        g_missing_vals = []
        
        for idx in visible_indices:
            idx_i = int(idx.item())
            mask = (contrib_indices == idx_i) & (contrib_weights > self.contribution_threshold)
            pix_mask = mask.any(dim=-1)
            if pix_mask.sum() == 0:
                g_depth_vals.append(0.0)
                g_rgb_vals.append(0.0)
                g_missing_vals.append(0.0)
            else:
                g_depth_vals.append(depth_grad[pix_mask].mean().item())
                g_rgb_vals.append(rgb_grad[pix_mask].mean().item())
                g_missing_vals.append(depth_missing[pix_mask].float().mean().item())
                
        g_depth = np.array(g_depth_vals)
        g_rgb = np.array(g_rgb_vals)
        g_missing = np.array(g_missing_vals)
        
        # Determine quantile thresholds
        depth_edge_th = np.percentile(g_depth, 70) if len(g_depth) > 0 else 0.05
        rgb_tex_th = np.percentile(g_rgb, 70) if len(g_rgb) > 0 else 0.05
        flat_depth_th = np.percentile(g_depth, 40) if len(g_depth) > 0 else 0.02
        flat_rgb_th = np.percentile(g_rgb, 40) if len(g_rgb) > 0 else 0.02
        missing_th = np.percentile(g_missing, 70) if len(g_missing) > 0 else 0.1
        
        strata_buckets = {
            'edge': [],
            'texture': [],
            'depth_discontinuity': [],
            'flat': [],
        }
        
        for i, idx_t in enumerate(visible_indices):
            idx = int(idx_t.item())
            if g_depth[i] >= depth_edge_th:
                strata_buckets['edge'].append(idx)
            elif g_missing[i] >= missing_th and g_missing[i] > 0.05:
                strata_buckets['depth_discontinuity'].append(idx)
            elif g_rgb[i] >= rgb_tex_th:
                strata_buckets['texture'].append(idx)
            elif g_depth[i] <= flat_depth_th and g_rgb[i] <= flat_rgb_th:
                strata_buckets['flat'].append(idx)
            else:
                strata_buckets['flat'].append(idx)
                
        # Sample balanced quota across strata
        k_target = min(n_samples, n_vis)
        quota_per_stratum = max(1, k_target // 4)
        sampled = []
        strata_map = {}
        
        for stratum, indices in strata_buckets.items():
            if len(indices) == 0:
                continue
            perm = np.random.permutation(len(indices))
            selected = [indices[p] for p in perm[:quota_per_stratum]]
            for s_idx in selected:
                sampled.append(s_idx)
                strata_map[s_idx] = stratum
                
        # Fill remaining up to k_target if some strata were under-represented
        if len(sampled) < k_target:
            all_remaining = [int(x.item()) for x in visible_indices if int(x.item()) not in strata_map]
            if len(all_remaining) > 0:
                perm = np.random.permutation(len(all_remaining))
                needed = k_target - len(sampled)
                for s_idx in [all_remaining[p] for p in perm[:needed]]:
                    sampled.append(s_idx)
                    strata_map[s_idx] = 'general_visible'
                    
        return sampled[:k_target], strata_map

    def sample_population(
        self,
        population_type: SamplingPopulation,
        num_gaussians: int,
        predicted_importance: torch.Tensor,
        visibility_mask: torch.Tensor,
        positions: torch.Tensor,
        n_samples: int,
        rgb: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        contrib_indices: Optional[torch.Tensor] = None,
        contrib_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], Dict[int, str]]:
        """Sample Gaussian indices according to specified population strategy."""
        visible_indices = torch.where(visibility_mask)[0]
        n_vis = len(visible_indices)
        if n_vis == 0:
            return [], {}
            
        sample_k = min(n_samples, n_vis)
        strata_map = {}
        
        if population_type == SamplingPopulation.RANDOM_VISIBLE:
            perm = torch.randperm(n_vis)[:sample_k]
            res = visible_indices[perm].tolist()
            return res, {idx: 'random_visible' for idx in res}
            
        elif population_type == SamplingPopulation.IMPORTANCE_STRATIFIED:
            vis_imp = predicted_importance[visible_indices]
            sorted_order = torch.argsort(vis_imp, descending=True)
            sorted_vis = visible_indices[sorted_order]
            
            n_third = max(1, sample_k // 3)
            high_stratum = sorted_vis[:max(n_third * 2, 1)]
            mid_start = len(sorted_vis) // 3
            mid_stratum = sorted_vis[mid_start:mid_start + max(n_third * 2, 1)]
            low_stratum = sorted_vis[-max(n_third * 2, 1):]
            
            p_h = torch.randperm(len(high_stratum))[:n_third]
            p_m = torch.randperm(len(mid_stratum))[:n_third]
            p_l = torch.randperm(len(low_stratum))[:(sample_k - 2 * n_third)]
            
            h_list = high_stratum[p_h].tolist()
            m_list = mid_stratum[p_m].tolist()
            l_list = low_stratum[p_l].tolist()
            
            for idx in h_list: strata_map[idx] = 'importance_high'
            for idx in m_list: strata_map[idx] = 'importance_mid'
            for idx in l_list: strata_map[idx] = 'importance_low'
            
            return h_list + m_list + l_list, strata_map
            
        elif population_type == SamplingPopulation.UNIFORM_VISIBLE:
            vis_pos = positions[visible_indices]
            z_vals = vis_pos[:, 2]
            z_order = torch.argsort(z_vals)
            sorted_vis = visible_indices[z_order]
            step = max(1, len(sorted_vis) // sample_k)
            res = sorted_vis[::step][:sample_k].tolist()
            return res, {idx: 'uniform_depth_bin' for idx in res}
            
        elif population_type == SamplingPopulation.GEOMETRY_STRATIFIED:
            if rgb is not None and depth is not None and contrib_indices is not None and contrib_weights is not None:
                return self.sample_geometry_stratified(
                    visible_indices=visible_indices,
                    rgb=rgb,
                    depth=depth,
                    contrib_indices=contrib_indices,
                    contrib_weights=contrib_weights,
                    n_samples=sample_k,
                )
            else:
                perm = torch.randperm(n_vis)[:sample_k]
                res = visible_indices[perm].tolist()
                return res, {idx: 'geometry_fallback' for idx in res}
            
        res = visible_indices[:sample_k].tolist()
        return res, {idx: 'default' for idx in res}

    def run_oracle_experiment(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        population_type: SamplingPopulation = SamplingPopulation.GEOMETRY_STRATIFIED,
        sample_indices: Optional[List[int]] = None,
        scene_name: str = "scene",
        frame_idx: int = 0,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run full oracle utility measurement on selected population with complete state tracking."""
        H, W = rgb.shape[:2]
        device = rgb.device
        model = self.pipeline.gaussian_model
        num_gaussians = model.num_gaussians
        run_seed = seed if seed is not None else getattr(self, 'seed', 42)
        
        # Determine dataset split strictly per protocol
        if split is None:
            sc = scene_name.lower()
            if "fr2" in sc or "xyz" in sc:
                split = "cross_scene_test"
            elif frame_idx <= 40:
                split = "train"
            else:
                split = "validation"
        
        # 1. Attribution & per-Gaussian state extraction
        with torch.no_grad():
            attr_result = self._render_with_attribution(H, W)
            contrib_indices = attr_result['contrib_indices']
            contrib_weights = attr_result['contrib_weights']
            
            cov3D = model.build_covariance()
            proj_areas = compute_gaussian_statistics(
                rendered_color=attr_result['color'],
                rendered_depth=attr_result['depth'],
                gt_color=rgb,
                gt_depth=depth,
                contrib_weights=contrib_weights,
                contrib_indices=contrib_indices,
                n_gaussians=num_gaussians
            )
            visibility_mask = proj_areas['visibility_mask']
            influence_mass = proj_areas['influence_mass']
            projected_area = proj_areas['projected_area']
            color_error = proj_areas['color_error']
            depth_error = proj_areas['depth_error']
            visibility = proj_areas['visibility']
            pixel_count = proj_areas.get('pixel_count', torch.zeros(num_gaussians, device=device))
            
        # 2. Extract pipeline diagnostics & persistent state features (Points III & IV)
        try:
            diagnostics = self.pipeline.get_importance_diagnostics()
        except RuntimeError:
            diagnostics = {
                'importance': torch.zeros(num_gaussians, device=device),
                'tiers': torch.zeros(num_gaussians, dtype=torch.long, device=device),
                'confidence': torch.full((num_gaussians,), 0.5, device=device),
                'components': {'temporal': torch.zeros(num_gaussians, device=device)},
            }
        predicted_importance = diagnostics['importance']
        tiers = diagnostics.get('tiers', torch.zeros(num_gaussians, dtype=torch.long, device=device))
        confidence = diagnostics.get('confidence', torch.full((num_gaussians,), 0.5, device=device))
        temporal_change = diagnostics.get('components', {}).get(
            'temporal', torch.zeros(num_gaussians, device=device)
        )

        # Extract synchronized state from GaussianStateStore if available
        store = getattr(model, 'state_store', None)
        if store is not None and store.num_gaussians >= num_gaussians:
            # Update store's visibility_count using attribution pixel_count (3-FIX-4)
            store.visibility_count[:num_gaussians] = pixel_count[:num_gaussians].float()
            pos_drift = store.position_drift[:num_gaussians]
            res_drift_ema = store.residual_drift_ema[:num_gaussians]
            store_age = store.ages[:num_gaussians]
            store_staleness = store.get_staleness(frame_idx)[:num_gaussians]
            store_vis_cnt = store.visibility_count[:num_gaussians]
            store_tiers = store.tiers[:num_gaussians]
            store_update_freq = store.get_update_frequency(frame_idx)[:num_gaussians]
        else:
            pos_drift = temporal_change
            res_drift_ema = torch.zeros(num_gaussians, device=device)
            creation_frame = getattr(self.pipeline.importance_estimator, '_creation_frame', None)
            if creation_frame is not None and creation_frame.shape[0] >= num_gaussians:
                store_age = (frame_idx - creation_frame[:num_gaussians]).clamp(min=0)
            else:
                store_age = torch.zeros(num_gaussians, dtype=torch.long, device=device)
            store_staleness = torch.zeros(num_gaussians, dtype=torch.long, device=device)
            store_vis_cnt = pixel_count
            store_tiers = tiers
            store_update_freq = torch.full((num_gaussians,), 0.5, device=device)
        
        # Cost model estimates
        cost_estimates_us = estimate_gaussian_costs(
            screen_areas=influence_mass,
            n_gaussians=num_gaussians,
            device=device
        )
        predicted_utility = predicted_importance / (cost_estimates_us / 1000.0 + 1e-6)
        uncertainty = (1.0 - confidence).clamp(0.0, 1.0)
            
        # 3. Candidate Sampling
        strata_map = {}
        if sample_indices is None:
            sample_indices, strata_map = self.sample_population(
                population_type=population_type,
                num_gaussians=num_gaussians,
                predicted_importance=predicted_importance,
                visibility_mask=visibility_mask,
                positions=model.positions,
                n_samples=self.n_samples,
                rgb=rgb,
                depth=depth,
                contrib_indices=contrib_indices,
                contrib_weights=contrib_weights,
            )
        else:
            strata_map = {idx: 'prespecified' for idx in sample_indices}
            
        # 3b. Compute 3D KNN Contextual Features (Phase 5 - Points XI, XII, XXVIII)
        knn_features = {}
        if len(sample_indices) > 0 and num_gaussians > 1:
            try:
                cand_t = torch.tensor(sample_indices, dtype=torch.long, device=model.positions.device)
                cand_pos = model.positions[cand_t]
                all_pos = model.positions
                dists = torch.cdist(cand_pos, all_pos)
                k_val = min(5, num_gaussians)
                topk_dists, topk_idx = torch.topk(dists, k=k_val, largest=False, dim=-1)
                
                n_idx = topk_idx[:, 1:]
                n_dst = topk_dists[:, 1:]
                all_err = (color_error + depth_error)
                all_z = model.positions[:, 2]
                
                d_mean = n_dst.mean(dim=-1).cpu().numpy()
                e_mean = all_err[n_idx].mean(dim=-1).cpu().numpy()
                z_var = all_z[n_idx].var(dim=-1).cpu().numpy() if k_val > 2 else np.zeros(len(sample_indices))
                
                for si, s_idx in enumerate(sample_indices):
                    knn_features[s_idx] = {
                        'knn_density': float(d_mean[si]),
                        'knn_error_mean': float(e_mean[si]),
                        'knn_depth_var': float(z_var[si]) if not np.isnan(z_var[si]) else 0.0,
                    }
            except Exception:
                pass
            
        # 4. Group Partitioning
        if self.group_size > 1:
            groups = [sample_indices[i:i + self.group_size]
                      for i in range(0, len(sample_indices), self.group_size)]
        else:
            groups = [[idx] for idx in sample_indices]
            
        results = []
        
        # 5. Isolated Trial Execution with Snapshot/Restore
        for gi, group in enumerate(groups):
            influence_mask = self._get_influence_mask(group, contrib_indices, contrib_weights)
            n_pixels = int(influence_mask.sum().item())
            is_filtered = bool(n_pixels < self.min_influence_pixels)
            filter_reason = "min_influence_pixels" if is_filtered else "none"
            
            if n_pixels == 0:
                for idx in group:
                    knn_f = knn_features.get(idx, {'knn_density': 0.0, 'knn_error_mean': 0.0, 'knn_depth_var': 0.0})
                    p_id = int(model.persistent_ids[idx].item()) if hasattr(model, 'persistent_ids') and idx < len(model.persistent_ids) else idx
                    results.append({
                        "seed": int(run_seed),
                        "scene": scene_name,
                        "frame": frame_idx,
                        "split": split,
                        "gaussian_id": idx,
                        "persistent_id": p_id,
                        "population": population_type.value if hasattr(population_type, 'value') else str(population_type),
                        "geometry_stratum": strata_map.get(idx, 'none'),
                        "group_size": self.group_size,
                        "features": {
                            "rgb_error": float(color_error[idx]),
                            "depth_error": float(depth_error[idx]),
                            "gradient_norm": float(influence_mass[idx] * (color_error[idx] + depth_error[idx])),
                            "visibility_count": float(store_vis_cnt[idx]),
                            "visibility": float(visibility[idx]),
                            "influence_mass": float(influence_mass[idx]),
                            "position_drift": float(pos_drift[idx]),
                            "residual_drift_ema": float(res_drift_ema[idx]),
                            "uncertainty": float(uncertainty[idx]),
                            "uncertainty_var": float(uncertainty[idx]),
                            "projected_area": float(projected_area[idx]),
                            "age": int(store_age[idx].item() if hasattr(store_age[idx], 'item') else store_age[idx]),
                            "staleness": int(store_staleness[idx].item() if hasattr(store_staleness[idx], 'item') else store_staleness[idx]),
                            "update_frequency": float(store_update_freq[idx].item() if hasattr(store_update_freq[idx], 'item') else store_update_freq[idx]),
                            "tier": int(store_tiers[idx].item() if hasattr(store_tiers[idx], 'item') else store_tiers[idx]),
                            "knn_density": knn_f['knn_density'],
                            "knn_error_mean": knn_f['knn_error_mean'],
                            "knn_depth_var": knn_f['knn_depth_var'],
                            "sh_degree": int(getattr(model, 'sh_degree', 0)),
                        },
                        "predicted_importance": float(predicted_importance[idx]),
                        "predicted_utility": float(predicted_utility[idx]),
                        # Primary Global Metrics (3-FIX-1)
                        "psnr_before": 0.0,
                        "psnr_after": 0.0,
                        "delta_psnr": 0.0,
                        "ssim_before": 0.0,
                        "ssim_after": 0.0,
                        "delta_ssim": 0.0,
                        "depth_before": 0.0,
                        "depth_after": 0.0,
                        "delta_depth": 0.0,
                        "loss_before": 0.0,
                        "loss_after": 0.0,
                        "delta_loss": 0.0,
                        "delta_quality": 0.0,
                        "delta_quality_global": 0.0,
                        "oracle_utility": 0.0,
                        "oracle_utility_joint": 0.0,
                        "oracle_utility_rgb": 0.0,
                        "oracle_utility_depth": 0.0,
                        "oracle_utility_loss": 0.0,
                        "oracle_utility_joint_global": 0.0,
                        "oracle_utility_rgb_global": 0.0,
                        "oracle_utility_depth_global": 0.0,
                        "oracle_utility_loss_global": 0.0,
                        # Local secondary diagnostics
                        "psnr_local_before": 0.0,
                        "psnr_local_after": 0.0,
                        "delta_psnr_local": 0.0,
                        "ssim_local_before": 0.0,
                        "ssim_local_after": 0.0,
                        "delta_ssim_local": 0.0,
                        "depth_local_before": 0.0,
                        "depth_local_after": 0.0,
                        "delta_depth_gain_local": 0.0,
                        "loss_local_before": 0.0,
                        "loss_local_after": 0.0,
                        "delta_loss_local": 0.0,
                        "delta_quality_local": 0.0,
                        "oracle_utility_joint_local": 0.0,
                        "oracle_utility_rgb_local": 0.0,
                        "oracle_utility_depth_local": 0.0,
                        "oracle_utility_loss_local": 0.0,
                        "delta_time_ms": 0.0,
                        "measured_trial_cost_ms": 0.0,
                        "modeled_marginal_cost_us": float(cost_estimates_us[idx]),
                        "n_influence_pixels": 0,
                        "filtered": True,
                        "filter_reason": "zero_influence_pixels",
                        "visible": False,
                    })
                continue
                
            snapshot = self.snapshot_state()
            try:
                metrics = self.optimize_gaussian_group(
                    group, self.n_opt_steps, rgb, depth, influence_mask)
                
                trial_cost = metrics['measured_trial_cost_ms']
                per_gauss_cost = trial_cost / len(group)
                delta_q_global = metrics['delta_quality_global']
                delta_q_local = metrics['delta_quality_local']
                
                for idx in group:
                    knn_f = knn_features.get(idx, {'knn_density': 0.0, 'knn_error_mean': 0.0, 'knn_depth_var': 0.0})
                    p_id = int(model.persistent_ids[idx].item()) if hasattr(model, 'persistent_ids') and idx < len(model.persistent_ids) else idx
                    results.append({
                        "seed": int(run_seed),
                        "scene": scene_name,
                        "frame": frame_idx,
                        "split": split,
                        "gaussian_id": idx,
                        "persistent_id": p_id,
                        "population": population_type.value if hasattr(population_type, 'value') else str(population_type),
                        "geometry_stratum": strata_map.get(idx, 'none'),
                        "group_size": self.group_size,
                        "features": {
                            "rgb_error": float(color_error[idx]),
                            "depth_error": float(depth_error[idx]),
                            "gradient_norm": float(influence_mass[idx] * (color_error[idx] + depth_error[idx])),
                            "visibility_count": float(store_vis_cnt[idx]),
                            "visibility": float(visibility[idx]),
                            "influence_mass": float(influence_mass[idx]),
                            "position_drift": float(pos_drift[idx]),
                            "residual_drift_ema": float(res_drift_ema[idx]),
                            "uncertainty": float(uncertainty[idx]),
                            "uncertainty_var": float(uncertainty[idx]),
                            "projected_area": float(projected_area[idx]),
                            "age": int(store_age[idx].item() if hasattr(store_age[idx], 'item') else store_age[idx]),
                            "staleness": int(store_staleness[idx].item() if hasattr(store_staleness[idx], 'item') else store_staleness[idx]),
                            "update_frequency": float(store_update_freq[idx].item() if hasattr(store_update_freq[idx], 'item') else store_update_freq[idx]),
                            "tier": int(store_tiers[idx].item() if hasattr(store_tiers[idx], 'item') else store_tiers[idx]),
                            "knn_density": knn_f['knn_density'],
                            "knn_error_mean": knn_f['knn_error_mean'],
                            "knn_depth_var": knn_f['knn_depth_var'],
                            "sh_degree": int(getattr(model, 'sh_degree', 0)),
                        },
                        "raw_metrics": {
                            "psnr_before": float(metrics['psnr_global_before']),
                            "psnr_after": float(metrics['psnr_global_after']),
                            "delta_psnr": float(metrics['delta_psnr_global']),
                            "ssim_before": float(metrics['ssim_global_before']),
                            "ssim_after": float(metrics['ssim_global_after']),
                            "delta_ssim": float(metrics['delta_ssim_global']),
                            "depth_l1_before": float(metrics['depth_l1_global_before']),
                            "depth_l1_after": float(metrics['depth_l1_global_after']),
                            "delta_depth_gain": float(metrics['delta_depth_gain_global']),
                            "loss_before": float(metrics['loss_global_before']),
                            "loss_after": float(metrics['loss_global_after']),
                            "delta_loss": float(metrics['delta_loss_global']),
                            "measured_trial_cost_ms": float(trial_cost),
                            "psnr_local_before": float(metrics['psnr_local_before']),
                            "psnr_local_after": float(metrics['psnr_local_after']),
                            "delta_psnr_local": float(metrics['delta_psnr_local']),
                            "depth_l1_local_before": float(metrics['depth_l1_before']),
                            "depth_l1_local_after": float(metrics['depth_l1_after']),
                            "delta_depth_gain_local": float(metrics['delta_depth_gain_local']),
                            "delta_quality_local": float(delta_q_local),
                        },
                        # Primary Global Metrics (3-FIX-1)
                        "psnr_before": float(metrics['psnr_global_before']),
                        "psnr_after": float(metrics['psnr_global_after']),
                        "delta_psnr": float(metrics['delta_psnr_global']),
                        "ssim_before": float(metrics['ssim_global_before']),
                        "ssim_after": float(metrics['ssim_global_after']),
                        "delta_ssim": float(metrics['delta_ssim_global']),
                        "depth_before": float(metrics['depth_l1_global_before']),
                        "depth_after": float(metrics['depth_l1_global_after']),
                        "delta_depth": float(metrics['delta_depth_gain_global']),
                        "loss_before": float(metrics['loss_global_before']),
                        "loss_after": float(metrics['loss_global_after']),
                        "delta_loss": float(metrics['delta_loss_global']),
                        "delta_quality": float(delta_q_global),
                        "delta_quality_global": float(delta_q_global),
                        "delta_time_ms": float(per_gauss_cost),
                        "measured_trial_cost_ms": float(per_gauss_cost),
                        "modeled_marginal_cost_us": float(cost_estimates_us[idx]),
                        "oracle_utility": float(metrics['oracle_utility_joint_global']),
                        "oracle_utility_joint": float(metrics['oracle_utility_joint_global']),
                        "oracle_utility_rgb": float(metrics['oracle_utility_rgb_global']),
                        "oracle_utility_depth": float(metrics['oracle_utility_depth_global']),
                        "oracle_utility_loss": float(metrics['oracle_utility_loss_global']),
                        "oracle_utility_joint_global": float(metrics['oracle_utility_joint_global']),
                        "oracle_utility_rgb_global": float(metrics['oracle_utility_rgb_global']),
                        "oracle_utility_depth_global": float(metrics['oracle_utility_depth_global']),
                        "oracle_utility_loss_global": float(metrics['oracle_utility_loss_global']),
                        # Secondary Local Diagnostics
                        "psnr_local_before": float(metrics['psnr_local_before']),
                        "psnr_local_after": float(metrics['psnr_local_after']),
                        "delta_psnr_local": float(metrics['delta_psnr_local']),
                        "ssim_local_before": float(metrics['ssim_local_before']),
                        "ssim_local_after": float(metrics['ssim_local_after']),
                        "delta_ssim_local": float(metrics['delta_ssim_local']),
                        "depth_local_before": float(metrics['depth_l1_before']),
                        "depth_local_after": float(metrics['depth_l1_after']),
                        "delta_depth_gain_local": float(metrics['delta_depth_gain_local']),
                        "loss_local_before": float(metrics['loss_local_before']),
                        "loss_local_after": float(metrics['loss_local_after']),
                        "delta_loss_local": float(metrics['delta_loss_local']),
                        "delta_quality_local": float(delta_q_local),
                        "oracle_utility_joint_local": float(metrics['oracle_utility_joint_local']),
                        "oracle_utility_rgb_local": float(metrics['oracle_utility_rgb_local']),
                        "oracle_utility_depth_local": float(metrics['oracle_utility_depth_local']),
                        "oracle_utility_loss_local": float(metrics['oracle_utility_loss_local']),
                        # Diagnostics and status
                        "predicted_importance": float(predicted_importance[idx]),
                        "predicted_utility": float(predicted_utility[idx]),
                        "influence_mass": float(influence_mass[idx]),
                        "projected_area": float(projected_area[idx]),
                        "n_influence_pixels": n_pixels,
                        "filtered": is_filtered,
                        "filter_reason": filter_reason,
                        "visible": True,
                    })
            finally:
                self.restore_state(snapshot)
                
        return results

    def run_stability_check(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        candidate_indices: List[int],
        n_repeats: int = 3,
        scene_name: str = "scene",
        frame_idx: int = 0,
    ) -> Dict[str, Any]:
        """Run n repeated trials on candidate subset to measure oracle noise & stability (Point 6)."""
        H, W = rgb.shape[:2]
        
        with torch.no_grad():
            attr_result = self._render_with_attribution(H, W)
            contrib_indices = attr_result['contrib_indices']
            contrib_weights = attr_result['contrib_weights']
            
        candidate_stats = []
        cvs = []
        
        for idx in candidate_indices:
            influence_mask = self._get_influence_mask([idx], contrib_indices, contrib_weights)
            if influence_mask.sum() == 0:
                continue
                
            trial_utilities = []
            trial_times = []
            trial_gains = []
            
            for r in range(n_repeats):
                snapshot = self.snapshot_state()
                try:
                    m = self.optimize_gaussian_group([idx], self.n_opt_steps, rgb, depth, influence_mask)
                    trial_utilities.append(m['oracle_utility_joint'])
                    trial_times.append(m['measured_trial_cost_ms'])
                    trial_gains.append(m['delta_quality'])
                finally:
                    self.restore_state(snapshot)
                    
            u_arr = np.array(trial_utilities)
            mu_u = float(np.mean(u_arr))
            sigma_u = float(np.std(u_arr))
            cv = float(sigma_u / (abs(mu_u) + 1e-6))
            cvs.append(cv)
            
            p_positive_gain = float(np.mean([1.0 if g > 1e-6 else 0.0 for g in trial_gains])) if trial_gains else 1.0

            candidate_stats.append({
                "gaussian_id": idx,
                "n_repeats": n_repeats,
                "mean_utility": mu_u,
                "std_utility": sigma_u,
                "coefficient_of_variation": cv,
                "is_stable": cv <= 0.35,
                "p_positive_gain": p_positive_gain,
                "mean_time_ms": float(np.mean(trial_times)),
                "std_time_ms": float(np.std(trial_times)),
                "mean_gain": float(np.mean(trial_gains)),
            })
            
        mean_cv = float(np.mean(cvs)) if cvs else 0.0
        median_cv = float(np.median(cvs)) if cvs else 0.0
        stable_frac = float(np.mean([1.0 if c <= 0.35 else 0.0 for c in cvs])) if cvs else 1.0
        mean_sign_stability = float(np.mean([c['p_positive_gain'] for c in candidate_stats])) if candidate_stats else 1.0
        
        pos_candidates = [c for c in candidate_stats if c['mean_utility'] > 0]
        neg_candidates = [c for c in candidate_stats if c['mean_utility'] < 0]
        pos_cv = float(np.mean([c['coefficient_of_variation'] for c in pos_candidates])) if pos_candidates else 0.0
        neg_cv = float(np.mean([c['coefficient_of_variation'] for c in neg_candidates])) if neg_candidates else 0.0
        
        return {
            "n_candidates": len(candidate_stats),
            "n_repeats": n_repeats,
            "mean_cv": mean_cv,
            "median_cv": median_cv,
            "positive_utility_count": len(pos_candidates),
            "negative_utility_count": len(neg_candidates),
            "positive_utility_cv": pos_cv,
            "negative_utility_cv": neg_cv,
            "stable_fraction": stable_frac,
            "mean_sign_stability": mean_sign_stability,
            "gate1_passed": (mean_cv <= 0.35 or median_cv <= 0.35) and (mean_sign_stability >= 0.70),
            "candidates": candidate_stats,
        }


    def evaluate_group_interaction(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        candidate_indices: List[int],
        group_sizes: List[int] = [1, 4, 8, 16],
        n_groups_per_size: int = 4,
    ) -> Dict[str, Any]:
        """Evaluate non-additivity and interaction error across group sizes (Point IX).
        
        Measures:
            Interaction Error = |U(S) - Σ_{i in S} U_i| / (|U(S)| + ε)
            Additivity Ratio = U(S) / (Σ_{i in S} U_i + ε)
        """
        H, W = rgb.shape[:2]
        with torch.no_grad():
            attr = self._render_with_attribution(H, W)
            c_idx, c_wt = attr['contrib_indices'], attr['contrib_weights']
            
        group_interaction_results = {}
        
        # Precompute individual utilities for candidate_indices
        individual_utilities = {}
        for idx in candidate_indices:
            mask = self._get_influence_mask([idx], c_idx, c_wt)
            if mask.sum() == 0:
                continue
            snap = self.snapshot_state()
            try:
                m = self.optimize_gaussian_group([idx], self.n_opt_steps, rgb, depth, mask)
                individual_utilities[idx] = m['oracle_utility_joint']
            finally:
                self.restore_state(snap)
                
        valid_candidates = list(individual_utilities.keys())
        if len(valid_candidates) < 4:
            return {'error': 'Insufficient valid candidates for group interaction test'}
            
        for g_size in group_sizes:
            if g_size == 1:
                group_interaction_results['group_size_1'] = {
                    'group_size': 1,
                    'interaction_error_mean': 0.0,
                    'interaction_error_median': 0.0,
                    'additivity_ratio_mean': 1.0,
                    'n_groups': len(valid_candidates),
                }
                continue
                
            errors = []
            ratios = []
            
            # Form candidate groups
            np_cand = np.array(valid_candidates)
            max_possible_groups = max(1, len(valid_candidates) // g_size)
            for g_step in range(min(n_groups_per_size, max_possible_groups)):
                perm = np.random.permutation(len(np_cand))
                group = np_cand[perm[:g_size]].tolist()
                
                sum_indiv = sum(individual_utilities[i] for i in group)
                
                # Joint group trial
                g_mask = self._get_influence_mask(group, c_idx, c_wt)
                snap = self.snapshot_state()
                try:
                    m_group = self.optimize_gaussian_group(group, self.n_opt_steps, rgb, depth, g_mask)
                    u_joint = m_group['oracle_utility_joint']
                finally:
                    self.restore_state(snap)
                    
                inter_err = abs(u_joint - sum_indiv) / (abs(u_joint) + 1e-6)
                add_ratio = u_joint / (sum_indiv + 1e-6)
                errors.append(float(inter_err))
                ratios.append(float(add_ratio))
                
            group_interaction_results[f'group_size_{g_size}'] = {
                'group_size': g_size,
                'interaction_error_mean': float(np.mean(errors)) if errors else 0.0,
                'interaction_error_median': float(np.median(errors)) if errors else 0.0,
                'additivity_ratio_mean': float(np.mean(ratios)) if ratios else 1.0,
                'n_groups': len(errors),
            }
            
        return group_interaction_results

    def evaluate_diminishing_returns(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        candidate_indices: List[int],
        n_trials: int = 10,
        size_a: int = 2,
        size_b: int = 6,
    ) -> Dict[str, Any]:
        """Empirically evaluate diminishing marginal returns: Delta_i(A) >= Delta_i(B) for A subset B."""
        H, W = rgb.shape[:2]
        full_mask = torch.ones(H, W, dtype=torch.bool, device=rgb.device)
        
        candidates = list(candidate_indices)
        if len(candidates) < size_b + 1:
            return {'error': 'Insufficient candidates for diminishing returns test'}
            
        def _get_joint_gain(indices):
            snap = self.snapshot_state()
            try:
                res = self.optimize_gaussian_group(
                    indices, self.n_opt_steps, rgb, depth, influence_mask=full_mask
                )
                return res['delta_quality_global']
            finally:
                self.restore_state(snap)
                
        trials = []
        diminishing_count = 0
        
        for t in range(n_trials):
            perm = np.random.permutation(len(candidates))
            idx_b = [candidates[p] for p in perm[:size_b]]
            idx_a = idx_b[:size_a]
            i_elem = candidates[perm[size_b]]
            
            # Gains
            q_a = _get_joint_gain(idx_a)
            q_a_plus_i = _get_joint_gain(idx_a + [i_elem])
            delta_i_a = q_a_plus_i - q_a
            
            q_b = _get_joint_gain(idx_b)
            q_b_plus_i = _get_joint_gain(idx_b + [i_elem])
            delta_i_b = q_b_plus_i - q_b
            
            is_diminishing = bool(delta_i_a >= delta_i_b - 1e-7)
            if is_diminishing:
                diminishing_count += 1
                
            trials.append({
                'delta_i_A': float(delta_i_a),
                'delta_i_B': float(delta_i_b),
                'diminishing': is_diminishing,
            })
            
        mean_delta_a = float(np.mean([t['delta_i_A'] for t in trials]))
        mean_delta_b = float(np.mean([t['delta_i_B'] for t in trials]))
        consistency_rate = float(diminishing_count / len(trials)) if trials else 0.0
        
        return {
            'n_trials': len(trials),
            'size_A': size_a,
            'size_B': size_b,
            'mean_marginal_gain_A': mean_delta_a,
            'mean_marginal_gain_B': mean_delta_b,
            'diminishing_rate': consistency_rate,
            'is_diminishing_consistent': bool(mean_delta_a >= mean_delta_b),
            'trials': trials,
        }

    def compute_correlation_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute Spearman rank correlations, Overlap@K, Realized Gain, and Regret."""
        visible = [r for r in results if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
        if len(visible) < 5:
            return {'error': 'Insufficient visible Gaussians for statistical evaluation', 'n_visible': len(visible)}
            
        def _safe_spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
            if len(x) < 3 or np.std(x) < 1e-7 or np.std(y) < 1e-7:
                return float('nan'), float('nan')
            r, p = spearmanr(x, y)
            return (float(r) if not np.isnan(r) else float('nan')), (float(p) if not np.isnan(p) else float('nan'))

        def _compute_ndcg(pred_scores: np.ndarray, true_scores: np.ndarray, k_val: int) -> float:
            if k_val <= 0 or len(pred_scores) == 0:
                return 0.0
            k_eval = min(k_val, len(pred_scores))
            p_idx = np.argsort(-pred_scores)[:k_eval]
            i_idx = np.argsort(-true_scores)[:k_eval]
            min_val = min(0.0, float(np.min(true_scores)))
            rel = true_scores - min_val
            discounts = np.log2(np.arange(2, k_eval + 2))
            dcg = np.sum(rel[p_idx] / discounts)
            idcg = np.sum(rel[i_idx] / discounts)
            return float(dcg / (idcg + 1e-8)) if idcg > 0 else 1.0

        pred_imp = np.array([r['predicted_importance'] for r in visible])
        pred_util = np.array([r['predicted_utility'] for r in visible])
        oracle_util = np.array([r.get('oracle_utility_joint', r.get('oracle_utility', 0.0)) for r in visible])
        oracle_rgb = np.array([r.get('oracle_utility_rgb', 0.0) for r in visible])
        oracle_depth = np.array([r.get('oracle_utility_depth', 0.0) for r in visible])
        oracle_util_global = np.array([r.get('oracle_utility_joint_global', 0.0) for r in visible])
        delta_q = np.array([r['delta_quality_local'] for r in visible])
        delta_psnr = np.array([r['delta_psnr_local'] for r in visible])
        
        rho_util_oracle, p_util_oracle = _safe_spearmanr(pred_util, oracle_util)
        rho_imp_oracle, p_imp_oracle = _safe_spearmanr(pred_imp, oracle_util)
        rho_imp_deltaq, p_imp_deltaq = _safe_spearmanr(pred_imp, delta_q)
        rho_util_rgb, p_util_rgb = _safe_spearmanr(pred_util, oracle_rgb)
        rho_util_depth, p_util_depth = _safe_spearmanr(pred_util, oracle_depth)
        rho_util_global, p_util_global = _safe_spearmanr(pred_util, oracle_util_global)
        
        n = len(visible)
        imp_ranks = np.argsort(-pred_imp)
        util_ranks = np.argsort(-pred_util)
        oracle_ranks = np.argsort(-oracle_util)
        
        overlaps = {}
        realized_gains = {}
        regrets = {}
        regrets_abs = {}
        ose_metrics = {}
        ndcg_metrics = {}
        lifts = {}
        coverages = {}
        total_positive_gain = float(np.sum(np.maximum(0.0, delta_q)))
        
        for k_pct in [0.05, 0.10, 0.20]:
            k = max(1, int(n * k_pct))
            tag = f'top_{int(k_pct*100)}pct'
            top_k_util = set(util_ranks[:k].tolist())
            top_k_oracle = set(oracle_ranks[:k].tolist())
            
            overlaps[tag] = len(top_k_util & top_k_oracle) / k
            
            gain_util = delta_q[list(top_k_util)].sum()
            gain_oracle = delta_q[list(top_k_oracle)].sum()
            gain_ratio = float(gain_util / (gain_oracle + 1e-8)) if gain_oracle > 0 else 1.0
            
            realized_gains[f'{tag}_ratio'] = gain_ratio
            ose_metrics[tag] = gain_ratio
            regrets[tag] = max(0.0, 1.0 - gain_ratio)
            regrets_abs[tag] = float(gain_oracle - gain_util)
            ndcg_metrics[tag] = _compute_ndcg(pred_util, oracle_util, k)
            
            gain_random = float(np.mean(delta_q) * k)
            lifts[tag] = float(gain_util / (gain_random + 1e-8)) if gain_random > 0 else 1.0
            coverages[tag] = float(
                np.sum(np.maximum(0.0, delta_q[list(top_k_util)])) / (total_positive_gain + 1e-8)
            )
            
        return {
            'n_visible': len(visible),
            'n_total': len(results),
            'spearman_utility_vs_oracle': float(rho_util_oracle),
            'spearman_utility_p': float(p_util_oracle),
            'spearman_importance_vs_oracle': float(rho_imp_oracle),
            'spearman_importance_p': float(p_imp_oracle),
            'spearman_importance_vs_deltaQ': float(rho_imp_deltaq),
            'spearman_deltaQ_p': float(p_imp_deltaq),
            'spearman_utility_vs_rgb': float(rho_util_rgb),
            'spearman_utility_vs_depth': float(rho_util_depth),
            'spearman_utility_vs_oracle_global': float(rho_util_global),
            'spearman_utility_global_p': float(p_util_global),
            'overlaps': overlaps,
            'realized_gains': realized_gains,
            'ose_metrics': ose_metrics,
            'ndcg_metrics': ndcg_metrics,
            'regrets': regrets,
            'regrets_abs': regrets_abs,
            'lifts': lifts,
            'coverages': coverages,
            'delta_quality_stats': {
                'mean': float(np.mean(delta_q)),
                'std': float(np.std(delta_q)),
                'min': float(np.min(delta_q)),
                'max': float(np.max(delta_q)),
            }
        }

    def export_oracle_dataset(self, results: List[Dict], save_path: str):
        """Export tabular dataset rows X_i → Y_i for Offline Learned Utility modeling (Point 38, Step 2)."""
        import pandas as pd
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        csv_path = os.path.splitext(save_path)[0] + '.csv'
        flat_rows = []
        for r in results:
            flat = {
                'seed': r.get('seed', 42),
                'scene': r.get('scene', 'scene'),
                'frame': r.get('frame', 0),
                'split': r.get('split', 'train'),
                'gaussian_id': r.get('gaussian_id', -1),
                'persistent_id': r.get('persistent_id', r.get('gaussian_id', -1)),
                'population': r.get('population', ''),
                'geometry_stratum': r.get('geometry_stratum', 'none'),
                'group_size': r.get('group_size', 1),
                # Primary Global Metrics (3-FIX-1)
                'psnr_before': r.get('psnr_before', 0.0),
                'psnr_after': r.get('psnr_after', 0.0),
                'delta_psnr': r.get('delta_psnr', 0.0),
                'ssim_before': r.get('ssim_before', 0.0),
                'ssim_after': r.get('ssim_after', 0.0),
                'delta_ssim': r.get('delta_ssim', 0.0),
                'depth_before': r.get('depth_before', 0.0),
                'depth_after': r.get('depth_after', 0.0),
                'delta_depth': r.get('delta_depth', 0.0),
                'loss_before': r.get('loss_before', 0.0),
                'loss_after': r.get('loss_after', 0.0),
                'delta_loss': r.get('delta_loss', 0.0),
                'delta_quality': r.get('delta_quality', 0.0),
                'delta_time_ms': r.get('delta_time_ms', r.get('measured_trial_cost_ms', 0.0)),
                'oracle_utility_joint': r.get('oracle_utility_joint', r.get('oracle_utility', 0.0)),
                'oracle_utility_rgb': r.get('oracle_utility_rgb', 0.0),
                'oracle_utility_depth': r.get('oracle_utility_depth', 0.0),
                'oracle_utility_loss': r.get('oracle_utility_loss', 0.0),
                # Local Secondary Diagnostics
                'delta_psnr_local': r.get('delta_psnr_local', 0.0),
                'delta_ssim_local': r.get('delta_ssim_local', 0.0),
                'delta_depth_local': r.get('delta_depth_gain_local', 0.0),
                'delta_loss_local': r.get('delta_loss_local', 0.0),
                'delta_quality_local': r.get('delta_quality_local', 0.0),
                'oracle_utility_joint_local': r.get('oracle_utility_joint_local', 0.0),
                # Predictors & Diagnostics
                'predicted_importance': r.get('predicted_importance', 0.0),
                'predicted_utility': r.get('predicted_utility', 0.0),
                'modeled_marginal_cost_us': r.get('modeled_marginal_cost_us', 0.0),
                'n_influence_pixels': r.get('n_influence_pixels', 0),
                'filtered': r.get('filtered', False),
                'filter_reason': r.get('filter_reason', 'none'),
                'visible': r.get('visible', True),
            }
            if 'features' in r:
                for fk, fv in r['features'].items():
                    flat[f'feat_{fk}'] = fv
            flat_rows.append(flat)
            
        df = pd.DataFrame(flat_rows)
        df.to_csv(csv_path, index=False)
        print(f"[Dataset] Exported {len(flat_rows)} records to {save_path} and {csv_path}")
