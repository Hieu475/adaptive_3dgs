"""Oracle Utility Experiment for Adaptive 3DGS.

Core research module: measures the TRUE marginal utility of optimizing
each Gaussian by isolating its individual or group contribution.

Math:
    ΔQ_{local, i} = w_{rgb} · ΔPSNR_{local, i} + w_{depth} · ΔDepthGain_{local, i}
    
    U_i^{oracle} = ΔQ_{local, i} / (Cost_i + ε)

Terminology:
    - measured_trial_cost_ms: Wall-clock duration of the isolated single-Gaussian
      optimization trial (includes local render + backward pass).
    - modeled_marginal_cost_us: Intrinsic estimated compute workload based on
      projected footprint, visible pixels, and SH degree:
          Cost_i = a + b · ProjectedArea_i · (1 + 0.1 · SH_degree)

Sampling Populations:
    - IMPORTANCE_STRATIFIED: Stratified across high/mid/low predicted importance
    - RANDOM_VISIBLE: Uniform random subset among visible Gaussians
    - UNIFORM_VISIBLE: Spatially distributed sample across image plane

Dataset Generation:
    Exports full feature vectors X_i → Y_i to `results/oracle_dataset/` for
    offline training of lightweight Learned Utility MLPs f_θ(s_i).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import math
import numpy as np
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


class OracleUtilityExperiment:
    """Oracle Utility Experiment — Ground truth marginal utility engine."""
    
    def __init__(
        self,
        pipeline,
        n_samples: int = 150,
        n_opt_steps: int = 10,
        w_rgb: float = 0.7,
        w_depth: float = 0.3,
        seed: int = 42,
        contribution_threshold: float = 0.01,
        group_size: int = 1,
    ):
        """
        Args:
            pipeline: OnlineReconstructionPipeline instance (must be initialized)
            n_samples: number of Gaussians to evaluate per population
            n_opt_steps: gradient steps per Gaussian/group
            w_rgb: weight of photometric improvement in ΔQ (default: 0.7)
            w_depth: weight of geometric depth improvement in ΔQ (default: 0.3)
            seed: random seed for reproducibility
            contribution_threshold: minimum w_{u,i} to include pixel in local region
            group_size: Gaussians per optimization group (1=individual, >1=group)
        """
        self.pipeline = pipeline
        self.n_samples = n_samples
        self.n_opt_steps = n_opt_steps
        self.w_rgb = w_rgb
        self.w_depth = w_depth
        self.seed = seed
        self.contribution_threshold = contribution_threshold
        self.group_size = group_size
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
        snapshot['optimizer_state'] = deepcopy(self.pipeline.optimizer.state_dict())
        return snapshot
        
    def restore_state(self, snapshot: Dict):
        """Restore all parameters and optimizer from snapshot."""
        model = self.pipeline.gaussian_model
        for name, param in model.named_parameters():
            key = f'param_{name}'
            if key in snapshot:
                param.data.copy_(snapshot[key])
        for name, buf in model.named_buffers():
            key = f'buffer_{name}'
            if key in snapshot:
                buf.copy_(snapshot[key])
        self.pipeline.optimizer.load_state_dict(snapshot['optimizer_state'])

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
        """Optimize a group of Gaussians and measure RGB-D local quality improvement."""
        model = self.pipeline.gaussian_model
        opt = self.pipeline.optimizer
        H, W = rgb.shape[:2]
        device = rgb.device
        
        n_influence_pixels = int(influence_mask.sum().item())
        
        # === 1. Pre-optimization measurements ===
        with torch.no_grad():
            before_out = self._render(H, W)
            before_color = before_out['color']
            before_depth = before_out['depth']
            
            psnr_local_before = self._compute_local_psnr(before_color, rgb, influence_mask)
            depth_l1_before = self._compute_local_depth_l1(before_depth, depth, influence_mask)
        
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
        
        # === 3. Post-optimization measurements ===
        with torch.no_grad():
            after_out = self._render(H, W)
            after_color = after_out['color']
            after_depth = after_out['depth']
            
            psnr_local_after = self._compute_local_psnr(after_color, rgb, influence_mask)
            depth_l1_after = self._compute_local_depth_l1(after_depth, depth, influence_mask)
            
        delta_psnr_local = psnr_local_after - psnr_local_before
        delta_depth_gain_local = max(0.0, depth_l1_before - depth_l1_after)
        # Normalized depth gain: scale by local standard deviation or stable factor
        normalized_depth_gain = delta_depth_gain_local / (depth_l1_before + 1e-4)
        delta_quality_local = self.w_rgb * delta_psnr_local + self.w_depth * (10.0 * normalized_depth_gain)
        
        # Combined RGB-D Quality Gain
        # Normalize depth gain scale: 0.1m reduction ≈ 1.0 dB equivalent
        depth_gain_scaled = delta_depth_gain_local * 10.0
        delta_quality_local = self.w_rgb * delta_psnr_local + self.w_depth * depth_gain_scaled
        
        return {
            'delta_psnr_local': delta_psnr_local,
            'delta_depth_gain_local': delta_depth_gain_local,
            'delta_quality_local': delta_quality_local,
            'measured_trial_cost_ms': measured_trial_cost_ms,
            'n_influence_pixels': n_influence_pixels,
            'psnr_local_before': psnr_local_before,
            'psnr_local_after': psnr_local_after,
            'depth_l1_before': depth_l1_before,
            'depth_l1_after': depth_l1_after,
        }

    def sample_population(
        self,
        population_type: SamplingPopulation,
        num_gaussians: int,
        predicted_importance: torch.Tensor,
        visibility_mask: torch.Tensor,
        positions: torch.Tensor,
        n_samples: int,
    ) -> List[int]:
        """Sample Gaussian indices according to specified population strategy."""
        visible_indices = torch.where(visibility_mask)[0]
        n_vis = len(visible_indices)
        if n_vis == 0:
            return []
            
        sample_k = min(n_samples, n_vis)
        
        if population_type == SamplingPopulation.RANDOM_VISIBLE:
            perm = torch.randperm(n_vis)[:sample_k]
            return visible_indices[perm].tolist()
            
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
            
            sampled = torch.cat([high_stratum[p_h], mid_stratum[p_m], low_stratum[p_l]])
            return sampled.tolist()
            
        elif population_type == SamplingPopulation.UNIFORM_VISIBLE:
            # Spatial uniform binning based on projected position
            vis_pos = positions[visible_indices]
            z_vals = vis_pos[:, 2]
            z_order = torch.argsort(z_vals)
            sorted_vis = visible_indices[z_order]
            step = max(1, len(sorted_vis) // sample_k)
            return sorted_vis[::step][:sample_k].tolist()
            
        return visible_indices[:sample_k].tolist()

    def run_oracle_experiment(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        population_type: SamplingPopulation = SamplingPopulation.IMPORTANCE_STRATIFIED,
        sample_indices: Optional[List[int]] = None,
        scene_name: str = "scene",
        frame_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """Run full oracle utility measurement on selected population."""
        H, W = rgb.shape[:2]
        device = rgb.device
        model = self.pipeline.gaussian_model
        num_gaussians = model.num_gaussians
        
        # 1. Attribution
        with torch.no_grad():
            attr_result = self._render_with_attribution(H, W)
            contrib_indices = attr_result['contrib_indices']
            contrib_weights = attr_result['contrib_weights']
            
            # Extract 2D projected area
            means2D = getattr(model, 'positions', model.positions)
            # Estimate 2D covariance
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
            
        # 2. Predicted scores
        diagnostics = self.pipeline.get_importance_diagnostics()
        predicted_importance = diagnostics['importance']
        
        cost_estimates_us = estimate_gaussian_costs(
            screen_areas=influence_mass,
            n_gaussians=num_gaussians,
            device=device
        )
        predicted_utility = predicted_importance / (cost_estimates_us / 1000.0 + 1e-6)
        
        # 3. Sampling
        if sample_indices is None:
            sample_indices = self.sample_population(
                population_type=population_type,
                num_gaussians=num_gaussians,
                predicted_importance=predicted_importance,
                visibility_mask=visibility_mask,
                positions=model.positions,
                n_samples=self.n_samples
            )
            
        # Group indices if group_size > 1
        if self.group_size > 1:
            groups = [sample_indices[i:i + self.group_size]
                      for i in range(0, len(sample_indices), self.group_size)]
        else:
            groups = [[idx] for idx in sample_indices]
            
        results = []
        total = len(sample_indices)
        
        # 4. Isolated trial execution
        for gi, group in enumerate(groups):
            influence_mask = self._get_influence_mask(group, contrib_indices, contrib_weights)
            n_pixels = int(influence_mask.sum().item())
            
            if n_pixels == 0:
                for idx in group:
                    results.append({
                        "scene": scene_name,
                        "frame": frame_idx,
                        "gaussian_id": idx,
                        "population": population_type.value if hasattr(population_type, 'value') else str(population_type),
                        "group_size": self.group_size,
                        "predicted_importance": float(predicted_importance[idx]),
                        "predicted_utility": float(predicted_utility[idx]),
                        "delta_psnr_local": 0.0,
                        "delta_depth_gain_local": 0.0,
                        "delta_quality_local": 0.0,
                        "measured_trial_cost_ms": 0.0,
                        "modeled_marginal_cost_us": float(cost_estimates_us[idx]),
                        "oracle_utility": 0.0,
                        "influence_mass": float(influence_mass[idx]),
                        "projected_area": float(projected_area[idx]),
                        "n_influence_pixels": 0,
                        "visible": False,
                    })
                continue
                
            snapshot = self.snapshot_state()
            try:
                metrics = self.optimize_gaussian_group(
                    group, self.n_opt_steps, rgb, depth, influence_mask)
                
                trial_cost = metrics['measured_trial_cost_ms']
                delta_q = metrics['delta_quality_local']
                # Oracle utility based on trial cost
                oracle_util = delta_q / (trial_cost + 1e-6)
                
                for idx in group:
                    results.append({
                        "scene": scene_name,
                        "frame": frame_idx,
                        "gaussian_id": idx,
                        "population": population_type.value if hasattr(population_type, 'value') else str(population_type),
                        "group_size": self.group_size,
                        "predicted_importance": float(predicted_importance[idx]),
                        "predicted_utility": float(predicted_utility[idx]),
                        "delta_psnr_local": float(metrics['delta_psnr_local']),
                        "delta_depth_gain_local": float(metrics['delta_depth_gain_local']),
                        "delta_quality_local": float(delta_q),
                        "measured_trial_cost_ms": float(trial_cost / len(group)),
                        "modeled_marginal_cost_us": float(cost_estimates_us[idx]),
                        "oracle_utility": float(oracle_util),
                        "influence_mass": float(influence_mass[idx]),
                        "projected_area": float(projected_area[idx]),
                        "n_influence_pixels": n_pixels,
                        "visible": True,
                    })
            finally:
                self.restore_state(snapshot)
                
        return results

    def compute_correlation_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute comprehensive Spearman rank correlations, Top-K overlap, and realized gain."""
        visible = [r for r in results if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
        if len(visible) < 5:
            return {'error': 'Insufficient visible Gaussians for statistical evaluation', 'n_visible': len(visible)}
            
        pred_imp = np.array([r['predicted_importance'] for r in visible])
        pred_util = np.array([r['predicted_utility'] for r in visible])
        oracle_util = np.array([r['oracle_utility'] for r in visible])
        delta_q = np.array([r['delta_quality_local'] for r in visible])
        delta_psnr = np.array([r['delta_psnr_local'] for r in visible])
        
        rho_util_oracle, p_util_oracle = spearmanr(pred_util, oracle_util)
        rho_imp_oracle, p_imp_oracle = spearmanr(pred_imp, oracle_util)
        rho_imp_deltaq, p_imp_deltaq = spearmanr(pred_imp, delta_q)
        
        n = len(visible)
        imp_ranks = np.argsort(-pred_imp)
        oracle_ranks = np.argsort(-oracle_util)
        
        overlaps = {}
        realized_gains = {}
        regrets = {}
        for k_pct in [0.05, 0.10, 0.20]:
            k = max(1, int(n * k_pct))
            top_k_imp = set(imp_ranks[:k].tolist())
            top_k_oracle = set(oracle_ranks[:k].tolist())
            
            overlaps[f'top_{int(k_pct*100)}pct'] = len(top_k_imp & top_k_oracle) / k
            
            gain_imp = delta_q[list(top_k_imp)].sum()
            gain_oracle = delta_q[list(top_k_oracle)].sum()
            gain_ratio = float(gain_imp / (gain_oracle + 1e-8))
            realized_gains[f'top_{int(k_pct*100)}pct_ratio'] = gain_ratio
            regrets[f'top_{int(k_pct*100)}pct'] = max(0.0, 1.0 - gain_ratio)
            
            gain_random = float(np.mean(delta_q) * k)
            lifts[f'top_{int(k_pct*100)}pct'] = float(gain_imp / (gain_random + 1e-8)) if gain_random > 0 else 1.0
            
        return {
            'n_visible': len(visible),
            'n_total': len(results),
            'spearman_utility_vs_oracle': float(rho_util_oracle),
            'spearman_utility_p': float(p_util_oracle),
            'spearman_importance_vs_oracle': float(rho_imp_oracle),
            'spearman_importance_p': float(p_imp_oracle),
            'spearman_importance_vs_deltaQ': float(rho_imp_deltaq),
            'spearman_deltaQ_p': float(p_imp_deltaq),
            'overlaps': overlaps,
            'realized_gains': realized_gains,
            'regrets': regrets,
            'lifts': lifts,
            'delta_quality_stats': {
                'mean': float(np.mean(delta_q)),
                'std': float(np.std(delta_q)),
                'min': float(np.min(delta_q)),
                'max': float(np.max(delta_q)),
            }
        }

    def export_oracle_dataset(self, results: List[Dict], save_path: str):
        """Export tabular dataset rows X_i → Y_i for Offline Learned Utility modeling."""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
