"""Oracle Utility Experiment for Adaptive 3DGS.

Core research experiment: measure TRUE marginal utility of optimizing
each Gaussian by isolating its quality contribution.

Key insight from v1 results:
    Global ΔPSNR ≈ constant when optimizing 1/N Gaussians
    → Must measure LOCAL quality change at pixels where the Gaussian contributes.

Oracle utility:
    U_i^oracle = ΔQ_local_i / (Cost_i + ε)

where ΔQ_local_i is measured ONLY at pixels where Gaussian i has
significant contribution weight (w_{u,i} > threshold).

Three measurement modes:
    1. LOCAL: ΔPSNR only at pixels with w_{u,i} > ε (primary)  
    2. GLOBAL: ΔPSNR over full image (too noisy for single Gaussians)
    3. GROUP: Optimize clusters of nearby Gaussians together
"""
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr
from copy import deepcopy

from .rasterizer import render as rasterize_scene
from .attribution import render_with_attribution, compute_gaussian_statistics
from .scheduler import estimate_gaussian_costs


class OracleUtilityExperiment:
    """Oracle Utility Experiment — measures true marginal value of each Gaussian.
    
    Workflow:
        1. Render with attribution to get per-pixel Gaussian contributions
        2. For each sampled Gaussian i:
            a. Identify its "influence region" (pixels where w_{u,i} > threshold)
            b. Snapshot state
            c. Optimize only Gaussian i for N steps
            d. Re-render, measure LOCAL quality change in influence region
            e. Compute oracle utility = ΔQ_local / cost
            f. Restore state
        3. Compare predicted utility vs oracle utility
    """
    
    def __init__(
        self,
        pipeline,
        n_samples: int = 200,
        n_opt_steps: int = 10,
        seed: int = 42,
        contribution_threshold: float = 0.01,
        group_size: int = 1,
    ):
        """
        Args:
            pipeline: OnlineReconstructionPipeline instance (must be initialized)
            n_samples: number of Gaussians to evaluate
            n_opt_steps: gradient steps per Gaussian/group
            seed: random seed for reproducibility
            contribution_threshold: minimum w_{u,i} to include pixel in local region
            group_size: Gaussians per optimization group (1=individual, >1=group)
        """
        self.pipeline = pipeline
        self.n_samples = n_samples
        self.n_opt_steps = n_opt_steps
        self.seed = seed
        self.contribution_threshold = contribution_threshold
        self.group_size = group_size
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def snapshot_state(self) -> Dict:
        """Save all Gaussian parameters and optimizer state (deep copy)."""
        model = self.pipeline.gaussian_model
        snapshot = {}
        # Save all parameters
        for name, param in model.named_parameters():
            snapshot[f'param_{name}'] = param.data.clone()
        # Save buffers
        for name, buf in model.named_buffers():
            snapshot[f'buffer_{name}'] = buf.clone()
        # Save optimizer state
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
        """Compute PSNR only within masked region.
        
        Args:
            rendered: (H, W, 3) rendered color
            gt: (H, W, 3) ground truth color
            pixel_mask: (H, W) boolean mask
            
        Returns:
            PSNR in dB within the masked region. 0.0 if mask is empty.
        """
        if pixel_mask.sum() == 0:
            return 0.0
        mse = ((rendered[pixel_mask] - gt[pixel_mask]) ** 2).mean().item()
        if mse < 1e-10:
            return 50.0
        return -10.0 * math.log10(mse)

    def _compute_local_depth_error(
        self,
        rendered_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> float:
        """Compute mean absolute depth error in masked region."""
        valid = pixel_mask & (gt_depth > 0)
        if valid.sum() == 0:
            return 0.0
        return (rendered_depth[valid] - gt_depth[valid]).abs().mean().item()

    def _get_influence_mask(
        self,
        gaussian_indices: List[int],
        contrib_indices: torch.Tensor,
        contrib_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Get pixel mask where given Gaussians have significant contribution.
        
        Args:
            gaussian_indices: list of Gaussian indices to check
            contrib_indices: (H, W, top_k) contributing Gaussian indices per pixel
            contrib_weights: (H, W, top_k) contribution weights per pixel
            
        Returns:
            pixel_mask: (H, W) boolean — True where any of the given Gaussians
                        contributes with weight > threshold
        """
        H, W, K = contrib_indices.shape
        mask = torch.zeros(H, W, dtype=torch.bool, device=contrib_indices.device)
        
        for idx in gaussian_indices:
            # Check if this Gaussian appears in any pixel's top-K contributors
            idx_match = (contrib_indices == idx)  # (H, W, K) bool
            weight_ok = (contrib_weights > self.contribution_threshold)  # (H, W, K)
            significant = (idx_match & weight_ok).any(dim=-1)  # (H, W)
            mask |= significant
        
        return mask

    def optimize_gaussian_group(
        self,
        indices: List[int],
        n_steps: int,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        influence_mask: torch.Tensor,
    ) -> Dict:
        """Optimize a group of Gaussians and measure LOCAL quality change.
        
        Args:
            indices: list of Gaussian indices to optimize
            n_steps: number of gradient steps
            rgb: (H, W, 3) GT color
            depth: (H, W) GT depth
            influence_mask: (H, W) boolean mask of pixels influenced by these Gaussians
            
        Returns:
            Dict with delta_psnr_local, delta_psnr_global, delta_depth_local, cost_ms,
                  n_influence_pixels
        """
        model = self.pipeline.gaussian_model
        opt = self.pipeline.optimizer
        H, W = rgb.shape[:2]
        
        n_influence_pixels = int(influence_mask.sum().item())
        
        # === Before optimization: measure quality ===
        with torch.no_grad():
            before_out = self._render(H, W)
            before_color = before_out['color']
            before_depth = before_out['depth']
            
            psnr_local_before = self._compute_local_psnr(before_color, rgb, influence_mask)
            psnr_global_before = self._compute_local_psnr(
                before_color, rgb, torch.ones(H, W, dtype=torch.bool, device=rgb.device))
            depth_local_before = self._compute_local_depth_error(
                before_depth, depth, influence_mask)
        
        # === Optimization: update only selected Gaussians ===
        opt_mask = torch.zeros(model.num_gaussians, dtype=torch.bool, device=rgb.device)
        opt_mask[indices] = True
        
        start_time = time.time()
        for step in range(n_steps):
            opt.zero_grad()
            out = self._render(H, W)
            
            # Loss: can use local loss for stronger signal
            if n_influence_pixels > 0 and n_influence_pixels < H * W * 0.8:
                # Local loss: MSE only at influenced pixels
                loss = ((out['color'][influence_mask] - rgb[influence_mask]) ** 2).mean()
            else:
                # Fallback to global loss
                loss = ((out['color'] - rgb) ** 2).mean()
            
            loss.backward()
            
            # Zero gradients for non-selected Gaussians
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None and param.shape[0] == model.num_gaussians:
                        param.grad[~opt_mask] = 0.0
            
            opt.step()
        
        cost_ms = (time.time() - start_time) * 1000.0
        
        # === After optimization: measure quality ===
        with torch.no_grad():
            after_out = self._render(H, W)
            after_color = after_out['color']
            after_depth = after_out['depth']
            
            psnr_local_after = self._compute_local_psnr(after_color, rgb, influence_mask)
            psnr_global_after = self._compute_local_psnr(
                after_color, rgb, torch.ones(H, W, dtype=torch.bool, device=rgb.device))
            depth_local_after = self._compute_local_depth_error(
                after_depth, depth, influence_mask)
        
        return {
            'delta_psnr_local': psnr_local_after - psnr_local_before,
            'delta_psnr_global': psnr_global_after - psnr_global_before,
            'delta_depth_local': depth_local_before - depth_local_after,  # positive = improved
            'cost_ms': cost_ms,
            'n_influence_pixels': n_influence_pixels,
            'psnr_local_before': psnr_local_before,
            'psnr_local_after': psnr_local_after,
        }

    def run_oracle_experiment(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        sample_indices: Optional[List[int]] = None,
    ) -> List[Dict]:
        """Run full oracle utility experiment.
        
        Steps:
            1. Render with attribution to get per-Gaussian contribution maps
            2. Get predicted importance from heuristic
            3. For each sampled Gaussian:
                a. Find its influence region
                b. Snapshot → optimize → measure local ΔQ → restore
            4. Return per-Gaussian oracle results
            
        Args:
            rgb: (H, W, 3) ground truth color
            depth: (H, W) ground truth depth
            sample_indices: optional specific Gaussian indices to evaluate
            
        Returns:
            List of per-Gaussian result dicts
        """
        H, W = rgb.shape[:2]
        device = rgb.device
        model = self.pipeline.gaussian_model
        num_gaussians = model.num_gaussians
        
        # === Step 1: Render with attribution ===
        print("  [Oracle] Rendering with attribution...")
        with torch.no_grad():
            attr_result = self._render_with_attribution(H, W)
            contrib_indices = attr_result['contrib_indices']  # (H, W, top_k)
            contrib_weights = attr_result['contrib_weights']  # (H, W, top_k)
        
        # === Step 2: Get predicted importance ===
        diagnostics = self.pipeline.get_importance_diagnostics()
        predicted_importance = diagnostics['importance']  # (N,)
        
        # Get per-Gaussian cost estimates
        screen_areas = diagnostics.get('screen_area', None)
        cost_estimates = estimate_gaussian_costs(
            screen_areas=screen_areas,
            n_gaussians=num_gaussians,
            device=device,
        )
        # Compute predicted utility = importance / cost
        predicted_utility = predicted_importance / (cost_estimates + 1e-8)
        
        # === Step 3: Sample Gaussians ===
        if sample_indices is None:
            n_sample = min(self.n_samples, num_gaussians)
            # Stratified sampling: mix high/mid/low importance Gaussians
            sorted_idx = torch.argsort(predicted_importance, descending=True)
            n_high = n_sample // 3
            n_mid = n_sample // 3
            n_low = n_sample - n_high - n_mid
            
            high_idx = sorted_idx[:max(n_high * 3, 1)]
            mid_start = len(sorted_idx) // 3
            mid_idx = sorted_idx[mid_start:mid_start + max(n_mid * 3, 1)]
            low_idx = sorted_idx[-max(n_low * 3, 1):]
            
            # Random subsample from each stratum
            perm_h = torch.randperm(len(high_idx))[:n_high]
            perm_m = torch.randperm(len(mid_idx))[:n_mid]
            perm_l = torch.randperm(len(low_idx))[:n_low]
            
            sample_indices = torch.cat([
                high_idx[perm_h], mid_idx[perm_m], low_idx[perm_l]
            ]).tolist()
        
        # === Step 4: Oracle measurement loop ===
        print(f"  [Oracle] Evaluating {len(sample_indices)} Gaussians "
              f"(group_size={self.group_size}, n_steps={self.n_opt_steps})...")
        
        results = []
        total = len(sample_indices)
        
        # Group indices if group_size > 1
        if self.group_size > 1:
            groups = [sample_indices[i:i+self.group_size] 
                     for i in range(0, len(sample_indices), self.group_size)]
        else:
            groups = [[idx] for idx in sample_indices]
        
        for gi, group in enumerate(groups):
            # Find influence region for this group
            influence_mask = self._get_influence_mask(
                group, contrib_indices, contrib_weights)
            
            n_pixels = int(influence_mask.sum().item())
            
            # Skip if Gaussian has zero influence (not visible)
            if n_pixels == 0:
                for idx in group:
                    results.append({
                        "gaussian_id": idx,
                        "predicted_utility": float(predicted_utility[idx]),
                        "predicted_importance": float(predicted_importance[idx]),
                        "delta_psnr_local": 0.0,
                        "delta_psnr_global": 0.0,
                        "delta_depth_local": 0.0,
                        "cost_ms": 0.0,
                        "oracle_utility": 0.0,
                        "n_influence_pixels": 0,
                        "visible": False,
                    })
                continue
            
            # Snapshot → optimize → measure → restore
            snapshot = self.snapshot_state()
            try:
                metrics = self.optimize_gaussian_group(
                    group, self.n_opt_steps, rgb, depth, influence_mask)
                
                cost = metrics['cost_ms']
                delta_q = metrics['delta_psnr_local']
                oracle_util = delta_q / (cost + 1e-6)
                
                for idx in group:
                    results.append({
                        "gaussian_id": idx,
                        "predicted_utility": float(predicted_utility[idx]),
                        "predicted_importance": float(predicted_importance[idx]),
                        "delta_psnr_local": float(delta_q),
                        "delta_psnr_global": float(metrics['delta_psnr_global']),
                        "delta_depth_local": float(metrics['delta_depth_local']),
                        "cost_ms": float(cost / len(group)),  # per-Gaussian cost
                        "oracle_utility": float(oracle_util),
                        "n_influence_pixels": n_pixels,
                        "visible": True,
                    })
            finally:
                self.restore_state(snapshot)
            
            # Progress
            done = min((gi + 1) * self.group_size, total)
            if done % max(1, total // 10) < self.group_size or done == total:
                print(f"    [{done}/{total}] "
                      f"ΔQ_local={delta_q:+.4f} dB | "
                      f"pixels={n_pixels} | "
                      f"cost={cost:.1f} ms")
        
        return results

    def compute_correlation_metrics(self, results: List[Dict]) -> Dict:
        """Compute Spearman correlation, top-K overlap, and realized gains.
        
        Metrics:
            1. Spearman ρ(U_pred, U_oracle) — rank correlation
            2. Overlap@K at 5%, 10%, 20% — set intersection of top-K
            3. Realized gain ratio — actual ΔQ from predicted top-K vs oracle top-K
        """
        # Filter to visible Gaussians only
        visible = [r for r in results if r.get('visible', True) and r['n_influence_pixels'] > 0]
        
        if len(visible) < 5:
            return {'error': 'Too few visible Gaussians for correlation', 'n_visible': len(visible)}
        
        pred = np.array([r['predicted_utility'] for r in visible])
        pred_imp = np.array([r['predicted_importance'] for r in visible])
        oracle = np.array([r['oracle_utility'] for r in visible])
        delta_psnr = np.array([r['delta_psnr_local'] for r in visible])
        
        # === Metric 1: Spearman correlation ===
        rho_utility, p_utility = spearmanr(pred, oracle)
        rho_importance, p_importance = spearmanr(pred_imp, oracle)
        # Also correlate with raw delta_psnr (ignoring cost)
        rho_delta_q, p_delta_q = spearmanr(pred_imp, delta_psnr)
        
        # === Metric 2: Top-K overlap ===
        n = len(visible)
        pred_ranks = np.argsort(-pred)
        oracle_ranks = np.argsort(-oracle)
        imp_ranks = np.argsort(-pred_imp)
        
        overlaps = {}
        realized_gains = {}
        
        for k_pct in [0.05, 0.10, 0.20]:
            k = max(1, int(n * k_pct))
            
            # Overlap: predicted utility vs oracle
            top_k_pred = set(pred_ranks[:k].tolist())
            top_k_oracle = set(oracle_ranks[:k].tolist())
            top_k_imp = set(imp_ranks[:k].tolist())
            
            overlaps[f'utility_top_{int(k_pct*100)}pct'] = len(top_k_pred & top_k_oracle) / k
            overlaps[f'importance_top_{int(k_pct*100)}pct'] = len(top_k_imp & top_k_oracle) / k
            
            # === Metric 3: Realized quality gain ===
            gain_pred = delta_psnr[list(top_k_pred)].sum()
            gain_oracle = delta_psnr[list(top_k_oracle)].sum()
            gain_imp = delta_psnr[list(top_k_imp)].sum()
            
            realized_gains[f'utility_top_{int(k_pct*100)}pct_ratio'] = float(
                gain_pred / (gain_oracle + 1e-8))
            realized_gains[f'importance_top_{int(k_pct*100)}pct_ratio'] = float(
                gain_imp / (gain_oracle + 1e-8))
        
        return {
            'n_visible': len(visible),
            'n_total': len(results),
            'spearman_utility_vs_oracle': float(rho_utility),
            'spearman_utility_p': float(p_utility),
            'spearman_importance_vs_oracle': float(rho_importance),
            'spearman_importance_p': float(p_importance),
            'spearman_importance_vs_deltaQ': float(rho_delta_q),
            'spearman_deltaQ_p': float(p_delta_q),
            'overlaps': overlaps,
            'realized_gains': realized_gains,
            'delta_psnr_stats': {
                'mean': float(np.mean(delta_psnr)),
                'std': float(np.std(delta_psnr)),
                'min': float(np.min(delta_psnr)),
                'max': float(np.max(delta_psnr)),
                'range': float(np.max(delta_psnr) - np.min(delta_psnr)),
            },
        }

    def save_results(self, results: List[Dict], path: str):
        """Save oracle results to JSON."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
