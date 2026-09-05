"""Online Reconstruction Pipeline for Adaptive 3D Gaussian Splatting.

Ties together all research modules into a complete per-frame pipeline:
    initialize → [for each frame: track → render → errors → densify → schedule → optimize → prune]
"""
import torch
import torch.optim as optim
import time
from typing import Dict, Optional, Any
import yaml

from .gaussian_repr import GaussianModel, GaussianState
from .projection import world_to_camera, project_to_screen, compute_2d_covariance
from .rasterizer import render as rasterize_scene
from .losses import total_loss, color_loss, depth_loss
from .depth_render import render_depth_surface_aware
from .importance import GaussianImportanceEstimator, Tier
from .attribution import render_with_attribution, compute_gaussian_statistics
from .scheduler import BudgetScheduler, OptimizationPolicy, estimate_gaussian_costs
from .densification import (
    compute_error_masks, sample_candidates,
    create_gaussians_from_candidates, prune_low_value
)
from .tracker import ICPTracker
from .background_cache import FrozenBackgroundCache
from .selective_optimizer import SelectiveAdam
from enum import Enum


class ExecutionMode(str, Enum):
    """Pipeline execution mode."""
    FULL = "full"             # Ground-truth quality upper bound
    SELECTIVE = "selective"   # Production adaptive policy
    ORACLE = "oracle"         # Research evaluator


class OnlineReconstructionPipeline:
    """Main online reconstruction pipeline for Adaptive 3D Gaussian Splatting.
    
    Manages the full lifecycle: initialization from first frame,
    per-frame tracking + mapping, and metric collection.
    """
    
    @classmethod
    def _merge_config(cls, base: Dict, update: Optional[Dict]) -> Dict:
        """Recursively merge update dict into base dict."""
        if not update:
            return base
        merged = base.copy()
        for k, v in update.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = cls._merge_config(merged[k], v)
            else:
                merged[k] = v
        return merged

    def __init__(self, config: Optional[Dict] = None, device: str = 'cpu'):
        """Initialize pipeline.
        
        Args:
            config: configuration dict (from YAML file)
            device: 'cpu' or 'cuda'
        """
        self.config = self._merge_config(self._default_config(), config)
        self.device = device
        
        # Core modules
        self.gaussian_model = GaussianModel(
            sh_degree=self.config['gaussian']['sh_degree'],
            device=device,
        )
        self.tracker = ICPTracker()
        self.importance_estimator = GaussianImportanceEstimator(
            weights=self.config.get('importance', {}),
            tau_high=self.config['scheduler']['tier_thresholds'][0],
            tau_low=self.config['scheduler']['tier_thresholds'][2],
        )
        self.scheduler = BudgetScheduler(
            gpu_budget_ms=self.config['scheduler']['gpu_budget_ms'],
        )
        
        # Optimizer (initialized after first frame)
        self.optimizer: Optional[optim.Adam] = None
        
        # Frozen Background Cache for True Selective Optimization
        self.bg_cache = FrozenBackgroundCache(device=device)
        
        # State
        self.frame_count = 0
        self.current_pose = torch.eye(4, device=device)
        self.intrinsics: Optional[torch.Tensor] = None
        self.initialized = False
        
        # Metrics collection
        self.metrics_history = []
        
        # Custom selector function (Phase 5: Policy Path Unification)
        # Allows external policy (e.g. select_budget_constrained_subset) to run
        # at Step 6 with frame-t state, ensuring Stage A and Stage B share identical policy semantics.
        # Signature: selector_fn(pipeline: OnlineReconstructionPipeline, N: int) -> torch.Tensor (bool mask)
        self._custom_selector_fn = None
        self._pre_scheduling_hook = None
    
    @staticmethod
    def _default_config() -> Dict:
        return {
            'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 500000},
            'rendering': {
                'tile_size': 16,
                'image_width': 640,
                'image_height': 480,
                'use_surface_aware_depth': True,
                'depth_threshold_opaque': 0.5,
                'attribution_top_k': 8,
            },
            'losses': {'weight_color': 1.0, 'weight_depth': 0.5, 'weight_normal': 0.1, 'weight_regularization': 0.01},
            'importance': {'depth_error': 1.0, 'color_error': 1.0, 'normal_error': 0.5, 'visibility': 0.1, 'temporal': 0.5, 'screen_space': 0.2},
            'scheduler': {
                'gpu_budget_ms': 16.6,
                'tier_thresholds': [0.8, 0.5, 0.2],
                'optimize_every_n_frames': 5,
                'policy': 'budget_aware',
                'optimize_ratio': 0.5,
                'cost_per_gaussian_us': 0.5,
            },
            'densification': {
                'max_new_per_frame': 500,
                'strategy': 'importance',
                'use_adaptive_thresholds': True,
                'adaptive_k': 2.0,
                'error_threshold_color': 0.1,
                'error_threshold_depth': 0.05,
                'transmission_threshold': 0.5,
                'lambda_color': 1.0,
                'lambda_depth': 1.0,
                'lambda_transmission': 0.5,
            },
            'training': {'learning_rate': {'position': 1.6e-4, 'scale': 5e-3, 'rotation': 1e-3, 'opacity': 5e-2, 'sh': 2.5e-3}},
        }
    
    @classmethod
    def from_config_file(cls, config_path: str, device: str = 'cpu') -> 'OnlineReconstructionPipeline':
        """Create pipeline from YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config=config, device=device)
    
    def _setup_optimizer(self):
        """Setup Adam optimizer with per-parameter learning rates."""
        lr = self.config['training']['learning_rate']
        params = [
            {'params': [self.gaussian_model._xyz], 'lr': lr.get('position', 1.6e-4)},
            {'params': [self.gaussian_model._scaling], 'lr': lr.get('scale', 5e-3)},
            {'params': [self.gaussian_model._rotation], 'lr': lr.get('rotation', 1e-3)},
            {'params': [self.gaussian_model._opacity], 'lr': lr.get('opacity', 5e-2)},
            {'params': [self.gaussian_model._features_dc], 'lr': lr.get('sh', 2.5e-3)},
            {'params': [self.gaussian_model._features_rest], 'lr': lr.get('sh', 2.5e-3)},
            {'params': [self.gaussian_model._normals], 'lr': lr.get('position', 1.6e-4)},
        ]
        self.optimizer = SelectiveAdam(params)
    
    def _reregister_optimizer_params(self):
        """Re-register model parameters with the optimizer after compaction.
        
        After compact() creates new nn.Parameter objects, the optimizer's param_groups
        reference stale parameter objects. This method updates the references while
        preserving the accumulated momentum and variance state.
        """
        if self.optimizer is None:
            return
        
        new_params = [
            self.gaussian_model._xyz,
            self.gaussian_model._scaling,
            self.gaussian_model._rotation,
            self.gaussian_model._opacity,
            self.gaussian_model._features_dc,
            self.gaussian_model._features_rest,
            self.gaussian_model._normals,
        ]
        
        for group, new_p in zip(self.optimizer.param_groups, new_params):
            old_p = group['params'][0]
            if old_p in self.optimizer.state:
                self.optimizer.state[new_p] = self.optimizer.state.pop(old_p)
            group['params'] = [new_p]
    
    def initialize(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
    ):
        """Initialize Gaussian map from first RGB-D frame.
        
        Args:
            rgb: (H, W, 3) RGB image in [0, 1]
            depth: (H, W) depth map
            intrinsics: (3, 3) camera intrinsic matrix
            pose: (4, 4) initial camera pose (default: identity)
        """
        self.intrinsics = intrinsics.to(self.device)
        self.current_pose = (pose if pose is not None else torch.eye(4)).to(self.device)
        
        H, W = depth.shape
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        
        # Subsample pixels to create initial Gaussians
        stride = 4  # Every 4th pixel
        v_coords = torch.arange(0, H, stride, device=self.device)
        u_coords = torch.arange(0, W, stride, device=self.device)
        vv, uu = torch.meshgrid(v_coords, u_coords, indexing='ij')
        uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)  # (K, 2)
        
        # Filter by valid depth
        d_vals = depth[uv[:, 1].long(), uv[:, 0].long()]
        valid = d_vals > 0
        uv = uv[valid]
        
        # Unproject to 3D
        from .densification import unproject_pixels
        points = unproject_pixels(uv, depth, intrinsics, self.current_pose)
        
        # Get colors
        colors = rgb[uv[:, 1].long(), uv[:, 0].long()]
        
        # Initialize Gaussians
        self.gaussian_model.initialize_from_points(
            points, colors=colors,
            initial_scale=self.config['gaussian'].get('initial_scale', 0.01),
            initial_opacity=self.config['gaussian']['initial_opacity'],
        )
        
        self._setup_optimizer()
        self.initialized = True
        self.frame_count = 1
        
        print(f"[Init] Created {self.gaussian_model.num_gaussians} Gaussians from first frame")
    
    def process_frame(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        gt_pose: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Process a single RGB-D frame through the full pipeline.
        
        Pipeline: track → render → compute errors → densify → schedule → optimize → prune
        
        Args:
            rgb: (H, W, 3) RGB image in [0, 1]
            depth: (H, W) depth map
            gt_pose: (4, 4) optional ground truth pose (skip tracking)
        
        Returns:
            Dict with per-frame metrics
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        frame_start = time.time()
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        H, W = depth.shape
        
        # === 1. Camera Tracking ===
        if gt_pose is not None:
            self.current_pose = gt_pose.to(self.device)
        else:
            self.current_pose = self.tracker.track_frame(rgb, depth, self.gaussian_model).to(self.device)
        
        # === 2. Render Current Map (with per-Gaussian attribution) ===
        with torch.no_grad():
            cov3D = self.gaussian_model.build_covariance()
            render_result = render_with_attribution(
                means3D=self.gaussian_model.positions,
                cov3D=cov3D,
                colors=self.gaussian_model.get_colors(),
                opacities=self.gaussian_model.opacities.squeeze(-1),
                extrinsics=self.current_pose,
                intrinsics=self.intrinsics,
                image_width=W,
                image_height=H,
                tile_size=self.config['rendering']['tile_size'],
                top_k=self.config['rendering'].get('attribution_top_k', 8),
            )
            
            rendered_color = render_result['color']  # (H, W, 3)
            transmission = render_result['transmission']  # (H, W)
            
            if self.config['rendering'].get('use_surface_aware_depth', True):
                depth_result = render_depth_surface_aware(
                    means3D=self.gaussian_model.positions,
                    normals=self.gaussian_model._normals,
                    opacities=self.gaussian_model.opacities.squeeze(-1),
                    cov3D=cov3D,
                    extrinsics=self.current_pose,
                    intrinsics=self.intrinsics,
                    image_width=W,
                    image_height=H,
                    opacity_threshold=self.config['rendering'].get('depth_threshold_opaque', 0.5),
                    tile_size=self.config['rendering']['tile_size'],
                )
                rendered_depth = depth_result['depth']  # (H, W)
            else:
                rendered_depth = render_result['depth']  # (H, W)
        
        # === 3. Per-Gaussian Error Attribution ===
        # Compute true per-Gaussian statistics from pixel-level contributions
        N = self.gaussian_model.num_gaussians
        
        gaussian_stats = compute_gaussian_statistics(
            rendered_color=rendered_color,
            rendered_depth=rendered_depth,
            gt_color=rgb,
            gt_depth=depth,
            contrib_weights=render_result['contrib_weights'],
            contrib_indices=render_result['contrib_indices'],
            n_gaussians=N,
        )
        
        per_gaussian_color_err = gaussian_stats['color_error']      # (N,)
        per_gaussian_depth_err = gaussian_stats['depth_error']      # (N,)
        visibility_mask = gaussian_stats['visibility_mask']          # (N,) bool
        per_gaussian_screen_area = gaussian_stats['screen_area']    # (N,)
        
        # Pixel-level errors for densification masks
        color_err = (rendered_color - rgb).abs().mean(dim=-1)  # (H, W)
        depth_valid = depth > 0
        depth_err = torch.zeros_like(depth)
        depth_err[depth_valid] = (rendered_depth[depth_valid] - depth[depth_valid]).abs()
        
        # === 4. Importance Estimation ===
        self.importance_estimator.update_statistics(
            depth_errors=per_gaussian_depth_err,
            color_errors=per_gaussian_color_err,
            normal_errors=None,
            visibility_mask=visibility_mask,
            positions=self.gaussian_model.positions.detach(),
            screen_areas=per_gaussian_screen_area,
        )
        importance = self.importance_estimator.compute_importance()
        # Update Gaussian confidence from importance feedback
        if hasattr(self.gaussian_model, '_confidence'):
            new_confidence = self.importance_estimator.update_confidence(
                self.gaussian_model._confidence,
                importance,
            )
            self.gaussian_model._confidence.data.copy_(new_confidence)
        tiers = self.importance_estimator.classify_tier(importance)
        
        # === 5. Densification ===
        dense_cfg = self.config['densification']
        
        # Adaptive thresholds or fixed thresholds
        if dense_cfg.get('use_adaptive_thresholds', True):
            depth_thresh, color_thresh = self.scheduler.adaptive_threshold(
                depth_errors=depth_err[depth_valid] if depth_valid.any() else torch.tensor([0.05], device=self.device),
                color_errors=color_err,
                k=dense_cfg.get('adaptive_k', 2.0),
            )
        else:
            color_thresh = dense_cfg['error_threshold_color']
            depth_thresh = dense_cfg['error_threshold_depth']

        error_masks = compute_error_masks(
            color_err, depth_err, transmission,
            color_threshold=color_thresh,
            depth_threshold=depth_thresh,
            transmission_threshold=dense_cfg['transmission_threshold'],
        )
        
        max_new = min(
            dense_cfg['max_new_per_frame'],
            self.scheduler.compute_max_new_gaussians(),
            self.config['gaussian']['max_gaussians'] - self.gaussian_model.num_gaussians,
        )
        
        if max_new > 0 and error_masks['combined_mask'].any():
            candidates = sample_candidates(
                error_mask=error_masks['combined_mask'],
                num_samples=max_new,
                strategy=dense_cfg.get('strategy', 'importance'),
                color_err=color_err,
                depth_err=depth_err,
                transmission=transmission,
                lambda_color=dense_cfg.get('lambda_color', 1.0),
                lambda_depth=dense_cfg.get('lambda_depth', 1.0),
                lambda_transmission=dense_cfg.get('lambda_transmission', 0.5),
            )
            if candidates.shape[0] > 0:
                new_gaussians = create_gaussians_from_candidates(
                    candidates, rgb, depth,
                    self.intrinsics, self.current_pose,
                )
                if new_gaussians['xyz'].shape[0] > 0:
                    self.gaussian_model.add_gaussians(new_gaussians)
                    self.importance_estimator.expand_buffers(
                        new_gaussians['xyz'].shape[0], self.device
                    )
                    # Extend optimizer state continuously without destroying historical momentum
                    if self.optimizer is not None:
                        self.optimizer.extend_state(new_gaussians['xyz'].shape[0], device=self.device)
                    else:
                        self._setup_optimizer()
        
        # === 6. Budget-Aware Scheduling ===
        # Recompute importance for updated Gaussian set
        N_updated = self.gaussian_model.num_gaussians
        if importance.shape[0] != N_updated:
            # Pad importance for new Gaussians
            importance = torch.cat([
                importance,
                torch.full((N_updated - importance.shape[0],), 0.5, device=self.device)
            ])
            tiers = self.importance_estimator.classify_tier(importance)
        
        # Estimate per-Gaussian compute costs (A2: Controlled Cost Model)
        use_cost_model = self.config.get('scheduler', {}).get('use_cost_model', True)
        use_attribution = self.config.get('importance', {}).get('use_attribution', True)
        if not use_cost_model:
            cost_estimates = torch.full((N_updated,), 0.5, device=self.device)
        else:
            cost_estimates = estimate_gaussian_costs(
                screen_areas=getattr(self.importance_estimator, '_screen_areas', None) if use_attribution else None,
                n_gaussians=N_updated,
                base_cost_us=self.config['scheduler'].get('cost_per_gaussian_us', 0.5),
                sh_degree=self.gaussian_model.sh_degree,
                device=self.device,
            )
        
        policy = self.config['scheduler'].get('policy', 'budget_aware')
        use_knapsack = self.config.get('scheduler', {}).get('use_knapsack', True)
        if not use_knapsack and policy in ['budget_aware', 'ours']:
            policy = 'top_k'  # Greedy top-k utility ranking
            
        ratio = self.config['scheduler'].get('optimize_ratio', 0.5)
        
        error_scores = None
        error_influence_scores = None
        if self.importance_estimator._running_depth_error is not None and self.importance_estimator._running_color_error is not None:
            error_scores = self.importance_estimator._running_depth_error + self.importance_estimator._running_color_error
            error_influence_scores = self.importance_estimator.compute_error_influence_score()
        
        top_k = self.config['scheduler'].get('top_k', None)
        binary_threshold = self.config['scheduler'].get('binary_threshold', 0.5)

        if getattr(self, '_custom_selector_fn', None) is not None:
            optimize_mask = self._custom_selector_fn(self, N_updated)
        else:
            optimize_mask = self.scheduler.select_by_policy(
                policy=policy,
                importance_scores=importance,
                tiers=tiers,
                confidence=self.gaussian_model._confidence if hasattr(self.gaussian_model, '_confidence') else None,
                cost_estimates=cost_estimates,
                error_scores=error_scores,
                error_influence_scores=error_influence_scores,
                ratio=ratio,
                top_k=top_k,
                frame_idx=self.frame_count,
                binary_threshold=binary_threshold,
                utility_scores=getattr(self, '_learned_utility_scores', None),
            )
        self._last_optimize_mask = optimize_mask
        
        # === 7. True Selective Optimization with Frozen Background Cache (R21/R29) ===
        n_optimized = 0
        opt_loss_val = 0.0
        cache_time = 0.0
        opt_time = 0.0
        
        if optimize_mask.any() and self.optimizer is not None:
            # 1. Build / refresh background cache for frozen Gaussians once per frame
            frozen_mask = ~optimize_mask[:self.gaussian_model.num_gaussians]
            if frozen_mask.any():
                c_start = time.time()
                self.bg_cache.build_cache(
                    model=self.gaussian_model,
                    frozen_mask=frozen_mask,
                    extrinsics=self.current_pose,
                    intrinsics=self.intrinsics,
                    image_width=W,
                    image_height=H,
                    tile_size=self.config['rendering']['tile_size'],
                )
                cache_time = time.time() - c_start
            else:
                self.bg_cache.invalidate()
                
            # 2. Pure Selective Optimization Step (M only)
            opt_start = time.time()
            self.optimizer.zero_grad()
            active_subset = self.gaussian_model.get_optimization_subset(optimize_mask)
            
            composite_opt = self.bg_cache.composite_with_active(
                active_subset=active_subset,
                extrinsics=self.current_pose,
                intrinsics=self.intrinsics,
                image_width=W,
                image_height=H,
                tile_size=self.config['rendering']['tile_size'],
            )
            
            rendered_opt_color = composite_opt['color']
            rendered_opt_depth = composite_opt['depth']
            depth_valid_mask = (rendered_opt_depth > 0) & (depth > 0)
            
            # 3. Compute loss
            weights = {
                'color': self.config['losses']['weight_color'],
                'depth': self.config['losses']['weight_depth'],
            }
            losses = total_loss(
                rendered_opt_color, rgb,
                rendered_opt_depth, depth,
                weights,
                depth_valid_mask=depth_valid_mask,
            )
            
            # 4. Backward: only computes gradients for the active subset M <= N
            losses['total'].backward()
            
            # 5. Selective Optimizer update (O(M) arithmetic & memory bandwidth)
            self.optimizer.step(active_idx=active_subset['indices'])
            n_optimized = optimize_mask.sum().item()
            opt_loss_val = losses['total'].item()
            opt_time = time.time() - opt_start
        
        # === 7b. Post-Optimization Quality Assessment ===
        # Re-render after optimization to measure actual quality improvement
        if n_optimized > 0:
            with torch.no_grad():
                cov3D_post = self.gaussian_model.build_covariance()
                post_render = render_with_attribution(
                    means3D=self.gaussian_model.positions,
                    cov3D=cov3D_post,
                    colors=self.gaussian_model.get_colors(),
                    opacities=self.gaussian_model.opacities.squeeze(-1),
                    extrinsics=self.current_pose,
                    intrinsics=self.intrinsics,
                    image_width=W,
                    image_height=H,
                    tile_size=self.config['rendering']['tile_size'],
                    top_k=self.config['rendering'].get('attribution_top_k', 8),
                )
                rendered_color_post = post_render['color']
        else:
            rendered_color_post = rendered_color

        # === 8. Pruning ===
        prune_low_value(
            self.gaussian_model, importance[:self.gaussian_model.num_gaussians],
            opacity_threshold=0.005,
            zero_contrib_frames=self.importance_estimator._zero_contrib_frames,
            prune_patience=self.importance_estimator.prune_patience,
        )
        
        # Compact every 100 frames — preserve optimizer momentum and state store identity
        if self.frame_count % 100 == 0:
            keep_mask = self.gaussian_model.compact()
            self.importance_estimator.prune_buffers(keep_mask)
            if self.optimizer is not None:
                # Prune optimizer state to match compacted parameters
                self.optimizer.prune_state(keep_mask)
                # Re-register new parameter objects (compact() creates new nn.Parameter)
                self._reregister_optimizer_params()
            else:
                self._setup_optimizer()
        
        # === Collect Metrics & Feedback ===
        frame_time = time.time() - frame_start
        
        # Closed-loop profiling feedback to scheduler
        self.scheduler.adjust_budget_from_profiling(
            actual_frame_ms=frame_time * 1000.0,
            actual_opt_ms=opt_time * 1000.0,
            n_optimized=n_optimized,
        )
        
        # Compute quality metrics (pre and post optimization)
        with torch.no_grad():
            # Pre-optimization PSNR (diagnostic)
            psnr_pre = -10 * torch.log10(
                ((rendered_color - rgb) ** 2).mean() + 1e-8
            ).item()
            # Post-optimization PSNR (primary metric)
            psnr_post = -10 * torch.log10(
                ((rendered_color_post - rgb) ** 2).mean() + 1e-8
            ).item()
            psnr = psnr_post  # Primary metric is post-optimization
            depth_l1 = depth_err[depth_valid].mean().item() if depth_valid.any() else 0.0
            
            # SSIM computation (structural similarity)
            def _compute_ssim_simple(img1, img2):
                """Compute mean SSIM between two (H,W,3) images."""
                c1, c2 = 0.01 ** 2, 0.03 ** 2
                mu1 = img1.mean(dim=(0, 1))
                mu2 = img2.mean(dim=(0, 1))
                sig1_sq = ((img1 - mu1) ** 2).mean(dim=(0, 1))
                sig2_sq = ((img2 - mu2) ** 2).mean(dim=(0, 1))
                sig12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=(0, 1))
                ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sig12 + c2)) / (
                    (mu1 ** 2 + mu2 ** 2 + c1) * (sig1_sq + sig2_sq + c2)
                )
                return ssim_map.mean().item()
            
            ssim = _compute_ssim_simple(rendered_color_post, rgb)
        
        budget_ms = self.config['scheduler'].get('gpu_budget_ms', 16.6)
        budget_violated = (frame_time * 1000.0) > budget_ms if budget_ms > 0 else False
        overshoot_ms = max(0.0, (frame_time * 1000.0) - budget_ms) if budget_ms > 0 else 0.0

        metrics = {
            'frame': self.frame_count,
            'psnr': psnr,
            'psnr_pre': psnr_pre,
            'psnr_post': psnr_post,
            'ssim': ssim,
            'depth_l1': depth_l1,
            'color_loss': per_gaussian_color_err.mean().item(),
            'n_gaussians': self.gaussian_model.num_gaussians,
            'n_optimized': n_optimized,
            'n_tier_a': (tiers == Tier.A).sum().item(),
            'n_tier_b': (tiers == Tier.B).sum().item(),
            'n_tier_c': (tiers == Tier.C).sum().item(),
            'n_tier_d': (tiers == Tier.D).sum().item(),
            'frame_time_ms': frame_time * 1000.0,
            'opt_time_ms': opt_time * 1000.0,
            'cache_time_ms': cache_time * 1000.0,
            'fps': 1.0 / max(frame_time, 1e-8),
            'loss': opt_loss_val,
            # Budget metrics
            'budget_ms': budget_ms,
            'budget_violated': budget_violated,
            'overshoot_ms': overshoot_ms,
            # Attribution metrics
            'n_visible': visibility_mask.sum().item(),
            'importance_std': importance.std().item() if importance.numel() > 0 else 0.0,
            'importance_min': importance.min().item() if importance.numel() > 0 else 0.0,
            'importance_max': importance.max().item() if importance.numel() > 0 else 0.0,
            'avg_screen_area': per_gaussian_screen_area.mean().item(),
        }
        self.metrics_history.append(metrics)
        self.frame_count += 1
        
        return metrics
    
    def get_importance_diagnostics(self) -> Dict[str, torch.Tensor]:
        """Expose current research state and per-Gaussian diagnostics.
        
        Returns:
            Dict containing:
                'importance': Tensor[N]
                'color_error': Tensor[N]
                'depth_error': Tensor[N]
                'visibility': Tensor[N]
                'screen_area': Tensor[N]
                'temporal_change': Tensor[N]
                'tiers': Tensor[N]
                'confidence': Tensor[N]
                'components': Dict[str, Tensor[N]]
        """
        if not self.initialized or self.importance_estimator._running_depth_error is None:
            raise RuntimeError("Pipeline has not processed any frame or is uninitialized.")
        
        N = self.gaussian_model.num_gaussians
        importance = self.importance_estimator.compute_importance()[:N]
        tiers = self.importance_estimator.classify_tier(importance)[:N]
        
        color_err = (
            self.importance_estimator._running_color_error[:N] 
            if self.importance_estimator._running_color_error is not None 
            else torch.zeros(N, device=self.device)
        )
        depth_err = (
            self.importance_estimator._running_depth_error[:N] 
            if self.importance_estimator._running_depth_error is not None 
            else torch.zeros(N, device=self.device)
        )
        visibility = (
            self.importance_estimator._visibility_count[:N] 
            if self.importance_estimator._visibility_count is not None 
            else torch.zeros(N, device=self.device)
        )
        
        screen_area = getattr(self.importance_estimator, '_screen_areas', None)
        if screen_area is not None:
            screen_area = screen_area[:N]
        else:
            screen_area = torch.zeros(N, device=self.device)
            
        temporal_change = torch.zeros(N, device=self.device)
        if self.importance_estimator._prev_positions is not None and self.importance_estimator._positions is not None:
            min_len = min(
                self.importance_estimator._prev_positions.shape[0], 
                self.importance_estimator._positions.shape[0], 
                N
            )
            temporal_change[:min_len] = (
                self.importance_estimator._positions[:min_len] - self.importance_estimator._prev_positions[:min_len]
            ).norm(dim=-1)
        if self.importance_estimator._running_error_fast is not None and self.importance_estimator._running_error_slow is not None:
            res_drift = (self.importance_estimator._running_error_fast[:N] - self.importance_estimator._running_error_slow[:N]).abs()
            temporal_change = temporal_change + res_drift

            
        if hasattr(self.gaussian_model, '_confidence') and self.gaussian_model._confidence is not None:
            confidence = self.gaussian_model._confidence[:N].squeeze(-1)
        else:
            confidence = torch.full((N,), 0.5, device=self.device)
            
        components = {
            'color': color_err,
            'depth': depth_err,
            'visibility': visibility,
            'temporal': temporal_change,
            'screen_area': screen_area,
        }
            
        return {
            'importance': importance,
            'color_error': color_err,
            'depth_error': depth_err,
            'visibility': visibility,
            'screen_area': screen_area,
            'temporal_change': temporal_change,
            'tiers': tiers,
            'confidence': confidence,
            'components': components,
        }
    
    def get_gaussian_map(self) -> GaussianModel:
        """Returns the current Gaussian model."""
        return self.gaussian_model
    
    def get_metrics_summary(self) -> Dict:
        """Compute summary statistics over all processed frames."""
        if not self.metrics_history:
            return {}
        
        import numpy as np
        psnrs = [m['psnr'] for m in self.metrics_history]
        depths = [m['depth_l1'] for m in self.metrics_history]
        fps_list = [m['fps'] for m in self.metrics_history]
        violations = [m.get('budget_violated', False) for m in self.metrics_history]
        
        latency_stats = self.scheduler.get_latency_statistics()
        
        return {
            'total_frames': len(self.metrics_history),
            'avg_psnr': float(np.mean(psnrs)),
            'avg_depth_l1': float(np.mean(depths)),
            'avg_fps': float(np.mean(fps_list)),
            'final_n_gaussians': self.metrics_history[-1]['n_gaussians'],
            'avg_frame_time_ms': float(np.mean([m['frame_time_ms'] for m in self.metrics_history])),
            'budget_violation_rate': float(np.mean(violations)),
            'latency_stats': latency_stats,
        }

    def evaluate_gaussian_update(
        self,
        gaussian_indices: torch.Tensor,
        frame: Dict[str, torch.Tensor],
        n_steps: int = 5,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """Execute isolated selective optimization trial sharing the exact same pipeline execution model (R34/R35).
        
        Saves and restores model & optimizer states to ensure non-destructive evaluation.
        
        Args:
            gaussian_indices: (K,) indices of Gaussians to evaluate
            frame: dict with 'rgb', 'depth', 'pose'
            n_steps: optimization steps
            lr: learning rate
            
        Returns:
            Dict containing delta_psnr, delta_depth_l1, delta_quality, measured_trial_cost_ms.
        """
        device = self.device
        rgb = frame['rgb'].to(device)
        depth = frame['depth'].to(device)
        pose = frame.get('pose', self.current_pose).to(device)
        H, W = rgb.shape[:2]
        
        # Backup state
        saved_state = {k: v.clone() for k, v in self.gaussian_model.state_dict().items()}
        
        # 1. Pre-optimization measurements
        with torch.no_grad():
            render_before = rasterize_scene(
                means3D=self.gaussian_model.positions,
                cov3D=self.gaussian_model.build_covariance(),
                colors=self.gaussian_model.get_colors(),
                opacities=self.gaussian_model.opacities.squeeze(-1),
                extrinsics=pose,
                intrinsics=self.intrinsics,
                image_width=W,
                image_height=H,
            )
            psnr_before = -10 * torch.log10(((render_before['color'] - rgb)**2).mean() + 1e-8).item()
            valid_d = depth > 0
            depth_l1_before = (render_before['depth'][valid_d] - depth[valid_d]).abs().mean().item() if valid_d.any() else 0.0
            
        # 2. Setup selective optimization trial
        opt_mask = torch.zeros(self.gaussian_model.num_gaussians, dtype=torch.bool, device=device)
        opt_mask[gaussian_indices] = True
        
        trial_opt = SelectiveAdam([{'params': list(self.gaussian_model.parameters()), 'lr': lr}])
        trial_cache = FrozenBackgroundCache(device=device)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        for _ in range(n_steps):
            trial_opt.zero_grad()
            active_subset = self.gaussian_model.get_optimization_subset(opt_mask)
            frozen_mask = ~opt_mask
            
            if frozen_mask.any():
                trial_cache.build_cache(self.gaussian_model, frozen_mask, pose, self.intrinsics, W, H)
                
            comp_out = trial_cache.composite_with_active(active_subset, pose, self.intrinsics, W, H)
            losses = total_loss(comp_out['color'], rgb, comp_out['depth'], depth, {'color': 1.0, 'depth': 0.5})
            losses['total'].backward()
            trial_opt.step(active_idx=active_subset['indices'])
            
        if device == 'cuda':
            torch.cuda.synchronize()
        measured_trial_cost_ms = (time.perf_counter() - t0) * 1000.0
        
        # 3. Post-optimization measurements
        with torch.no_grad():
            render_after = rasterize_scene(
                means3D=self.gaussian_model.positions,
                cov3D=self.gaussian_model.build_covariance(),
                colors=self.gaussian_model.get_colors(),
                opacities=self.gaussian_model.opacities.squeeze(-1),
                extrinsics=pose,
                intrinsics=self.intrinsics,
                image_width=W,
                image_height=H,
            )
            psnr_after = -10 * torch.log10(((render_after['color'] - rgb)**2).mean() + 1e-8).item()
            depth_l1_after = (render_after['depth'][valid_d] - depth[valid_d]).abs().mean().item() if valid_d.any() else 0.0
            
        delta_psnr = psnr_after - psnr_before
        delta_depth_gain = depth_l1_before - depth_l1_after  # Unclamped: positive is gain, negative is degradation
        delta_quality = 0.70 * delta_psnr + 0.30 * (10.0 * delta_depth_gain)
        
        # Restore state
        self.gaussian_model.load_state_dict(saved_state)
        
        return {
            'delta_psnr': float(delta_psnr),
            'delta_depth_gain': float(delta_depth_gain),
            'delta_quality': float(delta_quality),
            'measured_trial_cost_ms': float(measured_trial_cost_ms),
            'oracle_utility': float(delta_quality / (measured_trial_cost_ms + 1e-6)),
        }



