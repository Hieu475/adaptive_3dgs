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
from .scheduler import BudgetScheduler
from .densification import (
    compute_error_masks, sample_candidates,
    create_gaussians_from_candidates, prune_low_value
)
from .tracker import ICPTracker


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
        
        # State
        self.frame_count = 0
        self.current_pose = torch.eye(4, device=device)
        self.intrinsics: Optional[torch.Tensor] = None
        self.initialized = False
        
        # Metrics collection
        self.metrics_history = []
    
    @staticmethod
    def _default_config() -> Dict:
        return {
            'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 500000},
            'rendering': {'tile_size': 16, 'image_width': 640, 'image_height': 480},
            'losses': {'weight_color': 1.0, 'weight_depth': 0.5, 'weight_normal': 0.1, 'weight_regularization': 0.01},
            'importance': {'depth_error': 1.0, 'color_error': 1.0, 'normal_error': 0.5, 'visibility': 0.1, 'temporal': 0.5, 'screen_space': 0.2},
            'scheduler': {'gpu_budget_ms': 16.6, 'tier_thresholds': [0.8, 0.5, 0.2], 'optimize_every_n_frames': 5},
            'densification': {'max_new_per_frame': 500, 'error_threshold_color': 0.1, 'error_threshold_depth': 0.05, 'transmission_threshold': 0.5},
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
        self.optimizer = optim.Adam(params)
    
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
            self.current_pose = self.tracker.track_frame(rgb, depth, self.gaussian_model)
        
        # === 2. Render Current Map ===
        with torch.no_grad():
            cov3D = self.gaussian_model.build_covariance()
            render_result = rasterize_scene(
                means3D=self.gaussian_model.positions,
                cov3D=cov3D,
                colors=self.gaussian_model.get_colors(),
                opacities=self.gaussian_model.opacities.squeeze(-1),
                extrinsics=self.current_pose,
                intrinsics=self.intrinsics,
                image_width=W,
                image_height=H,
                tile_size=self.config['rendering']['tile_size'],
            )
        
        rendered_color = render_result['color']  # (H, W, 3)
        rendered_depth = render_result['depth']  # (H, W)
        transmission = render_result['transmission']  # (H, W)
        
        # === 3. Compute Errors ===
        color_err = (rendered_color - rgb).abs().mean(dim=-1)  # (H, W)
        depth_valid = depth > 0
        depth_err = torch.zeros_like(depth)
        depth_err[depth_valid] = (rendered_depth[depth_valid] - depth[depth_valid]).abs()
        
        # Per-Gaussian errors (approximate via pixel-to-Gaussian mapping)
        N = self.gaussian_model.num_gaussians
        per_gaussian_color_err = torch.zeros(N, device=self.device)
        per_gaussian_depth_err = torch.zeros(N, device=self.device)
        visibility_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        
        # Simple approximation: average error in Gaussian's projected region
        # (Full implementation would use the Gaussian index map)
        mean_color_err = color_err.mean().item()
        mean_depth_err = depth_err[depth_valid].mean().item() if depth_valid.any() else 0.0
        per_gaussian_color_err.fill_(mean_color_err)
        per_gaussian_depth_err.fill_(mean_depth_err)
        visibility_mask.fill_(True)  # Simplified: assume all visible
        
        # === 4. Importance Estimation ===
        self.importance_estimator.update_statistics(
            depth_errors=per_gaussian_depth_err,
            color_errors=per_gaussian_color_err,
            normal_errors=None,
            visibility_mask=visibility_mask,
            positions=self.gaussian_model.positions.detach(),
        )
        importance = self.importance_estimator.compute_importance()
        tiers = self.importance_estimator.classify_tier(importance)
        
        # === 5. Densification ===
        dense_cfg = self.config['densification']
        error_masks = compute_error_masks(
            color_err, depth_err, transmission,
            color_threshold=dense_cfg['error_threshold_color'],
            depth_threshold=dense_cfg['error_threshold_depth'],
            transmission_threshold=dense_cfg['transmission_threshold'],
        )
        
        max_new = min(
            dense_cfg['max_new_per_frame'],
            self.scheduler.compute_max_new_gaussians(),
            self.config['gaussian']['max_gaussians'] - self.gaussian_model.num_gaussians,
        )
        
        if max_new > 0 and error_masks['combined_mask'].any():
            candidates = sample_candidates(error_masks['combined_mask'], max_new)
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
                    self._setup_optimizer()  # Re-create optimizer with new params
        
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
        
        optimize_mask = self.scheduler.select_for_optimization(
            importance, tiers, frame_idx=self.frame_count
        )
        
        # === 7. Selective Optimization ===
        n_optimized = 0
        opt_loss_val = 0.0
        
        if optimize_mask.any() and self.optimizer is not None:
            self.optimizer.zero_grad()
            
            # Re-render for gradient computation
            cov3D = self.gaussian_model.build_covariance()
            render_opt = rasterize_scene(
                means3D=self.gaussian_model.positions,
                cov3D=cov3D,
                colors=self.gaussian_model.get_colors(),
                opacities=self.gaussian_model.opacities.squeeze(-1),
                extrinsics=self.current_pose,
                intrinsics=self.intrinsics,
                image_width=W,
                image_height=H,
                tile_size=self.config['rendering']['tile_size'],
            )
            
            # Compute loss
            weights = {
                'color': self.config['losses']['weight_color'],
                'depth': self.config['losses']['weight_depth'],
            }
            losses = total_loss(
                render_opt['color'], rgb,
                render_opt['depth'], depth,
                weights,
            )
            
            losses['total'].backward()
            
            # Zero gradients for non-selected Gaussians
            with torch.no_grad():
                non_optimize = ~optimize_mask[:self.gaussian_model._xyz.shape[0]]
                if self.gaussian_model._xyz.grad is not None:
                    self.gaussian_model._xyz.grad[non_optimize] = 0
                if self.gaussian_model._scaling.grad is not None:
                    self.gaussian_model._scaling.grad[non_optimize] = 0
                if self.gaussian_model._rotation.grad is not None:
                    self.gaussian_model._rotation.grad[non_optimize] = 0
                if self.gaussian_model._opacity.grad is not None:
                    self.gaussian_model._opacity.grad[non_optimize] = 0
            
            self.optimizer.step()
            n_optimized = optimize_mask.sum().item()
            opt_loss_val = losses['total'].item()
        
        # === 8. Pruning ===
        prune_low_value(
            self.gaussian_model, importance[:self.gaussian_model.num_gaussians],
            opacity_threshold=0.005,
            importance_threshold=0.01,
        )
        
        # Compact every 100 frames
        if self.frame_count % 100 == 0:
            self.gaussian_model.compact()
            self._setup_optimizer()
        
        # === Collect Metrics ===
        frame_time = time.time() - frame_start
        
        # Compute quality metrics
        with torch.no_grad():
            psnr = -10 * torch.log10(
                ((rendered_color - rgb) ** 2).mean() + 1e-8
            ).item()
            depth_l1 = depth_err[depth_valid].mean().item() if depth_valid.any() else 0.0
        
        metrics = {
            'frame': self.frame_count,
            'psnr': psnr,
            'depth_l1': depth_l1,
            'color_loss': mean_color_err,
            'n_gaussians': self.gaussian_model.num_gaussians,
            'n_optimized': n_optimized,
            'n_tier_a': (tiers == Tier.A).sum().item(),
            'n_tier_b': (tiers == Tier.B).sum().item(),
            'n_tier_c': (tiers == Tier.C).sum().item(),
            'n_tier_d': (tiers == Tier.D).sum().item(),
            'frame_time_ms': frame_time * 1000,
            'fps': 1.0 / max(frame_time, 1e-8),
            'loss': opt_loss_val,
        }
        self.metrics_history.append(metrics)
        self.frame_count += 1
        
        return metrics
    
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
        
        return {
            'total_frames': len(self.metrics_history),
            'avg_psnr': np.mean(psnrs),
            'avg_depth_l1': np.mean(depths),
            'avg_fps': np.mean(fps_list),
            'final_n_gaussians': self.metrics_history[-1]['n_gaussians'],
            'avg_frame_time_ms': np.mean([m['frame_time_ms'] for m in self.metrics_history]),
        }
