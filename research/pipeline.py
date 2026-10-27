"""Online Reconstruction Pipeline for Adaptive 3D Gaussian Splatting.

Ties together all research modules into a complete per-frame pipeline:
    initialize → [for each frame: track → render → errors → densify → schedule → optimize → prune]

CANONICAL CONFIG SOURCE: _default_config() below. configs/default.yaml mirrors it.
OURS (Phase 13): coverage throttling (κ≥0.90, no warmup, K=5).
Backlog throttling is an independent opt-in for the corrected factorial study.
Learned utility path is retained only as documented negative result.
Split plan (do NOT split before paper freeze): map.py (init/densify/prune) |
sched.py (selection) | optim.py (SelectiveAdam steps). See experiments/README.md.
"""
import torch
import torch.optim as optim
import time
import hashlib
from typing import Dict, Optional, Any
import numpy as np
import yaml


from .gaussian_repr import GaussianModel, GaussianState
from .projection import world_to_camera, project_to_screen, compute_2d_covariance
from .rasterizer import render as rasterize_scene, get_renderer_backend
from .losses import total_loss, color_loss, depth_loss
from .depth_render import render_depth_surface_aware
from .importance import GaussianImportanceEstimator, Tier
from .attribution import (
    render_with_attribution,
    compute_gaussian_statistics,
    compute_fast_gaussian_statistics,
)
from .scheduler import (
    BudgetScheduler,
    OptimizationPolicy,
    estimate_gaussian_costs,
    estimate_gaussian_cost_components,
)
from .densification import (
    compute_error_masks, sample_candidates,
    create_gaussians_from_candidates, prune_low_value,
    compute_depth_adaptive_scale,
)
from .tracker import ICPTracker
from .background_cache import FrozenBackgroundCache
from .selective_optimizer import SelectiveAdam
from .utility_features import extract_online_feature_tensor
from .utility_predictor import FrozenUtilityPredictor
from .mechanism.metrics import depth_metric_bundle, rgb_metric_bundle
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
        self.config.setdefault('rendering', {})['backend'] = get_renderer_backend(
            explicit_backend=self.config.get('rendering', {}).get('backend'),
            device_type=torch.device(device).type,
        )
        
        # Memory Guard: Cap PyTorch allocation to protect OS and desktop compositor
        if str(self.device).startswith('cuda') and torch.cuda.is_available():
            max_vram_fraction = self.config.get('system', {}).get('max_vram_fraction', 0.70)
            if max_vram_fraction is not None and 0.0 < max_vram_fraction <= 1.0:
                try:
                    torch.cuda.set_per_process_memory_fraction(max_vram_fraction, 0)
                except (RuntimeError, ValueError):
                    pass
        
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
        if 'densify_budget_override_ms' in self.config['scheduler']:
            self.scheduler.densify_budget_override_ms = float(
                self.config['scheduler']['densify_budget_override_ms'])
        
        # Optimizer (initialized after first frame)
        self.optimizer: Optional[optim.Adam] = None
        
        # Frozen Background Cache for True Selective Optimization
        self.bg_cache = FrozenBackgroundCache(device=device)

        # Learned Utility Predictor (Phase 4 / Phase 13)
        self.utility_predictor: Optional[FrozenUtilityPredictor] = None
        self._learned_utility_scores: Optional[torch.Tensor] = None
        self._predicted_delta_q: Optional[torch.Tensor] = None
        self._predicted_delta_t: Optional[torch.Tensor] = None
        policy_cfg = self.config.get('scheduler', {}).get('policy', 'budget_aware')
        use_learned = self.config.get('scheduler', {}).get('use_learned_utility', False)
        if policy_cfg in ("learned_utility", OptimizationPolicy.LEARNED_UTILITY.value):
            use_learned = True

        if use_learned:
            seed = self.config.get('seed', 42)
            try:
                self.utility_predictor = FrozenUtilityPredictor(seed=seed, device=device)
                import logging
                logging.getLogger(__name__).info(
                    f"[Pipeline] Successfully loaded FrozenUtilityPredictor for seed {seed} (in_features={self.utility_predictor.in_features})."
                )
            except Exception as e:
                if policy_cfg in ("learned_utility", OptimizationPolicy.LEARNED_UTILITY.value):
                    raise RuntimeError(
                        f"Policy '{policy_cfg}' strictly requires FrozenUtilityPredictor for seed {seed}, "
                        f"but loading failed: {e}. Aborting to avoid invalid fallback."
                    )
                import logging
                logging.getLogger(__name__).warning(
                    f"[Pipeline] Could not load FrozenUtilityPredictor for seed {seed}: {e}. "
                    f"Policy '{policy_cfg}' will operate in HEURISTIC mode, NOT learned utility."
                )
                self.utility_predictor = None
        
        # Optional LPIPS Perceptual Metric
        self._lpips_metric = None
        if self.config.get('rendering', {}).get('compute_lpips', False):
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
                self._lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
                self._lpips_metric.eval()
                for p in self._lpips_metric.parameters():
                    p.requires_grad = False
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Could not load LPIPS metric: {e}")
                self._lpips_metric = None

        # State

        self.frame_count = 0
        self._spawn_control_count = 0
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
        # Expensive, opt-in per-stage hashes used only to localize run-to-run
        # divergence. Disabled in all benchmark configs by default.
        self._repro_trace_enabled = bool(self.config.get('system', {}).get('repro_trace', False))
        self._repro_trace = []
        self._init_repro_trace = []
        self._init_gradient_snapshots = {}

    @staticmethod
    def _hash_trace_tensors(tensors: Dict[str, Optional[torch.Tensor]]) -> Dict[str, str]:
        """Return exact SHA256 hashes for a small set of diagnostic tensors."""
        result = {}
        for name, tensor in tensors.items():
            if tensor is None:
                result[name] = 'none'
                continue
            value = tensor.detach().contiguous().cpu()
            digest = hashlib.sha256()
            digest.update(str(value.dtype).encode('ascii'))
            digest.update(repr(tuple(value.shape)).encode('ascii'))
            digest.update(value.numpy().tobytes())
            result[name] = digest.hexdigest()
        return result

    def _trace_stage(self, stage: str, tensors: Dict[str, Optional[torch.Tensor]]):
        if self._repro_trace_enabled:
            self._repro_trace.append({
                'stage': stage,
                'hashes': self._hash_trace_tensors(tensors),
            })

    def _trace_init_stage(self, stage: str, tensors: Dict[str, Optional[torch.Tensor]]):
        if self._repro_trace_enabled:
            if (stage == 'init_step_0_gradients'
                    and self.config.get('system', {}).get('repro_trace_dump_init_gradients', False)):
                self._init_gradient_snapshots = {
                    name: tensor.detach().float().cpu().clone()
                    for name, tensor in tensors.items() if tensor is not None
                }
            trace_row = {
                'stage': stage,
                'hashes': self._hash_trace_tensors(tensors),
            }
            if (self.config.get('system', {}).get('repro_trace_summaries', False)
                    and 'gradients' in stage):
                summaries = {}
                for name, tensor in tensors.items():
                    if tensor is None:
                        summaries[name] = None
                        continue
                    value = tensor.detach().float()
                    finite = torch.isfinite(value)
                    finite_values = value[finite]
                    summaries[name] = {
                        'shape': list(value.shape),
                        'finite_count': int(finite.sum().item()),
                        'nonfinite_count': int((~finite).sum().item()),
                        'mean_abs': float(finite_values.abs().mean().item()) if finite_values.numel() else None,
                        'rms': float(torch.sqrt((finite_values.square().mean())).item()) if finite_values.numel() else None,
                        'max_abs': float(finite_values.abs().max().item()) if finite_values.numel() else None,
                    }
                trace_row['summaries'] = summaries
            self._init_repro_trace.append(trace_row)
    
    @staticmethod
    def _default_config() -> Dict:
        return {
            'gaussian': {
                'sh_degree': 0,
                'initial_opacity': 0.5,
                'max_gaussians': 500000,
                'init_refine_steps': 0,
                'init_warmup_mature': False,
            },
            'system': {
                'max_vram_fraction': 0.70,
                'empty_cache_frequency': 0,
            },
            'rendering': {
                'tile_size': 16,
                'image_width': 640,
                'image_height': 480,
                'use_surface_aware_depth': False,
                'depth_threshold_opaque': 0.5,
                'attribution_top_k': 8,
                'use_fast_attribution': True,
            },
            'losses': {'weight_color': 0.8, 'weight_depth': 0.5, 'weight_ssim': 0.2, 'weight_normal': 0.1, 'weight_regularization': 0.01},
            'importance': {'depth_error': 1.0, 'color_error': 1.0, 'normal_error': 0.5, 'visibility': 0.1, 'temporal': 0.5, 'screen_space': 0.2},
            'scheduler': {
                'gpu_budget_ms': 16.6,
                'tier_thresholds': [0.8, 0.5, 0.2],
                'optimize_every_n_frames': 5,
                'policy': 'budget_aware',
                'optimize_ratio': 0.5,
                'cost_per_gaussian_us': 0.5,
                'enable_warmup': False,  # Ablation: warmup is harmful (-0.75 dB)
                'warmup_steps': 3,
                'warmup_budget_ratio': 0.20,
                'warmup_k': 2,
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
                'enable_coverage_throttling': True,
                'throttle_coverage_threshold': 0.90,
                'throttle_factor': 0.20,
                'throttle_error_threshold_mult': 1.5,
            },
            'training': {
                'n_micro_steps': 5,
                'learning_rate': {'position': 4.0e-4, 'scale': 5e-3, 'rotation': 1e-3, 'opacity': 5e-2, 'sh': 1.0e-2}
            },
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
        self._init_repro_trace = []
        self._trace_init_stage('frame0_inputs', {
            'rgb': rgb, 'depth': depth,
            'intrinsics': intrinsics, 'pose': self.current_pose,
        })
        # Anchor the visual tracker on the init frame (GT pose if given).
        try:
            if hasattr(self.tracker, "reset"):
                self.tracker.reset(rgb.to(self.device), depth.to(self.device),
                                   self.current_pose.clone())
            self.tracker._K = self.intrinsics.detach().float()
        except Exception:
            pass

        H, W = depth.shape
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        
        # Subsample pixels to create initial Gaussians.
        # NOTE: at 320x240 (post 2x downsample), stride=4 previously produced only
        # ~4,800 initial Gaussians for an entire room — 20-200x sparser than
        # published online 3DGS-SLAM systems (SplaTAM ~635k-969k, RTG-SLAM
        # ~84k-273k on comparable TUM scenes), which was the dominant cause of
        # both low absolute PSNR and near-zero headroom between No-Op and Full
        # Optimization bounds. Default lowered to stride=2 (~4x more initial
        # points); combined with adaptive per-frame densification (see
        # process_frame below) this is expected to reach tens of thousands of
        # Gaussians within the trajectory.
        stride = self.config.get('gaussian', {}).get('init_stride', 2)
        v_coords = torch.arange(0, H, stride, device=self.device)
        u_coords = torch.arange(0, W, stride, device=self.device)
        vv, uu = torch.meshgrid(v_coords, u_coords, indexing='ij')
        uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)  # (K, 2)
        
        # Filter by valid depth
        d_vals = depth[uv[:, 1].long(), uv[:, 0].long()]
        valid = d_vals > 0
        uv = uv[valid]
        d_vals = d_vals[valid]  # keep in sync with filtered uv (needed for depth-adaptive scale)
        
        # Unproject to 3D
        from .densification import unproject_pixels
        points = unproject_pixels(uv, depth, intrinsics, self.current_pose)
        self._trace_init_stage('initial_point_selection', {
            'uv': uv, 'depth_values': d_vals, 'points': points,
        })
        
        # Get colors
        colors = rgb[uv[:, 1].long(), uv[:, 0].long()]
        
        # Geometry-aware initial scale: r = depth / focal_length (SplaTAM-style),
        # instead of one global constant applied regardless of depth. Falls back
        # to the legacy constant if explicitly disabled via config.
        if self.config['gaussian'].get('scale_mode', 'depth_adaptive') == 'depth_adaptive':
            init_scale = compute_depth_adaptive_scale(
                d_vals, intrinsics,
                pixel_multiplier=self.config['gaussian'].get('scale_pixel_multiplier', 1.0),
            )
        else:
            init_scale = self.config['gaussian'].get('initial_scale', 0.01)
        
        # Initialize Gaussians
        self.gaussian_model.initialize_from_points(
            points, colors=colors,
            initial_scale=init_scale,
            initial_opacity=self.config['gaussian']['initial_opacity'],
        )
        self._trace_init_stage('initial_gaussian_map', {
            'xyz': self.gaussian_model._xyz,
            'scaling': self.gaussian_model._scaling,
            'rotation': self.gaussian_model._rotation,
            'opacity': self.gaussian_model._opacity,
            'features_dc': self.gaussian_model._features_dc,
            'features_rest': self.gaussian_model._features_rest,
        })
        
        self._setup_optimizer()
        self.initialized = True
        self.frame_count = 1

        # Frame-0 Map Refinement Burst (fits initial unprojected primitives to frame 0)
        init_refine_steps = self.config.get('gaussian', {}).get('init_refine_steps', 0)
        if init_refine_steps > 0 and self.optimizer is not None:
            weights = {
                'color': self.config.get('losses', {}).get('weight_color', 0.8),
                'depth': self.config.get('losses', {}).get('weight_depth', 0.5),
                'ssim': self.config.get('losses', {}).get('weight_ssim', 0.2),
                'ssim_implementation': self.config.get('losses', {}).get('ssim_implementation', 'direct'),
            }
            tile_size = self.config.get('rendering', {}).get('tile_size', 16)
            for init_step in range(init_refine_steps):
                self.optimizer.zero_grad()
                cov3D = self.gaussian_model.build_covariance()
                rendered = rasterize_scene(
                    means3D=self.gaussian_model.positions,
                    cov3D=cov3D,
                    colors=self.gaussian_model.get_colors(),
                    opacities=self.gaussian_model.opacities.squeeze(-1),
                    extrinsics=self.current_pose,
                    intrinsics=self.intrinsics,
                    image_width=W,
                    image_height=H,
                    tile_size=tile_size,
                    backend=self.config.get('rendering', {}).get('backend'),
                )
                if self._repro_trace_enabled:
                    rendered['color'].register_hook(
                        lambda grad, step=init_step: self._trace_init_stage(
                            f'init_step_{step}_render_color_gradient', {'grad': grad}))
                    rendered['depth'].register_hook(
                        lambda grad, step=init_step: self._trace_init_stage(
                            f'init_step_{step}_render_depth_gradient', {'grad': grad}))
                self._trace_init_stage(f'init_step_{init_step}_render', {
                    'color': rendered['color'], 'depth': rendered['depth'],
                })
                initial_alpha_mask = rendered.get('alpha', torch.ones_like(depth)) >= self.config.get(
                    'rendering', {}).get('depth_threshold_opaque', 0.5)
                d_mask = initial_alpha_mask & torch.isfinite(depth) & (depth > 0)
                losses = total_loss(
                    rendered['color'], rgb,
                    rendered['depth'], depth,
                    weights,
                    depth_valid_mask=d_mask,
                )
                self._trace_init_stage(f'init_step_{init_step}_loss', {
                    'loss': losses['total'].detach().reshape(1),
                })
                if losses['total'].requires_grad:
                    losses['total'].backward()
                    self._trace_init_stage(f'init_step_{init_step}_gradients', {
                        'xyz_grad': self.gaussian_model._xyz.grad,
                        'scale_grad': self.gaussian_model._scaling.grad,
                        'opacity_grad': self.gaussian_model._opacity.grad,
                        'features_dc_grad': self.gaussian_model._features_dc.grad,
                    })
                    self.optimizer.step()
                    self._trace_init_stage(f'init_step_{init_step}_updated_map', {
                        'xyz': self.gaussian_model._xyz,
                        'scaling': self.gaussian_model._scaling,
                        'rotation': self.gaussian_model._rotation,
                        'opacity': self.gaussian_model._opacity,
                        'features_dc': self.gaussian_model._features_dc,
                    })

        # Set update_counts of initial primitives to warmup_steps so they enter stream mature
        if hasattr(self.gaussian_model, 'state_store') and self.gaussian_model.state_store is not None:
            if init_refine_steps > 0 or self.config.get('gaussian', {}).get('init_warmup_mature', False):
                warmup_steps = self.config.get('scheduler', {}).get('warmup_steps', 3)
                self.gaussian_model.state_store.update_counts.fill_(warmup_steps)
        
        print(f"[Init] Created {self.gaussian_model.num_gaussians} Gaussians from first frame (refine_steps={init_refine_steps})")
    
    def process_frame(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        gt_pose: Optional[torch.Tensor] = None,
        frame_noise: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process a single RGB-D frame through the full pipeline.

        Pipeline: track → route → render → compute errors → densify → schedule → optimize → prune

        Args:
            rgb: (H, W, 3) RGB image in [0, 1]
            depth: (H, W) depth map
            gt_pose: (4, 4) optional ground truth pose (skip tracking)
            frame_noise: optional v3 sensor-noise score for this frame (see
                research/noise_adaptive.py). When ``adaptive_routing.enabled``
                is set, routes this frame's selection policy + kappa from the
                score; otherwise the configured policy/kappa apply. The routed
                values are recorded in metrics (routed_policy, frame_noise).

        Returns:
            Dict with per-frame metrics
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        frame_start = time.time()
        if self._repro_trace_enabled:
            self._repro_trace = []
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        H, W = depth.shape

        # === 1. Camera Tracking ===
        if gt_pose is not None:
            self.current_pose = gt_pose.to(self.device)
        else:
            self.current_pose = self.tracker.track_frame(
                rgb, depth, self.gaussian_model,
                intrinsics=self.intrinsics).to(self.device)

        self._trace_stage('inputs_and_pose', {
            'rgb': rgb, 'depth': depth, 'pose': self.current_pose,
        })
        self._trace_stage('map_before_render', {
            'xyz': self.gaussian_model._xyz,
            'scaling': self.gaussian_model._scaling,
            'rotation': self.gaussian_model._rotation,
            'opacity': self.gaussian_model._opacity,
            'features_dc': self.gaussian_model._features_dc,
            'features_rest': self.gaussian_model._features_rest,
            'normals': self.gaussian_model._normals,
            'confidence': self.gaussian_model._confidence,
            'state': self.gaussian_model._state,
        })

        # === 1b. Per-frame adaptive routing (no-op unless enabled) ===
        adapt_cfg = self.config.get('adaptive_routing', {})
        routed_policy: Optional[str] = None
        routed_kappa: Optional[float] = None
        if adapt_cfg.get('enabled', False) and frame_noise is not None:
            from .noise_adaptive import NoiseAdaptiveController, NoiseAdaptiveConfig
            _ctrl = NoiseAdaptiveController(NoiseAdaptiveConfig(
                kappa_noisy=adapt_cfg.get('kappa_noisy', 0.90),
                kappa_clean=adapt_cfg.get('kappa_clean', 0.98),
                blend_lo=adapt_cfg.get('blend_lo', 0.10),
                blend_hi=adapt_cfg.get('blend_hi', 0.30)))
            routed_policy, routed_kappa = _ctrl.route(float(frame_noise))
        
        # === 2. Render Current Map (with per-Gaussian attribution) ===
        use_fast_attribution = self.config['rendering'].get(
            'use_fast_attribution',
            self.config.get('rendering', {}).get('backend', 'gsplat') == 'gsplat'
        )
        with torch.no_grad():
            cov3D = self.gaussian_model.build_covariance()
            if self.gaussian_model.sh_degree > 0:
                from .rasterizer import _view_directions
                current_dirs = _view_directions(self.gaussian_model.positions, self.current_pose)
            else:
                current_dirs = None
            current_colors = self.gaussian_model.get_colors(current_dirs)

            if use_fast_attribution:
                renderer_backend = self.config.get('rendering', {}).get(
                    'backend', 'gsplat' if str(self.device).startswith('cuda') else 'reference'
                )
                render_result = rasterize_scene(
                    means3D=self.gaussian_model.positions,
                    cov3D=cov3D,
                    colors=current_colors,
                    opacities=self.gaussian_model.opacities.squeeze(-1),
                    extrinsics=self.current_pose,
                    intrinsics=self.intrinsics,
                    image_width=W,
                    image_height=H,
                    tile_size=self.config['rendering']['tile_size'],
                    backend=renderer_backend,
                )
            else:
                render_result = render_with_attribution(
                    means3D=self.gaussian_model.positions,
                    cov3D=cov3D,
                    colors=current_colors,
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
            alpha_threshold = float(self.config['rendering'].get('depth_threshold_opaque', 0.5))
            
            if self.config['rendering'].get('use_surface_aware_depth', False):
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
                rendered_depth_support = depth_result['hit_mask']
            else:
                rendered_depth = render_result['depth']  # (H, W)
                rendered_depth_support = render_result.get(
                    'alpha', 1.0 - transmission) >= alpha_threshold
            rendered_depth = torch.where(
                rendered_depth_support, rendered_depth, torch.zeros_like(rendered_depth))

        self._trace_stage('pre_update_render', {
            'color': rendered_color, 'depth': rendered_depth,
            'transmission': transmission,
        })
        
        # === 3. Per-Gaussian Error Attribution ===
        # Compute true per-Gaussian statistics
        N = self.gaussian_model.num_gaussians
        
        if use_fast_attribution:
            gaussian_stats = compute_fast_gaussian_statistics(
                means3D=self.gaussian_model.positions,
                cov3D=cov3D,
                opacities=self.gaussian_model.opacities.squeeze(-1),
                rendered_color=rendered_color,
                rendered_depth=rendered_depth,
                gt_color=rgb,
                gt_depth=depth,
                extrinsics=self.current_pose,
                intrinsics=self.intrinsics,
            )
        else:
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
        depth_valid = torch.isfinite(depth) & (depth > 0)
        depth_err = torch.zeros_like(depth)
        depth_err[depth_valid] = (rendered_depth[depth_valid] - depth[depth_valid]).abs()
        self._trace_stage('attribution_and_residuals', {
            'color_error': per_gaussian_color_err,
            'depth_error': per_gaussian_depth_err,
            'visibility': visibility_mask,
            'screen_area': per_gaussian_screen_area,
            'pixel_color_error': color_err,
            'pixel_depth_error': depth_err,
        })
        
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
        
        # Compute scene coverage from transmission (< 0.5 indicates solid geometry coverage)
        if depth_valid.any():
            current_coverage = float((transmission[depth_valid] < 0.5).float().mean().item())
        else:
            current_coverage = float((transmission < 0.5).float().mean().item())

        enable_throttling = dense_cfg.get('enable_coverage_throttling', True)
        thresh_target = routed_kappa if routed_kappa is not None else dense_cfg.get('throttle_coverage_threshold', 0.90)
        max_multiplier = dense_cfg.get('throttle_error_threshold_mult', 1.5)
        if enable_throttling and current_coverage >= thresh_target:
            thresh_multiplier = max_multiplier
        elif enable_throttling and current_coverage > 0.80:
            scale_range = max(thresh_target - 0.80, 1e-4)
            thresh_multiplier = 1.0 + (max_multiplier - 1.0) * ((current_coverage - 0.80) / scale_range)
        else:
            thresh_multiplier = 1.0

        # Adaptive thresholds or fixed thresholds (scaled by coverage-aware multiplier)
        if dense_cfg.get('use_adaptive_thresholds', True):
            depth_thresh, color_thresh = self.scheduler.adaptive_threshold(
                depth_errors=depth_err[depth_valid] if depth_valid.any() else torch.tensor([0.05], device=self.device),
                color_errors=color_err,
                k=dense_cfg.get('adaptive_k', 2.0) * thresh_multiplier,
            )
        else:
            color_thresh = dense_cfg['error_threshold_color'] * thresh_multiplier
            depth_thresh = dense_cfg['error_threshold_depth'] * thresh_multiplier

        error_masks = compute_error_masks(
            color_err, depth_err, transmission,
            color_threshold=color_thresh,
            depth_threshold=depth_thresh,
            transmission_threshold=dense_cfg['transmission_threshold'],
        )
        
        sched_cfg = self.config.get('scheduler', {})
        enable_warmup = sched_cfg.get('enable_warmup', True) and sched_cfg.get('warmup_budget_ratio', 0.20) > 0
        n_warmup_current = None
        if (enable_warmup or sched_cfg.get('enable_backlog_throttling', False)) and hasattr(self.gaussian_model, 'update_counts'):
            warmup_steps = sched_cfg.get('warmup_steps', 3)
            n_warmup_current = int((self.gaussian_model.update_counts < warmup_steps).sum().item())

        n_err_pixels = int(error_masks['combined_mask'].sum().item())
        error_ratio = n_err_pixels / float(H * W)
        # Only throttle coverage when current view is settled (error_ratio < 0.15)
        effective_coverage = current_coverage if (enable_throttling and error_ratio < 0.15) else None

        max_new = min(
            dense_cfg['max_new_per_frame'],
            self.scheduler.compute_max_new_gaussians(
                n_error_pixels=n_err_pixels,
                current_coverage=effective_coverage,
                n_warmup=n_warmup_current,
                max_warmup_queue=sched_cfg.get('max_warmup_queue', 500),
                coverage_hi=routed_kappa if routed_kappa is not None else dense_cfg.get('throttle_coverage_threshold', 0.90),
                coverage_lo=dense_cfg.get('throttle_coverage_lo', 0.80),
                throttle_factor_hi=dense_cfg.get('throttle_factor', 0.20),
            ),
            self.config['gaussian']['max_gaussians'] - self.gaussian_model.num_gaussians,
        )
        # Optional experimental intervention. A fixed total spawn target is
        # allocated within a configured frame window; the per-frame allowance
        # adapts to any missed quota so the run can catch up before the window
        # closes. Default mapping behavior is unchanged when absent.
        spawn_control = dense_cfg.get('spawn_control')
        spawn_quota_this_frame = None
        if spawn_control:
            target_total = max(0, int(spawn_control['target_total']))
            start_frame = int(spawn_control.get('start_frame', 1))
            end_frame = int(spawn_control.get('end_frame', 150))
            remaining = max(0, target_total - self._spawn_control_count)
            # The control owns the spawn allowance; do not let the ordinary
            # error-pressure scheduler silently reduce the requested quota.
            available_capacity = max(0, self.config['gaussian']['max_gaussians']
                                     - self.gaussian_model.num_gaussians)
            max_new = min(dense_cfg['max_new_per_frame'], available_capacity)
            if start_frame <= self.frame_count <= end_frame and remaining:
                active_frames_left = end_frame - self.frame_count + 1
                spawn_quota_this_frame = (remaining + active_frames_left - 1) // active_frames_left
                max_new = min(max_new, spawn_quota_this_frame)
            else:
                spawn_quota_this_frame = 0
                max_new = 0
        
        n_spawned = 0
        n_spawned_fallback = 0
        trace_candidates = None
        if max_new > 0 and (error_masks['combined_mask'].any() or spawn_control):
            if spawn_control and spawn_control.get('candidate_mode') == 'uniform_depth':
                candidates = sample_candidates(
                    error_mask=depth_valid, num_samples=max_new, strategy='uniform')
                if candidates.numel():
                    in_error_mask = error_masks['combined_mask'][candidates[:, 1], candidates[:, 0]]
                    n_spawned_fallback = int((~in_error_mask).sum().item())
            else:
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
            if spawn_control and spawn_control.get('candidate_mode') != 'uniform_depth':
                if candidates.numel():
                    valid_candidates = depth[candidates[:, 1], candidates[:, 0]] > 0
                    candidates = candidates[valid_candidates]
            if (spawn_control and spawn_control.get('candidate_mode') != 'uniform_depth'
                    and candidates.shape[0] < max_new):
                # For matched-count experiments, fill a shortfall from distinct
                # observed-depth pixels. This keeps the requested spawn count
                # independent of how large the current error mask happens to be.
                fallback_mask = depth_valid.clone()
                if candidates.numel():
                    fallback_mask[candidates[:, 1], candidates[:, 0]] = False
                n_needed = max_new - int(candidates.shape[0])
                fallback = sample_candidates(
                    error_mask=fallback_mask,
                    num_samples=n_needed,
                    strategy='uniform',
                )
                n_spawned_fallback = int(fallback.shape[0])
                if n_spawned_fallback:
                    candidates = torch.cat([candidates, fallback], dim=0)
            if candidates.shape[0] > 0:
                trace_candidates = candidates
                new_gaussians = create_gaussians_from_candidates(
                    candidates, rgb, depth,
                    self.intrinsics, self.current_pose,
                    scale_mode=self.config['gaussian'].get('scale_mode', 'depth_adaptive'),
                    scale_pixel_multiplier=self.config['gaussian'].get('scale_pixel_multiplier', 1.0),
                    initial_scale=self.config['gaussian'].get('initial_scale', 0.01),
                    initial_opacity=self.config['gaussian'].get('initial_opacity', 0.5),
                )
                if new_gaussians['xyz'].shape[0] > 0:
                    n_spawned = int(new_gaussians['xyz'].shape[0])
                    self._spawn_control_count += n_spawned
                    self.gaussian_model.add_gaussians(new_gaussians, frame_idx=self.frame_count)
                    self.importance_estimator.expand_buffers(
                        new_gaussians['xyz'].shape[0], self.device
                    )
                    # Re-register newly created Parameter objects and extend optimizer state
                    # without destroying historical momentum/variance buffers
                    if self.optimizer is not None:
                        self._reregister_optimizer_params()
                        self.optimizer.extend_state(new_gaussians['xyz'].shape[0], device=self.device)
                    else:
                        self._setup_optimizer()

        self._trace_stage('spawn_decision_and_map', {
            'candidate_pixels': trace_candidates,
            'xyz': self.gaussian_model._xyz,
            'scaling': self.gaussian_model._scaling,
            'rotation': self.gaussian_model._rotation,
            'opacity': self.gaussian_model._opacity,
            'features_dc': self.gaussian_model._features_dc,
            'features_rest': self.gaussian_model._features_rest,
        })
        
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
        
        # Estimate per-Gaussian compute costs (A2: Controlled Cost Model with K and backend awareness)
        use_cost_model = self.config.get('scheduler', {}).get('use_cost_model', True)
        use_attribution = self.config.get('importance', {}).get('use_attribution', True)
        n_micro_steps = self.config.get('training', {}).get('n_micro_steps', 5)
        renderer_backend = self.config.get('rendering', {}).get('backend', 'gsplat')

        if not use_cost_model:
            base_costs = torch.full((N_updated,), 0.5, device=self.device)
            step_costs = torch.full((N_updated,), 0.5, device=self.device)
            cost_estimates = base_costs + float(n_micro_steps) * step_costs
        else:
            sched_cfg = self.config.get('scheduler', {})
            base_costs, step_costs = estimate_gaussian_cost_components(
                screen_areas=getattr(self.importance_estimator, '_screen_areas', None) if (use_attribution and renderer_backend != 'gsplat') else None,
                n_gaussians=N_updated,
                base_cost_us=sched_cfg.get('cost_per_gaussian_us', 0.10 if renderer_backend == 'gsplat' else 0.5),
                area_cost_factor=sched_cfg.get('area_cost_factor', 0.0 if renderer_backend == 'gsplat' else 0.002),
                cost_backward_per_step_us=sched_cfg.get('cost_backward_per_step_us', 0.05 if renderer_backend == 'gsplat' else 0.35),
                cost_optimizer_per_step_us=sched_cfg.get('cost_optimizer_per_step_us', 0.03 if renderer_backend == 'gsplat' else 0.15),
                sh_degree=self.gaussian_model.sh_degree,
                backend=renderer_backend,
                device=self.device,
            )
            cost_estimates = base_costs + float(n_micro_steps) * step_costs
        
        policy = self.config['scheduler'].get('policy', 'budget_aware')
        use_knapsack = self.config.get('scheduler', {}).get('use_knapsack', True)
        if not use_knapsack and policy in ['budget_aware', 'ours']:
            policy = 'top_k'  # Greedy top-k utility ranking
        if routed_policy is not None:
            policy = routed_policy
            
        ratio = self.config['scheduler'].get('optimize_ratio', 0.5)
        
        error_scores = None
        error_influence_scores = None
        error_influence_temporal_scores = None
        if self.importance_estimator._running_depth_error is not None and self.importance_estimator._running_color_error is not None:
            error_scores = self.importance_estimator._running_depth_error + self.importance_estimator._running_color_error
            error_influence_scores = self.importance_estimator.compute_error_influence_score(use_temporal=False)
            error_influence_temporal_scores = self.importance_estimator.compute_error_influence_score(use_temporal=True)
        
        top_k = self.config['scheduler'].get('top_k', None)
        binary_threshold = self.config['scheduler'].get('binary_threshold', 0.5)

        # Warm-up configuration (Age-Aware Guaranteed Warm-up)
        sched_cfg = self.config.get('scheduler', {})
        enable_warmup = sched_cfg.get('enable_warmup', True)
        warmup_steps = sched_cfg.get('warmup_steps', 3)
        warmup_budget_ratio = sched_cfg.get('warmup_budget_ratio', 0.20) if enable_warmup else 0.0
        warmup_k = sched_cfg.get('warmup_k', 2)

        # Ensure visibility mask matches N_updated
        if visibility_mask.shape[0] < N_updated:
            visibility_mask = torch.cat([
                visibility_mask,
                torch.ones(N_updated - visibility_mask.shape[0], dtype=torch.bool, device=self.device)
            ])
        update_counts = self.gaussian_model.update_counts

        # Compute learned utility predictions for current frame Gaussians if predictor is active
        if self.utility_predictor is not None:
            feat_tensor = extract_online_feature_tensor(self, N_updated)
            pred_out = self.utility_predictor.predict_tensor(feat_tensor)
            raw_u = pred_out['predicted_utility']
            if raw_u.numel() > 0 and (raw_u <= 0).all():
                self._learned_utility_scores = raw_u - raw_u.min() + 1e-4
            else:
                self._learned_utility_scores = raw_u
            self._predicted_delta_q = pred_out['predicted_delta_q']
            self._predicted_delta_t = pred_out['predicted_delta_t']

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
                error_influence_temporal_scores=error_influence_temporal_scores,
                ratio=ratio,
                top_k=top_k,
                frame_idx=self.frame_count,
                binary_threshold=binary_threshold,
                utility_scores=getattr(self, '_learned_utility_scores', None),
                seed=self.config.get('seed', 42),
                update_counts=update_counts if enable_warmup else None,
                visibility_mask=visibility_mask if enable_warmup else None,
                warmup_steps=warmup_steps,
                warmup_budget_ratio=warmup_budget_ratio,
            )
        use_adaptive_k = self.config.get('training', {}).get('use_adaptive_k', False)
        k_alloc = None
        if policy in ("no_op", OptimizationPolicy.NO_OP.value):
            optimize_mask = torch.zeros(N_updated, dtype=torch.bool, device=self.device)
            k_alloc = None
        elif policy in ("full", OptimizationPolicy.FULL.value):
            optimize_mask = torch.ones(N_updated, dtype=torch.bool, device=self.device)
            k_alloc = None
        elif use_adaptive_k and self.optimizer is not None and policy in ("ours", "budget_aware", "learned_utility", OptimizationPolicy.OURS.value, OptimizationPolicy.BUDGET_AWARE.value, OptimizationPolicy.LEARNED_UTILITY.value):
            if policy in ("ours", "learned_utility", OptimizationPolicy.OURS.value, OptimizationPolicy.LEARNED_UTILITY.value) and self._learned_utility_scores is not None:
                step_importance = self._learned_utility_scores
            elif error_influence_temporal_scores is not None:
                step_importance = error_influence_temporal_scores
            elif error_influence_scores is not None:
                step_importance = error_influence_scores
            else:
                step_importance = importance

            k_alloc = self.scheduler.allocate_adaptive_micro_steps(
                importance_scores=step_importance,
                base_costs=base_costs,
                step_costs=step_costs,
                max_k=self.config.get('training', {}).get('n_micro_steps', 5),
                update_counts=update_counts if enable_warmup else None,
                visibility_mask=visibility_mask if enable_warmup else None,
                warmup_steps=warmup_steps,
                warmup_budget_ratio=warmup_budget_ratio,
                warmup_k=warmup_k,
            )
            optimize_mask = (k_alloc > 0)
        else:
            k_alloc = None

        # Optional causal-control hook: force an exact number of Gaussian
        # micro-updates per frame while retaining the current importance
        # ranking. This matches optimizer update count, not wall-clock cost.
        optimization_control = self.config.get('training', {}).get('optimization_control')
        age_quota_mature_updates = None
        age_quota_young_updates = None
        stale_quota_micro_updates = None
        other_quota_micro_updates = None
        randomized_assignment_diagnostics = None
        if optimization_control:
            target_updates = max(0, int(optimization_control['micro_updates_per_frame']))
            max_k = max(1, int(self.config.get('training', {}).get('n_micro_steps', 5)))
            scores = importance[:N_updated].float()
            scores = torch.nan_to_num(scores, nan=float('-inf'), posinf=float('inf'), neginf=float('-inf'))
            k_alloc = torch.zeros(N_updated, dtype=torch.long, device=self.device)
            if 'fixed_persistent_id_k_alloc' in optimization_control:
                fixed_entries = optimization_control['fixed_persistent_id_k_alloc']
                store_ids = self.gaussian_model.state_store.persistent_ids[:N_updated]
                order_ids = torch.argsort(store_ids)
                sorted_ids = store_ids[order_ids]
                requested_ids = torch.as_tensor(
                    [int(entry['persistent_id']) for entry in fixed_entries],
                    dtype=torch.long, device=self.device)
                requested_k = torch.as_tensor(
                    [int(entry['k']) for entry in fixed_entries],
                    dtype=torch.long, device=self.device)
                if requested_ids.numel():
                    positions = torch.searchsorted(sorted_ids, requested_ids)
                    in_range = positions < sorted_ids.numel()
                    safe_positions = positions.clamp(max=max(sorted_ids.numel() - 1, 0))
                    found = (in_range & (sorted_ids[safe_positions] == requested_ids)
                             if sorted_ids.numel() else torch.zeros_like(in_range))
                    if (optimization_control.get('fixed_persistent_id_k_alloc_strict', True)
                            and not bool(found.all())):
                        missing = requested_ids[~found].detach().cpu().tolist()[:8]
                        raise RuntimeError(
                            f"Fixed update schedule references missing persistent IDs: {missing}")
                    resolved = order_ids[safe_positions[found]]
                    resolved_k = requested_k[found]
                    if (optimization_control.get('fixed_persistent_id_k_alloc_strict', True)
                            and (resolved_k < 0).any()):
                        raise ValueError("Fixed per-ID update counts must be nonnegative")
                    k_alloc[resolved] = resolved_k.clamp(min=0, max=max_k)
            elif 'mature_update_fraction' in optimization_control:
                ages_control = self.gaussian_model.state_store.creation_frames[:N_updated]
                mature_control = (self.frame_count - ages_control) >= int(optimization_control.get('mature_age_frames', 10))
                visible_control = visibility_mask[:N_updated].bool()
                young_candidates = visible_control & ~mature_control
                mature_candidates = visible_control & mature_control
                cap_young = int(young_candidates.sum().item()) * max_k
                cap_mature = int(mature_candidates.sum().item()) * max_k
                alloc_total = min(target_updates, cap_young + cap_mature)
                fraction_setting = optimization_control['mature_update_fraction']
                if isinstance(fraction_setting, str):
                    if fraction_setting != 'coverage_saturating':
                        raise ValueError(
                            f"Unknown mature_update_fraction policy: {fraction_setting}"
                        )
                    # Allocate at most one update to each visible mature
                    # Gaussian, while retaining a fixed upper bound so new
                    # Gaussians keep a minimum share of the frame budget.
                    fraction_cap = min(1.0, max(0.0, float(
                        optimization_control.get('mature_update_fraction_cap', 0.75)
                    )))
                    desired_mature_updates = min(
                        round(alloc_total * fraction_cap),
                        int(mature_candidates.sum().item()),
                    )
                    fraction = (desired_mature_updates / alloc_total
                                if alloc_total > 0 else 0.0)
                else:
                    fraction = min(1.0, max(0.0, float(fraction_setting)))
                age_quota_mature_updates = min(round(alloc_total * fraction), cap_mature)
                age_quota_young_updates = min(alloc_total - age_quota_mature_updates, cap_young)
                # Transfer quota that one age group cannot use to the other.
                unallocated = alloc_total - age_quota_mature_updates - age_quota_young_updates
                extra_mature = min(unallocated, cap_mature - age_quota_mature_updates)
                age_quota_mature_updates += extra_mature
                unallocated -= extra_mature
                age_quota_young_updates += min(unallocated, cap_young - age_quota_young_updates)

                def assign_age_quota(mask, quota):
                    if quota <= 0 or not mask.any():
                        return
                    ids = torch.where(mask)[0]
                    order = ids[torch.argsort(scores[ids], descending=True, stable=True)]
                    left = int(quota)
                    for _ in range(max_k):
                        if left <= 0:
                            break
                        take = min(left, len(order))
                        k_alloc[order[:take]] += 1
                        left -= take

                assign_age_quota(mature_candidates, age_quota_mature_updates)
                assign_age_quota(young_candidates, age_quota_young_updates)
            elif 'stale_update_fraction' in optimization_control:
                # Mechanism experiment: reserve a controlled share of the same
                # update budget for visible mature Gaussians that have not been
                # optimized for a configurable number of frames. This is an
                # opt-in intervention; ordinary scheduler behavior is unchanged.
                store = self.gaussian_model.state_store
                mature_age = int(optimization_control.get('mature_age_frames', 10))
                stale_after = int(optimization_control.get('stale_after_frames', 10))
                ages_control = store.creation_frames[:N_updated]
                last_updates_control = store.last_update_frames[:N_updated]
                mature_control = (self.frame_count - ages_control) >= mature_age
                stale_control = (self.frame_count - last_updates_control) >= stale_after
                visible_control = visibility_mask[:N_updated].bool()
                stale_candidates = visible_control & mature_control & stale_control
                # The stale cohort is visibility constrained. Fill the rest of
                # the same global update budget from every non-stale Gaussian,
                # matching the ordinary exact-update selector's candidate
                # domain. Restricting this pool to visible Gaussians can leave
                # the rescue arm with fewer updates than its matched control on
                # frames with sparse visibility.
                other_candidates = ~stale_candidates
                cap_stale = int(stale_candidates.sum().item()) * max_k
                cap_other = int(other_candidates.sum().item()) * max_k
                alloc_total = min(target_updates, cap_stale + cap_other)
                fraction = min(1.0, max(0.0, float(optimization_control['stale_update_fraction'])))
                stale_quota = min(round(alloc_total * fraction), cap_stale)
                other_quota = min(alloc_total - stale_quota, cap_other)
                unallocated = alloc_total - stale_quota - other_quota
                extra_stale = min(unallocated, cap_stale - stale_quota)
                stale_quota += extra_stale
                unallocated -= extra_stale
                other_quota += min(unallocated, cap_other - other_quota)

                def assign_stale_quota(mask, quota):
                    if quota <= 0 or not mask.any():
                        return
                    ids = torch.where(mask)[0]
                    order = ids[torch.argsort(scores[ids], descending=True, stable=True)]
                    left = int(quota)
                    for _ in range(max_k):
                        if left <= 0:
                            break
                        available = order[k_alloc[order] < max_k]
                        take = min(left, len(available))
                        if take <= 0:
                            break
                        chosen = available[:take]
                        k_alloc[chosen] += 1
                        left -= take

                assign_stale_quota(stale_candidates, stale_quota)
                assign_stale_quota(other_candidates, other_quota)
                stale_quota_micro_updates = stale_quota
                other_quota_micro_updates = other_quota
            elif 'randomized_candidate_persistent_ids' in optimization_control:
                # Local randomized trial: treatment is assigned within the
                # supplied pre-snapshot cohort, further blocked by age,
                # staleness, and current-view residual. One optimizer step is
                # assigned to treated IDs; controls receive none.
                store = self.gaussian_model.state_store
                requested_ids = torch.as_tensor(
                    optimization_control['randomized_candidate_persistent_ids'],
                    dtype=torch.long, device=self.device)
                ids_now = store.persistent_ids[:N_updated]
                order_ids = torch.argsort(ids_now)
                sorted_ids = ids_now[order_ids]
                positions = torch.searchsorted(sorted_ids, requested_ids)
                in_range = positions < sorted_ids.numel()
                safe_pos = positions.clamp(max=max(sorted_ids.numel() - 1, 0))
                found = (in_range & (sorted_ids[safe_pos] == requested_ids)
                         if sorted_ids.numel() else torch.zeros_like(in_range))
                cohort_idx = order_ids[safe_pos[found]] if sorted_ids.numel() else torch.empty(0, dtype=torch.long, device=self.device)
                eligible = torch.zeros(N_updated, dtype=torch.bool, device=self.device)
                if cohort_idx.numel():
                    cohort_idx = cohort_idx[visibility_mask[cohort_idx].bool()]
                    cohort_idx = cohort_idx[self.gaussian_model._state[cohort_idx] != GaussianState.PRUNED]
                    eligible[cohort_idx] = True

                ages_now = (self.frame_count - store.creation_frames[:N_updated]).clamp(min=0)
                stale_now = (self.frame_count - store.last_update_frames[:N_updated]).clamp(min=0)
                rgb_now = torch.nan_to_num(per_gaussian_color_err[:N_updated].float(), nan=0.0, posinf=0.0, neginf=0.0)
                age_groups = {'young': ages_now < int(optimization_control.get('mature_age_frames', 10)),
                              'mature': ages_now >= int(optimization_control.get('mature_age_frames', 10))}
                stale_groups = {'0_2': stale_now <= 2,
                                '3_9': (stale_now >= 3) & (stale_now <= 9),
                                '10_plus': stale_now >= 10}
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(optimization_control.get('randomization_seed', 0)))
                treated_indices = []
                assignment_strata = []
                assignment_units = []
                cohort_base = eligible.clone()
                # Residual median is calculated within each age/staleness block
                # before assignment; ties remain in the lower half.
                for age_name, age_mask in age_groups.items():
                    for stale_name, stale_mask in stale_groups.items():
                        block = torch.where(cohort_base & age_mask & stale_mask)[0]
                        if block.numel() < 2:
                            if block.numel():
                                eligible[block] = False
                                assignment_units.extend({'persistent_id': int(store.persistent_ids[idx].item()),
                                    'stratum': f'{age_name}/{stale_name}/not_randomizable', 'treated': False}
                                    for idx in block)
                                assignment_strata.append({'stratum': f'{age_name}/{stale_name}',
                                    'residual_band': 'not_randomizable', 'eligible': int(block.numel()),
                                    'treated': 0, 'control': int(block.numel())})
                            continue
                        vals = rgb_now[block]
                        median = torch.quantile(vals, 0.5)
                        bands = {'residual_low': block[vals <= median],
                                 'residual_high': block[vals > median]}
                        # If ties collapse a band, retain one residual band so
                        # the block can still be randomized without selection.
                        nonempty = [(name, idx) for name, idx in bands.items() if idx.numel()]
                        if len(nonempty) == 1:
                            nonempty = [('residual_all', block)]
                        for band_name, candidates in nonempty:
                            if candidates.numel() < 2:
                                eligible[candidates] = False
                                assignment_units.extend({'persistent_id': int(store.persistent_ids[idx].item()),
                                    'stratum': f'{age_name}/{stale_name}/{band_name}/not_randomizable', 'treated': False}
                                    for idx in candidates)
                                assignment_strata.append({'stratum': f'{age_name}/{stale_name}',
                                    'residual_band': f'{band_name}/not_randomizable', 'eligible': int(candidates.numel()),
                                    'treated': 0, 'control': int(candidates.numel())})
                                continue
                            perm = torch.randperm(candidates.numel(), generator=generator, device=self.device)
                            n_treated = int(candidates.numel() // 2)
                            if candidates.numel() % 2 and bool(torch.randint(0, 2, (), generator=generator, device=self.device).item()):
                                n_treated += 1
                            chosen = candidates[perm[:n_treated]]
                            chosen_set = set(chosen.detach().cpu().tolist())
                            assignment_units.extend({'persistent_id': int(store.persistent_ids[idx].item()),
                                'stratum': f'{age_name}/{stale_name}/{band_name}',
                                'treated': int(idx.item()) in chosen_set}
                                for idx in candidates)
                            treated_indices.append(chosen)
                            assignment_strata.append({'stratum': f'{age_name}/{stale_name}',
                                'residual_band': band_name, 'eligible': int(candidates.numel()),
                                'treated': int(n_treated), 'control': int(candidates.numel() - n_treated)})
                if treated_indices:
                    chosen = torch.cat(treated_indices)
                    k_alloc[chosen] = 1
                else:
                    chosen = torch.empty(0, dtype=torch.long, device=self.device)
                selected_ids = store.persistent_ids[chosen]
                randomized_assignment_diagnostics = {
                    'assignment': '1:1 randomization within age x staleness x current-view RGB-residual median blocks',
                    'seed': int(optimization_control.get('randomization_seed', 0)),
                    'cohort_requested': int(requested_ids.numel()),
                    'cohort_present_and_visible': int(cohort_idx.numel()),
                    'randomizable_ids': int(sum(x['eligible'] for x in assignment_strata if not x['residual_band'].endswith('not_randomizable'))),
                    'treated_ids': int(chosen.numel()),
                    'control_ids': int(sum(x['control'] for x in assignment_strata if not x['residual_band'].endswith('not_randomizable'))),
                    'selected_persistent_ids': selected_ids.detach().cpu().tolist(),
                    'strata': assignment_strata,
                    'assignment_units': assignment_units,
                    'micro_update_policy': 'one update per treated ID; all other cohort IDs receive zero',
                }
            else:
                order = torch.argsort(scores, descending=True, stable=True)
                remaining = min(target_updates, N_updated * max_k)
                for _ in range(max_k):
                    if remaining <= 0:
                        break
                    take = min(remaining, N_updated)
                    k_alloc[order[:take]] += 1
                    remaining -= take
            optimize_mask = k_alloc > 0

        self._last_optimize_mask = optimize_mask
        self._last_k_alloc = (k_alloc.detach().clone() if k_alloc is not None
                              else optimize_mask.to(torch.long))
        self._trace_stage('update_selection', {
            'optimize_mask': optimize_mask,
            'k_alloc': k_alloc,
            'importance': importance,
        })
        # Snapshot before optimization/pruning: these are actual selected primitives
        # and modeled optimization costs, grouped by age at the current frame.
        ages = self.gaussian_model.state_store.creation_frames[:N_updated]
        young = (self.frame_count - ages) < 10
        selected_young = optimize_mask[:N_updated] & young
        selected_old = optimize_mask[:N_updated] & ~young
        selected_young_count = int(selected_young.sum().item())
        selected_old_count = int(selected_old.sum().item())
        young_population = int(young.sum().item())
        old_population = N_updated - young_population
        mature_eligible_mask = (~young) & visibility_mask[:N_updated].bool()
        mature_updated_mask = mature_eligible_mask & optimize_mask[:N_updated]
        mature_eligible_count = int(mature_eligible_mask.sum().item())
        mature_updated_count = int(mature_updated_mask.sum().item())
        mature_update_coverage = mature_updated_count / mature_eligible_count if mature_eligible_count else 0.0
        stale_after_frames = int((optimization_control or {}).get('stale_after_frames', 10))
        stale_frames = (self.frame_count - self.gaussian_model.state_store.last_update_frames[:N_updated]).clamp(min=0)
        stale_mature_eligible_mask = mature_eligible_mask & (stale_frames >= stale_after_frames)
        stale_mature_updated_mask = stale_mature_eligible_mask & optimize_mask[:N_updated]
        stale_mature_eligible_count = int(stale_mature_eligible_mask.sum().item())
        stale_mature_updated_count = int(stale_mature_updated_mask.sum().item())
        stale_mature_micro_updates = (int(k_alloc[stale_mature_eligible_mask].sum().item())
                                      if k_alloc is not None and stale_mature_eligible_count else 0)
        stale_mature_update_coverage = (stale_mature_updated_count / stale_mature_eligible_count
                                        if stale_mature_eligible_count else 0.0)

        # Optional read-only summaries for same-snapshot quota replay studies.
        # Attribution tensors describe the pre-spawn render, so newly spawned
        # Gaussians are deliberately excluded from residual statistics.
        age_stratified_diagnostics = None
        identity_trace = None
        if optimization_control and optimization_control.get('capture_age_diagnostics', False):
            n_attributed = min(
                int(per_gaussian_color_err.shape[0]),
                int(per_gaussian_depth_err.shape[0]),
                int(visibility_mask.shape[0]),
                int(self.gaussian_model.state_store.creation_frames.shape[0]),
            )
            diag_ages = self.frame_count - self.gaussian_model.state_store.creation_frames[:n_attributed]
            diag_active = self.gaussian_model._state[:n_attributed] != GaussianState.PRUNED
            diag_visible = visibility_mask[:n_attributed].bool() & diag_active
            diag_color = per_gaussian_color_err[:n_attributed]
            diag_depth = per_gaussian_depth_err[:n_attributed]

            def age_summary(age_mask):
                population = age_mask & diag_active
                visible = age_mask & diag_visible
                count = int(visible.sum().item())
                if count:
                    color_mean = float(diag_color[visible].float().mean().item())
                    depth_mean = float(diag_depth[visible].float().mean().item())
                else:
                    color_mean = 0.0
                    depth_mean = 0.0
                return {
                    'population_attributed': int(population.sum().item()),
                    'visible_attributed': count,
                    'visible_fraction_attributed': count / max(int(population.sum().item()), 1),
                    'visible_color_error_mean': color_mean,
                    'visible_depth_error_mean': depth_mean,
                }

            diag_mature = diag_ages >= 10
            diag_young = ~diag_mature
            diag_staleness = (
                self.frame_count - self.gaussian_model.state_store.last_update_frames[:n_attributed]
            ).clamp(min=0)
            diag_update_counts = self.gaussian_model.state_store.update_counts[:n_attributed]
            revisit_bins = (("staleness_0_2_frames", 0, 2),
                            ("staleness_3_5_frames", 3, 5),
                            ("staleness_6_9_frames", 6, 9),
                            ("staleness_10_19_frames", 10, 19),
                            ("staleness_20_plus_frames", 20, None))
            mature_revisit_strata = {}
            for bin_name, lower, upper in revisit_bins:
                in_staleness = diag_staleness >= lower
                if upper is not None:
                    in_staleness &= diag_staleness <= upper
                group = diag_mature & diag_active & in_staleness
                visible_group = group & diag_visible
                group_count = int(group.sum().item())
                visible_count = int(visible_group.sum().item())
                mature_revisit_strata[bin_name] = {
                    'population_attributed': group_count,
                    'visible_attributed': visible_count,
                    'visible_fraction_attributed': visible_count / max(group_count, 1),
                    'mean_staleness_frames': (float(diag_staleness[group].float().mean().item())
                                              if group_count else 0.0),
                    'mean_update_count': (float(diag_update_counts[group].float().mean().item())
                                          if group_count else 0.0),
                    'visible_color_error_mean': (float(diag_color[visible_group].float().mean().item())
                                                 if visible_count else 0.0),
                    'visible_depth_error_mean': (float(diag_depth[visible_group].float().mean().item())
                                                 if visible_count else 0.0),
                }
            age_stratified_diagnostics = {
                'stage': 'pre_optimization; attribution from pre_spawn_render',
                'mature_age_frames': 10,
                'n_gaussians_attributed': n_attributed,
                'n_gaussians_after_spawn': int(N_updated),
                'unattributed_after_spawn': max(0, int(N_updated) - n_attributed),
                'view_coverage_valid_depth': float(current_coverage),
                'young_lt_10_frames': age_summary(diag_young),
                'mature_ge_10_frames': age_summary(diag_mature),
                'mature_revisit_strata': mature_revisit_strata,
            }

        # Optional bounded trace of a cohort selected at the common snapshot.
        # IDs are resolved on every frame, so compaction/reordering does not
        # change cohort identity. Residuals below are attribution errors for
        # the current input view, not post-optimizer residuals.
        trace_ids_cfg = (optimization_control or {}).get('trace_persistent_ids')
        if trace_ids_cfg:
            store = self.gaussian_model.state_store
            trace_ids = torch.as_tensor(trace_ids_cfg, dtype=torch.long, device=self.device)
            n_trace_state = min(N_updated, int(store.persistent_ids.shape[0]),
                                int(visibility_mask.shape[0]),
                                int(per_gaussian_color_err.shape[0]),
                                int(per_gaussian_depth_err.shape[0]))
            ids_now = store.persistent_ids[:n_trace_state]
            order_ids = torch.argsort(ids_now)
            sorted_ids = ids_now[order_ids]
            positions = torch.searchsorted(sorted_ids, trace_ids)
            in_range = positions < sorted_ids.numel()
            safe_positions = positions.clamp(max=max(sorted_ids.numel() - 1, 0))
            present = in_range & (sorted_ids[safe_positions] == trace_ids) if sorted_ids.numel() else torch.zeros_like(in_range)
            indices = order_ids[safe_positions] if sorted_ids.numel() else torch.zeros_like(safe_positions)
            active_trace = present & (self.gaussian_model._state[indices] != GaussianState.PRUNED) if sorted_ids.numel() else present
            present = active_trace
            def trace_values(tensor, default=0):
                out = torch.full(trace_ids.shape, default, dtype=tensor.dtype, device=self.device)
                if present.any():
                    out[present] = tensor[indices[present]]
                return out.detach().cpu().tolist()
            identity_trace = {
                'persistent_ids': trace_ids.detach().cpu().tolist(),
                'present_pre': present.detach().cpu().tolist(),
                'visible_pre': (visibility_mask[indices].bool() & present).detach().cpu().tolist() if sorted_ids.numel() else [False] * len(trace_ids),
                'age_frames_pre': (self.frame_count - store.creation_frames[indices]).clamp(min=0).detach().cpu().tolist() if sorted_ids.numel() else [0] * len(trace_ids),
                'staleness_frames_pre': (self.frame_count - store.last_update_frames[indices]).clamp(min=0).detach().cpu().tolist() if sorted_ids.numel() else [0] * len(trace_ids),
                'rgb_residual_pre': (per_gaussian_color_err[indices].detach().float().cpu().tolist() if sorted_ids.numel() else [0.0] * len(trace_ids)),
                'depth_residual_pre': (per_gaussian_depth_err[indices].detach().float().cpu().tolist() if sorted_ids.numel() else [0.0] * len(trace_ids)),
                'selected_pre': (optimize_mask[indices].detach().cpu().tolist() if sorted_ids.numel() else [False] * len(trace_ids)),
                'micro_updates_allocated_pre': (k_alloc[indices].detach().cpu().tolist() if sorted_ids.numel() else [0] * len(trace_ids)),
                'last_update_frame_pre': trace_values(store.last_update_frames),
                'update_count_pre': trace_values(store.update_counts),
                'stage': 'pre_optimizer_attribution; residual is current-view pre-update error',
            }
        budget_young_us = float(cost_estimates[selected_young].sum().item())
        budget_old_us = float(cost_estimates[selected_old].sum().item())
        
        # === 7. True Selective Optimization with Frozen Background Cache (R21/R29) ===
        n_optimized = int(optimize_mask.sum().item()) if optimize_mask is not None else 0
        opt_loss_val = 0.0
        cache_time = 0.0
        n_cache_rebuilds = 0
        cache_cuda_event_pairs = []
        cache_cuda_event_ms = 0.0
        optimizer_forward_cuda_event_pairs = []
        optimizer_backward_cuda_event_pairs = []
        optimizer_backward_only_cuda_event_pairs = []
        optimizer_step_cuda_event_pairs = []
        optimizer_iteration_count = 0
        optimizer_forward_cuda_event_ms = 0.0
        optimizer_backward_cuda_event_ms = 0.0
        optimizer_backward_only_cuda_event_ms = 0.0
        optimizer_step_cuda_event_ms = 0.0
        opt_time = 0.0
        opt_stage_sync_ms = 0.0
        opt_stage_cuda_event_ms = 0.0
        sync_stage_timing = bool(self.config.get('system', {}).get('synchronized_stage_timing', False))
        sync_stage_start = None
        sync_stage_start_unix_ns = None
        sync_stage_end_unix_ns = None
        cuda_stage_start_event = None
        cuda_stage_end_event = None
        if sync_stage_timing and str(self.device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            cuda_stage_start_event = torch.cuda.Event(enable_timing=True)
            cuda_stage_end_event = torch.cuda.Event(enable_timing=True)
            cuda_stage_start_event.record()
            sync_stage_start_unix_ns = time.time_ns()
            sync_stage_start = time.perf_counter()
        
        if k_alloc is not None and (k_alloc > 0).any():
            mean_k = float(k_alloc[k_alloc > 0].float().mean().item())
        elif n_optimized > 0:
            mean_k = float(self.config.get('training', {}).get('n_micro_steps', 5))
        else:
            mean_k = 0.0
        
        if optimize_mask.any() and self.optimizer is not None:
            # 1. Build / refresh background cache for frozen Gaussians once per frame
            frozen_mask = ~optimize_mask[:self.gaussian_model.num_gaussians]
            if frozen_mask.any():
                c_start = time.time()
                cache_event_start = None
                if sync_stage_timing and str(self.device).startswith('cuda') and torch.cuda.is_available():
                    cache_event_start = torch.cuda.Event(enable_timing=True)
                    cache_event_start.record()
                self.bg_cache.build_cache(
                    model=self.gaussian_model,
                    frozen_mask=frozen_mask,
                    extrinsics=self.current_pose,
                    intrinsics=self.intrinsics,
                    image_width=W,
                    image_height=H,
                    tile_size=self.config['rendering']['tile_size'],
                    backend=renderer_backend,
                )
                if cache_event_start is not None:
                    cache_event_end = torch.cuda.Event(enable_timing=True)
                    cache_event_end.record()
                    cache_cuda_event_pairs.append((cache_event_start, cache_event_end))
                cache_time = time.time() - c_start
                n_cache_rebuilds = 1
            else:
                self.bg_cache.invalidate()
                
            # 2. Pure Selective Optimization Step (M only) with Multi-Step Convergence
            opt_start = time.time()
            n_micro_steps = int(k_alloc.max().item()) if (use_adaptive_k and k_alloc is not None and k_alloc.numel() > 0) else self.config.get('training', {}).get('n_micro_steps', 5)
            weights = {
                'color': self.config.get('losses', {}).get('weight_color', 0.8),
                'depth': self.config.get('losses', {}).get('weight_depth', 0.5),
                'ssim': self.config.get('losses', {}).get('weight_ssim', 0.2),
                'ssim_implementation': self.config.get('losses', {}).get('ssim_implementation', 'direct'),
            }
            
            n_optimized = 0
            n_gaussians = self.gaussian_model.num_gaussians
            for step_i in range(n_micro_steps):
                step_mask = (k_alloc > step_i) if (use_adaptive_k and k_alloc is not None) else optimize_mask
                if not step_mask.any():
                    break
                optimizer_iteration_count += 1

                # Progressive Background Caching (True Adaptive-K Compute Scaling):
                # When primitives graduate at step_i (k_alloc == step_i), absorb them into
                # the frozen background cache with their latest updated parameters.
                # active_subset then strictly extracts M_k <= M elements for forward & backward.
                if use_adaptive_k and k_alloc is not None and step_i > 0 and (k_alloc[:n_gaussians] == step_i).any():
                    frozen_mask = (k_alloc[:n_gaussians] <= step_i)
                    rb_start = time.time()
                    cache_event_start = None
                    if sync_stage_timing and str(self.device).startswith('cuda') and torch.cuda.is_available():
                        cache_event_start = torch.cuda.Event(enable_timing=True)
                        cache_event_start.record()
                    self.bg_cache.build_cache(
                        model=self.gaussian_model,
                        frozen_mask=frozen_mask,
                        extrinsics=self.current_pose,
                        intrinsics=self.intrinsics,
                        image_width=W,
                        image_height=H,
                        tile_size=self.config['rendering']['tile_size'],
                        backend=renderer_backend,
                    )
                    if cache_event_start is not None:
                        cache_event_end = torch.cuda.Event(enable_timing=True)
                        cache_event_end.record()
                        cache_cuda_event_pairs.append((cache_event_start, cache_event_end))
                    cache_time += (time.time() - rb_start)
                    n_cache_rebuilds += 1

                self.optimizer.zero_grad()
                # Active subset contains strictly {i | K_i > step_i} (O(M_k) backward compute)
                forward_event_start = None
                if sync_stage_timing and str(self.device).startswith('cuda') and torch.cuda.is_available():
                    forward_event_start = torch.cuda.Event(enable_timing=True)
                    forward_event_start.record()
                active_subset = self.gaussian_model.get_optimization_subset(step_mask, extrinsics=self.current_pose)
                
                composite_opt = self.bg_cache.composite_with_active(
                    active_subset=active_subset,
                    extrinsics=self.current_pose,
                    intrinsics=self.intrinsics,
                    image_width=W,
                    image_height=H,
                    tile_size=self.config['rendering']['tile_size'],
                    backend=renderer_backend,
                )
                
                rendered_opt_color = composite_opt['color']
                rendered_opt_depth = composite_opt['depth']
                depth_valid_mask = (
                    (composite_opt['alpha'] >= alpha_threshold)
                    & torch.isfinite(depth) & (depth > 0)
                )
                
                losses = total_loss(
                    rendered_opt_color, rgb,
                    rendered_opt_depth, depth,
                    weights,
                    depth_valid_mask=depth_valid_mask,
                )
                if forward_event_start is not None:
                    forward_event_end = torch.cuda.Event(enable_timing=True)
                    forward_event_end.record()
                    optimizer_forward_cuda_event_pairs.append(
                        (forward_event_start, forward_event_end))
                
                # 4. Backward: computes gradients strictly for active_subset M_k
                backward_event_start = None
                if sync_stage_timing and str(self.device).startswith('cuda') and torch.cuda.is_available():
                    backward_event_start = torch.cuda.Event(enable_timing=True)
                    backward_event_start.record()
                if losses['total'].requires_grad:
                    backward_only_event_start = None
                    if sync_stage_timing and str(self.device).startswith('cuda') and torch.cuda.is_available():
                        backward_only_event_start = torch.cuda.Event(enable_timing=True)
                        backward_only_event_start.record()
                    losses['total'].backward()
                    if backward_only_event_start is not None:
                        backward_only_event_end = torch.cuda.Event(enable_timing=True)
                        backward_only_event_end.record()
                        optimizer_backward_only_cuda_event_pairs.append(
                            (backward_only_event_start, backward_only_event_end))
                    # 5. Selective Optimizer update: updates strictly M_k active parameters
                    optimizer_step_event_start = None
                    if sync_stage_timing and str(self.device).startswith('cuda') and torch.cuda.is_available():
                        optimizer_step_event_start = torch.cuda.Event(enable_timing=True)
                        optimizer_step_event_start.record()
                    self.optimizer.step(active_idx=active_subset['indices'])
                    if optimizer_step_event_start is not None:
                        optimizer_step_event_end = torch.cuda.Event(enable_timing=True)
                        optimizer_step_event_end.record()
                        optimizer_step_cuda_event_pairs.append(
                            (optimizer_step_event_start, optimizer_step_event_end))
                    n_optimized = max(n_optimized, step_mask.sum().item())
                else:
                    break
                if backward_event_start is not None:
                    backward_event_end = torch.cuda.Event(enable_timing=True)
                    backward_event_end.record()
                    optimizer_backward_cuda_event_pairs.append(
                        (backward_event_start, backward_event_end))
                    
            opt_loss_val = losses['total'].item()
            opt_time = time.time() - opt_start

        self._trace_stage('post_optimization', {
            'xyz': self.gaussian_model._xyz,
            'scaling': self.gaussian_model._scaling,
            'rotation': self.gaussian_model._rotation,
            'opacity': self.gaussian_model._opacity,
            'features_dc': self.gaussian_model._features_dc,
            'features_rest': self.gaussian_model._features_rest,
        })
        
        # End the optimization-stage timer before state bookkeeping, pruning,
        # and final-map quality rendering. End-to-end frame latency includes
        # all of those stages and is measured by the benchmark runner.
        if sync_stage_start is not None:
            cuda_stage_end_event.record()
            torch.cuda.synchronize(self.device)
            opt_stage_sync_ms = (time.perf_counter() - sync_stage_start) * 1000.0
            opt_stage_cuda_event_ms = float(cuda_stage_start_event.elapsed_time(cuda_stage_end_event))
            cache_cuda_event_ms = sum(float(start.elapsed_time(end))
                                      for start, end in cache_cuda_event_pairs)
            optimizer_forward_cuda_event_ms = sum(float(start.elapsed_time(end))
                                                  for start, end in optimizer_forward_cuda_event_pairs)
            optimizer_backward_cuda_event_ms = sum(float(start.elapsed_time(end))
                                                   for start, end in optimizer_backward_cuda_event_pairs)
            optimizer_backward_only_cuda_event_ms = sum(float(start.elapsed_time(end))
                                                        for start, end in optimizer_backward_only_cuda_event_pairs)
            optimizer_step_cuda_event_ms = sum(float(start.elapsed_time(end))
                                               for start, end in optimizer_step_cuda_event_pairs)
            sync_stage_end_unix_ns = time.time_ns()

        # Synchronize StateStore update counts and signals (Age-Aware Warmup lifecycle)
        if hasattr(self.gaussian_model, 'state_store') and self.gaussian_model.state_store is not None:
            padded_color_err = per_gaussian_color_err
            padded_depth_err = per_gaussian_depth_err
            padded_vis = visibility_mask
            if padded_color_err.shape[0] < N_updated:
                padded_color_err = torch.cat([
                    padded_color_err,
                    torch.zeros(N_updated - padded_color_err.shape[0], device=self.device)
                ])
            if padded_depth_err.shape[0] < N_updated:
                padded_depth_err = torch.cat([
                    padded_depth_err,
                    torch.zeros(N_updated - padded_depth_err.shape[0], device=self.device)
                ])
            if padded_vis.shape[0] < N_updated:
                padded_vis = torch.cat([
                    padded_vis,
                    torch.ones(N_updated - padded_vis.shape[0], dtype=torch.bool, device=self.device)
                ])
            self.gaussian_model.state_store.update_frame(
                frame_idx=self.frame_count,
                rgb_errors=padded_color_err,
                depth_errors=padded_depth_err,
                visibility_mask=padded_vis,
                optimized_mask=optimize_mask,
                positions=self.gaussian_model.positions.detach(),
            )
            if identity_trace is not None:
                identity_trace['last_update_frame_post'] = trace_values(store.last_update_frames)
                identity_trace['update_count_post'] = trace_values(store.update_counts)
                identity_trace['ema_rgb_post'] = trace_values(store.ema_rgb)
                identity_trace['ema_depth_post'] = trace_values(store.ema_depth)

        # === 8. Pruning ===
        enable_persistent_prune = self.config.get('densification', {}).get('enable_persistent_prune', True)
        if not self.config.get('densification', {}).get('disable_pruning', False):
            zero_frames = self.importance_estimator._zero_contrib_frames if enable_persistent_prune else None
            prune_low_value(
                self.gaussian_model, importance[:self.gaussian_model.num_gaussians],
                opacity_threshold=0.005,
                zero_contrib_frames=zero_frames,
                prune_patience=self.importance_estimator.prune_patience,
            )
        
        # Compact every 100 frames — preserve optimizer momentum and state store identity.
        # Fixed-decision causal rollouts may disable compaction so persistent IDs
        # remain comparable across the full replay horizon.
        if (self.frame_count % 100 == 0
                and not self.config.get('system', {}).get('disable_periodic_compaction', False)):
            keep_mask = self.gaussian_model.compact()
            self.importance_estimator.prune_buffers(keep_mask)
            if self.optimizer is not None:
                # Prune optimizer state to match compacted parameters
                self.optimizer.prune_state(keep_mask)
                # Re-register new parameter objects (compact() creates new nn.Parameter)
                self._reregister_optimizer_params()
            else:
                self._setup_optimizer()

        self._trace_stage('post_prune_compact', {
            'xyz': self.gaussian_model._xyz,
            'scaling': self.gaussian_model._scaling,
            'rotation': self.gaussian_model._rotation,
            'opacity': self.gaussian_model._opacity,
            'features_dc': self.gaussian_model._features_dc,
            'features_rest': self.gaussian_model._features_rest,
            'state': self.gaussian_model._state,
            'persistent_ids': self.gaussian_model.state_store.persistent_ids,
        })

        # Evaluate the final map after optimization, spawn/prune, and compaction.
        # The pre-action render above is retained only for pre-update diagnostics.
        with torch.no_grad():
            cov3D_post = self.gaussian_model.build_covariance()
            if self.gaussian_model.sh_degree > 0:
                from .rasterizer import _view_directions
                post_dirs = _view_directions(self.gaussian_model.positions, self.current_pose)
            else:
                post_dirs = None
            renderer_backend = self.config.get('rendering', {}).get(
                'backend', 'gsplat' if str(self.device).startswith('cuda') else 'reference')
            post_render = rasterize_scene(
                means3D=self.gaussian_model.positions,
                cov3D=cov3D_post,
                colors=self.gaussian_model.get_colors(post_dirs),
                opacities=self.gaussian_model.opacities.squeeze(-1),
                extrinsics=self.current_pose,
                intrinsics=self.intrinsics,
                image_width=W,
                image_height=H,
                tile_size=self.config['rendering']['tile_size'],
                backend=renderer_backend,
            )
            rendered_color_post = post_render['color']
            if self.config['rendering'].get('use_surface_aware_depth', False):
                depth_result_post = render_depth_surface_aware(
                    means3D=self.gaussian_model.positions,
                    normals=self.gaussian_model._normals,
                    opacities=self.gaussian_model.opacities.squeeze(-1),
                    cov3D=cov3D_post,
                    extrinsics=self.current_pose,
                    intrinsics=self.intrinsics,
                    image_width=W,
                    image_height=H,
                    opacity_threshold=self.config['rendering'].get('depth_threshold_opaque', 0.5),
                    tile_size=self.config['rendering']['tile_size'],
                )
                rendered_depth_post = depth_result_post['depth']
                rendered_depth_post_support = depth_result_post['hit_mask']
            else:
                rendered_depth_post = post_render['depth']
                rendered_depth_post_support = post_render.get(
                    'alpha', 1.0 - post_render['transmission']) >= alpha_threshold
            rendered_depth_post = torch.where(
                rendered_depth_post_support,
                rendered_depth_post,
                torch.zeros_like(rendered_depth_post),
            )
        
        # === Collect Metrics & Feedback ===
        frame_time = time.time() - frame_start
        
        # Closed-loop profiling feedback to scheduler
        self.scheduler.adjust_budget_from_profiling(
            actual_frame_ms=frame_time * 1000.0,
            actual_opt_ms=opt_time * 1000.0,
            n_optimized=n_optimized,
            actual_cache_ms=cache_time * 1000.0,
            n_cache_rebuilds=n_cache_rebuilds,
        )
        
        # Compute quality metrics (pre and post optimization). Canonical RGB
        # SSIM is spatial-window SSIM; the historic global-statistics score is
        # retained under ssim_legacy for artifact compatibility audits.
        with torch.no_grad():
            rgb_pre_metrics = rgb_metric_bundle(rendered_color, rgb, valid_gt_mask=depth_valid)
            rgb_post_metrics = rgb_metric_bundle(rendered_color_post, rgb, valid_gt_mask=depth_valid)
            psnr_pre = rgb_pre_metrics['psnr_valid_gt']
            psnr_post = rgb_post_metrics['psnr_valid_gt']
            psnr = psnr_post
            ssim = rgb_post_metrics['ssim_standard']
            ssim_legacy = rgb_post_metrics['ssim_legacy']
            depth_post_metrics = depth_metric_bundle(rendered_depth_post, depth,
                                                     valid_gt_mask=depth_valid,
                                                     pred_valid_mask=rendered_depth_post_support)
            depth_l1_pre = float(depth_err[depth_valid].mean().item()) if depth_valid.any() else float('nan')
            depth_l1_post = depth_post_metrics['depth_l1_gt_valid']
            depth_l1 = depth_l1_post
            
            lpips_val = 0.0
            if self._lpips_metric is not None:
                img_pred = rendered_color_post.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
                img_gt = rgb.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
                lpips_val = float(self._lpips_metric(img_pred, img_gt).item())
        
        budget_ms = float(self.config['scheduler'].get('gpu_budget_ms', 16.6))
        cuda_timing_available = bool(sync_stage_timing and str(self.device).startswith('cuda')
                                    and torch.cuda.is_available())
        time_opt_cuda_ms = opt_stage_cuda_event_ms if cuda_timing_available else None
        time_opt_stage_wall_ms = opt_stage_sync_ms if cuda_timing_available else None
        frame_deadline_ms = self.config.get('system', {}).get('frame_deadline_ms')
        violation_opt = (time_opt_cuda_ms > budget_ms) if (
            time_opt_cuda_ms is not None and budget_ms > 0) else None
        violation_frame = (frame_time * 1000.0 > float(frame_deadline_ms)) if (
            frame_deadline_ms is not None and float(frame_deadline_ms) > 0) else None
        overshoot_opt_ms = (max(0.0, time_opt_cuda_ms - budget_ms)
                            if time_opt_cuda_ms is not None and budget_ms > 0 else None)
        gain_train_psnr_db = float(psnr_post - psnr_pre) if (
            psnr_post is not None and psnr_pre is not None) else None
        legacy_frame_budget_violated = (frame_time * 1000.0) > budget_ms if budget_ms > 0 else False
        legacy_frame_overshoot_ms = max(0.0, (frame_time * 1000.0) - budget_ms) if budget_ms > 0 else 0.0

        metrics = {
            'frame': self.frame_count,
            'psnr': psnr,
            'psnr_valid_gt': psnr_post,
            'psnr_pre': psnr_pre,
            'psnr_post': psnr_post,
            'ssim': ssim,
            'ssim_standard': ssim,
            'ssim_standard_valid_gt': rgb_post_metrics['ssim_standard_valid_gt'],
            'ssim_legacy': ssim_legacy,
            'lpips': lpips_val,
            'depth_l1': depth_l1,
            'depth_l1_pre': depth_l1_pre,
            'depth_l1_post': depth_l1_post,
            'depth_l1_gt_valid': depth_post_metrics['depth_l1_gt_valid'],
            'depth_rmse_gt_valid': depth_post_metrics['depth_rmse_gt_valid'],
            'depth_l1_overlap': depth_post_metrics['depth_l1_overlap'],
            'depth_rmse_overlap': depth_post_metrics['depth_rmse_overlap'],
            'pred_depth_coverage': depth_post_metrics['pred_depth_coverage'],
            'valid_gt_pixel_count': depth_post_metrics['valid_gt_pixel_count'],
            'overlap_pixel_count': depth_post_metrics['overlap_pixel_count'],
            'color_loss': per_gaussian_color_err.mean().item(),
            'n_gaussians': self.gaussian_model.num_gaussians,
            'n_active_gaussians': int((self.gaussian_model._state != GaussianState.PRUNED).sum().item()),
            'n_optimized': n_optimized,
            'n_cache_rebuilds': n_cache_rebuilds,
            'n_cache_frozen_gaussians': int(getattr(self.bg_cache, 'n_frozen', 0)),
            'optimizer_iteration_count': optimizer_iteration_count,
            'n_micro_updates': int(k_alloc.sum().item()) if k_alloc is not None else int(n_optimized * self.config.get('training', {}).get('n_micro_steps', 5)),
            'age_quota_mature_micro_updates': age_quota_mature_updates,
            'age_quota_young_micro_updates': age_quota_young_updates,
            'stale_quota_micro_updates': stale_quota_micro_updates,
            'other_quota_micro_updates': other_quota_micro_updates,
            'randomized_assignment': randomized_assignment_diagnostics,
            'n_spawned': n_spawned,
            'n_spawned_fallback': n_spawned_fallback,
            'spawn_quota_this_frame': spawn_quota_this_frame,
            'spawn_control_total': self._spawn_control_count if spawn_control else None,
            'n_backlog': n_warmup_current if n_warmup_current is not None else 0,
            'selected_young': selected_young_count,
            'selected_old': selected_old_count,
            'population_young': young_population,
            'population_old': old_population,
            'mature_eligible': mature_eligible_count,
            'mature_updated': mature_updated_count,
            'mature_update_coverage': mature_update_coverage,
            'stale_after_frames': stale_after_frames,
            'stale_mature_eligible': stale_mature_eligible_count,
            'stale_mature_updated': stale_mature_updated_count,
            'stale_mature_update_coverage': stale_mature_update_coverage,
            'stale_mature_micro_updates': stale_mature_micro_updates,
            'growth_pressure': float(n_spawned) / max(int((self.gaussian_model._state != GaussianState.PRUNED).sum().item()), 1),
            'compute_return_db_per_opt_ms': (float(psnr_post - psnr_pre) /
                (opt_stage_sync_ms if opt_stage_sync_ms > 0 else opt_time * 1000.0))
                if (opt_stage_sync_ms > 0 or opt_time > 0) else 0.0,
            'compute_return_db_per_opt_ms_legacy_training_frame': (
                float(psnr_post - psnr_pre) /
                (opt_stage_sync_ms if opt_stage_sync_ms > 0 else opt_time * 1000.0)
            ) if (opt_stage_sync_ms > 0 or opt_time > 0) else 0.0,
            'gain_train_psnr_db': gain_train_psnr_db,
            'gain_holdout_psnr_db': None,
            'compute_return_timing_source': ('synchronized_stage_wall_ms' if opt_stage_sync_ms > 0
                else 'pipeline_host_timer_unsynchronized'),
            'budget_young_us': budget_young_us,
            'budget_old_us': budget_old_us,
            'mean_opacity': float(self.gaussian_model.opacities.mean().item()),
            'mean_k': mean_k,
            'peak_vram_mb': float(torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)) if str(self.device).startswith('cuda') and torch.cuda.is_available() else 0.0,
            'n_tier_a': (tiers == Tier.A).sum().item(),
            'n_tier_b': (tiers == Tier.B).sum().item(),
            'n_tier_c': (tiers == Tier.C).sum().item(),
            'n_tier_d': (tiers == Tier.D).sum().item(),
            'frame_time_ms': frame_time * 1000.0,
            'time_frame_wall_ms': frame_time * 1000.0,
            'opt_time_ms': opt_time * 1000.0,
            'opt_stage_sync_ms': opt_stage_sync_ms,
            'opt_stage_cuda_event_ms': opt_stage_cuda_event_ms,
            'time_opt_cuda_ms': time_opt_cuda_ms,
            'time_opt_stage_wall_ms': time_opt_stage_wall_ms,
            'cache_cuda_event_ms': cache_cuda_event_ms,
            'optimizer_forward_cuda_event_ms': optimizer_forward_cuda_event_ms,
            'optimizer_backward_cuda_event_ms': optimizer_backward_cuda_event_ms,
            'optimizer_backward_only_cuda_event_ms': optimizer_backward_only_cuda_event_ms,
            'optimizer_step_cuda_event_ms': optimizer_step_cuda_event_ms,
            'opt_stage_start_unix_ns': sync_stage_start_unix_ns,
            'opt_stage_end_unix_ns': sync_stage_end_unix_ns,
            'cache_time_ms': cache_time * 1000.0,
            'fps': 1.0 / max(frame_time, 1e-8),
            'loss': opt_loss_val,
            # Budget metrics
            'budget_ms': budget_ms,
            'budget_opt_modeled_ms': budget_ms,
            'frame_deadline_ms': frame_deadline_ms,
            'violation_opt': violation_opt,
            'violation_frame': violation_frame,
            'budget_violated': violation_opt if violation_opt is not None else False,
            'overshoot_opt_ms': overshoot_opt_ms,
            'legacy_frame_budget_violated': legacy_frame_budget_violated,
            'legacy_frame_budget_overshoot_ms': legacy_frame_overshoot_ms,
            # Attribution metrics
            'n_visible': visibility_mask.sum().item(),
            'importance_std': importance.std().item() if importance.numel() > 0 else 0.0,
            'importance_min': importance.min().item() if importance.numel() > 0 else 0.0,
            'importance_max': importance.max().item() if importance.numel() > 0 else 0.0,
            'avg_screen_area': per_gaussian_screen_area.mean().item(),
            # Coverage & Warmup metrics
            'coverage': current_coverage,
            'routed_policy': routed_policy if routed_policy is not None else policy,
            'frame_noise': float(frame_noise) if frame_noise is not None else -1.0,
            'routed_kappa': float(routed_kappa) if routed_kappa is not None else -1.0,
            'n_warmup': int(((update_counts < warmup_steps) & visibility_mask[:N_updated]).sum().item()) if enable_warmup else 0,
            'n_warmup_optimized': int((optimize_mask[:N_updated] & (update_counts < warmup_steps) & visibility_mask[:N_updated]).sum().item()) if enable_warmup else 0,
        }
        if age_stratified_diagnostics is not None:
            metrics['age_stratified_diagnostics'] = age_stratified_diagnostics
        if identity_trace is not None:
            metrics['identity_trace'] = identity_trace
        if self._repro_trace_enabled:
            metrics['repro_trace'] = self._repro_trace
        self.metrics_history.append(metrics)
        self.frame_count += 1

        # Periodic VRAM hygiene to prevent fragmentation and desktop lag
        empty_cache_freq = self.config.get('system', {}).get('empty_cache_frequency', 0)
        if empty_cache_freq > 0 and self.frame_count % empty_cache_freq == 0:
            if str(self.device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return metrics

    def eval_frame(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
        trace_persistent_ids: Optional[list] = None,
        return_images: bool = False,
    ) -> Dict[str, Any]:
        """Render-only evaluation at a held-out viewpoint. NO state change.

        Sets no optimizer/StateStore/densification side effects: renders the
        current map at ``pose`` and scores PSNR/SSIM/depth-L1 with the same
        formulas as :meth:`process_frame` (post-optimization branch). Used by
        the novel-view holdout harness (every Kth frame is evaluated but never
        mapped). Restores ``current_pose`` on exit.

        Returns:
            Dict with standard/legacy RGB metrics, GT-valid and overlap depth
            metrics, coverage, and n_gaussians. If ``return_images`` is true,
            the rendered/GT tensors and coverage/error maps are also returned.
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        H, W = depth.shape
        prev_pose = self.current_pose
        self.current_pose = pose.to(self.device)
        try:
            with torch.no_grad():
                cov3D = self.gaussian_model.build_covariance()
                if self.gaussian_model.sh_degree > 0:
                    from .rasterizer import _view_directions
                    dirs = _view_directions(self.gaussian_model.positions, self.current_pose)
                else:
                    dirs = None
                colors = self.gaussian_model.get_colors(dirs)
                backend = self.config.get('rendering', {}).get(
                    'backend', 'gsplat' if str(self.device).startswith('cuda') else 'reference')
                res = rasterize_scene(
                    means3D=self.gaussian_model.positions, cov3D=cov3D, colors=colors,
                    opacities=self.gaussian_model.opacities.squeeze(-1),
                    extrinsics=self.current_pose, intrinsics=self.intrinsics,
                    image_width=W, image_height=H,
                    tile_size=self.config['rendering']['tile_size'], backend=backend)
                rc, rd = res['color'], res['depth']
                valid = torch.isfinite(depth) & (depth > 0)
                rgb_metrics = rgb_metric_bundle(rc, rgb, valid_gt_mask=valid)
                alpha_threshold = float(self.config['rendering'].get('depth_threshold_opaque', 0.5))
                pred_depth_support = res.get('alpha', 1.0 - res['transmission']) >= alpha_threshold
                rd = torch.where(pred_depth_support, rd, torch.zeros_like(rd))
                depth_metrics = depth_metric_bundle(
                    rd, depth, valid_gt_mask=valid,
                    pred_valid_mask=pred_depth_support)
                result = {
                          'psnr': rgb_metrics['psnr_valid_gt'],
                          'psnr_valid_gt': rgb_metrics['psnr_valid_gt'],
                          'ssim': rgb_metrics['ssim_standard'],
                          'ssim_standard': rgb_metrics['ssim_standard'],
                          'ssim_standard_valid_gt': rgb_metrics['ssim_standard_valid_gt'],
                          'ssim_legacy': rgb_metrics['ssim_legacy'],
                          'depth_l1': depth_metrics['depth_l1_gt_valid'],
                          **depth_metrics,
                          'n_gaussians': self.gaussian_model.num_gaussians,
                          'eval_only': True, 'frame': self.frame_count}
                if return_images:
                    pred_valid = pred_depth_support
                    safe_rd = torch.where(pred_valid, rd, torch.zeros_like(rd))
                    result['images'] = {
                        'rgb_prediction': rc.detach().cpu(),
                        'rgb_ground_truth': rgb.detach().cpu(),
                        'depth_prediction': rd.detach().cpu(),
                        'depth_accumulated': res.get(
                            'depth_accumulated', rd).detach().cpu(),
                        'render_alpha': res.get(
                            'alpha', 1.0 - res['transmission']).detach().cpu(),
                        'depth_ground_truth': depth.detach().cpu(),
                        'depth_absolute_error_gt_valid': torch.where(
                            valid, (safe_rd - depth).abs(), torch.full_like(depth, float('nan'))
                        ).detach().cpu(),
                        'pred_depth_coverage_mask': (valid & pred_valid).detach().cpu(),
                        'gt_valid_mask': valid.detach().cpu(),
                    }
                if trace_persistent_ids:
                    stats = compute_fast_gaussian_statistics(
                        means3D=self.gaussian_model.positions,
                        cov3D=cov3D,
                        opacities=self.gaussian_model.opacities.squeeze(-1),
                        rendered_color=rc, rendered_depth=rd,
                        gt_color=rgb, gt_depth=depth,
                        extrinsics=self.current_pose, intrinsics=self.intrinsics)
                    store = self.gaussian_model.state_store
                    trace_ids = torch.as_tensor(trace_persistent_ids,
                                                dtype=torch.long, device=self.device)
                    ids_now = store.persistent_ids[:self.gaussian_model.num_gaussians]
                    order_ids = torch.argsort(ids_now)
                    sorted_ids = ids_now[order_ids]
                    positions = torch.searchsorted(sorted_ids, trace_ids)
                    in_range = positions < sorted_ids.numel()
                    safe_positions = positions.clamp(max=max(sorted_ids.numel() - 1, 0))
                    present = (in_range & (sorted_ids[safe_positions] == trace_ids)
                               if sorted_ids.numel() else torch.zeros_like(in_range))
                    indices = (order_ids[safe_positions] if sorted_ids.numel()
                               else torch.zeros_like(safe_positions))
                    if sorted_ids.numel():
                        present &= self.gaussian_model._state[indices] != GaussianState.PRUNED
                    visible = torch.zeros_like(present)
                    rgb_err = torch.zeros(trace_ids.shape, dtype=stats['color_error'].dtype,
                                          device=self.device)
                    depth_err = torch.zeros_like(rgb_err)
                    if present.any():
                        visible[present] = stats['visibility_mask'][indices[present]]
                        rgb_err[present] = stats['color_error'][indices[present]]
                        depth_err[present] = stats['depth_error'][indices[present]]
                    result['identity_reference_residual'] = {
                        'persistent_ids': trace_ids.detach().cpu().tolist(),
                        'present': present.detach().cpu().tolist(),
                        'visible': visible.detach().cpu().tolist(),
                        'rgb_center_abs_error': rgb_err.detach().cpu().tolist(),
                        'depth_center_abs_error': depth_err.detach().cpu().tolist(),
                        'metric_definition': 'projected Gaussian-center pixel residual on this fixed GT view; not contribution-weighted attribution',
                    }
            return result
        finally:
            self.current_pose = prev_pose

    def cleanup(self):
        """Release cached GPU memory and optimizer states."""
        self.optimizer = None
        self.bg_cache.invalidate()
        if str(self.device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
        measured_opt_violations = [m['violation_opt'] for m in self.metrics_history
                                   if m.get('violation_opt') is not None]
        measured_frame_violations = [m['violation_frame'] for m in self.metrics_history
                                     if m.get('violation_frame') is not None]
        
        latency_stats = self.scheduler.get_latency_statistics()
        
        return {
            'total_frames': len(self.metrics_history),
            'avg_psnr': float(np.mean(psnrs)),
            'avg_depth_l1': float(np.mean(depths)),
            'avg_fps': float(np.mean(fps_list)),
            'final_n_gaussians': self.metrics_history[-1]['n_gaussians'],
            'avg_frame_time_ms': float(np.mean([m['frame_time_ms'] for m in self.metrics_history])),
            'budget_violation_rate': (float(np.mean(measured_opt_violations))
                                      if measured_opt_violations else None),
            'optimization_budget_violation_rate': (
                float(np.mean(measured_opt_violations)) if measured_opt_violations else None),
            'frame_deadline_violation_rate': (
                float(np.mean(measured_frame_violations)) if measured_frame_violations else None),
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
        renderer_backend = self.config.get('rendering', {}).get(
            'backend', 'gsplat' if str(self.device).startswith('cuda') else 'reference')
        
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
                backend=renderer_backend,
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
                trial_cache.build_cache(
                    self.gaussian_model, frozen_mask, pose, self.intrinsics, W, H,
                    backend=renderer_backend,
                )
                
            comp_out = trial_cache.composite_with_active(
                active_subset, pose, self.intrinsics, W, H,
                backend=renderer_backend,
            )
            losses = total_loss(comp_out['color'], rgb, comp_out['depth'], depth, {'color': 1.0, 'depth': 0.5})
            if losses['total'].requires_grad:
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
                backend=renderer_backend,
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

    def get_global_context(self, budget_ms: float = 15.0) -> np.ndarray:
        """Extracts canonical 12D global frame context vector c_t (Phase 12D).

        Schema:
            0: glob_gaussian_count        (total active Gaussians N_G)
            1: glob_visible_count         (visible Gaussians N_vis)
            2: glob_visible_fraction      (N_vis / N_G)
            3: glob_mean_rgb_err          (frame mean color error)
            4: glob_std_rgb_err           (frame std color error)
            5: glob_mean_depth_err        (frame mean depth error)
            6: glob_std_depth_err         (frame std depth error)
            7: glob_mean_grad_norm        (frame mean gradient norm)
            8: glob_std_grad_norm         (frame std gradient norm)
            9: glob_mean_influence        (frame mean influence mass)
            10: glob_selected_fraction    (mean historical update frequency)
            11: glob_normalized_frame_idx (normalized frame index t / 60.0)

        Returns:
            c_t: [12] float32 numpy array. Guaranteed finite.
        """
        N = self.gaussian_model.num_gaussians
        if N == 0:
            return np.zeros(12, dtype=np.float32)

        est = self.importance_estimator
        store = getattr(self.gaussian_model, "state_store", None)

        color_err = est._running_color_error[:N] if est._running_color_error is not None else None
        depth_err = est._running_depth_error[:N] if est._running_depth_error is not None else None
        vis_count = est._visibility_count[:N] if est._visibility_count is not None else None
        inf_mass = getattr(est, "_influence_weights", None)
        inf_t = inf_mass[:N] if inf_mass is not None and inf_mass.shape[0] >= N else None

        n_vis = float(torch.sum(vis_count > 0).item()) if vis_count is not None else float(N)
        vis_frac = n_vis / max(float(N), 1.0)

        mean_rgb = float(torch.mean(color_err).item()) if color_err is not None and len(color_err) > 0 else 0.0
        std_rgb = float(torch.std(color_err).item()) if color_err is not None and len(color_err) > 0 else 0.0

        mean_depth = float(torch.mean(depth_err).item()) if depth_err is not None and len(depth_err) > 0 else 0.0
        std_depth = float(torch.std(depth_err).item()) if depth_err is not None and len(depth_err) > 0 else 0.0

        if inf_t is not None and color_err is not None and depth_err is not None:
            grad_norm = inf_t * (color_err + depth_err)
            mean_grad = float(torch.mean(grad_norm).item())
            std_grad = float(torch.std(grad_norm).item())
        else:
            mean_grad = 0.0
            std_grad = 0.0

        mean_inf = float(torch.mean(inf_t).item()) if inf_t is not None and len(inf_t) > 0 else 1.0

        if store is not None and store.num_gaussians >= N:
            mean_update_freq = float(torch.mean(store.get_update_frequency(self.frame_count)[:N]).item())
        else:
            mean_update_freq = 0.0

        norm_frame = float(self.frame_count) / 60.0

        ctx = np.array([
            float(N),
            float(n_vis),
            float(vis_frac),
            float(mean_rgb),
            float(std_rgb),
            float(mean_depth),
            float(std_depth),
            float(mean_grad),
            float(std_grad),
            float(mean_inf),
            float(mean_update_freq),
            float(norm_frame),
        ], dtype=np.float32)

        return np.nan_to_num(ctx, nan=0.0, posinf=1.0, neginf=0.0)
