"""
Persistent Gaussian State Store (Points III & IV).

Manages persistent Gaussian identity and lifecycle state across:
    create -> optimize -> densify -> prune -> reorder

Guarantees that state signals (EMA errors, influence, age, temporal drift,
gradients, tiers) track unique Gaussian identities (persistent_id),
eliminating tensor index aliasing after pruning, compaction, and reordering.
"""
import copy
import torch
from typing import Dict, List, Optional, Tuple, Any, Union


class GaussianStateStore:
    """Identity-preserving persistent state store for 3D Gaussians."""

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self._next_id: int = 0
        self._current_frame: int = 0
        
        # Core vectorized tensors for the active N Gaussians (strictly synchronized with GaussianModel parameters)
        self.persistent_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.parent_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.creation_frames = torch.empty(0, dtype=torch.long, device=self.device)
        self.last_update_frames = torch.empty(0, dtype=torch.long, device=self.device)
        self.ages = torch.empty(0, dtype=torch.long, device=self.device)
        
        # Per-Gaussian state signals
        self.ema_rgb = torch.empty(0, dtype=torch.float32, device=self.device)
        self.ema_depth = torch.empty(0, dtype=torch.float32, device=self.device)
        self.ema_influence = torch.empty(0, dtype=torch.float32, device=self.device)
        self.visibility_count = torch.empty(0, dtype=torch.float32, device=self.device)
        self.ema_visibility = torch.empty(0, dtype=torch.float32, device=self.device)
        self.uncertainty = torch.empty(0, dtype=torch.float32, device=self.device)
        self.position_drift = torch.empty(0, dtype=torch.float32, device=self.device)
        self.residual_drift_ema = torch.empty(0, dtype=torch.float32, device=self.device)
        self.gradient_ema = torch.empty(0, dtype=torch.float32, device=self.device)
        self.tiers = torch.empty(0, dtype=torch.long, device=self.device)
        
        # Tracking buffers for differential quantities
        self.last_positions: Optional[torch.Tensor] = None
        self.last_rgb_errors: Optional[torch.Tensor] = None
        
        # Historical registry for lineage and lifecycle queries
        self._id_to_metadata: Dict[int, Dict[str, Any]] = {}
        self._pruned_registry: Dict[int, Dict[str, Any]] = {}

    @property
    def num_gaussians(self) -> int:
        return self.persistent_ids.shape[0]

    @property
    def uncertainty_var(self) -> torch.Tensor:
        """Protocol-aligned alias for uncertainty."""
        return self.uncertainty

    @uncertainty_var.setter
    def uncertainty_var(self, value: torch.Tensor) -> None:
        self.uncertainty = value

    @property
    def temporal_drift(self) -> torch.Tensor:
        """Backward-compatibility alias for position_drift (Point B)."""
        return self.position_drift

    @temporal_drift.setter
    def temporal_drift(self, value: torch.Tensor) -> None:
        self.position_drift = value

    def create(
        self,
        count: int,
        frame_idx: int = 0,
        parent_ids: Optional[torch.Tensor] = None,
        initial_uncertainty: float = 0.5,
        initial_tier: int = 1,
    ) -> torch.Tensor:
        """Create new Gaussians with unique persistent IDs.
        
        Args:
            count: number of new Gaussians to register
            frame_idx: video frame index of creation
            parent_ids: optional (count,) tensor of parent IDs (for split/clone)
            initial_uncertainty: prior uncertainty value
            initial_tier: initial priority tier (0=A, 1=B, 2=C, 3=D)
            
        Returns:
            new_ids: (count,) tensor of unique persistent IDs
        """
        if count <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
            
        new_ids_list = list(range(self._next_id, self._next_id + count))
        self._next_id += count
        new_ids = torch.tensor(new_ids_list, dtype=torch.long, device=self.device)
        
        if parent_ids is None:
            parents = torch.full((count,), -1, dtype=torch.long, device=self.device)
        else:
            parents = parent_ids.to(self.device)
            
        creation_t = torch.full((count,), frame_idx, dtype=torch.long, device=self.device)
        last_up_t = torch.full((count,), frame_idx, dtype=torch.long, device=self.device)
        init_ages = torch.zeros(count, dtype=torch.long, device=self.device)
        
        zeros = torch.zeros(count, dtype=torch.float32, device=self.device)
        uncert = torch.full((count,), initial_uncertainty, dtype=torch.float32, device=self.device)
        tier_t = torch.full((count,), initial_tier, dtype=torch.long, device=self.device)
        
        # Append to active state tensors
        self.persistent_ids = torch.cat([self.persistent_ids, new_ids], dim=0)
        self.parent_ids = torch.cat([self.parent_ids, parents], dim=0)
        self.creation_frames = torch.cat([self.creation_frames, creation_t], dim=0)
        self.last_update_frames = torch.cat([self.last_update_frames, last_up_t], dim=0)
        self.ages = torch.cat([self.ages, init_ages], dim=0)
        
        self.ema_rgb = torch.cat([self.ema_rgb, zeros], dim=0)
        self.ema_depth = torch.cat([self.ema_depth, zeros], dim=0)
        self.ema_influence = torch.cat([self.ema_influence, zeros], dim=0)
        self.visibility_count = torch.cat([self.visibility_count, zeros], dim=0)
        self.ema_visibility = torch.cat([self.ema_visibility, zeros], dim=0)
        self.uncertainty = torch.cat([self.uncertainty, uncert], dim=0)
        self.position_drift = torch.cat([self.position_drift, zeros], dim=0)
        self.residual_drift_ema = torch.cat([self.residual_drift_ema, zeros], dim=0)
        self.gradient_ema = torch.cat([self.gradient_ema, zeros], dim=0)
        self.tiers = torch.cat([self.tiers, tier_t], dim=0)
        
        if self.last_positions is not None:
            self.last_positions = torch.cat([self.last_positions, torch.zeros(count, 3, device=self.device)], dim=0)
        if self.last_rgb_errors is not None:
            self.last_rgb_errors = torch.cat([self.last_rgb_errors, zeros], dim=0)
            
        # Log metadata
        for idx, g_id in enumerate(new_ids_list):
            p_id = int(parents[idx].item())
            self._id_to_metadata[g_id] = {
                'persistent_id': g_id,
                'parent_id': p_id,
                'creation_frame': frame_idx,
                'children': [],
            }
            if p_id in self._id_to_metadata:
                self._id_to_metadata[p_id]['children'].append(g_id)
                
        return new_ids

    def get_staleness(self, frame_idx: Optional[int] = None) -> torch.Tensor:
        """Compute staleness: frame_idx - last_update_frames (clamped >= 0)."""
        t = self._current_frame if frame_idx is None else frame_idx
        return (t - self.last_update_frames).clamp(min=0)

    def update_frame(
        self,
        frame_idx: int,
        rgb_errors: Optional[torch.Tensor] = None,
        depth_errors: Optional[torch.Tensor] = None,
        influence_scores: Optional[torch.Tensor] = None,
        visibility_count: Optional[torch.Tensor] = None,
        visibility_mask: Optional[torch.Tensor] = None,
        gradient_norms: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        optimized_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        ema_decay: float = 0.9,
    ) -> None:
        """Update active Gaussian state signals for the current frame using identity-stable EMA."""
        self._current_frame = frame_idx
        N = self.num_gaussians
        if N == 0:
            return
            
        # Age follows exact semantic: age = frame_idx - creation_frame (clamped >= 0)
        self.ages = (frame_idx - self.creation_frames).clamp(min=0)
        
        # Track last optimization update frame
        if optimized_mask is not None:
            opt = optimized_mask[:N].to(self.device)
            self.last_update_frames[opt] = frame_idx
            
        # Position drift: d_i(t) = ||mu_i(t) - mu_i(t-1)||_2 (Point B)
        if positions is not None:
            pos_t = positions[:N].to(self.device)
            if self.last_positions is not None and self.last_positions.shape == pos_t.shape:
                pos_diff = pos_t - self.last_positions
                self.position_drift = torch.norm(pos_diff, p=2, dim=-1)
            else:
                self.position_drift = torch.zeros(N, dtype=torch.float32, device=self.device)
            self.last_positions = pos_t.clone()
            
        # EMA error updates & residual drift
        if rgb_errors is not None:
            rgb_e = rgb_errors[:N].to(self.device)
            # Residual drift: r_i(t) = |e_i(t) - e_i(t-1)|
            if self.last_rgb_errors is not None and self.last_rgb_errors.shape == rgb_e.shape:
                r_t = torch.abs(rgb_e - self.last_rgb_errors)
                self.residual_drift_ema = ema_decay * self.residual_drift_ema + (1.0 - ema_decay) * r_t
            else:
                self.residual_drift_ema = torch.zeros(N, dtype=torch.float32, device=self.device)
            self.last_rgb_errors = rgb_e.clone()
            self.ema_rgb = ema_decay * self.ema_rgb + (1.0 - ema_decay) * rgb_e
            
        if depth_errors is not None:
            depth_e = depth_errors[:N].to(self.device)
            self.ema_depth = ema_decay * self.ema_depth + (1.0 - ema_decay) * depth_e
            
        if influence_scores is not None:
            inf = influence_scores[:N].to(self.device)
            self.ema_influence = ema_decay * self.ema_influence + (1.0 - ema_decay) * inf
            
        # Visibility count and EMA (Point A)
        if visibility_count is not None:
            vis = visibility_count[:N].to(self.device).float()
            self.visibility_count = vis
            self.ema_visibility = ema_decay * self.ema_visibility + (1.0 - ema_decay) * vis
        elif visibility_mask is not None:
            vis = visibility_mask[:N].to(self.device).float()
            self.visibility_count = vis
            self.ema_visibility = ema_decay * self.ema_visibility + (1.0 - ema_decay) * vis
            
        if gradient_norms is not None:
            gn = gradient_norms[:N].to(self.device)
            self.gradient_ema = ema_decay * self.gradient_ema + (1.0 - ema_decay) * gn
            
        if uncertainty is not None:
            self.uncertainty = uncertainty[:N].to(self.device)

    def reorder(self, permutation: torch.Tensor) -> None:
        """Permute active state tensors according to permutation indices.
        
        Guarantees Invariant B: State tracks persistent ID, NOT tensor index.
        """
        N = self.num_gaussians
        if N == 0:
            return
        perm = permutation.to(self.device)
        if len(perm) != N:
            raise ValueError(f"Permutation length ({len(perm)}) must match num_gaussians ({N})")
            
        self.persistent_ids = self.persistent_ids[perm]
        self.parent_ids = self.parent_ids[perm]
        self.creation_frames = self.creation_frames[perm]
        self.last_update_frames = self.last_update_frames[perm]
        self.ages = self.ages[perm]
        
        self.ema_rgb = self.ema_rgb[perm]
        self.ema_depth = self.ema_depth[perm]
        self.ema_influence = self.ema_influence[perm]
        self.visibility_count = self.visibility_count[perm]
        self.ema_visibility = self.ema_visibility[perm]
        self.uncertainty = self.uncertainty[perm]
        self.position_drift = self.position_drift[perm]
        self.residual_drift_ema = self.residual_drift_ema[perm]
        self.gradient_ema = self.gradient_ema[perm]
        self.tiers = self.tiers[perm]
        
        if self.last_positions is not None:
            self.last_positions = self.last_positions[perm]
        if self.last_rgb_errors is not None:
            self.last_rgb_errors = self.last_rgb_errors[perm]

    def remap_after_pruning(self, keep_mask: torch.Tensor) -> torch.Tensor:
        """Remap state tensors after pruning/compact.
        
        Args:
            keep_mask: (N,) boolean tensor where True = kept, False = pruned
            
        Returns:
            pruned_ids: (P,) tensor of IDs that were pruned
        """
        N = self.num_gaussians
        mask = keep_mask[:N].to(self.device)
        
        pruned_mask = ~mask
        pruned_ids = self.persistent_ids[pruned_mask]
        
        # Record pruned Gaussians in registry with historical metadata
        for p_id in pruned_ids.cpu().tolist():
            if p_id in self._id_to_metadata:
                record = self._id_to_metadata.pop(p_id)
                record['pruned_frame'] = self._current_frame
                self._pruned_registry[p_id] = record
                
        # Squeeze active tensors
        self.persistent_ids = self.persistent_ids[mask]
        self.parent_ids = self.parent_ids[mask]
        self.creation_frames = self.creation_frames[mask]
        self.last_update_frames = self.last_update_frames[mask]
        self.ages = self.ages[mask]
        
        self.ema_rgb = self.ema_rgb[mask]
        self.ema_depth = self.ema_depth[mask]
        self.ema_influence = self.ema_influence[mask]
        self.visibility_count = self.visibility_count[mask]
        self.ema_visibility = self.ema_visibility[mask]
        self.uncertainty = self.uncertainty[mask]
        self.position_drift = self.position_drift[mask]
        self.residual_drift_ema = self.residual_drift_ema[mask]
        self.gradient_ema = self.gradient_ema[mask]
        self.tiers = self.tiers[mask]
        
        if self.last_positions is not None:
            self.last_positions = self.last_positions[mask]
        if self.last_rgb_errors is not None:
            self.last_rgb_errors = self.last_rgb_errors[mask]
            
        return pruned_ids

    def register_densification(
        self,
        parent_indices: torch.Tensor,
        n_children_per_parent: int = 1,
        frame_idx: int = 0,
        policy: str = 'fresh',
    ) -> torch.Tensor:
        """Register child Gaussians generated by densification (clone/split).
        
        Maintains parent-child lineage (e.g. parent #105 -> child #82731).
        
        Args:
            parent_indices: (P,) indices in current active tensor of parent Gaussians
            n_children_per_parent: number of children spawned per parent (e.g. 1 for clone, 2 for split)
            frame_idx: current video frame index
            policy: 'fresh' (clean initial state) or 'inherit' (copy parent EMA)
            
        Returns:
            child_ids: (P * n_children,) tensor of newly assigned IDs
        """
        N = self.num_gaussians
        p_idx = parent_indices[parent_indices < N].to(self.device)
        p_ids = self.persistent_ids[p_idx]
        
        # Expand parent IDs for each child
        repeated_parent_ids = p_ids.repeat_interleave(n_children_per_parent)
        n_new = repeated_parent_ids.shape[0]
        
        child_ids = self.create(
            count=n_new,
            frame_idx=frame_idx,
            parent_ids=repeated_parent_ids,
            initial_tier=1,
        )
        
        if policy == 'inherit' and n_new > 0:
            # Child inherits parent's historical EMA
            parent_ema_rgb = self.ema_rgb[p_idx].repeat_interleave(n_children_per_parent)
            parent_ema_depth = self.ema_depth[p_idx].repeat_interleave(n_children_per_parent)
            parent_ema_inf = self.ema_influence[p_idx].repeat_interleave(n_children_per_parent)
            parent_ema_vis = self.ema_visibility[p_idx].repeat_interleave(n_children_per_parent)
            parent_vis_cnt = self.visibility_count[p_idx].repeat_interleave(n_children_per_parent)
            
            child_slice = slice(N, N + n_new)
            self.ema_rgb[child_slice] = parent_ema_rgb
            self.ema_depth[child_slice] = parent_ema_depth
            self.ema_influence[child_slice] = parent_ema_inf
            self.ema_visibility[child_slice] = parent_ema_vis
            self.visibility_count[child_slice] = parent_vis_cnt
            
        return child_ids

    def get_state_matrix(self, indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Extract state feature dictionary aligned with the requested Gaussian indices."""
        if indices is None:
            idx = slice(None)
        else:
            idx = indices.to(self.device)
            
        return {
            'persistent_id': self.persistent_ids[idx],
            'parent_id': self.parent_ids[idx],
            'age': self.ages[idx],
            'creation_frame': self.creation_frames[idx],
            'last_update_frame': self.last_update_frames[idx],
            'staleness': self.get_staleness()[idx],
            'ema_rgb': self.ema_rgb[idx],
            'ema_depth': self.ema_depth[idx],
            'ema_influence': self.ema_influence[idx],
            'visibility_count': self.visibility_count[idx],
            'ema_visibility': self.ema_visibility[idx],
            'uncertainty': self.uncertainty[idx],
            'uncertainty_var': self.uncertainty[idx],
            'position_drift': self.position_drift[idx],
            'residual_drift_ema': self.residual_drift_ema[idx],
            'temporal_drift': self.position_drift[idx],  # legacy alias
            'gradient_ema': self.gradient_ema[idx],
            'tier': self.tiers[idx],
        }

    def get_lineage(self, persistent_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve lineage record for an active or pruned Gaussian."""
        if persistent_id in self._id_to_metadata:
            return dict(self._id_to_metadata[persistent_id], status='active')
        if persistent_id in self._pruned_registry:
            return dict(self._pruned_registry[persistent_id], status='pruned')
        return None

    def state_dict(self) -> Dict[str, Any]:
        """Export complete persistent state for snapshotting/checkpointing."""
        return {
            '_next_id': self._next_id,
            '_current_frame': self._current_frame,
            'persistent_ids': self.persistent_ids.clone(),
            'parent_ids': self.parent_ids.clone(),
            'creation_frames': self.creation_frames.clone(),
            'last_update_frames': self.last_update_frames.clone(),
            'ages': self.ages.clone(),
            'ema_rgb': self.ema_rgb.clone(),
            'ema_depth': self.ema_depth.clone(),
            'ema_influence': self.ema_influence.clone(),
            'visibility_count': self.visibility_count.clone(),
            'ema_visibility': self.ema_visibility.clone(),
            'uncertainty': self.uncertainty.clone(),
            'position_drift': self.position_drift.clone(),
            'temporal_drift': self.position_drift.clone(),  # legacy alias
            'residual_drift_ema': self.residual_drift_ema.clone(),
            'gradient_ema': self.gradient_ema.clone(),
            'tiers': self.tiers.clone(),
            'last_positions': self.last_positions.clone() if self.last_positions is not None else None,
            'last_rgb_errors': self.last_rgb_errors.clone() if self.last_rgb_errors is not None else None,
            '_id_to_metadata': copy.deepcopy(self._id_to_metadata),
            '_pruned_registry': copy.deepcopy(self._pruned_registry),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore complete persistent state from snapshot."""
        self._next_id = int(state['_next_id'])
        self._current_frame = int(state['_current_frame'])
        self.persistent_ids = state['persistent_ids'].clone().to(self.device)
        self.parent_ids = state['parent_ids'].clone().to(self.device)
        self.creation_frames = state['creation_frames'].clone().to(self.device)
        self.last_update_frames = state['last_update_frames'].clone().to(self.device)
        self.ages = state['ages'].clone().to(self.device)
        self.ema_rgb = state['ema_rgb'].clone().to(self.device)
        self.ema_depth = state['ema_depth'].clone().to(self.device)
        self.ema_influence = state['ema_influence'].clone().to(self.device)
        self.visibility_count = state.get('visibility_count', state.get('ema_visibility')).clone().to(self.device)
        self.ema_visibility = state['ema_visibility'].clone().to(self.device)
        self.uncertainty = state['uncertainty'].clone().to(self.device)
        self.position_drift = state.get('position_drift', state.get('temporal_drift', torch.zeros_like(self.ema_rgb))).clone().to(self.device)
        self.residual_drift_ema = state.get('residual_drift_ema', torch.zeros_like(self.ema_rgb)).clone().to(self.device)
        self.gradient_ema = state['gradient_ema'].clone().to(self.device)
        self.tiers = state['tiers'].clone().to(self.device)
        self.last_positions = state['last_positions'].clone().to(self.device) if state.get('last_positions') is not None else None
        self.last_rgb_errors = state['last_rgb_errors'].clone().to(self.device) if state.get('last_rgb_errors') is not None else None
        self._id_to_metadata = copy.deepcopy(state['_id_to_metadata'])
        self._pruned_registry = copy.deepcopy(state['_pruned_registry'])

