"""Continuous Gaussian Importance / Confidence Scoring.

Research novelty: Replace RTG-SLAM's binary stable/unstable classification
with a continuous importance score that drives adaptive optimization.

Importance formula:
    I_i = w_g · E_{depth,i} + w_p · E_{color,i} + w_n · E_{normal,i}
          + w_v · V_i + w_t · ΔT_i + w_s · S_i

where:
    E_{depth,i} = per-Gaussian average depth error from recent frames
    E_{color,i} = per-Gaussian average color error from recent frames  
    E_{normal,i} = per-Gaussian normal consistency error
    V_i = visibility frequency (how often this Gaussian is rendered)
    ΔT_i = temporal change magnitude
    S_i = screen-space importance (area × gradient)

Tier classification:
    Tier A (I > τ_high): Optimize every frame
    Tier B (τ_low ≤ I ≤ τ_high): Optimize every N frames
    Tier C (I < τ_low): Freeze (forward render only)
    Tier D (zero contribution long-term): Prune candidate
"""
import torch
from typing import Dict, Tuple, Optional
from enum import IntEnum


class Tier(IntEnum):
    """Optimization tier classification."""
    A = 0  # High importance - optimize every frame
    B = 1  # Medium importance - optimize every N frames
    C = 2  # Low importance - frozen, render only
    D = 3  # Outlier / zero contribution - prune candidate


class GaussianImportanceEstimator:
    """Estimates per-Gaussian importance scores for adaptive optimization.
    
    Maintains running statistics of per-Gaussian errors and visibility,
    and computes a continuous importance score used by the BudgetScheduler.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        tau_high: float = 0.8,
        tau_low: float = 0.2,
        tau_prune: float = 0.05,
        ema_decay: float = 0.95,
        prune_patience: int = 50,
    ):
        """Initialize importance estimator.
        
        Args:
            weights: importance component weights {depth_error, color_error,
                     normal_error, visibility, temporal, screen_space}
            tau_high: threshold for Tier A
            tau_low: threshold for Tier C
            tau_prune: threshold for Tier D (prune candidates)
            ema_decay: exponential moving average decay for running stats
            prune_patience: frames of zero contribution before prune
        """
        self.weights = weights or {
            'depth_error': 1.0,
            'color_error': 1.0,
            'normal_error': 0.5,
            'visibility': 0.1,
            'temporal': 0.5,
            'screen_space': 0.2,
        }
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.tau_prune = tau_prune
        self.ema_decay = ema_decay
        self.prune_patience = prune_patience
        
        # Running statistics (initialized on first call)
        self._running_depth_error: Optional[torch.Tensor] = None
        self._running_color_error: Optional[torch.Tensor] = None
        self._running_normal_error: Optional[torch.Tensor] = None
        self._visibility_count: Optional[torch.Tensor] = None
        self._prev_positions: Optional[torch.Tensor] = None
        self._zero_contrib_frames: Optional[torch.Tensor] = None
        self._frame_count = 0
    
    def _ensure_buffers(self, N: int, device: torch.device):
        """Lazily initialize running buffers when Gaussian count is known."""
        if self._running_depth_error is None or self._running_depth_error.shape[0] != N:
            self._running_depth_error = torch.zeros(N, device=device)
            self._running_color_error = torch.zeros(N, device=device)
            self._running_normal_error = torch.zeros(N, device=device)
            self._visibility_count = torch.zeros(N, device=device)
            self._prev_positions = None
            self._zero_contrib_frames = torch.zeros(N, dtype=torch.long, device=device)
    
    def update_statistics(
        self,
        depth_errors: torch.Tensor,
        color_errors: torch.Tensor,
        normal_errors: Optional[torch.Tensor],
        visibility_mask: torch.Tensor,
        positions: torch.Tensor,
        screen_areas: Optional[torch.Tensor] = None,
    ):
        """Update running statistics with current frame observations.
        
        Uses exponential moving average to smooth statistics over time.
        
        Args:
            depth_errors: (N,) per-Gaussian depth error this frame
            color_errors: (N,) per-Gaussian color error this frame
            normal_errors: (N,) per-Gaussian normal error (optional)
            visibility_mask: (N,) boolean - True if Gaussian was visible
            positions: (N, 3) current Gaussian positions
            screen_areas: (N,) screen-space area of each Gaussian (optional)
        """
        N = depth_errors.shape[0]
        device = depth_errors.device
        self._ensure_buffers(N, device)
        
        decay = self.ema_decay
        
        # EMA update for errors
        self._running_depth_error = (
            decay * self._running_depth_error + (1 - decay) * depth_errors
        )
        self._running_color_error = (
            decay * self._running_color_error + (1 - decay) * color_errors
        )
        if normal_errors is not None:
            self._running_normal_error = (
                decay * self._running_normal_error + (1 - decay) * normal_errors
            )
        
        # Visibility counting
        self._visibility_count = (
            decay * self._visibility_count + (1 - decay) * visibility_mask.float()
        )
        
        # Zero-contribution tracking for pruning
        no_contrib = ~visibility_mask | (depth_errors + color_errors < 1e-6)
        self._zero_contrib_frames[no_contrib] += 1
        self._zero_contrib_frames[~no_contrib] = 0
        
        self._screen_areas = screen_areas
        self._positions = positions.detach().clone()
        self._frame_count += 1
    
    def compute_importance(self) -> torch.Tensor:
        """Compute continuous importance score for all Gaussians.
        
        I_i = w_g·E_{depth,i} + w_p·E_{color,i} + w_n·E_{normal,i}
              + w_v·V_i + w_t·ΔT_i + w_s·S_i
        
        Returns:
            importance: (N,) importance scores in [0, 1] (normalized)
        """
        if self._running_depth_error is None:
            raise RuntimeError("Must call update_statistics before compute_importance")
        
        w = self.weights
        
        # Component scores
        score = torch.zeros_like(self._running_depth_error)
        
        score += w['depth_error'] * self._running_depth_error
        score += w['color_error'] * self._running_color_error
        score += w['normal_error'] * self._running_normal_error
        score += w['visibility'] * self._visibility_count
        
        # Temporal change: ||μ_t - μ_{t-1}||₂
        if self._prev_positions is not None and self._positions is not None:
            N_score = score.shape[0]
            temporal_change = torch.zeros(N_score, device=score.device)
            min_len = min(self._prev_positions.shape[0], self._positions.shape[0], N_score)
            temporal_change[:min_len] = (self._positions[:min_len] - self._prev_positions[:min_len]).norm(dim=-1)
            score += w['temporal'] * temporal_change
        
        # Screen-space importance
        if self._screen_areas is not None:
            N_score = score.shape[0]
            if self._screen_areas.shape[0] == N_score:
                score += w['screen_space'] * self._screen_areas
            else:
                sa_len = min(self._screen_areas.shape[0], N_score)
                score[:sa_len] += w['screen_space'] * self._screen_areas[:sa_len]
        
        # Normalize to [0, 1]
        score_min = score.min()
        score_max = score.max()
        if score_max - score_min > 1e-8:
            score = (score - score_min) / (score_max - score_min)
        else:
            score = torch.zeros_like(score)
        
        # Update prev positions for next frame
        if self._positions is not None:
            self._prev_positions = self._positions.clone()
        
        return score
    
    def classify_tier(self, importance: torch.Tensor) -> torch.Tensor:
        """Classify each Gaussian into optimization tiers.
        
        Tier A (I > τ_high): optimize every frame
        Tier B (τ_low ≤ I ≤ τ_high): optimize periodically
        Tier C (I < τ_low): frozen
        Tier D (long-term zero contribution): prune
        
        Args:
            importance: (N,) importance scores
        
        Returns:
            tiers: (N,) integer tier labels (0=A, 1=B, 2=C, 3=D)
        """
        N = importance.shape[0]
        device = importance.device
        tiers = torch.full((N,), Tier.C, dtype=torch.long, device=device)
        
        tiers[importance > self.tau_high] = Tier.A
        tiers[(importance >= self.tau_low) & (importance <= self.tau_high)] = Tier.B
        tiers[importance < self.tau_low] = Tier.C
        
        # Override to Tier D for long-term zero-contribution Gaussians
        if self._zero_contrib_frames is not None:
            tiers[self._zero_contrib_frames > self.prune_patience] = Tier.D
        
        return tiers
    
    def update_confidence(
        self,
        current_confidence: torch.Tensor,
        importance: torch.Tensor,
        learning_rate: float = 0.1,
    ) -> torch.Tensor:
        """Update Gaussian confidence using importance feedback.
        
        confidence_new = (1 - lr) · confidence_old + lr · importance
        
        Args:
            current_confidence: (N, 1) current confidence values
            importance: (N,) new importance scores
            learning_rate: update rate
        
        Returns:
            updated_confidence: (N, 1)
        """
        updated = (
            (1 - learning_rate) * current_confidence.squeeze(-1) 
            + learning_rate * importance
        )
        return updated.unsqueeze(-1).clamp(0, 1)
    
    @torch.no_grad()
    def expand_buffers(self, n_new: int, device: torch.device):
        """Expand running buffers when new Gaussians are added."""
        if self._running_depth_error is None:
            return
        self._running_depth_error = torch.cat([
            self._running_depth_error, torch.zeros(n_new, device=device)])
        self._running_color_error = torch.cat([
            self._running_color_error, torch.zeros(n_new, device=device)])
        self._running_normal_error = torch.cat([
            self._running_normal_error, torch.zeros(n_new, device=device)])
        self._visibility_count = torch.cat([
            self._visibility_count, torch.zeros(n_new, device=device)])
        self._zero_contrib_frames = torch.cat([
            self._zero_contrib_frames, torch.zeros(n_new, dtype=torch.long, device=device)])
        if hasattr(self, '_screen_areas') and self._screen_areas is not None:
            self._screen_areas = torch.cat([
                self._screen_areas, torch.zeros(n_new, device=device)])
