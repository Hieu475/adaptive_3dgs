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
        
        self.hysteresis_enabled: bool = True
        self.tau_a_enter: float = 0.8
        self.tau_a_leave: float = 0.65
        self.tau_c_enter: float = 0.2
        self.tau_c_leave: float = 0.35
        self._prev_tiers: Optional[torch.Tensor] = None
        self._state_switch_count: Optional[torch.Tensor] = None
        
        # Novelty boost: new Gaussians get importance boost that decays over time.
        # Fixes cold-start bias where new (unfitted) Gaussians get 0 importance.
        self.novelty_weight: float = weights.get('novelty', 0.5) if weights else 0.5
        self.novelty_warmup_frames: int = 20  # boost decays to 0 over this many frames
        
        # Uncertainty boost: high-uncertainty Gaussians need more optimization.
        # Wire in a GaussianUncertaintyEstimator via set_uncertainty_estimator().
        self.uncertainty_weight: float = weights.get('uncertainty', 0.3) if weights else 0.3
        self._uncertainty_estimator = None  # set via set_uncertainty_estimator()
        
        # Error prior: initialize new Gaussians with scene mean error instead of zero.
        self.use_error_prior: bool = True
        
        # Running statistics (initialized on first call)
        self._running_depth_error: Optional[torch.Tensor] = None
        self._running_color_error: Optional[torch.Tensor] = None
        self._running_normal_error: Optional[torch.Tensor] = None
        self._visibility_count: Optional[torch.Tensor] = None
        self._prev_positions: Optional[torch.Tensor] = None
        self._zero_contrib_frames: Optional[torch.Tensor] = None
        self._creation_frame: Optional[torch.Tensor] = None  # frame when each Gaussian was created
        self._frame_count = 0
    
    def set_uncertainty_estimator(self, estimator):
        """Wire in a GaussianUncertaintyEstimator for uncertainty-boosted importance."""
        self._uncertainty_estimator = estimator
    
    def _ensure_buffers(self, N: int, device: torch.device):
        """Lazily initialize running buffers when Gaussian count is known."""
        if self._running_depth_error is None or self._running_depth_error.shape[0] != N:
            self._running_depth_error = torch.zeros(N, device=device)
            self._running_color_error = torch.zeros(N, device=device)
            self._running_normal_error = torch.zeros(N, device=device)
            self._visibility_count = torch.zeros(N, device=device)
            self._prev_positions = None
            self._zero_contrib_frames = torch.zeros(N, dtype=torch.long, device=device)
            self._creation_frame = torch.full((N,), self._frame_count, dtype=torch.long, device=device)
    
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
        
        # Novelty boost: new (unfitted) Gaussians get higher importance.
        # Decays linearly from novelty_weight to 0 over novelty_warmup_frames.
        # Fixes cold-start bias: oracle experiment showed new Gaussians with
        # importance=0 actually have the highest marginal utility.
        if self._creation_frame is not None and self.novelty_weight > 0:
            N_score = score.shape[0]
            age = self._frame_count - self._creation_frame[:N_score].float()
            novelty = (1.0 - age / max(1, self.novelty_warmup_frames)).clamp(min=0.0)
            score += self.novelty_weight * novelty
        
        # Uncertainty boost: high-uncertainty Gaussians need more optimization.
        # UQ_i = EMA(E_i²) - EMA(E_i)² — variance of error signal.
        # High uncertainty → error is unstable → Gaussian hasn't converged.
        if (self._uncertainty_estimator is not None 
                and self.uncertainty_weight > 0):
            uq = self._uncertainty_estimator.compute_uncertainty()
            if uq.numel() > 0:
                N_score = score.shape[0]
                uq_len = min(uq.shape[0], N_score)
                # Normalize uncertainty to [0, 1] range
                uq_max = uq[:uq_len].max()
                if uq_max > 1e-8:
                    uq_norm = uq[:uq_len] / uq_max
                else:
                    uq_norm = uq[:uq_len]
                score[:uq_len] += self.uncertainty_weight * uq_norm
        
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
    
    def classify_tier_with_hysteresis(self, importance: torch.Tensor) -> torch.Tensor:
        if self._prev_tiers is None or self._prev_tiers.shape[0] != importance.shape[0]:
            self._prev_tiers = None # Reset if shape mismatch
            
        N = importance.shape[0]
        device = importance.device
        
        # If no prev tiers, initialize using simple classification
        if self._prev_tiers is None:
            tiers = torch.full((N,), Tier.C, dtype=torch.long, device=device)
            tiers[importance > self.tau_high] = Tier.A
            tiers[(importance >= self.tau_low) & (importance <= self.tau_high)] = Tier.B
            tiers[importance < self.tau_low] = Tier.C
            if self._zero_contrib_frames is not None:
                tiers[self._zero_contrib_frames > self.prune_patience] = Tier.D
                
            self._prev_tiers = tiers.clone()
            self._state_switch_count = torch.zeros(N, dtype=torch.long, device=device)
            return tiers
            
        tiers = self._prev_tiers.clone()
        
        in_a = (self._prev_tiers == Tier.A)
        in_b = (self._prev_tiers == Tier.B)
        in_c = (self._prev_tiers == Tier.C)
        
        leave_a = in_a & (importance < self.tau_a_leave)
        tiers[leave_a] = Tier.B
        
        enter_a = in_b & (importance > self.tau_a_enter)
        enter_c = in_b & (importance < self.tau_c_enter)
        tiers[enter_a] = Tier.A
        tiers[enter_c] = Tier.C
        
        leave_c = in_c & (importance > self.tau_c_leave)
        tiers[leave_c] = Tier.B
        
        if self._zero_contrib_frames is not None:
            tiers[self._zero_contrib_frames > self.prune_patience] = Tier.D
            
        changed = (tiers != self._prev_tiers)
        if self._state_switch_count is not None:
            self._state_switch_count[changed] += 1
            
        self._prev_tiers = tiers.clone()
        return tiers

    def get_hysteresis_diagnostics(self) -> Dict:
        if self._state_switch_count is None or self._prev_tiers is None:
            return {}
        N = self._state_switch_count.shape[0]
        if N == 0:
            return {}
        switches = self._state_switch_count.float()
        return {
            "avg_switches_per_gaussian": switches.mean().item(),
            "max_switches": switches.max().item(),
            "switch_rate_per_frame": switches.sum().item() / max(1, self._frame_count),
            "tier_distribution": {
                "Tier A": (self._prev_tiers == Tier.A).sum().item() / N,
                "Tier B": (self._prev_tiers == Tier.B).sum().item() / N,
                "Tier C": (self._prev_tiers == Tier.C).sum().item() / N,
                "Tier D": (self._prev_tiers == Tier.D).sum().item() / N,
            }
        }

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
        if getattr(self, 'hysteresis_enabled', False):
            return self.classify_tier_with_hysteresis(importance)
            
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
        """Expand running buffers when new Gaussians are added.
        
        New Gaussians are initialized with scene-mean error (error prior)
        instead of zero, to avoid cold-start bias where unfitted Gaussians
        get 0 importance and are never selected for optimization.
        """
        if self._running_depth_error is None:
            return
        
        # Error prior: initialize new Gaussians with current scene mean error
        # This prevents new Gaussians from starting with importance=0
        if self.use_error_prior:
            depth_prior = self._running_depth_error.mean().item()
            color_prior = self._running_color_error.mean().item()
            normal_prior = self._running_normal_error.mean().item()
        else:
            depth_prior = 0.0
            color_prior = 0.0
            normal_prior = 0.0
        
        self._running_depth_error = torch.cat([
            self._running_depth_error, torch.full((n_new,), depth_prior, device=device)])
        self._running_color_error = torch.cat([
            self._running_color_error, torch.full((n_new,), color_prior, device=device)])
        self._running_normal_error = torch.cat([
            self._running_normal_error, torch.full((n_new,), normal_prior, device=device)])
        self._visibility_count = torch.cat([
            self._visibility_count, torch.zeros(n_new, device=device)])
        self._zero_contrib_frames = torch.cat([
            self._zero_contrib_frames, torch.zeros(n_new, dtype=torch.long, device=device)])
        # Creation frame: mark when these Gaussians were born (for novelty boost)
        if self._creation_frame is not None:
            self._creation_frame = torch.cat([
                self._creation_frame,
                torch.full((n_new,), self._frame_count, dtype=torch.long, device=device)])
        if hasattr(self, '_screen_areas') and self._screen_areas is not None:
            self._screen_areas = torch.cat([
                self._screen_areas, torch.zeros(n_new, device=device)])
        if hasattr(self, '_prev_tiers') and self._prev_tiers is not None:
            self._prev_tiers = torch.cat([
                self._prev_tiers, torch.full((n_new,), Tier.C, dtype=torch.long, device=device)])
        if hasattr(self, '_state_switch_count') and self._state_switch_count is not None:
            self._state_switch_count = torch.cat([
                self._state_switch_count, torch.zeros(n_new, dtype=torch.long, device=device)])

