"""
Uncertainty Estimation for Gaussian Splatting

This module implements uncertainty estimation for each Gaussian based on 
error variance using exponential moving averages.
UQ_i = EMA(E_i^2) - EMA(E_i)^2
"""

import torch
from torch import Tensor
from typing import Optional

class GaussianUncertaintyEstimator:
    """Estimates uncertainty of Gaussians using moving averages of errors."""

    def __init__(self, ema_decay: float = 0.95):
        self.ema_decay = ema_decay
        self._ema_error: Optional[Tensor] = None
        self._ema_error_sq: Optional[Tensor] = None
        self._ema_depth_error: Optional[Tensor] = None
        self._ema_depth_error_sq: Optional[Tensor] = None

    def update(self, color_errors: Tensor, depth_errors: Tensor):
        """Update the EMA buffers with new errors."""
        n = color_errors.shape[0]
        device = color_errors.device

        if self._ema_error is None:
            self._ema_error = torch.zeros(n, device=device)
            self._ema_error_sq = torch.zeros(n, device=device)
            self._ema_depth_error = torch.zeros(n, device=device)
            self._ema_depth_error_sq = torch.zeros(n, device=device)

        decay = self.ema_decay

        self._ema_error = decay * self._ema_error + (1 - decay) * color_errors
        self._ema_error_sq = decay * self._ema_error_sq + (1 - decay) * (color_errors ** 2)

        self._ema_depth_error = decay * self._ema_depth_error + (1 - decay) * depth_errors
        self._ema_depth_error_sq = decay * self._ema_depth_error_sq + (1 - decay) * (depth_errors ** 2)

    def compute_color_uncertainty(self) -> Tensor:
        """Uncertainty from color errors only."""
        if self._ema_error is None:
            return torch.empty(0)
        var = self._ema_error_sq - (self._ema_error ** 2)
        return torch.clamp(var, min=0.0)

    def compute_depth_uncertainty(self) -> Tensor:
        """Uncertainty from depth errors only."""
        if self._ema_depth_error is None:
            return torch.empty(0)
        var = self._ema_depth_error_sq - (self._ema_depth_error ** 2)
        return torch.clamp(var, min=0.0)

    def compute_uncertainty(self) -> Tensor:
        """Combined uncertainty from color and depth."""
        color_u = self.compute_color_uncertainty()
        depth_u = self.compute_depth_uncertainty()
        if len(color_u) == 0:
            return torch.empty(0)
        return color_u + depth_u

    def expand_buffers(self, n_new: int, device: torch.device):
        """Expand when new Gaussians added."""
        if self._ema_error is None:
            return
            
        new_err = torch.zeros(n_new, device=device)
        new_err_sq = torch.zeros(n_new, device=device)
        new_depth_err = torch.zeros(n_new, device=device)
        new_depth_err_sq = torch.zeros(n_new, device=device)
        
        self._ema_error = torch.cat([self._ema_error, new_err])
        self._ema_error_sq = torch.cat([self._ema_error_sq, new_err_sq])
        self._ema_depth_error = torch.cat([self._ema_depth_error, new_depth_err])
        self._ema_depth_error_sq = torch.cat([self._ema_depth_error_sq, new_depth_err_sq])

    def get_high_uncertainty_mask(self, threshold: Optional[float] = None) -> Tensor:
        """Get mask for high uncertainty Gaussians."""
        uq = self.compute_uncertainty()
        if len(uq) == 0:
            return torch.zeros(0, dtype=torch.bool)
            
        if threshold is None:
            threshold = uq.mean().item() + 2 * uq.std().item()
            
        return uq > threshold
