"""Utility Estimation Models and Heuristic Baseline Scorers for Phase 4.

This module provides the complete baseline ladder B0 to B7:
  - B0: RandomScorer
  - B1: RGBErrorScorer
  - B2: RGBDepthErrorScorer
  - B3: ErrorInfluenceScorer
  - B4: BinaryThresholdScorer
  - B5: LinearUtilityModel (Single-Head direct utility prediction)
  - B6: TwoHeadLinear (Independent linear quality & cost heads)
  - B7: TwoHeadMLP (Shared backbone + Quality Head + Cost Head)
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class BaseScorer:
    """Base interface for heuristic and non-trainable baseline scorers."""
    def score(self, X: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        raise NotImplementedError


class RandomScorer(BaseScorer):
    """B0: Uniform random baseline."""
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def score(self, X: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        n = len(X)
        return self.rng.random(n).astype(np.float32)


class RGBErrorScorer(BaseScorer):
    """B1: Single appearance heuristic (pre-intervention rgb_error)."""
    def __init__(self, rgb_idx: int = 0):
        self.rgb_idx = rgb_idx

    def score(self, X: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return X[:, self.rgb_idx].copy().astype(np.float32)


class RGBDepthErrorScorer(BaseScorer):
    """B2: Combined RGB + Depth error heuristic."""
    def __init__(self, rgb_idx: int = 0, depth_idx: int = 1, w_rgb: float = 0.70, w_depth: float = 0.30):
        self.rgb_idx = rgb_idx
        self.depth_idx = depth_idx
        self.w_rgb = w_rgb
        self.w_depth = w_depth

    def score(self, X: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return (self.w_rgb * X[:, self.rgb_idx] + self.w_depth * X[:, self.depth_idx]).astype(np.float32)


class ErrorInfluenceScorer(BaseScorer):
    """B3: Error mass heuristic: (RGB_error + Depth_error) * influence_mass."""
    def __init__(self, rgb_idx: int = 0, depth_idx: int = 1, inf_idx: int = 4):
        self.rgb_idx = rgb_idx
        self.depth_idx = depth_idx
        self.inf_idx = inf_idx

    def score(self, X: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        err = X[:, self.rgb_idx] + X[:, self.depth_idx]
        mass = X[:, self.inf_idx]
        return (err * mass).astype(np.float32)


class BinaryThresholdScorer(BaseScorer):
    """B4: Binary threshold on combined error (above median = 1.0, else 0.0)."""
    def __init__(self, rgb_idx: int = 0, depth_idx: int = 1):
        self.rgb_idx = rgb_idx
        self.depth_idx = depth_idx

    def score(self, X: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        err = X[:, self.rgb_idx] + X[:, self.depth_idx]
        med = np.median(err)
        return (err > med).astype(np.float32)


class LinearUtilityModel(nn.Module):
    """B5: Direct single-head linear regression model predicting U* directly."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns direct utility prediction [N]."""
        return self.linear(x).squeeze(-1)

    def predict_utility(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TwoHeadLinear(nn.Module):
    """B6: Two-head linear model predicting Delta Q and Delta T independently."""
    def __init__(self, in_features: int, eps_cost: float = 0.001):
        super().__init__()
        self.eps_cost = eps_cost
        self.linear_q = nn.Linear(in_features, 1)
        self.linear_t = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_q = self.linear_q(x).squeeze(-1)
        delta_t = self.linear_t(x).squeeze(-1) + self.eps_cost
        utility = delta_q / delta_t
        return delta_q, delta_t, utility

    def predict_utility(self, x: torch.Tensor) -> torch.Tensor:
        _, _, u = self.forward(x)
        return u


class TwoHeadMLP(nn.Module):
    """B7: Two-head neural network predicting Quality Gain Delta Q and Cost Delta T independently.
    
    Architecture:
      Shared Backbone: Linear(in_features, hidden_dim) -> LeakyReLU(0.1)
      Head Q:          Linear(hidden_dim, 32) -> LeakyReLU(0.1) -> Linear(32, 1)
      Head T:          Linear(hidden_dim, 32) -> LeakyReLU(0.1) -> Linear(32, 1) -> Softplus()
      Derived Utility: delta_q / (delta_t + eps)
    """
    def __init__(self, in_features: int, hidden_dim: int = 64, eps_cost: float = 0.001):
        super().__init__()
        self.eps_cost = eps_cost
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.1),
        )
        self.head_q = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )
        self.head_t = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Softplus(),  # Execution cost is strictly positive
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        delta_q = self.head_q(feat).squeeze(-1)
        delta_t = self.head_t(feat).squeeze(-1) + self.eps_cost
        utility = delta_q / delta_t
        return delta_q, delta_t, utility

    def predict_utility(self, x: torch.Tensor) -> torch.Tensor:
        _, _, u = self.forward(x)
        return u
