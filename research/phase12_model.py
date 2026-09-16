#!/usr/bin/env python3
r"""Phase 12 Model: Two-Stage Decision-Aware Utility Estimation with Global Frame Context.

Implements the overhauled multi-task model architecture:
    Stage 1: s_i, c_t -> P(U_i* > 0)          (Positive Utility Classification Head)
    Stage 2: s_i, c_t -> \hat{\Delta Q_i}, \hat{C_i} (Decoupled Quality and Cost Heads)
    Combined Decision:
        \hat{U}_i = P(U_i* > 0) * \frac{\hat{\Delta Q_i}}{\hat{C_i} + \epsilon}
    Or Thresholded Decision:
        \hat{U}_i = \hat{\Delta Q_i} / \hat{C_i} if P(U_i* > 0) > \tau else -\infty

Architecture:
    Local State (11D)   -> LocalEncoder (64)
                                \
                                 -> SharedFusionMLP (64) -> PositiveUtilityHead -> p_i in [0, 1]
                                /                        -> QualityHead        -> \hat{\Delta Q_i}
    Global Context (12D)-> GlobalContextEncoder (32)     -> CostHead (Softplus)-> \hat{C_i} > 0
                                                                                   |
                                                                \hat{U}_i = p_i * \hat{\Delta Q} / (\hat{C} + \epsilon)

Design Constraints (Strict Protocol Adherence):
    - NO GNN.
    - NO Transformer / Cross-Attention.
    - Compact MLP with shared fusion backbone (<15k parameters).
    - GPU inference latency < 0.5 ms.
"""
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from research.phase12_protocol import (
    CANONICAL_LOCAL_FEATURES,
    CANONICAL_GLOBAL_FEATURES,
    SAFETY_FACTOR,
    DEFAULT_BUDGET_MS,
)
from research.utility_metrics import safe_spearmanr, compute_ndcg_at_k


# ─────────────────────────────────────────────────────────────────────────────
# Encoders & Heads
# ─────────────────────────────────────────────────────────────────────────────

class LocalEncoder(nn.Module):
    """Encodes 11-dimensional local Gaussian state s_i into latent feature h_loc."""

    def __init__(self, in_dim: int = 11, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x_loc: torch.Tensor) -> torch.Tensor:
        return self.net(x_loc)


class GlobalContextEncoder(nn.Module):
    """Encodes 12-dimensional global frame context c_t into latent feature h_glob."""

    def __init__(self, in_dim: int = 12, hidden_dim: int = 32, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x_glob: torch.Tensor) -> torch.Tensor:
        return self.net(x_glob)


class PositiveUtilityHead(nn.Module):
    """Predicts probability of strictly positive utility p_i = P(U_i* > 0)."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class QualityHead(nn.Module):
    """Predicts continuous quality gain \\hat{\\Delta Q_i}."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class CostHead(nn.Module):
    """Predicts strictly positive execution compute cost \\hat{C_i} > 0."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 32, eps_cost: float = 0.001):
        super().__init__()
        self.eps_cost = eps_cost
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1) + self.eps_cost


# ─────────────────────────────────────────────────────────────────────────────
# Full Two-Stage Multi-Task Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TwoStageModelConfig:
    """Configuration for TwoStageUtilityModel."""
    local_in_dim: int = 11
    global_in_dim: int = 12
    local_hidden_dim: int = 64
    global_hidden_dim: int = 32
    fusion_hidden_dim: int = 64
    head_hidden_dim: int = 32
    dropout: float = 0.05
    eps_cost: float = 0.001
    use_global: bool = True
    use_positive_head: bool = True
    decision_threshold: float = 0.50


class TwoStageUtilityModel(nn.Module):
    """Two-Stage Multi-Task Utility Model with optional Global Context.

    Capable of instantiating:
        M2: use_global=False, use_positive_head=False (Current TwoHeadMLP equivalent)
        M3: use_global=False, use_positive_head=True  (Local + Positive Head)
        M4: use_global=True,  use_positive_head=True  (Local + Global Context + Positive Head)
    """

    def __init__(self, config: Optional[TwoStageModelConfig] = None):
        super().__init__()
        self.config = config or TwoStageModelConfig()
        c = self.config
        self.use_global = c.use_global
        self.use_positive_head = c.use_positive_head
        self.eps_cost = c.eps_cost
        self.decision_threshold = c.decision_threshold

        # 1. Local observable state encoder
        self.local_encoder = LocalEncoder(
            in_dim=c.local_in_dim,
            hidden_dim=c.local_hidden_dim,
            dropout=c.dropout,
        )

        # 2. Global scene context encoder (if enabled)
        if self.use_global:
            self.global_encoder = GlobalContextEncoder(
                in_dim=c.global_in_dim,
                hidden_dim=c.global_hidden_dim,
                dropout=c.dropout,
            )
            fused_in_dim = c.local_hidden_dim + c.global_hidden_dim
        else:
            self.global_encoder = None
            fused_in_dim = c.local_hidden_dim

        # 3. Shared Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_in_dim, c.fusion_hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(c.dropout),
        )

        # 4. Multi-Task Heads
        if self.use_positive_head:
            self.head_positive = PositiveUtilityHead(
                in_dim=c.fusion_hidden_dim,
                hidden_dim=c.head_hidden_dim,
            )
        else:
            self.head_positive = None

        self.head_quality = QualityHead(
            in_dim=c.fusion_hidden_dim,
            hidden_dim=c.head_hidden_dim,
        )

        self.head_cost = CostHead(
            in_dim=c.fusion_hidden_dim,
            hidden_dim=c.head_hidden_dim,
            eps_cost=c.eps_cost,
        )

    def forward(
        self,
        x_loc: torch.Tensor,
        x_glob: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_loc:  [N, 11] local Gaussian state tensor.
            x_glob: [12] or [N, 12] optional global context tensor.

        Returns:
            p: [N] positive utility probability (or all-ones if positive head disabled).
            q: [N] continuous predicted quality gain.
            t: [N] strictly positive execution cost.
            u: [N] decision utility = p * (q / t) or (q / t).
        """
        N = x_loc.shape[0]
        h_loc = self.local_encoder(x_loc)

        if self.use_global and self.global_encoder is not None:
            if x_glob is None:
                # Fallback to zero context if not provided
                x_glob = torch.zeros(
                    (N, self.config.global_in_dim),
                    dtype=x_loc.dtype,
                    device=x_loc.device,
                )
            elif x_glob.dim() == 1:
                # Expand [12] to [N, 12]
                x_glob = x_glob.unsqueeze(0).expand(N, -1)

            h_glob = self.global_encoder(x_glob)
            h_fused = self.fusion_mlp(torch.cat([h_loc, h_glob], dim=-1))
        else:
            h_fused = self.fusion_mlp(h_loc)

        q = self.head_quality(h_fused)
        t = self.head_cost(h_fused)

        if self.use_positive_head and self.head_positive is not None:
            p = self.head_positive(h_fused)
            # Decision utility: probability-modulated expected return
            u = p * (q / t)
        else:
            p = torch.ones_like(q)
            u = q / t

        return p, q, t, u

    def predict_utility(
        self,
        x_loc: torch.Tensor,
        x_glob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience utility inference method."""
        _, _, _, u = self.forward(x_loc, x_glob)
        return u

    def predict_thresholded_utility(
        self,
        x_loc: torch.Tensor,
        x_glob: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
    ) -> torch.Tensor:
        """Thresholded utility: drops candidates with p_i <= tau."""
        t_thresh = tau if tau is not None else self.decision_threshold
        p, q, t, u = self.forward(x_loc, x_glob)
        if self.use_positive_head:
            drop_mask = p <= t_thresh
            u_thresh = q / t
            u_thresh[drop_mask] = -1e9
            return u_thresh
        return u


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Loss Formulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TwoStageLossConfig:
    """Weights for the composite multi-task objective."""
    lambda_pos: float = 2.0      # Weight for BCE classification of U* > 0
    lambda_q: float = 1.0        # Weight for quality gain regression
    lambda_t: float = 0.5        # Weight for compute cost regression
    lambda_rank: float = 0.5     # Weight for pairwise ranking margin
    ranking_margin: float = 0.05
    q_scale: float = 1e4         # Quality scale factor for numerical balance
    t_scale: float = 0.01        # Cost scale factor (maps ms to ~1.0)


class TwoStageLoss(nn.Module):
    r"""Composite loss: BCE(p, U* > 0) + SmoothL1(q, \Delta Q*) + SmoothL1(t, \Delta T*) + RankLoss."""

    def __init__(self, config: Optional[TwoStageLossConfig] = None):
        super().__init__()
        self.config = config or TwoStageLossConfig()
        self.bce = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        pred_p: torch.Tensor,
        pred_q: torch.Tensor,
        pred_t: torch.Tensor,
        pred_u: torch.Tensor,
        target_q: torch.Tensor,
        target_t: torch.Tensor,
        target_u: torch.Tensor,
        use_positive_head: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        c = self.config
        loss_components: Dict[str, float] = {}

        # 1. Positive Utility Classification Loss
        target_pos = (target_u > 0.0).float()
        if use_positive_head:
            loss_pos = self.bce(pred_p, target_pos)
            loss_components["loss_pos"] = float(loss_pos.item())
        else:
            loss_pos = torch.tensor(0.0, device=pred_q.device)
            loss_components["loss_pos"] = 0.0

        # 2. Quality Gain Regression Loss (scaled for gradient parity)
        scaled_target_q = target_q * c.q_scale
        scaled_pred_q = pred_q * c.q_scale
        loss_q = self.smooth_l1(scaled_pred_q, scaled_target_q)
        loss_components["loss_q"] = float(loss_q.item())

        # 3. Cost Regression Loss
        scaled_target_t = target_t * c.t_scale
        scaled_pred_t = pred_t * c.t_scale
        loss_t = self.smooth_l1(scaled_pred_t, scaled_target_t)
        loss_components["loss_t"] = float(loss_t.item())

        # 4. Pairwise Ranking Loss
        N = len(target_u)
        if N > 1 and c.lambda_rank > 0.0:
            diff_target = torch.sign(target_u.unsqueeze(1) - target_u.unsqueeze(0))
            diff_pred = pred_u.unsqueeze(1) - pred_u.unsqueeze(0)
            # Sample up to 500 random pairs to maintain fast backpropagation
            mask = (torch.rand((N, N), device=pred_u.device) < 0.10) & (diff_target != 0)
            if mask.any():
                margin_violations = torch.relu(
                    -diff_target[mask] * diff_pred[mask] + c.ranking_margin
                )
                loss_rank = margin_violations.mean()
            else:
                loss_rank = torch.tensor(0.0, device=pred_u.device)
        else:
            loss_rank = torch.tensor(0.0, device=pred_u.device)
        loss_components["loss_rank"] = float(loss_rank.item())

        total_loss = (
            c.lambda_pos * loss_pos
            + c.lambda_q * loss_q
            + c.lambda_t * loss_t
            + c.lambda_rank * loss_rank
        )
        loss_components["loss_total"] = float(total_loss.item())

        return total_loss, loss_components


# ─────────────────────────────────────────────────────────────────────────────
# Dataset & Preprocessing Utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_local_features_from_df(df: pd.DataFrame) -> np.ndarray:
    """Extracts canonical 11D local observable features s_i from DataFrame."""
    cols = CANONICAL_LOCAL_FEATURES
    X = df[cols].to_numpy().astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)


def compute_global_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Computes canonical 12D global frame context c_t for all frame groups."""
    rows = []
    for (scene, frame, seed), g in df.groupby(["scene", "frame", "seed"]):
        vis = g["feat_visibility"].to_numpy() > 0
        vis_count = float(np.sum(vis))
        n_total = float(len(g))

        rows.append({
            "scene": scene,
            "frame": int(frame),
            "seed": int(seed),
            "glob_gaussian_count": n_total,
            "glob_visible_count": vis_count,
            "glob_visible_fraction": vis_count / max(n_total, 1.0),
            "glob_mean_rgb_err": float(g["feat_rgb_error"].mean()),
            "glob_std_rgb_err": float(g["feat_rgb_error"].std(ddof=0)),
            "glob_mean_depth_err": float(g["feat_depth_error"].mean()),
            "glob_std_depth_err": float(g["feat_depth_error"].std(ddof=0)),
            "glob_mean_grad_norm": float(g["feat_gradient_norm"].mean()),
            "glob_std_grad_norm": float(g["feat_gradient_norm"].std(ddof=0)),
            "glob_mean_influence": float(g["feat_influence_mass"].mean()),
            "glob_selected_fraction": float(g["feat_update_frequency"].mean()),
            "glob_normalized_frame_idx": float(frame) / 60.0,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Model Trainer & Checkpointer
# ─────────────────────────────────────────────────────────────────────────────

class TwoStageModelTrainer:
    """Trains and validates TwoStageUtilityModel with reproducible checkpoints."""

    def __init__(
        self,
        model: TwoStageUtilityModel,
        loss_config: Optional[TwoStageLossConfig] = None,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = TwoStageLoss(config=loss_config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(
        self,
        X_loc: torch.Tensor,
        y_q: torch.Tensor,
        y_t: torch.Tensor,
        y_u: torch.Tensor,
        X_glob: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        X_loc = X_loc.to(self.device)
        X_glob = X_glob.to(self.device) if X_glob is not None else None
        y_q = y_q.to(self.device)
        y_t = y_t.to(self.device)
        y_u = y_u.to(self.device)

        p, q, t, u = self.model(X_loc, X_glob)
        loss, loss_dict = self.loss_fn(
            pred_p=p,
            pred_q=q,
            pred_t=t,
            pred_u=u,
            target_q=y_q,
            target_t=y_t,
            target_u=y_u,
            use_positive_head=self.model.use_positive_head,
        )
        loss.backward()
        self.optimizer.step()

        return loss_dict

    def evaluate(
        self,
        X_loc: torch.Tensor,
        y_u_true: np.ndarray,
        X_glob: Optional[torch.Tensor] = None,
        k: int = 20,
    ) -> Dict[str, float]:
        """Evaluates model against ground-truth targets."""
        self.model.eval()
        with torch.no_grad():
            X_loc = X_loc.to(self.device)
            X_glob = X_glob.to(self.device) if X_glob is not None else None
            p, q, t, u = self.model(X_loc, X_glob)

            p_np = p.cpu().numpy()
            u_np = u.cpu().numpy()

        rho, _ = safe_spearmanr(u_np, y_u_true)
        ndcg = compute_ndcg_at_k(u_np, y_u_true, k=k)

        # AUROC for positive utility detection
        y_pos = (y_u_true > 0.0).astype(int)
        if len(np.unique(y_pos)) > 1:
            if self.model.use_positive_head:
                auroc = float(roc_auc_score(y_pos, p_np))
            else:
                auroc = float(roc_auc_score(y_pos, u_np))
        else:
            auroc = 0.50

        return {
            "spearman_rho": float(rho),
            "auroc": float(auroc),
            "ndcg_20": float(ndcg),
        }

    def fit(
        self,
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
        epochs: int = 250,
        eval_interval: int = 10,
    ) -> Dict[str, Any]:
        """Trains model with validation tracking and best-checkpoint restoration."""
        best_val_score = -1e9
        best_state = None

        X_loc_tr = train_data["X_loc"]
        X_glob_tr = train_data.get("X_glob", None)
        y_q_tr = train_data["y_q"]
        y_t_tr = train_data["y_t"]
        y_u_tr = train_data["y_u"]

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(
                X_loc=X_loc_tr,
                y_q=y_q_tr,
                y_t=y_t_tr,
                y_u=y_u_tr,
                X_glob=X_glob_tr,
            )

            if val_data is not None and (epoch % eval_interval == 0 or epoch == epochs):
                val_eval = self.evaluate(
                    X_loc=val_data["X_loc"],
                    y_u_true=val_data["y_u_np"],
                    X_glob=val_data.get("X_glob", None),
                )
                # Composite validation score: 0.5 * rho + 0.5 * auroc
                composite = 0.5 * val_eval["spearman_rho"] + 0.5 * val_eval["auroc"]
                if composite > best_val_score:
                    best_val_score = composite
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "best_val_score": best_val_score,
            "final_train_loss": train_metrics.get("loss_total", 0.0),
        }

    def save_checkpoint(self, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        save_dict = {
            "state_dict": self.model.state_dict(),
            "config": asdict(self.model.config),
            "metadata": metadata or {},
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, str(p))

    @staticmethod
    def load_checkpoint(
        path: Union[str, Path],
        device: str = "cpu",
    ) -> TwoStageUtilityModel:
        checkpoint = torch.load(str(path), map_location=device)
        cfg_dict = checkpoint.get("config", {})
        config = TwoStageModelConfig(**cfg_dict)
        model = TwoStageUtilityModel(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
