"""Loss Functions for Learned Utility Estimation and Learning-to-Rank (Phase 4).

Provides modular loss functions decoupled by component:
  - quality_loss: SmoothL1 loss on quality head (Delta Q).
  - cost_loss: SmoothL1 loss on execution cost head (Delta T).
  - pairwise_utility_loss: Pairwise margin-weighted logistic ranking loss on U_hat.
  - TwoHeadUtilityLoss: Configurable total loss:
      L = lambda_rank * L_rank + lambda_q * L_q + lambda_t * L_t
  - DirectUtilityRegressionLoss: Direct SmoothL1 loss on U* (for single-head baselines).
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class LossConfig:
    """Hyperparameters and weighting for two-head utility objectives."""
    lambda_rank: float = 1.0
    lambda_q: float = 0.25
    lambda_t: float = 0.125
    margin_eps: float = 1e-5
    max_pairs: int = 25000


def quality_loss(
    pred_q: torch.Tensor,
    target_q: torch.Tensor,
    loss_fn: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Pointwise reconstruction loss on Quality head."""
    fn = loss_fn or nn.SmoothL1Loss()
    return fn(pred_q, target_q)


def cost_loss(
    pred_t: torch.Tensor,
    target_t: torch.Tensor,
    loss_fn: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Pointwise reconstruction loss on Cost head."""
    fn = loss_fn or nn.SmoothL1Loss()
    return fn(pred_t, target_t)


def pairwise_utility_loss(
    pred_u: torch.Tensor,
    pairs_i: torch.Tensor,
    pairs_j: torch.Tensor,
    pair_weights: torch.Tensor,
) -> torch.Tensor:
    """Margin-weighted logistic pairwise ranking loss."""
    if len(pairs_i) == 0:
        return torch.tensor(0.0, device=pred_u.device, requires_grad=True)
    diff_pred = pred_u[pairs_i] - pred_u[pairs_j]
    return (pair_weights * torch.log1p(torch.exp(-diff_pred.clamp(-15.0, 15.0)))).mean()


def two_head_loss(
    pred_q: torch.Tensor,
    pred_t: torch.Tensor,
    pred_u: torch.Tensor,
    target_q: torch.Tensor,
    target_t: torch.Tensor,
    pairs_i: torch.Tensor,
    pairs_j: torch.Tensor,
    pair_weights: torch.Tensor,
    lambda_rank: float = 1.0,
    lambda_q: float = 0.25,
    lambda_t: float = 0.125,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Functional two-head loss: L = lambda_rank * L_rank + lambda_q * L_q + lambda_t * L_t."""
    l_rank = pairwise_utility_loss(pred_u, pairs_i, pairs_j, pair_weights)
    l_q = quality_loss(pred_q, target_q)
    l_t = cost_loss(pred_t, target_t)
    total = lambda_rank * l_rank + lambda_q * l_q + lambda_t * l_t
    metrics = {
        "loss_total": float(total.item()),
        "loss_rank": float(l_rank.item()),
        "loss_quality": float(l_q.item()),
        "loss_cost": float(l_t.item()),
    }
    return total, metrics


class PairwiseRankingLoss(nn.Module):
    """Encapsulates pair mining and pairwise ranking loss computation."""
    def __init__(self, margin_eps: float = 1e-5, max_pairs: int = 25000):
        super().__init__()
        self.margin_eps = margin_eps
        self.max_pairs = max_pairs

    def find_pairs(self, target_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = target_u.device
        u_np = target_u.detach().cpu().numpy()
        n = len(u_np)
        
        pairs_i = []
        pairs_j = []
        weights = []
        
        for i in range(n):
            for j in range(n):
                diff = float(u_np[i] - u_np[j])
                if diff > self.margin_eps:
                    pairs_i.append(i)
                    pairs_j.append(j)
                    weights.append(diff)
                    if len(pairs_i) >= self.max_pairs:
                        break
            if len(pairs_i) >= self.max_pairs:
                break
                
        if len(pairs_i) == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
            )
            
        i_t = torch.tensor(pairs_i, dtype=torch.long, device=device)
        j_t = torch.tensor(pairs_j, dtype=torch.long, device=device)
        w_t = torch.tensor(weights, dtype=torch.float32, device=device)
        w_t = w_t / (w_t.mean() + 1e-8)
        return i_t, j_t, w_t

    def forward(
        self,
        pred_u: torch.Tensor,
        pairs_i: torch.Tensor,
        pairs_j: torch.Tensor,
        pair_weights: torch.Tensor,
    ) -> torch.Tensor:
        return pairwise_utility_loss(pred_u, pairs_i, pairs_j, pair_weights)


class TwoHeadUtilityLoss(nn.Module):
    """Canonical Two-Head Loss balancing ranking and pointwise calibration:
    
    L = lambda_rank * L_rank + lambda_q * L_q + lambda_t * L_t
    """
    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()
        self.ranking_loss = PairwiseRankingLoss(
            margin_eps=self.config.margin_eps,
            max_pairs=self.config.max_pairs,
        )
        self.loss_fn_q = nn.SmoothL1Loss()
        self.loss_fn_t = nn.SmoothL1Loss()

    def forward(
        self,
        pred_q: torch.Tensor,
        pred_t: torch.Tensor,
        pred_u: torch.Tensor,
        target_q: torch.Tensor,
        target_t: torch.Tensor,
        pairs_i: torch.Tensor,
        pairs_j: torch.Tensor,
        pair_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_rank = pairwise_utility_loss(pred_u, pairs_i, pairs_j, pair_weights)
        l_q = quality_loss(pred_q, target_q, self.loss_fn_q)
        l_t = cost_loss(pred_t, target_t, self.loss_fn_t)
        
        total_loss = (
            self.config.lambda_rank * l_rank +
            self.config.lambda_q * l_q +
            self.config.lambda_t * l_t
        )
        
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_rank": float(l_rank.item()),
            "loss_pairwise": float(l_rank.item()),
            "loss_quality": float(l_q.item()),
            "loss_cost": float(l_t.item()),
            "loss_pointwise": float((self.config.lambda_q * l_q + self.config.lambda_t * l_t).item()),
        }
        return total_loss, metrics


# Backward-compatible alias
class JointRankingAndPointwiseLoss(TwoHeadUtilityLoss):
    """Backward-compatible wrapper for TwoHeadUtilityLoss."""
    def __init__(
        self,
        lambda_pointwise: float = 0.25,
        cost_weight: float = 0.5,
        margin_eps: float = 1e-5,
    ):
        config = LossConfig(
            lambda_rank=1.0,
            lambda_q=lambda_pointwise,
            lambda_t=lambda_pointwise * cost_weight,
            margin_eps=margin_eps,
        )
        super().__init__(config=config)


class PointwiseTwoHeadLoss(nn.Module):
    """Calibrates independent quality and cost heads with Smooth L1 regression."""
    def __init__(self, cost_weight: float = 0.5):
        super().__init__()
        self.cost_weight = cost_weight
        self.loss_fn = nn.SmoothL1Loss()

    def forward(
        self,
        pred_q: torch.Tensor,
        pred_t: torch.Tensor,
        target_q: torch.Tensor,
        target_t: torch.Tensor,
    ) -> torch.Tensor:
        loss_q = self.loss_fn(pred_q, target_q)
        loss_t = self.loss_fn(pred_t, target_t)
        return loss_q + self.cost_weight * loss_t


class DirectUtilityRegressionLoss(nn.Module):
    """Direct loss for single-head models predicting U*."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, pred_u: torch.Tensor, target_u: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred_u, target_u)
