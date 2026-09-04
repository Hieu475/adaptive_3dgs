"""Loss Functions for Learned Utility Estimation and Learning-to-Rank (Phase 4).

Provides:
  - PointwiseTwoHeadLoss: SmoothL1 loss on Quality (Delta Q) and Cost (Delta T).
  - PairwiseRankingLoss: Logistic pairwise ranking loss on derived utility.
  - JointRankingAndPointwiseLoss: Combined ranking + calibration loss.
  - DirectUtilityRegressionLoss: Direct regression loss on utility U* (for single-head models).
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


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


class PairwiseRankingLoss(nn.Module):
    """Pairwise margin-weighted logistic ranking loss over utility values.
    
    Pairs (i, j) are formed whenever target_u[i] - target_u[j] > margin_eps.
    """
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
        if len(pairs_i) == 0:
            return torch.tensor(0.0, device=pred_u.device, requires_grad=True)
            
        diff_pred = pred_u[pairs_i] - pred_u[pairs_j]
        loss_pair = (pair_weights * torch.log1p(torch.exp(-diff_pred.clamp(-15.0, 15.0)))).mean()
        return loss_pair


class JointRankingAndPointwiseLoss(nn.Module):
    """Joint objective balancing pairwise ranking ordering with pointwise metric calibration.
    
    L_total = L_pairwise + lambda_pointwise * (L_q + cost_weight * L_t)
    """
    def __init__(
        self,
        lambda_pointwise: float = 0.25,
        cost_weight: float = 0.5,
        margin_eps: float = 1e-5,
    ):
        super().__init__()
        self.lambda_pointwise = lambda_pointwise
        self.pointwise_loss = PointwiseTwoHeadLoss(cost_weight=cost_weight)
        self.ranking_loss = PairwiseRankingLoss(margin_eps=margin_eps)

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
        loss_pt = self.pointwise_loss(pred_q, pred_t, target_q, target_t)
        loss_pair = self.ranking_loss(pred_u, pairs_i, pairs_j, pair_weights)
        
        total_loss = loss_pair + self.lambda_pointwise * loss_pt
        
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_pairwise": float(loss_pair.item()),
            "loss_pointwise": float(loss_pt.item()),
        }
        return total_loss, metrics


class DirectUtilityRegressionLoss(nn.Module):
    """Direct loss for single-head models predicting U*."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, pred_u: torch.Tensor, target_u: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred_u, target_u)
