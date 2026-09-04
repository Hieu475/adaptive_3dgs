"""Training Pipeline and Checkpointing for Learned Utility Models (Phase 4).

Handles:
  - Supervised training for TwoHeadMLP, TwoHeadLinear, and LinearUtilityModel.
  - Joint ranking + pointwise calibration objectives.
  - Validation monitoring on independent validation split (no test leakage).
  - Reproducible multi-seed training across protocol seeds.
  - Model serialization and checkpoint loading.
"""
from dataclasses import dataclass, field
import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from research.utility_dataset import UtilityDataset
from research.utility_losses import (
    LossConfig,
    TwoHeadUtilityLoss,
    DirectUtilityRegressionLoss,
    PairwiseRankingLoss,
)
from research.utility_metrics import safe_spearmanr


@dataclass
class TrainingConfig:
    """Hyperparameters for utility model training."""
    epochs: int = 200
    learning_rate: float = 0.005
    weight_decay: float = 1e-4
    loss_config: LossConfig = field(default_factory=LossConfig)
    device: str = "cpu"


class UtilityModelTrainer:
    """Encapsulates model training, validation tracking, and model checkpointing."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu")

    def train_two_head_model(
        self,
        model: nn.Module,
        train_ds: UtilityDataset,
        val_ds: Optional[UtilityDataset] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Train a two-head model (TwoHeadMLP or TwoHeadLinear) using TwoHeadUtilityLoss."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model.to(self.device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        loss_fn = TwoHeadUtilityLoss(config=self.config.loss_config)

        # Precompute pairwise constraints on train targets
        X_train = train_ds.X.to(self.device)
        y_q_train = train_ds.delta_q.to(self.device)
        y_t_train = train_ds.delta_t.to(self.device)
        y_u_train = train_ds.utility.to(self.device)

        pairs_i, pairs_j, pair_weights = loss_fn.ranking_loss.find_pairs(y_u_train)

        history: List[Dict[str, float]] = []
        best_val_rho = -1.0
        best_state: Optional[Dict[str, Any]] = None

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred_q, pred_t, pred_u = model(X_train)
            
            loss, loss_components = loss_fn(
                pred_q=pred_q,
                pred_t=pred_t,
                pred_u=pred_u,
                target_q=y_q_train,
                target_t=y_t_train,
                pairs_i=pairs_i,
                pairs_j=pairs_j,
                pair_weights=pair_weights,
            )
            
            loss.backward()
            optimizer.step()

            log_entry = {
                "epoch": epoch,
                "loss_total": loss_components["loss_total"],
                "loss_rank": loss_components["loss_rank"],
                "loss_quality": loss_components["loss_quality"],
                "loss_cost": loss_components["loss_cost"],
            }

            # Optional validation step
            if val_ds is not None and (epoch % 10 == 0 or epoch == self.config.epochs):
                model.eval()
                with torch.no_grad():
                    X_val = val_ds.X.to(self.device)
                    _, _, pred_u_val = model(X_val)
                    val_rho, _ = safe_spearmanr(pred_u_val.cpu().numpy(), val_ds.utility_np)
                    log_entry["val_spearman_rho"] = val_rho
                    if val_rho > best_val_rho:
                        best_val_rho = val_rho
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            history.append(log_entry)

        # Restore best validation weights if tracked
        if best_state is not None:
            model.load_state_dict(best_state)

        return {
            "seed": seed,
            "final_loss": history[-1]["loss_total"],
            "best_val_rho": best_val_rho if best_state is not None else float("nan"),
            "history": history,
        }

    def train_linear_utility_model(
        self,
        model: nn.Module,
        train_ds: UtilityDataset,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Train a direct linear utility model (B5) with SmoothL1 regression."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        loss_fn = DirectUtilityRegressionLoss()

        X_train = train_ds.X.to(self.device)
        y_u_train = train_ds.utility.to(self.device)

        for _ in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            pred_u = model(X_train)
            loss = loss_fn(pred_u, y_u_train)
            loss.backward()
            optimizer.step()

        return {"seed": seed, "final_loss": float(loss.item())}

    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        save_path: str,
        feature_names: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves model weights and architectural metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        checkpoint = {
            "state_dict": model.state_dict(),
            "in_features": len(feature_names),
            "feature_names": feature_names,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, save_path)

    @staticmethod
    def load_checkpoint(model: nn.Module, checkpoint_path: str) -> Dict[str, Any]:
        """Loads weights from checkpoint into model instance."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        return ckpt
