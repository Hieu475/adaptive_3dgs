import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, ndcg_score

# Relative imports as per conventions
# from .utility_features import CANONICAL_FEATURE_NAMES
# from .phase12_protocol import ... 

class LocalEncoder(torch_nn.Module):
    """
    Encodes the 11-dimensional local candidate state (s_i) into a dense embedding.
    """
    def __init__(self, input_dim: int = 11, hidden_dim: int = 32, embedding_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = torch_nn.Sequential(
            torch_nn.Linear(input_dim, hidden_dim),
            torch_nn.GELU(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden_dim, embedding_dim),
            torch_nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, s_i: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_i: (batch_size, input_dim) tensor of local features
        Returns:
            (batch_size, embedding_dim) local embedding
        """
        return self.net(s_i)

class ContextEncoder(torch_nn.Module):
    """
    Encodes the 51-dimensional permutation-invariant context representation (h_S).
    """
    def __init__(self, input_dim: int = 51, hidden_dim: int = 64, embedding_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = torch_nn.Sequential(
            torch_nn.Linear(input_dim, hidden_dim),
            torch_nn.GELU(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden_dim, embedding_dim),
            torch_nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, h_S: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_S: (batch_size, input_dim) tensor of context features
        Returns:
            (batch_size, embedding_dim) context embedding
        """
        return self.net(h_S)

class ConditionalFusion(torch_nn.Module):
    """
    Fuses the local embedding and context embedding to produce a joint representation.
    """
    def __init__(self, local_dim: int = 32, context_dim: int = 32, hidden_dim: int = 64, fusion_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = torch_nn.Sequential(
            torch_nn.Linear(local_dim + context_dim, hidden_dim),
            torch_nn.GELU(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden_dim, fusion_dim),
            torch_nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, local_emb: torch.Tensor, ctx_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_emb: (batch_size, local_dim) tensor
            ctx_emb: (batch_size, context_dim) tensor
        Returns:
            (batch_size, fusion_dim) fused representation
        """
        combined = torch.cat([local_emb, ctx_emb], dim=-1)
        return self.net(combined)

class InteractionAwareModel(torch_nn.Module):
    """
    Full Phase 12 Interaction-Aware Conditional Utility Model V_theta(i | S, B_rem).
    Outputs a probability of positive utility, predicted conditional quality gain,
    and predicted conditional cost.
    """
    def __init__(self, local_dim: int = 11, context_dim: int = 51, embedding_dim: int = 32, 
                 fusion_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
        self.local_encoder = LocalEncoder(input_dim=local_dim, hidden_dim=hidden_dim, 
                                          embedding_dim=embedding_dim, dropout=dropout)
        self.context_encoder = ContextEncoder(input_dim=context_dim, hidden_dim=hidden_dim, 
                                              embedding_dim=embedding_dim, dropout=dropout)
        
        self.fusion = ConditionalFusion(local_dim=embedding_dim, context_dim=embedding_dim, 
                                        hidden_dim=hidden_dim, fusion_dim=fusion_dim, dropout=dropout)
        
        # Heads
        self.positive_head = torch_nn.Linear(fusion_dim, 1)  # p_i = P(U* > 0)
        self.quality_head = torch_nn.Linear(fusion_dim, 1)   # delta_q_hat
        self.cost_head = torch_nn.Linear(fusion_dim, 1)      # delta_c_hat
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch_nn.Linear):
                torch_nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch_nn.init.zeros_(m.bias)

    def forward(self, s_i: torch.Tensor, h_S: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            s_i: (batch_size, 11) local candidate state
            h_S: (batch_size, 51) permutation-invariant context
        Returns:
            Dict containing:
                p_i: (batch_size, 1) Probability U*(i|S) > 0
                delta_q: (batch_size, 1) Predicted quality gain
                delta_c: (batch_size, 1) Predicted cost
                utility: (batch_size, 1) Predicted composite utility
        """
        local_emb = self.local_encoder(s_i)
        ctx_emb = self.context_encoder(h_S)
        fused = self.fusion(local_emb, ctx_emb)
        
        p_i = torch.sigmoid(self.positive_head(fused))
        delta_q = self.quality_head(fused)
        delta_c = F.softplus(self.cost_head(fused)) + self.eps  # ensure strictly positive
        
        # U_hat(i|S) = p_i * delta_q_hat / delta_c_hat
        utility = p_i * delta_q / delta_c
        
        return {
            'p_i': p_i,
            'delta_q': delta_q,
            'delta_c': delta_c,
            'utility': utility
        }

class InteractionAwareLoss(torch_nn.Module):
    """
    Combined loss for the 3-head interaction-aware utility model.
    """
    def __init__(self, lambda_q: float = 1.0, lambda_c: float = 0.5, 
                 lambda_p: float = 0.3, lambda_rank: float = 0.2, margin: float = 0.1):
        super().__init__()
        self.lambda_q = lambda_q
        self.lambda_c = lambda_c
        self.lambda_p = lambda_p
        self.lambda_rank = lambda_rank
        self.margin = margin
        
        self.mse = torch_nn.MSELoss()
        self.bce = torch_nn.BCELoss()
        
    def pairwise_rank_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise margin ranking loss.
        """
        # Create pairs: preds_i - preds_j, targets_i - targets_j
        diff_preds = preds.unsqueeze(1) - preds.unsqueeze(0)
        diff_targets = targets.unsqueeze(1) - targets.unsqueeze(0)
        
        # Target sign: 1 if target_i > target_j, -1 if target_i < target_j, 0 if equal
        target_sign = torch.sign(diff_targets)
        
        # Margin ranking loss: max(0, -y * (x1 - x2) + margin)
        # We only consider pairs where target_i != target_j
        mask = (target_sign != 0)
        
        if not mask.any():
            return torch.tensor(0.0, device=preds.device)
            
        loss = F.relu(-target_sign[mask] * diff_preds[mask] + self.margin)
        return loss.mean()
        
    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            preds: Outputs from InteractionAwareModel
            targets: Dict with 'delta_q', 'delta_c', 'utility'
        """
        # 1. Quality Regression Loss
        loss_q = self.mse(preds['delta_q'], targets['delta_q'])
        
        # 2. Cost Regression Loss
        loss_c = self.mse(preds['delta_c'], targets['delta_c'])
        
        # 3. Probability Classification Loss
        # Binary target: 1 if utility > 0, 0 otherwise
        target_p = (targets['utility'] > 0).float()
        loss_p = self.bce(preds['p_i'], target_p)
        
        # 4. Pairwise Ranking Loss on Utility
        loss_rank = self.pairwise_rank_loss(preds['utility'], targets['utility'])
        
        total_loss = (self.lambda_q * loss_q + 
                      self.lambda_c * loss_c + 
                      self.lambda_p * loss_p + 
                      self.lambda_rank * loss_rank)
                      
        return total_loss

class ConditionalUtilityDataset(Dataset):
    """
    PyTorch Dataset for conditional utility data collected in Phase 12-J.
    """
    def __init__(self, records: List[Dict[str, Any]], normalizer: Optional[Any] = None):
        """
        Args:
            records: List of dictionaries containing sample data. Expected keys:
                     's_i', 'h_S', 'delta_q', 'delta_c', 'utility'
            normalizer: Optional normalizer object
        """
        self.records = records
        self.normalizer = normalizer
        
        # Extract features and targets
        self.s_i = torch.tensor([r['s_i'] for r in records], dtype=torch.float32)
        self.h_S = torch.tensor([r['h_S'] for r in records], dtype=torch.float32)
        
        # Reshape to (N, 1)
        self.delta_q = torch.tensor([r['delta_q'] for r in records], dtype=torch.float32).unsqueeze(1)
        self.delta_c = torch.tensor([r.get('delta_c', r.get('delta_c_ms', 1.0)) for r in records], dtype=torch.float32).unsqueeze(1)
        self.utility = torch.tensor([r['utility'] for r in records], dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.records)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        s_i = self.s_i[idx]
        h_S = self.h_S[idx]
        targets = {
            'delta_q': self.delta_q[idx],
            'delta_c': self.delta_c[idx],
            'utility': self.utility[idx]
        }
        return s_i, h_S, targets

class InteractionModelTrainer:
    """
    Trains the interaction-aware model on conditional dataset.
    """
    def __init__(self, model: InteractionAwareModel, 
                 train_data: ConditionalUtilityDataset, 
                 val_data: ConditionalUtilityDataset, 
                 config: Dict[str, Any]):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        
        self.device = config.get('device', 'cpu')
        self.model = self.model.to(self.device)
        
        batch_size = config.get('batch_size', 32)
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
        
        lr = config.get('lr', 1e-3)
        weight_decay = config.get('weight_decay', 1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        loss_kwargs = config.get('loss_kwargs', {})
        self.criterion = InteractionAwareLoss(**loss_kwargs).to(self.device)
        
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        
        for s_i, h_S, targets in self.train_loader:
            s_i, h_S = s_i.to(self.device), h_S.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            preds = self.model(s_i, h_S)
            loss = self.criterion(preds, targets)
            
            loss.backward()
            torch_nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * s_i.size(0)
            
        return {'loss': total_loss / len(self.train_data)}
        
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model and compute metrics: MSE, Spearman rho, AUROC, NDCG@K
        """
        self.model.eval()
        total_loss = 0.0
        
        all_utility_preds = []
        all_utility_targets = []
        all_p_preds = []
        all_p_targets = []
        
        with torch.no_grad():
            for s_i, h_S, targets in dataloader:
                s_i, h_S = s_i.to(self.device), h_S.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                preds = self.model(s_i, h_S)
                loss = self.criterion(preds, targets)
                
                total_loss += loss.item() * s_i.size(0)
                
                all_utility_preds.extend(preds['utility'].cpu().numpy().flatten())
                all_utility_targets.extend(targets['utility'].cpu().numpy().flatten())
                
                all_p_preds.extend(preds['p_i'].cpu().numpy().flatten())
                all_p_targets.extend((targets['utility'] > 0).float().cpu().numpy().flatten())
                
        # Compute metrics
        u_preds_np = np.array(all_utility_preds)
        u_targets_np = np.array(all_utility_targets)
        
        metrics = {'loss': total_loss / len(dataloader.dataset)}
        
        if len(u_preds_np) > 1:
            metrics['mse'] = np.mean((u_preds_np - u_targets_np) ** 2)
            
            # Spearman correlation
            rho, _ = spearmanr(u_preds_np, u_targets_np)
            metrics['spearman'] = rho if not np.isnan(rho) else 0.0
            
            # AUROC for p_i (skip if only one class in targets)
            p_targets_np = np.array(all_p_targets)
            if len(np.unique(p_targets_np)) > 1:
                metrics['auroc'] = roc_auc_score(p_targets_np, np.array(all_p_preds))
                
            # NDCG (ranking evaluation)
            # Reshape to (1, N) for ndcg_score
            try:
                metrics['ndcg'] = ndcg_score(u_targets_np.reshape(1, -1), u_preds_np.reshape(1, -1))
            except Exception:
                pass
                
        return metrics
        
    def train(self, n_epochs: int = 100) -> Dict[str, Any]:
        """
        Full training loop.
        """
        history = {'train_loss': [], 'val_loss': [], 'val_spearman': []}
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(n_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            if 'spearman' in val_metrics:
                history['val_spearman'].append(val_metrics['spearman'])
                
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | Val Spear: {val_metrics.get('spearman', 0.0):.4f}")
                
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return history
