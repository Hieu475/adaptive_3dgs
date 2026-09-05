"""Phase 6: Context-Aware Two-Head Utility Model.

Architecture:
    Self (11)     → f_self  → h_self  (64)
                                        ↘
    Neighbor (8)  → f_neigh → h_neigh (32) → Fusion MLP (160→128→64) → Quality Head → ΔQ_hat
                                        ↗                              → Cost Head   → ΔT_hat (softplus)
    Overlap (5)   → f_over  → h_over  (32)                                              ↓
                                        /                              U_hat = ΔQ / (ΔT + ε)
    Selected (8)  → f_sel   → h_sel   (32)

Keeps the Two-Head philosophy from Phase 4 (separate quality gain and cost heads).
Extends the input from 11-dim pointwise features to 32-dim contextual features.

Variants (for ablation):
    V8:  Self only (11-dim)               → same as Phase 4 TwoHeadMLP
    V9:  Self + Neighbor (19-dim)
    V10: Self + Neighbor + Overlap (24-dim)
    V11: Self + Neighbor + Overlap + Selected (32-dim)  ← FULL
    V12: V11 + Adaptive Greedy (selection-time re-ranking)

Loss:
    L = λ_Q · SmoothL1(ΔQ_hat, ΔQ*) + λ_C · SmoothL1(ΔT_hat, ΔT*) + λ_R · MarginRankingLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .phase6_context import (
    PHASE6_FEATURE_DIM,
    SELF_SLICE,
    NEIGHBOR_SLICE,
    OVERLAP_SLICE,
    SELECTED_SLICE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Phase6ModelConfig:
    """Configuration for the context-aware utility model.

    Attributes:
        self_dim: Input dimension for self features (11 canonical).
        neighbor_dim: Input dimension for neighbor context.
        overlap_dim: Input dimension for overlap context.
        selected_dim: Input dimension for selected-set context.
        self_hidden: Hidden dimension for self encoder.
        neighbor_hidden: Hidden dimension for neighbor encoder.
        overlap_hidden: Hidden dimension for overlap encoder.
        selected_hidden: Hidden dimension for selected encoder.
        fusion_hidden: Hidden dimension for fusion MLP.
        head_hidden: Hidden dimension for quality/cost heads.
        dropout: Dropout rate.
        eps_cost: Minimum cost output (prevents division by zero).
        use_neighbor: Whether to use neighbor context.
        use_overlap: Whether to use overlap context.
        use_selected: Whether to use selected-set context.
    """
    self_dim: int = 11
    neighbor_dim: int = 8
    overlap_dim: int = 5
    selected_dim: int = 8
    self_hidden: int = 64
    neighbor_hidden: int = 32
    overlap_hidden: int = 32
    selected_hidden: int = 32
    fusion_hidden: int = 128
    head_hidden: int = 64
    dropout: float = 0.1
    eps_cost: float = 0.001
    use_neighbor: bool = True
    use_overlap: bool = True
    use_selected: bool = True

    @property
    def total_input_dim(self) -> int:
        """Total input dimension based on enabled context groups."""
        dim = self.self_dim
        if self.use_neighbor:
            dim += self.neighbor_dim
        if self.use_overlap:
            dim += self.overlap_dim
        if self.use_selected:
            dim += self.selected_dim
        return dim

    @property
    def fusion_input_dim(self) -> int:
        """Fusion layer input dimension (sum of encoder hidden dims)."""
        dim = self.self_hidden
        if self.use_neighbor:
            dim += self.neighbor_hidden
        if self.use_overlap:
            dim += self.overlap_hidden
        if self.use_selected:
            dim += self.selected_hidden
        return dim

    @property
    def variant_name(self) -> str:
        """Human-readable variant name for ablation."""
        if not self.use_neighbor and not self.use_overlap and not self.use_selected:
            return "V8_self_only"
        if self.use_neighbor and not self.use_overlap and not self.use_selected:
            return "V9_self_neighbor"
        if self.use_neighbor and self.use_overlap and not self.use_selected:
            return "V10_self_neigh_overlap"
        if self.use_neighbor and self.use_overlap and self.use_selected:
            return "V11_full_context"
        return "custom"


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ContextAwareTwoHeadMLP(nn.Module):
    """Context-aware Two-Head MLP for conditional utility estimation.

    Extends Phase 4 TwoHeadMLP with modular context encoders for:
    - Spatial neighborhood (KNN features)
    - Co-visibility overlap (projected IoU features)
    - Already-selected set (budget/group features)

    Each context group has its own encoder, outputs are fused via concatenation
    and passed through a shared fusion MLP, then split into quality and cost heads.
    """

    def __init__(self, config: Optional[Phase6ModelConfig] = None):
        super().__init__()
        self.config = config or Phase6ModelConfig()
        c = self.config

        # ─── Context Encoders ───
        self.self_encoder = nn.Sequential(
            nn.Linear(c.self_dim, c.self_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(c.dropout),
        )

        if c.use_neighbor:
            self.neighbor_encoder = nn.Sequential(
                nn.Linear(c.neighbor_dim, c.neighbor_hidden),
                nn.LeakyReLU(0.1),
                nn.Dropout(c.dropout),
            )

        if c.use_overlap:
            self.overlap_encoder = nn.Sequential(
                nn.Linear(c.overlap_dim, c.overlap_hidden),
                nn.LeakyReLU(0.1),
                nn.Dropout(c.dropout),
            )

        if c.use_selected:
            self.selected_encoder = nn.Sequential(
                nn.Linear(c.selected_dim, c.selected_hidden),
                nn.LeakyReLU(0.1),
                nn.Dropout(c.dropout),
            )

        # ─── Fusion MLP ───
        self.fusion = nn.Sequential(
            nn.Linear(c.fusion_input_dim, c.fusion_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(c.dropout),
            nn.Linear(c.fusion_hidden, c.head_hidden),
            nn.LeakyReLU(0.1),
        )

        # ─── Quality Head: predicts ΔQ (unconstrained sign) ───
        self.head_q = nn.Sequential(
            nn.Linear(c.head_hidden, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )

        # ─── Cost Head: predicts ΔT > 0 (Softplus ensures positivity) ───
        self.head_t = nn.Sequential(
            nn.Linear(c.head_hidden, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self.eps_cost = c.eps_cost

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: context-aware utility prediction.

        Args:
            x: Input feature tensor of shape (N, D) where D depends on enabled
               context groups. When all enabled: D = 32.
               Features must be ordered: [self(11), neighbor(8), overlap(5), selected(8)].

        Returns:
            Tuple of (delta_q, delta_t, utility), each shape (N,).
        """
        c = self.config

        # ─── Split input into context groups ───
        offset = 0
        x_self = x[:, offset:offset + c.self_dim]
        offset += c.self_dim

        h_self = self.self_encoder(x_self)
        encoded_parts = [h_self]

        if c.use_neighbor:
            x_neigh = x[:, offset:offset + c.neighbor_dim]
            offset += c.neighbor_dim
            h_neigh = self.neighbor_encoder(x_neigh)
            encoded_parts.append(h_neigh)

        if c.use_overlap:
            x_overlap = x[:, offset:offset + c.overlap_dim]
            offset += c.overlap_dim
            h_overlap = self.overlap_encoder(x_overlap)
            encoded_parts.append(h_overlap)

        if c.use_selected:
            x_sel = x[:, offset:offset + c.selected_dim]
            offset += c.selected_dim
            h_sel = self.selected_encoder(x_sel)
            encoded_parts.append(h_sel)

        # ─── Fusion ───
        h_fused = torch.cat(encoded_parts, dim=-1)
        h_shared = self.fusion(h_fused)

        # ─── Decoupled heads ───
        delta_q = self.head_q(h_shared).squeeze(-1)
        delta_t = self.head_t(h_shared).squeeze(-1) + self.eps_cost
        utility = delta_q / delta_t

        return delta_q, delta_t, utility

    def predict_utility(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method for inference: returns only utility."""
        _, _, u = self.forward(x)
        return u

    def get_input_dim(self) -> int:
        """Returns expected input dimension based on config."""
        return self.config.total_input_dim


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class Phase6Loss(nn.Module):
    """Composite loss for Phase 6 context-aware utility training.

    L = λ_Q · SmoothL1(ΔQ_hat, ΔQ*) + λ_C · SmoothL1(ΔT_hat, ΔT*) + λ_R · MarginRankingLoss

    The margin ranking loss encourages the model to correctly rank candidates
    by utility: if U*(i) > U*(j), then U_hat(i) should be > U_hat(j).
    """

    def __init__(
        self,
        lambda_q: float = 1.0,
        lambda_c: float = 0.5,
        lambda_r: float = 0.1,
        margin: float = 0.0,
    ):
        super().__init__()
        self.lambda_q = lambda_q
        self.lambda_c = lambda_c
        self.lambda_r = lambda_r
        self.margin = margin
        self.smooth_l1 = nn.SmoothL1Loss()
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(
        self,
        pred_q: torch.Tensor,
        pred_t: torch.Tensor,
        pred_u: torch.Tensor,
        target_q: torch.Tensor,
        target_t: torch.Tensor,
        target_u: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute composite loss.

        Args:
            pred_q: Predicted ΔQ (N,)
            pred_t: Predicted ΔT (N,)
            pred_u: Predicted utility (N,)
            target_q: Oracle ΔQ* (N,)
            target_t: Oracle ΔT* (N,)
            target_u: Oracle U* (N,)

        Returns:
            Dict with 'total', 'loss_q', 'loss_t', 'loss_r' keys.
        """
        loss_q = self.smooth_l1(pred_q, target_q)
        loss_t = self.smooth_l1(pred_t, target_t)

        # Margin ranking loss on utility pairs
        loss_r = torch.tensor(0.0, device=pred_u.device)
        N = pred_u.shape[0]
        if N >= 2 and self.lambda_r > 0:
            # Sample random pairs for efficiency
            n_pairs = min(N * 2, N * (N - 1) // 2)
            idx_i = torch.randint(0, N, (n_pairs,), device=pred_u.device)
            idx_j = torch.randint(0, N, (n_pairs,), device=pred_u.device)
            # Ensure i != j
            different = idx_i != idx_j
            idx_i = idx_i[different]
            idx_j = idx_j[different]

            if len(idx_i) > 0:
                u_i = pred_u[idx_i]
                u_j = pred_u[idx_j]
                target_sign = torch.sign(target_u[idx_i] - target_u[idx_j])
                # MarginRankingLoss expects target in {-1, 1}
                target_sign = target_sign.clamp(-1, 1)
                # Replace zeros with 1 (tie → prefer i)
                target_sign[target_sign == 0] = 1.0
                loss_r = self.margin_loss(u_i, u_j, target_sign)

        total = self.lambda_q * loss_q + self.lambda_c * loss_t + self.lambda_r * loss_r

        return {
            'total': total,
            'loss_q': loss_q.detach(),
            'loss_t': loss_t.detach(),
            'loss_r': loss_r.detach() if isinstance(loss_r, torch.Tensor) else torch.tensor(0.0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Frozen Predictor Wrapper (for Phase 6 inference/selection)
# ─────────────────────────────────────────────────────────────────────────────

class FrozenContextPredictor:
    """Frozen Phase 6 model wrapper for budget-constrained selection.

    Mirrors FrozenUtilityPredictor from Phase 5 but uses the context-aware model.
    Enforces eval mode, no gradients, and frozen normalization.
    """

    def __init__(
        self,
        checkpoint_path: str,
        normalizer_path: str,
        device: Optional[str] = None,
        config: Optional[Phase6ModelConfig] = None,
    ):
        """Load frozen Phase 6 model and normalizer.

        Args:
            checkpoint_path: Path to model checkpoint (.pt).
            normalizer_path: Path to Phase 6 normalization JSON.
            device: Device for inference.
            config: Model config (overridden by checkpoint if available).
        """
        import time
        from .phase6_dataset import Phase6FeatureNormalizer

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load normalizer
        self.normalizer = Phase6FeatureNormalizer.load_json(normalizer_path)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Reconstruct config from checkpoint
        if 'config' in ckpt and config is None:
            config = Phase6ModelConfig(**ckpt['config'])
        elif config is None:
            config = Phase6ModelConfig()

        self.config = config
        self.model = ContextAwareTwoHeadMLP(config).to(self.device)

        state_dict = ckpt.get('model_state', ckpt.get('state_dict'))
        self.model.load_state_dict(state_dict)

        # Freeze
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.metadata = ckpt.get('metadata', {})

    def predict(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict ΔQ, ΔT, utility for a batch of 32-dim context features.

        Args:
            features: (N, 32) tensor of Phase 6 feature vectors.

        Returns:
            Dict with 'delta_q', 'delta_t', 'utility' tensors, each (N,).
        """
        import time

        if features.ndim == 1:
            features = features.unsqueeze(0)

        # Normalize
        norm_feat = self.normalizer.transform(features)
        if isinstance(norm_feat, torch.Tensor):
            input_t = norm_feat.to(self.device)
        else:
            input_t = torch.tensor(norm_feat, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            delta_q, delta_t, utility = self.model(input_t)

        return {
            'delta_q': delta_q,
            'delta_t': delta_t,
            'utility': utility,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Variant Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_ablation_variant(variant: str) -> Phase6ModelConfig:
    """Create model config for a specific ablation variant.

    Args:
        variant: One of 'V8', 'V9', 'V10', 'V11'.

    Returns:
        Phase6ModelConfig with appropriate context groups enabled.
    """
    configs = {
        'V8': Phase6ModelConfig(use_neighbor=False, use_overlap=False, use_selected=False),
        'V9': Phase6ModelConfig(use_neighbor=True, use_overlap=False, use_selected=False),
        'V10': Phase6ModelConfig(use_neighbor=True, use_overlap=True, use_selected=False),
        'V11': Phase6ModelConfig(use_neighbor=True, use_overlap=True, use_selected=True),
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    return configs[variant]
