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
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

from .phase6_context import (
    PHASE6_FEATURE_DIM,
    SELF_SLICE,
    NEIGHBOR_SLICE,
    OVERLAP_SLICE,
    SELECTED_SLICE,
)
from research.utility_models import TwoHeadMLP


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

    Enhancements (Phase 6 Reform):
      1. Meaningful pair filtering: excludes pairs where |U_i* - U_j*| <= tau
         (tau = 0.05 * std(U*)). Completely eliminates the artificial tie-to-preference hack!
      2. Listwise KL loss option (lambda_list) across context groups:
         P* = softmax(U* / tau_list), P_hat = softmax(U_hat / tau_list), L_list = KL(P* || P_hat)
      3. SmoothL1 on scaled quality and cost components.
    """

    def __init__(
        self,
        lambda_q: float = 2.0,
        lambda_c: float = 0.5,
        lambda_r: float = 2.0,
        lambda_list: float = 0.5,
        lambda_res: float = 1.0,
        lambda_zero: float = 0.5,
        margin: float = 0.0,
        scale_q: float = 1e4,
        scale_t: float = 0.05,
        scale_u: float = 1e4,
        tau_factor: float = 0.05,
        tau_list: float = 1.0,
        require_group_ids: bool = True,
    ):
        super().__init__()
        self.lambda_q = lambda_q
        self.lambda_c = lambda_c
        self.lambda_r = lambda_r
        self.lambda_list = lambda_list
        self.lambda_res = lambda_res
        self.lambda_zero = lambda_zero
        self.margin = margin
        self.scale_q = scale_q
        self.scale_t = scale_t
        self.scale_u = scale_u
        self.tau_factor = tau_factor
        self.tau_list = tau_list
        self.require_group_ids = require_group_ids
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
        pred_r: Optional[torch.Tensor] = None,
        target_r: Optional[torch.Tensor] = None,
        is_empty: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_q = self.smooth_l1(pred_q * self.scale_q, target_q * self.scale_q)
        loss_t = self.smooth_l1(pred_t * self.scale_t, target_t * self.scale_t)

        device = pred_u.device
        N = pred_u.shape[0]

        # ─── Group-Aware Ranking & Listwise Losses (P0.1, Sửa số 16) ───
        # Strictly enforce context group partitioning g = (scene, frame, S_t)
        if group_ids is None:
            if self.require_group_ids:
                raise ValueError(
                    "Phase6Loss requires group_ids to partition ranking and listwise losses strictly by candidate context group. "
                    "Set require_group_ids=False to allow un-grouped computation if explicitly intended."
                )
            group_ids = torch.zeros(N, dtype=torch.long, device=device)
        else:
            group_ids = group_ids.to(device)

        unique_groups = torch.unique(group_ids)

        loss_r_list = []
        loss_list_list = []
        total_pairs_eval = 0
        retained_pairs_eval = 0
        n_groups_used_pairwise = 0
        n_groups_used_listwise = 0

        for g in unique_groups:
            g_mask = (group_ids == g)
            g_indices = torch.nonzero(g_mask, as_tuple=False).squeeze(-1)
            g_n = len(g_indices)

            if g_n < 2:
                continue

            g_pred_u = pred_u[g_indices]
            g_target_u = target_u[g_indices]

            # 1. Pairwise ranking within group g (deterministic combinations if <= 30)
            if self.lambda_r > 0:
                u_std = g_target_u.std()
                tau = float(self.tau_factor * u_std) if u_std > 1e-8 else 1e-7

                if g_n <= 30:
                    comb_pairs = torch.combinations(torch.arange(g_n, device=device), r=2)
                    sub_i = comb_pairs[:, 0]
                    sub_j = comb_pairs[:, 1]
                else:
                    n_pairs = min(g_n * 6, g_n * (g_n - 1) // 2)
                    rng = torch.Generator(device=device)
                    rng.manual_seed(42 + int(g.item()))
                    sub_i = torch.randint(0, g_n, (n_pairs,), generator=rng, device=device)
                    sub_j = torch.randint(0, g_n, (n_pairs,), generator=rng, device=device)

                diff_targets = g_target_u[sub_i] - g_target_u[sub_j]
                meaningful = (sub_i != sub_j) & (torch.abs(diff_targets) > tau)

                total_pairs_eval += len(sub_i)
                retained_pairs_eval += int(meaningful.sum().item())

                if meaningful.any():
                    n_groups_used_pairwise += 1
                    valid_i = sub_i[meaningful]
                    valid_j = sub_j[meaningful]
                    u_i = g_pred_u[valid_i] * self.scale_u
                    u_j = g_pred_u[valid_j] * self.scale_u
                    target_sign = torch.sign(g_target_u[valid_i] - g_target_u[valid_j]).clamp(-1, 1)
                    loss_r_list.append(self.margin_loss(u_i, u_j, target_sign))

            # 2. Listwise KL ranking within group g (P0.1, Sửa số 20)
            if self.lambda_list > 0 and g_n >= 3:
                n_groups_used_listwise += 1
                t_std = max(float(g_target_u.detach().std() + 1e-6), 1e-6)
                p_target = torch.softmax(g_target_u / t_std / self.tau_list, dim=0)
                u_pred_std = max(float(g_pred_u.detach().std() + 1e-6), 1e-6)
                log_p_pred = torch.log_softmax(g_pred_u / u_pred_std / self.tau_list, dim=0)
                kl = torch.sum(p_target * (torch.log(p_target + 1e-9) - log_p_pred))
                loss_list_list.append(kl)

        loss_r = torch.mean(torch.stack(loss_r_list)) if loss_r_list else torch.tensor(0.0, device=device)
        loss_list = torch.mean(torch.stack(loss_list_list)) if loss_list_list else torch.tensor(0.0, device=device)

        fraction_pairs_retained = (retained_pairs_eval / total_pairs_eval) if total_pairs_eval > 0 else 1.0
        n_groups_skipped = len(unique_groups) - n_groups_used_pairwise

        # ─── Residual Target Loss (P0.4) ───
        loss_res = torch.tensor(0.0, device=device)
        if pred_r is not None and target_r is not None and self.lambda_res > 0:
            loss_res = self.smooth_l1(pred_r * self.scale_u, target_r * self.scale_u)

        # ─── Empty Context Zero Regularization (P0.4): r(∅) ≈ 0 ───
        loss_zero = torch.tensor(0.0, device=device)
        if pred_r is not None and is_empty is not None and self.lambda_zero > 0:
            empty_mask = (is_empty > 0.5)
            if empty_mask.any():
                loss_zero = self.smooth_l1(pred_r[empty_mask] * self.scale_u, torch.zeros_like(pred_r[empty_mask]))

        total = (
            self.lambda_q * loss_q
            + self.lambda_c * loss_t
            + self.lambda_r * loss_r
            + self.lambda_list * loss_list
            + self.lambda_res * loss_res
            + self.lambda_zero * loss_zero
        )

        return {
            'total': total,
            'loss_q': loss_q.detach(),
            'loss_t': loss_t.detach(),
            'loss_r': loss_r.detach() if isinstance(loss_r, torch.Tensor) else torch.tensor(0.0),
            'loss_list': loss_list.detach() if isinstance(loss_list, torch.Tensor) else torch.tensor(0.0),
            'loss_res': loss_res.detach() if isinstance(loss_res, torch.Tensor) else torch.tensor(0.0),
            'loss_zero': loss_zero.detach() if isinstance(loss_zero, torch.Tensor) else torch.tensor(0.0),
            'n_total_groups': len(unique_groups),
            'n_groups_used_pairwise': n_groups_used_pairwise,
            'n_groups_used_listwise': n_groups_used_listwise,
            'n_groups_skipped': n_groups_skipped,
            'fraction_pairs_retained': fraction_pairs_retained,
        }



class ResidualContextModel(nn.Module):
    """Contextual Residual Utility Model for Phase 6.

    Architecture:
      Baseline:    U_hat_P4(i) = f_P4(s_i)           [frozen or pretrained Phase 4]
      Correction:  r_i = g(s_i, N_i, O_i, S_t)       [learnable context residual]
      Final:       U_hat_P6(i|S) = U_hat_P4(i) + r_i

    Guarantees:
      - Inherits Phase 4's strong representation (rho_P4 = 0.3178).
      - Directly learns the interaction correction r_i* = U*(i|S) - U*(i|∅).
      - When context is empty (S = ∅), r_i converges to 0, preventing degradation.
    """

    def __init__(
        self,
        config: Optional[Phase6ModelConfig] = None,
        p4_model: Optional[nn.Module] = None,
        eps_cost: float = 0.001,
    ):
        super().__init__()
        self.config = config or Phase6ModelConfig()
        c = self.config
        self.eps_cost = eps_cost

        # Phase 4 backbone (11 self features)
        if p4_model is not None:
            self.p4_model = p4_model
        else:
            self.p4_model = TwoHeadMLP(in_features=c.self_dim, hidden_dim=64, eps_cost=eps_cost)

        # Strictly freeze Phase 4 backbone (P0.2)
        self.p4_model.eval()
        for p in self.p4_model.parameters():
            p.requires_grad = False

        # Context Encoders
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

        # Context Fusion MLP
        self.context_fusion = nn.Sequential(
            nn.Linear(c.fusion_input_dim, c.fusion_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(c.dropout),
            nn.Linear(c.fusion_hidden, c.head_hidden),
            nn.LeakyReLU(0.1),
        )

        # Residual Head: predicts residual utility correction r_i
        self.head_residual_u = nn.Sequential(
            nn.Linear(c.head_hidden, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )

        # Zero-initialize residual projection so initial state is exact P4 baseline: r_i(x) = 0
        nn.init.zeros_(self.head_residual_u[-1].weight)
        nn.init.zeros_(self.head_residual_u[-1].bias)

    def context_parameters(self) -> List[nn.Parameter]:
        """Returns only trainable context parameters, strictly excluding frozen P4 (P0.2)."""
        return [p for n, p in self.named_parameters() if not n.startswith("p4_model") and p.requires_grad]

    def forward(
        self,
        x: torch.Tensor,
        return_residual: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        c = self.config

        # 1. Compute Phase 4 baseline from self features x[:, :c.self_dim]
        # P4 backbone is strictly executed with no gradients (P0.2)
        with torch.no_grad():
            offset = 0
            self_x = x[:, offset:offset + c.self_dim]
            offset += c.self_dim
            p4_dq, p4_dt, p4_u = self.p4_model(self_x)

        # 2. Compute Context Features
        parts = [self.self_encoder(self_x)]
        if c.use_neighbor:
            neigh_x = x[:, offset:offset + c.neighbor_dim]
            offset += c.neighbor_dim
            parts.append(self.neighbor_encoder(neigh_x))
        if c.use_overlap:
            overlap_x = x[:, offset:offset + c.overlap_dim]
            offset += c.overlap_dim
            parts.append(self.overlap_encoder(overlap_x))
        if c.use_selected:
            sel_x = x[:, offset:offset + c.selected_dim]
            offset += c.selected_dim
            parts.append(self.selected_encoder(sel_x))

        fused = torch.cat(parts, dim=-1)
        h_ctx = self.context_fusion(fused)

        # 3. Residual prediction
        r_u = self.head_residual_u(h_ctx).squeeze(-1)

        # 4. Additive residual combination with mathematical consistency (P0.3):
        # U_P6 = U_P4 + r_U
        # T_P6 = T_P4
        # Q_P6 = U_P6 * T_P6
        final_u = p4_u + r_u
        final_dt = p4_dt
        final_dq = final_u * final_dt

        if return_residual:
            return final_dq, final_dt, final_u, r_u
        return final_dq, final_dt, final_u


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
        arch = ckpt.get('architecture', ckpt.get('metadata', {}).get('model_type', 'direct'))
        is_residual = arch in ('residual', 'residual_context') or ckpt.get('metadata', {}).get('model_type') == 'residual_context'

        if is_residual:
            p4_net = None
            p4_ckpt = ckpt.get('p4_checkpoint') or ckpt.get('metadata', {}).get('phase4_checkpoint')
            if p4_ckpt and os.path.exists(p4_ckpt):
                from research.utility_models import TwoHeadMLP
                p4_net = TwoHeadMLP(in_features=config.self_dim)
                p4_data = torch.load(p4_ckpt, map_location="cpu", weights_only=False)
                p4_net.load_state_dict(p4_data.get("model_state", p4_data))
                p4_net.eval()
                for p in p4_net.parameters():
                    p.requires_grad = False
            self.model = ResidualContextModel(config, p4_model=p4_net).to(self.device)
        else:
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
        'self_only': Phase6ModelConfig(use_neighbor=False, use_overlap=False, use_selected=False),

        'V9': Phase6ModelConfig(use_neighbor=True, use_overlap=False, use_selected=False),
        'self_neighbor': Phase6ModelConfig(use_neighbor=True, use_overlap=False, use_selected=False),

        'self_overlap': Phase6ModelConfig(use_neighbor=False, use_overlap=True, use_selected=False),
        'self_selected': Phase6ModelConfig(use_neighbor=False, use_overlap=False, use_selected=True),

        'V10': Phase6ModelConfig(use_neighbor=True, use_overlap=True, use_selected=False),
        'self_neighbor_overlap': Phase6ModelConfig(use_neighbor=True, use_overlap=True, use_selected=False),

        'self_neighbor_selected': Phase6ModelConfig(use_neighbor=True, use_overlap=False, use_selected=True),
        'self_overlap_selected': Phase6ModelConfig(use_neighbor=False, use_overlap=True, use_selected=True),

        'V11': Phase6ModelConfig(use_neighbor=True, use_overlap=True, use_selected=True),
        'all_features': Phase6ModelConfig(use_neighbor=True, use_overlap=True, use_selected=True),
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    return configs[variant]
