r"""Phase 10: End-to-End Adaptive 3DGS Runtime Engine.

Implements the complete online closed-loop reconstruction pipeline (Phase 10A & 10B):
    S_t -> X_t -> \hat{X}_t -> \hat{U}_t -> A_t -> S_{t+1}

Core Modules:
    1. Phase10ModelBundle:
       Loads and manages the frozen TwoHeadMLP model and B2 OnlineEMANormalizer.
       Guarantees zero model parameter updates and strictly online EMA covariate adaptation.

    2. Feature Extractor:
       Extracts canonical 11-dimensional Gaussian state representation strictly from
       pre-intervention observable state S_t.

    3. Phase10Selector:
       Executes budget-constrained subset selection under scheduler budget B = 15.0 ms
       for NO_OP, ERROR_ONLY, OURS (B2), and FULL.

    4. StateStore Closed-Loop Updater:
       Maintains continuous Gaussian lifecycle and state signals across frames.

    5. Dual Budget Accounting:
       Tracks predicted cost, scheduled cost, actual wall-clock optimization time,
       and total frame latency concurrently.
"""
import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from research.phase9_protocol import CANONICAL_FEATURE_SCHEMA
from research.phase9_features import transform_features_array
from research.phase9b_normalization import OnlineEMANormalizer
from research.utility_models import TwoHeadMLP
from research.utility_training import UtilityModelTrainer
from research.state_store import GaussianStateStore
from research.pipeline import OnlineReconstructionPipeline
from research.phase10_protocol import (
    get_repo_root,
    get_checkpoint_path_for_seed,
    get_normalizer_path_for_seed,
    DEFAULT_BUDGET_MS,
    SAFETY_FACTOR,
    B2_BETA,
    MODEL_IN_FEATURES,
    MODEL_HIDDEN_DIM,
    MODEL_EPS_COST,
    BASE_REPRESENTATION,
)
from datasets.tum_dataset import TUMDataset


class Phase10ModelBundle:
    """Manages frozen TwoHeadMLP and OnlineEMANormalizer for online deployment."""

    def __init__(
        self,
        seed: int = 42,
        checkpoint_path: Optional[Union[str, Path]] = None,
        normalizer_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        beta: float = B2_BETA,
    ):
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.beta = beta

        # 1. Initialize model architecture (strictly frozen)
        self.model = TwoHeadMLP(
            in_features=MODEL_IN_FEATURES,
            hidden_dim=MODEL_HIDDEN_DIM,
            eps_cost=MODEL_EPS_COST,
        ).to(self.device)

        # 2. Load checkpoint weights
        ckpt_p = Path(checkpoint_path) if checkpoint_path else get_checkpoint_path_for_seed(seed)
        UtilityModelTrainer.load_checkpoint(self.model, str(ckpt_p))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # 3. Load online normalizer initialized from train reference statistics
        norm_p = Path(normalizer_path) if normalizer_path else get_normalizer_path_for_seed(seed)
        self.normalizer = OnlineEMANormalizer.load_json(str(norm_p))
        # Enforce protocol beta and reset to train initial reference statistics
        self.normalizer.beta = self.beta
        self.normalizer.reset()

    def reset_normalizer(self) -> None:
        """Reset normalizer state to initial train reference statistics."""
        self.normalizer.reset()

    def predict(
        self,
        X_raw: np.ndarray,
        frame_id: Optional[int] = None,
        update_normalizer: bool = True,
    ) -> Dict[str, Any]:
        r"""Runs the full A1 + B2 + TwoHeadMLP inference pipeline.

        Steps:
            1. Scale-invariant A1 transformation: X_raw -> Z_a1 (geometry_relative).
            2. Online covariate adaptation: Z_a1 -> \hat{X} (OnlineEMANormalizer update & transform).
            3. Model forward pass: \hat{X} -> (\hat{\Delta Q}, \hat{\Delta T}, \hat{U}).

        Args:
            X_raw: [N, 11] raw observable feature matrix.
            frame_id: video frame index for tracking temporal dynamics.
            update_normalizer: whether to update online statistics with current frame.

        Returns:
            Dict containing predicted_utility, predicted_cost, predicted_quality, and norm_metrics.
        """
        N = len(X_raw)
        if N == 0:
            return {
                "predicted_utility": np.empty(0, dtype=np.float32),
                "predicted_cost": np.empty(0, dtype=np.float32),
                "predicted_quality": np.empty(0, dtype=np.float32),
                "norm_metrics": {},
            }

        # 1. Scale-invariant A1 transformation (geometry_relative)
        Z_a1 = transform_features_array(X_raw, variant=BASE_REPRESENTATION)

        # 2. B2 Online EMA normalizer update & transform
        if update_normalizer:
            Z_norm, norm_metrics = self.normalizer.update_and_transform_frame(Z_a1, frame_id=frame_id)
        else:
            Z_norm = self.normalizer.transform_static(Z_a1)
            norm_metrics = {}

        # 3. TwoHeadMLP inference (strictly no_grad)
        Z_tensor = torch.tensor(Z_norm, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred_q, pred_t, pred_u = self.model(Z_tensor)

        return {
            "predicted_utility": pred_u.detach().cpu().numpy().astype(np.float32),
            "predicted_cost": pred_t.detach().cpu().numpy().astype(np.float32),
            "predicted_quality": pred_q.detach().cpu().numpy().astype(np.float32),
            "norm_metrics": norm_metrics,
        }


def extract_online_features(
    pipeline: OnlineReconstructionPipeline,
    N: int,
) -> np.ndarray:
    """Extracts canonical 11-dimensional feature vectors strictly from pre-intervention observable state S_t.

    Schema:
        0: rgb_error          (running color attribution error)
        1: depth_error        (running depth attribution error)
        2: gradient_norm      (attribution mass * (rgb_error + depth_error))
        3: visibility_count   (running visible frame count)
        4: influence_mass     (pixel influence mass / screen area)
        5: position_drift     (state_store position drift ||mu_t - mu_{t-1}||)
        6: residual_drift_ema (state_store residual error drift EMA)
        7: uncertainty_var    (1.0 - confidence score)
        8: projected_area     (projected screen area)
        9: update_frequency   (state_store update count / frame count)
        10: age               (state_store current_frame - creation_frame)

    Returns:
        X: [N, 11] float32 numpy array. Guaranteed finite (no NaN, no Inf).
    """
    model = pipeline.gaussian_model
    store: Optional[GaussianStateStore] = getattr(model, "state_store", None)
    est = pipeline.importance_estimator
    device = pipeline.device

    # Error attribution signals
    color_err = est._running_color_error[:N] if est._running_color_error is not None else torch.zeros(N, device=device)
    depth_err = est._running_depth_error[:N] if est._running_depth_error is not None else torch.zeros(N, device=device)
    vis_count = est._visibility_count[:N] if est._visibility_count is not None else torch.zeros(N, device=device)

    screen_areas = getattr(est, "_screen_areas", None)
    if screen_areas is not None and screen_areas.shape[0] >= N:
        proj_area = screen_areas[:N]
    else:
        proj_area = torch.ones(N, device=device)

    inf_mass = getattr(est, "_influence_weights", None)
    if inf_mass is not None and inf_mass.shape[0] >= N:
        inf_mass_t = inf_mass[:N]
    else:
        inf_mass_t = proj_area

    grad_norm = inf_mass_t * (color_err + depth_err)

    # Persistent StateStore signals
    if store is not None and store.num_gaussians >= N:
        pos_drift = store.position_drift[:N]
        res_drift = store.residual_drift_ema[:N]
        ages = store.ages[:N].float()
        update_freq = store.get_update_frequency(pipeline.frame_count)[:N]
    else:
        pos_drift = torch.zeros(N, device=device)
        res_drift = torch.zeros(N, device=device)
        ages = torch.ones(N, device=device)
        update_freq = torch.full((N,), 0.5, device=device)

    # Uncertainty from model confidence
    if hasattr(model, "_confidence") and model._confidence is not None and model._confidence.shape[0] >= N:
        conf = model._confidence[:N].squeeze(-1)
        unc_var = (1.0 - conf).clamp(0.0, 1.0)
    else:
        unc_var = torch.full((N,), 0.5, device=device)

    mat = torch.stack([
        color_err, depth_err, grad_norm, vis_count.float(), inf_mass_t,
        pos_drift, res_drift, unc_var, proj_area, update_freq, ages,
    ], dim=-1)

    mat_np = mat.detach().cpu().numpy().astype(np.float32)

    # Strict finiteness enforcement
    if not np.all(np.isfinite(mat_np)):
        mat_np = np.nan_to_num(mat_np, nan=0.0, posinf=1.0, neginf=0.0)

    return mat_np


def update_statestore_closed_loop(
    pipeline: OnlineReconstructionPipeline,
    frame_idx: int,
    optimize_mask: torch.Tensor,
) -> None:
    """Synchronizes GaussianStateStore with current frame state, closing S_t -> S_{t+1}.

    Args:
        pipeline: active reconstruction pipeline instance.
        frame_idx: current video frame index.
        optimize_mask: [N] boolean tensor of Gaussians selected and optimized in frame t.
    """
    model = pipeline.gaussian_model
    store: Optional[GaussianStateStore] = getattr(model, "state_store", None)
    if store is None:
        return

    N = model.num_gaussians
    if N == 0:
        return

    # Ensure store size matches model size
    if store.num_gaussians < N:
        diff = N - store.num_gaussians
        store.create(diff, frame_idx=frame_idx)

    est = pipeline.importance_estimator
    color_err = est._running_color_error[:N] if est._running_color_error is not None else None
    depth_err = est._running_depth_error[:N] if est._running_depth_error is not None else None
    vis_count = est._visibility_count[:N] if est._visibility_count is not None else None
    inf_mass = getattr(est, "_influence_weights", None)
    inf_t = inf_mass[:N] if inf_mass is not None and inf_mass.shape[0] >= N else None

    # Gradient norm approximation
    grad_norm = None
    if inf_t is not None and color_err is not None and depth_err is not None:
        grad_norm = inf_t * (color_err + depth_err)

    # Uncertainty
    unc = None
    if hasattr(model, "_confidence") and model._confidence is not None and model._confidence.shape[0] >= N:
        unc = (1.0 - model._confidence[:N].squeeze(-1)).clamp(0.0, 1.0)

    store.update_frame(
        frame_idx=frame_idx,
        rgb_errors=color_err,
        depth_errors=depth_err,
        influence_scores=inf_t,
        visibility_count=vis_count,
        gradient_norms=grad_norm,
        uncertainty=unc,
        optimized_mask=optimize_mask[:N],
        positions=model.positions[:N].detach(),
        ema_decay=0.90,
    )


class Phase10Selector:
    """Selects candidate Gaussians subject to scheduler budget B for Phase 10 policies."""

    def __init__(
        self,
        model_bundle: Phase10ModelBundle,
        budget_ms: float = DEFAULT_BUDGET_MS,
        safety_factor: float = SAFETY_FACTOR,
        device: str = "cuda",
    ):
        self.model_bundle = model_bundle
        self.budget_ms = budget_ms
        self.safety_factor = safety_factor
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.prev_selected_indices: Optional[set] = None

    def select(
        self,
        pipeline: OnlineReconstructionPipeline,
        policy: str,
        frame_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Executes budget-aware subset selection for the given policy.

        Returns:
            mask: [N] boolean tensor on pipeline device.
            diagnostics: dictionary with predicted_cost, scheduled_cost, overlap, etc.
        """
        N = pipeline.gaussian_model.num_gaussians
        mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        pol = policy.lower()

        if N == 0 or pol in ("no_op", "noop"):
            diag = {
                "policy": pol,
                "n_gaussians": N,
                "n_selected": 0,
                "fraction_selected": 0.0,
                "predicted_cost": 0.0,
                "scheduled_cost": 0.0,
                "budget_ms": self.budget_ms,
                "jaccard_overlap": 0.0,
                "rejected_negative_count": 0,
                "selection_time_ms": 0.0,
                "norm_metrics": {},
            }
            self.prev_selected_indices = set()
            return mask, diag

        if pol == "full":
            mask = torch.ones(N, dtype=torch.bool, device=self.device)
            current_set = set(range(N))
            overlap = 1.0 if self.prev_selected_indices is not None else 0.0
            self.prev_selected_indices = current_set
            diag = {
                "policy": "full",
                "n_gaussians": N,
                "n_selected": N,
                "fraction_selected": 1.0,
                "predicted_cost": float(N * 1.5),
                "scheduled_cost": float(N * 1.5 * self.safety_factor),
                "budget_ms": self.budget_ms,
                "jaccard_overlap": overlap,
                "rejected_negative_count": 0,
                "selection_time_ms": 0.0,
                "norm_metrics": {},
            }
            return mask, diag

        t_sel_start = time.perf_counter()

        # 1. Extract canonical 11-D features from state S_t
        X = extract_online_features(pipeline, N)

        # 2. Predict cost & utility using frozen model bundle
        # Both ERROR_ONLY and OURS share the exact same cost head for fair packing
        preds = self.model_bundle.predict(
            X_raw=X,
            frame_id=frame_idx,
            update_normalizer=(pol == "ours"),  # Only B2 updates its online normalizer
        )
        pred_costs = preds["predicted_cost"]
        pred_utils = preds["predicted_utility"]
        norm_metrics = preds.get("norm_metrics", {})

        # Unified packing costs
        scheduled_costs = pred_costs * self.safety_factor

        selected_indices: List[int] = []
        cur_scheduled_cost = 0.0
        rejected_neg = 0

        if pol in ("error_only", "error"):
            # Score = rgb_error + depth_error
            err_scores = X[:, 0] + X[:, 1]
            order = np.argsort(-err_scores)
            for idx in order:
                c = float(scheduled_costs[idx])
                if cur_scheduled_cost + c <= self.budget_ms + 1e-7:
                    selected_indices.append(int(idx))
                    cur_scheduled_cost += c

        elif pol in ("ours", "b2", "learned_utility"):
            # Rank by TwoHeadMLP utility; reject non-positive utility
            order = np.argsort(-pred_utils)
            for idx in order:
                u_val = float(pred_utils[idx])
                if u_val <= 0.0:
                    rejected_neg += 1
                    continue
                c = float(scheduled_costs[idx])
                if cur_scheduled_cost + c <= self.budget_ms + 1e-7:
                    selected_indices.append(int(idx))
                    cur_scheduled_cost += c
        else:
            raise ValueError(f"Unknown selection policy '{policy}'")

        # Construct boolean mask
        if selected_indices:
            sel_arr = np.array(selected_indices, dtype=np.int64)
            mask[sel_arr] = True

        sel_time_ms = (time.perf_counter() - t_sel_start) * 1000.0

        # Compute selection overlap with previous frame
        current_set = set(selected_indices)
        if self.prev_selected_indices is not None and (len(current_set) + len(self.prev_selected_indices)) > 0:
            inter = len(current_set.intersection(self.prev_selected_indices))
            union = len(current_set.union(self.prev_selected_indices))
            jaccard = float(inter / union) if union > 0 else 0.0
        else:
            jaccard = 0.0
        self.prev_selected_indices = current_set

        k_count = len(selected_indices)
        pred_cost_sum = float(np.sum(pred_costs[selected_indices])) if k_count > 0 else 0.0

        diag = {
            "policy": pol,
            "n_gaussians": N,
            "n_selected": k_count,
            "fraction_selected": float(k_count / max(N, 1)),
            "predicted_cost": pred_cost_sum,
            "scheduled_cost": cur_scheduled_cost,
            "budget_ms": self.budget_ms,
            "jaccard_overlap": jaccard,
            "rejected_negative_count": rejected_neg,
            "selection_time_ms": sel_time_ms,
            "norm_metrics": norm_metrics,
        }
        return mask, diag


def load_phase10_sequence(
    scene_name: str = "tum_fr2_xyz",
    n_frames: int = 30,
    H: int = 240,
    W: int = 320,
    device: str = "cuda",
) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
    """Loads scaled TUM sequence strictly aligned with protocol resolution.

    Ensures f_x', f_y', c_x', c_y' camera intrinsic scaling.
    """
    repo = get_repo_root()
    if scene_name == "tum_fr2_xyz":
        data_path = str(repo / "datasets/TUM/rgbd_dataset_freiburg2_xyz")
        cam = "freiburg2"
    elif scene_name == "tum_fr1_desk":
        data_path = str(repo / "datasets/TUM/rgbd_dataset_freiburg1_desk")
        cam = "freiburg1"
    else:
        raise ValueError(f"Unknown scene '{scene_name}'")

    dataset = TUMDataset(data_path, max_frames=n_frames, camera=cam)
    frames = []
    orig_W, orig_H = 640.0, 480.0
    scale_x = W / orig_W
    scale_y = H / orig_H

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    intrinsics = torch.tensor([
        [dataset.fx * scale_x, 0.0, dataset.cx * scale_x],
        [0.0, dataset.fy * scale_y, dataset.cy * scale_y],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=dev)

    for i in range(min(n_frames, len(dataset))):
        item = dataset[i]
        rgb = item["rgb"].unsqueeze(0).permute(0, 3, 1, 2)
        depth = item["depth"].unsqueeze(0).unsqueeze(0)

        rgb_scaled = torch.nn.functional.interpolate(
            rgb, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        depth_scaled = torch.nn.functional.interpolate(
            depth, size=(H, W), mode="nearest"
        ).squeeze(0).squeeze(0)

        frames.append({
            "frame_id": i,
            "rgb": rgb_scaled.to(dev),
            "depth": depth_scaled.to(dev),
            "pose": item["pose"].to(dev),
        })

    return frames, intrinsics


def get_pipeline_config(
    policy: str,
    budget_ms: float = DEFAULT_BUDGET_MS,
    seed: int = 42,
    W: int = 320,
    H: int = 240,
) -> Dict[str, Any]:
    """Builds pipeline configuration for Phase 10 execution."""
    is_full = (policy.lower() == "full")
    return {
        "seed": seed,
        "gaussian": {
            "sh_degree": 0,
            "initial_opacity": 0.5,
            "max_gaussians": 30000,
            "initial_scale": 0.02,
        },
        "rendering": {
            "tile_size": 16,
            "image_width": W,
            "image_height": H,
            "use_surface_aware_depth": True,
            "attribution_top_k": 4,
        },
        "scheduler": {
            "gpu_budget_ms": 500.0 if is_full else budget_ms,
            "policy": policy,
            "cost_per_gaussian_us": 2.0,
        },
        "densification": {
            "max_new_per_frame": 80,
            "strategy": "importance",
            "use_adaptive_thresholds": True,
        },
    }
