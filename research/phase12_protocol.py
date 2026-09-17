#!/usr/bin/env python3
r"""Phase 12 Protocol: Paper Evidence & Benchmark Expansion.

Formal protocol defining the restructured multi-pillar evaluation suite, canonicalized
cost definitions, two-stage decision model, global context representations, and controlled
ablation specifications for publication.

Restructured Phase 12 Structure:
    12A: Scientific Audit (immutability, invariant verification, zero leakage, metric definitions)
    12B: Baseline ML Model (frozen TwoHeadMLP Phase 4/10 reference)
    12C: Positive-Utility Head (P(U* > 0) classification to eliminate negative-utility updates)
    12D: Global Scene Context (12-dim frame-level statistics without cross-attention overhead)
    12E: Decision-Aware Scheduler (probability-aware ranking, thresholded knapsack, oracle filter)
    12F: Controlled Ablation (M0 Error, M1 GradNorm, M2 Current, M3 +PosHead, M4 +PosHead+Global)
    12G: Final Paper Evidence (tables, Wilcoxon tests, effect sizes, audited systems profiles)

Core Cost Accounting Invariant:
    - Modeled Optimization Kernel Budget (B_opt = 15.0 ms):
        Scheduler packs Gaussians such that \sum_{i \in S} C_i^{kernel} \le B_opt.
    - Total System Frame Latency (T_frame ~ 6000-7000 ms):
        End-to-end wall-clock time encompassing tracking, rendering, error attribution,
        densification, scheduling, CUDA backward passes, and state synchronization.
        T_frame is disclosed separately and NEVER conflated with B_opt.
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Repository Root
REPO_ROOT = Path(__file__).resolve().parent.parent

# Output Directory for Phase 12 Paper Evidence
PHASE12_OUTPUT_DIR = REPO_ROOT / "results" / "phase12_paper_evidence"

# Protocol Invariants & Experimental Setup
SEEDS: List[int] = [42, 43, 44, 45, 46]

# Canonical Safety Factor: 1.10 is canonical across all runtime and scheduler code.
# (1.05 was used in early draft text as an informal margin; 1.10 is the strictly enforced protocol constant).
SAFETY_FACTOR: float = 1.10

# Dual Budget Accounting Definitions
DEFAULT_BUDGET_MS: float = 15.0          # B_opt: Optimization kernel compute budget
DEFAULT_KERNEL_BUDGET_MS: float = 15.0   # Alias for clarity: C_i^{kernel} knapsack limit
DEFAULT_TRAJECTORY_FRAMES: int = 150  # was 30 — see research/phase10_protocol.py for rationale
DEFAULT_IMAGE_WIDTH: int = 320
DEFAULT_IMAGE_HEIGHT: int = 240

# Canonical Systems Latency & Resource Reference Disclosures
# (Resolves previous text discrepancies between mean vs p95 and allocated vs reserved VRAM)
SYSTEMS_REFERENCE_PROFILE: Dict[str, Any] = {
    "t_scheduler_mean_ms": 2.54,
    "t_scheduler_p95_ms": 4.32,
    "vram_allocated_min_mb": 50.8,
    "vram_allocated_max_mb": 60.9,
    "vram_peak_allocated_mb": 70.0,
    "vram_reserved_cuda_mb": 536.6,
    "t_infer_twohead_ms": 0.456,
}

# Datasets
TRAIN_SCENE: str = "tum_fr1_desk"
VAL_SCENE: str = "tum_fr1_desk"

# 4 Unseen Zero-Shot Test Sequences spanning FR1, FR2, FR3 sensors:
ZERO_SHOT_SCENES: Dict[str, Dict[str, Any]] = {
    "tum_fr2_xyz": {
        "camera": "freiburg2",
        "scene_path": "datasets/TUM/rgbd_dataset_freiburg2_xyz",
        "description": "Primary benchmark: desk environment, translational motion (FR2 sensor)",
        "sensor": "Freiburg 2",
    },
    "tum_fr1_rpy": {
        "camera": "freiburg1",
        "scene_path": "datasets/TUM/rgbd_dataset_freiburg1_rpy",
        "description": "Unseen sequence A: table environment, aggressive 3D roll-pitch-yaw rotations (FR1 sensor)",
        "sensor": "Freiburg 1",
    },
    "tum_fr1_xyz": {
        "camera": "freiburg1",
        "scene_path": "datasets/TUM/rgbd_dataset_freiburg1_xyz",
        "description": "Unseen sequence B: table environment, smooth 3D translations (FR1 sensor)",
        "sensor": "Freiburg 1",
    },
    "tum_fr3_sitting_static": {
        "camera": "freiburg3",
        "scene_path": "datasets/TUM/rgbd_dataset_freiburg3_sitting_static",
        "description": "Unseen sequence C: office room with human sitting, distinct room & intrinsics (FR3 sensor)",
        "sensor": "Freiburg 3",
    },
}

# Policies for Multi-Sequence Evaluation
EVAL_POLICIES: List[str] = ["no_op", "error_only", "ours", "full"]

# Budget Sweep Values
BUDGET_SWEEP_MS: List[float] = [5.0, 10.0, 15.0, 20.0, 30.0]

# Canonical 11D Local Observable State Schema (s_i):
# Organized into 5 physical domains:
#   1. Local appearance:       rgb_error, depth_error
#   2. Sensitivity:            gradient_norm, position_drift, residual_drift_ema
#   3. Visibility / footprint: visibility_count, influence_mass, projected_area
#   4. Temporal dynamics:      age, update_frequency
#   5. Geometric uncertainty:  uncertainty_var
CANONICAL_LOCAL_FEATURES: List[str] = [
    "feat_rgb_error",            # 0: e_rgb
    "feat_depth_error",          # 1: e_depth
    "feat_gradient_norm",        # 2: ||\nabla L|| sensitivity proxy
    "feat_visibility_count",     # 3: v_count
    "feat_influence_mass",       # 4: m_attr
    "feat_position_drift",       # 5: d_pos
    "feat_residual_drift_ema",   # 6: d_res
    "feat_uncertainty_var",      # 7: \sigma^2
    "feat_projected_area",       # 8: A_proj
    "feat_update_frequency",     # 9: update_frequency
    "feat_age",                  # 10: age
]

# Canonical 12D Global Frame Context Schema (c_t):
# Map-level & frame-level summary statistics (no cross-attention overhead):
CANONICAL_GLOBAL_FEATURES: List[str] = [
    "glob_gaussian_count",        # 0: Total active Gaussians N_G
    "glob_visible_count",         # 1: Visible Gaussians N_vis
    "glob_visible_fraction",      # 2: N_vis / N_G
    "glob_mean_rgb_err",          # 3: Frame mean e_rgb
    "glob_std_rgb_err",           # 4: Frame std e_rgb
    "glob_mean_depth_err",        # 5: Frame mean e_depth
    "glob_std_depth_err",         # 6: Frame std e_depth
    "glob_mean_grad_norm",        # 7: Frame mean ||\nabla L||
    "glob_std_grad_norm",         # 8: Frame std ||\nabla L||
    "glob_mean_influence",        # 9: Frame mean m_attr
    "glob_selected_fraction",     # 10: Mean historical update rate
    "glob_normalized_frame_idx",  # 11: Frame index normalized (t / 60.0)
]

# Controlled Model Ablation Specifications (M0 through M4)
ABLATION_MODELS: Dict[str, Dict[str, Any]] = {
    "M0_Error": {
        "name": "Error Heuristic",
        "description": "Baseline proxy: e_rgb + e_depth (appearance error only)",
        "has_learned_model": False,
        "uses_positive_head": False,
        "uses_global_context": False,
    },
    "M1_GradNorm": {
        "name": "Gradient Sensitivity (Grad-Norm)",
        "description": "Baseline proxy: ||\\nabla L|| attribution mass * composite error",
        "has_learned_model": False,
        "uses_positive_head": False,
        "uses_global_context": False,
    },
    "M2_CurrentMLP": {
        "name": "Current TwoHeadMLP (11D Local)",
        "description": "Frozen Phase 4/10 baseline: \\hat{U} = \\hat{\\Delta Q} / \\hat{C}",
        "has_learned_model": True,
        "uses_positive_head": False,
        "uses_global_context": False,
    },
    "M3_PositiveHead": {
        "name": "MLP + Positive Classifier Head (11D Local)",
        "description": "Two-stage model: p_i = P(U* > 0), \\hat{U} = p_i * (\\hat{\\Delta Q} / \\hat{C})",
        "has_learned_model": True,
        "uses_positive_head": True,
        "uses_global_context": False,
    },
    "M4_PositiveGlobal": {
        "name": "MLP + Positive Head + Global Context (11D Local + 12D Global)",
        "description": "Two-stage multi-task model: f(s_i, c_t) -> p_i, \\hat{\\Delta Q}, \\hat{C}",
        "has_learned_model": True,
        "uses_positive_head": True,
        "uses_global_context": True,
    },
}

# 4 Primary Metric Dimensions for Evaluation:
PRIMARY_METRICS: Dict[str, List[str]] = {
    "selection_quality": ["ndcg_20", "ose_budget"],
    "positive_detection": ["auroc", "f1_pos"],
    "budget_efficiency": ["delta_q_per_ms", "realized_delta_q"],
    "reconstruction_quality": ["psnr", "ssim", "depth_l1"],
}


def get_phase12_output_dir(subfolder: str = "") -> Path:
    out = PHASE12_OUTPUT_DIR / subfolder if subfolder else PHASE12_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out
