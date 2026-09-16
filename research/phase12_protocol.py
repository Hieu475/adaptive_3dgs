#!/usr/bin/env python3
r"""Phase 12 Protocol: Paper Evidence & Benchmark Expansion.

Formal protocol defining the multi-sequence evaluation suite, external baselines,
quality-vs-budget sweeps, oracle recovery, and systems profiling for publication.

Key Experimental Pillars:
    1. Multi-Sequence Zero-Shot Benchmark (4 unseen sequences across FR1, FR2, FR3)
    2. External Baseline Comparisons (Gradient/Sensitivity, Importance, Error-only)
    3. Quality vs. Budget Curves (B in {5, 10, 15, 20, 30} ms) and AUC_{Q-B}
    4. Direct Error != Utility Evidence (rho(error, U*) vs rho(U_hat, U*), U* < 0 regions)
    5. Oracle Recovery Headroom Quantification (Q_ours - Q_rand) / (Q_oracle - Q_rand)
    6. Tracking & Memory Resource Profiles (ATE RMSE, GPU memory, primitive counts)
    7. Systems Latency Breakdown Disclosures
"""
import os
from pathlib import Path
from typing import Dict, List, Any

# Repository Root
REPO_ROOT = Path(__file__).resolve().parent.parent

# Output Directory for Phase 12 Paper Evidence
PHASE12_OUTPUT_DIR = REPO_ROOT / "results" / "phase12_paper_evidence"

# Protocol Invariants & Experimental Setup
SEEDS: List[int] = [42, 43, 44, 45, 46]
DEFAULT_BUDGET_MS: float = 15.0
SAFETY_FACTOR: float = 1.10
DEFAULT_TRAJECTORY_FRAMES: int = 30
DEFAULT_IMAGE_WIDTH: int = 320
DEFAULT_IMAGE_HEIGHT: int = 240

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

# Budget Sweep Values (for Step 5 & Step 8)
BUDGET_SWEEP_MS: List[float] = [5.0, 10.0, 15.0, 20.0, 30.0]


def get_phase12_output_dir(subfolder: str = "") -> Path:
    out = PHASE12_OUTPUT_DIR / subfolder if subfolder else PHASE12_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out
