"""Phase 9: Robust Utility Representation Protocol.

Single source of truth for Phase 9 experimental configuration.
Frozen before any experiment is run.

Research Question:
    Can we make the Gaussian utility representation more robust under
    cross-scene distribution shift without losing the in-domain benefits
    of Phase 4 and Phase 8?

    Baseline:
        U_hat_i = f_theta(x_i),  x_i in R^11 (raw canonical state)
    Phase 9A:
        z_i = phi(x_i),          z_i in R^11 (scale-invariant representation)
        U_hat_i = f_theta'(z_i)

Hypothesis (Phase 9A):
    Scale-relative geometric features (depth_error, position_drift, projected_area)
    reduce sensitivity to camera distance and scene metric scale, diminishing the
    generalization degradation gap (Delta_rho = rho_in_domain - rho_zero_shot)
    when transferring from tum_fr1_desk to unseen tum_fr2_xyz.

Protocol Invariants:
    - Only representation phi(x_i) changes across variants A0, A1, A2.
    - Model architecture strictly preserved: TwoHeadMLP(in_features=11, hidden_dim=64).
    - Loss function strictly preserved: LossConfig(lambda_rank=1.0, lambda_q=0.25, lambda_t=0.125).
    - Optimizer & training schedule strictly preserved: Adam(lr=0.005, epochs=200).
    - Oracle U* = Delta Q / Cost strictly preserved (same intervention protocol).
    - Data splits strictly preserved:
        Train: tum_fr1_desk frames 0-40 (N=375)
        Val:   tum_fr1_desk frames 41-60 (N=250)
        Test:  tum_fr2_xyz zero-shot (unseen scene)
    - Budgets: [0.10, 0.20, 0.40, 0.60, 0.80].
    - Seeds: n=5 seeds [42, 43, 44, 45, 46].
    - Normalization: train-domain reference statistics fit strictly on train split.
      Contextual relative features use current-frame unlabeled population statistics
      (never oracle U*, never future frames, never test labels).
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


def get_repo_root() -> Path:
    """Return repository root (parent of research/)."""
    return Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════
# Seeds & Protocol Constants
# ═══════════════════════════════════════════════════════════════════════

SEEDS: List[int] = [42, 43, 44, 45, 46]
N_SEEDS: int = len(SEEDS)  # 5 — unit of statistical inference

EPS: float = 1e-6
CONFIDENCE_LEVEL: float = 0.95
BOOTSTRAP_RESAMPLES: int = 1000


# ═══════════════════════════════════════════════════════════════════════
# Scene / Split Configuration — identical to Phase 4 / Phase 8
# ═══════════════════════════════════════════════════════════════════════

TRAIN_SCENE: str = "tum_fr1_desk"
TRAIN_FRAME_RANGE: Tuple[int, int] = (0, 40)

VAL_SCENE: str = "tum_fr1_desk"
VAL_FRAME_RANGE: Tuple[int, int] = (41, 60)

TEST_SCENE: str = "tum_fr2_xyz"


# ═══════════════════════════════════════════════════════════════════════
# Budget Configuration
# ═══════════════════════════════════════════════════════════════════════

BUDGETS: List[float] = [0.10, 0.20, 0.40, 0.60, 0.80]
TOP_K_FRACTIONS: List[float] = [0.10, 0.20]


# ═══════════════════════════════════════════════════════════════════════
# Representation Variants (Phase 9A)
# ═══════════════════════════════════════════════════════════════════════

FEATURE_VARIANTS: List[str] = [
    "raw",                             # A0: 11 raw canonical features + P4 normalizer
    "geometry_relative",               # A1: scale-relative depth_error, position_drift, projected_area
    "geometry_optimization_relative",  # A2: A1 + relative gradient, influence, uncertainty, residual
]

VARIANT_ALIASES: Dict[str, str] = {
    "A0": "raw",
    "A1": "geometry_relative",
    "A2": "geometry_optimization_relative",
}

CANONICAL_FEATURE_SCHEMA: List[str] = [
    "rgb_error",           # 0: appearance (L1) - unchanged in A0, A1, A2
    "depth_error",         # 1: geometry - relative in A1, A2
    "gradient_norm",       # 2: geometry grad - relative in A2
    "visibility_count",    # 3: visibility - unchanged in A0, A1, A2
    "influence_mass",      # 4: attribution - relative in A2
    "position_drift",      # 5: temporal drift - relative in A1, A2
    "residual_drift_ema",  # 6: temporal residual - relative in A2
    "uncertainty_var",     # 7: uncertainty - relative in A2
    "projected_area",      # 8: footprint/cost - relative in A1, A2
    "update_frequency",    # 9: update rate - unchanged in A0, A1, A2
    "age",                 # 10: lifecycle age - unchanged in A0, A1, A2
]


# ═══════════════════════════════════════════════════════════════════════
# Model Architecture & Training Protocol — identical to Phase 4
# ═══════════════════════════════════════════════════════════════════════

MODEL_IN_FEATURES: int = 11
MODEL_HIDDEN_DIM: int = 64
MODEL_EPS_COST: float = 0.001

TRAIN_EPOCHS: int = 200
TRAIN_LR: float = 0.005
LOSS_LAMBDA_RANK: float = 1.0
LOSS_LAMBDA_Q: float = 0.25
LOSS_LAMBDA_T: float = 0.125


# ═══════════════════════════════════════════════════════════════════════
# Baseline Methods
# ═══════════════════════════════════════════════════════════════════════

BASELINES: List[str] = [
    "random",
    "error_only",      # rgb_error + depth_error
    "heuristic",       # pipeline built-in heuristic utility
    "learned_A0",      # Phase 4 TwoHeadMLP (raw canonical features)
    "learned_A1",      # TwoHeadMLP trained on geometry_relative features
    "learned_A2",      # TwoHeadMLP trained on geometry_optimization_relative features
    "oracle",          # oracle U* upper bound
]


# ═══════════════════════════════════════════════════════════════════════
# Gate Criteria — DEFINED BEFORE experiments run
# ═══════════════════════════════════════════════════════════════════════

GATE_CRITERIA: Dict[str, Dict[str, Any]] = {
    "Gate_9A_1_protocol_integrity": {
        "description": "Train/test strictly separated; no oracle leakage; "
                       "same data split, loss, architecture, and seeds.",
        "type": "checklist",
        "criteria": [
            "no_oracle_leakage",
            "same_split_as_phase4",
            "same_oracle_utility_definition",
            "same_training_protocol",
            "feature_transform_unlabeled_only",
        ],
    },
    "Gate_9A_2_representation_validity": {
        "description": "Representation validity across all 5 seeds without numerical pathology.",
        "type": "quantitative",
        "criteria": [
            "no_nan_or_inf_in_features",
            "all_5_seeds_converge",
            "deterministic_transforms",
        ],
    },
    "Gate_9A_3_generalization_improvement": {
        "description": "Generalization gap reduction: Delta_rho_A1 < Delta_rho_A0 while "
                       "preserving in-domain selection quality (regression protection).",
        "type": "decision_matrix",
        "primary_metric": "delta_rho = rho_in_domain - rho_zero_shot",
        "regression_protection": "rho_in_domain_A1 >= rho_in_domain_A0 - 0.05",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Output Structure
# ═══════════════════════════════════════════════════════════════════════

OUTPUT_DIR: str = "results/phase9a_scale_invariant"

OUTPUT_FILES: Dict[str, str] = {
    "protocol":               "protocol.json",
    "manifest":               "manifest.json",
    "summary":                "summary.md",
    "representation_metrics": "representation_metrics.csv",
    "prediction_metrics":     "prediction_metrics.csv",
    "selection_metrics":      "selection_metrics.csv",
    "generalization_gap":     "generalization_gap.csv",
    "figures_dir":            "figures/",
}

SEED_RESULT_PATTERN: str = "seed_{seed}.json"


def get_output_dir() -> Path:
    """Return absolute path to output directory."""
    path = get_repo_root() / OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_protocol_integrity() -> Dict[str, bool]:
    """Verify that protocol preconditions are satisfied."""
    checks = {
        "seeds_count": len(SEEDS) == 5,
        "features_count": len(CANONICAL_FEATURE_SCHEMA) == MODEL_IN_FEATURES == 11,
        "budgets_count": len(BUDGETS) == 5,
        "variants_count": len(FEATURE_VARIANTS) == 3,
        "train_val_test_disjoint": (
            TRAIN_SCENE != TEST_SCENE or
            (TRAIN_FRAME_RANGE[1] < VAL_FRAME_RANGE[0])
        ),
    }
    for k, v in checks.items():
        assert v, f"Protocol integrity check failed: {k}"
    return checks


def to_dict() -> Dict[str, Any]:
    """Serialize protocol to dictionary."""
    return {
        "phase": "Phase 9A: Scale-Invariant Features",
        "primary_objective": "Make utility representation robust under distribution shift",
        "seeds": list(SEEDS),
        "train_scene": TRAIN_SCENE,
        "train_frame_range": list(TRAIN_FRAME_RANGE),
        "val_scene": VAL_SCENE,
        "val_frame_range": list(VAL_FRAME_RANGE),
        "test_scene": TEST_SCENE,
        "budgets": list(BUDGETS),
        "top_k_fractions": list(TOP_K_FRACTIONS),
        "feature_variants": list(FEATURE_VARIANTS),
        "canonical_feature_schema": list(CANONICAL_FEATURE_SCHEMA),
        "model_params": {
            "in_features": MODEL_IN_FEATURES,
            "hidden_dim": MODEL_HIDDEN_DIM,
            "eps_cost": MODEL_EPS_COST,
        },
        "training_params": {
            "epochs": TRAIN_EPOCHS,
            "learning_rate": TRAIN_LR,
            "lambda_rank": LOSS_LAMBDA_RANK,
            "lambda_q": LOSS_LAMBDA_Q,
            "lambda_t": LOSS_LAMBDA_T,
        },
        "gate_criteria": GATE_CRITERIA,
    }
