"""Phase 9B: Robust Normalization Protocol.

Single source of truth for Phase 9B experimental configuration.
Frozen before any experiment is run.

Scientific Objective:
    Determine whether robust static normalization (B1: median/MAD) or
    online test-time adaptive normalization (B2: EMA) recovers the in-domain
    utility quality lost under scale-invariant geometric representation (A1)
    while preserving cross-scene zero-shot transfer robustness on tum_fr2_xyz.

Pipeline:
    x_i  --(phi_A1 fixed)-->  z_i  --(psi_norm)-->  tilde_z_i  --(TwoHeadMLP f_theta)-->  U_hat_i

Variants:
    B0: "A1_standard"
        A1 geometry-relative features + standard train-domain FeatureNormalizer (mean/std).
        Baseline of Phase 9B.
    B1: "A1_robust_static"
        A1 geometry-relative features + train-only Median/MAD normalizer:
            tilde_z_ij = (z_ij - median(z_j)) / (MAD(z_j) + eps)
        Mitigates outlier sensitivity in training distribution. Test data never enters fitting.
    B2: "A1_online_adaptive"
        A1 geometry-relative features + online test-time EMA covariate adaptation:
            mu_t = beta * mu_{t-1} + (1 - beta) * mu_t^{frame}
            sigma_t = beta * sigma_{t-1} + (1 - beta) * sigma_t^{frame}
            mu_0 = mu_train, sigma_0 = sigma_train (beta = 0.9)
        Unsupervised test-time covariate adaptation. Model weights f_theta remain strictly frozen.

Protocol Invariants:
    - Representation phi_A1 is STRICTLY FIXED (depth_error, position_drift, projected_area relative).
    - Model architecture strictly preserved: TwoHeadMLP(in_features=11, hidden_dim=64).
    - Loss function strictly preserved: LossConfig(lambda_rank=1.0, lambda_q=0.25, lambda_t=0.125).
    - Optimizer & training schedule strictly preserved: Adam(lr=0.005, epochs=200).
    - Oracle U* = Delta Q / Cost strictly preserved (same intervention ground truth).
    - Data splits strictly preserved:
        Train: tum_fr1_desk frames 0-40 (N=375)
        Val:   tum_fr1_desk frames 41-60 (N=250)
        Test:  tum_fr2_xyz zero-shot (unseen scene)
    - Budgets: [0.10, 0.20, 0.40, 0.60, 0.80].
    - Seeds: n=5 seeds [42, 43, 44, 45, 46].
    - Online adaptation invariant (B2):
        Uses ONLY current-frame and past unlabeled feature statistics.
        Zero access to future frames, oracle U*, Delta Q, Delta T, or test labels.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from research.phase9_protocol import (
    CANONICAL_FEATURE_SCHEMA,
    SEEDS,
    N_SEEDS,
    TRAIN_SCENE,
    TRAIN_FRAME_RANGE,
    VAL_SCENE,
    VAL_FRAME_RANGE,
    TEST_SCENE,
    BUDGETS,
    TOP_K_FRACTIONS,
    MODEL_IN_FEATURES,
    MODEL_HIDDEN_DIM,
    MODEL_EPS_COST,
    TRAIN_EPOCHS,
    TRAIN_LR,
    LOSS_LAMBDA_RANK,
    LOSS_LAMBDA_Q,
    LOSS_LAMBDA_T,
    CONFIDENCE_LEVEL,
    BOOTSTRAP_RESAMPLES,
)


def get_repo_root() -> Path:
    """Return repository root."""
    return Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════
# Phase 9B Normalization Variants
# ═══════════════════════════════════════════════════════════════════════

NORMALIZATION_VARIANTS: List[str] = [
    "A1_standard",          # B0: standard mean/std normalizer fit on train
    "A1_robust_static",     # B1: median/MAD normalizer fit on train
    "A1_online_adaptive",   # B2: online EMA adaptive normalizer initialized from train
]

VARIANT_ALIASES_9B: Dict[str, str] = {
    "B0": "A1_standard",
    "B1": "A1_robust_static",
    "B2": "A1_online_adaptive",
    "standard": "A1_standard",
    "robust": "A1_robust_static",
    "adaptive": "A1_online_adaptive",
}

# Fixed EMA adaptation coefficient for B2
B2_EMA_BETA: float = 0.90
EPS: float = 1e-6

# Fixed Phase 9A baseline reference representation
BASE_REPRESENTATION: str = "geometry_relative"


# ═══════════════════════════════════════════════════════════════════════
# Baseline Methods Evaluated
# ═══════════════════════════════════════════════════════════════════════

BASELINES_9B: List[str] = [
    "random",
    "error_only",
    "heuristic",
    "learned",
    "oracle",
]


# ═══════════════════════════════════════════════════════════════════════
# Gate Criteria (Defined Before Experiments)
# ═══════════════════════════════════════════════════════════════════════

GATE_CRITERIA_9B: Dict[str, Dict[str, Any]] = {
    "Gate_9B_1_protocol_integrity": {
        "description": "A1 representation strictly fixed; no oracle leakage; "
                       "train-only fitting for B0/B1; online unlabeled stats only for B2; "
                       "identical splits, architecture, seeds, and loss.",
        "type": "checklist",
        "criteria": [
            "fixed_a1_representation",
            "no_oracle_leakage",
            "no_future_frame_access",
            "train_only_fit_b0_b1",
            "unlabeled_online_stats_b2",
            "same_training_protocol",
            "frozen_model_weights_at_test",
        ],
    },
    "Gate_9B_2_numerical_stability": {
        "description": "Zero NaN/Inf across all variants and seeds; safe fallback for MAD=0 or std=0.",
        "type": "quantitative",
        "criteria": [
            "no_nan_or_inf_in_features",
            "no_nan_or_inf_in_predictions",
            "all_5_seeds_converge",
            "stable_mad_and_std_fallback",
        ],
    },
    "Gate_9B_3_in_domain_recovery": {
        "description": "In-domain recovery: rho_in_B >= rho_in_A1 - 0.05 (target: rho_in_B > rho_in_A1 = 0.1911).",
        "type": "quantitative_decision",
        "baseline_metric": "rho_in_A1 = 0.1911",
        "margin": 0.05,
    },
    "Gate_9B_4_transfer_preservation": {
        "description": "Transfer robustness preserved: rho_zs_B >= rho_zs_A1 - 0.03 (A1 zero-shot rho = 0.2709).",
        "type": "quantitative_decision",
        "baseline_metric": "rho_zs_A1 = 0.2709",
        "tolerance": 0.03,
    },
    "Gate_9B_5_selection_protection": {
        "description": "Selection quality maintained on unseen test scene: OSE@20_zs_B >= OSE@20_zs_A1 - 0.05 (A1 OSE = 0.560).",
        "type": "quantitative_decision",
        "baseline_metric": "ose_zs_A1 = 0.560",
        "tolerance": 0.05,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Output Structure
# ═══════════════════════════════════════════════════════════════════════

OUTPUT_DIR_9B: str = "results/phase9b_robust_normalization"

OUTPUT_FILES_9B: Dict[str, str] = {
    "protocol":                 "protocol.json",
    "stage_results":            "stage_results.json",
    "prediction_metrics":       "prediction_metrics.csv",
    "selection_metrics":        "selection_metrics.csv",
    "generalization_gap":       "generalization_gap.csv",
    "normalization_stability":  "normalization_stability.csv",
    "runtime_overhead":         "runtime_overhead.csv",
    "summary":                  "summary.md",
    "manifest":                 "manifest.json",
    "figures_dir":              "figures/",
}

SEED_RESULT_PATTERN_9B: str = "seed_{seed}.json"


def get_output_dir_9b() -> Path:
    """Return absolute path to Phase 9B output directory."""
    repo = get_repo_root()
    p = repo / OUTPUT_DIR_9B
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_protocol_integrity_9b() -> bool:
    """Verify core constants match protocol invariants."""
    assert len(SEEDS) == 5, f"Expected 5 seeds, got {len(SEEDS)}"
    assert len(BUDGETS) == 5, f"Expected 5 budgets, got {len(BUDGETS)}"
    assert len(CANONICAL_FEATURE_SCHEMA) == 11, f"Expected 11 features, got {len(CANONICAL_FEATURE_SCHEMA)}"
    assert B2_EMA_BETA > 0.0 and B2_EMA_BETA < 1.0, f"Invalid EMA beta: {B2_EMA_BETA}"
    assert BASE_REPRESENTATION == "geometry_relative", f"Expected base A1 representation, got {BASE_REPRESENTATION}"
    return True


def to_dict() -> Dict[str, Any]:
    """Serialize protocol specification to JSON-compatible dictionary."""
    return {
        "phase": "Phase 9B: Robust Normalization",
        "protocol_version": "1.0.0",
        "status": "frozen",
        "base_representation": BASE_REPRESENTATION,
        "normalization_variants": NORMALIZATION_VARIANTS,
        "b2_ema_beta": B2_EMA_BETA,
        "eps": EPS,
        "seeds": list(SEEDS),
        "scenes": {
            "train": TRAIN_SCENE,
            "val": VAL_SCENE,
            "test": TEST_SCENE,
        },
        "model": {
            "in_features": MODEL_IN_FEATURES,
            "hidden_dim": MODEL_HIDDEN_DIM,
            "eps_cost": MODEL_EPS_COST,
        },
        "training": {
            "epochs": TRAIN_EPOCHS,
            "learning_rate": TRAIN_LR,
            "loss_lambda_rank": LOSS_LAMBDA_RANK,
            "loss_lambda_q": LOSS_LAMBDA_Q,
            "loss_lambda_t": LOSS_LAMBDA_T,
        },
        "budgets": list(BUDGETS),
        "gate_criteria": GATE_CRITERIA_9B,
    }
