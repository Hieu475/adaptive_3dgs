"""Phase 8: Generalization / Zero-Shot Transfer Protocol.

Single source of truth for Phase 8 experimental configuration.
Frozen before any experiment is run.

Research Question:
    Does a utility predictor trained on tum_fr1_desk generalize
    to unseen scenes (tum_fr2_xyz) without any fine-tuning?

RQ8.1: Does ρ(Û, U*) remain positive on unseen scenes?
RQ8.2: Does Û → S_B produce good selections on unseen scenes?
RQ8.3: How does performance degrade with distribution shift?

Protocol Invariants:
    - Test scene NEVER participates in training or normalizer fitting
    - Checkpoints are frozen from Phase 4 (no retraining)
    - Normalizer is frozen from Phase 4 (fit on train split only)
    - Feature schema is the 11 canonical features
    - n=5 seeds for statistical inference
    - Gate criteria are defined BEFORE experiments run
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════
# Repository paths
# ═══════════════════════════════════════════════════════════════════════

def get_repo_root() -> Path:
    """Return repository root (parent of research/)."""
    return Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════
# Seeds — identical to protocol_v1.yaml
# ═══════════════════════════════════════════════════════════════════════

SEEDS: List[int] = [42, 43, 44, 45, 46]


# ═══════════════════════════════════════════════════════════════════════
# Scene / Split Configuration
# ═══════════════════════════════════════════════════════════════════════

TRAIN_SCENES: List[str] = ["tum_fr1_desk"]
TRAIN_FRAME_RANGE: Tuple[int, int] = (0, 40)  # inclusive

VAL_SCENES: List[str] = ["tum_fr1_desk"]
VAL_FRAME_RANGE: Tuple[int, int] = (41, 60)  # inclusive

TEST_SCENES: List[str] = ["tum_fr2_xyz"]
# tum_fr2_xyz is zero-shot: entire scene is test, no frame restriction

# Transfer hierarchy for RQ8.3
TRANSFER_LEVELS: Dict[str, str] = {
    "in_domain":      "tum_fr1_desk frames 41-60 (temporal holdout, same scene)",
    "zero_shot":      "tum_fr2_xyz (unseen scene, different camera/motion)",
}


# ═══════════════════════════════════════════════════════════════════════
# Budget Configuration — from protocol_v1.yaml
# ═══════════════════════════════════════════════════════════════════════

BUDGETS: List[float] = [0.10, 0.20, 0.40, 0.60, 0.80]

TOP_K_FRACTIONS: List[float] = [0.10, 0.20]  # for NDCG@k, OSE@k, Overlap@k


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint Configuration — Phase 4 frozen
# ═══════════════════════════════════════════════════════════════════════

CHECKPOINT_PATTERN: str = "results/learned_utility/checkpoints/two_head_mlp_seed_{seed}.pt"

NORMALIZER_SOURCE: str = "results/learned_utility/normalization.json"

# Phase 4 model architecture parameters (for verification)
MODEL_IN_FEATURES: int = 11
MODEL_HIDDEN_DIM: int = 64
MODEL_EPS_COST: float = 0.001


# ═══════════════════════════════════════════════════════════════════════
# Feature Schema — 11 canonical features
# ═══════════════════════════════════════════════════════════════════════

FEATURE_SCHEMA: List[str] = [
    "rgb_error",
    "depth_error",
    "gradient_norm",
    "visibility_count",
    "influence_mass",
    "position_drift",
    "residual_drift_ema",
    "uncertainty_var",
    "projected_area",
    "update_frequency",
    "age",
]


# ═══════════════════════════════════════════════════════════════════════
# Oracle Configuration — for generating ground truth on test scene
# ═══════════════════════════════════════════════════════════════════════

ORACLE_N_SAMPLES: int = 40
ORACLE_N_OPT_STEPS: int = 5
ORACLE_W_RGB: float = 0.70
ORACLE_W_DEPTH: float = 0.30
ORACLE_MIN_INFLUENCE_PIXELS: int = 25


# ═══════════════════════════════════════════════════════════════════════
# Statistical Configuration
# ═══════════════════════════════════════════════════════════════════════

CONFIDENCE_LEVEL: float = 0.95
BOOTSTRAP_RESAMPLES: int = 1000
N_SEEDS: int = len(SEEDS)  # 5 — unit of statistical inference

# WARNING: Frame-level data is descriptive diagnostics only.
# Primary inference is at the seed level (n=5).


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Configuration — matching Phase 7
# ═══════════════════════════════════════════════════════════════════════

PIPELINE_CONFIG: Dict[str, Any] = {
    "gaussian": {
        "sh_degree": 0,
        "initial_opacity": 0.5,
        "max_gaussians": 30000,
        "initial_scale": 0.02,
    },
    "rendering": {
        "tile_size": 16,
        "image_width": 320,
        "image_height": 240,
        "use_surface_aware_depth": True,
        "attribution_top_k": 4,
    },
    "scheduler": {
        "gpu_budget_ms": 25.0,
        "policy": "budget_aware",
    },
    "densification": {
        "max_new_per_frame": 80,
        "strategy": "importance",
        "use_adaptive_thresholds": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Baselines — methods to compare
# ═══════════════════════════════════════════════════════════════════════

BASELINES: List[str] = [
    "random",
    "error_only",      # rgb_error + depth_error
    "heuristic",       # pipeline's built-in heuristic utility
    "learned",         # Phase 4 TwoHeadMLP (frozen)
    "oracle",          # oracle U* (upper bound)
]


# ═══════════════════════════════════════════════════════════════════════
# Gate Criteria — DEFINED BEFORE experiments run
# ═══════════════════════════════════════════════════════════════════════

GATE_CRITERIA: Dict[str, Dict[str, Any]] = {
    "Gate_8A_protocol_integrity": {
        "description": "Train/test strictly separated; checkpoint frozen; "
                       "normalizer fit on train only; no test leakage",
        "type": "checklist",
        "criteria": [
            "train_test_separated",
            "checkpoint_frozen",
            "normalizer_train_only",
            "no_test_leakage",
        ],
    },
    "Gate_8B_zero_shot_prediction": {
        "description": "Zero-shot prediction quality on unseen scene",
        "type": "quantitative",
        "primary_metric": "spearman_rho",
        "pass_criterion": "rho_learned > rho_random AND rho_learned > 0",
        "secondary_metrics": ["ndcg_20pct", "ose_20pct"],
        "secondary_criterion": "ndcg_learned > ndcg_random",
        "note": "If rho_learned <= 0, model has no zero-shot transfer signal",
    },
    "Gate_8C_zero_shot_selection": {
        "description": "Budget selection quality on unseen scene",
        "type": "quantitative",
        "primary_metric": "realized_delta_q",
        "pass_criterion": "delta_q_learned >= delta_q_error_only at majority of budgets",
        "secondary_metrics": ["regret", "ose"],
        "note": "Phase 7 found no advantage on in-domain; zero-shot may be worse",
    },
    "Gate_8D_transfer_robustness": {
        "description": "Generalization gap measurement",
        "type": "descriptive",
        "metrics": [
            "delta_rho = rho_in_domain - rho_zero_shot",
            "delta_ndcg = ndcg_in_domain - ndcg_zero_shot",
            "delta_ose = ose_in_domain - ose_zero_shot",
        ],
        "note": "No pass/fail — this is a measurement gate. "
                "Reports how much performance degrades under distribution shift.",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Output Structure
# ═══════════════════════════════════════════════════════════════════════

OUTPUT_DIR: str = "results/phase8_generalization"

OUTPUT_FILES: Dict[str, str] = {
    "manifest":             "manifest.json",
    "protocol":             "protocol.json",
    "summary":              "generalization_summary.md",
    "prediction_metrics":   "prediction_metrics.csv",
    "selection_metrics":    "selection_metrics.csv",
    "regret_metrics":       "regret_metrics.csv",
    "figures_dir":          "figures/",
}

# Per-seed result files
SEED_RESULT_PATTERN: str = "seed_{seed}.json"


# ═══════════════════════════════════════════════════════════════════════
# Execution Stages — ordered, gate-checked
# ═══════════════════════════════════════════════════════════════════════

EXECUTION_STAGES: List[Dict[str, str]] = [
    {
        "stage": "A",
        "name": "Utility Prediction",
        "description": "Evaluate ρ, NDCG, OSE of frozen model on test scene",
        "gate": "Gate_8B",
        "required": True,
    },
    {
        "stage": "B",
        "name": "Budget Selection",
        "description": "Evaluate ΔQ, regret at equal budgets on test scene",
        "gate": "Gate_8C",
        "required": True,
    },
    {
        "stage": "C",
        "name": "Generalization Analysis",
        "description": "Compute generalization gap metrics",
        "gate": "Gate_8D",
        "required": True,
    },
    {
        "stage": "D",
        "name": "Online Transfer (Optional)",
        "description": "Full online reconstruction on test scene",
        "gate": None,
        "required": False,
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Freeze Checklist
# ═══════════════════════════════════════════════════════════════════════

FREEZE_CHECKLIST: List[str] = [
    "protocol frozen",
    "checkpoint frozen",
    "no leakage",
    "prediction evaluation done",
    "selection evaluation done",
    "generalization gap reported",
    "n=5 statistics",
    "raw results saved",
    "figures generated",
    "manifest created",
    "checksums computed",
    "README updated",
    "pytest passing",
    "git diff --check clean",
    "phase8-frozen tag created",
]


# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════

def get_checkpoint_path(seed: int) -> str:
    """Return absolute path to Phase 4 checkpoint for given seed."""
    repo = get_repo_root()
    return str(repo / CHECKPOINT_PATTERN.format(seed=seed))


def get_normalizer_path() -> str:
    """Return absolute path to Phase 4 normalizer."""
    repo = get_repo_root()
    return str(repo / NORMALIZER_SOURCE)


def get_output_dir() -> str:
    """Return absolute path to Phase 8 output directory."""
    repo = get_repo_root()
    return str(repo / OUTPUT_DIR)


def get_seed_result_path(seed: int) -> str:
    """Return absolute path to per-seed result file."""
    repo = get_repo_root()
    return str(repo / OUTPUT_DIR / SEED_RESULT_PATTERN.format(seed=seed))


def validate_no_leakage() -> bool:
    """Verify that test scenes are not in train scenes."""
    train_set = set(TRAIN_SCENES)
    test_set = set(TEST_SCENES)
    overlap = train_set & test_set
    if overlap:
        raise ValueError(
            f"LEAKAGE DETECTED: {overlap} appears in both train and test scenes"
        )
    return True


def to_dict() -> Dict[str, Any]:
    """Serialize protocol to dictionary for manifest."""
    return {
        "phase": 8,
        "title": "Generalization / Zero-Shot Transfer",
        "protocol_version": "1.0.0",
        "seeds": SEEDS,
        "train_scenes": TRAIN_SCENES,
        "train_frame_range": list(TRAIN_FRAME_RANGE),
        "val_scenes": VAL_SCENES,
        "val_frame_range": list(VAL_FRAME_RANGE),
        "test_scenes": TEST_SCENES,
        "transfer_levels": TRANSFER_LEVELS,
        "budgets": BUDGETS,
        "top_k_fractions": TOP_K_FRACTIONS,
        "checkpoint_pattern": CHECKPOINT_PATTERN,
        "normalizer_source": NORMALIZER_SOURCE,
        "model_in_features": MODEL_IN_FEATURES,
        "model_hidden_dim": MODEL_HIDDEN_DIM,
        "feature_schema": FEATURE_SCHEMA,
        "oracle_config": {
            "n_samples": ORACLE_N_SAMPLES,
            "n_opt_steps": ORACLE_N_OPT_STEPS,
            "w_rgb": ORACLE_W_RGB,
            "w_depth": ORACLE_W_DEPTH,
            "min_influence_pixels": ORACLE_MIN_INFLUENCE_PIXELS,
        },
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "n_seeds": N_SEEDS,
        "baselines": BASELINES,
        "gate_criteria": GATE_CRITERIA,
        "execution_stages": EXECUTION_STAGES,
        "output_dir": OUTPUT_DIR,
        "freeze_checklist": FREEZE_CHECKLIST,
    }


# ═══════════════════════════════════════════════════════════════════════
# Self-check on import
# ═══════════════════════════════════════════════════════════════════════

# Verify no train/test leakage at import time
validate_no_leakage()

# Verify seed count
assert len(SEEDS) == 5, f"Expected 5 seeds, got {len(SEEDS)}"

# Verify feature schema length
assert len(FEATURE_SCHEMA) == MODEL_IN_FEATURES, (
    f"Feature schema has {len(FEATURE_SCHEMA)} features but model expects {MODEL_IN_FEATURES}"
)
