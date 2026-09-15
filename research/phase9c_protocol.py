"""Phase 9C: B2 Robustness and Adaptation Necessity Protocol.

Single source of truth for Phase 9C experimental configuration.
Frozen before any experiment is run.

Scientific Objective:
    Determine whether the online EMA covariate adaptation mechanism (B2)
    is genuinely robust and necessary, addressing:
        1. 9C-1 EMA Sensitivity: Is performance stable across adaptation timescales
           beta in {0.80, 0.90, 0.95}?
        2. 9C-2 Adaptation Ablation: Is B2_update > B2_static ~= B0, proving that
           test-time adaptation is active and the implementation is uncorrupted?
        3. 9C-3 Temporal Dynamics: Does EMA exhibit structured convergence
           (early frames -> adaptation -> stabilization -> steady state)
           rather than erratic stochastic drift?
        4. 9C-4 Perturbation Robustness: Does online adaptation protect against
           controlled feature scale (a in {0.8, 1.0, 1.2}) and offset (b in {-0.2, 0.0, +0.2})
           covariate shifts compared to static normalization (B0)?

Protocol Invariants:
    - Representation phi_A1 is STRICTLY FIXED (geometry_relative).
    - Model architecture strictly preserved: TwoHeadMLP(in_features=11, hidden_dim=64).
    - Checkpoints: Frozen Phase-9B B2 model checkpoints evaluated at test time.
    - Loss function strictly preserved: LossConfig(lambda_rank=1.0, lambda_q=0.25, lambda_t=0.125).
    - Optimizer & training schedule strictly preserved: Adam(lr=0.005, epochs=200).
    - Oracle U* = Delta Q / Cost strictly preserved (same intervention ground truth).
    - Data splits strictly preserved:
        Train: tum_fr1_desk frames 0-40 (N=375)
        Val:   tum_fr1_desk frames 41-60 (N=250)
        Test:  tum_fr2_xyz zero-shot (unseen scene, N=250)
    - Budgets: [0.10, 0.20, 0.40, 0.60, 0.80].
    - Seeds: n=5 seeds [42, 43, 44, 45, 46].
    - Online adaptation invariant:
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
from research.phase9b_protocol import (
    BASE_REPRESENTATION,
    EPS,
    B2_EMA_BETA as DEFAULT_B2_BETA,
)


def get_repo_root() -> Path:
    """Return repository root."""
    return Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════
# Phase 9C Configurations
# ═══════════════════════════════════════════════════════════════════════

# 9C-1: Sensitivity grid for EMA adaptation parameter beta
BETA_VALUES: List[float] = [0.80, 0.90, 0.95]
BETA_DESCRIPTIONS: Dict[float, str] = {
    0.80: "fast adaptation (shorter temporal memory, ~5 frames)",
    0.90: "standard Phase 9B reference configuration (~10 frames)",
    0.95: "slow adaptation (longer temporal memory, ~20 frames)",
}

# 9C-2: Ablation variants
ABLATION_VARIANTS: List[str] = [
    "B0",           # Static train normalizer (A1_standard)
    "B2",           # Dynamic online EMA normalizer (beta=0.90)
    "B2_static",    # Online normalizer initialized from train stats with test updates OFF
]

ABLATION_VARIANT_NAMES: Dict[str, str] = {
    "B0": "A1_standard",
    "B2": "A1_online_adaptive",
    "B2_static": "A1_b2_static_start",
}

# 9C-4: Controlled perturbation suite
# Scale factor perturbations: x' = a * x
PERTURBATION_SCALES: List[float] = [0.8, 1.0, 1.2]

# Additive offset perturbations: x' = x + b
PERTURBATION_OFFSETS: List[float] = [-0.2, 0.0, 0.2]

# Baseline methods evaluated
BASELINES_9C: List[str] = [
    "random",
    "error_only",
    "heuristic",
    "learned",
    "oracle",
]

# ═══════════════════════════════════════════════════════════════════════
# Gate Criteria (Defined Before Experiments)
# ═══════════════════════════════════════════════════════════════════════

GATE_CRITERIA_9C: Dict[str, Dict[str, Any]] = {
    "Gate_9C_1_protocol_integrity": {
        "description": "A1 representation strictly fixed; same dataset, architecture, loss, oracle, and seeds; "
                       "no future frame access; no test tuning; frozen models evaluated.",
        "type": "checklist",
        "criteria": [
            "fixed_a1_representation",
            "same_dataset_and_splits",
            "same_architecture_and_loss",
            "same_oracle_u_star",
            "same_5_seeds",
            "no_future_frame_access",
            "no_test_tuning",
        ],
    },
    "Gate_9C_2_sensitivity": {
        "description": "Assess sensitivity across beta in {0.80, 0.90, 0.95}. "
                       "Valid scientific result whether performance is stable or sensitive.",
        "type": "quantitative_analysis",
        "beta_values": BETA_VALUES,
        "criteria": [
            "report_spearman_rho",
            "report_ndcg_20",
            "report_ose_20",
            "report_delta_q",
            "report_drift_metrics",
            "report_runtime",
        ],
    },
    "Gate_9C_3_adaptation_necessity": {
        "description": "Adaptation necessity & equivalence check: "
                       "1. B2_static (update off) must be numerically equivalent to B0 within floating-point tolerance (< 1e-5). "
                       "2. Observe B2_update vs B2_static to verify active adaptation.",
        "type": "quantitative_equivalence",
        "tolerance": 1e-5,
    },
    "Gate_9C_4_temporal_stability": {
        "description": "Temporal dynamics verification: zero NaN/Inf, bounded drift D_t^norm and D_t^sigma, "
                       "no diverging statistics, smooth stabilization towards steady state.",
        "type": "quantitative",
        "criteria": [
            "zero_nan_or_inf",
            "bounded_mean_drift",
            "bounded_scale_drift",
            "non_zero_selection_overlap",
        ],
    },
    "Gate_9C_5_perturbation_robustness": {
        "description": "Under controlled feature perturbations (a in {0.8, 1.0, 1.2}, b in {-0.2, 0.0, +0.2}), "
                       "B2 must not degrade systematically worse than B0.",
        "type": "comparative_robustness",
        "scale_factors": PERTURBATION_SCALES,
        "offset_values": PERTURBATION_OFFSETS,
    },
}

# ═══════════════════════════════════════════════════════════════════════
# Output Structure
# ═══════════════════════════════════════════════════════════════════════

OUTPUT_DIR_9C: str = "results/phase9c_b2_robustness"

OUTPUT_FILES_9C: Dict[str, str] = {
    "protocol":                 "protocol.json",
    "stage_results":            "stage_results.json",
    "sensitivity_metrics":      "sensitivity_metrics.csv",
    "ablation_metrics":         "ablation_metrics.csv",
    "temporal_dynamics":        "temporal_dynamics.csv",
    "perturbation_metrics":     "perturbation_metrics.csv",
    "selection_metrics":        "selection_metrics.csv",
    "runtime_metrics":          "runtime_metrics.csv",
    "summary":                  "summary.md",
    "manifest":                 "manifest.json",
    "figures_dir":              "figures/",
}

SEED_RESULT_PATTERN_9C: str = "seed_{seed}.json"


def get_output_dir_9c() -> Path:
    """Return absolute path to Phase 9C output directory."""
    repo = get_repo_root()
    p = repo / OUTPUT_DIR_9C
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_protocol_integrity_9c() -> bool:
    """Verify core constants match Phase 9C protocol invariants."""
    assert len(SEEDS) == 5, f"Expected 5 seeds, got {len(SEEDS)}"
    assert len(BUDGETS) == 5, f"Expected 5 budgets, got {len(BUDGETS)}"
    assert len(CANONICAL_FEATURE_SCHEMA) == 11, f"Expected 11 features, got {len(CANONICAL_FEATURE_SCHEMA)}"
    assert BASE_REPRESENTATION == "geometry_relative", f"Expected base A1 representation, got {BASE_REPRESENTATION}"
    assert BETA_VALUES == [0.80, 0.90, 0.95], f"Unexpected beta values: {BETA_VALUES}"
    assert PERTURBATION_SCALES == [0.8, 1.0, 1.2], f"Unexpected scale perturbations: {PERTURBATION_SCALES}"
    assert PERTURBATION_OFFSETS == [-0.2, 0.0, 0.2], f"Unexpected offset perturbations: {PERTURBATION_OFFSETS}"
    return True


def to_dict() -> Dict[str, Any]:
    """Serialize Phase 9C protocol specification to JSON-compatible dictionary."""
    return {
        "phase": "Phase 9C: B2 Robustness & Adaptation Necessity",
        "protocol_version": "1.0.0",
        "status": "frozen",
        "base_representation": BASE_REPRESENTATION,
        "default_b2_beta": DEFAULT_B2_BETA,
        "beta_values": BETA_VALUES,
        "ablation_variants": ABLATION_VARIANTS,
        "perturbation_scales": PERTURBATION_SCALES,
        "perturbation_offsets": PERTURBATION_OFFSETS,
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
        "gate_criteria": GATE_CRITERIA_9C,
    }
