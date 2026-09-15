r"""Phase 10: End-to-End Adaptive 3DGS Integration Protocol.

Single source of truth for Phase 10 experimental configuration.
Strictly frozen before any benchmark execution.

Scientific Objective:
    Transition from research prototype to an end-to-end, closed-loop Adaptive 3DGS system.
    Evaluate whether the frozen utility model (A1 + B2 + beta=0.90 + Phase 9C checkpoint),
    when embedded directly inside the continuous online reconstruction trajectory:
        S_t -> X_t -> \hat{X}_t -> \hat{U}_t -> A_t -> S_{t+1}
    delivers robust, budget-aware Gaussian subset optimization without resetting state between frames.

Protocol Invariants (10.1):
    1. Research Model Frozen:
       - Base representation phi_A1 is STRICTLY FIXED ("geometry_relative", 11-D).
       - Model architecture is STRICTLY PRESERVED: TwoHeadMLP(in_features=11, hidden_dim=64, eps_cost=0.001).
       - Test-time normalizer is STRICTLY PRESERVED: OnlineEMANormalizer with beta=0.90.
       - Checkpoints: Frozen Phase 9B / 9C checkpoints (A1_online_adaptive_seed_{seed}.pt).
       - NO model retraining.
       - NO fine-tuning on test scenes.
       - NO modification of feature representations.
       - NO peeking at ground truth (zero oracle delta_q or delta_t at test time).
       - NO future frame access.

    2. Online Closed-Loop Trajectory (10.5):
       - Sequential frame processing: RGB-D frame 0 -> select -> optimize -> update -> frame 1 -> ... -> frame T.
       - Gaussian map state and StateStore persist continuously across all frames (NO reset between frames).

    3. Budget Accounting (10.4):
       - Explicitly distinguish modeled scheduler budget from physical wall-clock latency:
           predicted_cost:       \sum_{i in A_t} \hat{C}_i
           scheduled_cost:       \sum_{i in A_t} \alpha * \hat{C}_i   (\alpha = 1.10)
           actual_optimization:  actual wall-clock time spent in optimizer step (ms)
           total_frame_latency:  actual wall-clock time spent on entire frame (ms)
       - Do NOT report "budget satisfied" merely because predicted_cost <= budget.

    4. Evaluated Policies (10.5 & 10.7):
       - NO_OP:      zero optimization baseline (rendering + state update only).
       - ERROR_ONLY: strong heuristic baseline ranking by photometric + depth error (e_rgb + e_depth).
       - OURS (B2):  end-to-end adaptive scheduling with A1 + B2 (beta=0.90) + TwoHeadMLP.
       - FULL:       unconstrained full-scene optimization (reference upper bound).

    5. Reproducibility & Multi-Seed:
       - Multi-seed validation: seeds = [42, 43, 44, 45, 46] (n=5).
       - Primary test scene: tum_fr2_xyz (zero-shot cross-scene transfer, unseen during training).
       - Reference train scene: tum_fr1_desk (for initial reference statistics).
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from research.phase9_protocol import (
    CANONICAL_FEATURE_SCHEMA,
    SEEDS,
    N_SEEDS,
    TRAIN_SCENE,
    VAL_SCENE,
    TEST_SCENE,
    MODEL_IN_FEATURES,
    MODEL_HIDDEN_DIM,
    MODEL_EPS_COST,
    EPS,
)
from research.phase9b_protocol import (
    BASE_REPRESENTATION,
    B2_EMA_BETA as DEFAULT_B2_BETA,
)


def get_repo_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════
# Phase 10 Core Hyperparameters & Configurations
# ═══════════════════════════════════════════════════════════════════════

# Default per-frame scheduler budget (ms)
DEFAULT_BUDGET_MS: float = 15.0

# Safety factor alpha for knapsack budget scheduling
SAFETY_FACTOR: float = 1.10

# Trajectory length (frames)
DEFAULT_TRAJECTORY_FRAMES: int = 30

# Image resolution
DEFAULT_IMAGE_WIDTH: int = 320
DEFAULT_IMAGE_HEIGHT: int = 240

# Normalization parameter
B2_BETA: float = 0.90

# Evaluated policies
POLICIES_PHASE10: List[str] = [
    "no_op",
    "error_only",
    "ours",       # B2: A1 + B2 (beta=0.90) + TwoHeadMLP
    "full",       # Reference bound (run on reference subset)
]

POLICY_DESCRIPTIONS: Dict[str, str] = {
    "no_op": "No optimization; pass-through tracking, rendering, and state accumulation",
    "error_only": "Strong heuristic baseline selecting Gaussians by combined RGB+depth error under budget",
    "ours": "End-to-end Adaptive 3DGS with A1 geometry-relative + B2 online normalizer + TwoHeadMLP under budget",
    "full": "Unconstrained full-scene optimization (reference upper bound)",
}

# ═══════════════════════════════════════════════════════════════════════
# Gate Criteria (Defined Before Experiments)
# ═══════════════════════════════════════════════════════════════════════

GATE_CRITERIA_10: Dict[str, Dict[str, Any]] = {
    "Gate_10A_integration": {
        "description": "Model checkpoint loaded properly; B2 normalizer runs online; "
                       "StateStore updates per frame; selector uses B2 output; zero oracle/test labels.",
        "type": "verification",
        "evidence_required": [
            "checkpoint_sha256_verified",
            "model_eval_mode_verified",
            "all_params_requires_grad_false",
            "online_b2_normalizer_active",
            "statestore_updated_per_frame",
            "b2_utility_guides_selection",
            "zero_oracle_access",
        ],
    },
    "Gate_10B_closed_loop_correctness": {
        "description": "Continuous stateful trajectory S_t -> S_{t+1} across entire sequence. "
                       "No Gaussian state reset between frames. Population dynamics audited.",
        "type": "verification",
        "evidence_required": [
            "continuous_map_state",
            "no_per_frame_state_reset",
            "statestore_model_synchronization",
            "population_dynamics_logged",
            "zero_crashes_and_zero_nans",
        ],
    },
    "Gate_10C_budget_accounting": {
        "description": "Every frame logs predicted cost \\hat{C}_t, scheduled cost C_t^{scheduled}, "
                       "and actual wall-clock latency C_t^{actual}. Distinctly reports scheduler budget "
                       "vs physical wall-clock budget.",
        "type": "accounting",
        "evidence_required": [
            "log_predicted_cost",
            "log_scheduled_cost",
            "log_actual_opt_time",
            "log_total_frame_time",
            "log_fine_grained_latency_breakdown",
            "distinguish_scheduler_vs_physical_budget",
        ],
    },
    "Gate_10D_scientific_validity": {
        "description": "Evidence-driven audit: checkpoint hashes verified, model weights strictly immutable "
                       "(\\|\\theta_T - \\theta_0\\| = 0), zero oracle access in decision path, zero future frame leakage, "
                       "and multi-seed reproducibility.",
        "type": "audit",
        "evidence_required": [
            "checkpoint_hash_match",
            "model_weight_immutability_zero_diff",
            "zero_oracle_runtime_audit_pass",
            "zero_future_leakage_audit_pass",
            "fixed_5_seeds_evaluated",
        ],
    },
    "Gate_10E_performance_characterization": {
        "description": "Objective characterization of B2 vs ERROR_ONLY and B2 vs FULL across all seeds. "
                       "Valid scientific result regardless of whether B2 outperforms or matches baseline.",
        "type": "characterization",
        "evidence_required": [
            "evaluate_all_4_policies",
            "report_psnr_trajectory_and_ci",
            "report_delta_q_and_cumulative_q",
            "report_selection_dynamics",
            "report_systems_and_memory_metrics",
            "paired_statistical_tests_wilcoxon_bootstrap",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════
# Paths & Output Specifications
# ═══════════════════════════════════════════════════════════════════════

OUTPUT_DIR_10: str = "results/phase10_e2e"

OUTPUT_FILES_10: Dict[str, str] = {
    "protocol":             "protocol.json",
    "trajectory_metrics":   "trajectory_metrics.csv",
    "frame_metrics":        "frame_metrics.csv",
    "selection_metrics":    "selection_metrics.csv",
    "runtime_metrics":      "runtime_metrics.csv",
    "memory_metrics":       "memory_metrics.csv",
    "latency_breakdown":    "latency_breakdown.csv",
    "audit_metrics":        "audit_metrics.csv",
    "summary":              "summary.md",
    "manifest":             "manifest.json",
    "figures_dir":          "figures/",
    "checkpoints_dir":      "checkpoints/",
}

FIGURE_NAMES_10: List[str] = [
    "quality_vs_frame.png",
    "latency_vs_frame.png",
    "budget_vs_actual.png",
    "gaussian_selection.png",
    "trajectory_comparison.png",
]


def get_output_dir_10() -> Path:
    """Return absolute path to Phase 10 output directory."""
    repo = get_repo_root()
    p = repo / OUTPUT_DIR_10
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_checkpoint_path_for_seed(seed: int) -> Path:
    """Return path to frozen Phase 9B / 9C checkpoint for a given seed."""
    repo = get_repo_root()
    p = repo / "results" / "phase9b_robust_normalization" / "checkpoints" / f"A1_online_adaptive_seed_{seed}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing frozen checkpoint: {p}")
    return p


def get_normalizer_path_for_seed(seed: int) -> Path:
    """Return path to frozen Phase 9B / 9C normalizer JSON for a given seed."""
    repo = get_repo_root()
    p = repo / "results" / "phase9b_robust_normalization" / "normalizers" / f"A1_online_adaptive_seed_{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing frozen normalizer: {p}")
    return p


def validate_protocol_integrity_10() -> bool:
    """Verify core constants match Phase 10 protocol invariants."""
    assert len(SEEDS) == 5, f"Expected 5 seeds, got {len(SEEDS)}"
    assert len(CANONICAL_FEATURE_SCHEMA) == 11, f"Expected 11 features, got {len(CANONICAL_FEATURE_SCHEMA)}"
    assert BASE_REPRESENTATION == "geometry_relative", f"Expected geometry_relative, got {BASE_REPRESENTATION}"
    assert B2_BETA == 0.90, f"Expected beta=0.90, got {B2_BETA}"
    assert MODEL_IN_FEATURES == 11, f"Expected in_features=11, got {MODEL_IN_FEATURES}"
    assert MODEL_HIDDEN_DIM == 64, f"Expected hidden_dim=64, got {MODEL_HIDDEN_DIM}"
    assert DEFAULT_BUDGET_MS == 15.0, f"Expected budget=15.0ms, got {DEFAULT_BUDGET_MS}"
    assert SAFETY_FACTOR == 1.10, f"Expected safety_factor=1.10, got {SAFETY_FACTOR}"
    return True


def to_dict() -> Dict[str, Any]:
    """Serialize Phase 10 protocol specification to JSON-compatible dictionary."""
    return {
        "phase": "Phase 10: End-to-End Adaptive 3DGS Integration",
        "protocol_version": "1.0.0",
        "status": "frozen",
        "invariants": {
            "base_representation": BASE_REPRESENTATION,
            "b2_beta": B2_BETA,
            "model_architecture": f"TwoHeadMLP(in_features={MODEL_IN_FEATURES}, hidden_dim={MODEL_HIDDEN_DIM})",
            "model_eps_cost": MODEL_EPS_COST,
            "eps": EPS,
            "safety_factor": SAFETY_FACTOR,
            "zero_oracle_at_test": True,
            "zero_test_tuning": True,
        },
        "seeds": list(SEEDS),
        "scenes": {
            "primary_test": TEST_SCENE,
            "reference_train": TRAIN_SCENE,
        },
        "budget_ms": DEFAULT_BUDGET_MS,
        "default_trajectory_frames": DEFAULT_TRAJECTORY_FRAMES,
        "resolution": [DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH],
        "policies": POLICIES_PHASE10,
        "policy_descriptions": POLICY_DESCRIPTIONS,
        "gate_criteria": GATE_CRITERIA_10,
    }
