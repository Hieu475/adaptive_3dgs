"""Canonical Feature Schema for Learned Utility Estimation (Phase 4).

This module provides the single source of truth for the 11 canonical input features
s_i(t) used by learned utility models across Phase 4, Phase 5, and Phase 6.

Schema:
  s_i(t) = [
      rgb_error,           # 0: Appearance (pre-intervention)
      depth_error,         # 1: Geometry (pre-intervention)
      gradient_norm,       # 2: Geometry gradient (pre-intervention)
      visibility_count,    # 3: Visibility (StateStore / attribution)
      influence_mass,      # 4: Attribution (pixel influence mass)
      position_drift,      # 5: Temporal position drift (StateStore)
      residual_drift_ema,  # 6: Temporal residual drift EMA (StateStore)
      uncertainty_var,     # 7: Uncertainty variance (StateStore)
      projected_area,      # 8: Cost / footprint (renderer)
      update_frequency,    # 9: Temporal update frequency (StateStore history)
      age,                 # 10: Lifecycle age (t - t_creation)
  ]
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata specification for an individual Gaussian state feature."""
    name: str
    group: str
    source: str
    units: str
    min_val: float
    max_val: float
    normalization: str
    leakage_status: str
    description: str


CANONICAL_FEATURE_SPECS: List[FeatureSpec] = [
    FeatureSpec(
        name="rgb_error",
        group="appearance",
        source="pre-intervention rendering",
        units="L1 pixel error [0, 1]",
        min_val=0.0,
        max_val=1.0,
        normalization="train_z_score",
        leakage_status="clean_pre_intervention",
        description="Mean L1 RGB photometric error within Gaussian footprint before intervention.",
    ),
    FeatureSpec(
        name="depth_error",
        group="geometry",
        source="pre-intervention rendering",
        units="metric L1 depth error (meters)",
        min_val=0.0,
        max_val=10.0,
        normalization="train_z_score",
        leakage_status="clean_pre_intervention",
        description="Mean L1 depth residual within Gaussian footprint before intervention.",
    ),
    FeatureSpec(
        name="gradient_norm",
        group="geometry",
        source="pre-intervention backward pass",
        units="L2 norm of position gradient",
        min_val=0.0,
        max_val=100.0,
        normalization="train_z_score",
        leakage_status="clean_pre_intervention",
        description="Magnitude of pre-intervention positional gradient ||dLoss/dmu||_2.",
    ),
    FeatureSpec(
        name="visibility_count",
        group="visibility",
        source="attribution / StateStore",
        units="integer pixel count",
        min_val=0.0,
        max_val=1e6,
        normalization="train_z_score",
        leakage_status="clean_pre_intervention",
        description="Number of screen-space pixels where Gaussian alpha > 0.01.",
    ),
    FeatureSpec(
        name="influence_mass",
        group="attribution",
        source="attribution pipeline",
        units="summed alpha-blending transmittance",
        min_val=0.0,
        max_val=1e6,
        normalization="train_z_score",
        leakage_status="clean_pre_intervention",
        description="Integrated attribution mass across all influenced screen pixels.",
    ),
    FeatureSpec(
        name="position_drift",
        group="temporal",
        source="GaussianStateStore",
        units="meters",
        min_val=0.0,
        max_val=10.0,
        normalization="train_z_score",
        leakage_status="clean_history",
        description="L2 displacement ||mu(t) - mu(t_prev)||_2 from GaussianStateStore.",
    ),
    FeatureSpec(
        name="residual_drift_ema",
        group="temporal",
        source="GaussianStateStore",
        units="EMA units",
        min_val=0.0,
        max_val=10.0,
        normalization="train_z_score",
        leakage_status="clean_history",
        description="Exponential moving average of photometric residuals tracked in StateStore.",
    ),
    FeatureSpec(
        name="uncertainty_var",
        group="uncertainty",
        source="GaussianStateStore",
        units="variance units",
        min_val=0.0,
        max_val=10.0,
        normalization="train_z_score",
        leakage_status="clean_history",
        description="State uncertainty variance accumulated across past frames.",
    ),
    FeatureSpec(
        name="projected_area",
        group="cost",
        source="projection / renderer",
        units="pixels^2",
        min_val=0.0,
        max_val=1e6,
        normalization="train_z_score",
        leakage_status="clean_pre_intervention",
        description="Screen-space bounding box or ellipse area of the projected Gaussian.",
    ),
    FeatureSpec(
        name="update_frequency",
        group="temporal",
        source="GaussianStateStore",
        units="ratio in [0, 1]",
        min_val=0.0,
        max_val=1.0,
        normalization="train_z_score",
        leakage_status="clean_history",
        description="Fraction of recent frames where this Gaussian received active gradient updates.",
    ),
    FeatureSpec(
        name="age",
        group="lifecycle",
        source="GaussianStateStore",
        units="integer frames",
        min_val=0.0,
        max_val=1e5,
        normalization="train_z_score",
        leakage_status="clean_history",
        description="Age in frames: t_current - t_creation.",
    ),
]

class SchemaError(ValueError):
    """Raised when a dataset row violates the canonical schema contract."""
    pass

DatasetSchemaError = SchemaError


CANONICAL_FEATURE_NAMES: List[str] = [spec.name for spec in CANONICAL_FEATURE_SPECS]

UTILITY_FEATURES: Tuple[str, ...] = tuple(CANONICAL_FEATURE_NAMES)

FEATURE_SPEC_BY_NAME: Dict[str, FeatureSpec] = {spec.name: spec for spec in CANONICAL_FEATURE_SPECS}

# Feature ablation ladder V0-V7 (Phase 6 taxonomy)
ABLATION_SUBSETS: Dict[str, List[str]] = {
    "V0: RGB Error": [
        "rgb_error",
    ],
    "V1: + Depth Error": [
        "rgb_error", "depth_error",
    ],
    "V2: + Gradient Norm": [
        "rgb_error", "depth_error", "gradient_norm",
    ],
    "V3: + Visibility": [
        "rgb_error", "depth_error", "gradient_norm", "visibility_count",
    ],
    "V4: + Influence Mass": [
        "rgb_error", "depth_error", "gradient_norm", "visibility_count",
        "influence_mass",
    ],
    "V5: + Temporal State": [
        "rgb_error", "depth_error", "gradient_norm", "visibility_count",
        "influence_mass", "position_drift", "residual_drift_ema",
    ],
    "V6: + Uncertainty": [
        "rgb_error", "depth_error", "gradient_norm", "visibility_count",
        "influence_mass", "position_drift", "residual_drift_ema", "uncertainty_var",
    ],
    "V7: + Cost & Lifecycle": [
        "rgb_error", "depth_error", "gradient_norm", "visibility_count",
        "influence_mass", "position_drift", "residual_drift_ema", "uncertainty_var",
        "projected_area", "update_frequency", "age",
    ],
}


def get_canonical_feature_names() -> List[str]:
    """Return the list of 11 canonical feature names in deterministic order."""
    return list(CANONICAL_FEATURE_NAMES)


def get_feature_specs() -> List[FeatureSpec]:
    """Return all feature specifications."""
    return list(CANONICAL_FEATURE_SPECS)


def extract_feature_vector(row: Dict[str, Any], strict: bool = True) -> np.ndarray:
    """Extract canonical 11-dimensional feature vector s_i(t) from a raw oracle row.
    
    Args:
        row: Sample dictionary from oracle dataset.
        strict: If True, raises SchemaError if any canonical feature is missing.
                If False, falls back to legacy aliases or defaults.
    """
    f = row.get("features")
    if f is None or not isinstance(f, dict):
        if strict:
            raise SchemaError(f"Row {row.get('gaussian_id', '?')} is missing 'features' dictionary.")
        f = {}

    if strict:
        missing = [name for name in UTILITY_FEATURES if name not in f]
        if missing:
            raise SchemaError(f"Row {row.get('gaussian_id', '?')} missing required canonical features: {missing}")
        vec = [float(f[name]) for name in UTILITY_FEATURES]
    else:
        vec = [
            float(f.get("rgb_error", row.get("rgb_error", 0.0))),
            float(f.get("depth_error", row.get("depth_error", 0.0))),
            float(f.get("gradient_norm", row.get("gradient_norm", 0.0))),
            float(f.get("visibility_count", f.get("visibility", row.get("visibility_count", 0.0)))),
            float(f.get("influence_mass", row.get("influence_mass", 1.0))),
            float(f.get("position_drift", row.get("position_drift", 0.0))),
            float(f.get("residual_drift_ema", row.get("residual_drift_ema", 0.0))),
            float(f.get("uncertainty_var", f.get("uncertainty", row.get("uncertainty_var", 0.0)))),
            float(f.get("projected_area", row.get("projected_area", 1.0))),
            float(f.get("update_frequency", row.get("update_frequency", 0.0))),
            float(f.get("age", row.get("age", 1.0))),
        ]

    return np.array(vec, dtype=np.float32)


def generate_feature_schema_table() -> str:
    """Generate Markdown documentation table for the canonical feature schema."""
    lines = [
        "| Index | Feature Name | Group | Source | Units | Range | Normalization | Leakage Status | Description |",
        "|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---|",
    ]
    for idx, spec in enumerate(CANONICAL_FEATURE_SPECS):
        lines.append(
            f"| {idx} | `{spec.name}` | {spec.group} | {spec.source} | {spec.units} | "
            f"[{spec.min_val}, {spec.max_val}] | `{spec.normalization}` | {spec.leakage_status} | {spec.description} |"
        )
    return "\n".join(lines)
