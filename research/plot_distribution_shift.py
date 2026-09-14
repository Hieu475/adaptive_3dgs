"""Figure 19 & Distribution Shift Analysis for Phase 9A.

Quantifies and visualizes the distribution shift between tum_fr1_desk (train)
and tum_fr2_xyz (test) before (raw) and after (relative) scale-invariant transformation.

Generates:
    - results/phase9a_scale_invariant/representation_metrics.csv
    - results/phase9a_scale_invariant/figures/fig19_distribution_shift.png
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.phase9_protocol import (
    CANONICAL_FEATURE_SCHEMA,
    get_output_dir,
    get_repo_root,
)
from research.phase9_features import (
    FEAT_IDX,
    transform_grouped_features,
)


def analyze_feature_distributions() -> pd.DataFrame:
    """Compute statistical distribution metrics between fr1 and fr2 across representations."""
    repo_root = get_repo_root()
    dataset_path = repo_root / "results" / "oracle_dataset" / "oracle_dataset.json"

    with open(dataset_path) as f:
        data = json.load(f)

    fr1_samples = [d for d in data if d["scene"] == "tum_fr1_desk"]
    fr2_samples = [d for d in data if d["scene"] == "tum_fr2_xyz"]

    def extract_X_and_keys(samples):
        X = np.zeros((len(samples), 11), dtype=np.float32)
        keys = []
        for i, s in enumerate(samples):
            feats = s["features"]
            for j, name in enumerate(CANONICAL_FEATURE_SCHEMA):
                X[i, j] = float(feats.get(name, 0.0))
            keys.append((s["scene"], s["frame"]))
        return X, keys

    X_fr1, keys_fr1 = extract_X_and_keys(fr1_samples)
    X_fr2, keys_fr2 = extract_X_and_keys(fr2_samples)

    # A0: Raw
    Z_fr1_a0, Z_fr2_a0 = X_fr1, X_fr2

    # A1: Geometry Relative
    Z_fr1_a1 = transform_grouped_features(X_fr1, keys_fr1, variant="geometry_relative")
    Z_fr2_a1 = transform_grouped_features(X_fr2, keys_fr2, variant="geometry_relative")

    # A2: Geometry + Optimization Relative
    Z_fr1_a2 = transform_grouped_features(X_fr1, keys_fr1, variant="geometry_optimization_relative")
    Z_fr2_a2 = transform_grouped_features(X_fr2, keys_fr2, variant="geometry_optimization_relative")

    rows = []
    variants_data = [
        ("A0_raw", Z_fr1_a0, Z_fr2_a0),
        ("A1_geometry_relative", Z_fr1_a1, Z_fr2_a1),
        ("A2_geometry_optimization_relative", Z_fr1_a2, Z_fr2_a2),
    ]

    for var_name, v_fr1, v_fr2 in variants_data:
        for j, feat_name in enumerate(CANONICAL_FEATURE_SCHEMA):
            vals1 = v_fr1[:, j]
            vals2 = v_fr2[:, j]

            w_dist = float(wasserstein_distance(vals1, vals2))
            ks_res = ks_2samp(vals1, vals2)

            med1 = float(np.median(vals1))
            med2 = float(np.median(vals2))
            mean1 = float(np.mean(vals1))
            mean2 = float(np.mean(vals2))
            std1 = float(np.std(vals1))
            std2 = float(np.std(vals2))

            rows.append({
                "variant": var_name,
                "feature_index": j,
                "feature_name": feat_name,
                "fr1_median": med1,
                "fr2_median": med2,
                "median_ratio_fr2_fr1": med2 / (med1 + 1e-6),
                "fr1_mean": mean1,
                "fr2_mean": mean2,
                "fr1_std": std1,
                "fr2_std": std2,
                "wasserstein_distance": w_dist,
                "ks_statistic": float(ks_res.statistic),
                "ks_pvalue": float(ks_res.pvalue),
            })

    df = pd.DataFrame(rows)
    out_dir = get_output_dir()
    csv_path = out_dir / "representation_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved representation metrics to {csv_path}")

    # Generate Figure 19
    plot_figure_19(X_fr1, X_fr2, Z_fr1_a1, Z_fr2_a1, Z_fr1_a2, Z_fr2_a2)

    return df


def plot_figure_19(
    X_fr1: np.ndarray,
    X_fr2: np.ndarray,
    Z_fr1_a1: np.ndarray,
    Z_fr2_a1: np.ndarray,
    Z_fr1_a2: np.ndarray,
    Z_fr2_a2: np.ndarray,
) -> None:
    """Generate Figure 19: Multi-panel distribution shift comparison (Raw vs Transformed)."""
    fig_dir = get_output_dir() / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "fig19_distribution_shift.png"

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panels:
    # (0, 0): depth_error raw vs A1
    # (0, 1): projected_area raw vs A1
    # (0, 2): gradient_norm raw vs A2
    # (1, 0): influence_mass raw vs A2
    # (1, 1): uncertainty_var raw vs A2
    # (1, 2): summary Wasserstein distance comparison

    c_fr1 = "#1f77b4"  # blue
    c_fr2 = "#d62728"  # red

    # 1. Depth error: Raw vs A1
    ax = axes[0, 0]
    idx = FEAT_IDX["depth_error"]
    bins = np.linspace(0, 1.5, 30)
    ax.hist(X_fr1[:, idx], bins=bins, alpha=0.5, density=True, color=c_fr1, label=f"fr1 (Raw med={np.median(X_fr1[:, idx]):.2f})")
    ax.hist(X_fr2[:, idx], bins=bins, alpha=0.5, density=True, color=c_fr2, label=f"fr2 (Raw med={np.median(X_fr2[:, idx]):.2f})")
    ax.set_title("depth_error (Raw A0) [Meters]", fontweight="bold", fontsize=11)
    ax.set_xlabel("Depth Error (m)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Depth error: A1
    ax = axes[0, 1]
    bins_rel = np.linspace(0, 4.0, 30)
    ax.hist(Z_fr1_a1[:, idx], bins=bins_rel, alpha=0.5, density=True, color=c_fr1, label=f"fr1 (A1 med={np.median(Z_fr1_a1[:, idx]):.2f})")
    ax.hist(Z_fr2_a1[:, idx], bins=bins_rel, alpha=0.5, density=True, color=c_fr2, label=f"fr2 (A1 med={np.median(Z_fr2_a1[:, idx]):.2f})")
    ax.set_title("depth_error (Relative A1) [Unitless]", fontweight="bold", fontsize=11)
    ax.set_xlabel("Relative Depth Error (e / median)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Projected area: Raw vs A1
    ax = axes[0, 2]
    idx_a = FEAT_IDX["projected_area"]
    bins_a = np.linspace(0, 30, 30)
    ax.hist(X_fr1[:, idx_a], bins=bins_a, alpha=0.5, density=True, color=c_fr1, label=f"fr1 (Raw med={np.median(X_fr1[:, idx_a]):.1f})")
    ax.hist(X_fr2[:, idx_a], bins=bins_a, alpha=0.5, density=True, color=c_fr2, label=f"fr2 (Raw med={np.median(X_fr2[:, idx_a]):.1f})")
    ax.set_title("projected_area (Raw A0) [Pixels^2]", fontweight="bold", fontsize=11)
    ax.set_xlabel("Screen Area")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Projected area: A1
    ax = axes[1, 0]
    bins_arel = np.linspace(0, 4.0, 30)
    ax.hist(Z_fr1_a1[:, idx_a], bins=bins_arel, alpha=0.5, density=True, color=c_fr1, label=f"fr1 (A1 med={np.median(Z_fr1_a1[:, idx_a]):.2f})")
    ax.hist(Z_fr2_a1[:, idx_a], bins=bins_arel, alpha=0.5, density=True, color=c_fr2, label=f"fr2 (A1 med={np.median(Z_fr2_a1[:, idx_a]):.2f})")
    ax.set_title("projected_area (Relative A1) [Unitless]", fontweight="bold", fontsize=11)
    ax.set_xlabel("Relative Area (A / median)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. Gradient norm: Raw vs A2
    ax = axes[1, 1]
    idx_g = FEAT_IDX["gradient_norm"]
    bins_g = np.linspace(0, 20, 30)
    ax.hist(X_fr1[:, idx_g], bins=bins_g, alpha=0.5, density=True, color=c_fr1, label=f"fr1 (Raw med={np.median(X_fr1[:, idx_g]):.2f})")
    ax.hist(X_fr2[:, idx_g], bins=bins_g, alpha=0.5, density=True, color=c_fr2, label=f"fr2 (Raw med={np.median(X_fr2[:, idx_g]):.2f})")
    ax.set_title("gradient_norm (Raw A0 vs fr1/fr2)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Gradient Norm ||dLoss/dmu||")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 6. Summary: Median Alignment Table / Bar Chart
    ax = axes[1, 2]
    feats_to_show = ["depth_error", "projected_area", "gradient_norm", "influence_mass"]
    raw_ratios = []
    a1_ratios = []
    for f in feats_to_show:
        i = FEAT_IDX[f]
        raw_ratios.append(float(np.median(X_fr2[:, i]) / (np.median(X_fr1[:, i]) + 1e-6)))
        a1_ratios.append(float(np.median(Z_fr2_a1[:, i]) / (np.median(Z_fr1_a1[:, i]) + 1e-6)))

    x = np.arange(len(feats_to_show))
    width = 0.35
    ax.bar(x - width/2, raw_ratios, width, label="Raw Ratio (fr2/fr1)", color="#e74c3c")
    ax.bar(x + width/2, a1_ratios, width, label="A1 Relative Ratio (fr2/fr1)", color="#2ecc71")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Perfect Invariance (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in feats_to_show], fontsize=9)
    ax.set_title("Scale Invariance Ratio (Ideal = 1.0)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Median Ratio (fr2 / fr1)")
    ax.set_ylim(0, 1.6)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 19: Distribution Shift Alignment Under Scale-Invariant Representation (Phase 9A)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 19 to {fig_path}")


if __name__ == "__main__":
    analyze_feature_distributions()
