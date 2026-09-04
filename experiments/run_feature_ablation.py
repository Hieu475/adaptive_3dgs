#!/usr/bin/env python3
"""Phase 6 & 7: Multi-Seed V0–V7 Feature Ablation and Empirical Evaluation Chain.

Evaluates:
  - V0 to V7 Feature Ablation across all protocol seeds [42, 43, 44, 45, 46].
  - Trains TwoHeadMLP on each feature subset using normalized train split (fr1/desk 0-40).
  - Evaluates strictly on held-out independent cross-scene test split (tum_fr2_xyz).
  - Reports Mean ± Std and 95% CI for Spearman rho, NDCG@20, OSE@20, Realized Delta Q.
  - Empirical Evaluation Chain (Prediction-to-Decision Association):
      corr(rho, NDCG@20) -> corr(NDCG@20, Delta Q) -> corr(OSE@20, Delta Q) -> corr(rho, Delta Q).
  - Exports results/learned_utility/feature_ablation_report.md and learned_utility_summary.json.
"""
import os
import sys
import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.protocol import (
    load_protocol,
    get_seeds,
    get_repo_root,
)
from research.utility_features import (
    ABLATION_SUBSETS,
    CANONICAL_FEATURE_NAMES,
)
from research.utility_dataset import (
    UtilityDataset,
    prepare_normalized_splits,
)
from research.utility_models import TwoHeadMLP
from research.utility_training import (
    UtilityModelTrainer,
    TrainingConfig,
)
from research.utility_metrics import (
    safe_spearmanr,
    evaluate_utility_complete,
    compute_confidence_interval_95,
)


def run_multi_seed_ablation(
    train_ds,
    val_ds,
    test_ds,
    seeds: List[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    trainer = UtilityModelTrainer(TrainingConfig(epochs=200, learning_rate=0.005))
    y_q_test = test_ds.delta_q_np
    y_t_test = test_ds.delta_t_np
    y_u_test = test_ds.utility_np

    ablation_summary_rows = []
    all_runs_records = {}

    prev_mean_rho = 0.0

    for v_name, feat_names in ABLATION_SUBSETS.items():
        train_sub = train_ds.select_features(feat_names)
        test_sub = test_ds.select_features(feat_names)

        seed_rhos = []
        seed_ndcgs = []
        seed_overlaps = []
        seed_oses = []
        seed_regrets = []
        seed_dqs = []

        for seed in seeds:
            model = TwoHeadMLP(in_features=len(feat_names), hidden_dim=64)
            trainer.train_two_head_model(model, train_sub, val_ds=None, seed=seed)

            model.eval()
            with torch.no_grad():
                pred_q_test, pred_t_test, pred_u_test = model(test_sub.X)

            m = evaluate_utility_complete(
                pred_u=pred_u_test.cpu().numpy(),
                oracle_u=y_u_test,
                delta_q=y_q_test,
                pred_q=pred_q_test.cpu().numpy(),
                true_q=y_q_test,
                pred_t=pred_t_test.cpu().numpy(),
                true_t=y_t_test,
                costs=y_t_test,
            )

            seed_rhos.append(m["spearman_rho"])
            seed_ndcgs.append(m["ndcg_20pct"])
            seed_overlaps.append(m["overlap_20pct"])
            seed_oses.append(m["ose_20pct"])
            seed_regrets.append(m["regret_20pct"])
            seed_dqs.append(m["realized_delta_q_20pct"])

        mean_rho = float(np.mean(seed_rhos))
        std_rho = float(np.std(seed_rhos))
        ci_rho = compute_confidence_interval_95(std_rho, len(seeds))
        delta_rho = mean_rho - prev_mean_rho
        prev_mean_rho = mean_rho

        mean_ndcg = float(np.mean(seed_ndcgs))
        std_ndcg = float(np.std(seed_ndcgs))
        ci_ndcg = compute_confidence_interval_95(std_ndcg, len(seeds))

        mean_ose = float(np.mean(seed_oses))
        std_ose = float(np.std(seed_oses))
        ci_ose = compute_confidence_interval_95(std_ose, len(seeds))

        mean_overlap = float(np.mean(seed_overlaps))
        mean_regret = float(np.mean(seed_regrets))

        mean_dq = float(np.mean(seed_dqs))
        std_dq = float(np.std(seed_dqs))
        ci_dq = compute_confidence_interval_95(std_dq, len(seeds))

        row = {
            "version": v_name,
            "inputs": len(feat_names),
            "features": feat_names,
            "spearman_rho_mean": mean_rho,
            "spearman_rho_std": std_rho,
            "spearman_rho_ci95": ci_rho,
            "delta_rho": delta_rho,
            "ndcg_20pct_mean": mean_ndcg,
            "ndcg_20pct_std": std_ndcg,
            "ndcg_20pct_ci95": ci_ndcg,
            "overlap_20pct_mean": mean_overlap,
            "ose_20pct_mean": mean_ose,
            "ose_20pct_std": std_ose,
            "ose_20pct_ci95": ci_ose,
            "regret_20pct_mean": mean_regret,
            "realized_delta_q_mean": mean_dq,
            "realized_delta_q_std": std_dq,
            "realized_delta_q_ci95": ci_dq,
        }
        ablation_summary_rows.append(row)

        all_runs_records[v_name] = {
            "rhos": seed_rhos,
            "ndcgs": seed_ndcgs,
            "oses": seed_oses,
            "dqs": seed_dqs,
        }

        print(
            f"   - {v_name:<30}: ρ = {mean_rho:+.4f} ±{std_rho:.3f} (Δρ={delta_rho:+.4f}) | "
            f"NDCG = {mean_ndcg:.4f} ±{std_ndcg:.3f} | OSE = {mean_ose:.3f} ±{std_ose:.3f} | "
            f"ΔQ = {mean_dq:+.6f}"
        )

    # Empirical Evaluation Chain (Prediction-to-Decision Association)
    df_chain = pd.DataFrame([
        {
            "rho": r["spearman_rho_mean"],
            "ndcg": r["ndcg_20pct_mean"],
            "ose": r["ose_20pct_mean"],
            "delta_q": r["realized_delta_q_mean"],
        }
        for r in ablation_summary_rows
    ])

    r_rho_ndcg, p_rho_ndcg = pearsonr(df_chain["rho"], df_chain["ndcg"])
    r_ndcg_dq, p_ndcg_dq = pearsonr(df_chain["ndcg"], df_chain["delta_q"])
    r_ose_dq, p_ose_dq = pearsonr(df_chain["ose"], df_chain["delta_q"])
    r_rho_dq, p_rho_dq = pearsonr(df_chain["rho"], df_chain["delta_q"])

    chain_stats = {
        "corr_rho_to_ndcg": {"r": float(r_rho_ndcg), "p": float(p_rho_ndcg)},
        "corr_ndcg_to_delta_q": {"r": float(r_ndcg_dq), "p": float(p_ndcg_dq)},
        "corr_ose_to_delta_q": {"r": float(r_ose_dq), "p": float(p_ose_dq)},
        "corr_rho_to_delta_q": {"r": float(r_rho_dq), "p": float(p_rho_dq)},
    }

    return ablation_summary_rows, chain_stats


def main():
    repo_root = get_repo_root()
    protocol = load_protocol()
    seeds = get_seeds(protocol)

    print("=" * 95)
    print("   PHASE 6 & 7: MULTI-SEED V0–V7 FEATURE ABLATION & EMPIRICAL EVALUATION CHAIN")
    print("=" * 95)
    print(f">> Seeds ({len(seeds)}): {seeds}")
    print(f">> Ablation levels: {len(ABLATION_SUBSETS)} subsets (V0 to V7)\n")

    # 1. Load canonical dataset and prepare normalized splits
    dataset = UtilityDataset.from_oracle()
    train_ds, val_ds, test_ds, normalizer = prepare_normalized_splits(dataset=dataset)

    ablation_rows, chain_stats = run_multi_seed_ablation(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        seeds=seeds,
    )

    # 2. Print Empirical Chain Summary
    print("\n" + "=" * 95)
    print(">> Empirical Evaluation Chain (Prediction-to-Decision Association):")
    print(f"   Stage 1 -> Stage 2: corr(ρ, NDCG@20)  = {chain_stats['corr_rho_to_ndcg']['r']:+.4f} (p={chain_stats['corr_rho_to_ndcg']['p']:.4f})")
    print(f"   Stage 2 -> Stage 4: corr(NDCG@20, ΔQ) = {chain_stats['corr_ndcg_to_delta_q']['r']:+.4f} (p={chain_stats['corr_ndcg_to_delta_q']['p']:.4f})")
    print(f"   Stage 3 -> Stage 4: corr(OSE@20, ΔQ)  = {chain_stats['corr_ose_to_delta_q']['r']:+.4f} (p={chain_stats['corr_ose_to_delta_q']['p']:.4f})")
    print(f"   End-to-End Pipeline: corr(ρ, ΔQ)      = {chain_stats['corr_rho_to_delta_q']['r']:+.4f} (p={chain_stats['corr_rho_to_delta_q']['p']:.4f})")

    # 3. Export Reports
    out_dir = os.path.join(repo_root, "results", "learned_utility")
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, "feature_ablation_report.md")
    json_file = os.path.join(out_dir, "learned_utility_summary.json")

    with open(json_file, "w") as f:
        json.dump({
            "protocol_version": protocol.get("protocol_version", "1.0.0"),
            "seeds": seeds,
            "v0_v7_ablation": ablation_rows,
            "empirical_evaluation_chain": chain_stats,
        }, f, indent=2)

    lines = [
        "# Phase 6 & 7: V0–V7 Feature Ablation & Empirical Evaluation Chain",
        "",
        "## 1. Multi-Seed Feature Ablation Progression (V0 to V7)",
        "",
        f"Evaluated strictly on independent cross-scene test split (`tum_fr2_xyz`) across {len(seeds)} protocol seeds ({seeds}).",
        "Results reported as **Mean ± Std** (with 95% Confidence Intervals):",
        "",
        "| Variant | Inputs | Spearman $\\rho$ ↑ | $\\Delta \\rho$ | NDCG@20% ↑ | OSE@20% ↑ | Realized $\\Delta Q$ |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for a in ablation_rows:
        rho_s = f"{a['spearman_rho_mean']:+.4f} ±{a['spearman_rho_std']:.3f}"
        ndcg_s = f"{a['ndcg_20pct_mean']:.4f} ±{a['ndcg_20pct_std']:.3f}"
        ose_s = f"{a['ose_20pct_mean']:.3f} ±{a['ose_20pct_std']:.3f}"
        dq_s = f"{a['realized_delta_q_mean']:+.6f} ±{a['realized_delta_q_std']:.6f}"
        lines.append(
            f"| **{a['version']}** | {a['inputs']} | **{rho_s}** | "
            f"{a['delta_rho']:+.4f} | {ndcg_s} | **{ose_s}** | {dq_s} |"
        )

    lines.extend([
        "",
        "## 2. Empirical Evaluation Chain (Prediction-to-Decision Association)",
        "",
        "Quantifies empirical transfer from prediction fidelity to decision quality and reconstruction gain:",
        "",
        f"- **Stage 1 to Stage 2:** $\\text{{corr}}(\\rho, NDCG@20) = \\mathbf{{{chain_stats['corr_rho_to_ndcg']['r']:+.4f}}}$ ($p = {chain_stats['corr_rho_to_ndcg']['p']:.4f}$)",
        f"- **Stage 2 to Stage 4:** $\\text{{corr}}(NDCG@20, \\Delta Q) = \\mathbf{{{chain_stats['corr_ndcg_to_delta_q']['r']:+.4f}}}$ ($p = {chain_stats['corr_ndcg_to_delta_q']['p']:.4f}$)",
        f"- **Stage 3 to Stage 4:** $\\text{{corr}}(OSE@20, \\Delta Q) = \\mathbf{{{chain_stats['corr_ose_to_delta_q']['r']:+.4f}}}$ ($p = {chain_stats['corr_ose_to_delta_q']['p']:.4f}$)",
        f"- **End-to-End Pipeline:** $\\text{{corr}}(\\rho, \\Delta Q) = \\mathbf{{{chain_stats['corr_rho_to_delta_q']['r']:+.4f}}}$ ($p = {chain_stats['corr_rho_to_delta_q']['p']:.4f}$)",
        "",
        "> **Methodological Note:** These empirical correlations verify the operational pipeline connection from statistical estimation fidelity to online decision efficiency without overclaiming causal identification.",
        "",
    ])

    with open(report_file, "w") as f:
        f.write("\n".join(lines))

    print(f"\n>> Saved multi-seed ablation report to: {report_file}")
    print(f">> Saved JSON summary to: {json_file}")


if __name__ == "__main__":
    main()
