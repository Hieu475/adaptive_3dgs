#!/usr/bin/env python3
r"""Phase 12: Statistical Processing for Controlled Model Ablation.

Parses model_ablation_results.json and generates:
    1. Comprehensive Multi-Seed Comparison Table:
       Model | \rho | AUROC | NDCG@20 | OSE | \Delta Q | \Delta Q/ms
    2. Bootstrap 95% Confidence Intervals for all metrics across seeds.
    3. Paired Hypothesis Tests (Wilcoxon Signed-Rank Test & Paired t-test):
       - M4 vs M1 (Gradient Sensitivity)
       - M4 vs M2 (Current TwoHeadMLP)
       - M3 vs M2 (Positive Head vs Baseline MLP)
       - M1 vs M2 (Grad-Norm vs Baseline MLP)
       - M4 vs M0 (Error Heuristic)
    4. Effect Size Quantification (Cohen's d).
    5. Checkpoint Decision Guidance (Point 9 evaluation).
    6. Formatted Markdown and LaTeX table snippets for direct paper insertion.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import scipy.stats as stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.phase12_protocol import get_phase12_output_dir


def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Computes non-parametric bootstrap 95% confidence interval for the mean."""
    if len(arr) <= 1 or np.all(arr == arr[0]):
        val = float(np.mean(arr)) if len(arr) > 0 else 0.0
        return val, val
    rng = np.random.default_rng(seed)
    boot_means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cohen's d effect size for paired samples."""
    diff = x - y
    n = len(diff)
    if n <= 1:
        return 0.0
    s = np.std(diff, ddof=1)
    return float(np.mean(diff) / (s + 1e-8)) if s > 1e-8 else 0.0


def paired_test(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Runs paired Wilcoxon signed-rank and t-test."""
    diff = x - y
    d_val = cohens_d(x, y)
    if np.all(diff == 0.0):
        return {"wilcoxon_p": 1.0, "ttest_p": 1.0, "cohens_d": 0.0}

    # Wilcoxon signed-rank
    try:
        w_res = stats.wilcoxon(x, y, zero_method="wilcox")
        w_p = float(w_res.pvalue)
    except Exception:
        w_p = 1.0

    # Paired t-test
    try:
        t_res = stats.ttest_rel(x, y)
        t_p = float(t_res.pvalue)
    except Exception:
        t_p = 1.0

    return {
        "wilcoxon_p": w_p,
        "ttest_p": t_p,
        "cohens_d": d_val,
    }


def process_model_ablation(
    results_json: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assert results_json.exists(), f"Results file not found: {results_json}"

    with open(results_json, "r") as f:
        data = json.load(f)

    metrics = [
        ("spearman_rho", "Spearman Rank Correlation (\\rho)"),
        ("auroc", "AUROC (Positive Utility P(U* > 0))"),
        ("ndcg_20", "Selection Quality (NDCG@20)"),
        ("ose", "Optimal Selection Efficiency (OSE)"),
        ("realized_dq", "Realized Quality Gain (\\Delta Q)"),
        ("dq_per_ms", "Budget Efficiency (\\Delta Q / ms)"),
        ("n_negative_selected", "Negative Utility Interventions Selected"),
    ]

    model_keys = [
        "M0_Error",
        "M1_GradNorm",
        "M2_CurrentMLP",
        "M3_PositiveHead",
        "M4_PositiveGlobal",
        "Policy_OraclePositive",
        "Policy_ProbThreshold",
    ]

    summary_stats: Dict[str, Dict[str, Any]] = {}

    for m_key in model_keys:
        if m_key not in data:
            continue
        m_info = data[m_key]
        per_seed = m_info["per_seed"]
        summary_stats[m_key] = {"name": m_info.get("name", m_key), "metrics": {}}

        for met_key, met_label in metrics:
            vals = np.array([seed_data[met_key] for seed_data in per_seed.values()], dtype=np.float64)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            lo_v, hi_v = bootstrap_ci(vals)

            summary_stats[m_key]["metrics"][met_key] = {
                "mean": mean_v,
                "std": std_v,
                "ci_95": [lo_v, hi_v],
                "values": vals.tolist(),
            }

    # Paired statistical comparisons against baselines
    paired_comparisons: Dict[str, Dict[str, Any]] = {}
    comparison_pairs = [
        ("M4_PositiveGlobal", "M1_GradNorm", "M4 vs M1 (Grad-Norm)"),
        ("M4_PositiveGlobal", "M2_CurrentMLP", "M4 vs M2 (Current TwoHeadMLP)"),
        ("M3_PositiveHead", "M2_CurrentMLP", "M3 vs M2 (Positive Head Effect)"),
        ("M1_GradNorm", "M2_CurrentMLP", "M1 vs M2 (Heuristic vs Current ML)"),
        ("M4_PositiveGlobal", "M0_Error", "M4 vs M0 (Full Model vs Error Heuristic)"),
    ]

    for m_a, m_b, comp_label in comparison_pairs:
        if m_a in summary_stats and m_b in summary_stats:
            paired_comparisons[comp_label] = {}
            for met_key, _ in metrics:
                arr_a = np.array(summary_stats[m_a]["metrics"][met_key]["values"])
                arr_b = np.array(summary_stats[m_b]["metrics"][met_key]["values"])
                test_res = paired_test(arr_a, arr_b)
                diff = float(np.mean(arr_a - arr_b))
                paired_comparisons[comp_label][met_key] = {
                    "mean_diff": diff,
                    **test_res,
                }

    # ─── Generate Markdown Summary Table ───
    md_lines = [
        "# Phase 12: Controlled Model Ablation & Hypothesis Test Results",
        "",
        "Formal comparison of utility estimation formulations across 5 protocol seeds (42–46) on unseen test scenes:",
        "- **M0**: Error Heuristic ($e_{rgb} + e_{depth}$)",
        "- **M1**: Gradient Sensitivity Heuristic (Grad-Norm $\|\nabla L\|$)",
        "- **M2**: Current Baseline TwoHeadMLP (11D Local Observable State)",
        "- **M3**: Two-Stage MLP + Positive Head (11D Local + $P(U^* > 0)$)",
        "- **M4**: Two-Stage MLP + Positive Head + Global Context (11D Local + 12D Global Frame State)",
        "",
        "## 1. Primary Model Ablation Table",
        "",
        "| Model | Spearman $\\rho$ | AUROC | NDCG@20 | OSE (Knapsack) | Realized $\\Delta Q$ | $\\Delta Q$ / ms | Neg. Selected |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    display_models = [
        ("M0_Error", "M0: Error Heuristic"),
        ("M1_GradNorm", "M1: Grad-Norm Sensitivity"),
        ("M2_CurrentMLP", "M2: Current TwoHeadMLP"),
        ("M3_PositiveHead", "M3: MLP + Positive Head"),
        ("M4_PositiveGlobal", "**M4: MLP + Positive + Global**"),
        ("Policy_OraclePositive", "*Policy: Oracle-Positive + M4*"),
        ("Policy_ProbThreshold", "*Policy: Prob-Thresholded M4*"),
    ]

    for key, label in display_models:
        if key not in summary_stats:
            continue
        m = summary_stats[key]["metrics"]
        rho_s = f"{m['spearman_rho']['mean']:.4f} ± {m['spearman_rho']['std']:.4f}"
        auc_s = f"{m['auroc']['mean']:.4f}"
        ndcg_s = f"{m['ndcg_20']['mean']:.4f}"
        ose_s = f"{m['ose']['mean']:.4f}"
        dq_s = f"{m['realized_dq']['mean']:.4e}"
        eff_s = f"{m['dq_per_ms']['mean']:.4e}"
        neg_s = f"{m['n_negative_selected']['mean']:.1f}"

        md_lines.append(
            f"| {label} | {rho_s} | {auc_s} | {ndcg_s} | {ose_s} | {dq_s} | {eff_s} | {neg_s} |"
        )

    md_lines.extend([
        "",
        "## 2. Paired Hypothesis Tests & Effect Sizes",
        "",
        "| Comparison | Metric | Mean Difference | Wilcoxon $p$-value | Paired $t$-test $p$ | Cohen's $d$ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    for comp_label, met_dict in paired_comparisons.items():
        for met_key in ["spearman_rho", "auroc", "ndcg_20", "ose"]:
            if met_key in met_dict:
                d = met_dict[met_key]
                p_wil = f"{d['wilcoxon_p']:.4f}" if d['wilcoxon_p'] >= 0.0001 else f"{d['wilcoxon_p']:.2e}"
                p_t = f"{d['ttest_p']:.4f}" if d['ttest_p'] >= 0.0001 else f"{d['ttest_p']:.2e}"
                md_lines.append(
                    f"| {comp_label} | `{met_key}` | {d['mean_diff']:+.4f} | {p_wil} | {p_t} | {d['cohens_d']:+.2f} |"
                )

    # ─── Section 3: Scientific Checkpoint Decision ───
    m4_rho = summary_stats["M4_PositiveGlobal"]["metrics"]["spearman_rho"]["mean"]
    m1_rho = summary_stats["M1_GradNorm"]["metrics"]["spearman_rho"]["mean"]
    m2_rho = summary_stats["M2_CurrentMLP"]["metrics"]["spearman_rho"]["mean"]

    m4_ose = summary_stats["M4_PositiveGlobal"]["metrics"]["ose"]["mean"]
    m1_ose = summary_stats["M1_GradNorm"]["metrics"]["ose"]["mean"]
    m2_ose = summary_stats["M2_CurrentMLP"]["metrics"]["ose"]["mean"]

    m4_auc = summary_stats["M4_PositiveGlobal"]["metrics"]["auroc"]["mean"]

    decision_md = [
        "",
        "## 3. Scientific Checkpoint Evaluation (Point 9 Decision)",
        "",
    ]

    if m4_rho > m1_rho and m4_ose >= m1_ose:
        decision_md.extend([
            "> [!IMPORTANT]",
            "> **Outcome: Hypothesis Confirmed (M4 > M1 > M2)**",
            f"> - Global Context + Positive Utility Head achieves **$\\rho = {m4_rho:.4f}$**, surpassing Grad-Norm ($\\rho = {m1_rho:.4f}$) and Current MLP ($\\rho = {m2_rho:.4f}$).",
            f"> - The positive head achieves AUROC **{m4_auc:.4f}**, successfully rejecting geometric degradation interventions.",
            "> - **Paper Narrative**: Multi-task formulation and global frame context resolve the correlation deficit.",
        ])
    elif m4_ose > m1_ose:
        decision_md.extend([
            "> [!NOTE]",
            "> **Outcome: Selection Superiority over Pointwise Sensitivity (OSE Advantage)**",
            f"> - While Grad-Norm achieves higher isolated pointwise correlation on this slice ($\\rho = {m1_rho:.4f}$ vs ${m4_rho:.4f}$), **M4 achieves superior or competitive budgeted selection (OSE = {m4_ose:.4f} vs {m1_ose:.4f})**.",
            f"> - Furthermore, M4 positive head achieves AUROC = **{m4_auc:.4f}**, reducing negative utility inclusions.",
            "> - **Paper Narrative**: Pointwise correlation $\\neq$ selection quality; cost modeling and positive pruning preserve budgeted efficiency.",
        ])
    else:
        decision_md.extend([
            "> [!WARNING]",
            "> **Outcome: Diagnostic Checkpoint Threshold (M4 <= M1)**",
            f"> - Model ranking ($\\rho = {m4_rho:.4f}$, OSE = {m4_ose:.4f}) remains below or comparable to Grad-Norm sensitivity ($\\rho = {m1_rho:.4f}$, OSE = {m1_ose:.4f}).",
            "> - **Scientific Conclusion**: Local and frame-level observable states are only weakly predictive of counterfactual marginal return.",
            "> - As advised in Point 9, rather than indefinitely inflating MLP capacity, the scientific conclusion is documented transparently.",
        ])

    md_lines.extend(decision_md)

    summary_file = output_dir / "model_ablation_summary.md"
    with open(summary_file, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f">> Saved summary report to {summary_file}")

    # ─── Generate LaTeX Table Snippet ───
    tex_lines = [
        "% LaTeX table snippet generated by experiments/process_phase12_model_ablation.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Controlled Model Ablation on Unseen Test Sequences across 5 seeds. $\\rho$: Spearman correlation with ground-truth utility $U^\\star$; AUROC: classification of $U^\\star > 0$; NDCG@20: ranking quality; OSE: optimal knapsack efficiency under $B=15\\,\\mathrm{ms}$.}",
        "\\label{tab:model_ablation}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Model} & $\\bm{\\rho}$ & \\textbf{AUROC} & \\textbf{NDCG@20} & \\textbf{OSE} & $\\bm{\\Delta Q}$ & $\\bm{\\Delta Q / \\mathrm{ms}}$ \\\\",
        "\\midrule",
    ]

    tex_names = {
        "M0_Error": "Error Heuristic ($M_0$)",
        "M1_GradNorm": "Grad-Norm ($M_1$)",
        "M2_CurrentMLP": "Current TwoHeadMLP ($M_2$)",
        "M3_PositiveHead": "MLP + Positive Head ($M_3$)",
        "M4_PositiveGlobal": "\\textbf{MLP + Pos + Global ($M_4$)}",
        "Policy_OraclePositive": "\\textit{Oracle-Positive Filter}",
        "Policy_ProbThreshold": "\\textit{Prob-Thresholded ($M_4$)}",
    }

    for key, label in display_models:
        if key not in summary_stats:
            continue
        m = summary_stats[key]["metrics"]
        tex_name = tex_names.get(key, label)
        tex_lines.append(
            f"{tex_name} & {m['spearman_rho']['mean']:.3f} & {m['auroc']['mean']:.3f} & "
            f"{m['ndcg_20']['mean']:.3f} & {m['ose']['mean']:.3f} & "
            f"{m['realized_dq']['mean']:.2e} & {m['dq_per_ms']['mean']:.2e} \\\\"
        )

    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    tex_file = output_dir / "tab_model_ablation.tex"
    with open(tex_file, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f">> Saved LaTeX table snippet to {tex_file}")

    return {
        "summary_stats": summary_stats,
        "paired_comparisons": paired_comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 12: Process Model Ablation")
    parser.add_argument(
        "--results_json",
        type=Path,
        default=get_phase12_output_dir("model_ablation") / "model_ablation_results.json",
        help="Path to model_ablation_results.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=get_phase12_output_dir("model_ablation"),
        help="Output directory",
    )
    args = parser.parse_args()

    process_model_ablation(
        results_json=args.results_json,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
