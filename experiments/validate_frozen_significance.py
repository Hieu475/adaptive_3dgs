import sys
from pathlib import Path
REPO_ROOT = Path("/home/nguyen_quoc_hieu/Documents/adaptive_3dgs")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import pandas as pd
from experiments.run_statistical_validation import (
    bootstrap_ci_95,
    compute_cohens_d,
    safe_wilcoxon,
)

def holm_bonferroni(p_values: list) -> list:
    """Computes Holm-Bonferroni step-down adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj_p = min(1.0, (m - rank) * p_val)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max
    return adjusted


def analyze_frozen_benchmark():
    json_path = REPO_ROOT / "results/phase13_frozen_benchmark/phase13_frozen_results.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    raw_runs = data["raw_runs"]
    df = pd.DataFrame(raw_runs)

    print("=" * 95)
    print("  STATISTICAL SIGNIFICANCE & EFFECT SIZE (PHASE 13 FROZEN BENCHMARK)")
    print("=" * 95)
    print(f"  Source: {json_path}")
    print(f"  Policies: {df['policy'].unique().tolist()}")
    print(f"  Seeds: {sorted(df['seed'].unique().tolist())}")
    print("=" * 95)

    # Extract per-seed metrics for each policy
    metrics_by_policy = {}
    for policy in df['policy'].unique():
        sub = df[df['policy'] == policy].sort_values('seed')
        metrics_by_policy[policy] = {
            'mean_psnr': sub['mean_psnr'].to_numpy(),
            'final_psnr': sub['final_psnr'].to_numpy(),
            'mean_ssim': sub['mean_ssim'].to_numpy(),
            'final_ssim': sub['final_ssim'].to_numpy(),
            'fps': sub['fps'].to_numpy(),
        }

    ours = metrics_by_policy.get('ours')
    comparisons = ['error_only', 'error_influence', 'error_influence_temporal', 'no_op', 'full']

    raw_rows = []
    
    print("\n--- PAIRED STATISTICAL TESTS: OURS vs BASELINES (n=5 Seeds) ---\n")
    print("  Note: For paired Wilcoxon with n=5, min two-sided p is 0.0625; min one-sided p is 0.03125.")
    print("  Reporting both pre-specified two-sided p and pre-specified superiority p (H1: OURS > baseline).\n")

    for comp in comparisons:
        if comp not in metrics_by_policy:
            continue
        base = metrics_by_policy[comp]

        # 1. Mean PSNR comparison
        ours_psnr = ours['mean_psnr']
        base_psnr = base['mean_psnr']
        diff_psnr = ours_psnr - base_psnr
        mean_diff_psnr = float(np.mean(diff_psnr))
        dz_psnr = compute_cohens_d(ours_psnr, base_psnr)  # paired Cohen's dz: mean(diff)/std(diff)
        p_psnr_twosided = safe_wilcoxon(ours_psnr, base_psnr, alternative='two-sided')
        p_psnr_greater = safe_wilcoxon(ours_psnr, base_psnr, alternative='greater')
        ci_psnr_low, ci_psnr_high = bootstrap_ci_95(diff_psnr)

        # 2. Mean SSIM comparison
        ours_ssim = ours['mean_ssim']
        base_ssim = base['mean_ssim']
        diff_ssim = ours_ssim - base_ssim
        mean_diff_ssim = float(np.mean(diff_ssim))
        dz_ssim = compute_cohens_d(ours_ssim, base_ssim)
        p_ssim_twosided = safe_wilcoxon(ours_ssim, base_ssim, alternative='two-sided')
        p_ssim_greater = safe_wilcoxon(ours_ssim, base_ssim, alternative='greater')
        ci_ssim_low, ci_ssim_high = bootstrap_ci_95(diff_ssim)

        row = {
            'comparison': f"OURS vs {comp.upper()}",
            'delta_psnr_mean': mean_diff_psnr,
            'ci95_delta_psnr': f"[{ci_psnr_low:+.2f}, {ci_psnr_high:+.2f}]",
            'cohens_dz_psnr': dz_psnr,
            'p_val_psnr_twosided': p_psnr_twosided,
            'p_val_psnr_superiority': p_psnr_greater,
            'delta_ssim_mean': mean_diff_ssim,
            'ci95_delta_ssim': f"[{ci_ssim_low:+.4f}, {ci_ssim_high:+.4f}]",
            'cohens_dz_ssim': dz_ssim,
            'p_val_ssim_twosided': p_ssim_twosided,
            'p_val_ssim_superiority': p_ssim_greater,
        }
        raw_rows.append(row)

    # Apply Holm-Bonferroni correction across the family of baseline comparisons
    p_psnr_two = [r['p_val_psnr_twosided'] for r in raw_rows]
    p_psnr_sup = [r['p_val_psnr_superiority'] for r in raw_rows]
    p_ssim_two = [r['p_val_ssim_twosided'] for r in raw_rows]
    p_ssim_sup = [r['p_val_ssim_superiority'] for r in raw_rows]

    p_psnr_two_holm = holm_bonferroni(p_psnr_two)
    p_psnr_sup_holm = holm_bonferroni(p_psnr_sup)
    p_ssim_two_holm = holm_bonferroni(p_ssim_two)
    p_ssim_sup_holm = holm_bonferroni(p_ssim_sup)

    for i, r in enumerate(raw_rows):
        r['p_psnr_twosided_holm'] = p_psnr_two_holm[i]
        r['p_psnr_sup_holm'] = p_psnr_sup_holm[i]
        r['p_ssim_twosided_holm'] = p_ssim_two_holm[i]
        r['p_ssim_sup_holm'] = p_ssim_sup_holm[i]
        r['significant_raw_p05'] = (r['p_val_psnr_superiority'] < 0.05 or r['p_val_ssim_superiority'] < 0.05)
        r['significant_holm_p05'] = (r['p_psnr_sup_holm'] < 0.05 or r['p_ssim_sup_holm'] < 0.05)

        comp = r['comparison']
        print(f"[{comp}]:")
        print(f"  Δ Mean PSNR: {r['delta_psnr_mean']:+.2f} dB (95% CI: {r['ci95_delta_psnr']}, paired dz: {r['cohens_dz_psnr']:.2f})")
        print(f"    p (two-sided): {r['p_val_psnr_twosided']:.4f} (Holm: {r['p_psnr_twosided_holm']:.4f}) | p (superiority): {r['p_val_psnr_superiority']:.4f} (Holm: {r['p_psnr_sup_holm']:.4f})")
        print(f"  Δ Mean SSIM: {r['delta_ssim_mean']:+.4f} (95% CI: {r['ci95_delta_ssim']}, paired dz: {r['cohens_dz_ssim']:.2f})")
        print(f"    p (two-sided): {r['p_val_ssim_twosided']:.4f} (Holm: {r['p_ssim_twosided_holm']:.4f}) | p (superiority): {r['p_val_ssim_superiority']:.4f} (Holm: {r['p_ssim_sup_holm']:.4f})")
        print()

    out_df = pd.DataFrame(raw_rows)
    out_dir = REPO_ROOT / "results/phase13_frozen_benchmark"
    out_csv = out_dir / "statistical_significance_validation.csv"
    out_df.to_csv(out_csv, index=False)
    print(f">> Saved statistical validation table to {out_csv}")

    # Print Summary Markdown Table
    print("=" * 135)
    print(f"{'Comparison':<28} | {'Δ PSNR':<10} | {'95% CI (PSNR)':<18} | {'dz':<6} | {'p (two)':<8} | {'p (sup)':<8} | {'p (Holm)':<8} | {'Sig (Holm)':<10}")
    print("-" * 135)
    for _, r in out_df.iterrows():
        comp = r['comparison']
        d_p = f"{r['delta_psnr_mean']:+.2f} dB"
        ci_p = r['ci95_delta_psnr']
        dz = f"{r['cohens_dz_psnr']:.2f}"
        p2 = f"{r['p_val_psnr_twosided']:.4f}"
        ps = f"{r['p_val_psnr_superiority']:.4f}"
        ph = f"{r['p_psnr_sup_holm']:.4f}"
        sig = "Yes" if r['significant_holm_p05'] else "No"
        print(f"{comp:<28} | {d_p:<10} | {ci_p:<18} | {dz:<6} | {p2:<8} | {ps:<8} | {ph:<8} | {sig:<10}")
    print("=" * 135)

if __name__ == "__main__":
    analyze_frozen_benchmark()
