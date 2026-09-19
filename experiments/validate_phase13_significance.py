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

def analyze_frozen_benchmark():
    json_path = REPO_ROOT / "results/phase13_frozen_benchmark/phase13_frozen_results.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    raw_runs = data["raw_runs"]
    df = pd.DataFrame(raw_runs)

    print("=" * 95)
    print("  ITEM 3: STATISTICAL SIGNIFICANCE & EFFECT SIZE (PHASE 13 FROZEN BENCHMARK)")
    print("=" * 95)
    print(f"  Source: {json_path}")
    print(f"  Policies: {df['policy'].unique().tolist()}")
    print(f"  Seeds: {sorted(df['seed'].unique().tolist())}")
    print("=" * 95)

    seeds = sorted(df['seed'].unique().tolist())

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

    results_table = []
    
    print("\n--- PAIRED STATISTICAL TESTS: OURS vs BASELINES (Across 5 Independent Seeds) ---\n")

    for comp in comparisons:
        if comp not in metrics_by_policy:
            continue
        base = metrics_by_policy[comp]

        # 1. Mean PSNR comparison
        ours_psnr = ours['mean_psnr']
        base_psnr = base['mean_psnr']
        diff_psnr = ours_psnr - base_psnr
        mean_diff_psnr = float(np.mean(diff_psnr))
        d_psnr = compute_cohens_d(ours_psnr, base_psnr)
        p_psnr = safe_wilcoxon(ours_psnr, base_psnr, alternative='greater' if mean_diff_psnr >= 0 else 'less')
        ci_psnr_low, ci_psnr_high = bootstrap_ci_95(diff_psnr)

        # 2. Mean SSIM comparison
        ours_ssim = ours['mean_ssim']
        base_ssim = base['mean_ssim']
        diff_ssim = ours_ssim - base_ssim
        mean_diff_ssim = float(np.mean(diff_ssim))
        d_ssim = compute_cohens_d(ours_ssim, base_ssim)
        p_ssim = safe_wilcoxon(ours_ssim, base_ssim, alternative='greater' if mean_diff_ssim >= 0 else 'less')
        ci_ssim_low, ci_ssim_high = bootstrap_ci_95(diff_ssim)

        row = {
            'comparison': f"OURS vs {comp.upper()}",
            'delta_psnr_mean': mean_diff_psnr,
            'ci95_delta_psnr': f"[{ci_psnr_low:+.2f}, {ci_psnr_high:+.2f}]",
            'cohens_d_psnr': d_psnr,
            'p_val_psnr': p_psnr,
            'delta_ssim_mean': mean_diff_ssim,
            'ci95_delta_ssim': f"[{ci_ssim_low:+.4f}, {ci_ssim_high:+.4f}]",
            'cohens_d_ssim': d_ssim,
            'p_val_ssim': p_ssim,
            'significant_p05': (p_psnr < 0.05 or p_ssim < 0.05),
        }
        results_table.append(row)

        print(f"[{comp.upper()}]:")
        print(f"  Δ Mean PSNR: {mean_diff_psnr:+.2f} dB (95% CI: [{ci_psnr_low:+.2f}, {ci_psnr_high:+.2f}], Cohen's d: {d_psnr:.2f}, p-value: {p_psnr:.4f})")
        print(f"  Δ Mean SSIM: {mean_diff_ssim:+.4f} (95% CI: [{ci_ssim_low:+.4f}, {ci_ssim_high:+.4f}], Cohen's d: {d_ssim:.2f}, p-value: {p_ssim:.4f})")
        print()

    out_df = pd.DataFrame(results_table)
    out_dir = REPO_ROOT / "results/phase13_frozen_benchmark"
    out_csv = out_dir / "statistical_significance_validation.csv"
    out_df.to_csv(out_csv, index=False)
    print(f">> Saved statistical validation table to {out_csv}")

    # Print Summary Markdown Table
    print("=" * 110)
    print(f"{'Comparison':<28} | {'Δ PSNR':<10} | {'95% CI (PSNR)':<18} | {'Cohen d':<8} | {'p-val':<8} | {'Δ SSIM':<8} | {'Cohen d':<8} | {'p-val':<8}")
    print("-" * 110)
    for _, r in out_df.iterrows():
        comp = r['comparison']
        d_p = f"{r['delta_psnr_mean']:+.2f} dB"
        ci_p = r['ci95_delta_psnr']
        cd_p = f"{r['cohens_d_psnr']:.2f}"
        pv_p = f"{r['p_val_psnr']:.4f}"
        d_s = f"{r['delta_ssim_mean']:+.4f}"
        cd_s = f"{r['cohens_d_ssim']:.2f}"
        pv_s = f"{r['p_val_ssim']:.4f}"
        print(f"{comp:<28} | {d_p:<10} | {ci_p:<18} | {cd_p:<8} | {pv_p:<8} | {d_s:<8} | {cd_s:<8} | {pv_s:<8}")
    print("=" * 110)

if __name__ == "__main__":
    analyze_frozen_benchmark()
