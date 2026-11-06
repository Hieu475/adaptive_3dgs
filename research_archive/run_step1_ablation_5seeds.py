#!/usr/bin/env python3
"""Clean Multi-Seed Ablation Study for Step 1 Innovations.

Conditions Evaluated Across 5 Seeds [42, 43, 44, 45, 46] on TUM fr2_xyz (150 frames):
  - A0: Old Substrate (No Warm-up, No Coverage Throttling, No Backlog Throttling, Fixed K=5)
  - A1: + Age-Aware Warm-up (warmup_steps=3, warmup_ratio=0.20, warmup_k=2)
  - A2: + Coverage Throttling (throttle_threshold=0.90, factor=0.20)
  - A3: + Backlog Throttling (max_warmup_queue=500)
  - A4: All (Stage 1 Full Substrate + Learned Utility + Adaptive-K)

Statistical Rigor:
  - Paired two-sided Wilcoxon signed-rank test vs A0
  - Cohen's d effect sizes
  - 95% Bootstrap confidence intervals
  - Holm-Bonferroni family-wise error rate multiplicity correction
  - Dedicated output folder: results/step1_ablation_5seeds/ (preserves phase13_frozen_benchmark)
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pipeline import OnlineReconstructionPipeline
from experiments.run_phase13_frozen_benchmark import (
    load_phase10_sequence,
    build_pipeline_config,
    run_policy_trajectory,
)


def bootstrap_ci_95(data: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval on 1D numpy array."""
    arr = np.asarray(data)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    boot_means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired or independent samples."""
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-12:
        return 0.0
    return float(np.mean(diff) / std_diff)


def safe_wilcoxon_two_sided(x: np.ndarray, y: np.ndarray) -> float:
    """Compute exact two-sided Wilcoxon signed-rank test p-value."""
    diff = x - y
    if np.all(diff == 0):
        return 1.0
    try:
        res = wilcoxon(diff, alternative='two-sided')
        return float(res.pvalue)
    except Exception:
        return 1.0


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down correction for family-wise error rate control."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = (m - rank) * p
        running_max = max(running_max, val)
        adj[orig_idx] = min(1.0, running_max)
    return adj


def get_ablation_configs(seed: int, budget_ms: float, W: int, H: int, device: str) -> Dict[str, Dict[str, Any]]:
    """Build the 5 ablation configurations A0 to A4 for a given seed."""
    base_cfg = build_pipeline_config(policy="ours", seed=seed, budget_ms=budget_ms, W=W, H=H, device=device)

    # A0: Old substrate (No warmup, No coverage throttling, No backlog throttling, Fixed K=5)
    cfg_a0 = json.loads(json.dumps(base_cfg))
    cfg_a0["scheduler"]["enable_warmup"] = False
    cfg_a0["scheduler"]["warmup_budget_ratio"] = 0.0
    cfg_a0["scheduler"]["max_warmup_queue"] = 999999
    cfg_a0["densification"]["enable_coverage_throttling"] = False
    cfg_a0["training"]["use_adaptive_k"] = False

    # A1: + Age-Aware Warmup only
    cfg_a1 = json.loads(json.dumps(base_cfg))
    cfg_a1["scheduler"]["enable_warmup"] = True
    cfg_a1["scheduler"]["warmup_steps"] = 3
    cfg_a1["scheduler"]["warmup_budget_ratio"] = 0.20
    cfg_a1["scheduler"]["warmup_k"] = 2
    cfg_a1["scheduler"]["max_warmup_queue"] = 999999
    cfg_a1["densification"]["enable_coverage_throttling"] = False
    cfg_a1["training"]["use_adaptive_k"] = False

    # A2: + Coverage Throttling
    cfg_a2 = json.loads(json.dumps(base_cfg))
    cfg_a2["scheduler"]["enable_warmup"] = True
    cfg_a2["scheduler"]["warmup_steps"] = 3
    cfg_a2["scheduler"]["warmup_budget_ratio"] = 0.20
    cfg_a2["scheduler"]["warmup_k"] = 2
    cfg_a2["scheduler"]["max_warmup_queue"] = 999999
    cfg_a2["densification"]["enable_coverage_throttling"] = True
    cfg_a2["densification"]["throttle_coverage_threshold"] = 0.90
    cfg_a2["densification"]["throttle_factor"] = 0.20
    cfg_a2["training"]["use_adaptive_k"] = False

    # A3: + Backlog Throttling
    cfg_a3 = json.loads(json.dumps(base_cfg))
    cfg_a3["scheduler"]["enable_warmup"] = True
    cfg_a3["scheduler"]["warmup_steps"] = 3
    cfg_a3["scheduler"]["warmup_budget_ratio"] = 0.20
    cfg_a3["scheduler"]["warmup_k"] = 2
    cfg_a3["scheduler"]["max_warmup_queue"] = 500
    cfg_a3["densification"]["enable_coverage_throttling"] = True
    cfg_a3["densification"]["throttle_coverage_threshold"] = 0.90
    cfg_a3["densification"]["throttle_factor"] = 0.20
    cfg_a3["training"]["use_adaptive_k"] = False

    # A4: All (Warmup + Coverage Throttling + Backlog Throttling + Learned Utility + Adaptive-K)
    cfg_a4 = json.loads(json.dumps(base_cfg))
    cfg_a4["scheduler"]["enable_warmup"] = True
    cfg_a4["scheduler"]["warmup_steps"] = 3
    cfg_a4["scheduler"]["warmup_budget_ratio"] = 0.20
    cfg_a4["scheduler"]["warmup_k"] = 2
    cfg_a4["scheduler"]["max_warmup_queue"] = 500
    cfg_a4["densification"]["enable_coverage_throttling"] = True
    cfg_a4["densification"]["throttle_coverage_threshold"] = 0.90
    cfg_a4["densification"]["throttle_factor"] = 0.20
    cfg_a4["training"]["use_adaptive_k"] = True

    return {
        "A0_old_substrate": cfg_a0,
        "A1_plus_warmup": cfg_a1,
        "A2_plus_coverage_throttling": cfg_a2,
        "A3_plus_backlog_throttling": cfg_a3,
        "A4_all": cfg_a4,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean Step 1 Ablation Study Across 5 Seeds")
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz")
    parser.add_argument("--n_frames", type=int, default=150)
    parser.add_argument("--budget_ms", type=float, default=15.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="results/step1_ablation_5seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    # Memory Guard: cap CUDA allocation
    if args.device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.70, 0)
        except (RuntimeError, ValueError):
            pass

    out_path = REPO_ROOT / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240

    print("=" * 90)
    print("  STAGE 1 COMPONENT ABLATION STUDY: 5 SEEDS x 150 FRAMES")
    print("=" * 90)
    print(f"  Scene:      {args.scene}")
    print(f"  Frames:     {args.n_frames}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Device:     {args.device}")
    print(f"  Output Dir: {out_path}")
    print("=" * 90)

    # 1. Preload sequence
    print(f"\n>> Preloading sequence {args.scene} ({args.n_frames + 1} frames)...")
    frames, intrinsics = load_phase10_sequence(
        scene_name=args.scene,
        n_frames=args.n_frames + 1,
        H=H,
        W=W,
        device=args.device,
    )
    print(f">> Sequence preloaded successfully ({len(frames)} frames).")

    condition_names = [
        "A0_old_substrate",
        "A1_plus_warmup",
        "A2_plus_coverage_throttling",
        "A3_plus_backlog_throttling",
        "A4_all",
    ]

    raw_results = []

    total_runs = len(condition_names) * len(args.seeds)
    run_idx = 0

    t_all_start = time.perf_counter()

    for cond in condition_names:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] Condition: {cond} | Seed: {seed} ...")
            cfgs = get_ablation_configs(seed, args.budget_ms, W, H, args.device)
            target_cfg = cfgs[cond]

            summary, frame_logs = run_policy_trajectory(
                policy="ours",
                seed=seed,
                frames=frames,
                intrinsics=intrinsics,
                budget_ms=args.budget_ms,
                device=args.device,
                W=W,
                H=H,
                custom_config=target_cfg,
            )

            rec = {
                "condition": cond,
                "seed": seed,
                "mean_psnr": summary["mean_psnr"],
                "final_psnr": summary["final_psnr"],
                "mean_ssim": summary["mean_ssim"],
                "final_ssim": summary["final_ssim"],
                "mean_depth_l1": summary["mean_depth_l1"],
                "final_depth_l1": summary["final_depth_l1"],
                "mean_n_optimized": summary["mean_n_optimized"],
                "mean_k": summary["mean_k"],
                "N_final": summary["N_final"],
                "fps": summary["fps"],
                "mean_opt_time_ms": summary["mean_opt_time_ms"],
                "mean_frame_time_ms": summary["mean_frame_time_ms"],
            }
            raw_results.append(rec)
            print(f"   -> Mean PSNR: {rec['mean_psnr']:.2f} dB | Final PSNR: {rec['final_psnr']:.2f} dB | SSIM: {rec['mean_ssim']:.4f} | FPS: {rec['fps']:.1f} | N_final: {rec['N_final']}")

    t_all_dur = time.perf_counter() - t_all_start
    print(f"\n>> All {total_runs} trajectories completed in {t_all_dur:.1f}s ({t_all_dur / 60.0:.2f} min).")

    # 2. Convert to DataFrame
    df = pd.DataFrame(raw_results)
    df.to_csv(out_path / "step1_ablation_raw.csv", index=False)

    # 3. Statistical Analysis: Paired Wilcoxon Tests Against A0 + Holm Multiplicity Correction
    a0_df = df[df["condition"] == "A0_old_substrate"].sort_values("seed")
    a0_mean_psnr = a0_df["mean_psnr"].to_numpy()
    a0_final_psnr = a0_df["final_psnr"].to_numpy()
    a0_mean_ssim = a0_df["mean_ssim"].to_numpy()

    stats_records = []
    comparisons = [c for c in condition_names if c != "A0_old_substrate"]

    p_vals_mean_psnr = []
    p_vals_final_psnr = []
    p_vals_mean_ssim = []

    for comp in comparisons:
        comp_df = df[df["condition"] == comp].sort_values("seed")
        comp_mean_psnr = comp_df["mean_psnr"].to_numpy()
        comp_final_psnr = comp_df["final_psnr"].to_numpy()
        comp_mean_ssim = comp_df["mean_ssim"].to_numpy()

        p_m_psnr = safe_wilcoxon_two_sided(comp_mean_psnr, a0_mean_psnr)
        p_f_psnr = safe_wilcoxon_two_sided(comp_final_psnr, a0_final_psnr)
        p_m_ssim = safe_wilcoxon_two_sided(comp_mean_ssim, a0_mean_ssim)

        p_vals_mean_psnr.append(p_m_psnr)
        p_vals_final_psnr.append(p_f_psnr)
        p_vals_mean_ssim.append(p_m_ssim)

    # Apply Holm-Bonferroni correction
    p_adj_mean_psnr = holm_bonferroni_correction(p_vals_mean_psnr)
    p_adj_final_psnr = holm_bonferroni_correction(p_vals_final_psnr)
    p_adj_mean_ssim = holm_bonferroni_correction(p_vals_mean_ssim)

    for i, comp in enumerate(comparisons):
        comp_df = df[df["condition"] == comp].sort_values("seed")
        c_mean_psnr = comp_df["mean_psnr"].to_numpy()
        c_final_psnr = comp_df["final_psnr"].to_numpy()
        c_mean_ssim = comp_df["mean_ssim"].to_numpy()

        diff_m_psnr = c_mean_psnr - a0_mean_psnr
        diff_f_psnr = c_final_psnr - a0_final_psnr
        diff_m_ssim = c_mean_ssim - a0_mean_ssim

        ci_m_psnr = bootstrap_ci_95(diff_m_psnr)
        ci_f_psnr = bootstrap_ci_95(diff_f_psnr)
        ci_m_ssim = bootstrap_ci_95(diff_m_ssim)

        d_m_psnr = compute_cohens_d(c_mean_psnr, a0_mean_psnr)
        d_f_psnr = compute_cohens_d(c_final_psnr, a0_final_psnr)

        stats_records.append({
            "comparison": f"{comp} vs A0",
            "delta_mean_psnr": float(np.mean(diff_m_psnr)),
            "ci_mean_psnr": ci_m_psnr,
            "p_mean_psnr_raw": p_vals_mean_psnr[i],
            "p_mean_psnr_holm": p_adj_mean_psnr[i],
            "cohens_d_mean_psnr": d_m_psnr,
            "delta_final_psnr": float(np.mean(diff_f_psnr)),
            "ci_final_psnr": ci_f_psnr,
            "p_final_psnr_raw": p_vals_final_psnr[i],
            "p_final_psnr_holm": p_adj_final_psnr[i],
            "cohens_d_final_psnr": d_f_psnr,
            "delta_mean_ssim": float(np.mean(diff_m_ssim)),
            "ci_mean_ssim": ci_m_ssim,
            "p_mean_ssim_holm": p_adj_mean_ssim[i],
        })

    # 4. Aggregated Condition Summary
    cond_summary = []
    for cond in condition_names:
        sub = df[df["condition"] == cond]
        cond_summary.append({
            "condition": cond,
            "mean_psnr_mean": float(sub["mean_psnr"].mean()),
            "mean_psnr_std": float(sub["mean_psnr"].std()),
            "final_psnr_mean": float(sub["final_psnr"].mean()),
            "final_psnr_std": float(sub["final_psnr"].std()),
            "mean_ssim_mean": float(sub["mean_ssim"].mean()),
            "mean_ssim_std": float(sub["mean_ssim"].std()),
            "fps_mean": float(sub["fps"].mean()),
            "mean_k_mean": float(sub["mean_k"].mean()),
            "N_final_mean": float(sub["N_final"].mean()),
        })

    summary_df = pd.DataFrame(cond_summary)
    summary_df.to_csv(out_path / "step1_ablation_summary.csv", index=False)

    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv(out_path / "step1_ablation_statistics.csv", index=False)

    full_output = {
        "metadata": {
            "scene": args.scene,
            "n_frames": args.n_frames,
            "seeds": args.seeds,
            "device": args.device,
            "budget_ms": args.budget_ms,
            "total_wall_sec": t_all_dur,
        },
        "condition_summary": cond_summary,
        "statistical_tests_vs_a0": stats_records,
        "raw_runs": raw_results,
    }

    with open(out_path / "step1_ablation_results.json", "w") as f:
        json.dump(full_output, f, indent=2)

    # 5. Generate Markdown Report
    lines = [
        "# Stage 1 Component Ablation Study Report (5 Seeds x 150 Frames)",
        f"\n**Scene:** `{args.scene}` | **Frames:** {args.n_frames} | **Seeds:** {args.seeds} | **Budget:** {args.budget_ms} ms\n",
        "## 1. Condition Aggregate Performance",
        "",
        "| Condition | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | Mean K | Final Map Size | FPS |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for s in cond_summary:
        lines.append(
            f"| `{s['condition']}` | {s['mean_psnr_mean']:.2f} ± {s['mean_psnr_std']:.2f} | "
            f"{s['final_psnr_mean']:.2f} ± {s['final_psnr_std']:.2f} | "
            f"{s['mean_ssim_mean']:.4f} ± {s['mean_ssim_std']:.4f} | "
            f"{s['mean_k_mean']:.2f} | {int(s['N_final_mean']):,} | {s['fps_mean']:.1f} |"
        )

    lines.extend([
        "",
        "## 2. Statistical Significance Tests vs A0 Baseline (Two-Sided Wilcoxon + Holm Multiplicity Correction)",
        "",
        "| Comparison | Δ Mean PSNR [95% CI] | p_Holm | Cohen's d | Δ Final PSNR [95% CI] | p_Holm | Cohen's d | Δ SSIM |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for st in stats_records:
        ci_m = f"[{st['ci_mean_psnr'][0]:+.2f}, {st['ci_mean_psnr'][1]:+.2f}]"
        ci_f = f"[{st['ci_final_psnr'][0]:+.2f}, {st['ci_final_psnr'][1]:+.2f}]"
        lines.append(
            f"| **{st['comparison']}** | {st['delta_mean_psnr']:+.2f} dB {ci_m} | {st['p_mean_psnr_holm']:.4f} | {st['cohens_d_mean_psnr']:.2f} | "
            f"{st['delta_final_psnr']:+.2f} dB {ci_f} | {st['p_final_psnr_holm']:.4f} | {st['cohens_d_final_psnr']:.2f} | {st['delta_mean_ssim']:+.4f} |"
        )

    report_content = "\n".join(lines)
    with open(out_path / "step1_ablation_report.md", "w") as f:
        f.write(report_content)

    print("\n" + "=" * 90)
    print("  STAGE 1 ABLATION SUMMARY")
    print("=" * 90)
    print(report_content)
    print("=" * 90)


if __name__ == "__main__":
    main()
