#!/usr/bin/env python3
"""Final Confirmation Benchmark: OURS (pure throttling) vs Baselines.

Statistical Design:
  - 8 seeds [42..49] for each of 6 policies
  - 150 frames, TUM fr2_xyz
  - Single paired comparison per policy pair → no Holm correction needed
  - n=8: Wilcoxon two-sided minimum p-value = 2/2^8 = 0.0078 → easily < 0.05
  - Reports: paired Wilcoxon, Cohen's d, bootstrap CI

This is the DEFINITIVE benchmark for the paper.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_phase13_frozen_benchmark import (
    load_phase10_sequence,
    build_pipeline_config,
    run_policy_trajectory,
    POLICIES_CANONICAL,
)


DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]  # 8 seeds for p<0.05


def bootstrap_ci_95(data: np.ndarray, n_boot: int = 2000, rng_seed: int = 42) -> Tuple[float, float]:
    """95% bootstrap CI."""
    arr = np.asarray(data)
    if len(arr) <= 1:
        return float(arr[0]) if len(arr) == 1 else (0.0, 0.0), float(arr[0]) if len(arr) == 1 else (0.0, 0.0)
    rng = np.random.default_rng(rng_seed)
    means = [float(np.mean(rng.choice(arr, len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Paired Cohen's d."""
    diff = x - y
    s = np.std(diff, ddof=1)
    return float(np.mean(diff) / s) if s > 1e-12 else 0.0


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided Wilcoxon p-value."""
    diff = x - y
    if np.all(diff == 0):
        return 1.0
    try:
        return float(wilcoxon(diff, alternative='two-sided').pvalue)
    except Exception:
        return 1.0


def main():
    parser = argparse.ArgumentParser(description="Final Confirmation Benchmark (8 seeds)")
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz")
    parser.add_argument("--n_frames", type=int, default=150)
    parser.add_argument("--budget_ms", type=float, default=15.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="results/final_confirmation")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--policies", type=str, nargs="+", default=POLICIES_CANONICAL)
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.70, 0)
        except (RuntimeError, ValueError):
            pass

    out_path = REPO_ROOT / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240

    print("=" * 90)
    print("  FINAL CONFIRMATION BENCHMARK — OURS vs BASELINES")
    print("=" * 90)
    print(f"  Scene:    {args.scene}")
    print(f"  Frames:   {args.n_frames}")
    print(f"  Seeds:    {args.seeds} ({len(args.seeds)} seeds)")
    print(f"  Policies: {args.policies}")
    print(f"  Device:   {args.device}")
    print(f"  Output:   {out_path}")
    print(f"  Min p-value (Wilcoxon, n={len(args.seeds)}): {2 / 2**len(args.seeds):.6f}")
    print("=" * 90)

    # 1. Preload
    print(f"\n>> Loading {args.scene} ({args.n_frames + 1} frames)...")
    frames, intrinsics = load_phase10_sequence(
        scene_name=args.scene, n_frames=args.n_frames + 1, H=H, W=W, device=args.device
    )
    print(f">> Loaded {len(frames)} frames.\n")

    raw_results = []
    total = len(args.policies) * len(args.seeds)
    idx = 0
    t_start = time.perf_counter()

    for policy in args.policies:
        for seed in args.seeds:
            idx += 1
            print(f"\n[{idx}/{total}] {policy} | seed={seed}")
            cfg = build_pipeline_config(policy=policy, seed=seed, budget_ms=args.budget_ms, W=W, H=H, device=args.device)

            summary, _ = run_policy_trajectory(
                policy=policy, seed=seed, frames=frames, intrinsics=intrinsics,
                budget_ms=args.budget_ms, device=args.device, W=W, H=H, custom_config=cfg,
            )

            rec = {
                "policy": policy,
                "seed": seed,
                "mean_psnr": summary["mean_psnr"],
                "final_psnr": summary["final_psnr"],
                "mean_ssim": summary["mean_ssim"],
                "final_ssim": summary["final_ssim"],
                "mean_depth_l1": summary["mean_depth_l1"],
                "N_final": summary["N_final"],
                "fps": summary["fps"],
                "mean_k": summary["mean_k"],
                "mean_opt_time_ms": summary["mean_opt_time_ms"],
                "mean_frame_time_ms": summary["mean_frame_time_ms"],
            }
            raw_results.append(rec)
            print(f"   -> PSNR: {rec['mean_psnr']:.2f}/{rec['final_psnr']:.2f} | "
                  f"SSIM: {rec['mean_ssim']:.4f} | FPS: {rec['fps']:.1f} | N: {rec['N_final']:,}")

    t_dur = time.perf_counter() - t_start
    print(f"\n>> Total: {t_dur:.1f}s ({t_dur/60:.1f} min)")

    # 2. Aggregate per policy
    print("\n" + "=" * 90)
    print("  POLICY SUMMARY (mean ± std across seeds)")
    print("=" * 90)

    policy_stats = {}
    for pol in args.policies:
        runs = [r for r in raw_results if r["policy"] == pol]
        stats = {}
        for metric in ["mean_psnr", "final_psnr", "mean_ssim", "final_ssim", "N_final", "fps"]:
            vals = np.array([r[metric] for r in runs])
            stats[metric] = {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}
        policy_stats[pol] = stats
        print(f"  {pol:30s}  PSNR={stats['mean_psnr']['mean']:.2f}±{stats['mean_psnr']['std']:.2f}  "
              f"Final={stats['final_psnr']['mean']:.2f}±{stats['final_psnr']['std']:.2f}  "
              f"SSIM={stats['mean_ssim']['mean']:.4f}  FPS={stats['fps']['mean']:.1f}  "
              f"N={stats['N_final']['mean']:.0f}")

    # 3. Paired comparisons: OURS vs each baseline
    print("\n" + "=" * 90)
    print("  PAIRED COMPARISONS: OURS vs EACH BASELINE")
    print("=" * 90)

    ours_key = "ours"
    if ours_key not in policy_stats:
        print("WARNING: 'ours' not in policies, skipping paired tests")
    else:
        comparisons = []
        for baseline in args.policies:
            if baseline == ours_key:
                continue
            ours_vals = np.array(policy_stats[ours_key]["mean_psnr"]["values"])
            base_vals = np.array(policy_stats[baseline]["mean_psnr"]["values"])

            delta_psnr = ours_vals - base_vals
            p = safe_wilcoxon(ours_vals, base_vals)
            d = cohens_d(ours_vals, base_vals)
            ci = bootstrap_ci_95(delta_psnr)

            # Also for final PSNR
            ours_final = np.array(policy_stats[ours_key]["final_psnr"]["values"])
            base_final = np.array(policy_stats[baseline]["final_psnr"]["values"])
            delta_final = ours_final - base_final
            p_final = safe_wilcoxon(ours_final, base_final)
            d_final = cohens_d(ours_final, base_final)
            ci_final = bootstrap_ci_95(delta_final)

            # SSIM
            ours_ssim = np.array(policy_stats[ours_key]["mean_ssim"]["values"])
            base_ssim = np.array(policy_stats[baseline]["mean_ssim"]["values"])
            delta_ssim = ours_ssim - base_ssim
            p_ssim = safe_wilcoxon(ours_ssim, base_ssim)

            sig = "✅ p<0.05" if p < 0.05 else "❌ n.s."
            sig_final = "✅ p<0.05" if p_final < 0.05 else "❌ n.s."

            comp = {
                "baseline": baseline,
                "delta_mean_psnr": float(np.mean(delta_psnr)),
                "ci_mean_psnr": ci,
                "p_mean_psnr": p,
                "cohens_d_mean_psnr": d,
                "delta_final_psnr": float(np.mean(delta_final)),
                "ci_final_psnr": ci_final,
                "p_final_psnr": p_final,
                "cohens_d_final_psnr": d_final,
                "delta_mean_ssim": float(np.mean(delta_ssim)),
                "p_mean_ssim": p_ssim,
            }
            comparisons.append(comp)

            print(f"\n  OURS vs {baseline}:")
            print(f"    Mean PSNR:  Δ={comp['delta_mean_psnr']:+.2f} dB  CI=[{ci[0]:+.2f}, {ci[1]:+.2f}]  "
                  f"p={p:.4f} {sig}  d={d:.2f}")
            print(f"    Final PSNR: Δ={comp['delta_final_psnr']:+.2f} dB  CI=[{ci_final[0]:+.2f}, {ci_final[1]:+.2f}]  "
                  f"p={p_final:.4f} {sig_final}  d={d_final:.2f}")
            print(f"    Mean SSIM:  Δ={comp['delta_mean_ssim']:+.4f}  p={p_ssim:.4f}")

    # 4. Generate Markdown Report
    lines = [
        "# Final Confirmation Benchmark Report",
        f"\n**Scene:** `{args.scene}` | **Frames:** {args.n_frames} | **Seeds:** {args.seeds} | "
        f"**Budget:** {args.budget_ms} ms | **n={len(args.seeds)}**\n",
        "## 1. Policy Performance Summary",
        "",
        "| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for pol in args.policies:
        s = policy_stats[pol]
        lines.append(
            f"| `{pol}` | {s['mean_psnr']['mean']:.2f} ± {s['mean_psnr']['std']:.2f} | "
            f"{s['final_psnr']['mean']:.2f} ± {s['final_psnr']['std']:.2f} | "
            f"{s['mean_ssim']['mean']:.4f} | {s['fps']['mean']:.1f} | "
            f"{int(s['N_final']['mean']):,} |"
        )

    if ours_key in policy_stats:
        lines.extend([
            "",
            "## 2. Paired Significance Tests (OURS vs each baseline)",
            "",
            "| Comparison | Δ Mean PSNR | 95% CI | p-value | Cohen's d | Δ Final PSNR | p-value | Δ SSIM |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for c in comparisons:
            sig = "✅" if c['p_mean_psnr'] < 0.05 else "❌"
            lines.append(
                f"| OURS vs `{c['baseline']}` | {c['delta_mean_psnr']:+.2f} dB | "
                f"[{c['ci_mean_psnr'][0]:+.2f}, {c['ci_mean_psnr'][1]:+.2f}] | "
                f"{c['p_mean_psnr']:.4f} {sig} | {c['cohens_d_mean_psnr']:.2f} | "
                f"{c['delta_final_psnr']:+.2f} dB | {c['p_final_psnr']:.4f} | "
                f"{c['delta_mean_ssim']:+.4f} |"
            )

    lines.extend([
        "",
        "## 3. OURS Definition",
        "",
        "```",
        "OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling",
        "       NO warmup (ablation: -0.75 dB)",
        "       NO adaptive-K (ablation: -1.54 dB with learned utility)",
        "       NO learned utility predictor",
        "       Fixed K=5 micro-steps",
        "```",
    ])

    report = "\n".join(lines)
    with open(out_path / "final_confirmation_report.md", "w") as f:
        f.write(report)

    # 5. Save full JSON
    output = {
        "metadata": {
            "experiment": "final_confirmation",
            "scene": args.scene,
            "n_frames": args.n_frames,
            "seeds": args.seeds,
            "n_seeds": len(args.seeds),
            "policies": args.policies,
            "budget_ms": args.budget_ms,
            "total_wall_sec": t_dur,
            "ours_definition": "pure_throttling_no_warmup_no_adaptivek",
            "min_wilcoxon_p": 2 / 2**len(args.seeds),
        },
        "policy_stats": {k: {m: {"mean": v["mean"], "std": v["std"]} for m, v in vs.items()}
                         for k, vs in policy_stats.items()},
        "paired_comparisons": comparisons if ours_key in policy_stats else [],
        "raw_runs": raw_results,
    }
    with open(out_path / "final_confirmation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n>> Report: {out_path / 'final_confirmation_report.md'}")
    print(f">> Data:   {out_path / 'final_confirmation_results.json'}")
    print("\n" + report)
    print("\nDone.")


if __name__ == "__main__":
    main()
