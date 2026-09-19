#!/usr/bin/env python3
"""Quick Experiment: Throttling WITHOUT Warmup.

Motivation:
  A2/A3 in the cumulative ablation always *included* A1's warmup (which was shown harmful).
  This experiment tests the PURE throttling config:
    - enable_warmup = False
    - enable_coverage_throttling = True
    - enable_backlog_throttling = True
    - use_adaptive_k = False (fixed K=5)

  We also re-run A0 (baseline) and A3 (warmup + throttling) for in-session comparison.

Conditions:
  - A0: No warmup, No throttling, Fixed K=5  (baseline)
  - A3_with_warmup: Warmup + Coverage + Backlog Throttling, Fixed K=5  (old A3)
  - A_THROTTLE_ONLY: No warmup + Coverage + Backlog Throttling, Fixed K=5  (NEW)

Usage:
  python experiments/run_throttle_no_warmup.py --seeds 42        # 1 seed quick check
  python experiments/run_throttle_no_warmup.py --seeds 42 43 44 45 46  # full
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_phase13_frozen_benchmark import (
    load_phase10_sequence,
    build_pipeline_config,
    run_policy_trajectory,
)


def make_a0_config(seed: int, budget_ms: float, W: int, H: int, device: str) -> Dict[str, Any]:
    """A0: Old substrate — no warmup, no throttling, fixed K=5."""
    cfg = build_pipeline_config(policy="ours", seed=seed, budget_ms=budget_ms, W=W, H=H, device=device)
    cfg["scheduler"]["enable_warmup"] = False
    cfg["scheduler"]["warmup_budget_ratio"] = 0.0
    cfg["scheduler"]["max_warmup_queue"] = 999999
    cfg["densification"]["enable_coverage_throttling"] = False
    cfg["training"]["use_adaptive_k"] = False
    return cfg


def make_a3_warmup_config(seed: int, budget_ms: float, W: int, H: int, device: str) -> Dict[str, Any]:
    """A3 (old): warmup + coverage throttling + backlog throttling, fixed K=5."""
    cfg = build_pipeline_config(policy="ours", seed=seed, budget_ms=budget_ms, W=W, H=H, device=device)
    cfg["scheduler"]["enable_warmup"] = True
    cfg["scheduler"]["warmup_steps"] = 3
    cfg["scheduler"]["warmup_budget_ratio"] = 0.20
    cfg["scheduler"]["warmup_k"] = 2
    cfg["scheduler"]["max_warmup_queue"] = 500
    cfg["densification"]["enable_coverage_throttling"] = True
    cfg["densification"]["throttle_coverage_threshold"] = 0.90
    cfg["densification"]["throttle_factor"] = 0.20
    cfg["training"]["use_adaptive_k"] = False
    return cfg


def make_throttle_only_config(seed: int, budget_ms: float, W: int, H: int, device: str) -> Dict[str, Any]:
    """NEW: Coverage + Backlog Throttling WITHOUT warmup, fixed K=5."""
    cfg = build_pipeline_config(policy="ours", seed=seed, budget_ms=budget_ms, W=W, H=H, device=device)
    # NO warmup
    cfg["scheduler"]["enable_warmup"] = False
    cfg["scheduler"]["warmup_budget_ratio"] = 0.0
    cfg["scheduler"]["max_warmup_queue"] = 999999
    # YES throttling
    cfg["densification"]["enable_coverage_throttling"] = True
    cfg["densification"]["throttle_coverage_threshold"] = 0.90
    cfg["densification"]["throttle_factor"] = 0.20
    cfg["scheduler"]["max_warmup_queue"] = 500  # backlog throttling
    # NO adaptive-K
    cfg["training"]["use_adaptive_k"] = False
    return cfg


CONDITION_BUILDERS = {
    "A0_baseline": make_a0_config,
    "A3_warmup_throttle": make_a3_warmup_config,
    "A_THROTTLE_ONLY": make_throttle_only_config,
}


def main():
    parser = argparse.ArgumentParser(description="Throttling-Without-Warmup Quick Experiment")
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz")
    parser.add_argument("--n_frames", type=int, default=150)
    parser.add_argument("--budget_ms", type=float, default=15.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="results/throttle_no_warmup")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.70, 0)
        except (RuntimeError, ValueError):
            pass

    out_path = REPO_ROOT / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240

    print("=" * 80)
    print("  THROTTLING WITHOUT WARMUP — Quick Experiment")
    print("=" * 80)
    print(f"  Scene:   {args.scene}")
    print(f"  Frames:  {args.n_frames}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Device:  {args.device}")
    print(f"  Output:  {out_path}")
    print("=" * 80)

    # 1. Preload sequence
    print(f"\n>> Preloading sequence {args.scene} ({args.n_frames + 1} frames)...")
    frames, intrinsics = load_phase10_sequence(
        scene_name=args.scene,
        n_frames=args.n_frames + 1,
        H=H,
        W=W,
        device=args.device,
    )
    print(f">> Loaded {len(frames)} frames.\n")

    conditions = list(CONDITION_BUILDERS.keys())
    raw_results = []
    total_runs = len(conditions) * len(args.seeds)
    run_idx = 0

    t_start = time.perf_counter()

    for cond_name in conditions:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] {cond_name} | seed={seed}")
            cfg = CONDITION_BUILDERS[cond_name](seed, args.budget_ms, W, H, args.device)

            summary, frame_logs = run_policy_trajectory(
                policy="ours",
                seed=seed,
                frames=frames,
                intrinsics=intrinsics,
                budget_ms=args.budget_ms,
                device=args.device,
                W=W,
                H=H,
                custom_config=cfg,
            )

            rec = {
                "condition": cond_name,
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
            print(f"   -> Mean PSNR: {rec['mean_psnr']:.2f} dB | Final PSNR: {rec['final_psnr']:.2f} dB | "
                  f"SSIM: {rec['mean_ssim']:.4f} | FPS: {rec['fps']:.1f} | N_final: {rec['N_final']}")

    t_dur = time.perf_counter() - t_start
    print(f"\n>> All {total_runs} runs completed in {t_dur:.1f}s ({t_dur / 60:.1f} min)")

    # 2. Summary table
    print("\n" + "=" * 80)
    print("  RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Condition':<25} {'Mean PSNR':>10} {'Final PSNR':>11} {'SSIM':>8} {'FPS':>6} {'N_final':>8}")
    print("-" * 80)
    for rec in raw_results:
        print(f"{rec['condition']:<25} {rec['mean_psnr']:>10.2f} {rec['final_psnr']:>11.2f} "
              f"{rec['mean_ssim']:>8.4f} {rec['fps']:>6.1f} {rec['N_final']:>8,}")

    # Reference from previous ablation
    print("\n>> Reference from previous 5-seed ablation (means across seeds 42-46):")
    print("   A0: Mean PSNR=15.53, Final PSNR=12.93, SSIM=0.6973")
    print("   A3: Mean PSNR=16.61, Final PSNR=14.68, SSIM=0.7713")
    print("   (A3 had warmup=ON — 'cõng' the harmful warmup component)")
    print()

    # 3. Key comparison: throttle-only vs A0 and A3
    a0_runs = [r for r in raw_results if r["condition"] == "A0_baseline"]
    to_runs = [r for r in raw_results if r["condition"] == "A_THROTTLE_ONLY"]
    a3_runs = [r for r in raw_results if r["condition"] == "A3_warmup_throttle"]

    if a0_runs and to_runs:
        for seed in args.seeds:
            a0 = next((r for r in a0_runs if r["seed"] == seed), None)
            to = next((r for r in to_runs if r["seed"] == seed), None)
            a3 = next((r for r in a3_runs if r["seed"] == seed), None)
            if a0 and to:
                d_mean = to["mean_psnr"] - a0["mean_psnr"]
                d_final = to["final_psnr"] - a0["final_psnr"]
                d_ssim = to["mean_ssim"] - a0["mean_ssim"]
                print(f"  [seed={seed}] THROTTLE_ONLY vs A0:  Δ Mean PSNR = {d_mean:+.2f} dB | "
                      f"Δ Final PSNR = {d_final:+.2f} dB | Δ SSIM = {d_ssim:+.4f}")
            if a3 and to:
                d_mean = to["mean_psnr"] - a3["mean_psnr"]
                d_final = to["final_psnr"] - a3["final_psnr"]
                d_ssim = to["mean_ssim"] - a3["mean_ssim"]
                print(f"  [seed={seed}] THROTTLE_ONLY vs A3:  Δ Mean PSNR = {d_mean:+.2f} dB | "
                      f"Δ Final PSNR = {d_final:+.2f} dB | Δ SSIM = {d_ssim:+.4f}")

    # 4. Save results
    output = {
        "metadata": {
            "experiment": "throttle_no_warmup",
            "scene": args.scene,
            "n_frames": args.n_frames,
            "seeds": args.seeds,
            "budget_ms": args.budget_ms,
            "total_wall_sec": t_dur,
        },
        "raw_runs": raw_results,
    }
    with open(out_path / "throttle_no_warmup_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n>> Results saved to {out_path / 'throttle_no_warmup_results.json'}")

    # 5. Verdict
    if to_runs and a0_runs:
        to_mean = np.mean([r["mean_psnr"] for r in to_runs])
        a0_mean = np.mean([r["mean_psnr"] for r in a0_runs])
        delta = to_mean - a0_mean
        print(f"\n>> VERDICT: THROTTLE_ONLY mean PSNR = {to_mean:.2f} dB (Δ vs A0 = {delta:+.2f} dB)")
        if delta > 0.5:
            print(">> ✅ THROTTLE_ONLY is BETTER than A0 — proceed to lock as OURS definition")
        elif delta > 0:
            print(">> ⚠️  THROTTLE_ONLY is slightly better — consider more seeds for confirmation")
        else:
            print(">> ❌ THROTTLE_ONLY is NOT better — keep A3 as OURS")
    if to_runs and a3_runs:
        to_mean = np.mean([r["mean_psnr"] for r in to_runs])
        a3_mean = np.mean([r["mean_psnr"] for r in a3_runs])
        delta = to_mean - a3_mean
        print(f">> THROTTLE_ONLY vs A3 (with warmup): Δ Mean PSNR = {delta:+.2f} dB")
        if delta > 0:
            print(">> ✅ Removing warmup HELPS — warmup was indeed dragging A3 down")
        else:
            print(">> ⚠️  Removing warmup doesn't help — warmup is neutral/helpful within throttling context")

    print("\nDone.")


if __name__ == "__main__":
    main()
