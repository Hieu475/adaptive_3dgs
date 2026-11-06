"""Validation trajectory script for Step 1: Age-Aware Warmup & Coverage-based Densification Throttling.

Executes a 150-frame evaluation on TUM fr2_xyz with OURS policy (seed 42)
and logs per-frame evolution of PSNR, SSIM, N_gaussians, N_optimized, coverage, and warm-up statistics.
"""
import os
import sys
from pathlib import Path

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import time
import numpy as np
import torch
import pandas as pd

from research.pipeline import OnlineReconstructionPipeline
from experiments.run_phase13_frozen_benchmark import (
    load_phase10_sequence,
    build_pipeline_config,
    run_policy_trajectory,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Step 1 Validation")
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz")
    parser.add_argument("--policy", type=str, default="ours")
    parser.add_argument("--n_frames", type=int, default=150)
    parser.add_argument("--budget_ms", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/step1_warmup_throttling_eval")
    args = parser.parse_args()

    device = args.device
    policy = args.policy
    n_frames = args.n_frames
    seed = args.seed
    budget_ms = args.budget_ms
    W, H = 320, 240
    scene_name = args.scene

    output_dir = Path(args.output_dir) / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  STEP 1 VALIDATION: AGE-AWARE WARM-UP & COVERAGE THROTTLING")
    print("=" * 80)
    print(f"  Scene:      {scene_name}")
    print(f"  Frames:     {n_frames}")
    print(f"  Resolution: {W}x{H}")
    print(f"  Budget:     {budget_ms} ms")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")
    print("=" * 80)

    # 1. Load sequence
    print(f"\n>> Loading {n_frames + 1} frames from {scene_name}...")
    t0 = time.time()
    frames, intrinsics = load_phase10_sequence(
        scene_name=scene_name,
        n_frames=n_frames + 1,
        H=H,
        W=W,
        device=device,
    )
    print(f">> Loaded {len(frames)} frames in {time.time() - t0:.2f}s.")

    # 2. Run policy trajectory
    print(f"\n>> Executing {policy.upper()} policy trajectory (seed={seed}, {n_frames} frames)...")
    summary, frame_logs = run_policy_trajectory(
        policy=policy,
        seed=seed,
        frames=frames,
        intrinsics=intrinsics,
        budget_ms=budget_ms,
        device=device,
        W=W,
        H=H,
    )

    # 3. Save trajectory CSV and summary JSON
    df = pd.DataFrame(frame_logs)
    csv_path = output_dir / f"{policy}_trajectory.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n>> Trajectory CSV saved to {csv_path}")

    summary_path = output_dir / f"{policy}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f">> Summary JSON saved to {summary_path}")

    # 4. Print key milestones across trajectory (e.g. frame 1, 30, 75, 150)
    print("\n" + "=" * 80)
    print(f"{'Frame':<8} | {'N_after':<10} | {'N_opt':<8} | {'% Updated':<10} | {'Coverage':<10} | {'N_warmup':<10} | {'PSNR':<10} | {'SSIM':<8}")
    print("-" * 80)
    milestones = [1, 15, 30, 50, 75, 100, 125, 150]
    for frame_id in milestones:
        matching = df[df["frame"] == frame_id]
        if not matching.empty:
            row = matching.iloc[0]
            n_tot = int(row["n_gaussians"])
            n_opt = int(row["n_optimized"])
            pct_upd = (n_opt / max(1, n_tot)) * 100.0
            cov = float(row.get("coverage", 0.0)) * 100.0
            n_w = int(row.get("n_warmup", 0))
            psnr = float(row["psnr"])
            ssim = float(row["ssim"])
            print(f"{frame_id:<8} | {n_tot:<10,d} | {n_opt:<8,d} | {pct_upd:<9.1f}% | {cov:<9.1f}% | {n_w:<10,d} | {psnr:<9.2f}dB | {ssim:<8.4f}")

    print("=" * 80)
    print(f"Final Statistics: Mean PSNR={summary['mean_psnr']:.2f} dB, Final PSNR={summary['final_psnr']:.2f} dB, Mean SSIM={summary['mean_ssim']:.4f}, FPS={summary['fps']:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
