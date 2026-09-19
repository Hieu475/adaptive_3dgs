import sys
from pathlib import Path
REPO_ROOT = Path("/home/nguyen_quoc_hieu/Documents/adaptive_3dgs")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
import json
import time
import torch
import pandas as pd
import numpy as np

from experiments.run_phase13_frozen_benchmark import (
    load_phase10_sequence,
    build_pipeline_config,
    run_policy_trajectory,
)

def run_warmup_ablation(n_frames: int = 40, scene_name: str = "tum_fr2_xyz", seed: int = 42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W, H = 320, 240
    budget_ms = 15.0

    print("=" * 80)
    print("  ITEM 2: WARM-UP PARAMETER GRID ABLATION (OURS vs ERROR_INFLUENCE)")
    print("=" * 80)
    print(f"  Scene:    {scene_name}")
    print(f"  Frames:   {n_frames}")
    print(f"  Seed:     {seed}")
    print(f"  Device:   {device}")
    print("=" * 80)

    # 1. Preload sequence
    frames, intrinsics = load_phase10_sequence(
        scene_name=scene_name,
        n_frames=n_frames + 1,
        H=H,
        W=W,
        device=device,
    )

    results = []

    # 2. First, run baseline ERROR_INFLUENCE
    print("\n>> Running baseline: ERROR_INFLUENCE...")
    ei_summary, ei_logs = run_policy_trajectory(
        policy="error_influence",
        seed=seed,
        frames=frames,
        intrinsics=intrinsics,
        budget_ms=budget_ms,
        device=device,
        W=W,
        H=H,
    )
    ei_res = {
        "policy": "error_influence",
        "warmup_ratio": 0.0,
        "warmup_k": 0,
        "mean_psnr": ei_summary["mean_psnr"],
        "final_psnr": ei_summary["final_psnr"],
        "mean_ssim": ei_summary["mean_ssim"],
        "final_ssim": ei_summary["final_ssim"],
        "fps": ei_summary["fps"],
        "n_gaussians": ei_summary.get("N_final", 0),
    }
    results.append(ei_res)
    print(f"   ERROR_INFLUENCE: Mean PSNR={ei_res['mean_psnr']:.2f} dB, Final PSNR={ei_res['final_psnr']:.2f} dB, SSIM={ei_res['mean_ssim']:.4f}, FPS={ei_res['fps']:.1f}")

    # 3. Grid search over warmup_budget_ratio and warmup_k
    ratios = [0.0, 0.10, 0.20, 0.30, 0.40]
    k_vals = [1, 2, 3, 5]

    for ratio in ratios:
        if ratio == 0.0:
            # warmup turned off for OURS
            print(f"\n>> Running OURS with warmup_budget_ratio=0.0 (Warmup OFF)...")
            cfg = build_pipeline_config(policy="ours", seed=seed, budget_ms=budget_ms, W=W, H=H, device=device)
            cfg["scheduler"]["enable_warmup"] = False
            cfg["scheduler"]["warmup_budget_ratio"] = 0.0
            
            summary, _ = run_policy_trajectory(
                policy="ours", seed=seed, frames=frames, intrinsics=intrinsics,
                budget_ms=budget_ms, device=device, W=W, H=H,
            )
            res = {
                "policy": "ours",
                "warmup_ratio": 0.0,
                "warmup_k": 0,
                "mean_psnr": summary["mean_psnr"],
                "final_psnr": summary["final_psnr"],
                "mean_ssim": summary["mean_ssim"],
                "final_ssim": summary["final_ssim"],
                "fps": summary["fps"],
                "n_gaussians": summary.get("N_final", 0),
            }
            results.append(res)
            print(f"   OURS (ratio=0.0): Mean PSNR={res['mean_psnr']:.2f} dB, Final={res['final_psnr']:.2f} dB, SSIM={res['mean_ssim']:.4f}, FPS={res['fps']:.1f}")
            continue

        for k in k_vals:
            print(f"\n>> Running OURS with warmup_budget_ratio={ratio:.2f}, warmup_k={k}...")
            cfg = build_pipeline_config(policy="ours", seed=seed, budget_ms=budget_ms, W=W, H=H, device=device)
            cfg["scheduler"]["enable_warmup"] = True
            cfg["scheduler"]["warmup_budget_ratio"] = ratio
            cfg["scheduler"]["warmup_k"] = k

            # Custom runner using modified config
            set_seed = torch.manual_seed(seed)
            from research.pipeline import OnlineReconstructionPipeline
            pipeline = OnlineReconstructionPipeline(config=cfg, device=device)
            pipeline.initialize(
                rgb=frames[0]["rgb"],
                depth=frames[0]["depth"],
                intrinsics=intrinsics,
                pose=frames[0].get("pose"),
            )
            t_start = time.perf_counter()
            f_logs = []
            for t in range(1, len(frames)):
                m = pipeline.process_frame(rgb=frames[t]["rgb"], depth=frames[t]["depth"], gt_pose=frames[t].get("pose"))
                f_logs.append(m)
            traj_time = time.perf_counter() - t_start
            pipeline.cleanup()

            df_sub = pd.DataFrame(f_logs)
            mean_psnr = float(df_sub["psnr"].mean())
            final_psnr = float(df_sub["psnr"].iloc[-1])
            mean_ssim = float(df_sub["ssim"].mean())
            final_ssim = float(df_sub["ssim"].iloc[-1])
            fps = float(len(f_logs) / max(1e-4, traj_time))

            res = {
                "policy": "ours",
                "warmup_ratio": ratio,
                "warmup_k": k,
                "mean_psnr": mean_psnr,
                "final_psnr": final_psnr,
                "mean_ssim": mean_ssim,
                "final_ssim": final_ssim,
                "fps": fps,
                "n_gaussians": pipeline.gaussian_model.num_gaussians,
            }
            results.append(res)
            delta_psnr = mean_psnr - ei_res["mean_psnr"]
            delta_ssim = mean_ssim - ei_res["mean_ssim"]
            print(f"   ratio={ratio:.2f}, k={k}: Mean PSNR={mean_psnr:.2f} dB (Δ={delta_psnr:+.2f} dB), SSIM={mean_ssim:.4f} (Δ={delta_ssim:+.4f}), FPS={fps:.1f}")

    # Output summary table
    out_df = pd.DataFrame(results)
    out_dir = Path("results/warmup_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "warmup_grid_results.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n>> Saved ablation results to {out_csv}")

    print("\n" + "=" * 95)
    print(f"{'Policy':<16} | {'Warmup Ratio':<12} | {'Warmup K':<8} | {'Mean PSNR':<10} | {'Final PSNR':<10} | {'Mean SSIM':<10} | {'FPS':<6}")
    print("-" * 95)
    for _, row in out_df.iterrows():
        p_name = row['policy']
        w_r = f"{row['warmup_ratio']:.2f}"
        w_k = str(int(row['warmup_k']))
        m_p = f"{row['mean_psnr']:.2f} dB"
        f_p = f"{row['final_psnr']:.2f} dB"
        m_s = f"{row['mean_ssim']:.4f}"
        fps_str = f"{row['fps']:.1f}"
        print(f"{p_name:<16} | {w_r:<12} | {w_k:<8} | {m_p:<10} | {f_p:<10} | {m_s:<10} | {fps_str:<6}")
    print("=" * 95)

if __name__ == "__main__":
    run_warmup_ablation(n_frames=40)
