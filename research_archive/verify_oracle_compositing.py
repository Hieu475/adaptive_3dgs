#!/usr/bin/env python3
r"""Oracle Verification Experiment: Exact Full Rendering vs Background Cache Compositing.

Verification Objectives:
  1. Measure photometric loss discrepancy: |L_full - L_selective|
  2. Measure relative parameter gradient error on Active Gaussians A:
       Error_grad = ||\nabla_A L_full - \nabla_A L_selective|| / (||\nabla_A L_full|| + \epsilon)
  3. Quantify ray depth-order violation effect in affine compositing:
       C = C_active + T_active * C_frozen

Outputs:
  - results/oracle/oracle_verification_gradient_report.json
  - results/oracle/oracle_verification_gradient_report.md
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pipeline import OnlineReconstructionPipeline
from research.background_cache import FrozenBackgroundCache
from research.rasterizer import render as rasterize_scene
from research.phase10_runtime import load_phase10_sequence
from experiments.run_phase13_frozen_benchmark import build_pipeline_config


def run_oracle_compositing_verification(
    scene: str = "tum_fr2_xyz",
    n_frames: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    active_ratio: float = 0.20,
    out_dir: str = "results/oracle",
) -> Dict[str, Any]:
    """Compare Full rendering vs Background Cache Compositing gradients on active Gaussians."""
    out_path = REPO_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240

    print("=" * 90)
    print("  ORACLE COMPOSITING & GRADIENT FIDELITY VERIFICATION EXPERIMENT")
    print("=" * 90)
    print(f"  Scene:        {scene}")
    print(f"  Frames:       {n_frames}")
    print(f"  Active Ratio: {active_ratio * 100:.1f}%")
    print(f"  Device:       {device}")
    print(f"  Output:       {out_path}")
    print("=" * 90)

    # 1. Load sequence
    frames, intrinsics = load_phase10_sequence(
        scene_name=scene, n_frames=n_frames, H=H, W=W, device=device
    )

    # 2. Initialize pipeline to get authentic production Gaussian model
    cfg = build_pipeline_config(policy="error_influence", seed=42, budget_ms=15.0, W=W, H=H, device=device)
    pipeline = OnlineReconstructionPipeline(config=cfg, device=device)
    pipeline.initialize(
        rgb=frames[0]["rgb"],
        depth=frames[0]["depth"],
        intrinsics=intrinsics,
        pose=frames[0].get("pose", torch.eye(4, device=torch.device(device))),
    )

    N = pipeline.gaussian_model.num_gaussians
    print(f">> Initialized pipeline GaussianModel with {N:,} primitives.")

    bg_cache = FrozenBackgroundCache(device=device)
    results = []

    for t in range(1, len(frames)):
        frame = frames[t]
        rgb_gt = frame["rgb"]
        depth_gt = frame["depth"]
        pose = frame.get("pose", torch.eye(4, device=torch.device(device)))

        # Define active mask (simulating active selected primitives)
        torch.manual_seed(42 + t * 100)
        perm = torch.randperm(N, device=device)
        n_active = max(10, int(N * active_ratio))
        active_mask = torch.zeros(N, dtype=torch.bool, device=device)
        active_mask[perm[:n_active]] = True
        frozen_mask = ~active_mask

        # --- A. SELECTIVE COMPOSITED RENDERING ---
        bg_cache.build_cache(
            model=pipeline.gaussian_model,
            frozen_mask=frozen_mask,
            extrinsics=pose,
            intrinsics=intrinsics,
            image_width=W,
            image_height=H,
            tile_size=16,
        )
        active_subset = pipeline.gaussian_model.get_optimization_subset(active_mask, extrinsics=pose)
        active_means = active_subset['means3D']
        active_means.retain_grad()

        comp_out = bg_cache.composite_with_active(
            active_subset=active_subset,
            extrinsics=pose,
            intrinsics=intrinsics,
            image_width=W,
            image_height=H,
            tile_size=16,
        )
        l_sel = torch.mean((comp_out['color'] - rgb_gt) ** 2)
        grad_sel = torch.autograd.grad(l_sel, active_means, retain_graph=False)[0]

        # --- B. FULL EXACT RENDERING ---
        all_mask = torch.ones(N, dtype=torch.bool, device=device)
        all_subset = pipeline.gaussian_model.get_optimization_subset(all_mask, extrinsics=pose)
        all_means = all_subset['means3D']
        all_means.retain_grad()

        full_out = rasterize_scene(
            means3D=all_means,
            cov3D=all_subset['cov3D'],
            colors=all_subset['colors'],
            opacities=all_subset['opacities'],
            extrinsics=pose,
            intrinsics=intrinsics,
            image_width=W,
            image_height=H,
            tile_size=16,
        )
        l_full = torch.mean((full_out['color'] - rgb_gt) ** 2)
        grad_all = torch.autograd.grad(l_full, all_means, retain_graph=False)[0]
        grad_full_active = grad_all[active_mask]

        # --- C. METRICS ---
        loss_diff = float(torch.abs(l_full - l_sel).item())
        grad_norm_full = float(torch.norm(grad_full_active).item())
        grad_norm_diff = float(torch.norm(grad_full_active - grad_sel).item())
        rel_grad_error = float(grad_norm_diff / (grad_norm_full + 1e-7))
        psnr_full = float(-10.0 * np.log10(max(1e-8, l_full.item())))
        psnr_sel = float(-10.0 * np.log10(max(1e-8, l_sel.item())))

        res = {
            "frame": t,
            "N_total": N,
            "N_active": n_active,
            "loss_full": float(l_full.item()),
            "loss_selective": float(l_sel.item()),
            "loss_diff": loss_diff,
            "psnr_full": psnr_full,
            "psnr_selective": psnr_sel,
            "grad_norm_full": grad_norm_full,
            "grad_norm_diff": grad_norm_diff,
            "relative_gradient_error": rel_grad_error,
        }
        results.append(res)
        print(f"Frame {t}: Loss Full={res['loss_full']:.5f} | Selective={res['loss_selective']:.5f} | "
              f"Rel Grad Error={rel_grad_error:.4f} ({rel_grad_error*100:.2f}%)")

    mean_rel_grad_err = float(np.mean([r["relative_gradient_error"] for r in results]))
    max_rel_grad_err = float(np.max([r["relative_gradient_error"] for r in results]))
    mean_loss_diff = float(np.mean([r["loss_diff"] for r in results]))

    report = {
        "scene": scene,
        "n_frames_evaluated": len(results),
        "active_ratio": active_ratio,
        "mean_relative_gradient_error": mean_rel_grad_err,
        "max_relative_gradient_error": max_rel_grad_err,
        "mean_loss_diff": mean_loss_diff,
        "frame_details": results,
    }

    with open(out_path / "oracle_verification_gradient_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Markdown Report
    md_lines = [
        "# Oracle Compositing & Gradient Fidelity Verification Report",
        "",
        f"**Scene:** `{scene}` | **Frames:** {len(results)} | **Active Ratio:** {active_ratio*100:.1f}%",
        "",
        "## 1. Summary of Gradient Discrepancy",
        "",
        f"- **Mean Relative Gradient Error**: `{mean_rel_grad_err:.4f}` ({mean_rel_grad_err*100:.2f}%)",
        f"- **Max Relative Gradient Error**: `{max_rel_grad_err:.4f}` ({max_rel_grad_err*100:.2f}%)",
        f"- **Mean Photometric Loss Discrepancy**: `{mean_loss_diff:.6f}`",
        "",
        "## 2. Per-Frame Measurements",
        "",
        "| Frame | $L_{full}$ | $L_{selective}$ | PSNR Full | PSNR Sel | Relative Grad Error |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        md_lines.append(
            f"| {r['frame']} | {r['loss_full']:.5f} | {r['loss_selective']:.5f} | "
            f"{r['psnr_full']:.2f} dB | {r['psnr_selective']:.2f} dB | "
            f"**{r['relative_gradient_error']:.4f}** ({r['relative_gradient_error']*100:.1f}%) |"
        )

    md_lines.extend([
        "",
        "## 3. Scientific Finding on Oracle Validity",
        "",
    ])
    if mean_rel_grad_err > 0.05:
        md_lines.append(
            "> [!WARNING]\n"
            f"> **Depth Sorting Violation Confirmed:** Relative gradient error is {mean_rel_grad_err*100:.1f}% (> 5%). "
            "Because active and frozen Gaussians interleave along viewing rays, affine compositing "
            "introduces systematic gradient distortion. "
            "For authoritative oracle intervention targets ($U_i^\\star$), exact full rendering must be used."
        )
    else:
        md_lines.append(
            "> [!NOTE]\n"
            f"> **Compositing Fidelity High:** Relative gradient error is {mean_rel_grad_err*100:.1f}% (<= 5%)."
        )

    md_report = "\n".join(md_lines)
    with open(out_path / "oracle_verification_gradient_report.md", "w") as f:
        f.write(md_report)

    print(f"\n>> Verification completed. Report saved to: {out_path / 'oracle_verification_gradient_report.md'}\n")
    print(md_report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle Compositing Verification")
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz")
    parser.add_argument("--n_frames", type=int, default=5)
    parser.add_argument("--active_ratio", type=float, default=0.20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="results/oracle")
    args = parser.parse_args()

    run_oracle_compositing_verification(
        scene=args.scene,
        n_frames=args.n_frames,
        device=args.device,
        active_ratio=args.active_ratio,
        out_dir=args.out_dir,
    )
