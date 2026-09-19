#!/usr/bin/env python3
r"""Benchmark Multi-step Micro-convergence: Q(K) vs T(K) for K in {1, 2, 3, 5}.

Evaluates systems trade-offs across micro-step counts:
  - Latency: p50 opt time, p95 opt time, end-to-end frame time
  - Quality: Mean Delta Q (PSNR improvement), final PSNR, SSIM
  - Compute Efficiency: Delta Q / opt_time, Delta Q / frame_time

Usage:
  python experiments/benchmark_k_microsteps.py --scene tum_fr2_xyz --n_frames 20 --device cuda
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pipeline import OnlineReconstructionPipeline
from research.phase10_runtime import load_phase10_sequence
from research.reproducibility import create_provenance_manifest, get_hardware_info


def run_single_k_benchmark(
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    k_micro_steps: int,
    use_adaptive_k: bool = False,
    budget_ms: float = 15.0,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute pipeline for a given micro-step count K or adaptive K."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    W = frames[0]["rgb"].shape[1]
    H = frames[0]["rgb"].shape[0]

    config = {
        "seed": seed,
        "gaussian": {
            "sh_degree": 0,
            "initial_opacity": 0.8,
            "max_gaussians": 150000,
            "initial_scale": 0.02,
            "init_stride": 2,
            "scale_mode": "depth_adaptive",
            "scale_pixel_multiplier": 1.5,
        },
        "system": {
            "max_vram_fraction": 0.70,
            "empty_cache_frequency": 2,
        },
        "rendering": {
            "tile_size": 16,
            "image_width": W,
            "image_height": H,
            "use_surface_aware_depth": False,
            "attribution_top_k": 8,
            "use_fast_attribution": True,
            "backend": "gsplat" if device == "cuda" else "reference",
        },
        "losses": {
            "weight_color": 0.8,
            "weight_depth": 0.5,
            "weight_ssim": 0.2,
        },
        "scheduler": {
            "gpu_budget_ms": budget_ms,
            "policy": "budget_aware",
            "use_knapsack": True,
            "cost_per_gaussian_us": 0.10 if device == "cuda" else 0.5,
            "optimize_ratio": 0.50,
        },
        "training": {
            "n_micro_steps": k_micro_steps,
            "use_adaptive_k": use_adaptive_k,
        },
        "densification": {
            "max_new_per_frame": 4000,
            "strategy": "importance",
            "use_adaptive_thresholds": True,
        },
    }

    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    pipeline.initialize(
        rgb=frames[0]["rgb"],
        depth=frames[0]["depth"],
        intrinsics=intrinsics,
        pose=frames[0]["pose"],
    )

    opt_times_ms: List[float] = []
    frame_times_ms: List[float] = []
    delta_qs: List[float] = []
    post_psnrs: List[float] = []
    ssims: List[float] = []
    mean_ks: List[float] = []

    for f in frames[1:]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        res = pipeline.process_frame(
            rgb=f["rgb"],
            depth=f["depth"],
            gt_pose=f["pose"],
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_frame_ms = (time.perf_counter() - t0) * 1000.0

        opt_t_ms = res.get("opt_time_ms", 0.0)
        psnr_pre = res.get("psnr_pre", 0.0)
        psnr_post = res.get("psnr_post", res.get("psnr", 0.0))
        delta_q = psnr_post - psnr_pre
        ssim_val = res.get("ssim", 0.0)
        k_val = res.get("mean_k", float(k_micro_steps))

        frame_times_ms.append(t_frame_ms)
        post_psnrs.append(psnr_post)
        ssims.append(ssim_val)
        mean_ks.append(k_val)
        if opt_t_ms > 0:
            opt_times_ms.append(opt_t_ms)
            delta_qs.append(delta_q)

    p50_opt = float(np.percentile(opt_times_ms, 50)) if opt_times_ms else 0.0
    p95_opt = float(np.percentile(opt_times_ms, 95)) if opt_times_ms else 0.0
    mean_frame_time = float(np.mean(frame_times_ms))
    mean_delta_q = float(np.mean(delta_qs)) if delta_qs else 0.0
    mean_post_psnr = float(np.mean(post_psnrs))
    final_psnr = float(post_psnrs[-1]) if post_psnrs else 0.0
    mean_ssim = float(np.mean(ssims)) if ssims else 0.0
    fps = 1000.0 / mean_frame_time if mean_frame_time > 0 else 0.0
    overall_mean_k = float(np.mean(mean_ks)) if mean_ks else float(k_micro_steps)

    efficiency_opt = mean_delta_q / (p50_opt + 1e-6)
    efficiency_frame = mean_delta_q / (mean_frame_time + 1e-6)

    label = f"Adaptive (1-{k_micro_steps})" if use_adaptive_k else f"K={k_micro_steps}"

    n_gauss = pipeline.gaussian_model.num_gaussians
    pipeline.cleanup()
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "k": label,
        "mean_k": round(overall_mean_k, 2),
        "mean_delta_q_db": round(mean_delta_q, 4),
        "mean_post_psnr_db": round(mean_post_psnr, 2),
        "final_psnr_db": round(final_psnr, 2),
        "mean_ssim": round(mean_ssim, 4),
        "p50_opt_ms": round(p50_opt, 2),
        "p95_opt_ms": round(p95_opt, 2),
        "frame_time_ms": round(mean_frame_time, 2),
        "fps": round(fps, 1),
        "delta_q_per_opt_ms": round(efficiency_opt, 4),
        "delta_q_per_frame_ms": round(efficiency_frame, 4),
        "n_gaussians": n_gauss,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark K micro-steps")
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz")
    parser.add_argument("--n_frames", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--budget_ms", type=float, default=15.0)
    parser.add_argument("--out_dir", type=str, default="results/k_microsteps_benchmark")
    args = parser.parse_args()

    # Memory Guard: cap CUDA allocation to prevent desktop compositor starvation
    if args.device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.70, 0)
        except (RuntimeError, ValueError):
            pass

    out_path = REPO_ROOT / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Multi-step Micro-convergence Benchmark: K in [1, 2, 3, 5] & Adaptive K ===")
    print(f"Scene: {args.scene} | Frames: {args.n_frames} | Device: {args.device} | Budget: {args.budget_ms} ms")

    frames, intrinsics = load_phase10_sequence(
        scene_name=args.scene,
        n_frames=args.n_frames,
        H=240,
        W=320,
        device=args.device,
    )

    configs_to_run = [
        (1, False),
        (2, False),
        (3, False),
        (5, False),
        (5, True),  # Adaptive K in {1, 2, 3, 5} (Pyramid: K=1 nhiều, K=2 vừa, K=3 ít, K=5 rất ít)
    ]
    records = []

    for k_val, adaptive in configs_to_run:
        tag = f"Adaptive K (1-{k_val})" if adaptive else f"Fixed K = {k_val}"
        print(f"\n>> Benchmarking {tag} ...")
        rec = run_single_k_benchmark(
            frames=frames,
            intrinsics=intrinsics,
            k_micro_steps=k_val,
            use_adaptive_k=adaptive,
            budget_ms=args.budget_ms,
            device=args.device,
        )
        records.append(rec)
        print(f"   {rec['k']} -> Mean K: {rec['mean_k']:.2f} | PSNR: {rec['mean_post_psnr_db']:.2f} dB | SSIM: {rec['mean_ssim']:.4f} | p50 opt: {rec['p50_opt_ms']:.2f} ms | FPS: {rec['fps']:.1f} | ΔQ/opt_ms: {rec['delta_q_per_opt_ms']:.4f}")

    # Manifest with strict provenance
    manifest = create_provenance_manifest(
        dataset_name=args.scene,
        renderer_backend=os.environ.get("ADAPTIVE_3DGS_RENDERER", "gsplat" if args.device == "cuda" else "reference"),
        init_stride=2,
        scale_pixel_multiplier=1.5,
        initial_opacity=0.8,
        n_micro_steps=None,
        psnr_mask="valid_depth",
        extra_metadata={
            "experiment": "K Micro-steps Systems Efficiency Benchmark",
            "configs_tested": [r["k"] for r in records],
            "scene": args.scene,
            "n_frames": args.n_frames,
            "device": args.device,
        }
    )

    with open(out_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(out_path / "k_microsteps_results.json", "w") as f:
        json.dump({"benchmark_results": records, "manifest": manifest}, f, indent=2)

    # Print summary table
    print("\n" + "=" * 110)
    print("AI Systems Micro-convergence Evaluation Table: Q(K) vs T(K)")
    print("=" * 110)
    print(f"| {'Configuration':<18} | {'Mean K':<8} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'p50 opt (ms)':<14} | {'FPS':<6} | {'ΔQ / opt_ms':<12} |")
    print(f"|{'-'*20}|{'-'*10}|{'-'*12}|{'-'*10}|{'-'*16}|{'-'*8}|{'-'*14}|")
    for r in records:
        print(f"| {str(r['k']):<18} | {r['mean_k']:<8.2f} | {r['mean_post_psnr_db']:<10.2f} | {r['mean_ssim']:<8.4f} | {r['p50_opt_ms']:<14.2f} | {r['fps']:<6.1f} | {r['delta_q_per_opt_ms']:<12.4f} |")
    print("=" * 110)


if __name__ == "__main__":
    main()
