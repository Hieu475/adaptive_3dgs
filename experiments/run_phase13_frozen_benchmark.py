#!/usr/bin/env python3
"""Phase 13: Unified Authoritative Frozen Benchmark Protocol.

Executes the definitive Phase 13 scientific evaluation across:
  - 6 Policies: NO_OP, ERROR_ONLY, ERROR_INFLUENCE, ERROR_INFLUENCE_TEMPORAL, OURS, FULL
  - 5 Seeds: [42, 43, 44, 45, 46]
  - 150 Frames on TUM RGB-D (default: tum_fr2_xyz)

Measures complete 10 metrics with 95% bootstrap confidence intervals:
  1. Mean & Final PSNR (dB)
  2. Mean & Final SSIM
  3. Mean & Final Depth L1
  4. Final Gaussians (N_final)
  5. Mean New / Frame (N_new)
  6. Mean Selected / Frame (N_opt)
  7. Mean K Micro-steps / Frame
  8. Mean Optimization Time (ms)
  9. Mean Frame Time (ms) & FPS
  10. Peak VRAM (MB)

Produces:
  - results/phase13_frozen_benchmark/phase13_frozen_results.json
  - results/phase13_frozen_benchmark/phase13_frozen_summary.md
  - results/phase13_frozen_benchmark/manifest.json
"""
import os
import sys
import time
import json
import gc
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import torch

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pipeline import OnlineReconstructionPipeline
from research.scheduler import OptimizationPolicy
from research.phase10_runtime import load_phase10_sequence
from research.reproducibility import (
    create_provenance_manifest,
    bootstrap_ci,
    set_seed,
    get_hardware_info,
    get_git_commit,
    get_git_branch,
)


POLICIES_CANONICAL = [
    "no_op",
    "error_only",
    "error_influence",
    "error_influence_temporal",
    "ours",
    "full",
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def build_pipeline_config(
    policy: str,
    seed: int,
    budget_ms: float = 15.0,
    W: int = 320,
    H: int = 240,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Build standardized pipeline config for Phase 13 benchmark."""
    is_ours = policy in ["ours", "budget_aware", OptimizationPolicy.OURS.value, OptimizationPolicy.BUDGET_AWARE.value]
    return {
        "seed": seed,
        "system": {
            "max_vram_fraction": 0.70,
            "empty_cache_frequency": 10,
        },
        "gaussian": {
            "sh_degree": 0,
            "initial_opacity": 0.8,
            "max_gaussians": 150000,
            "initial_scale": 0.02,
            "init_stride": 2,
            "scale_mode": "depth_adaptive",
            "scale_pixel_multiplier": 1.5,
        },
        "rendering": {
            "tile_size": 16,
            "image_width": W,
            "image_height": H,
            "use_surface_aware_depth": False,
            "attribution_top_k": 4,
            "use_fast_attribution": True,
            "backend": "gsplat" if device.startswith("cuda") else "reference",
        },
        "scheduler": {
            "gpu_budget_ms": budget_ms,
            "policy": policy,
            "cost_per_gaussian_us": 2.0,
            "use_knapsack": True,
        },
        "training": {
            "n_micro_steps": 5,
            "use_adaptive_k": is_ours,
            "lr_position": 0.00016,
            "lr_rotation": 0.001,
            "lr_scale": 0.005,
            "lr_opacity": 0.05,
            "lr_color": 0.0025,
        },
        "losses": {
            "weight_color": 0.8,
            "weight_depth": 0.5,
            "weight_ssim": 0.2,
        },
        "densification": {
            "max_new_per_frame": 4000,
            "strategy": "importance",
            "use_adaptive_thresholds": True,
        },
    }


def run_policy_trajectory(
    policy: str,
    seed: int,
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    budget_ms: float = 15.0,
    device: str = "cuda",
    W: int = 320,
    H: int = 240,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute a single policy trajectory across preloaded frames."""
    set_seed(seed)
    
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    cfg = build_pipeline_config(
        policy=policy,
        seed=seed,
        budget_ms=budget_ms,
        W=W,
        H=H,
        device=device,
    )

    pipeline = OnlineReconstructionPipeline(config=cfg, device=device)
    
    # Initialize on frame 0
    t_init_0 = time.perf_counter()
    pipeline.initialize(
        rgb=frames[0]["rgb"],
        depth=frames[0]["depth"],
        intrinsics=intrinsics,
        pose=frames[0].get("pose", torch.eye(4, device=torch.device(device))),
    )
    n_init = pipeline.gaussian_model.num_gaussians

    frame_logs: List[Dict[str, Any]] = []
    t_traj_start = time.perf_counter()

    for t in range(1, len(frames)):
        n_before = pipeline.gaussian_model.num_gaussians
        t_frame_start = time.perf_counter()

        m = pipeline.process_frame(
            rgb=frames[t]["rgb"],
            depth=frames[t]["depth"],
            gt_pose=frames[t].get("pose", None),
        )
        t_frame_ms = (time.perf_counter() - t_frame_start) * 1000.0
        n_after = pipeline.gaussian_model.num_gaussians
        n_new = max(0, n_after - n_before)

        record = {
            "frame": t,
            "psnr": float(m["psnr"]),
            "ssim": float(m["ssim"]),
            "depth_l1": float(m["depth_l1"]),
            "n_gaussians": int(m["n_gaussians"]),
            "n_new": n_new,
            "n_optimized": int(m["n_optimized"]),
            "mean_k": float(m.get("mean_k", 0.0)),
            "opt_time_ms": float(m["opt_time_ms"]),
            "frame_time_ms": t_frame_ms,
            "peak_vram_mb": float(m.get("peak_vram_mb", 0.0)),
        }
        frame_logs.append(record)

    total_wall_sec = time.perf_counter() - t_traj_start

    # Trajectory-level summary aggregation
    psnrs = np.array([r["psnr"] for r in frame_logs])
    ssims = np.array([r["ssim"] for r in frame_logs])
    depths = np.array([r["depth_l1"] for r in frame_logs])
    n_news = np.array([r["n_new"] for r in frame_logs])
    n_opts = np.array([r["n_optimized"] for r in frame_logs])
    ks = np.array([r["mean_k"] for r in frame_logs])
    opt_times = np.array([r["opt_time_ms"] for r in frame_logs])
    frame_times = np.array([r["frame_time_ms"] for r in frame_logs])
    vrams = np.array([r["peak_vram_mb"] for r in frame_logs])

    peak_vram = float(np.max(vrams)) if len(vrams) > 0 else 0.0
    if str(device).startswith("cuda") and torch.cuda.is_available():
        peak_vram = max(peak_vram, float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)))

    summary = {
        "policy": policy,
        "seed": seed,
        "n_frames": len(frame_logs),
        "N_init": n_init,
        "N_final": int(frame_logs[-1]["n_gaussians"]) if frame_logs else n_init,
        "N_new_total": int(np.sum(n_news)),
        "mean_new_per_frame": float(np.mean(n_news)),
        "final_psnr": float(psnrs[-1]) if len(psnrs) > 0 else 0.0,
        "mean_psnr": float(np.mean(psnrs)),
        "final_ssim": float(ssims[-1]) if len(ssims) > 0 else 0.0,
        "mean_ssim": float(np.mean(ssims)),
        "final_depth_l1": float(depths[-1]) if len(depths) > 0 else 0.0,
        "mean_depth_l1": float(np.mean(depths)),
        "mean_n_optimized": float(np.mean(n_opts)),
        "mean_k": float(np.mean(ks)),
        "mean_opt_time_ms": float(np.mean(opt_times)),
        "mean_frame_time_ms": float(np.mean(frame_times)),
        "fps": float(len(frame_logs) / max(total_wall_sec, 1e-4)),
        "peak_vram_mb": peak_vram,
        "total_wall_sec": total_wall_sec,
        "catastrophic_failures": int(np.sum(np.isnan(psnrs) | np.isinf(psnrs))),
    }

    # Safe resource cleanup
    pipeline.cleanup()
    del pipeline
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, frame_logs


def aggregate_policy_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean, std, and 95% bootstrap confidence intervals across seeds."""
    if not runs:
        return {}

    def stats_for(metric_key: str, do_ci: bool = True) -> Dict[str, Any]:
        vals = [r[metric_key] for r in runs]
        res = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        if do_ci:
            ci_low, ci_high = bootstrap_ci(vals, stat_fn=np.mean, n_boot=1000)
            res["ci_95"] = [ci_low, ci_high]
        return res

    return {
        "policy": runs[0]["policy"],
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "mean_psnr": stats_for("mean_psnr"),
        "final_psnr": stats_for("final_psnr"),
        "mean_ssim": stats_for("mean_ssim"),
        "final_ssim": stats_for("final_ssim"),
        "mean_depth_l1": stats_for("mean_depth_l1"),
        "final_depth_l1": stats_for("final_depth_l1"),
        "N_final": stats_for("N_final", do_ci=False),
        "mean_new_per_frame": stats_for("mean_new_per_frame", do_ci=False),
        "mean_n_optimized": stats_for("mean_n_optimized", do_ci=False),
        "mean_k": stats_for("mean_k", do_ci=False),
        "mean_opt_time_ms": stats_for("mean_opt_time_ms"),
        "mean_frame_time_ms": stats_for("mean_frame_time_ms"),
        "fps": stats_for("fps", do_ci=False),
        "peak_vram_mb": stats_for("peak_vram_mb", do_ci=False),
        "catastrophic_failures": int(sum(r["catastrophic_failures"] for r in runs)),
    }


def generate_markdown_summary(
    aggregated_results: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    n_frames: int,
    budget_ms: float,
) -> str:
    """Format publication-quality Markdown benchmark summary report."""
    md = []
    md.append("# Phase 13 Authoritative Frozen Benchmark Report\n")
    md.append(f"**Execution Timestamp**: `{manifest['timestamp']}`  ")
    md.append(f"**Git Commit**: `{manifest['git_sha']}` (Branch: `{manifest['git_branch']}`)  ")
    md.append(f"**Dataset**: `{manifest['dataset_name']}` ({n_frames} frames, resolution: `{manifest.get('resolution', '320x240')}`)  ")
    md.append(f"**Optimization Budget**: `{budget_ms} ms` | **Backend**: `{manifest['renderer_backend']}`  ")
    md.append(f"**Hardware**: `{manifest['gpu']}` ({manifest['vram_gb']} GB VRAM, Driver / CUDA `{manifest['cuda_version']}`)  \n")

    md.append("## 1. Primary Benchmark Results Across Policies (5 Seeds Mean ± Std [95% CI])\n")
    header = (
        "| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | Depth L1 | "
        "Selected/Frame | Mean K | Opt Time (ms) | FPS | Peak VRAM |"
    )
    sep = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    md.append(header)
    md.append(sep)

    for agg in aggregated_results:
        p_name = agg["policy"].upper()
        p_mean = f"{agg['mean_psnr']['mean']:.2f} ± {agg['mean_psnr']['std']:.2f}"
        p_final = f"{agg['final_psnr']['mean']:.2f} ± {agg['final_psnr']['std']:.2f}"
        s_mean = f"{agg['mean_ssim']['mean']:.4f} ± {agg['mean_ssim']['std']:.4f}"
        d_mean = f"{agg['mean_depth_l1']['mean']:.4f} ± {agg['mean_depth_l1']['std']:.4f}"
        n_opt = f"{agg['mean_n_optimized']['mean']:.0f}"
        k_val = f"{agg['mean_k']['mean']:.1f}"
        t_opt = f"{agg['mean_opt_time_ms']['mean']:.1f} ms"
        fps = f"{agg['fps']['mean']:.1f}"
        vram = f"{agg['peak_vram_mb']['mean']:.1f} MB"

        md.append(
            f"| **{p_name}** | {p_mean} | {p_final} | {s_mean} | {d_mean} | "
            f"{n_opt} | {k_val} | {t_opt} | {fps} | {vram} |"
        )

    md.append("\n## 2. Scientific Headroom & Policy Gain Analysis\n")
    
    # Extract means
    by_pol = {a["policy"]: a for a in aggregated_results}
    if "full" in by_pol and "no_op" in by_pol:
        h_mean = by_pol["full"]["mean_psnr"]["mean"] - by_pol["no_op"]["mean_psnr"]["mean"]
        h_final = by_pol["full"]["final_psnr"]["mean"] - by_pol["no_op"]["final_psnr"]["mean"]
        md.append(f"- **Substrate Headroom ($H_{{substrate}}$)**: `{h_mean:+.2f} dB` (Mean), `{h_final:+.2f} dB` (Final)")

    if "ours" in by_pol and "error_only" in by_pol:
        d_err = by_pol["ours"]["mean_psnr"]["mean"] - by_pol["error_only"]["mean_psnr"]["mean"]
        md.append(rf"- **Gain over Raw Error-Only ($\Delta Q$)**: `{d_err:+.2f} dB` (SSIM $\Delta$: `{by_pol['ours']['mean_ssim']['mean'] - by_pol['error_only']['mean_ssim']['mean']:+.4f}`)")

    if "ours" in by_pol and "error_influence_temporal" in by_pol:
        d_temp = by_pol["ours"]["mean_psnr"]["mean"] - by_pol["error_influence_temporal"]["mean_psnr"]["mean"]
        md.append(rf"- **Gain over Temporal Error×Influence ($\Delta Q$)**: `{d_temp:+.2f} dB`")

    md.append("\n## 3. Provenance & Reproducibility Guarantees\n")
    md.append(f"- **Git Commit Hash**: `{manifest['git_sha']}`")
    md.append(f"- **CUDA Rendering Backend**: `{manifest['renderer_backend']}` (Production: gsplat, Prototypes: custom_cuda)")
    md.append(f"- **VRAM Guard Applied**: Yes (`max_vram_fraction=0.70`, reserved OS headroom: 1.8 GB)")
    md.append(f"- **Catastrophic Failures (NaN/Inf)**: `{sum(a['catastrophic_failures'] for a in aggregated_results)}`")
    md.append("- **Verification**: Frozen benchmark conducted on current repository HEAD with exact seed control.\n")

    return "\n".join(md)


def run_phase13_frozen_benchmark(
    dataset_name: str = "tum_fr2_xyz",
    n_frames: int = 150,
    seeds: Optional[List[int]] = None,
    policies: Optional[List[str]] = None,
    budget_ms: float = 15.0,
    device: str = "cuda",
    output_dir: Optional[str] = None,
    W: int = 320,
    H: int = 240,
) -> Dict[str, Any]:
    """Execute authoritative Phase 13 multi-seed frozen benchmark."""
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if policies is None:
        policies = POLICIES_CANONICAL

    out_path = Path(output_dir) if output_dir else REPO_ROOT / "results" / "phase13_frozen_benchmark"
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  PHASE 13: AUTHORITATIVE FROZEN BENCHMARK SUITE")
    print("=" * 80)
    print(f"  Dataset:    {dataset_name}")
    print(f"  Frames:     {n_frames}")
    print(f"  Resolution: {W}x{H}")
    print(f"  Budget:     {budget_ms} ms")
    print(f"  Policies:   {policies}")
    print(f"  Seeds:      {seeds}")
    print(f"  Device:     {device}")
    print(f"  Output:     {out_path}")
    print("=" * 80)

    # 1. Load sequence once
    n_load = n_frames + 1
    print(f"\n>> Loading {n_load} frames from {dataset_name}...")
    t_load_0 = time.time()
    frames, intrinsics = load_phase10_sequence(
        scene_name=dataset_name,
        n_frames=n_load,
        H=H,
        W=W,
        device=device,
    )
    print(f">> Loaded {len(frames)} frames in {time.time() - t_load_0:.2f}s.")

    # 2. Provenance Manifest
    manifest = create_provenance_manifest(
        dataset_name=dataset_name,
        renderer_backend=os.environ.get("ADAPTIVE_3DGS_RENDERER", "gsplat" if device.startswith("cuda") else "reference"),
        init_stride=2,
        scale_pixel_multiplier=1.5,
        initial_opacity=0.8,
        n_micro_steps=5,
        psnr_mask="valid_depth",
        extra_metadata={
            "protocol_version": "phase13-authoritative-v1",
            "experiment": "Phase 13 Frozen Benchmark Across 6 Policies x 5 Seeds",
            "scene": dataset_name,
            "seeds": seeds,
            "policies": policies,
            "budget_ms": budget_ms,
            "n_frames_evaluated": n_frames,
            "resolution": f"{W}x{H}",
        }
    )
    with open(out_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 3. Execution Loop
    all_raw_runs = []
    runs_by_policy: Dict[str, List[Dict[str, Any]]] = {pol: [] for pol in policies}

    total_runs = len(policies) * len(seeds)
    run_idx = 0

    for pol in policies:
        print(f"\n================================================================================")
        print(f"  POLICY: {pol.upper()} ({len(seeds)} seeds)")
        print(f"================================================================================")
        for seed in seeds:
            run_idx += 1
            print(f"[{run_idx:2d}/{total_runs:2d}] Running Policy={pol:<24} Seed={seed} ...", end=" ", flush=True)
            summary, frame_logs = run_policy_trajectory(
                policy=pol,
                seed=seed,
                frames=frames,
                intrinsics=intrinsics,
                budget_ms=budget_ms,
                device=device,
                W=W,
                H=H,
            )
            runs_by_policy[pol].append(summary)
            all_raw_runs.append(summary)
            print(
                f"Done: PSNR={summary['mean_psnr']:.2f}dB (final={summary['final_psnr']:.2f}), "
                f"SSIM={summary['mean_ssim']:.4f}, Opt={summary['mean_n_optimized']:.0f}, "
                f"K={summary['mean_k']:.1f}, OptTime={summary['mean_opt_time_ms']:.1f}ms, "
                f"Wall={summary['total_wall_sec']:.1f}s"
            )

    # 4. Aggregations across seeds
    aggregated_results = []
    for pol in policies:
        agg = aggregate_policy_runs(runs_by_policy[pol])
        aggregated_results.append(agg)

    # 5. Generate Markdown Report
    summary_md = generate_markdown_summary(
        aggregated_results=aggregated_results,
        manifest=manifest,
        n_frames=n_frames,
        budget_ms=budget_ms,
    )

    # 6. Save Artifacts
    full_export = {
        "manifest": manifest,
        "aggregated": aggregated_results,
        "raw_runs": all_raw_runs,
    }
    results_json_path = out_path / "phase13_frozen_results.json"
    with open(results_json_path, "w") as f:
        json.dump(full_export, f, indent=2)

    summary_md_path = out_path / "phase13_frozen_summary.md"
    with open(summary_md_path, "w") as f:
        f.write(summary_md)

    print("\n" + "=" * 80)
    print("  PHASE 13 FROZEN BENCHMARK EXECUTION COMPLETE")
    print("=" * 80)
    print(f"  Results JSON: {results_json_path}")
    print(f"  Summary MD:   {summary_md_path}")
    print(f"  Manifest:     {out_path / 'manifest.json'}")
    print("\n" + summary_md)

    return full_export


def main():
    parser = argparse.ArgumentParser(description="Phase 13 Unified Frozen Benchmark Protocol")
    parser.add_argument("--dataset", type=str, default="tum_fr2_xyz", help="TUM scene name")
    parser.add_argument("--n_frames", type=int, default=150, help="Number of evaluated frames (default: 150)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Seeds to evaluate")
    parser.add_argument("--policies", nargs="+", type=str, default=POLICIES_CANONICAL, help="Policies to evaluate")
    parser.add_argument("--budget_ms", type=float, default=15.0, help="GPU optimization budget in ms")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--width", type=int, default=320, help="Benchmark width (default: 320)")
    parser.add_argument("--height", type=int, default=240, help="Benchmark height (default: 240)")
    args = parser.parse_args()

    run_phase13_frozen_benchmark(
        dataset_name=args.dataset,
        n_frames=args.n_frames,
        seeds=args.seeds,
        policies=args.policies,
        budget_ms=args.budget_ms,
        device=args.device,
        output_dir=args.output_dir,
        W=args.width,
        H=args.height,
    )


if __name__ == "__main__":
    main()
