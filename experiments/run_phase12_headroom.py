#!/usr/bin/env python3
r"""Phase 12-R: Reconstruction Headroom Audit.

Measures the empirical optimization headroom H_substrate = Q_FULL - Q_NOOP
after fixing initialization (stride 4→2, constant→depth-adaptive scale) and
densification (fixed 80/frame→demand-driven) bottlenecks.

This script runs INDEPENDENTLY of Phase 10/12 evidence.  All output goes to
    results/phase12_headroom_audit/
and uses configs/protocol_v2_headroom.yaml.

Experiment stages (run sequentially via --stage):
    smoke     — 10-frame sanity check (single seed, single config)
    R1        — Initialization headroom (legacy vs new init, legacy dense)
    R2        — Densification headroom (legacy vs demand-driven dense, new init)
    R3        — Trajectory scaling (30/75/150 frames, best config)
    R4        — Full scheduler comparison (only if headroom exists)

Key design decisions:
    - NO model bundle or learned selector is used.  Only NO_OP and FULL
      policies (stage 1).  This isolates reconstruction substrate quality
      from scheduler intelligence.
    - Densification can be toggled off (max_new_per_frame=0) for the
      three-way decomposition: NoOp+Static / NoOp+Densify / Full+Densify.
    - Coverage metric added: fraction of pixels with Gaussian contribution > ε.
"""
import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import torch
import numpy as np

# Repository root setup
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pipeline import OnlineReconstructionPipeline
from research.phase10_runtime import load_phase10_sequence
from research.attribution import render_with_attribution
from research.rasterizer import render as rasterize_scene
from research.densification import compute_depth_adaptive_scale

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = REPO_ROOT / "results" / "phase12_headroom_audit"
PROTOCOL_FILE = REPO_ROOT / "configs" / "protocol_v2_headroom.yaml"

W = 320
H = 240
BUDGET_MS = 15.0
SAFETY_FACTOR = 1.10
SCENE = "tum_fr2_xyz"
DEPTH_SCALE = 5000.0


# ---------------------------------------------------------------------------
# Pipeline config builder
# ---------------------------------------------------------------------------
def build_pipeline_config(
    *,
    init_stride: int = 2,
    scale_mode: str = "depth_adaptive",
    initial_scale: float = 0.02,
    scale_pixel_multiplier: float = 1.5,
    max_new_per_frame: int = 4000,
    max_gaussians: int = 150000,
    budget_ms: float = BUDGET_MS,
    policy: str = "full",
    seed: int = 42,
) -> Dict[str, Any]:
    """Build a pipeline config dict with explicit control over experimental variables."""
    is_full = (policy == "full")
    return {
        "seed": seed,
        "gaussian": {
            "sh_degree": 0,
            "initial_opacity": 0.8,
            "max_gaussians": max_gaussians,
            "initial_scale": initial_scale,
            "init_stride": init_stride,
            "scale_mode": scale_mode,
            "scale_pixel_multiplier": scale_pixel_multiplier,
        },
        "rendering": {
            "tile_size": 16,
            "image_width": W,
            "image_height": H,
            "use_surface_aware_depth": False,
            "attribution_top_k": 4,
        },
        "scheduler": {
            "gpu_budget_ms": budget_ms,
            "policy": policy,
            "cost_per_gaussian_us": 2.0,
        },
        "densification": {
            "max_new_per_frame": max_new_per_frame,
            "strategy": "importance",
            "use_adaptive_thresholds": True,
        },
    }


# ---------------------------------------------------------------------------
# Coverage metric
# ---------------------------------------------------------------------------
def compute_coverage(
    pipeline: OnlineReconstructionPipeline,
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
    epsilon: float = 1e-4,
) -> float:
    """Compute pixel coverage: fraction of pixels with any Gaussian contribution > ε."""
    with torch.no_grad():
        cov3D = pipeline.gaussian_model.build_covariance()
        rend = rasterize_scene(
            means3D=pipeline.gaussian_model.positions,
            cov3D=cov3D,
            colors=pipeline.gaussian_model.get_colors(),
            opacities=pipeline.gaussian_model.opacities.squeeze(-1),
            extrinsics=pose,
            intrinsics=intrinsics,
            image_width=W,
            image_height=H,
            tile_size=16,
        )
        # Pixel is "covered" if transmittance < 1 - ε  (i.e., at least one
        # Gaussian contributed meaningfully)
        covered = (1.0 - rend["transmission"]) > epsilon
        return float(covered.float().mean().item())


# ---------------------------------------------------------------------------
# Compute initial PSNR (frame 0 quality check)
# ---------------------------------------------------------------------------
def compute_quality_metrics(
    pipeline: OnlineReconstructionPipeline,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Dict[str, float]:
    """Render and compute PSNR/SSIM/DepthL1 for a given frame."""
    with torch.no_grad():
        cov3D = pipeline.gaussian_model.build_covariance()
        rend = rasterize_scene(
            means3D=pipeline.gaussian_model.positions,
            cov3D=cov3D,
            colors=pipeline.gaussian_model.get_colors(),
            opacities=pipeline.gaussian_model.opacities.squeeze(-1),
            extrinsics=pose,
            intrinsics=intrinsics,
            image_width=W,
            image_height=H,
            tile_size=16,
        )
        rendered_color = rend["color"]
        rendered_depth = rend["depth"]

        # PSNR (evaluated on valid surface geometry: depth > 0)
        valid = depth > 0
        if valid.any():
            mse = ((rendered_color[valid] - rgb[valid]) ** 2).mean() + 1e-8
        else:
            mse = ((rendered_color - rgb) ** 2).mean() + 1e-8
        psnr = float(-10.0 * torch.log10(mse).item())

        # SSIM (simple global)
        c1, c2 = 0.01**2, 0.03**2
        mu1, mu2 = rendered_color.mean(), rgb.mean()
        sigma1_sq = ((rendered_color - mu1) ** 2).mean()
        sigma2_sq = ((rgb - mu2) ** 2).mean()
        sigma12 = ((rendered_color - mu1) * (rgb - mu2)).mean()
        ssim = float(
            ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2) /
             ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))).item()
        )

        # Depth L1
        valid = depth > 0
        if valid.any():
            depth_l1 = float((rendered_depth[valid] - depth[valid]).abs().mean().item())
        else:
            depth_l1 = 0.0

        return {"psnr": psnr, "ssim": ssim, "depth_l1": depth_l1}


# ---------------------------------------------------------------------------
# Single trajectory runner (simplified from Phase 10 — NO model bundle needed)
# ---------------------------------------------------------------------------
def run_headroom_trajectory(
    *,
    config_name: str,
    policy: str,
    seed: int,
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    pipeline_cfg: Dict[str, Any],
    device: str = "cpu",
    densification_enabled: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run a single trajectory and collect per-frame metrics.

    Args:
        config_name: human label for this configuration
        policy: "no_op" or "full"
        seed: random seed
        frames: pre-loaded RGB-D frames
        intrinsics: 3x3 camera matrix
        pipeline_cfg: full pipeline config dict
        device: execution device
        densification_enabled: if False, overrides max_new_per_frame=0

    Returns:
        summary: aggregate trajectory statistics
        frame_logs: per-frame metric records
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Override densification if disabled
    cfg = json.loads(json.dumps(pipeline_cfg))  # deep copy
    if not densification_enabled:
        cfg["densification"]["max_new_per_frame"] = 0

    # Hook for NO_OP: override the selector to select nothing
    is_noop = (policy == "no_op")

    pipeline = OnlineReconstructionPipeline(config=cfg, device=device)
    pipeline.initialize(
        rgb=frames[0]["rgb"],
        depth=frames[0]["depth"],
        intrinsics=intrinsics,
        pose=frames[0].get("pose", torch.eye(4, device=torch.device(device))),
    )

    n_init = pipeline.gaussian_model.num_gaussians

    # Initial quality
    init_metrics = compute_quality_metrics(
        pipeline, frames[0]["rgb"], frames[0]["depth"],
        frames[0].get("pose", torch.eye(4, device=torch.device(device))),
        intrinsics,
    )
    init_psnr = init_metrics["psnr"]

    # For NO_OP: inject a selector that always returns all-False mask
    if is_noop:
        def noop_selector(pipe, N):
            return torch.zeros(N, dtype=torch.bool, device=torch.device(device))
        pipeline._custom_selector_fn = noop_selector

    # Run trajectory
    frame_logs: List[Dict[str, Any]] = []
    prev_psnr = init_psnr

    for t in range(1, len(frames)):
        n_before = pipeline.gaussian_model.num_gaussians
        t0 = time.perf_counter()

        m = pipeline.process_frame(
            rgb=frames[t]["rgb"],
            depth=frames[t]["depth"],
            gt_pose=frames[t].get("pose", None),
        )

        t_frame = (time.perf_counter() - t0) * 1000.0
        n_after = pipeline.gaussian_model.num_gaussians
        n_new = max(0, n_after - n_before)

        # Retrieve metrics
        psnr = float(m.get("psnr_post", m.get("psnr", 0.0)))
        ssim_val = float(m.get("ssim", 0.0))
        depth_l1 = float(m.get("depth_l1", 0.0))
        n_visible = int(m.get("n_visible", 0))
        n_optimized = int(m.get("n_optimized", 0))
        opt_time_ms = float(m.get("opt_time_ms", 0.0))

        # Coverage
        cov = compute_coverage(
            pipeline,
            frames[t].get("pose", torch.eye(4, device=torch.device(device))),
            intrinsics,
        )

        # NaN/Inf check
        has_nan = bool(
            torch.isnan(pipeline.gaussian_model.positions).any().item()
            or torch.isinf(pipeline.gaussian_model.positions).any().item()
        )

        record = {
            "frame": t,
            "config": config_name,
            "policy": policy,
            "seed": seed,
            "densification_enabled": densification_enabled,
            "N_before": n_before,
            "N_new": n_new,
            "N_after": n_after,
            "N_visible": n_visible,
            "N_optimized": n_optimized,
            "PSNR": psnr,
            "SSIM": ssim_val,
            "depth_L1": depth_l1,
            "coverage": cov,
            "opt_time_ms": opt_time_ms,
            "frame_time_ms": t_frame,
            "delta_psnr_step": psnr - prev_psnr,
            "delta_psnr_cumul": psnr - init_psnr,
            "has_nan_inf": has_nan,
        }
        frame_logs.append(record)
        prev_psnr = psnr

    # Summary
    psnrs = np.array([r["PSNR"] for r in frame_logs])
    ssims = np.array([r["SSIM"] for r in frame_logs])
    depths = np.array([r["depth_L1"] for r in frame_logs])
    coverages = np.array([r["coverage"] for r in frame_logs])
    n_news = np.array([r["N_new"] for r in frame_logs])
    opt_times = np.array([r["opt_time_ms"] for r in frame_logs])

    summary = {
        "config": config_name,
        "policy": policy,
        "seed": seed,
        "densification_enabled": densification_enabled,
        "n_frames": len(frame_logs),
        "N_init": n_init,
        "N_final": int(frame_logs[-1]["N_after"]) if frame_logs else n_init,
        "N_new_total": int(np.sum(n_news)),
        "mean_new_per_frame": float(np.mean(n_news)),
        "max_new_per_frame": int(np.max(n_news)) if len(n_news) > 0 else 0,
        "init_PSNR": init_psnr,
        "final_PSNR": float(psnrs[-1]) if len(psnrs) > 0 else 0.0,
        "mean_PSNR": float(np.mean(psnrs)),
        "mean_SSIM": float(np.mean(ssims)),
        "mean_depth_L1": float(np.mean(depths)),
        "mean_coverage": float(np.mean(coverages)),
        "final_coverage": float(coverages[-1]) if len(coverages) > 0 else 0.0,
        "mean_opt_ms": float(np.mean(opt_times)),
        "catastrophic_failures": int(np.sum(np.isnan(psnrs) | np.isinf(psnrs))),
    }

    return summary, frame_logs


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------
def get_experiment_configs() -> Dict[str, Dict[str, Any]]:
    """Return named configurations for R1/R2 experiments."""
    return {
        # R1: Legacy initialization
        "legacy_init_legacy_dense": {
            "init_stride": 4,
            "scale_mode": "constant",
            "initial_scale": 0.02,
            "scale_pixel_multiplier": 1.0,
            "max_new_per_frame": 80,
            "max_gaussians": 30000,
        },
        # R1 & R2: New initialization, legacy densification
        "new_init_legacy_dense": {
            "init_stride": 2,
            "scale_mode": "depth_adaptive",
            "initial_scale": 0.02,
            "scale_pixel_multiplier": 1.0,
            "max_new_per_frame": 80,
            "max_gaussians": 150000,
        },
        # R2: New initialization, demand-driven densification
        "new_init_new_dense": {
            "init_stride": 2,
            "scale_mode": "depth_adaptive",
            "initial_scale": 0.02,
            "scale_pixel_multiplier": 1.0,
            "max_new_per_frame": 4000,  # Hard safety ceiling, not binding constraint
            "max_gaussians": 150000,
        },
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def print_frame_table(frame_logs: List[Dict[str, Any]]) -> str:
    """Format per-frame metrics as a readable table."""
    header = f"{'frame':>5} | {'N_before':>8} | {'N_new':>6} | {'N_after':>8} | {'visible':>7} | {'PSNR':>7} | {'SSIM':>6} | {'depthL1':>8} | {'coverage':>8}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in frame_logs:
        lines.append(
            f"{r['frame']:>5} | {r['N_before']:>8} | {r['N_new']:>6} | {r['N_after']:>8} | "
            f"{r['N_visible']:>7} | {r['PSNR']:>7.2f} | {r['SSIM']:>6.4f} | "
            f"{r['depth_L1']:>8.4f} | {r['coverage']:>8.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def print_summary(summary: Dict[str, Any]) -> str:
    """Format summary statistics."""
    lines = [
        f"  Config:             {summary['config']}",
        f"  Policy:             {summary['policy']}",
        f"  Densification:      {'ON' if summary['densification_enabled'] else 'OFF'}",
        f"  N_init:             {summary['N_init']}",
        f"  N_final:            {summary['N_final']}",
        f"  N_new_total:        {summary['N_new_total']}",
        f"  mean_new/frame:     {summary['mean_new_per_frame']:.1f}",
        f"  max_new/frame:      {summary['max_new_per_frame']}",
        f"  init PSNR:          {summary['init_PSNR']:.4f} dB",
        f"  final PSNR:         {summary['final_PSNR']:.4f} dB",
        f"  mean PSNR:          {summary['mean_PSNR']:.4f} dB",
        f"  mean SSIM:          {summary['mean_SSIM']:.4f}",
        f"  mean depth L1:      {summary['mean_depth_L1']:.4f}",
        f"  mean coverage:      {summary['mean_coverage']:.4f}",
        f"  final coverage:     {summary['final_coverage']:.4f}",
        f"  mean opt time:      {summary['mean_opt_ms']:.2f} ms",
        f"  NaN/Inf failures:   {summary['catastrophic_failures']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------
def run_smoke(frames, intrinsics, device, output_dir):
    """Stage: smoke — 10-frame sanity check."""
    print("\n" + "=" * 70)
    print("  STAGE: SMOKE TEST (10 frames)")
    print("=" * 70)

    smoke_frames = frames[:min(11, len(frames))]  # frame 0 + 10
    cfg_params = get_experiment_configs()["new_init_new_dense"]

    results = {}
    for policy in ["no_op", "full"]:
        cfg = build_pipeline_config(
            policy=policy, seed=42, **cfg_params
        )
        summary, frame_logs = run_headroom_trajectory(
            config_name="new_init_new_dense",
            policy=policy,
            seed=42,
            frames=smoke_frames,
            intrinsics=intrinsics,
            pipeline_cfg=cfg,
            device=device,
        )
        results[policy] = {"summary": summary, "frames": frame_logs}

        print(f"\n  --- {policy.upper()} ---")
        print(print_summary(summary))
        print(print_frame_table(frame_logs))

    # Gaussian count check
    n_init = results["full"]["summary"]["N_init"]
    expected_min = int(160 * 120 * 0.5)  # at least 50% valid depth
    count_pass = n_init > expected_min
    print(f"\n  CHECKPOINT: N_init = {n_init}  (expected >> {expected_min})")
    print(f"  RESULT: {'PASS' if count_pass else 'FAIL'}")

    # Headroom
    h = results["full"]["summary"]["mean_PSNR"] - results["no_op"]["summary"]["mean_PSNR"]
    print(f"\n  H_substrate (mean PSNR): Full - NoOp = {h:.4f} dB")

    # NaN check
    any_nan = any(
        r["has_nan_inf"]
        for p in results.values()
        for r in p["frames"]
    )
    print(f"  NaN/Inf detected: {any_nan}")

    # Save
    smoke_dir = output_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    with open(smoke_dir / "smoke_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_frames": len(smoke_frames) - 1,
            "gaussian_count_pass": count_pass,
            "N_init": n_init,
            "headroom_mean_psnr_dB": h,
            "nan_detected": any_nan,
            "no_op": results["no_op"]["summary"],
            "full": results["full"]["summary"],
        }, f, indent=2)

    # Save per-frame CSV
    import csv
    all_records = results["no_op"]["frames"] + results["full"]["frames"]
    if all_records:
        with open(smoke_dir / "smoke_frames.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
            writer.writeheader()
            writer.writerows(all_records)

    print(f"\n  Saved to: {smoke_dir}")
    return results


def run_experiment(
    *,
    stage_name: str,
    config_names: List[str],
    policies: List[str],
    n_frames: int,
    seed: int,
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    device: str,
    output_dir: Path,
    decomposition: bool = False,
):
    """Generic experiment runner for R1/R2/R3/R4 stages."""
    print(f"\n{'=' * 70}")
    print(f"  STAGE: {stage_name}")
    print(f"  Configs: {config_names}")
    print(f"  Policies: {policies}")
    print(f"  Frames: {n_frames}")
    print(f"{'=' * 70}")

    exp_frames = frames[:min(n_frames + 1, len(frames))]
    all_configs = get_experiment_configs()
    all_summaries = []
    all_frame_records = []

    for config_name in config_names:
        cfg_params = all_configs[config_name]

        conditions = []
        if decomposition:
            # Three-way decomposition
            conditions = [
                ("noop_static", "no_op", False),
                ("noop_densify", "no_op", True),
                ("full_densify", "full", True),
            ]
        else:
            for policy in policies:
                conditions.append((f"{config_name}_{policy}", policy, True))

        for label, policy, dense_on in conditions:
            full_label = f"{config_name}/{label}" if decomposition else config_name
            cfg = build_pipeline_config(
                policy=policy, seed=seed, **cfg_params
            )
            summary, frame_logs = run_headroom_trajectory(
                config_name=full_label,
                policy=policy,
                seed=seed,
                frames=exp_frames,
                intrinsics=intrinsics,
                pipeline_cfg=cfg,
                device=device,
                densification_enabled=dense_on,
            )
            all_summaries.append(summary)
            all_frame_records.extend(frame_logs)

            print(f"\n  --- {full_label} / {policy.upper()} (dense={'ON' if dense_on else 'OFF'}) ---")
            print(print_summary(summary))

    # Compute headroom for each config
    print(f"\n  {'─' * 50}")
    print(f"  HEADROOM ANALYSIS: {stage_name}")
    print(f"  {'─' * 50}")

    for config_name in config_names:
        full_results = [s for s in all_summaries if config_name in s["config"] and s["policy"] == "full" and s["densification_enabled"]]
        noop_results = [s for s in all_summaries if config_name in s["config"] and s["policy"] == "no_op" and s["densification_enabled"]]

        if full_results and noop_results:
            h_mean = full_results[0]["mean_PSNR"] - noop_results[0]["mean_PSNR"]
            h_final = full_results[0]["final_PSNR"] - noop_results[0]["final_PSNR"]
            print(f"  {config_name}:")
            print(f"    H_substrate (mean)  = {h_mean:+.4f} dB")
            print(f"    H_substrate (final) = {h_final:+.4f} dB")
            print(f"    N_init              = {full_results[0]['N_init']}")
            print(f"    N_final (Full)      = {full_results[0]['N_final']}")
            print(f"    N_final (NoOp)      = {noop_results[0]['N_final']}")

    if decomposition:
        for config_name in config_names:
            static = [s for s in all_summaries if "noop_static" in s["config"] and config_name in s["config"]]
            densify = [s for s in all_summaries if "noop_densify" in s["config"] and config_name in s["config"]]
            full = [s for s in all_summaries if "full_densify" in s["config"] and config_name in s["config"]]
            if static and densify and full:
                dq_densify = densify[0]["mean_PSNR"] - static[0]["mean_PSNR"]
                dq_opt = full[0]["mean_PSNR"] - densify[0]["mean_PSNR"]
                print(f"\n  DECOMPOSITION ({config_name}):")
                print(f"    ΔQ_densify = Q(NoOp+Densify) - Q(NoOp+Static) = {dq_densify:+.4f} dB")
                print(f"    ΔQ_opt     = Q(Full+Densify) - Q(NoOp+Densify) = {dq_opt:+.4f} dB")

    # Save results
    stage_dir = output_dir / stage_name.lower().replace(" ", "_").replace(":", "")
    stage_dir.mkdir(parents=True, exist_ok=True)

    with open(stage_dir / f"{stage_name.lower().replace(' ', '_')}_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "summaries": all_summaries,
        }, f, indent=2)

    # Save per-frame CSV
    import csv
    if all_frame_records:
        with open(stage_dir / "trajectory.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_frame_records[0].keys())
            writer.writeheader()
            writer.writerows(all_frame_records)

    print(f"\n  Saved to: {stage_dir}")
    return all_summaries, all_frame_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 12-R: Reconstruction Headroom Audit"
    )
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["smoke", "R1", "R2", "R2_decomp", "R3", "R4", "all"],
        help="Experiment stage to run"
    )
    parser.add_argument("--scene", type=str, default=SCENE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_frames", type=int, default=None,
                        help="Override number of frames (default: stage-specific)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine max frames needed
    max_frames_needed = {
        "smoke": 11,
        "R1": 31,
        "R2": 31,
        "R2_decomp": 31,
        "R3": 151,
        "R4": 151,
        "all": 151,
    }
    n_load = max_frames_needed.get(args.stage, 31)
    if args.n_frames:
        n_load = args.n_frames + 1

    print("=" * 70)
    print("  PHASE 12-R: RECONSTRUCTION HEADROOM AUDIT")
    print("=" * 70)
    print(f"  Stage:    {args.stage}")
    print(f"  Scene:    {args.scene}")
    print(f"  Seed:     {args.seed}")
    print(f"  Device:   {args.device}")
    print(f"  Loading:  {n_load} frames")
    print(f"  Output:   {out_dir}")
    print("=" * 70)

    # Load dataset
    print(f"\n>> Loading {n_load} frames from {args.scene}...")
    frames, intrinsics = load_phase10_sequence(
        scene_name=args.scene,
        n_frames=n_load,
        H=H,
        W=W,
        device=args.device,
    )
    print(f">> Loaded {len(frames)} frames successfully.")

    # Save manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "protocol_version": "2.0.0-headroom",
        "experiment": "Phase 12-R Reconstruction Headroom Audit",
        "scene": args.scene,
        "seed": args.seed,
        "device": args.device,
        "stage": args.stage,
        "n_frames_loaded": len(frames),
        "resolution": f"{W}x{H}",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Run stages
    if args.stage == "smoke" or args.stage == "all":
        run_smoke(frames, intrinsics, args.device, out_dir)

    if args.stage == "R1" or args.stage == "all":
        n = args.n_frames or 30
        run_experiment(
            stage_name="R1_initialization",
            config_names=["legacy_init_legacy_dense", "new_init_legacy_dense"],
            policies=["no_op", "full"],
            n_frames=n,
            seed=args.seed,
            frames=frames,
            intrinsics=intrinsics,
            device=args.device,
            output_dir=out_dir,
        )

    if args.stage == "R2" or args.stage == "all":
        n = args.n_frames or 30
        run_experiment(
            stage_name="R2_densification",
            config_names=["new_init_legacy_dense", "new_init_new_dense"],
            policies=["no_op", "full"],
            n_frames=n,
            seed=args.seed,
            frames=frames,
            intrinsics=intrinsics,
            device=args.device,
            output_dir=out_dir,
        )

    if args.stage == "R2_decomp" or args.stage == "all":
        n = args.n_frames or 30
        run_experiment(
            stage_name="R2_decomposition",
            config_names=["new_init_new_dense"],
            policies=["no_op", "full"],
            n_frames=n,
            seed=args.seed,
            frames=frames,
            intrinsics=intrinsics,
            device=args.device,
            output_dir=out_dir,
            decomposition=True,
        )

    if args.stage == "R3" or args.stage == "all":
        frame_lengths = [args.n_frames] if args.n_frames else [30, 75, 150]
        for n in frame_lengths:
            if len(frames) < n + 1:
                print(f"  [Skip] R3 at {n} frames: only {len(frames)} loaded.")
                continue
            run_experiment(
                stage_name=f"R3_trajectory_{n}f",
                config_names=["new_init_new_dense"],
                policies=["no_op", "full"],
                n_frames=n,
                seed=args.seed,
                frames=frames,
                intrinsics=intrinsics,
                device=args.device,
                output_dir=out_dir,
            )

    if args.stage == "R4":
        n = args.n_frames or 150
        run_experiment(
            stage_name="R4_scheduler",
            config_names=["new_init_new_dense"],
            policies=["no_op", "error_only", "ours", "full"],
            n_frames=n,
            seed=args.seed,
            frames=frames,
            intrinsics=intrinsics,
            device=args.device,
            output_dir=out_dir,
        )

    print("\n" + "=" * 70)
    print("  PHASE 12-R COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
