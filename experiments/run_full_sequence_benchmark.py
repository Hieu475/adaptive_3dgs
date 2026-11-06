#!/usr/bin/env python3
"""Full-sequence or long-horizon-prefix benchmark on TUM fr2_xyz.

Protocol & Design:
  1. Sequence scope: Evaluates the complete trajectory by default, or an explicit
     prefix for long-horizon gates. Prefix runs are not compared with full-sequence literature results.
  2. Complete Metric Triad: PSNR (dB) ↑, SSIM ↑, LPIPS (AlexNet) ↓, plus Depth L1 (m) ↓,
     Framerate (FPS) ↑, Map Size (N_final) ↓, and Peak VRAM (MB).
  3. Pareto Quality-Compute Curve:
     - OURS: Budget = 15.0 ms, real-time operating point (~8-9 FPS)
     - OURS-HQ: Budget = 30.0 ms, quality-oriented operating point (~5-6 FPS)
     - ERROR_INFLUENCE: Standard selective baseline (15.0 ms)
     - FULL: 100% upper bound (all primitives optimized each frame)
  4. Memory-Safe Streaming: Sequence is preloaded into system RAM as compact tensors
     (1.34 GB total), streaming frame-by-frame to GPU without VRAM bloat.
  5. Scientific Framing: Evaluated under 'Mapping-Only with Oracle (Ground-Truth) Pose',
     the standard protocol for isolating mapping and scheduling efficiency from tracking drift.
"""
import os
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.reproducibility import set_seed, get_hardware_info, get_git_commit
from experiments.run_phase13_frozen_benchmark import build_pipeline_config
from research.phase10_runtime import load_phase10_sequence


# Literature reference results on TUM fr2_xyz full sequence
# Note: SplaTAM and RTG-SLAM estimate tracking poses simultaneously.
LITERATURE_REFERENCE = {
    "Point-SLAM (ICCV'23)": {"psnr": 17.62, "ssim": 0.680, "lpips": 0.420, "fps": 0.2, "n_gaussians": 120000, "tracking": "Estimated (0.52cm)"},
    "MonoGS (CVPR'24)":     {"psnr": 16.17, "ssim": 0.650, "lpips": 0.460, "fps": 1.6, "n_gaussians": 180000, "tracking": "Estimated (1.55cm)"},
    "Photo-SLAM (CVPR'24)": {"psnr": 21.07, "ssim": 0.780, "lpips": 0.310, "fps": 10.0, "n_gaussians": 250000, "tracking": "Estimated (0.38cm)"},
    "SplaTAM (CVPR'24)":    {"psnr": 25.06, "ssim": 0.890, "lpips": 0.180, "fps": 0.6, "n_gaussians": 410000, "tracking": "Estimated (0.41cm)"},
    "RTG-SLAM (2024)":      {"psnr": 21.40, "ssim": 0.785, "lpips": 0.280, "fps": 20.0, "n_gaussians": 280000, "tracking": "Estimated (0.45cm)"},
}


def preload_tum_sequence_cpu(
    data_path: str,
    camera: str = "freiburg2",
    W: int = 320,
    H: int = 240,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    pose_max_dt_ms: Optional[float] = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float]:
    """Preloads downsampled sequence into compact CPU RAM tensors.

    Returns:
        rgb_cpu: (N, H, W, 3) uint8 [0..255]
        depth_cpu: (N, H, W) float16
        poses_cpu: (N, 4, 4) float32
        intrinsics_3x3: (3, 3) float32
        fx, fy, cx, cy
    """
    ds = TUMDataset(data_path, camera=camera, max_frames=None)
    if start_frame < 0 or start_frame >= len(ds):
        raise ValueError(f"start_frame={start_frame} outside sequence with {len(ds)} associated frames")
    stop_frame = len(ds) if max_frames is None else min(start_frame + max_frames, len(ds))
    selected_timestamps = np.asarray(ds.timestamps[start_frame:stop_frame], dtype=np.float64)
    gt_path = Path(data_path) / "groundtruth.txt"
    if pose_max_dt_ms is not None and gt_path.is_file():
        gt_times = np.loadtxt(gt_path, comments="#", usecols=(0,), dtype=np.float64)
        right = np.clip(np.searchsorted(gt_times, selected_timestamps), 1, len(gt_times) - 1)
        pose_dt_ms = np.minimum(
            np.abs(selected_timestamps - gt_times[right - 1]),
            np.abs(selected_timestamps - gt_times[right]),
        ) * 1000.0
        bad = np.flatnonzero(pose_dt_ms > pose_max_dt_ms)
        if len(bad):
            first_bad = [(int(start_frame + i), float(pose_dt_ms[i])) for i in bad[:8]]
            raise ValueError(
                f"Pose-sync gate failed: {len(bad)}/{len(selected_timestamps)} frames exceed "
                f"{pose_max_dt_ms:.1f} ms nearest-GT tolerance; first frame/dt_ms pairs: {first_bad}. "
                "Choose a contiguous valid trajectory window."
            )
    n = stop_frame - start_frame
    print(f">> Preloading {n} frames [{start_frame}, {stop_frame}) from '{data_path}' into CPU RAM...")
    t0 = time.perf_counter()

    rgb_cpu = torch.empty((n, H, W, 3), dtype=torch.uint8)
    depth_cpu = torch.empty((n, H, W), dtype=torch.float16)
    poses_cpu = torch.empty((n, 4, 4), dtype=torch.float32)

    scale_x = W / 640.0
    scale_y = H / 480.0
    fx, fy = ds.fx * scale_x, ds.fy * scale_y
    cx, cy = ds.cx * scale_x, ds.cy * scale_y

    intrinsics = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    for local_i, source_i in enumerate(range(start_frame, stop_frame)):
        item = ds[source_i]
        # RGB: [H_orig, W_orig, 3] float -> interpolate -> uint8
        rgb_in = item["rgb"].unsqueeze(0).permute(0, 3, 1, 2)
        rgb_scaled = torch.nn.functional.interpolate(
            rgb_in, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        rgb_cpu[local_i] = (rgb_scaled.clamp(0, 1) * 255.0).to(torch.uint8)

        # Depth: [H_orig, W_orig] float -> interpolate -> float16
        depth_in = item["depth"].unsqueeze(0).unsqueeze(0)
        depth_scaled = torch.nn.functional.interpolate(
            depth_in, size=(H, W), mode="nearest"
        ).squeeze(0).squeeze(0)
        depth_cpu[local_i] = depth_scaled.to(torch.float16)

        # TUMDataset.__getitem__ returns C2W. The mapping pipeline and gsplat
        # renderer consume W2C, so convert explicitly before storing this run.
        poses_cpu[local_i] = ds.w2c_pose(source_i)

        if (local_i + 1) % 500 == 0 or (local_i + 1) == n:
            elapsed = time.perf_counter() - t0
            print(f"   [{local_i + 1}/{n}] frames loaded ({elapsed:.1f}s, {(local_i + 1)/elapsed:.1f} fps)")

    ram_mb = (rgb_cpu.nbytes + depth_cpu.nbytes + poses_cpu.nbytes) / (1024 * 1024)
    print(f">> Preload complete: {n} frames | CPU RAM: {ram_mb:.1f} MB | Time: {time.perf_counter() - t0:.1f}s\n")
    return rgb_cpu, depth_cpu, poses_cpu, intrinsics, fx, fy, cx, cy


def build_experiment_configs(
    seed: int,
    W: int = 320,
    H: int = 240,
    device: str = "cuda",
    sh_degree: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Builds pipeline configurations for the Pareto evaluation."""
    # 1. OURS (Budget 15ms, Real-Time Operating Point)
    cfg_ours = build_pipeline_config("ours", seed=seed, budget_ms=15.0, W=W, H=H, device=device)
    cfg_ours["rendering"]["compute_lpips"] = True
    cfg_ours["gaussian"]["sh_degree"] = sh_degree

    # 2. OURS-HQ (Budget 30ms, High-Quality Operating Point)
    cfg_ours_hq = json.loads(json.dumps(cfg_ours))
    cfg_ours_hq["scheduler"]["gpu_budget_ms"] = 30.0
    cfg_ours_hq["gaussian"]["max_gaussians"] = 200000
    cfg_ours_hq["densification"]["throttle_coverage_threshold"] = 0.95
    cfg_ours_hq["densification"]["throttle_factor"] = 0.35
    cfg_ours_hq["scheduler"]["max_warmup_queue"] = 1000

    # 3. ERROR_INFLUENCE (Standard Selective Baseline, Budget 15ms)
    cfg_err = build_pipeline_config("error_influence", seed=seed, budget_ms=15.0, W=W, H=H, device=device)
    cfg_err["rendering"]["compute_lpips"] = True
    cfg_err["gaussian"]["sh_degree"] = sh_degree

    # 4. FULL (Quality Ceiling, Unconstrained)
    cfg_full = build_pipeline_config("full", seed=seed, budget_ms=15.0, W=W, H=H, device=device)
    cfg_full["rendering"]["compute_lpips"] = True
    cfg_full["gaussian"]["sh_degree"] = sh_degree

    return {
        "ours": cfg_ours,
        "ours_hq": cfg_ours_hq,
        "error_influence": cfg_err,
        "full": cfg_full,
    }


def run_full_sequence_trajectory(
    policy_name: str,
    pipeline_cfg: Dict[str, Any],
    rgb_cpu: torch.Tensor,
    depth_cpu: torch.Tensor,
    poses_cpu: torch.Tensor,
    intrinsics: torch.Tensor,
    device: str = "cuda",
    seed: int = 42,
    log_interval: int = 200,
    repro_trace: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Runs a full trajectory across the preloaded sequence with memory guards."""
    set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.empty_cache()

    if repro_trace:
        pipeline_cfg = json.loads(json.dumps(pipeline_cfg))
        pipeline_cfg.setdefault("system", {})["repro_trace"] = True
    pipeline = OnlineReconstructionPipeline(config=pipeline_cfg, device=device)
    intrinsics_gpu = intrinsics.to(dev)

    # Frame 0 Initialization
    rgb_0 = (rgb_cpu[0].float() / 255.0).to(dev)
    depth_0 = depth_cpu[0].float().to(dev)
    pose_0 = poses_cpu[0].to(dev)

    pipeline.initialize(
        rgb=rgb_0,
        depth=depth_0,
        intrinsics=intrinsics_gpu,
        pose=pose_0,
    )
    init_repro_trace = pipeline._init_repro_trace if repro_trace else []

    n_frames = rgb_cpu.shape[0]
    frame_logs: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    print(f"[{policy_name}] Starting trajectory: {n_frames} frames on {device}...")

    for t in range(1, n_frames):
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(dev)
        t_frame_start = time.perf_counter()

        # Stream single frame to GPU
        rgb_t = (rgb_cpu[t].float() / 255.0).to(dev)
        depth_t = depth_cpu[t].float().to(dev)
        pose_t = poses_cpu[t].to(dev)

        m = pipeline.process_frame(
            rgb=rgb_t,
            depth=depth_t,
            gt_pose=pose_t,
        )
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(dev)
        t_frame_ms = (time.perf_counter() - t_frame_start) * 1000.0

        rec = {
            "frame": t,
            "psnr": float(m["psnr"]),
            "ssim": float(m["ssim"]),
            "lpips": float(m.get("lpips", 0.0)),
            "depth_l1": float(m["depth_l1"]),
            "depth_l1_pre": float(m.get("depth_l1_pre", m["depth_l1"])),
            "depth_l1_post": float(m.get("depth_l1_post", m["depth_l1"])),
            "n_gaussians": int(m["n_gaussians"]),
            "n_active_gaussians": int(m.get("n_active_gaussians", m["n_gaussians"])),
            "n_optimized": int(m["n_optimized"]),
            "n_spawned": int(m.get("n_spawned", 0)),
            "n_spawned_fallback": int(m.get("n_spawned_fallback", 0)),
            "spawn_quota_this_frame": m.get("spawn_quota_this_frame"),
            "spawn_control_total": m.get("spawn_control_total"),
            "n_micro_updates": int(m.get("n_micro_updates", 0)),
            "optimizer_iteration_count": int(m.get("optimizer_iteration_count", 0)),
            "mature_update_coverage": float(m.get("mature_update_coverage", 0.0)),
            "growth_pressure": float(m.get("growth_pressure", 0.0)),
            "frame_time_ms": t_frame_ms,
            "opt_time_ms": float(m.get("opt_time_ms", 0.0)),
            "peak_vram_mb": float(m.get("peak_vram_mb", 0.0)),
        }
        if repro_trace:
            rec["repro_trace"] = m.get("repro_trace", [])
            if t == 1:
                rec["initialization_trace"] = init_repro_trace
        frame_logs.append(rec)

        # Periodic status report
        if t % log_interval == 0 or t == n_frames - 1:
            dur = time.perf_counter() - t_start
            current_fps = t / dur
            eta_sec = (n_frames - t) / max(current_fps, 0.1)
            print(f"   [{policy_name}] Frame {t:4d}/{n_frames} | "
                  f"PSNR: {rec['psnr']:5.2f} dB | SSIM: {rec['ssim']:.4f} | LPIPS: {rec['lpips']:.4f} | "
                  f"N: {rec['n_gaussians']:6,d} | FPS: {current_fps:4.1f} | ETA: {eta_sec/60:.1f}m")

        # Memory compaction & cache cleanup every 500 frames
        if t % 500 == 0 and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    total_wall_sec = time.perf_counter() - t_start
    overall_fps = (n_frames - 1) / total_wall_sec

    # Aggregate trajectory statistics
    psnrs = [r["psnr"] for r in frame_logs]
    ssims = [r["ssim"] for r in frame_logs]
    lpips_vals = [r["lpips"] for r in frame_logs if r["lpips"] > 0]
    depths = [r["depth_l1"] for r in frame_logs]
    depths_pre = [r["depth_l1_pre"] for r in frame_logs]
    vrams = [r["peak_vram_mb"] for r in frame_logs]
    opt_times = [r["opt_time_ms"] for r in frame_logs]
    frame_times = [r["frame_time_ms"] for r in frame_logs]

    summary = {
        "policy": policy_name,
        "n_frames": n_frames,
        "seed": seed,
        "mean_psnr": float(np.mean(psnrs)),
        "final_psnr": float(psnrs[-1]),
        "mean_ssim": float(np.mean(ssims)),
        "final_ssim": float(ssims[-1]),
        "mean_lpips": float(np.mean(lpips_vals)) if lpips_vals else 0.0,
        "final_lpips": float(lpips_vals[-1]) if lpips_vals else 0.0,
        "mean_depth_l1": float(np.mean(depths)),
        "final_depth_l1": float(depths[-1]),
        "mean_depth_l1_pre": float(np.mean(depths_pre)),
        "final_depth_l1_pre": float(depths_pre[-1]),
        "N_final": int(frame_logs[-1]["n_gaussians"]),
        "N_final_active": int(frame_logs[-1]["n_active_gaussians"]),
        "catastrophic_failures": int(sum(
            not (np.isfinite(r["psnr"]) and np.isfinite(r["depth_l1"]))
            for r in frame_logs)),
        "fps": float(overall_fps),
        "mean_frame_time_ms": float(np.mean(frame_times)),
        "p50_frame_time_ms": float(np.percentile(frame_times, 50)),
        "p95_frame_time_ms": float(np.percentile(frame_times, 95)),
        "p99_frame_time_ms": float(np.percentile(frame_times, 99)),
        "mean_opt_time_ms": float(np.mean(opt_times)),
        "peak_vram_mb": float(np.max(vrams)) if vrams else 0.0,
        "total_wall_sec": float(total_wall_sec),
    }
    return summary, frame_logs


def main():
    parser = argparse.ArgumentParser(description="Full-sequence or prefix benchmark on TUM fr2_xyz")
    parser.add_argument("--scene_path", type=str, default="datasets/TUM/rgbd_dataset_freiburg2_xyz")
    parser.add_argument("--replica_scene", type=str, default=None,
                        help="Replica scene name (e.g. office0); uses the Phase 10 loader and GT extrinsics")
    parser.add_argument("--camera", type=str, default="freiburg2")
    parser.add_argument("--max_frames", type=int, default=None, help="Default runs all 3669 frames")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="First source frame for TUM window runs (default: 0)")
    parser.add_argument("--pose_max_dt_ms", type=float, default=50.0,
                        help="Reject any TUM frame farther than this from nearest ground-truth pose; set <=0 to disable")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sh_degree", type=int, default=1, help="Spherical Harmonics degree (0 for diffuse, 1 for view-dependent color)")
    parser.add_argument("--ssim_implementation", choices=("direct", "separable"), default="direct",
                        help="SSIM filter implementation used by all selected policies")
    parser.add_argument("--policies", type=str, nargs="+", default=["ours", "ours_hq", "error_influence", "full"])
    parser.add_argument("--out_dir", type=str, default="results/full_sequence_benchmark")
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--strict_determinism", action="store_true",
                        help="Enable PyTorch deterministic algorithms for repeatability diagnostics; unsupported kernels may raise.")
    parser.add_argument("--repro_trace", action="store_true",
                        help="Hash inputs and pipeline state at each frame stage; diagnostic runs only (adds synchronization overhead).")
    args = parser.parse_args()

    if args.strict_determinism:
        if args.device.startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise RuntimeError("For strict CUDA determinism, set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python.")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if args.device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.70, 0)
        except (RuntimeError, ValueError):
            pass

    out_path = REPO_ROOT / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240
    if args.replica_scene:
        scene_id = f"replica_{args.replica_scene}"
        scene_label = f"Replica {args.replica_scene}"
    else:
        tum_sequence = Path(args.scene_path).name.removeprefix("rgbd_dataset_").replace("freiburg", "fr")
        scene_id = f"tum_{tum_sequence}"
        scene_label = f"TUM {tum_sequence}"

    print("=" * 90)
    print(f"  LONG-HORIZON / FULL-SEQUENCE BENCHMARK ON {scene_label}")
    print("=" * 90)
    print(f"  Dataset:      {scene_label if args.replica_scene else args.scene_path}")
    print(f"  Camera:       {args.camera}")
    print(f"  Resolution:   {W}x{H}")
    print(f"  Max Frames:   {args.max_frames if args.max_frames else 'FULL SEQUENCE (~3669)'}")
    print(f"  SH Degree:    {args.sh_degree} ({'diffuse' if args.sh_degree == 0 else 'view-dependent SH'})")
    print(f"  SSIM Filter:  {args.ssim_implementation}")
    print(f"  Policies:     {args.policies}")
    print(f"  Seed:         {args.seed}")
    print(f"  Device:       {args.device}")
    print(f"  Output Dir:   {out_path}")
    print(f"  Framing:      Mapping-Only under Oracle (Ground-Truth) Pose")
    print("=" * 90)

    # 1. Preload sequence into CPU RAM
    if args.replica_scene:
        n_load = args.max_frames or 500
        frames, intrinsics = load_phase10_sequence(
            scene_name=f"replica_{args.replica_scene}", n_frames=n_load,
            H=H, W=W, device="cpu",
        )
        if not frames:
            raise RuntimeError(f"No frames loaded for Replica {args.replica_scene}")
        rgb_cpu = torch.stack([(f["rgb"].clamp(0, 1) * 255).round().to(torch.uint8) for f in frames])
        depth_cpu = torch.stack([f["depth"].to(torch.float16) for f in frames])
        poses_cpu = torch.stack([f["pose"].to(torch.float32) for f in frames])
        del frames
        print(f">> Preloaded {len(rgb_cpu)} Replica frames in CPU RAM using Phase 10 loader (GT extrinsics).")
    else:
        rgb_cpu, depth_cpu, poses_cpu, intrinsics, fx, fy, cx, cy = preload_tum_sequence_cpu(
            data_path=str(REPO_ROOT / args.scene_path),
            camera=args.camera,
            W=W,
            H=H,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            pose_max_dt_ms=args.pose_max_dt_ms if args.pose_max_dt_ms > 0 else None,
        )
    actual_frames = rgb_cpu.shape[0]

    # 2. Build configs
    configs = build_experiment_configs(seed=args.seed, W=W, H=H, device=args.device, sh_degree=args.sh_degree)
    selected_configs = {pol: configs[pol] for pol in args.policies if pol in configs}
    for cfg in selected_configs.values():
        cfg.setdefault("losses", {})["ssim_implementation"] = args.ssim_implementation
    config_blob = json.dumps(selected_configs, sort_keys=True, separators=(",", ":"), default=str)
    config_sha256 = hashlib.sha256(config_blob.encode("utf-8")).hexdigest()
    renderer_backends = sorted({
        cfg.get("rendering", {}).get("backend", "unspecified")
        for cfg in selected_configs.values()
    })

    all_summaries = []
    all_logs = {}

    for pol in args.policies:
        if pol not in configs:
            print(f"WARNING: Unknown policy '{pol}', skipping.")
            continue

        cfg = configs[pol]
        print(f"\n>> Executing Policy: '{pol}' on {actual_frames} frames...")
        summary, logs = run_full_sequence_trajectory(
            policy_name=pol,
            pipeline_cfg=cfg,
            rgb_cpu=rgb_cpu,
            depth_cpu=depth_cpu,
            poses_cpu=poses_cpu,
            intrinsics=intrinsics,
            device=args.device,
            seed=args.seed,
            log_interval=args.log_interval,
            repro_trace=args.repro_trace,
        )
        all_summaries.append(summary)
        all_logs[pol] = logs

        # Save individual policy frame log to CSV
        df_logs = pd.DataFrame(logs)
        df_logs.to_csv(out_path / f"trajectory_log_{pol}.csv", index=False)
        if args.repro_trace:
            with open(out_path / f"repro_trace_{pol}.json", "w") as f:
                json.dump({
                    "policy": pol,
                    "seed": args.seed,
                    "scene": scene_id,
                    "initialization_stages": logs[0].get("initialization_trace", []) if logs else [],
                    "frames": [{"frame": row["frame"], "stages": row["repro_trace"]} for row in logs],
                }, f, indent=2)

        print(f"\n>> Policy '{pol}' Complete:")
        print(f"   Mean PSNR:   {summary['mean_psnr']:.2f} dB (Final: {summary['final_psnr']:.2f} dB)")
        print(f"   Mean SSIM:   {summary['mean_ssim']:.4f} (Final: {summary['final_ssim']:.4f})")
        print(f"   Mean LPIPS:  {summary['mean_lpips']:.4f} (Final: {summary['final_lpips']:.4f})")
        print(f"   Framerate:   {summary['fps']:.1f} FPS (Wall time: {summary['total_wall_sec']/60:.1f} min)")
        print(f"   Final Map:   {summary['N_final']:,d} Gaussians | Peak VRAM: {summary['peak_vram_mb']:.1f} MB\n")

    # 3. Generate Comparison & Pareto Report
    full_trajectory = (args.max_frames is None and args.start_frame == 0
                       and not args.replica_scene)
    protocol_version = ("full-sequence-v3-w2c-final-map-quality-synced-latency" if full_trajectory else
                        "long-horizon-window-v3-w2c-pose-gated-final-map-quality-synced-latency")
    source_start = args.start_frame if not args.replica_scene else 0
    source_end = source_start + actual_frames
    scope_line = (f"**Sequence Length**: {actual_frames} frames (complete trajectory).  "
                  if full_trajectory else
                  f"**Sequence Window**: source frames [{source_start}, {source_end}); {actual_frames} frames; not a full-trajectory literature comparison.  ")
    evaluation_scope = (
        f"a source-frame window [{source_start}, {source_end})"
        if source_start > 0 else f"a {actual_frames}-frame sequence prefix"
    )
    evaluation_heading = (
        "## 1. Full-Trajectory Evaluation Across Policies" if full_trajectory else
        "## 1. Long-Horizon Window Evaluation Across Policies" if source_start > 0 else
        "## 1. Long-Horizon Prefix Evaluation Across Policies"
    )
    lines = [
        (f"# Full-Sequence Benchmark Report ({scene_label})" if full_trajectory else
         f"# Long-Horizon {'Window' if source_start > 0 else 'Prefix'} Benchmark Report ({scene_label})"),
        "",
        scope_line,
        f"**Resolution**: {W} $\\times$ {H} | **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU  ",
        "**Framing**: **Mapping-Only with Oracle (Ground-Truth) Pose** (isolates mapping & budget scheduling from tracking drift)  ",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "",
        evaluation_heading,
        "",
        f"**Quality metrics:** PSNR, SSIM, and depth L1 use the final map after optimization, spawn/prune, and compaction. `depth_l1_pre` is retained separately. Protocol: `{protocol_version}`.",
        "",
        "| Method / Policy | PSNR mean/final | SSIM | LPIPS | Depth L1 ↓ | FPS | p50/p95/p99 frame ms | Final Map Size | Mean Opt ms | Peak VRAM |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|",
    ]

    for s in all_summaries:
        lines.append(
            f"| **{s['policy'].upper()}** | {s['mean_psnr']:.2f} / {s['final_psnr']:.2f} | "
            f"{s['mean_ssim']:.4f} | {s['mean_lpips']:.4f} | {s['mean_depth_l1']:.5f} | {s['fps']:.2f} | "
            f"{s['p50_frame_time_ms']:.1f}/{s['p95_frame_time_ms']:.1f}/{s['p99_frame_time_ms']:.1f} | "
            f"{s['N_final']:,d} | {s['mean_opt_time_ms']:.1f} | {s['peak_vram_mb']:.1f} MB |"
        )

    lines.extend([
        "",
        "*(Format: Mean / Final for PSNR. LPIPS evaluated with AlexNet backbone, lower is better)*",
        "",
    ])
    if full_trajectory and scene_id == "tum_fr2_xyz":
        lines.extend([
            "",
            "## 2. Literature Comparison (Full `tum_fr2_xyz` Trajectory)",
            "",
            "> [!NOTE]",
            "> Baselines marked 'Estimated' jointly estimate 6-DoF camera poses; this benchmark uses ground-truth poses and measures mapping only.",
            "",
            "| Method | Tracking Mode | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ | Map Size ↓ |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for m_name, m_data in LITERATURE_REFERENCE.items():
            lines.append(
                f"| {m_name} | {m_data['tracking']} | {m_data['psnr']:.2f} | {m_data['ssim']:.3f} | "
                f"{m_data['lpips']:.3f} | {m_data['fps']:.1f} | {m_data['n_gaussians']:,d} |"
            )
        lines.extend(["", "## 3. Full-Trajectory System Observations", "",
                      "Full-trajectory results are comparable only when dataset, pose, resolution, and metric protocols match.", ""])
    else:
        lines.extend([
            "",
            "## Scope and interpretation",
            "",
            f"This run evaluates {evaluation_scope} under ground-truth poses. It does not establish full-sequence performance or real-time operation. The table reports measured synchronized frame latency, map size, and reconstruction metrics for the tested policies.",
            "",
        ])

    report_str = "\n".join(lines)
    report_filename = "full_sequence_report.md" if full_trajectory else "long_horizon_report.md"
    results_filename = "full_sequence_results.json" if full_trajectory else "long_horizon_results.json"
    with open(out_path / report_filename, "w") as f:
        f.write(report_str)

    # Save full results JSON
    json_out = {
        "metadata": {
            "scene": scene_id,
            "n_frames": actual_frames,
            "max_frames_requested": args.max_frames,
            "source_frame_start": source_start,
            "source_frame_end_exclusive": source_end,
            "pose_sync_max_dt_ms": args.pose_max_dt_ms if not args.replica_scene and args.pose_max_dt_ms > 0 else None,
            "pose_sync_gate": "every selected TUM RGB timestamp must be within the configured tolerance of a ground-truth pose" if not args.replica_scene and args.pose_max_dt_ms > 0 else "not applied",
            "seed": args.seed,
            "policies": args.policies,
            "sh_degree": args.sh_degree,
            "ssim_implementation": args.ssim_implementation,
            "device": args.device,
            "framing": "Mapping-Only with Oracle (Ground-Truth) Pose",
            "pose_convention": "TUM ground-truth C2W inverted to W2C; Replica ground-truth extrinsics passed through as stored.",
            "protocol_version": protocol_version,
            "quality_metric_definition": "PSNR, SSIM, and depth L1 use the final map after optimization, spawn/prune, and compaction; depth_l1_pre retains the pre-update residual.",
            "per_frame_diagnostics": ["n_spawned", "n_spawned_fallback", "spawn_quota_this_frame", "spawn_control_total", "n_micro_updates", "optimizer_iteration_count", "mature_update_coverage", "growth_pressure"],
            "config_sha256": config_sha256,
            "config_sha256_scope": "Selected per-policy pipeline configs serialized as sorted compact JSON.",
            "renderer_backends": renderer_backends,
            "replicate_unit": "one complete trajectory run for the configured seed and policy",
            "statistical_unit": "per-seed trajectory summary; frames within a trajectory are repeated measurements, not independent samples",
            "seed_repeatability_scope": "seed initializes RNGs but does not guarantee bitwise-identical CUDA behavior for gsplat atomic gradient accumulation",
            "strict_determinism_scope": "PyTorch deterministic algorithms and cuBLAS settings do not constrain third-party gsplat CUDA atomic accumulation.",
            "synchronized_frame_timing": True,
            "strict_determinism_requested": bool(args.strict_determinism),
            "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "dataset_path": str(REPO_ROOT / "datasets" / "Replica" / args.replica_scene) if args.replica_scene else str(REPO_ROOT / args.scene_path),
            "git_commit": get_git_commit(),
            "hardware": get_hardware_info(),
        },
        "policies": all_summaries,
        "literature_baselines": LITERATURE_REFERENCE if full_trajectory and scene_id == "tum_fr2_xyz" else None,
    }
    with open(out_path / results_filename, "w") as f:
        json.dump(json_out, f, indent=2)

    print("\n" + "=" * 90)
    print("  BENCHMARK COMPLETE")
    print("=" * 90)
    print(report_str)
    print(f">> Report: {out_path / report_filename}")
    print(f">> Data:   {out_path / results_filename}")


if __name__ == "__main__":
    main()
