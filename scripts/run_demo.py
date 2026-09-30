#!/usr/bin/env python3
"""Interactive Demo for Adaptive 3D Gaussian Splatting (Online RGB-D).

Features:
  - Runs online 3DGS reconstruction frame-by-frame with budget scheduling
  - Fast CUDA backend (gsplat) with OS Memory Guard (no desktop stutter)
  - Exports:
      1. Side-by-side visual comparison frames (GT vs 3DGS vs Error vs Depth)
      2. Animated progress GIF showing the live reconstruction
      3. 3D point cloud & 3DGS PLY files for 3D viewers (MeshLab, SuperSplat, Blender)
      4. Quantitative metrics plot (PSNR, Gaussians, Latency) and summary JSON
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import cv2

# Project root setup
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pipeline import OnlineReconstructionPipeline
from research.rasterizer import render as rasterize_scene
from research.phase10_runtime import load_phase10_sequence
from experiments.run_phase13_frozen_benchmark import build_pipeline_config


def export_ply_colored_points(filepath: Path, positions: np.ndarray, colors: np.ndarray):
    """Export standard binary PLY colored point cloud.
    
    Compatible with MeshLab, CloudCompare, Blender, and any standard 3D viewer.
    """
    N = len(positions)
    colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    
    vertex_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    vertex_data = np.empty(N, dtype=vertex_dtype)
    vertex_data['x'] = positions[:, 0]
    vertex_data['y'] = positions[:, 1]
    vertex_data['z'] = positions[:, 2]
    vertex_data['red'] = colors_uint8[:, 0]
    vertex_data['green'] = colors_uint8[:, 1]
    vertex_data['blue'] = colors_uint8[:, 2]
    
    with open(filepath, "wb") as f:
        f.write(header.encode('ascii'))
        vertex_data.tofile(f)


def export_ply_3dgs(
    filepath: Path,
    positions: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    opacities: np.ndarray,
    colors_dc: np.ndarray,
):
    """Export standard 3D Gaussian Splatting binary PLY.
    
    Compatible with SuperSplat, Inria 3DGS viewer, and web splat viewers.
    """
    N = len(positions)
    # Standard Inria 3DGS SH DC coefficient: C0 = 0.28209479177387814
    # Inria rasterizer evaluates: RGB = 0.5 + C0 * f_dc => f_dc = (RGB - 0.5) / C0
    C0 = 0.28209479177387814
    c_flat = colors_dc.reshape(N, 3)
    # Check if colors_dc is RGB in [0, 1] or raw logit
    if np.all(c_flat >= -0.05) and np.all(c_flat <= 1.05):
        f_dc = (np.clip(c_flat, 0.0, 1.0) - 0.5) / C0
    else:
        # If already logit, convert logit -> sigmoid -> SH DC
        rgb_from_logit = 1.0 / (1.0 + np.exp(-np.clip(c_flat, -15.0, 15.0)))
        f_dc = (rgb_from_logit - 0.5) / C0

    # Inria standard: opacity is stored as logit = log(op / (1 - op))
    op_flat = opacities.squeeze()
    if np.all(op_flat >= -0.01) and np.all(op_flat <= 1.01):
        op_clipped = np.clip(op_flat, 1e-4, 1.0 - 1e-4)
        opacity_logit = np.log(op_clipped / (1.0 - op_clipped))
    else:
        opacity_logit = op_flat

    # Normalize quaternions (w, x, y, z)
    norm = np.linalg.norm(rotations, axis=-1, keepdims=True) + 1e-8
    q = rotations / norm

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )
    
    dtype_fields = [
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
        ('f_dc_0', '<f4'), ('f_dc_1', '<f4'), ('f_dc_2', '<f4'),
        ('opacity', '<f4'),
        ('scale_0', '<f4'), ('scale_1', '<f4'), ('scale_2', '<f4'),
        ('rot_0', '<f4'), ('rot_1', '<f4'), ('rot_2', '<f4'), ('rot_3', '<f4')
    ]
    
    vertex_dtype = np.dtype(dtype_fields)
    vertex_data = np.empty(N, dtype=vertex_dtype)
    
    vertex_data['x'] = positions[:, 0]
    vertex_data['y'] = positions[:, 1]
    vertex_data['z'] = positions[:, 2]
    vertex_data['nx'] = 0.0
    vertex_data['ny'] = 0.0
    vertex_data['nz'] = 0.0
    vertex_data['f_dc_0'] = f_dc[:, 0]
    vertex_data['f_dc_1'] = f_dc[:, 1]
    vertex_data['f_dc_2'] = f_dc[:, 2]
    vertex_data['opacity'] = opacity_logit
    vertex_data['scale_0'] = np.log(np.clip(scales[:, 0], 1e-7, None))
    vertex_data['scale_1'] = np.log(np.clip(scales[:, 1], 1e-7, None))
    vertex_data['scale_2'] = np.log(np.clip(scales[:, 2], 1e-7, None))
    vertex_data['rot_0'] = q[:, 0]
    vertex_data['rot_1'] = q[:, 1]
    vertex_data['rot_2'] = q[:, 2]
    vertex_data['rot_3'] = q[:, 3]
    
    with open(filepath, "wb") as f:
        f.write(header.encode('ascii'))
        vertex_data.tofile(f)


def create_side_by_side_panel(
    gt_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    gt_depth: np.ndarray,
    rendered_depth: np.ndarray,
    frame_idx: int,
    psnr: float,
    ssim: float,
    n_gaussians: int,
    fps: float,
) -> np.ndarray:
    """Create a 4-view comparison panel: [GT RGB | Rendered 3DGS | Error Heatmap | Rendered Depth]."""
    H, W, _ = gt_rgb.shape
    
    # 1. Error Heatmap
    err = np.abs(rendered_rgb - gt_rgb).mean(axis=-1)
    err_norm = np.clip(err / 0.3, 0.0, 1.0)
    err_heatmap = cv2.applyColorMap((err_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    err_heatmap = cv2.cvtColor(err_heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # 2. Depth visualization
    d_max = max(float(gt_depth.max()), 3.0)
    depth_vis = np.clip(rendered_depth / d_max, 0.0, 1.0)
    depth_vis = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Concatenate 4 views horizontally: GT | 3DGS | Error | Depth
    panel = np.concatenate([gt_rgb, rendered_rgb, err_heatmap, depth_vis], axis=1)
    
    # Convert to uint8 BGR for OpenCV drawing
    panel_bgr = (np.clip(panel, 0.0, 1.0) * 255.0).astype(np.uint8)
    panel_bgr = cv2.cvtColor(panel_bgr, cv2.COLOR_RGB2BGR)
    
    # Add top banner
    banner_height = 36
    total_W = panel_bgr.shape[1]
    banner = np.zeros((banner_height, total_W, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    
    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(banner, f"Frame {frame_idx:03d} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.3f} | Gaussians: {n_gaussians:,} | Speed: {fps:.1f} FPS",
                (12, 24), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Sub-view labels
    sub_w = W
    labels = ["1. Ground Truth RGB", "2. Rendered 3DGS", "3. Error (|Render - GT|)", "4. Rendered Depth"]
    full_img = np.vstack([banner, panel_bgr])
    
    for idx, lbl in enumerate(labels):
        x = idx * sub_w + 8
        y = banner_height + 20
        cv2.putText(full_img, lbl, (x, y), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
    return cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description="Run Adaptive 3DGS Online Reconstruction Demo")
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz",
                        help="TUM or Replica sequence name (e.g. replica_room2, replica_room0, tum_fr1_room)")
    parser.add_argument("--n_frames", type=int, default=30, help="Number of frames to process")
    parser.add_argument("--policy", type=str, default="ours",
                        choices=["ours", "error_only", "error_influence", "no_op", "full"],
                        help="Optimization / scheduling policy")
    parser.add_argument("--full_room", action="store_true", help="Reconstruct entire room across all frames with balanced high-density Gaussians")
    parser.add_argument("--high_res", action="store_true", help="Run at 640x480 resolution (4x more Gaussians)")
    parser.add_argument("--native", action="store_true", help="Run at true native resolution (640x480 for TUM, 1200x680 for Replica ~800k Gaussians)")
    parser.add_argument("--dense", action="store_true", help="Aggressive densification mode (disables throttling, allows rapid Gaussian spawning)")
    parser.add_argument("--max_gaussians", type=int, default=None, help="Maximum number of Gaussians allowed (e.g. 500000 or 1500000)")
    parser.add_argument("--steps", type=int, default=5, help="Number of micro-steps per frame (default: 5, use 10-15 for sharper reconstruction)")
    parser.add_argument("--stride", type=int, default=None, help="Pixel sampling stride for Gaussians (1 = dense/sharp, 2 = fast)")
    parser.add_argument("--budget_ms", type=float, default=15.0, help="Per-frame compute budget (ms)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save demo artifacts (defaults to results/demo_output/<scene>)")
    parser.add_argument("--save_gif", action="store_true", default=True, help="Save animated GIF of reconstruction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Resolution determination
    if args.full_room or args.native:
        if args.scene.startswith("replica_"):
            W, H = 1200, 680
        else:
            W, H = 640, 480
    elif args.high_res:
        W, H = 640, 480
    else:
        W, H = 320, 240

    if args.full_room and args.n_frames == 30:
        # User left default n_frames=30, expand to full sequence
        n_frames = 100 if args.scene.startswith("replica_") else 100
    else:
        n_frames = args.n_frames

    # Stride calculation: use stride 3 for full_room Replica, 2 for TUM so initial frame doesn't hoard budget
    if args.stride is not None:
        stride = args.stride
    elif args.full_room:
        stride = 2  # Dense 2px grid for thin structures, chair legs, and sharp edges
    elif args.native:
        stride = 2
    elif args.high_res or args.dense:
        stride = 1
    else:
        stride = 2

    # Max Gaussians default
    if args.max_gaussians is not None:
        max_gaussians = args.max_gaussians
    elif args.full_room:
        max_gaussians = 2500000
    elif args.native:
        max_gaussians = 1200000 if args.scene.startswith("replica_") else 500000
    elif args.high_res or args.dense:
        max_gaussians = 500000
    else:
        max_gaussians = 150000

    if args.out_dir is None:
        suffix_parts = []
        if args.full_room:
            suffix_parts.append("full_room")
        elif args.native:
            suffix_parts.append("native")
        elif args.high_res:
            suffix_parts.append("highres")
        if args.dense:
            suffix_parts.append("dense")
        suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
        out_dir = REPO_ROOT / "results" / "demo_output" / f"{args.scene}{suffix}"
    else:
        out_dir = REPO_ROOT / args.out_dir

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      ADAPTIVE 3D GAUSSIAN SPLATTING — ONLINE RECONSTRUCTION DEMO")
    print("=" * 80)
    print(f"  Scene:           {args.scene}")
    print(f"  Resolution:      {W}x{H}")
    print(f"  Max Gaussians:   {max_gaussians:,}")
    print(f"  Init Stride:     {stride} px")
    print(f"  Dense Mode:      {'ENABLED (Unthrottled expansion)' if args.dense else 'Standard'}")
    print(f"  Total Frames:    {n_frames}")
    print(f"  Policy:          {args.policy.upper()}")
    print(f"  Micro-steps:     {args.steps} steps/frame")
    print(f"  Compute Budget:  {args.budget_ms:.1f} ms/frame")
    print(f"  Device:          {args.device}")
    print(f"  Output Dir:      {out_dir}")
    print("=" * 80)

    # 1. Load Dataset
    print(f"\n[1/5] Loading RGB-D frames for scene '{args.scene}' at {W}x{H}...")
    frames, intrinsics = load_phase10_sequence(
        scene_name=args.scene,
        n_frames=n_frames,
        H=H,
        W=W,
        device=args.device,
    )
    print(f"      Loaded {len(frames)} frames successfully.")

    # 2. Initialize Pipeline
    print(f"\n[2/5] Initializing pipeline with policy '{args.policy}'...")
    cfg = build_pipeline_config(
        policy=args.policy,
        seed=args.seed,
        budget_ms=args.budget_ms,
        W=W,
        H=H,
        device=args.device,
    )
    # Apply density and sharpness overrides
    cfg["training"]["n_micro_steps"] = args.steps
    cfg["gaussian"]["init_stride"] = stride
    cfg["gaussian"]["max_gaussians"] = max_gaussians
    cfg["gaussian"]["scale_pixel_multiplier"] = 1.05  # Sharp, crisp splats avoiding blurry/fuzzy blobs
    cfg["gaussian"]["initial_opacity"] = 0.85        # Solid, opaque primitives from creation

    if args.dense or args.full_room:
        cfg["densification"]["enable_coverage_throttling"] = False
        cfg["densification"]["enable_persistent_prune"] = False  # Keep full scene map intact throughout video
        cfg["densification"]["max_new_per_frame"] = int(max_gaussians / max(n_frames, 1)) if args.full_room else 25000
        cfg["scheduler"]["max_warmup_queue"] = 999999
        cfg["scheduler"]["enable_warmup"] = False

    pipeline = OnlineReconstructionPipeline(config=cfg, device=args.device)

    if args.dense or args.full_room:
        pipeline.scheduler.cost_densify_us = 0.005  # unconstrain densification cap from budget cap

    # Init with frame 0
    t0 = time.perf_counter()
    pipeline.initialize(
        rgb=frames[0]["rgb"],
        depth=frames[0]["depth"],
        intrinsics=intrinsics,
        pose=frames[0].get("pose", torch.eye(4, device=torch.device(args.device))),
    )
    init_time = (time.perf_counter() - t0) * 1000.0
    print(f"      Pipeline initialized in {init_time:.1f} ms: {pipeline.gaussian_model.num_gaussians:,} initial Gaussians.")

    # 3. Online Processing Loop
    print(f"\n[3/5] Streaming online reconstruction across {len(frames)} frames...")
    print(f"      {'Frame':>5} | {'PSNR (dB)':>9} | {'SSIM':>6} | {'Gaussians':>10} | {'Opt/Frame':>9} | {'FPS':>6} | {'Wall (ms)':>9}")
    print("      " + "-" * 70)

    panel_images = []
    logs = []

    for t in range(len(frames)):
        t_start = time.perf_counter()
        current_pose = frames[t].get("pose", None)
        
        if t == 0:
            # Render frame 0
            with torch.no_grad():
                cov3D = pipeline.gaussian_model.build_covariance()
                colors = pipeline.gaussian_model.get_colors()
                out_render = rasterize_scene(
                    means3D=pipeline.gaussian_model.positions,
                    cov3D=cov3D,
                    colors=colors,
                    opacities=pipeline.gaussian_model.opacities.squeeze(-1),
                    extrinsics=current_pose,
                    intrinsics=intrinsics,
                    image_width=W,
                    image_height=H,
                    backend="gsplat" if args.device == "cuda" else "reference",
                )
                rendered_rgb = out_render["color"].detach().cpu().numpy()
                rendered_depth = out_render["depth"].detach().cpu().numpy()
                
            gt_rgb = frames[0]["rgb"].detach().cpu().numpy()
            gt_depth = frames[0]["depth"].detach().cpu().numpy()
            mse = np.mean((rendered_rgb - gt_rgb) ** 2) + 1e-8
            psnr = -10.0 * np.log10(mse)
            ssim = 0.5  # initial estimate
            fps = 1000.0 / max(init_time, 1.0)
            n_opt = 0
            wall_ms = init_time
        else:
            # Dynamically balance densification budget across remaining frames
            if args.dense or args.full_room:
                remaining_frames = max(len(frames) - t, 1)
                remaining_budget = max(0, max_gaussians - pipeline.gaussian_model.num_gaussians)
                even_share = int(remaining_budget / remaining_frames)
                dynamic_limit = max(3000, int(even_share * 1.15))
                pipeline.config["densification"]["max_new_per_frame"] = min(dynamic_limit, remaining_budget)

            # Process incoming streaming frame
            metrics = pipeline.process_frame(
                rgb=frames[t]["rgb"],
                depth=frames[t]["depth"],
                gt_pose=current_pose,
            )
            wall_ms = (time.perf_counter() - t_start) * 1000.0
            
            # Re-render updated Gaussian map for visual demo panel
            with torch.no_grad():
                cov3D = pipeline.gaussian_model.build_covariance()
                colors = pipeline.gaussian_model.get_colors()
                out_render = rasterize_scene(
                    means3D=pipeline.gaussian_model.positions,
                    cov3D=cov3D,
                    colors=colors,
                    opacities=pipeline.gaussian_model.opacities.squeeze(-1),
                    extrinsics=pipeline.current_pose,
                    intrinsics=intrinsics,
                    image_width=W,
                    image_height=H,
                    backend="gsplat" if args.device == "cuda" else "reference",
                )
                rendered_rgb = out_render["color"].detach().cpu().numpy()
                rendered_depth = out_render["depth"].detach().cpu().numpy()

            gt_rgb = frames[t]["rgb"].detach().cpu().numpy()
            gt_depth = frames[t]["depth"].detach().cpu().numpy()
            psnr = float(metrics["psnr"])
            ssim = float(metrics["ssim"])
            fps = float(metrics["fps"])
            n_opt = int(metrics["n_optimized"])

        n_gaussians = pipeline.gaussian_model.num_gaussians

        # Print progress row
        print(f"      {t:5d} | {psnr:9.2f} | {ssim:6.3f} | {n_gaussians:10,d} | {n_opt:9d} | {fps:6.1f} | {wall_ms:9.1f}")

        logs.append({
            "frame": t,
            "psnr": psnr,
            "ssim": ssim,
            "n_gaussians": n_gaussians,
            "n_optimized": n_opt,
            "fps": fps,
            "wall_ms": wall_ms,
        })

        # Generate 4-view comparison panel
        panel = create_side_by_side_panel(
            gt_rgb=gt_rgb,
            rendered_rgb=rendered_rgb,
            gt_depth=gt_depth,
            rendered_depth=rendered_depth,
            frame_idx=t,
            psnr=psnr,
            ssim=ssim,
            n_gaussians=n_gaussians,
            fps=fps,
        )

        # Save individual panel images for key frames
        if t % 5 == 0 or t == len(frames) - 1:
            frame_path = frames_dir / f"frame_{t:03d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        panel_images.append(panel)

    # 4. Generate Animated GIF & Video
    print(f"\n[4/5] Compiling visual demo animations and 3D models...")
    if args.save_gif and len(panel_images) > 0:
        import imageio
        gif_path = out_dir / "demo_recon_progress.gif"
        # Downsample slightly for fast GIF saving
        downsampled = [cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)) for img in panel_images]
        imageio.mimsave(str(gif_path), downsampled, fps=5, loop=0)
        print(f"      -> Animated GIF saved: {gif_path}")

    # 5. Export 3D Point Cloud & Gaussians
    print(f"\n[5/5] Exporting 3D Gaussian maps to PLY...")
    with torch.no_grad():
        positions = pipeline.gaussian_model.positions.detach().cpu().numpy()
        scales = pipeline.gaussian_model.scales.detach().cpu().numpy()
        rotations = pipeline.gaussian_model._rotation.detach().cpu().numpy()
        opacities = pipeline.gaussian_model.opacities.detach().cpu().numpy()
        colors = pipeline.gaussian_model.get_colors().detach().cpu().numpy()
        dc = pipeline.gaussian_model._features_dc.detach().cpu().numpy()

    pcd_ply_path = out_dir / "reconstructed_point_cloud.ply"
    export_ply_colored_points(pcd_ply_path, positions, colors)
    print(f"      -> Standard Point Cloud PLY: {pcd_ply_path} ({len(positions):,} points)")

    gs_ply_path = out_dir / "reconstructed_3dgs.ply"
    export_ply_3dgs(gs_ply_path, positions, scales, rotations, opacities, colors)
    print(f"      -> Full 3DGS Splat PLY:      {gs_ply_path} ({len(positions):,} splats)")

    # Save summary report & metrics JSON
    mean_psnr = float(np.mean([l["psnr"] for l in logs]))
    mean_ssim = float(np.mean([l["ssim"] for l in logs]))
    mean_fps = float(np.mean([l["fps"] for l in logs[1:]])) if len(logs) > 1 else logs[0]["fps"]
    final_gaussians = logs[-1]["n_gaussians"]

    summary = {
        "scene": args.scene,
        "policy": args.policy,
        "budget_ms": args.budget_ms,
        "n_frames": len(logs),
        "mean_psnr_db": round(mean_psnr, 2),
        "final_psnr_db": round(logs[-1]["psnr"], 2),
        "mean_ssim": round(mean_ssim, 4),
        "final_ssim": round(logs[-1]["ssim"], 4),
        "mean_fps": round(mean_fps, 1),
        "final_gaussians": final_gaussians,
        "ply_point_cloud": str(pcd_ply_path),
        "ply_3dgs": str(gs_ply_path),
    }

    summary_path = out_dir / "demo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      -> Summary JSON saved:        {summary_path}")

    # Plot metrics
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        frames_x = [l["frame"] for l in logs]
        psnrs = [l["psnr"] for l in logs]
        gaussians_count = [l["n_gaussians"] for l in logs]

        ax1.plot(frames_x, psnrs, "b-o", linewidth=2, markersize=4, label="PSNR (dB)")
        ax1.set_ylabel("PSNR (dB)", color="b", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.set_title(f"Adaptive 3DGS Online Reconstruction ({args.scene} - {args.policy.upper()})", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left")

        ax2.plot(frames_x, gaussians_count, "g-s", linewidth=2, markersize=4, label="Active Gaussians")
        ax2.set_xlabel("Frame Index", fontsize=11)
        ax2.set_ylabel("Gaussian Count", color="g", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper left")

        plt.tight_layout()
        plot_path = out_dir / "demo_metrics_curve.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"      -> Metrics Curve Plot saved:  {plot_path}")
    except Exception as e:
        print(f"      (Plot generation skipped: {e})")

    print("\n" + "=" * 80)
    print("                       DEMO COMPLETE! SUMMARY:")
    print("=" * 80)
    print(f"  • Mean PSNR:        {mean_psnr:.2f} dB  (Final: {logs[-1]['psnr']:.2f} dB)")
    print(f"  • Mean SSIM:        {mean_ssim:.4f}  (Final: {logs[-1]['ssim']:.4f})")
    print(f"  • Average Speed:    {mean_fps:.1f} FPS")
    print(f"  • Total Gaussians:  {final_gaussians:,}")
    print(f"  • Artifacts Dir:    {out_dir}")
    print(f"  • Interactive 3D:   python scripts/view_3d.py --path {out_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
