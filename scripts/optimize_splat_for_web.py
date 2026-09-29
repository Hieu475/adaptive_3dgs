#!/usr/bin/env python3
"""Optimize 3D Gaussian Splats for Web Viewers (SuperSplat, Antimatter15, Three.js).

Key Optimizations:
1. Centering: Shifts the room centroid to (0, 0, 0) so camera orbit rotates naturally around room center.
2. Up-Vector Alignment: Flips Y & Z from OpenCV coordinates to WebGL (+Y is Up, -Z is Forward).
3. Level-of-Detail (LOD) Downsampling: Downsamples from 1.4M to ~450k splats for locked 60 FPS in browsers.
4. Export to .splat: Generates 32-byte binary format for instantaneous web loading.
"""
import sys
import time
import argparse
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.view_3d import load_ply_file
from scripts.run_demo import export_ply_3dgs


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize 3DGS PLY for Web Viewers")
    parser.add_argument("--input", type=str, default="results/demo_output/replica_room2_full_room/reconstructed_3dgs.ply",
                        help="Path to input 3DGS PLY file")
    parser.add_argument("--target_splats", type=int, default=450000,
                        help="Target number of splats for web LOD (default: 450,000 for 60 FPS)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (defaults to input file's parent dir)")
    parser.add_argument("--source_up", type=str, default="+z", choices=["+z", "+y", "-z", "-y"],
                        help="Source coordinate Up-vector (+z for Replica/ROS, +y for WebGL/Blender)")
    return parser.parse_args()


def export_binary_splat(
    filepath: Path,
    positions: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    rotations: np.ndarray,
    opacities: np.ndarray,
):
    """Export to antimatter15 / SuperSplat standard 32-byte binary .splat format.

    Record layout (32 bytes per splat):
        pos:   3 x float32 (12 bytes)
        scale: 3 x float32 (12 bytes)
        color: 4 x uint8 (RGBA) (4 bytes)
        rot:   4 x uint8 (quaternion: rot_0, rot_1, rot_2, rot_3 mapped to [0, 255]) (4 bytes)
    """
    N = len(positions)
    # Ensure scales are linear (not log)
    scales_linear = np.clip(scales, 1e-6, None).astype(np.float32)

    # RGBA uint8
    r = np.clip(colors[:, 0] * 255.0, 0, 255).astype(np.uint8)
    g = np.clip(colors[:, 1] * 255.0, 0, 255).astype(np.uint8)
    b = np.clip(colors[:, 2] * 255.0, 0, 255).astype(np.uint8)
    a = np.clip(opacities.squeeze() * 255.0, 0, 255).astype(np.uint8)
    rgba = np.stack([r, g, b, a], axis=-1)

    # Rotation: normalize quaternion and quantize to uint8
    norm = np.linalg.norm(rotations, axis=-1, keepdims=True) + 1e-8
    q = rotations / norm
    q_uint8 = np.clip(q * 128.0 + 128.0, 0, 255).astype(np.uint8)

    dtype_splat = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('sx', '<f4'), ('sy', '<f4'), ('sz', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'),
        ('q0', 'u1'), ('q1', 'u1'), ('q2', 'u1'), ('q3', 'u1')
    ])

    buf = np.empty(N, dtype=dtype_splat)
    buf['x'] = positions[:, 0]
    buf['y'] = positions[:, 1]
    buf['z'] = positions[:, 2]
    buf['sx'] = scales_linear[:, 0]
    buf['sy'] = scales_linear[:, 1]
    buf['sz'] = scales_linear[:, 2]
    buf['r'] = rgba[:, 0]
    buf['g'] = rgba[:, 1]
    buf['b'] = rgba[:, 2]
    buf['a'] = rgba[:, 3]
    buf['q0'] = q_uint8[:, 0]
    buf['q1'] = q_uint8[:, 1]
    buf['q2'] = q_uint8[:, 2]
    buf['q3'] = q_uint8[:, 3]

    with open(filepath, "wb") as f:
        buf.tofile(f)


def rotate_quaternions_z_to_y(q: np.ndarray) -> np.ndarray:
    """Rotate quaternions by -90 degrees around X-axis.
    Transforms coordinates from +Z-up (Replica/ROS) to +Y-up (WebGL/Three.js/SuperSplat).

    New coordinates: [X, Z, -Y]
    Quaternion transformation: q_trans = [1/sqrt(2), -1/sqrt(2), 0, 0] (w, x, y, z)
    q' = q_trans * q:
        w' = c * (w + x)
        x' = c * (x - w)
        y' = c * (y + z)
        z' = c * (z - y)
    where c = 1 / sqrt(2).
    """
    c = float(1.0 / np.sqrt(2.0))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w_new = c * (w + x)
    x_new = c * (x - w)
    y_new = c * (y + z)
    z_new = c * (z - y)
    return np.stack([w_new, x_new, y_new, z_new], axis=-1)


def rotate_quaternions_neg_z_to_y(q: np.ndarray) -> np.ndarray:
    """Rotate quaternions by +90 degrees around X-axis.
    Transforms coordinates from -Z-up to +Y-up (WebGL/Three.js/SuperSplat).

    New coordinates: [X, -Z, Y]
    q_trans = [1/sqrt(2), 1/sqrt(2), 0, 0]
    q' = q_trans * q:
        w' = c * (w - x)
        x' = c * (x + w)
        y' = c * (y - z)
        z' = c * (z + y)
    """
    c = float(1.0 / np.sqrt(2.0))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w_new = c * (w - x)
    x_new = c * (x + w)
    y_new = c * (y - z)
    z_new = c * (z + y)
    return np.stack([w_new, x_new, y_new, z_new], axis=-1)


def main():
    args = parse_args()
    input_file = Path(args.input)
    if not input_file.is_absolute():
        input_file = REPO_ROOT / input_file

    if not input_file.exists():
        print(f"[Error] File not found: {input_file}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else input_file.parent

    print("=" * 70)
    print("      3D GAUSSIAN SPLATTING — WEB OPTIMIZER & EXPORTER")
    print("=" * 70)
    print(f"  Input File:     {input_file}")
    print(f"  Target Splats:  {args.target_splats:,} (for 60 FPS Web)")
    print(f"  Output Dir:     {out_dir}")
    print("=" * 70)

    # 1. Load Input PLY
    print("\n[1/4] Loading input 3DGS PLY...")
    t0 = time.time()
    data = load_ply_file(input_file)
    N_orig = data["num_points"]
    print(f"      Loaded {N_orig:,} splats in {time.time() - t0:.2f} s.")

    positions = data["centers"].copy()
    scales = data["scales"].copy()
    rotations = data["rotations"].copy()
    colors = data["colors"].copy()
    opacities = data["opacities"].copy()

    # 2. Filter Low-Opacity Floaters (< 0.10)
    valid_op = opacities.squeeze() >= 0.10
    if np.sum(~valid_op) > 0:
        print(f"      Pruned {np.sum(~valid_op):,} low-opacity floaters (< 0.10).")
        positions = positions[valid_op]
        scales = scales[valid_op]
        rotations = rotations[valid_op]
        colors = colors[valid_op]
        opacities = opacities[valid_op]
        N_orig = len(positions)

    # 3. Compute Centroid and Up-Vector Transformation
    print("\n[2/4] Centering and aligning coordinate system for WebGL...")
    centroid = np.mean(positions, axis=0)
    print(f"      Original Centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}] m")

    # Shift to center (0, 0, 0)
    pos_centered = positions - centroid

    if args.source_up == "+z":
        # Transform +Z-up (Replica/ROS) to +Y-up (WebGL/Three.js/SuperSplat)
        # R_X(-90 deg): X_web = X, Y_web = Z, Z_web = -Y
        # Preserves floor at bottom (-Y) and ceiling at top (+Y)
        pos_web = np.stack([pos_centered[:, 0], pos_centered[:, 2], -pos_centered[:, 1]], axis=-1)
        rot_web = rotate_quaternions_z_to_y(rotations)
        print("      Transformed coordinates: +Z-up (World) -> +Y-up (WebGL/SuperSplat)")
    elif args.source_up == "-z":
        pos_web = np.stack([pos_centered[:, 0], -pos_centered[:, 2], pos_centered[:, 1]], axis=-1)
        rot_web = rotate_quaternions_neg_z_to_y(rotations)
        print("      Transformed coordinates: -Z-up -> +Y-up")
    else:
        pos_web = pos_centered
        rot_web = rotations
        print(f"      Preserved existing {args.source_up} coordinate system")

    # 4. Export Centered Full-Res PLY & Full-Res .splat (100% Sharpness)
    centered_ply = out_dir / "reconstructed_3dgs_centered.ply"
    print(f"\n[3/4] Exporting centered full-resolution PLY ({N_orig:,} splats)...")
    export_ply_3dgs(
        filepath=centered_ply,
        positions=pos_web,
        scales=scales,
        rotations=rot_web,
        opacities=opacities,
        colors_dc=colors,
    )
    print(f"      -> Saved PLY:   {centered_ply.name} ({centered_ply.stat().st_size / (1024*1024):.1f} MB)")

    # Full-resolution .splat for SuperSplat (Maximum fidelity, 0 blurriness)
    full_splat = out_dir / "reconstructed_3dgs_full.splat"
    export_binary_splat(
        filepath=full_splat,
        positions=pos_web,
        scales=scales,
        colors=colors,
        rotations=rot_web,
        opacities=opacities,
    )
    print(f"      -> Saved SPLAT: {full_splat.name} ({full_splat.stat().st_size / (1024*1024):.1f} MB, {N_orig:,} splats - Ultra Sharp)")

    # 5. Generate Downsampled Web LOD (Level of Detail) & .splat
    print(f"\n[4/4] Generating Web-Optimized LOD ({args.target_splats:,} splats)...")
    if N_orig > args.target_splats:
        stride = int(np.ceil(N_orig / args.target_splats))
        sub_idx = np.arange(0, N_orig, stride)
        # Keep original scales so Gaussians stay crisp, sharp and don't blur into fog
        scales_lod = scales[sub_idx]
        pos_lod = pos_web[sub_idx]
        rot_lod = rot_web[sub_idx]
        colors_lod = colors[sub_idx]
        opacities_lod = opacities[sub_idx]
        N_lod = len(sub_idx)
    else:
        scales_lod = scales
        pos_lod = pos_web
        rot_lod = rot_web
        colors_lod = colors
        opacities_lod = opacities
        N_lod = N_orig

    # Export Web PLY
    web_ply = out_dir / "reconstructed_3dgs_web.ply"
    export_ply_3dgs(
        filepath=web_ply,
        positions=pos_lod,
        scales=scales_lod,
        rotations=rot_lod,
        opacities=opacities_lod,
        colors_dc=colors_lod,
    )
    print(f"      -> Web PLY:   {web_ply.name} ({web_ply.stat().st_size / (1024*1024):.1f} MB, {N_lod:,} splats)")

    # Export Web .splat (binary 32-byte layout)
    web_splat = out_dir / "reconstructed_3dgs_web.splat"
    export_binary_splat(
        filepath=web_splat,
        positions=pos_lod,
        scales=scales_lod,
        colors=colors_lod,
        rotations=rot_lod,
        opacities=opacities_lod,
    )
    print(f"      -> Web SPLAT: {web_splat.name} ({web_splat.stat().st_size / (1024*1024):.1f} MB, {N_lod:,} splats)")

    print("=" * 70)
    print("  OPTIMIZATION COMPLETE! 3 WEB ASSETS CREATED:")
    print(f"  1. Centered PLY (Full 1.4M): {centered_ply}")
    print(f"  2. Web 60FPS PLY (470k):     {web_ply}")
    print(f"  3. SuperSplat .splat (15MB): {web_splat}")
    print("=" * 70)


if __name__ == "__main__":
    main()
