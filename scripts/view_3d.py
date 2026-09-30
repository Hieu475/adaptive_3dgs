#!/usr/bin/env python3
"""Interactive 3D Scene Viewer for Adaptive 3DGS Reconstructions.

Visualizes reconstructed 3D Gaussian Splats or Point Clouds (.ply)
using a local interactive WebGL viewer powered by Viser.

Features:
- Native 3D Gaussian Splatting rendering with true covariance, rotation, scaling & SH/RGB colors
- Seamless toggling between 3D Gaussian Splats and Point Cloud modes
- Interactive Splat Scale multiplier and Min Opacity filter
- Point Size adjustment slider
- Dynamic file switcher to compare scenes and reconstructions
- Camera reset, floor grid, and coordinate axes frame

Usage:
    python scripts/view_3d.py --path results/demo_output/replica_room0
    python scripts/view_3d.py --path results/demo_output/tum_fr2_xyz_highres
    python scripts/view_3d.py --path results/demo_output/replica_office0/reconstructed_3dgs.ply
"""
import os
import sys
import time
import socket
import argparse
import webbrowser
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import trimesh
import logging
import websockets.asyncio.server

# 1. Increase Viser WebSocket max_size from 50MB to 500MB to handle large 3DGS point sets (> 2.5M splats)
_orig_serve = websockets.asyncio.server.serve
def _patched_serve(*args, **kwargs):
    if 'max_size' in kwargs:
        kwargs['max_size'] = 500 * 1024 * 1024
    return _orig_serve(*args, **kwargs)
websockets.asyncio.server.serve = _patched_serve

# 2. Suppress harmless EOFError / premature disconnects from browser pre-connections & tab switches
class WebsocketsConnectionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "opening handshake failed" in msg or "connection closed while reading HTTP" in msg:
            return False
        return True

logging.getLogger("websockets.server").addFilter(WebsocketsConnectionFilter())
logging.getLogger("websockets").addFilter(WebsocketsConnectionFilter())

import viser

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_available_port(start_port: int, max_attempts: int = 15) -> int:
    """Find next available TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    return start_port


def parse_ply_header(filepath: Path) -> Tuple[List[Tuple[str, str]], int, int]:
    """Parse PLY header to extract property types, names, vertex count, and header byte offset."""
    properties: List[Tuple[str, str]] = []
    num_vertices = 0
    header_bytes = 0
    with open(filepath, 'rb') as f:
        while True:
            line = f.readline()
            header_bytes += len(line)
            line_str = line.decode('ascii', errors='ignore').strip()
            if line_str.startswith('element vertex'):
                num_vertices = int(line_str.split()[-1])
            elif line_str.startswith('property'):
                parts = line_str.split()
                if len(parts) >= 3:
                    properties.append((parts[1], parts[2]))
            elif line_str == 'end_header':
                break
    return properties, num_vertices, header_bytes


def compute_covariances_numpy(scales: np.ndarray, rotations_wxyz: np.ndarray) -> np.ndarray:
    """Compute 3D covariance matrices Sigma = R * S * S^T * R^T from scale and quaternion.

    Args:
        scales: (N, 3) linear scales (already exponentiated if stored as log)
        rotations_wxyz: (N, 4) quaternions (w, x, y, z)
    Returns:
        covariances: (N, 3, 3) symmetric positive semi-definite covariance matrices
    """
    N = len(scales)
    # Normalize quaternion
    norm = np.linalg.norm(rotations_wxyz, axis=-1, keepdims=True) + 1e-8
    q = rotations_wxyz / norm
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Build 3x3 rotation matrices
    R = np.empty((N, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - w * z)
    R[:, 0, 2] = 2.0 * (x * z + w * y)
    R[:, 1, 0] = 2.0 * (x * y + w * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - w * x)
    R[:, 2, 0] = 2.0 * (x * z - w * y)
    R[:, 2, 1] = 2.0 * (y * z + w * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    # RS = R @ diag(s) = R * s[:, None, :]
    RS = R * scales[:, np.newaxis, :]
    covariances = np.matmul(RS, np.swapaxes(RS, 1, 2))
    return covariances


def load_ply_file(filepath: Path) -> Dict[str, Any]:
    """Load 3DGS or standard Point Cloud PLY file.

    Returns dict with keys:
        'type': '3dgs' or 'point_cloud'
        'num_points': int
        'centers': np.ndarray (N, 3)
        'colors': np.ndarray (N, 3) in [0, 1]
        'covariances': np.ndarray (N, 3, 3) [if 3dgs]
        'opacities': np.ndarray (N, 1) in [0, 1] [if 3dgs]
        'scales': np.ndarray (N, 3) [if 3dgs]
        'rotations': np.ndarray (N, 4) [if 3dgs]
    """
    filepath = Path(filepath)
    try:
        properties, num_vertices, header_bytes = parse_ply_header(filepath)
        prop_names = [p[1] for p in properties]
        is_3dgs = 'scale_0' in prop_names and 'rot_0' in prop_names and 'opacity' in prop_names

        type_map = {
            'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
            'uchar': 'u1', 'uint8': 'u1', 'int': '<i4', 'int32': '<i4',
            'short': '<i2', 'ushort': '<u2', 'uint': '<u4',
        }
        dtype_list = [(name, type_map.get(t, '<f4')) for t, name in properties]
        vertex_dtype = np.dtype(dtype_list)

        with open(filepath, 'rb') as f:
            f.seek(header_bytes)
            data = np.fromfile(f, dtype=vertex_dtype, count=num_vertices)

        centers = np.stack([data['x'], data['y'], data['z']], axis=-1).astype(np.float32)

        if is_3dgs:
            # Colors: evaluate Inria SH DC or Sigmoid
            C0 = 0.28209479177387814
            if 'f_dc_0' in prop_names:
                f_dc = np.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], axis=-1).astype(np.float32)
                colors = np.clip(0.5 + C0 * f_dc, 0.0, 1.0)
            elif 'red' in prop_names:
                colors = np.stack([data['red'], data['green'], data['blue']], axis=-1).astype(np.float32)
                if colors.max() > 1.0:
                    colors = colors / 255.0
            else:
                colors = np.ones_like(centers) * 0.7

            # Opacity (detect logit vs linear)
            raw_op = data['opacity'].astype(np.float32)[:, np.newaxis]
            if np.any(raw_op < 0.0) or np.any(raw_op > 1.0):
                opacities = 1.0 / (1.0 + np.exp(-np.clip(raw_op, -15.0, 15.0)))
            else:
                opacities = np.clip(raw_op, 0.0, 1.0)

            # Scales: in 3DGS PLY format, scales are typically stored in log-space
            scales_raw = np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=-1).astype(np.float32)
            # Check if scales are in log space (can be negative) or already linear
            if np.any(scales_raw < 0.0) or np.mean(scales_raw) < 0.0:
                scales = np.exp(np.clip(scales_raw, -15.0, 5.0))
            else:
                scales = np.clip(scales_raw, 1e-6, None)

            # Rotations (w, x, y, z)
            rotations = np.stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']], axis=-1).astype(np.float32)

            # Covariances
            covariances = compute_covariances_numpy(scales, rotations)

            return {
                'type': '3dgs',
                'num_points': len(centers),
                'centers': centers,
                'colors': colors,
                'opacities': opacities,
                'covariances': covariances,
                'scales': scales,
                'rotations': rotations,
            }
        else:
            if 'red' in prop_names:
                colors = np.stack([data['red'], data['green'], data['blue']], axis=-1).astype(np.float32)
                if colors.max() > 1.0:
                    colors = colors / 255.0
            else:
                colors = np.ones_like(centers) * 0.7
            return {
                'type': 'point_cloud',
                'num_points': len(centers),
                'centers': centers,
                'colors': colors,
            }

    except Exception as e:
        print(f"      [Warning] Fast binary PLY parser failed ({e}), falling back to trimesh...")
        mesh = trimesh.load(filepath)
        points = np.array(mesh.vertices, dtype=np.float32)
        colors = None
        if hasattr(mesh, 'colors') and mesh.colors is not None and len(mesh.colors) > 0:
            colors = np.array(mesh.colors[:, :3], dtype=np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
        else:
            colors = np.ones_like(points) * 0.7
        return {
            'type': 'point_cloud',
            'num_points': len(points),
            'centers': points,
            'colors': colors,
        }


def collect_ply_files(target_path: Path) -> List[Path]:
    """Collect available PLY files, sorting 3DGS files to the top."""
    if target_path.is_file() and target_path.suffix.lower() == '.ply':
        folder = target_path.parent
        siblings = [p for p in folder.glob("*.ply") if p != target_path]
        return [target_path] + sorted(siblings)

    folder = target_path if target_path.is_dir() else target_path.parent
    ply_files = list(folder.glob("*.ply"))
    if not ply_files:
        ply_files = list(folder.glob("**/*.ply"))

    # Priority sorting:
    # 0: Full-resolution centered 3DGS (100% sharpness, 1.41M splats, upright)
    # 1: Full-resolution raw 3DGS (1.41M splats)
    # 2: Downsampled web LOD (350k splats)
    # 3: Point cloud
    def sort_key(p: Path):
        name = p.name.lower()
        if "3dgs_centered" in name:
            return (0, name)
        elif "3dgs" in name and "web" not in name:
            return (1, name)
        elif "3dgs_web" in name:
            return (2, name)
        elif "point_cloud" in name or "pcd" in name:
            return (3, name)
        return (4, name)

    return sorted(ply_files, key=sort_key)


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D Viewer for Reconstructed Scenes")
    parser.add_argument("--path", type=str, default="results/demo_output/replica_room0",
                        help="Path to .ply file or demo_output scene directory")
    parser.add_argument("--port", type=int, default=8080, help="Web viewer port (default: 8080)")
    parser.add_argument("--point_size", type=float, default=0.015, help="Initial point cloud size (default: 0.015)")
    parser.add_argument("--splat_scale", type=float, default=0.8, help="Initial Gaussian splat scale factor (default: 0.8 for crisp rendering)")
    parser.add_argument("--no_browser", action="store_true", help="Don't auto-open web browser")
    args = parser.parse_args()

    input_path = Path(args.path)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path

    if not input_path.exists():
        print(f"[Error] Specified path does not exist: {input_path}")
        sys.exit(1)

    ply_files = collect_ply_files(input_path)
    if not ply_files:
        print(f"[Error] No .ply files found at: {input_path}")
        sys.exit(1)

    # Select initial file
    initial_ply = ply_files[0]

    # Resolve port to avoid collisions
    port = find_available_port(args.port)
    if port != args.port:
        print(f"[Notice] Port {args.port} is in use; dynamically switched to port {port}.")

    print("=" * 78)
    print("        ADAPTIVE 3DGS - INTERACTIVE 3D SCENE VIEWER (VISER)")
    print("=" * 78)
    print(f"  Target File:     {initial_ply.name}")
    print(f"  Full Path:       {initial_ply}")
    print(f"  Available PLYs:  {len(ply_files)} file(s) detected in directory")
    print(f"  Viewer Port:     {port}")
    print(f"  Web Address:     http://localhost:{port}")
    print("=" * 78)

    # Launch Viser Server
    server = viser.ViserServer(port=port)

    # State variables
    current_data: Dict[str, Any] = {}
    current_ply_path: Path = initial_ply
    splat_handle = None
    pcd_handle = None
    grid_handle = None
    axes_handle = None
    max_dim = 5.0
    centroid = np.zeros(3, dtype=np.float32)

    # 1. UI Controls Setup
    with server.gui.add_folder("Scene Selection"):
        file_options = [p.name for p in ply_files]
        file_dropdown = server.gui.add_dropdown(
            "Select PLY File",
            options=file_options,
            initial_value=initial_ply.name,
        )
        info_markdown = server.gui.add_markdown("Loading scene...")

    with server.gui.add_folder("Display Settings"):
        render_mode_dropdown = server.gui.add_dropdown(
            "Render Mode",
            options=["Gaussian Splats (Native 3DGS)", "Point Cloud (Points)"],
            initial_value="Gaussian Splats (Native 3DGS)",
        )
        splat_scale_slider = server.gui.add_slider(
            "Splat Scale", min=0.1, max=3.0, step=0.05, initial_value=args.splat_scale
        )
        opacity_slider = server.gui.add_slider(
            "Min Opacity Filter", min=0.0, max=0.95, step=0.01, initial_value=0.10
        )
        point_size_slider = server.gui.add_slider(
            "Point Size", min=0.001, max=0.05, step=0.001, initial_value=args.point_size
        )

    with server.gui.add_folder("Environment & Camera"):
        up_direction_dropdown = server.gui.add_dropdown(
            "Scene Up Vector",
            options=[
                "+Z is Up (Replica / ROS Raw)",
                "+Y is Up (WebGL / SuperSplat)",
                "-Z is Up (Inverted Z)",
                "-Y is Up (Inverted Y)",
            ],
            initial_value="+Z is Up (Replica / ROS Raw)",
        )
        flip_up_btn = server.gui.add_button("🔄 Flip Floor ⇕ Ceiling")
        show_grid_checkbox = server.gui.add_checkbox("Show Floor Grid", initial_value=True)
        show_axes_checkbox = server.gui.add_checkbox("Show Coordinate Frame", initial_value=True)
        reset_cam_btn = server.gui.add_button("Reset Camera to Scene")

    def get_current_up_key() -> str:
        val = up_direction_dropdown.value
        if "+z" in val.lower():
            return "+z"
        elif "-z" in val.lower():
            return "-z"
        elif "+y" in val.lower():
            return "+y"
        elif "-y" in val.lower():
            return "-y"
        return "+z"

    def apply_scene_up():
        nonlocal grid_handle
        up_key = get_current_up_key()
        try:
            server.scene.set_up_direction(up_key)
        except Exception:
            pass

        if current_data and 'centers' in current_data:
            centers = current_data['centers']
            z_min = float(np.min(centers[:, 2]))
            z_max = float(np.max(centers[:, 2]))
            y_min = float(np.min(centers[:, 1]))
            y_max = float(np.max(centers[:, 1]))

            if grid_handle is not None:
                grid_handle.remove()
                grid_handle = None

            if up_key == "+z":
                plane = "xy"
                pos = (float(centroid[0]), float(centroid[1]), z_min)
            elif up_key == "-z":
                plane = "xy"
                pos = (float(centroid[0]), float(centroid[1]), z_max)
            elif up_key == "+y":
                plane = "xz"
                pos = (float(centroid[0]), y_min, float(centroid[2]))
            else:  # -y
                plane = "xz"
                pos = (float(centroid[0]), y_max, float(centroid[2]))

            grid_handle = server.scene.add_grid(
                "floor_grid",
                plane=plane,
                width=max(max_dim * 1.6, 2.0),
                height=max(max_dim * 1.6, 2.0),
                position=pos,
                visible=show_grid_checkbox.value,
            )

    def reset_camera():
        up_key = get_current_up_key()
        cam_dist = max(max_dim * 1.3, 1.5)
        if up_key == "+z":
            cam_pos = centroid + np.array([0.0, -cam_dist, cam_dist * 0.5], dtype=np.float32)
            up_vec = (0.0, 0.0, 1.0)
        elif up_key == "-z":
            cam_pos = centroid + np.array([0.0, -cam_dist, -cam_dist * 0.5], dtype=np.float32)
            up_vec = (0.0, 0.0, -1.0)
        elif up_key == "+y":
            cam_pos = centroid + np.array([0.0, cam_dist * 0.4, cam_dist * 1.2], dtype=np.float32)
            up_vec = (0.0, 1.0, 0.0)
        else:  # -y
            cam_pos = centroid + np.array([0.0, -cam_dist * 0.4, cam_dist * 1.2], dtype=np.float32)
            up_vec = (0.0, -1.0, 0.0)

        server.initial_camera.position = tuple(cam_pos)
        server.initial_camera.look_at = tuple(centroid)
        server.initial_camera.up = up_vec
        for client in server.get_clients().values():
            client.camera.position = tuple(cam_pos)
            client.camera.look_at = tuple(centroid)
            try:
                client.camera.up_direction = np.array(up_vec, dtype=np.float32)
            except Exception:
                pass

    def apply_splat_updates():
        nonlocal splat_handle
        if splat_handle is None or current_data.get('type') != '3dgs':
            return
        
        scale_val = splat_scale_slider.value
        min_op = opacity_slider.value
        
        opacities = current_data['opacities']
        mask = opacities[:, 0] >= min_op
        
        if np.sum(mask) == 0:
            mask = np.ones(len(opacities), dtype=bool)

        sub_centers = current_data['centers'][mask]
        sub_cov = current_data['covariances'][mask] * float(scale_val ** 2)
        sub_rgbs = current_data['colors'][mask]
        sub_opacities = opacities[mask]
        
        splat_handle.set_gaussians(sub_centers, sub_cov, sub_rgbs, sub_opacities)

    def load_and_render_file(target_ply: Path):
        nonlocal current_data, current_ply_path, splat_handle, pcd_handle, grid_handle, axes_handle, max_dim, centroid

        current_ply_path = target_ply
        print(f"\n[Loading] Reading {target_ply.name} ...")
        t0 = time.time()
        current_data = load_ply_file(target_ply)
        elapsed = time.time() - t0

        centers = current_data['centers']
        colors = current_data['colors']
        N = current_data['num_points']
        scene_type = current_data['type']

        centroid = np.mean(centers, axis=0)
        extent = np.max(centers, axis=0) - np.min(centers, axis=0)
        max_dim = float(np.max(extent))

        print(f"          Loaded {N:,} elements ({scene_type.upper()}) in {elapsed:.3f}s.")
        print(f"          Bounding Box: {extent.round(2)} m (Max dim: {max_dim:.2f} m)")

        # Clear existing scene nodes
        if splat_handle is not None:
            splat_handle.remove()
            splat_handle = None
        if pcd_handle is not None:
            pcd_handle.remove()
            pcd_handle = None
        if grid_handle is not None:
            grid_handle.remove()
            grid_handle = None
        if axes_handle is not None:
            axes_handle.remove()
            axes_handle = None

        # Auto-detect Scene Up Vector based on file name
        is_web = ("web" in target_ply.name.lower() or "centered" in target_ply.name.lower())
        if is_web:
            up_direction_dropdown.value = "+Y is Up (WebGL / SuperSplat)"
        else:
            up_direction_dropdown.value = "+Z is Up (Replica / ROS Raw)"

        apply_scene_up()

        # Add Coordinate Axes Frame
        axes_handle = server.scene.add_frame(
            "coordinate_frame",
            show_axes=show_axes_checkbox.value,
            axes_length=max(max_dim * 0.1, 0.3),
            axes_radius=max(max_dim * 0.003, 0.008),
        )

        # Build 3DGS or Point Cloud
        if scene_type == '3dgs':
            cov = current_data['covariances'] * float(splat_scale_slider.value ** 2)
            op = current_data['opacities']

            splat_handle = server.scene.add_gaussian_splats(
                "reconstruction_3dgs",
                centers=centers,
                covariances=cov,
                rgbs=colors,
                opacities=op,
                visible=(render_mode_dropdown.value == "Gaussian Splats (Native 3DGS)"),
            )
            pcd_handle = server.scene.add_point_cloud(
                "reconstruction_pcd",
                points=centers,
                colors=colors,
                point_size=point_size_slider.value,
                visible=(render_mode_dropdown.value == "Point Cloud (Points)"),
            )
        else:
            # Point cloud only
            splat_handle = None
            pcd_handle = server.scene.add_point_cloud(
                "reconstruction_pcd",
                points=centers,
                colors=colors,
                point_size=point_size_slider.value,
                visible=True,
            )
            render_mode_dropdown.value = "Point Cloud (Points)"

        # Update GUI Info
        size_mb = target_ply.stat().st_size / (1024 * 1024)
        type_str = "✨ 3D Gaussian Splats" if scene_type == '3dgs' else "🔵 Point Cloud"
        info_markdown.content = (
            f"**File**: `{target_ply.name}` ({size_mb:.1f} MB)\n\n"
            f"**Format**: {type_str}\n\n"
            f"**Count**: **{N:,}** splats/points\n\n"
            f"**Extents**: `{extent[0]:.2f}m × {extent[1]:.2f}m × {extent[2]:.2f}m`\n\n"
            f"**Status**: Ready"
        )

        reset_camera()

    # Initial Load
    load_and_render_file(initial_ply)

    # Event Callbacks
    @file_dropdown.on_update
    def _(_):
        chosen_name = file_dropdown.value
        for p in ply_files:
            if p.name == chosen_name:
                load_and_render_file(p)
                break

    @render_mode_dropdown.on_update
    def _(_):
        mode = render_mode_dropdown.value
        is_splat = (mode == "Gaussian Splats (Native 3DGS)")
        if splat_handle is not None:
            splat_handle.visible = is_splat
        if pcd_handle is not None:
            pcd_handle.visible = not is_splat

    @splat_scale_slider.on_update
    def _(_):
        apply_splat_updates()

    @opacity_slider.on_update
    def _(_):
        apply_splat_updates()

    @point_size_slider.on_update
    def _(_):
        if pcd_handle is not None:
            pcd_handle.point_size = point_size_slider.value

    @show_grid_checkbox.on_update
    def _(_):
        if grid_handle is not None:
            grid_handle.visible = show_grid_checkbox.value

    @show_axes_checkbox.on_update
    def _(_):
        if axes_handle is not None:
            axes_handle.visible = show_axes_checkbox.value

    @reset_cam_btn.on_click
    def _(_):
        reset_camera()

    @flip_up_btn.on_click
    def _(_):
        curr = up_direction_dropdown.value
        if "+Z" in curr:
            up_direction_dropdown.value = "-Z is Up (Inverted Z)"
        elif "-Z" in curr:
            up_direction_dropdown.value = "+Z is Up (Replica / ROS Raw)"
        elif "+Y" in curr:
            up_direction_dropdown.value = "-Y is Up (Inverted Y)"
        elif "-Y" in curr:
            up_direction_dropdown.value = "+Y is Up (WebGL / SuperSplat)"
        apply_scene_up()
        reset_camera()

    @up_direction_dropdown.on_update
    def _(_):
        apply_scene_up()
        reset_camera()

    # Usage instructions
    print("\n" + "*" * 78)
    print(f"  >>> Trình duyệt 3D đã sẵn sàng tại: http://localhost:{port}")
    print("  • Chuột trái:      Xoay góc nhìn (Orbit 360°)")
    print("  • Chuột phải:      Di chuyển camera (Pan)")
    print("  • Con lăn chuột:   Phóng to / Thu nhỏ (Zoom)")
    print("  • Thanh công cụ bên phải:")
    print("      - Chọn PLY: Chuyển đổi giữa các bản dựng (3DGS vs Point Cloud)")
    print("      - Render Mode: Đổi giữa Gaussian Splats sắc nét và Point Cloud")
    print("      - Splat Scale: Tăng giảm kích thước hạt Gaussian (tránh thưa thớt/mờ)")
    print("      - Min Opacity: Lọc bớt các hạt Gaussian trong suốt")
    print("*" * 78 + "\n")

    if not args.no_browser:
        url = f"http://localhost:{port}"
        launched = False
        import subprocess
        import shutil

        # Force browser to run on discrete NVIDIA GPU with hardware acceleration
        gpu_env = os.environ.copy()
        gpu_env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        gpu_env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
        gpu_env["__VK_LAYER_NV_optimus"] = "NVIDIA_only"

        chrome_bin = shutil.which("google-chrome") or shutil.which("chromium")
        if chrome_bin:
            try:
                subprocess.Popen([
                    chrome_bin,
                    "--ignore-gpu-blocklist",
                    "--enable-gpu-rasterization",
                    "--enable-zero-copy",
                    "--enable-features=WebGPU",
                    url
                ], env=gpu_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                launched = True
            except Exception:
                pass
        if not launched:
            try:
                webbrowser.open(url)
            except Exception:
                pass

    # Keep server running
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping Viser server...")
        server.stop()


if __name__ == "__main__":
    main()
