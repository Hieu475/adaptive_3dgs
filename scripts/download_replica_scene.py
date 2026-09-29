#!/usr/bin/env python3
"""Download and preprocess Replica scenes directly from Hugging Face (voviktyl/Replica-SLAM).

Compatible with both OnlineReconstructionPipeline (phase10_runtime) and ReplicaDataset.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
from huggingface_hub import hf_hub_download

REPO_ID = "voviktyl/Replica-SLAM"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Replica camera parameters
DEFAULT_FX = 600.0
DEFAULT_FY = 600.0
DEFAULT_CX = 599.5
DEFAULT_CY = 339.5
DEFAULT_W = 1200
DEFAULT_H = 680
DEPTH_SCALE = 6553.5


def parse_args():
    parser = argparse.ArgumentParser(description="Download and format Replica dataset from Hugging Face.")
    parser.add_argument("--scene", type=str, default="room2", help="Scene name (e.g. room2, room1, office1)")
    parser.add_argument("--n_frames", type=int, default=120, help="Number of keyframes to sample")
    parser.add_argument("--workers", type=int, default=16, help="Parallel download threads")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    return parser.parse_args()


def download_single_frame(scene: str, frame_idx: int) -> dict:
    rgb_file = f"{scene}/results/frame{frame_idx:06d}.jpg"
    depth_file = f"{scene}/results/depth{frame_idx:06d}.png"
    
    p_rgb = hf_hub_download(repo_id=REPO_ID, filename=rgb_file, repo_type="dataset")
    p_depth = hf_hub_download(repo_id=REPO_ID, filename=depth_file, repo_type="dataset")
    
    return {
        "frame_idx": frame_idx,
        "rgb_path": p_rgb,
        "depth_path": p_depth,
    }


def main():
    args = parse_args()
    scene = args.scene
    n_frames = args.n_frames
    
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "datasets" / "Replica" / scene
    images_dir = out_dir / "images"
    results_dir = out_dir / "results"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Downloading Replica Scene: '{scene}' from Hugging Face ({REPO_ID})")
    print(f"  Target Frames: {n_frames} keyframes")
    print(f"  Output Directory: {out_dir}")
    print("=" * 70)

    # 1. Download trajectory
    print("\n[1/4] Fetching camera trajectory (traj.txt)...")
    traj_path = hf_hub_download(repo_id=REPO_ID, filename=f"{scene}/traj.txt", repo_type="dataset")
    with open(traj_path, "r") as f:
        traj_lines = [l.strip() for l in f if l.strip()]
    total_available = len(traj_lines)
    print(f"      Loaded {total_available} camera poses from traj.txt.")

    # Compute keyframe indices across video
    indices = np.linspace(0, total_available - 1, n_frames, dtype=int).tolist()
    print(f"      Sampling {len(indices)} keyframes across sequence (stride ~ {total_available / n_frames:.1f}).")

    # 2. Download frames concurrently
    print(f"\n[2/4] Downloading {len(indices)} RGB-D frame pairs with {args.workers} workers...")
    t0 = time.time()
    downloaded_map = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_single_frame, scene, idx): idx for idx in indices}
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            downloaded_map[res["frame_idx"]] = res
            completed += 1
            if completed % 20 == 0 or completed == len(indices):
                print(f"      Progress: {completed}/{len(indices)} frames downloaded ({completed/len(indices)*100:.0f}%)...")

    download_time = time.time() - t0
    print(f"      All {len(indices)} frames downloaded in {download_time:.1f} s.")

    # 3. Process into pipeline format
    print("\n[3/4] Processing depth maps and poses into standardized format...")
    all_depths = []
    all_exts = []
    all_ints = []
    frame_ids = []

    K_base = np.array([
        [DEFAULT_FX, 0.0, DEFAULT_CX],
        [0.0, DEFAULT_FY, DEFAULT_CY],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    for i, idx in enumerate(indices):
        d_info = downloaded_map[idx]
        frame_name = f"frame_{i:06d}"
        frame_ids.append(frame_name)

        # Copy RGB to images/
        rgb_dst = images_dir / f"{frame_name}.jpg"
        img = cv2.imread(d_info["rgb_path"])
        cv2.imwrite(str(rgb_dst), img)

        # Also link/copy to results/ for ReplicaDataset
        cv2.imwrite(str(results_dir / f"frame{i:06d}.jpg"), img)

        # Process depth (uint16 -> float32 meters)
        depth_png = cv2.imread(d_info["depth_path"], cv2.IMREAD_UNCHANGED)
        depth_m = depth_png.astype(np.float32) / DEPTH_SCALE
        all_depths.append(depth_m)
        cv2.imwrite(str(results_dir / f"depth{i:06d}.png"), depth_png)

        # Camera pose: traj.txt is C2W -> invert to W2C
        c2w = np.array([float(x) for x in traj_lines[idx].split()]).reshape(4, 4).astype(np.float32)
        w2c = np.linalg.inv(c2w)
        all_exts.append(w2c)
        all_ints.append(K_base.copy())

    depth_array = np.stack(all_depths, axis=0)       # (N, 680, 1200)
    exts_array = np.stack(all_exts, axis=0)           # (N, 4, 4)
    ints_array = np.stack(all_ints, axis=0)           # (N, 3, 3)

    # 4. Save artifacts
    print("\n[4/4] Writing numpy arrays and metadata...")
    np.save(str(out_dir / "depth.npy"), depth_array)
    np.save(str(out_dir / "extrinsics.npy"), exts_array)
    np.save(str(out_dir / "intrinsics.npy"), ints_array)

    # Write traj.txt for sampled frames
    with open(out_dir / "traj.txt", "w") as f:
        for idx in indices:
            f.write(traj_lines[idx] + "\n")

    # Write meta.json
    meta = {
        "scene_id": f"replica/{scene}",
        "dataset": "replica",
        "is_pseudo": False,
        "n_frames": len(indices),
        "frame_ids": frame_ids,
        "resolution": [DEFAULT_W, DEFAULT_H],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    (out_dir / ".done").touch()
    print("=" * 70)
    print(f"  SUCCESS! Replica scene '{scene}' ready at: {out_dir}")
    print(f"  Total frames: {len(indices)}")
    print(f"  Resolution: {DEFAULT_W}x{DEFAULT_H}")
    print(f"  Depth shape: {depth_array.shape}, range: [{depth_array.min():.2f}m, {depth_array.max():.2f}m]")
    print("=" * 70)


if __name__ == "__main__":
    main()
