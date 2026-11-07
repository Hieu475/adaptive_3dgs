#!/usr/bin/env python3
"""Compute ground-truth camera motion summaries for a loaded sequence prefix."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run_frozen_benchmark import load_benchmark_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--n_frames", type=int, default=61)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames, _ = load_benchmark_sequence(scene_name=args.scene,
        n_frames=args.n_frames, H=240, W=320, device="cpu")
    poses_w2c = torch.stack([frame["pose"].float() for frame in frames])
    # Convert to camera-to-world, then compute consecutive camera motion.
    poses_c2w = torch.linalg.inv(poses_w2c)
    relative = torch.linalg.inv(poses_c2w[:-1]) @ poses_c2w[1:]
    translation = torch.linalg.vector_norm(relative[:, :3, 3], dim=1).numpy()
    trace = relative[:, :3, :3].diagonal(dim1=1, dim2=2).sum(dim=1)
    rotation_deg = torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1, 1))).numpy()
    result = {
        "scene": args.scene,
        "pose_convention": "Phase 10 loader GT W2C inverted to C2W for consecutive camera motion",
        "frames_loaded": len(frames),
        "transitions": int(len(translation)),
        "translation_m_per_frame": {
            "mean": float(np.mean(translation)),
            "p95": float(np.percentile(translation, 95)),
        },
        "rotation_deg_per_frame": {
            "mean": float(np.mean(rotation_deg)),
            "p95": float(np.percentile(rotation_deg, 95)),
        },
        "window": "first n_frames loaded from Phase 10 sequence loader",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
