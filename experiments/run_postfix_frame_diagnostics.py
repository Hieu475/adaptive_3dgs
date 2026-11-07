#!/usr/bin/env python3
"""Capture per-frame scene/resource diagnostics for post-fix failure scenes.

Run this from a clean frozen checkout with the Phase 14 post-area-mask code.
Each complete trajectory is saved immediately so an interrupted run can resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_frozen_benchmark import (
    build_pipeline_config,
    load_benchmark_sequence,
    run_policy_trajectory,
)

DEFAULT_SCENES = ("replica_office1", "tum_fr1_rpy")
DEFAULT_POLICIES = ("ours", "error_only")
DEFAULT_SEEDS = tuple(range(42, 50))
W, H = 320, 240


def scene_features(frames: list[dict[str, Any]], intrinsics: torch.Tensor,
                   voxel_m: float = 0.05, pixel_stride: int = 4) -> list[dict[str, float]]:
    """Compute policy-independent motion, depth validity and observed-surface novelty."""
    K = intrinsics.detach().cpu().numpy()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    poses = [f["pose"].detach().cpu().numpy() for f in frames]
    centers = [np.linalg.inv(p)[:3, 3] for p in poses]
    cache: set[tuple[int, int, int]] = set()
    output: list[dict[str, float]] = []
    for i, frame in enumerate(frames):
        depth = frame["depth"].detach().cpu().numpy()
        if depth.ndim == 3:
            depth = depth.squeeze(0)
        sampled = depth[::pixel_stride, ::pixel_stride]
        valid = np.isfinite(sampled) & (sampled >= 0.4) & (sampled <= 4.0)
        valid_ratio = float(valid.mean())
        yy, xx = np.mgrid[0:depth.shape[0]:pixel_stride, 0:depth.shape[1]:pixel_stride]
        z = sampled[valid].astype(np.float64)
        x = (xx[valid] - cx) * z / fx
        y = (yy[valid] - cy) * z / fy
        camera_xyz = np.stack([x, y, z, np.ones_like(z)], axis=0)
        world_xyz = (np.linalg.inv(poses[i]) @ camera_xyz)[:3].T
        voxels = np.floor(world_xyz / voxel_m).astype(np.int32)
        unique_voxels = {tuple(v) for v in np.unique(voxels, axis=0)}
        novel = sum(v not in cache for v in unique_voxels)
        novelty = float(novel / max(1, len(unique_voxels)))
        cache.update(unique_voxels)

        if i == 0:
            translation_rate = 0.0
            angular_rate_deg = 0.0
        else:
            translation_rate = float(np.linalg.norm(centers[i] - centers[i - 1]))
            r0 = np.linalg.inv(poses[i - 1])[:3, :3]
            r1 = np.linalg.inv(poses[i])[:3, :3]
            relative = r0.T @ r1
            cos_angle = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
            angular_rate_deg = math.degrees(math.acos(cos_angle))
        output.append({
            "camera_translation_m_per_frame": translation_rate,
            "camera_angular_deg_per_frame": angular_rate_deg,
            "new_surface_ratio_voxel_5cm": novelty,
            "depth_valid_ratio_sampled": valid_ratio,
        })
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n_frames", type=int, default=150)
    parser.add_argument("--budget_ms", type=float, default=15.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="results/phase14_generalization_postfix/frame_diagnostics")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "phase14_postfix_frame_diagnostics",
        "status": "exploratory",
        "protocol_version": "phase14-postfix-frame-diagnostics-v1",
        "scene": args.scenes,
        "policies": args.policies,
        "seeds": args.seeds,
        "requested_frames": args.n_frames,
        "resolution_wh": [W, H],
        "budget_ms": args.budget_ms,
        "budget_scope": "modeled optimization budget, not end-to-end latency",
        "pose_convention": "ground-truth W2C from Phase 10 loader",
        "renderer_backend": "gsplat",
        "surface_novelty_definition": "fraction of unique world-space 5cm voxels from valid depth samples at 4-pixel stride that have not appeared in earlier trajectory frames; pose/depth based, policy independent",
        "mature_definition": "Gaussian age >= 10 frames; use the pipeline's mature_eligible and mature_update_coverage fields",
        "diagnostic_caveat": "Per-frame observations are repeated measures; comparisons are descriptive and uncertainty must be clustered by trajectory seed.",
        "source_revision": "75865c7921b5db9d5e8998fc94957935c798f1d6",
        "runner_sha256": sha256(Path(__file__).resolve()),
        "runs": [],
    }
    for scene in args.scenes:
        scene_frames, intrinsics = load_benchmark_sequence(
            scene_name=scene, n_frames=args.n_frames + 1, H=H, W=W, device=args.device
        )
        static_features = scene_features(scene_frames, intrinsics)
        print(f"{scene}: loaded {len(scene_frames)} frames; computed motion/depth/novelty features", flush=True)
        for policy in args.policies:
            for seed in args.seeds:
                target = out / scene / policy / f"seed{seed}.json"
                if args.resume and target.exists():
                    print(f"skip existing {target}", flush=True)
                    continue
                cfg = build_pipeline_config(policy=policy, seed=seed, budget_ms=args.budget_ms,
                                            W=W, H=H, device=args.device)
                t0 = time.perf_counter()
                summary, frame_logs = run_policy_trajectory(
                    policy=policy, seed=seed, frames=scene_frames, intrinsics=intrinsics,
                    budget_ms=args.budget_ms, device=args.device, W=W, H=H, custom_config=cfg,
                )
                joined = []
                for log in frame_logs:
                    idx = int(log["frame"])
                    item = dict(log)
                    item.update(static_features[idx])
                    joined.append(item)
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "metadata": {"scene": scene, "policy": policy, "seed": seed,
                                 "actual_frame_count": len(scene_frames), "budget_ms": args.budget_ms,
                                 "summary": summary, "diagnostics_wall_sec": time.perf_counter() - t0},
                    "frames": joined,
                }
                target.write_text(json.dumps(payload, allow_nan=False))
                manifest["runs"].append({"path": str(target.relative_to(out)), "sha256": sha256(target)})
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"saved {scene}/{policy}/seed{seed}: {len(joined)} rows", flush=True)


if __name__ == "__main__":
    main()
