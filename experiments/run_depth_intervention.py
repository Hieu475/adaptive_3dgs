#!/usr/bin/env python3
"""Within-scene depth interventions for the fixed selector comparison.

Uses identical perturbed RGB-D frames for both fixed policies, with the
coverage setting each policy used in the headline comparison. This estimates
the effect of depth corruption within a scene; RGB and poses stay fixed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run_frozen_benchmark import (
    build_pipeline_config, load_benchmark_sequence, run_policy_trajectory,
)


def modify_depth(frames, scene, level, seed):
    import cv2
    rng = np.random.default_rng(seed + 10000)
    output = []
    for f in frames:
        d = f["depth"].detach().cpu().numpy().astype(np.float32)
        if scene.startswith("replica_"):
            valid = np.isfinite(d) & (d > 0)
            holes = rng.random(d.shape) < level
            noisy = np.maximum(0, d + rng.normal(0, level * 0.05, d.shape).astype(np.float32))
            noisy[~valid | holes] = 0
            d = noisy
        elif level > 0:
            # The standard TUM loader already fills small holes. Smooth its
            # remaining valid measurements with an edge-preserving bilateral
            # filter; leave invalid pixels masked to avoid inventing geometry.
            valid = np.isfinite(d) & (d > 0)
            smoothed = cv2.bilateralFilter(d, 5, 0.05, 2)
            d = np.where(valid, (1 - level) * d + level * smoothed, 0)
        rec = dict(f)
        rec["depth"] = torch.from_numpy(np.ascontiguousarray(d)).to(f["depth"].device)
        output.append(rec)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["replica_office0", "tum_fr2_xyz"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "depth_intervention")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    for scene in args.scenes:
        frames, intr = load_benchmark_sequence(scene, args.n_frames + 1, 240, 320, "cuda")
        levels = (0.0, 0.15, 0.30) if scene.startswith("replica_") else (0.0, 0.5, 1.0)
        for level in levels:
            for seed in args.seeds:
                altered = modify_depth(frames, scene, level, seed)
                hole_rate = float(np.mean([float((f["depth"] <= 0).float().mean().item())
                                           for f in altered]))
                for policy in ("error_only", "error_influence_temporal"):
                    dest = args.out_dir / scene / f"level{level:g}_{policy}_seed{seed}.json"
                    if dest.exists() and not args.force:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    cfg = build_pipeline_config(policy, seed, 15, 320, 240, "cuda",
                                                throttling=(policy == "error_influence_temporal"))
                    summary, logs = run_policy_trajectory(policy, seed, altered, intr,
                        15, "cuda", 320, 240, custom_config=cfg)
                    rec = {"scene": scene, "level": level, "seed": seed, "policy": policy,
                           "intervention_method": "replica_random_holes_plus_gaussian_depth_noise" if scene.startswith("replica_") else "tum_bilateral_valid_depth_only",
                           "post_intervention_hole_rate": hole_rate,
                           "summary": summary, "frames": logs, "config": cfg}
                    temp = dest.with_suffix(".tmp")
                    temp.write_text(json.dumps(rec, indent=2))
                    temp.replace(dest)
                    print(f"{scene} level={level} seed={seed} {policy}: "
                          f"{summary['mean_psnr']:.3f} dB", flush=True)


if __name__ == "__main__":
    main()
