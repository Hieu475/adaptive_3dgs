#!/usr/bin/env python3
"""Pilot a continuous growth-control x stale-mature-rescue factorial."""
import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_dilution_factorial import config_for
from experiments.run_phase13_frozen_benchmark import load_phase10_sequence
from research.pipeline import OnlineReconstructionPipeline
from research.reproducibility import create_provenance_manifest, set_seed
from research.gaussian_repr import GaussianState

def run_arm(scene, seed, growth, allocation, frames, intrinsics, args):
    set_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
    condition = "error_coverage" if growth == "throttled" else "error_off"
    policy, cfg = config_for(condition, seed, args.budget_ms, args.device)
    cfg["training"]["optimization_control"] = {
        "micro_updates_per_frame": args.micro_updates,
        "mature_age_frames": 10,
        "stale_after_frames": args.stale_after_frames,
        "capture_age_diagnostics": True,
    }
    if allocation == "stale_rescue":
        cfg["training"]["optimization_control"]["stale_update_fraction"] = args.stale_fraction
    cfg["scheduler"]["enable_backlog_throttling"] = False
    cfg["system"]["synchronized_stage_timing"] = True

    pipeline = OnlineReconstructionPipeline(config=cfg, device=args.device)
    pipeline.initialize(
        rgb=frames[0]["rgb"], depth=frames[0]["depth"], intrinsics=intrinsics,
        pose=frames[0].get("pose", torch.eye(4, device=torch.device(args.device))),
    )
    logs = []
    for frame_idx in range(1, len(frames)):
        n_before = int(pipeline.gaussian_model.num_gaussians)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        m = pipeline.process_frame(
            rgb=frames[frame_idx]["rgb"], depth=frames[frame_idx]["depth"],
            gt_pose=frames[frame_idx].get("pose"),
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - start) * 1000.0
        active = int((pipeline.gaussian_model._state != GaussianState.PRUNED).sum().item())
        logs.append({
            "frame": int(frame_idx),
            "psnr": float(m["psnr"]),
            "psnr_pre": float(m.get("psnr_pre", m["psnr"])),
            "psnr_post": float(m.get("psnr_post", m["psnr"])),
            "ssim": float(m["ssim"]),
            "depth_l1": float(m["depth_l1"]),
            "depth_l1_post": float(m.get("depth_l1_post", m["depth_l1"])),
            "n_gaussians": int(m["n_gaussians"]),
            "n_active_gaussians": active,
            "n_spawned": int(m.get("n_spawned", 0)),
            "growth_pressure": float(m.get("growth_pressure", 0.0)),
            "n_micro_updates": int(m.get("n_micro_updates", 0)),
            "mature_eligible": int(m.get("mature_eligible", 0)),
            "mature_updated": int(m.get("mature_updated", 0)),
            "mature_update_coverage": float(m.get("mature_update_coverage", 0.0)),
            "stale_after_frames": int(m.get("stale_after_frames", args.stale_after_frames)),
            "stale_mature_eligible": int(m.get("stale_mature_eligible", 0)),
            "stale_mature_updated": int(m.get("stale_mature_updated", 0)),
            "stale_mature_update_coverage": float(m.get("stale_mature_update_coverage", 0.0)),
            "stale_mature_micro_updates": int(m.get("stale_mature_micro_updates", 0)),
            "stale_quota_micro_updates": m.get("stale_quota_micro_updates"),
            "opt_stage_sync_ms": float(m.get("opt_stage_sync_ms", 0.0)),
            "frame_wall_sync_ms": wall_ms,
            "peak_vram_mb": float(torch.cuda.max_memory_allocated(args.device) / 1048576)
                if args.device.startswith("cuda") else 0.0,
            "map_growth_this_frame": max(0, int(m["n_gaussians"]) - n_before),
        })

    eval_indices = sorted(set(i for i in args.eval_frames if 0 <= i < len(frames)))
    eval_rows = []
    for i in eval_indices:
        result = pipeline.eval_frame(
            rgb=frames[i]["rgb"], depth=frames[i]["depth"],
            pose=frames[i].get("pose", pipeline.current_pose),
        )
        eval_rows.append({"frame": int(i), "psnr": float(result["psnr"]),
                          "ssim": float(result["ssim"]), "depth_l1": float(result["depth_l1"])})
    pipeline.cleanup()
    del pipeline
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    def mean(key):
        return float(np.mean([r[key] for r in logs])) if logs else 0.0
    summary = {
        "scene": scene, "seed": int(seed), "growth": growth, "allocation": allocation,
        "policy": policy, "n_processed_frames": len(logs),
        "mean_psnr": mean("psnr"), "final_psnr": logs[-1]["psnr"] if logs else None,
        "mean_ssim": mean("ssim"), "mean_depth_l1": mean("depth_l1"),
        "final_gaussians": logs[-1]["n_gaussians"] if logs else None,
        "total_spawned": int(sum(r["n_spawned"] for r in logs)),
        "mean_growth_pressure": mean("growth_pressure"),
        "mean_mature_coverage": mean("mature_update_coverage"),
        "mean_stale_mature_coverage": mean("stale_mature_update_coverage"),
        "stale_quota_compliance": float(np.mean([
            abs((r["stale_mature_micro_updates"] / max(r["n_micro_updates"], 1)) - args.stale_fraction) < 1e-6
            for r in logs])) if allocation == "stale_rescue" and logs else None,
        "mean_opt_stage_sync_ms": mean("opt_stage_sync_ms"),
        "mean_frame_wall_sync_ms": mean("frame_wall_sync_ms"),
        "p95_frame_wall_sync_ms": float(np.percentile([r["frame_wall_sync_ms"] for r in logs], 95)) if logs else None,
        "peak_vram_mb": max([r["peak_vram_mb"] for r in logs], default=0.0),
        "fixed_eval_mean_psnr": float(np.mean([r["psnr"] for r in eval_rows])) if eval_rows else None,
        "fixed_eval_mean_depth_l1": float(np.mean([r["depth_l1"] for r in eval_rows])) if eval_rows else None,
    }
    return summary, logs, eval_rows, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=["replica_office1"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[211])
    parser.add_argument("--n_frames", type=int, default=120)
    parser.add_argument("--budget_ms", type=float, default=15.0)
    parser.add_argument("--micro_updates", type=int, default=5000)
    parser.add_argument("--stale_fraction", type=float, default=0.50)
    parser.add_argument("--stale_after_frames", type=int, default=10)
    parser.add_argument("--eval_frames", nargs="+", type=int, default=[20, 40, 60, 80, 100, 120])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.stale_fraction <= 1.0:
        raise ValueError("stale_fraction must lie in [0, 1]")
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = create_provenance_manifest(
            config={"scenes": args.scenes, "seeds": args.seeds, "n_frames": args.n_frames,
                    "growth_arms": ["throttled", "unthrottled"],
                    "allocation_arms": ["global", "stale_rescue"],
                    "micro_updates_per_frame": args.micro_updates,
                    "stale_fraction": args.stale_fraction,
                    "stale_after_frames": args.stale_after_frames,
                    "fixed_eval_frames": args.eval_frames,
                    "budget_ms_modeled": args.budget_ms},
            dataset_name=",".join(args.scenes),
            renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
            extra_metadata={"experiment": "growth_starvation_trajectory_factorial",
                            "status": "exploratory feasibility pilot",
                            "pose_convention": "Phase 10 loader; GT W2C for TUM and Replica"})
        manifest_path.write_text(json.dumps(manifest, indent=2))

    for scene in args.scenes:
        frames, intrinsics = load_phase10_sequence(scene_name=scene,
            n_frames=args.n_frames + 1, H=240, W=320, device=args.device)
        for seed in args.seeds:
            for growth in ("unthrottled", "throttled"):
                for allocation in ("global", "stale_rescue"):
                    name = f"{scene}_seed{seed}_{growth}_{allocation}.json"
                    target = args.out_dir / name
                    if target.exists() and not args.force:
                        print(f"skip {target}", flush=True)
                        continue
                    print(f"RUN {scene} seed={seed} growth={growth} allocation={allocation}", flush=True)
                    summary, logs, eval_rows, cfg = run_arm(
                        scene, seed, growth, allocation, frames, intrinsics, args)
                    record = {"summary": summary, "frames": logs, "fixed_eval": eval_rows,
                              "config": cfg}
                    temp = target.with_suffix(".tmp")
                    temp.write_text(json.dumps(record, indent=2))
                    temp.replace(target)
                    print(f"DONE {target.name}: mean_PSNR={summary['mean_psnr']:.3f}, "
                          f"stale_cov={summary['mean_stale_mature_coverage']:.4f}, "
                          f"spawned={summary['total_spawned']}", flush=True)

    source_paths = ["experiments/run_growth_starvation_trajectory_factorial.py",
                    "experiments/run_dilution_factorial.py",
                    "experiments/run_phase13_frozen_benchmark.py",
                    "research/pipeline.py", "research/state_store.py",
                    "research/gaussian_repr.py", "research/reproducibility.py"]
    source_hashes = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                     for rel in source_paths}
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip().splitlines()
    except Exception:
        git_head, dirty = None, ["unavailable"]
    provenance = {"experiment": "growth_starvation_trajectory_factorial",
                  "git_head": git_head, "git_dirty": bool(dirty),
                  "git_status_entry_count": len(dirty), "source_sha256": source_hashes,
                  "scenes": args.scenes, "seeds": args.seeds,
                  "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(args.out_dir / "provenance.json")


if __name__ == "__main__":
    main()
