#!/usr/bin/env python3
"""Decompose coverage throttling and compare it with simpler growth controls.

Coverage throttling does two things: (a) it multiplies the densification error thresholds
by up to 1.5, and (b) it cuts the per-frame spawn allowance to 20% once coverage saturates
and the view is settled. Raw-error selection throughout; conditions:

  error_off             no throttling (baseline)
  error_coverage        full coverage throttling (threshold multiplier + spawn cap)
  cov_spawn_only        spawn cap only (threshold multiplier fixed to 1)
  cov_thresh_only       threshold multiplier only (spawn cap disabled)
  spawn_cap_always20    spawn allowance x0.2 whenever the view is settled, no coverage gate
  spawn_cap_always50    same with x0.5
  error_budget_prop     map-proportional optimization budget, no throttling
  cov_plus_prop         full coverage throttling plus the map-proportional budget

Each run is written immediately, so interrupted runs resume.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_dilution_extension as ext  # noqa: E402  (installs the proportional-budget wrapper)
from experiments.run_dilution_factorial import config_for as base_config_for  # noqa: E402
from experiments.run_dilution_factorial import summarize  # noqa: E402
from experiments.run_frozen_benchmark import (  # noqa: E402
    load_benchmark_sequence, run_policy_trajectory,
)
from research.reproducibility import create_provenance_manifest  # noqa: E402

CONDITIONS = ("error_off", "error_coverage", "cov_spawn_only", "cov_thresh_only",
              "spawn_cap_always20", "spawn_cap_always50", "error_budget_prop", "cov_plus_prop")


def config_for(cond, seed, budget_ms, device):
    if cond in ("error_off", "error_coverage", "error_budget_prop"):
        return ext.config_for(cond, seed, budget_ms, device)
    policy, cfg = base_config_for("error_coverage", seed, budget_ms, device)
    d = cfg["densification"]
    if cond == "cov_spawn_only":
        d["throttle_error_threshold_mult"] = 1.0
    elif cond == "cov_thresh_only":
        d["throttle_factor"] = 1.0
    elif cond in ("spawn_cap_always20", "spawn_cap_always50"):
        d["throttle_coverage_threshold"] = 0.0   # coverage gate always open
        d["throttle_error_threshold_mult"] = 1.0
        d["throttle_factor"] = 0.2 if cond.endswith("20") else 0.5
    elif cond == "cov_plus_prop":
        cfg["scheduler"]["budget_prop_to_map"] = True
        cfg["scheduler"]["densify_budget_override_ms"] = budget_ms
    return policy, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0", "replica_office1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "throttle_ablation")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mp = args.out_dir / "manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps(create_provenance_manifest(
            config={"scenes": args.scenes, "seeds": args.seeds, "conditions": args.conditions,
                    "n_frames": args.n_frames, "budget_ms": args.budget_ms},
            dataset_name=",".join(args.scenes),
            renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
            extra_metadata={"experiment": "throttle_ablation", "status": "exploratory"}), indent=2))
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    for scene in args.scenes:
        frames, intrinsics = load_benchmark_sequence(scene_name=scene, n_frames=args.n_frames + 1,
                                                   H=240, W=320, device=args.device)
        for seed in args.seeds:
            for cond in args.conditions:
                target = args.out_dir / scene / f"{cond}_seed{seed}.json"
                if target.exists() and not args.force:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                policy, cfg = config_for(cond, seed, args.budget_ms, args.device)
                ext.BUDGET_SCALES.clear()
                print(f"RUN {scene} {cond} seed={seed}", flush=True)
                summary, logs = run_policy_trajectory(policy=policy, seed=seed, frames=frames,
                    intrinsics=intrinsics, budget_ms=args.budget_ms, device=args.device, W=320, H=240,
                    custom_config=cfg)
                res = summarize(summary, logs)
                if ext.BUDGET_SCALES:
                    res["mean_budget_scale"] = sum(ext.BUDGET_SCALES) / len(ext.BUDGET_SCALES)
                res["mean_n_spawned_per_frame"] = res.get("mean_n_spawned")
                rec = {"scene": scene, "seed": seed, "condition": cond, "config": cfg, "summary": res,
                       "frames": [{k: r[k] for k in ("frame", "psnr", "depth_l1", "n_gaussians", "n_spawned")} for r in logs]}
                tmp = target.with_suffix(".tmp")
                tmp.write_text(json.dumps(rec))
                tmp.replace(target)
                print(f"DONE {scene} {cond} seed={seed}: PSNR={summary['mean_psnr']:.3f} depth={summary['mean_depth_l1']:.4f} "
                      f"map={summary['N_final']} spawn/frame={res['mean_n_spawned']:.0f}", flush=True)


if __name__ == "__main__":
    main()
