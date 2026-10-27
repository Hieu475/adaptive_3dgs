#!/usr/bin/env python3
"""Corrected-pose selector x throttle study and two dilution controls.

Each scene/seed/condition is saved immediately, so interrupted runs resume.
The fourfold budget control changes the modeled optimization budget and also
the densification allowance; interpret its contrast with that caveat.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_phase13_frozen_benchmark import (
    build_pipeline_config, load_phase10_sequence, run_policy_trajectory,
)
from research.reproducibility import create_provenance_manifest

SELECTORS = {"error": "error_only", "temporal": "error_influence_temporal"}
THROTTLES = ("off", "coverage", "coverage_backlog")
CONDITIONS = tuple(f"{s}_{t}" for s in SELECTORS for t in THROTTLES) + (
    "error_budget4", "error_cap23k", "error_cap39k",
)


def config_for(condition, seed, budget_ms, device):
    selector = condition.split("_")[0]
    policy = SELECTORS[selector]
    coverage = condition.endswith("coverage") or condition.endswith("coverage_backlog")
    backlog = condition.endswith("coverage_backlog")
    effective_budget = budget_ms * (4 if condition == "error_budget4" else 1)
    cfg = build_pipeline_config(policy, seed, effective_budget, 320, 240, device,
                                throttling=coverage, use_learned_utility=False)
    cfg["scheduler"]["enable_backlog_throttling"] = backlog
    cfg["scheduler"]["max_warmup_queue"] = 500 if backlog else 999999
    if condition == "error_budget4":
        cfg["scheduler"]["densify_budget_override_ms"] = budget_ms
    if condition == "error_cap23k":
        cfg["gaussian"]["max_gaussians"] = 23000
    if condition == "error_cap39k":
        cfg["gaussian"]["max_gaussians"] = 39000
    return policy, cfg


def summarize(summary, logs):
    result = dict(summary)
    for field in ("n_spawned", "loss", "mean_opacity", "selected_young",
                  "selected_old", "population_young", "population_old",
                  "budget_young_us", "budget_old_us", "n_backlog"):
        result[f"mean_{field}"] = float(np.mean([r[field] for r in logs]))
    young = sum(r["budget_young_us"] for r in logs)
    old = sum(r["budget_old_us"] for r in logs)
    result["young_budget_share"] = young / (young + old) if young + old else 0.0
    # Age groups in this protocol define mature as age >= 10 frames.
    eligible = sum(r["population_old"] for r in logs)
    updated = sum(r["selected_old"] for r in logs)
    result["mature_update_coverage"] = updated / eligible if eligible else 0.0
    result["mature_age_threshold_frames"] = 10
    result["growth_pressure_mean"] = float(np.mean([
        r["n_new"] / max(r["n_gaussians"], 1) for r in logs
    ]))
    # Per-frame PSNR change is an observational proxy, not a causal action effect.
    returns = []
    for previous, current in zip(logs, logs[1:]):
        elapsed_ms = float(current["opt_time_ms"])
        if elapsed_ms > 0:
            returns.append((float(current["psnr"]) - float(previous["psnr"])) / elapsed_ms)
    result["compute_return_db_per_opt_ms_mean"] = float(np.mean(returns)) if returns else 0.0
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "tum_fr1_xyz"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 50)))
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "dilution_factorial")
    ap.add_argument("--force", action="store_true", help="Recompute and atomically replace matching runs")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = create_provenance_manifest(
            config={"scenes": args.scenes, "seeds": args.seeds,
                    "conditions": args.conditions, "n_frames": args.n_frames,
                    "budget_ms": args.budget_ms, "device": args.device,
                    "age_mature_frames": 10,
                    "compute_return": "delta per-frame PSNR / optimization kernel ms (observational proxy)"},
            dataset_name=",".join(args.scenes),
            renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
            extra_metadata={"experiment": "dilution_factorial", "status": "exploratory",
                            "pose_convention": "W2C via inverse TUM C2W groundtruth",
                            "seed_list": args.seeds,
                            "matched_map_size": False, "matched_spawn_count": False},
        )
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    for scene in args.scenes:
        frames, intrinsics = load_phase10_sequence(scene_name=scene,
            n_frames=args.n_frames + 1, H=240, W=320, device=args.device)
        for seed in args.seeds:
            for condition in args.conditions:
                target = args.out_dir / scene / f"{condition}_seed{seed}.json"
                if target.exists() and not args.force:
                    print(f"skip {target}", flush=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                policy, cfg = config_for(condition, seed, args.budget_ms, args.device)
                print(f"RUN {scene} {condition} seed={seed}", flush=True)
                summary, logs = run_policy_trajectory(policy=policy, seed=seed,
                    frames=frames, intrinsics=intrinsics, budget_ms=args.budget_ms,
                    device=args.device, W=320, H=240, custom_config=cfg)
                run_manifest = create_provenance_manifest(
                    config=cfg, dataset_name=scene,
                    renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
                    extra_metadata={"experiment": "dilution_factorial", "condition": condition,
                                    "seed": seed, "pose_convention": "W2C via inverse TUM C2W groundtruth",
                                    "status": "exploratory"},
                )
                record = {"scene": scene, "seed": seed, "condition": condition,
                          "manifest": run_manifest, "config": cfg,
                          "summary": summarize(summary, logs), "frames": logs}
                temp = target.with_suffix(".tmp")
                temp.write_text(json.dumps(record, indent=2))
                temp.replace(target)
                print(f"DONE {scene} {condition} seed={seed}: "
                      f"PSNR={summary['mean_psnr']:.3f}, depth={summary['mean_depth_l1']:.4f}, "
                      f"map={summary['N_final']}", flush=True)


if __name__ == "__main__":
    main()
