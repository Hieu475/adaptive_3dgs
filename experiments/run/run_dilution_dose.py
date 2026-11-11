#!/usr/bin/env python3
"""Budget-per-Gaussian dose-response: does PSNR depend on budget / map size?

Raw-error selection, no throttling. Crosses the modeled optimization budget
(multiplier m in {1, 2, 4} x budget_ms) with a map regime (free growth vs hard cap).
The spawn allowance is held at budget_ms in every cell (same convention as the x4
control), so extra compute cannot buy extra splats. If compute dilution is a
per-Gaussian allocation effect, quality should be a common function of
budget-per-Gaussian across the free and capped regimes.

Each cell is written immediately, so interrupted runs resume.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run.run_dilution_factorial import config_for as base_config_for  # noqa: E402
from experiments.run.run_dilution_factorial import summarize  # noqa: E402
from experiments.run.run_frozen_benchmark import (  # noqa: E402
    load_benchmark_sequence, run_policy_trajectory,
)
from src.evaluation.reproducibility import create_provenance_manifest  # noqa: E402

MULTIPLIERS = (1, 2, 4)
REGIMES = {"free": None, "cap": 23000}


def cell_name(regime, mult):
    return f"{regime}_x{mult}"


def config_for(regime, mult, seed, budget_ms, device):
    policy, cfg = base_config_for("error_off", seed, budget_ms * mult, device)
    if mult != 1:
        cfg["scheduler"]["densify_budget_override_ms"] = budget_ms
    if REGIMES[regime] is not None:
        cfg["gaussian"]["max_gaussians"] = REGIMES[regime]
    return policy, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 50)))
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "dilution" / "dilution_dose")
    ap.add_argument("--cap", type=int, default=23000, help="hard Gaussian cap for the capped regime")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    REGIMES["cap"] = args.cap
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(create_provenance_manifest(
            config={"scenes": args.scenes, "seeds": args.seeds, "multipliers": MULTIPLIERS,
                    "regimes": REGIMES, "n_frames": args.n_frames, "budget_ms": args.budget_ms},
            dataset_name=",".join(args.scenes),
            renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
            extra_metadata={"experiment": "dilution_dose", "status": "exploratory",
                            "spawn_allowance": "fixed at budget_ms in every cell"}), indent=2))
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    for scene in args.scenes:
        frames, intrinsics = load_benchmark_sequence(scene_name=scene, n_frames=args.n_frames + 1,
                                                   H=240, W=320, device=args.device)
        for seed in args.seeds:
            for regime in REGIMES:
                for mult in MULTIPLIERS:
                    name = cell_name(regime, mult)
                    target = args.out_dir / scene / f"{name}_seed{seed}.json"
                    if target.exists() and not args.force:
                        print(f"skip {target}", flush=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    policy, cfg = config_for(regime, mult, seed, args.budget_ms, args.device)
                    print(f"RUN {scene} {name} seed={seed}", flush=True)
                    summary, logs = run_policy_trajectory(
                        policy=policy, seed=seed, frames=frames, intrinsics=intrinsics,
                        budget_ms=args.budget_ms, device=args.device, W=320, H=240, custom_config=cfg)
                    result = summarize(summary, logs)
                    n = [r["n_gaussians"] for r in logs]
                    result["budget_multiplier"] = mult
                    result["regime"] = regime
                    result["mean_budget_per_gaussian_us"] = float(
                        sum(args.budget_ms * mult * 1000.0 / max(x, 1) for x in n) / len(n))
                    record = {"scene": scene, "seed": seed, "condition": name, "config": cfg,
                              "summary": result,
                              "frames": [{k: r[k] for k in ("frame", "psnr", "depth_l1", "n_gaussians",
                                                            "selected_old", "population_old")} for r in logs]}
                    temp = target.with_suffix(".tmp")
                    temp.write_text(json.dumps(record))
                    temp.replace(target)
                    print(f"DONE {scene} {name} seed={seed}: PSNR={summary['mean_psnr']:.3f} "
                          f"map={summary['N_final']} B/N={result['mean_budget_per_gaussian_us']:.3f}us "
                          f"mature_cov={result['mature_update_coverage']:.3f}", flush=True)


if __name__ == "__main__":
    main()
