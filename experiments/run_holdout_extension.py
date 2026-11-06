#!/usr/bin/env python3
"""Held-out-view check of the budget and throttling controls.

Every 8th frame is rendered but never mapped or optimized (a same-trajectory novel-view proxy,
not an independent trajectory). Conditions reuse run_throttle_ablation; the proportional-budget
wrapper is active because that module installs it. Each run is written immediately.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_throttle_ablation as abl  # noqa: E402
from experiments.run_holdout_benchmark import run_holdout_trajectory  # noqa: E402
from experiments.run_phase13_frozen_benchmark import load_phase10_sequence  # noqa: E402

CONDITIONS = ("error_off", "error_coverage", "spawn_cap_always20", "error_budget_prop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0", "replica_office1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--conditions", nargs="+", choices=abl.CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--holdout_every", type=int, default=8)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "holdout_extension")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    for scene in args.scenes:
        frames, intr = load_phase10_sequence(scene_name=scene, n_frames=args.n_frames + 1, H=240, W=320, device=args.device)
        for seed in args.seeds:
            for cond in args.conditions:
                target = args.out_dir / scene / f"{cond}_seed{seed}.json"
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                policy, cfg = abl.config_for(cond, seed, args.budget_ms, args.device)
                print(f"RUN {scene} {cond} seed={seed}", flush=True)
                s = run_holdout_trajectory(policy, seed, frames, intr, args.budget_ms, args.device, 320, 240,
                                           args.holdout_every, cfg)
                s.update({"scene": scene, "condition": cond, "n_frames_loaded": len(frames),
                          "holdout_every": args.holdout_every})
                tmp = target.with_suffix(".tmp")
                tmp.write_text(json.dumps(s, indent=1))
                tmp.replace(target)
                print(f"DONE {scene} {cond} seed={seed}: train={s['train_psnr']:.2f} holdout={s['holdout_psnr']:.2f} "
                      f"gap={s['gen_gap_db']:+.2f} N={s['N_final']}", flush=True)


if __name__ == "__main__":
    main()
