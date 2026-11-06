#!/usr/bin/env python3
"""Measured-compute comparison: budget multiplier x {free growth, spawn cap x0.2}.

Raw-error selection. For each regime the modeled optimization budget is 15 ms x {1, 2, 4}
with the spawn allowance fixed at 15 ms. Per-frame wall-clock (optimization and end-to-end)
is recorded so quality can be compared at equal measured time. Runs execute sequentially on an
otherwise idle GPU; timings are indicative of this laptop GPU only.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_throttle_ablation as abl  # noqa: E402
from experiments.run_dilution_factorial import summarize  # noqa: E402
from experiments.run_phase13_frozen_benchmark import load_phase10_sequence, run_policy_trajectory  # noqa: E402

REGIMES = {"free": "error_off", "spawn": "spawn_cap_always20"}
MULTS = (1, 2, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0", "replica_office1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "compute_matched")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    for scene in args.scenes:
        frames, intr = load_phase10_sequence(scene_name=scene, n_frames=args.n_frames + 1, H=240, W=320, device=args.device)
        for seed in args.seeds:
            for regime, base in REGIMES.items():
                for m in MULTS:
                    name = f"{regime}_x{m}"
                    target = args.out_dir / scene / f"{name}_seed{seed}.json"
                    if target.exists():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    policy, cfg = abl.config_for(base, seed, args.budget_ms * m, args.device)
                    if m != 1:
                        cfg["scheduler"]["densify_budget_override_ms"] = args.budget_ms
                    print(f"RUN {scene} {name} seed={seed}", flush=True)
                    summary, logs = run_policy_trajectory(policy=policy, seed=seed, frames=frames, intrinsics=intr,
                        budget_ms=args.budget_ms, device=args.device, W=320, H=240, custom_config=cfg)
                    res = summarize(summary, logs)
                    res["mean_frame_ms"] = float(sum(r["frame_time_ms"] for r in logs) / len(logs))
                    res["mean_opt_ms"] = float(sum(r["opt_time_ms"] for r in logs) / len(logs))
                    res.update({"regime": regime, "budget_multiplier": m})
                    tmp = target.with_suffix(".tmp")
                    tmp.write_text(json.dumps({"scene": scene, "seed": seed, "condition": name, "summary": res}))
                    tmp.replace(target)
                    print(f"DONE {scene} {name} seed={seed}: PSNR={res['mean_psnr']:.3f} frame={res['mean_frame_ms']:.0f}ms "
                          f"opt={res['mean_opt_ms']:.0f}ms map={res['N_final']}", flush=True)


if __name__ == "__main__":
    main()
