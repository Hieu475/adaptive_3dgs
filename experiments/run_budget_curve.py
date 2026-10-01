#!/usr/bin/env python3
"""Matched-budget curve (Gate-3 question, corrected substrate).

Supersedes experiments/run_phase6_budget_sweep.py (header: unfair non-unified
budgets, historical reference only). Here EVERY policy shares the same modeled
per-frame optimization budget; only the budget value varies.

Sweep: budgets {5,10,15,20,30} ms x policies {ours, error_only} x seeds.
Scene: tum_fr2_xyz, 30 frames (pilot horizons; 150f reserved for the main
confirmation). Output: results/phase14_budget_curve/budget_curve.json.

Usage:
  python experiments/run_budget_curve.py --seeds 42 43 44 --n_frames 30
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_phase13_frozen_benchmark import (
    load_phase10_sequence, build_pipeline_config, run_policy_trajectory,
)
from research.reproducibility import set_seed  # noqa: F401 (seed protocol parity)


def main():
    ap = argparse.ArgumentParser(description="Matched-budget quality curve")
    ap.add_argument("--scene", default="tum_fr2_xyz")
    ap.add_argument("--n_frames", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--budgets", type=float, nargs="+", default=[5.0, 10.0, 15.0, 20.0, 30.0])
    ap.add_argument("--policies", nargs="+", default=["ours", "error_only"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="results/phase14_budget_curve")
    a = ap.parse_args()
    if a.device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.70, 0)
        except Exception:
            pass
    out = REPO_ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240
    frames, intr = load_phase10_sequence(scene_name=a.scene, n_frames=a.n_frames + 1,
                                         H=H, W=W, device=a.device)
    rows = []
    for b in a.budgets:
        for pol in a.policies:
            ps, ns = [], []
            for seed in a.seeds:
                cfg = build_pipeline_config(policy=pol, seed=seed, budget_ms=b,
                                            W=W, H=H, device=a.device)
                s, _ = run_policy_trajectory(policy=pol, seed=seed, frames=frames,
                                             intrinsics=intr, budget_ms=b, device=a.device,
                                             W=W, H=H, custom_config=cfg)
                ps.append(s["mean_psnr"]); ns.append(s["N_final"])
            row = {"budget_ms": b, "policy": pol,
                   "mean_psnr": float(np.mean(ps)), "std_psnr": float(np.std(ps)),
                   "mean_N": float(np.mean(ns))}
            rows.append(row)
            print(f"B={b:5.1f}ms {pol:12s} PSNR={row['mean_psnr']:.2f}+-{row['std_psnr']:.2f} N={row['mean_N']:.0f}",
                  flush=True)
    json.dump({"metadata": {"scene": a.scene, "n_frames": a.n_frames, "seeds": a.seeds,
                            "budgets": a.budgets, "policies": a.policies,
                            "note": "matched modeled budgets; exploratory n=3"},
               "rows": rows}, open(out / "budget_curve.json", "w"), indent=1)
    print("wrote", out / "budget_curve.json")


if __name__ == "__main__":
    main()
