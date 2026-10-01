#!/usr/bin/env python3
"""Per-frame sensor-adaptive routing benchmark (no scene labels).

Each frame's v3 noise score (from loader raw stats) routes that frame's
selection policy + kappa inside pipeline.process_frame. No scene-level
switching: the same binary runs on every substrate.

Validation bar: per-frame adaptive must approximately recover the
scene-level oracle picks — OURS-level on TUM fr2_xyz, error_only-level on
Replica office0 — without being told which scene it is on.

Usage:
  python experiments/run_adaptive_policy.py --scenes tum_fr2_xyz replica_office0 \\
      --n_frames 150 --seeds 42 43 44
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_phase13_frozen_benchmark import load_phase10_sequence, build_pipeline_config
from research.pipeline import OnlineReconstructionPipeline
from research.noise_adaptive import frame_noise_score_v3
from research.reproducibility import set_seed


def run_adaptive(policy_base, seed, frames, intr, budget_ms=15.0, device="cuda",
                 W=320, H=240):
    set_seed(seed)
    cfg = build_pipeline_config(policy=policy_base, seed=seed, budget_ms=budget_ms,
                                W=W, H=H, device=device)
    cfg["adaptive_routing"] = {"enabled": True}
    pipe = OnlineReconstructionPipeline(config=cfg, device=device)
    pipe.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                    intrinsics=intr, pose=frames[0].get("pose", torch.eye(4)))
    psnrs, routes = [], []
    for t in range(1, len(frames)):
        f = frames[t]
        nz = frame_noise_score_v3(f.get("raw_holes", 0.25),
                                  f.get("raw_edge", 0.05))["noise"]
        m = pipe.process_frame(rgb=f["rgb"], depth=f["depth"],
                               gt_pose=f.get("pose"), frame_noise=nz)
        psnrs.append(m["psnr"]); routes.append(m["routed_policy"])
    uniq, cnt = np.unique(routes, return_counts=True)
    return {"mean_psnr": float(np.mean(psnrs)),
            "route_frac": {u: float(c / len(routes)) for u, c in zip(uniq, cnt)}}


def main():
    ap = argparse.ArgumentParser(description="Per-frame adaptive routing benchmark")
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "replica_office0"])
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="results/phase14_adaptive_perframe")
    a = ap.parse_args()
    if a.device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.70, 0)
        except Exception:
            pass
    out = REPO_ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240
    rows = []
    for sc in a.scenes:
        nf = min(a.n_frames, 99) if sc.startswith("replica_") else a.n_frames
        frames, intr = load_phase10_sequence(scene_name=sc, n_frames=nf + 1,
                                             H=H, W=W, device=a.device)
        for seed in a.seeds:
            s = run_adaptive("ours", seed, frames, intr, a.budget_ms, a.device, W, H)
            row = {"scene": sc, "seed": seed, **s}
            rows.append(row)
            print(f"{sc} seed={seed}: adaptive PSNR={s['mean_psnr']:.2f} routes={s['route_frac']}",
                  flush=True)
    json.dump({"metadata": {"scenes": a.scenes, "n_frames": a.n_frames, "seeds": a.seeds,
                            "note": "per-frame routing, no scene labels; exploratory n=3"},
               "rows": rows}, open(out / "adaptive_perframe.json", "w"), indent=1)
    print("wrote", out / "adaptive_perframe.json")


if __name__ == "__main__":
    main()
