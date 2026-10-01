#!/usr/bin/env python3
"""Holdout (novel-view proxy) + ATE benchmark on the corrected substrate.

Every --holdout_every-th frame is evaluated via pipeline.eval_frame (no mapping).
All other frames go through process_frame. GT poses (tracking decoupled).

Usage:
  python experiments/run_holdout_benchmark.py --scene tum_fr2_xyz --n_frames 30 \\
      --policies ours error_only --seeds 42 --holdout_every 8 --out_dir results/holdout_smoke
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_phase13_frozen_benchmark import (
    load_phase10_sequence, build_pipeline_config, run_policy_trajectory as _unused,
)
from research.pipeline import OnlineReconstructionPipeline
from research.holdout_eval import compute_ate, summarize_holdout
from research.reproducibility import set_seed


def run_holdout_trajectory(policy, seed, frames, intrinsics, budget_ms=15.0,
                           device="cuda", W=320, H=240, holdout_every=8, custom_config=None):
    set_seed(seed)
    cfg = custom_config or build_pipeline_config(policy=policy, seed=seed,
                                                 budget_ms=budget_ms, W=W, H=H, device=device)
    pipe = OnlineReconstructionPipeline(config=cfg, device=device)
    pipe.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                    intrinsics=intrinsics, pose=frames[0].get("pose", torch.eye(4)))
    train_p, hold_p, hold_s = [], [], []
    est_poses, gt_poses = [frames[0].get("pose", torch.eye(4))], [frames[0].get("pose", torch.eye(4))]
    for t in range(1, len(frames)):
        f = frames[t]
        gt = f.get("pose", torch.eye(4))
        if t % holdout_every == 0:
            m = pipe.eval_frame(rgb=f["rgb"], depth=f["depth"], pose=gt)
            hold_p.append(m["psnr"]); hold_s.append(m["ssim"])
        else:
            m = pipe.process_frame(rgb=f["rgb"], depth=f["depth"], gt_pose=gt)
            train_p.append(m["psnr"])
        est_poses.append(pipe.current_pose.detach().cpu() if isinstance(pipe.current_pose, torch.Tensor) else pipe.current_pose)
        gt_poses.append(gt)
    summ = summarize_holdout(train_p, hold_p)
    summ["holdout_ssim"] = float(np.mean(hold_s)) if hold_s else 0.0
    summ.update(compute_ate(est_poses, gt_poses))
    summ.update({"policy": policy, "seed": seed, "N_final": pipe.gaussian_model.num_gaussians})
    return summ


def main():
    ap = argparse.ArgumentParser(description="Holdout + ATE benchmark")
    ap.add_argument("--scene", default="tum_fr2_xyz")
    ap.add_argument("--n_frames", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--policies", nargs="+", default=["ours", "error_only"])
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--holdout_every", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="results/holdout_smoke")
    a = ap.parse_args()
    if a.device == "cuda" and torch.cuda.is_available():
        try: torch.cuda.set_per_process_memory_fraction(0.70, 0)
        except Exception: pass
    out = REPO_ROOT / a.out_dir; out.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240
    frames, intr = load_phase10_sequence(scene_name=a.scene, n_frames=a.n_frames + 1, H=H, W=W, device=a.device)
    rows = []
    for pol in a.policies:
        for seed in a.seeds:
            cfg = build_pipeline_config(policy=pol, seed=seed, budget_ms=a.budget_ms, W=W, H=H, device=a.device)
            s = run_holdout_trajectory(pol, seed, frames, intr, a.budget_ms, a.device, W, H, a.holdout_every, cfg)
            rows.append(s)
            print(f"{pol} seed={seed}: train={s['train_psnr']:.2f} holdout={s['holdout_psnr']:.2f} "
                  f"gap={s['gen_gap_db']:+.2f} ATE={s['ate_rmse_m']:.4f}m N={s['N_final']}")
    with open(out / "holdout_results.json", "w") as f:
        json.dump({"metadata": {"scene": a.scene, "n_frames": a.n_frames, "seeds": a.seeds,
                                "policies": a.policies, "holdout_every": a.holdout_every,
                                "note": "online-holdout proxy (unseen-for-optimization views on same trajectory), NOT independent novel trajectory; ATE GT-anchored (tracking decoupled)"},
                   "rows": rows}, f, indent=2)
    print(f">> wrote {out / 'holdout_results.json'}")


if __name__ == "__main__":
    main()
