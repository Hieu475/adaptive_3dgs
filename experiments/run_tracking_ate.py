#!/usr/bin/env python3
"""Baseline visual-odometry ATE evaluation (GT-free tracking).

Init on frame 0 with GT pose, then track WITHOUT any ground truth using
research.tracker.DenseRGBDTracker (frame-to-frame dense photometric+depth
Gauss-Newton; no keyframes/loop closure). Reports ATE RMSE vs GT trajectory
plus mapping PSNR under tracked poses.

Label: "baseline VO ATE" — NOT a SLAM system result. Mapping runs with the
configured policy (default ours) so tracking drift propagates honestly.

Usage:
  python experiments/run_tracking_ate.py --scenes tum_fr2_xyz tum_fr1_xyz --n_frames 30
"""
import sys, json, argparse
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_phase13_frozen_benchmark import load_phase10_sequence, build_pipeline_config
from research.pipeline import OnlineReconstructionPipeline
from research.holdout_eval import compute_ate
from research.reproducibility import set_seed


def main():
    ap = argparse.ArgumentParser(description="Baseline VO ATE eval")
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz"])
    ap.add_argument("--n_frames", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--policy", default="ours")
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="results/phase14_tracking")
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
        frames, intr = load_phase10_sequence(scene_name=sc, n_frames=a.n_frames + 1,
                                             H=H, W=W, device=a.device)
        for seed in a.seeds:
            set_seed(seed)
            cfg = build_pipeline_config(policy=a.policy, seed=seed, budget_ms=a.budget_ms,
                                        W=W, H=H, device=a.device)
            pipe = OnlineReconstructionPipeline(config=cfg, device=a.device)
            pipe.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                            intrinsics=intr, pose=frames[0].get("pose", torch.eye(4)))
            est, gt, ps = [frames[0]["pose"]], [frames[0]["pose"]], []
            import time
            t0 = time.perf_counter()
            for t in range(1, len(frames)):
                f = frames[t]
                m = pipe.process_frame(rgb=f["rgb"], depth=f["depth"], gt_pose=None)
                est.append(pipe.current_pose.detach().cpu())
                gt.append(f.get("pose", torch.eye(4)))
                ps.append(m["psnr"])
            wall = time.perf_counter() - t0
            ate = compute_ate(est, gt)
            import numpy as np
            row = {"scene": sc, "seed": seed, "policy": a.policy,
                   "ate_rmse_m": ate["ate_rmse_m"], "ate_max_m": ate["ate_max_m"],
                   "tracked_psnr": float(np.mean(ps)), "fps": len(frames) / wall,
                   "N_final": pipe.gaussian_model.num_gaussians}
            rows.append(row)
            print(f"{sc} seed={seed}: ATE={ate['ate_rmse_m']*100:.2f}cm max={ate['ate_max_m']*100:.2f}cm "
                  f"PSNR(tracked)={row['tracked_psnr']:.2f} N={row['N_final']}", flush=True)
    json.dump({"metadata": {"scenes": a.scenes, "n_frames": a.n_frames, "seeds": a.seeds,
                            "tracker": "DenseRGBDTracker frame-to-frame GN baseline (no KF/LC)",
                            "note": "baseline VO ATE; GT-anchored init frame 0"},
               "rows": rows}, open(out / "tracking_ate.json", "w"), indent=1)
    print("wrote", out / "tracking_ate.json")


if __name__ == "__main__":
    main()
