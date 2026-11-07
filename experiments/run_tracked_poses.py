#!/usr/bin/env python3
"""Do the budget and throttling effects survive estimated poses?

Mapping uses poses from the frame-to-frame dense RGB-D Gauss-Newton tracker (GT only for the
first frame). Per frame we log PSNR at the tracked pose (self-consistent) and PSNR when the same
map renders from the ground-truth pose (exposes alignment errors), plus ATE RMSE. The tracker does
not use the map, so ATE is shared across conditions; mapping quality changes only the PSNR terms.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_throttle_ablation as abl  # noqa: E402
from experiments.run_frozen_benchmark import load_benchmark_sequence  # noqa: E402
from research.holdout_eval import compute_ate  # noqa: E402
from research.pipeline import OnlineReconstructionPipeline  # noqa: E402
from research.reproducibility import set_seed  # noqa: E402

CONDITIONS = ("error_off", "error_coverage", "spawn_cap_always20", "error_budget_prop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "tum_fr1_xyz"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--conditions", nargs="+", choices=abl.CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "tracked_poses")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    for scene in args.scenes:
        frames, intr = load_benchmark_sequence(scene_name=scene, n_frames=args.n_frames + 1, H=240, W=320, device=args.device)
        for seed in args.seeds:
            for cond in args.conditions:
                target = args.out_dir / scene / f"{cond}_seed{seed}.json"
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                set_seed(seed)
                policy, cfg = abl.config_for(cond, seed, args.budget_ms, args.device)
                pipe = OnlineReconstructionPipeline(config=cfg, device=args.device)
                pipe.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"], intrinsics=intr, pose=frames[0]["pose"])
                est, gt, p_trk, p_gt = [frames[0]["pose"]], [frames[0]["pose"]], [], []
                print(f"RUN {scene} {cond} seed={seed}", flush=True)
                for t in range(1, len(frames)):
                    f = frames[t]
                    m = pipe.process_frame(rgb=f["rgb"], depth=f["depth"], gt_pose=None)
                    est.append(pipe.current_pose.detach().cpu())
                    gt.append(f["pose"].detach().cpu() if isinstance(f["pose"], torch.Tensor) else f["pose"])
                    p_trk.append(float(m["psnr"]))
                    p_gt.append(float(pipe.eval_frame(rgb=f["rgb"], depth=f["depth"], pose=f["pose"])["psnr"]))
                ate = compute_ate(est, gt)
                res = {"scene": scene, "seed": seed, "condition": cond, "ate_rmse_m": ate["ate_rmse_m"],
                       "psnr_tracked": float(np.mean(p_trk)), "psnr_at_gt_pose": float(np.mean(p_gt)),
                       "N_final": pipe.gaussian_model.num_gaussians, "n_frames_loaded": len(frames)}
                tmp = target.with_suffix(".tmp")
                tmp.write_text(json.dumps(res, indent=1))
                tmp.replace(target)
                print(f"DONE {scene} {cond} seed={seed}: tracked={res['psnr_tracked']:.2f} at_gt={res['psnr_at_gt_pose']:.2f} "
                      f"ATE={res['ate_rmse_m'] * 100:.2f}cm N={res['N_final']}", flush=True)


if __name__ == "__main__":
    main()
