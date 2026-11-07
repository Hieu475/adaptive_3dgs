#!/usr/bin/env python3
"""Closed-loop test: map-based tracking (research/map_tracker.py) feeds the mapper.

Poses come from rendering the *current map* and aligning to each new frame (GT only for frame 0), so
mapping quality feeds back into tracking. Logs ATE RMSE, PSNR at the tracked pose, PSNR when the
final map state renders the GT pose, tracking time and mapping time per frame. Controller conditions are
those of run_vgg.py (wrapper installed on import); the TT timer covers mapping only, tracking time is
reported separately.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_vgg as rv  # noqa: E402
from experiments.run_phase13_frozen_benchmark import load_phase10_sequence  # noqa: E402
from research.holdout_eval import compute_ate  # noqa: E402
from research.map_tracker import MapTracker  # noqa: E402
from research.pipeline import OnlineReconstructionPipeline  # noqa: E402
from research.reproducibility import set_seed  # noqa: E402

CONDITIONS = ("error_off", "error_budget_prop", "tt_vgg", "tt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--conditions", nargs="+", choices=rv.CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--iters", type=int, nargs=3, default=[20, 15, 10])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T_ms", type=float, default=100.0)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "closed_loop")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    ov = {}
    for sc in a.scenes:
        frames, intr = load_phase10_sequence(scene_name=sc, n_frames=a.offset + a.n_frames + 1, H=240, W=320, device="cuda")
        frames = frames[a.offset:]
        for seed in a.seeds:
            for cond in a.conditions:
                target = a.out_dir / sc / f"{cond}_seed{seed}.json"
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                set_seed(seed)
                _, cfg = rv.config_for(cond, seed, 15.0, "cuda", ov, {"T_ms": a.T_ms})
                rv.abl.ext.BUDGET_SCALES.clear(); rv.STATES.clear(); rv.TT_STATES.clear()
                pipe = OnlineReconstructionPipeline(config=cfg, device="cuda")
                f0 = frames[0]
                pipe.initialize(rgb=f0["rgb"], depth=f0["depth"], intrinsics=intr, pose=f0["pose"])
                mt = MapTracker(iters=a.iters, lr=a.lr)
                mt.reset(f0["pose"].cuda().float())
                est, gt, p_trk, t_trk, t_map = [f0["pose"].cpu()], [f0["pose"].cpu()], [], [], []
                print(f"RUN {sc} {cond} seed={seed}", flush=True)
                for f in frames[1:]:
                    rgb, depth = f["rgb"].cuda(), f["depth"].cuda()
                    torch.cuda.synchronize(); t0 = time.time()
                    T = mt.track(pipe, rgb, depth)
                    torch.cuda.synchronize(); t1 = time.time()
                    m = pipe.process_frame(rgb=rgb, depth=depth, gt_pose=T)
                    torch.cuda.synchronize(); t2 = time.time()
                    est.append(T.detach().cpu()); gt.append(f["pose"].cpu())
                    p_trk.append(float(m["psnr"])); t_trk.append(t1 - t0); t_map.append(t2 - t1)
                ate = compute_ate(est, gt)
                p_gt = [float(pipe.eval_frame(rgb=f["rgb"], depth=f["depth"], pose=f["pose"])["psnr"]) for f in frames[-10:]]
                res = {"scene": sc, "seed": seed, "condition": cond, "offset": a.offset,
                       "ate_rmse_m": ate["ate_rmse_m"], "ate_max_m": ate["ate_max_m"],
                       "psnr_tracked": float(np.mean(p_trk)), "psnr_final_map_at_gt_last10": float(np.mean(p_gt)),
                       "track_ms": 1000 * float(np.mean(t_trk)), "map_ms": 1000 * float(np.mean(t_map)),
                       "N_final": pipe.gaussian_model.num_gaussians,
                       "tt_m_final": rv.TT_STATES[0].m if rv.TT_STATES else None, "n_frames": len(frames)}
                tmp = target.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); tmp.replace(target)
                print(f"DONE {sc} {cond} seed={seed}: psnr={res['psnr_tracked']:.2f} ATE={100 * res['ate_rmse_m']:.2f}cm "
                      f"track={res['track_ms']:.0f}ms map={res['map_ms']:.0f}ms N={res['N_final']}", flush=True)


if __name__ == "__main__":
    main()
