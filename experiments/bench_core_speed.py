#!/usr/bin/env python3
"""Wall-clock and quality of the original budgeted configuration (error_off, 5 steps) on this code revision.

Run once per code revision (the paper compares the revision before and after the core speedups):
    python experiments/bench_core_speed.py --tag after --out results/core_speed
The per-frame time is the end-to-end process_frame latency including per-frame evaluation, averaged over
frames 10..N (CUDA-synchronized).
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

import experiments.run_throttle_ablation as abl  # noqa: E402
from src.core.pipeline import OnlineReconstructionPipeline  # noqa: E402
from src.evaluation.reproducibility import set_seed  # noqa: E402

try:
    from experiments.run_frozen_benchmark import load_benchmark_sequence as load_seq  # noqa: E402
except ImportError:  # revision before the rename
    from experiments.run_phase13_frozen_benchmark import load_phase10_sequence as load_seq  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "core_speed")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    res = {}
    for sc in a.scenes:
        fr, K = load_seq(scene_name=sc, n_frames=a.n_frames + 1, H=240, W=320, device="cuda")
        for seed in a.seeds:
            set_seed(seed)
            _, cfg = abl.config_for("error_off", seed, 15.0, "cuda")
            p = OnlineReconstructionPipeline(config=cfg, device="cuda")
            p.initialize(rgb=fr[0]["rgb"], depth=fr[0]["depth"], intrinsics=K, pose=fr[0]["pose"])
            ts, ps = [], []
            for t in range(1, len(fr)):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                m = p.process_frame(rgb=fr[t]["rgb"], depth=fr[t]["depth"], gt_pose=fr[t]["pose"])
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000); ps.append(float(m["psnr"]))
            res[f"{sc}/seed{seed}"] = {"ms": float(np.mean(ts[10:])), "psnr": float(np.mean(ps)), "N": p.gaussian_model.num_gaussians}
            print(a.tag, sc, seed, res[f"{sc}/seed{seed}"], flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
