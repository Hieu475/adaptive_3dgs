#!/usr/bin/env python3
"""Qualitative held-out comparison: original system vs uniform K=3 and K=10 steps.

Maps frames 1..N-1 of a scene, then renders the next frame (never mapped) from its pose and saves
ground truth, the three renders and absolute-error maps in one figure.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_vgg as vgg  # noqa: E402
from experiments.run_frozen_benchmark import load_benchmark_sequence  # noqa: E402
from src.core.pipeline import OnlineReconstructionPipeline  # noqa: E402
from src.evaluation.reproducibility import set_seed  # noqa: E402

CONDS = [("original", "error_off", {}), ("$K{=}3$", "tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}),
         ("$K{=}10$", "tts", {"T_ms": 1.0, "n_min": 10, "n_max": 10, "n_init": 10})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum_fr1_xyz", "tum_fr3_sitting_static"])
    ap.add_argument("--n_map", type=int, default=80, help="number of mapped frames before the held-out frame")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/paper/iccv_official/figs/fig_engine_qualitative.pdf")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    fig, axes = plt.subplots(len(a.scenes) * 2, 4, figsize=(7.0, 1.75 * 2 * len(a.scenes)))
    for r, sc in enumerate(a.scenes):
        frames, intr = load_benchmark_sequence(scene_name=sc, n_frames=a.n_map + 2, H=240, W=320, device="cuda")
        held = frames[a.n_map + 1]
        gt = held["rgb"].cpu().numpy()
        axes[2 * r, 0].imshow(np.clip(gt, 0, 1)); axes[2 * r, 0].set_title("ground truth", fontsize=7)
        axes[2 * r + 1, 0].axis("off")
        for c, (lab, cond, tt) in enumerate(CONDS):
            set_seed(42)
            _, cfg = vgg.config_for(cond, 42, 15.0, "cuda", {}, tt)
            pipe = OnlineReconstructionPipeline(config=cfg, device="cuda")
            pipe.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"], intrinsics=intr, pose=frames[0]["pose"])
            for f in frames[1:a.n_map + 1]:
                pipe.process_frame(rgb=f["rgb"], depth=f["depth"], gt_pose=f["pose"])
            out = pipe.eval_frame(rgb=held["rgb"], depth=held["depth"], pose=held["pose"], return_images=True)
            img = out["images"]["rgb_prediction"].numpy()
            err = np.abs(img - gt).mean(-1)
            axes[2 * r, c + 1].imshow(np.clip(img, 0, 1)); axes[2 * r, c + 1].set_title(f"{lab}", fontsize=7)
            axes[2 * r + 1, c + 1].imshow(err, cmap="magma", vmin=0, vmax=0.25)
            axes[2 * r + 1, c + 1].set_title(f"{out['psnr']:.1f} dB", fontsize=7)
            del pipe; torch.cuda.empty_cache()
        axes[2 * r, 0].set_ylabel(sc, fontsize=6)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.3); a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200); print("saved", a.out)


if __name__ == "__main__":
    main()
