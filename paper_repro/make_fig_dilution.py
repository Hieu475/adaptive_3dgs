#!/usr/bin/env python3
"""Fig. 1: map growth and PSNR over time, mean +/- 1 sd over seeds (rerun factorial)."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/dilution/rerun_paper1/dilution_factorial"
OUT = ROOT / "docs/paper/figs"
SCENES = [("tum_fr2_xyz", "TUM fr2_xyz"), ("tum_fr1_xyz", "TUM fr1_xyz")]
CURVES = [("error_off", "error_only, no throttle", "tab:blue", "--"),
          ("error_coverage", "error_only + coverage", "tab:green", "-"),
          ("temporal_coverage", "temporal + coverage", "tab:red", "-")]


def load(scene, cond):
    runs = [json.loads(p.read_text()) for p in sorted((SRC / scene).glob(f"{cond}_seed*.json"))]
    frames = [r["frame"] for r in runs[0]["frames"]]
    n = np.array([[f["n_gaussians"] for f in r["frames"]] for r in runs]) / 1000.0
    q = np.array([[f["psnr"] for f in r["frames"]] for r in runs])
    return frames, n, q, len(runs)


fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.4), sharex=True)
meta = {}
for col, (scene, title) in enumerate(SCENES):
    for cond, label, color, ls in CURVES:
        frames, n, q, k = load(scene, cond)
        meta[f"{scene}/{cond}"] = {"n_seeds": k, "final_map_k": float(n[:, -1].mean()), "mean_psnr": float(q.mean())}
        for row, arr in ((0, n), (1, q)):
            mu, sd = arr.mean(0), arr.std(0)
            axes[row, col].plot(frames, mu, color=color, ls=ls, lw=1.2, label=label)
            axes[row, col].fill_between(frames, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)
    axes[0, col].set_title(title, fontsize=8)
    axes[1, col].set_xlabel("frame", fontsize=8)
axes[0, 0].set_ylabel("map size (K)", fontsize=8)
axes[1, 0].set_ylabel("PSNR (dB)", fontsize=8)
axes[0, 0].legend(fontsize=6.5, loc="upper left")
for ax in axes.ravel():
    ax.tick_params(labelsize=7)
fig.tight_layout(pad=0.6)
fig.savefig(OUT / "fig7_dilution.pdf")
(OUT / "fig7.json").write_text(json.dumps({"source": "results/dilution/rerun_paper1/dilution_factorial", "curves": meta}, indent=1))
print("wrote", OUT / "fig7_dilution.pdf")
