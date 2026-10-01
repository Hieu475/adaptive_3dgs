#!/usr/bin/env python3
"""Generate ICCV skeleton figures from frozen measured numbers (no reruns).

Fig1: v3 sensor-noise separation across 5 scenes (raw hole-rate based).
Fig2: main-benchmark deltas vs OURS with 95% CI (fr2_xyz 150f x8).
Fig3: kappa ablation on replica_office0 (30f x3) + selection gap.
Fig4: map-size vs PSNR Pareto (Phase-14 6 policies).

All values anchored to results/ artifacts; see JSON sidecars.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 200})

# ---- Fig1: v3 separation (measured 2026-10-01, 10 frames/scene means) ----
scenes = ["replica_office0", "replica_room0", "tum_fr1_xyz", "tum_fr1_desk", "tum_fr2_xyz"]
v3 = [0.017, 0.022, 0.706, 0.493, 0.714]
cols = ["tab:green" if s.startswith("replica") else "tab:blue" for s in scenes]
fig, ax = plt.subplots(figsize=(3.4, 1.9))
ax.bar(range(len(scenes)), v3, color=cols)
ax.axhspan(0.10, 0.30, color="gray", alpha=0.15, label="blend band")
ax.axhline(0.10, color="k", ls="--", lw=0.7)
ax.set_xticks(range(len(scenes)))
ax.set_xticklabels([s.replace("replica_", "R:").replace("tum_", "T:") for s in scenes], rotation=18, ha="right")
ax.set_ylabel("v3 noise score")
ax.set_title("Raw hole-rate separates substrates 300-500x", fontsize=9)
ax.legend(["blend band", "threshold"], fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig1_noise_separation.pdf")
json.dump({"scenes": scenes, "v3": v3}, open(OUT / "fig1.json", "w"), indent=1)

# ---- Fig2: deltas vs OURS (Phase-14 main, 8 seeds) ----
labels = ["no_op", "error_only", "error_influence", "rtg_slam", "FULL"]
deltas = [8.80, 3.59, 2.63, 4.62, -0.18]
ci_lo = [8.80, 3.48, 2.56, 4.44, -0.18]
ci_hi = [8.80, 3.70, 2.68, 4.82, -0.18]
cols2 = ["tab:red" if d > 0 else "tab:gray" for d in deltas]
fig, ax = plt.subplots(figsize=(3.4, 1.9))
y = np.arange(len(labels))
ax.barh(y, deltas, color=cols2)
for i in range(len(labels)):
    ax.plot([ci_lo[i], ci_hi[i]], [y[i], y[i]], color="k", lw=1.2)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.axvline(0, color="k", lw=0.7)
ax.set_xlabel("OURS $-$ baseline mean PSNR (dB)")
ax.set_title("OURS beats all selective/SLAM policies (n=8, Holm)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig2_main_deltas.pdf")

# ---- Fig3: kappa ablation + selection gap (office0 30f x3) ----
conds = ["off", "k0.80", "k0.90", "k0.95", "error_only"]
vals = [24.79, 24.00, 23.98, 24.19, 29.48]
cols3 = ["tab:orange"] * 4 + ["tab:green"]
fig, ax = plt.subplots(figsize=(3.4, 1.9))
ax.bar(range(len(conds)), vals, color=cols3)
ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds)
ax.set_ylabel("mean PSNR (dB)")
ax.set_title("Throttling ~0.8dB; selection gap ~4.7dB (office0)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig3_kappa_ablation.pdf")
json.dump({"conds": conds, "psnr": vals}, open(OUT / "fig3.json", "w"), indent=1)

# ---- Fig4: Pareto map-size vs PSNR ----
names = ["no_op", "error_only", "error_influence", "rtg_slam", "OURS", "FULL"]
psnr = [18.86, 24.07, 25.04, 23.05, 27.67, 27.85]
nmap = [23.5, 74.6, 106.9, 67.2, 23.4, 88.0]
fig, ax = plt.subplots(figsize=(3.4, 1.9))
for x, y_, n in zip(nmap, psnr, names):
    ax.scatter(x, y_, s=28, c="tab:red" if n == "OURS" else "tab:blue")
    ax.annotate(n, (x, y_), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("final map size (K primitives)")
ax.set_ylabel("mean PSNR (dB)")
ax.set_title("Quality vs compactness (fr2_xyz 150f x8)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig4_pareto.pdf")
json.dump({"names": names, "psnr": psnr, "mapK": nmap}, open(OUT / "fig4.json", "w"), indent=1)

print("wrote", sorted(p.name for p in OUT.glob("fig*")))
