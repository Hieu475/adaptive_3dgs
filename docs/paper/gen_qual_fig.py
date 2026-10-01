#!/usr/bin/env python3
"""Qualitative figure: GT | OURS | error_only | FULL on both substrates.

Row 1: fr2_xyz view149 (OURS +8.5 over error_only, matches FULL).
Row 2: office0 view98 (error_only +2.4 over OURS; adaptive routes there).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
FIGS = REPO / "docs" / "paper" / "figs"

rows = [
    ("results/qualitative_fr2", 149, [("gt", None), ("ours", 27.40), ("error_only", 18.93), ("full", 28.63)],
     "TUM fr2_xyz (noisy): throttling wins"),
    ("results/qualitative_office0", 98, [("gt", None), ("ours", 25.93), ("error_only", 28.37), ("full", 31.67)],
     "Replica office0 (clean): raw-error wins"),
]

fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6))
for r, (d, v, cols, title) in enumerate(rows):
    axes[r, 0].set_ylabel(title, fontsize=7)
    for c, (pol, psnr) in enumerate(cols):
        img = plt.imread(REPO / d / f"{pol}_view{v:02d}.png")
        axes[r, c].imshow(img)
        axes[r, c].axis("off")
        tag = "GT" if pol == "gt" else f"{pol}\n{psnr:.1f} dB"
        axes[r, c].set_title(tag, fontsize=7)
fig.suptitle("Same fixed policies, opposite winners: adaptivity is necessary", fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "fig5_qualitative.pdf")
print("wrote fig5_qualitative.pdf")
