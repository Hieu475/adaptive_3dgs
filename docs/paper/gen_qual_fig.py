#!/usr/bin/env python3
"""Fig5 v2: full views with red-boxed most-discriminative region + zoom insets.

fr2_xyz view149 (OURS +8.5dB): crop where |err-gt|-|ours-gt| is maximal.
office0 view98 (error_only +2.4dB): same procedure (subtler gap, zoom makes it visible).
Crop auto-selected by error-gap scan (no hand-picking).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import cv2
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
FIGS = REPO / "docs" / "paper" / "figs"
CROP = 120


def best_crop(gt, a, b):
    """Region where b is worse than a by the largest margin (mean abs err gap)."""
    da = np.abs(a.astype(np.float32) - gt.astype(np.float32)).mean(axis=2)
    db = np.abs(b.astype(np.float32) - gt.astype(np.float32)).mean(axis=2)
    gap = db - da
    h, w = gap.shape
    best = None
    for y in range(0, h - CROP, 10):
        for x in range(0, w - CROP, 10):
            g = gap[y:y + CROP, x:x + CROP].mean()
            if best is None or g > best[0]:
                best = (g, x, y)
    return best[1], best[2], best[0]


def load(scene, pol, view):
    p = REPO / "results" / scene / f"{pol}_view{view}.png"
    img = cv2.imread(str(p))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


rows = [
    ("qualitative_fr2", 149, {"gt": None, "ours": 27.40, "error_only": 18.93, "full": 28.63},
     "TUM fr2_xyz view149: OURS wins by +8.5 dB"),
    ("qualitative_office0", 98, {"gt": None, "ours": 25.93, "error_only": 28.37, "full": 31.67},
     "Replica office0 view98: error_only wins by +2.4 dB"),
]
pols = ["gt", "ours", "error_only", "full"]
titles = {"gt": "GT", "ours": "OURS", "error_only": "error_only", "full": "FULL"}

fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.4))
for r, (scene, view, psnrs, subtitle) in enumerate(rows):
    imgs = {p: load(scene, p, view) for p in pols}
    # crop where the LOSER is worse than the WINNER by max margin
    winner = "ours" if psnrs["ours"] > psnrs["error_only"] else "error_only"
    loser = "error_only" if winner == "ours" else "ours"
    x, y, gap = best_crop(imgs["gt"], imgs[winner], imgs[loser])
    print(f"{scene} v{view}: crop=({x},{y}) gap={gap:.2f}")
    for c, p in enumerate(pols):
        ax = axes[r, c]
        ax.imshow(imgs[p])
        ax.axis("off")
        tag = titles[p] + ("" if p == "gt" else f"\n{psnrs[p]:.1f} dB")
        ax.set_title(tag, fontsize=8)
        rect = Rectangle((x, y), CROP, CROP, linewidth=1.5,
                         edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        axins = inset_axes(ax, width="42%", height="42%", loc="lower right",
                           borderpad=0.5)
        axins.imshow(imgs[p][y:y + CROP, x:x + CROP])
        axins.set_xticks([]); axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor("red"); spine.set_linewidth(1.5)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red", lw=0.8)
    axes[r, 0].set_ylabel(subtitle, fontsize=7)
fig.suptitle("Same fixed policies, opposite winners (red: most-discriminative region, zoomed)",
             fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "fig5_qualitative.pdf")
print("wrote fig5_qualitative.pdf")
