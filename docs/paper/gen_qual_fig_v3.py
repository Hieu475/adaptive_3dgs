#!/usr/bin/env python3
"""Fig5 v3: row1 fr2 gets GT/OURS/error_only + error-gap heatmap; row2 office0 keeps GT/OURS/error_only/FULL."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import cv2
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
FIGS = REPO / "docs" / "paper" / "figs"
CROP = 150
def best_crop(gt, a, b):
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
fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.0))
# Row 1: fr2 view149
scene, view = "qualitative_fr2", 149
psnrs = {"ours": 27.40, "error_only": 18.93}
imgs = {p: load(scene, p, view) for p in ["gt", "ours", "error_only"]}
x, y, gap = best_crop(imgs["gt"], imgs["ours"], imgs["error_only"])
print(f"fr2 v{view}: crop=({x},{y}) gap={gap:.2f}")
da = np.abs(imgs["ours"].astype(np.float32) - imgs["gt"].astype(np.float32)).mean(axis=2)
db = np.abs(imgs["error_only"].astype(np.float32) - imgs["gt"].astype(np.float32)).mean(axis=2)
heat = db - da  # >0 (red) where error_only worse
titles1 = ["GT", "OURS\n27.4 dB", "error_only\n18.9 dB"]
for c, p in enumerate(["gt", "ours", "error_only"]):
    ax = axes[0, c]
    ax.imshow(imgs[p]); ax.axis("off"); ax.set_title(titles1[c], fontsize=8)
    ax.add_patch(Rectangle((x, y), CROP, CROP, linewidth=1.8, edgecolor="red", facecolor="none"))
    axins = inset_axes(ax, width="45%", height="45%", loc="lower right", borderpad=0.5)
    axins.imshow(imgs[p][y:y + CROP, x:x + CROP]); axins.set_xticks([]); axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("red"); s.set_linewidth(1.8)
ax = axes[0, 3]
im = ax.imshow(heat, cmap="RdBu_r", vmin=-np.abs(heat).max(), vmax=np.abs(heat).max())
ax.set_title("error gap\n(red: error_only worse)", fontsize=8); ax.axis("off")
ax.add_patch(Rectangle((x, y), CROP, CROP, linewidth=1.8, edgecolor="k", facecolor="none"))
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
axes[0, 0].set_ylabel("TUM fr2_xyz v149\nOURS +8.5 dB", fontsize=7)
# Row 2: office0 view98 (keep 4 renders)
scene2, view2 = "qualitative_office0", 98
ps2 = {"ours": 25.93, "error_only": 28.37, "full": 31.67}
imgs2 = {p: load(scene2, p, view2) for p in ["gt", "ours", "error_only", "full"]}
x2, y2, gap2 = best_crop(imgs2["gt"], imgs2["error_only"], imgs2["ours"])
print(f"office0 v{view2}: crop=({x2},{y2}) gap={gap2:.2f}")
for c, p in enumerate(["gt", "ours", "error_only", "full"]):
    ax = axes[1, c]
    ax.imshow(imgs2[p]); ax.axis("off")
    tag = {"gt": "GT", "ours": "OURS\n25.9 dB", "error_only": "error_only\n28.4 dB", "full": "FULL\n31.7 dB"}[p]
    ax.set_title(tag, fontsize=8)
    ax.add_patch(Rectangle((x2, y2), CROP, CROP, linewidth=1.5, edgecolor="red", facecolor="none"))
    axins = inset_axes(ax, width="42%", height="42%", loc="lower right", borderpad=0.5)
    axins.imshow(imgs2[p][y2:y2 + CROP, x2:x2 + CROP]); axins.set_xticks([]); axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("red"); s.set_linewidth(1.5)
axes[1, 0].set_ylabel("Replica office0 v98\nerror_only +2.4 dB", fontsize=7)
fig.suptitle("Same fixed policies, opposite winners (red: most-discriminative region, zoomed)", fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "fig5_qualitative.pdf")
print("wrote fig5_qualitative.pdf")
