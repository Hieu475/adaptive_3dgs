#!/usr/bin/env python3
"""Figure 5: representative TUM and Replica qualitative comparisons.

The two rows show opposite fixed-policy winners.  TUM uses an intermediate
view rather than the terminal frame; Replica keeps its representative office0
view.  The signed RGB L1 gap is error_only minus OURS (red means OURS is
closer to GT).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np, cv2
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
FIGURES = (
    REPO / "docs" / "paper" / "figs" / "fig5_qualitative.pdf",
    REPO / "docs" / "paper" / "iccv_official" / "figs" / "fig5_qualitative.pdf",
)
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
    img = cv2.imread(str(REPO / "results" / scene / f"{pol}_view{view}.png"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Main figure: both evaluated substrates, same four render columns plus signed error gap.
rows = [
    ("qualitative_fr2", 90, {"gt": None, "ours": 22.71, "error_only": 20.14, "full": 22.59},
     "TUM fr2_xyz, view 90"),
    ("qualitative_office0", 98, {"gt": None, "ours": 25.93, "error_only": 28.37, "full": 31.67},
     "Replica office0, view 98"),
]
policies = ["gt", "ours", "error_only", "full"]
titles = {"gt": "GT", "ours": "OURS", "error_only": "error-only", "full": "FULL"}
fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.45), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1]})
for r, (scene, view, psnr, row_name) in enumerate(rows):
    imgs = {p: load(scene, p, view) for p in policies}
    # Select the crop where the two budgeted policies differ most; both
    # substrates use the same fixed crop size and crop-selection rule.
    winner = "ours" if psnr["ours"] >= psnr["error_only"] else "error_only"
    loser = "error_only" if winner == "ours" else "ours"
    x, y, gap = best_crop(imgs["gt"], imgs[winner], imgs[loser])
    err_ours = np.abs(imgs["ours"].astype(np.float32) - imgs["gt"].astype(np.float32)).mean(axis=2)
    err_error = np.abs(imgs["error_only"].astype(np.float32) - imgs["gt"].astype(np.float32)).mean(axis=2)
    heat = err_error - err_ours  # positive/red means OURS has lower RGB L1 error
    print(f"{scene} view{view}: crop=({x},{y}) winner={winner} gap={gap:.2f}")
    for c, p in enumerate(policies):
        ax = axes[r, c]
        ax.imshow(imgs[p]); ax.axis("off")
        tag = titles[p] if p == "gt" else f"{titles[p]}\n{psnr[p]:.1f} dB"
        ax.set_title(tag, fontsize=8)
        ax.add_patch(Rectangle((x, y), CROP, CROP, linewidth=1.3, edgecolor="red", facecolor="none"))
        ins = inset_axes(ax, width="43%", height="43%", loc="lower right", borderpad=0.35)
        ins.imshow(imgs[p][y:y + CROP, x:x + CROP]); ins.set_xticks([]); ins.set_yticks([])
        for spine in ins.spines.values():
            spine.set_edgecolor("red"); spine.set_linewidth(1.3)
    ax = axes[r, 4]
    vmax = max(float(np.percentile(np.abs(heat), 99)), 1e-6)
    ax.imshow(heat, cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax.axis("off")
    ax.set_title("RGB L1 gap\n(red: OURS better)", fontsize=8)
    ax.add_patch(Rectangle((x, y), CROP, CROP, linewidth=1.3, edgecolor="k", facecolor="none"))
    axes[r, 0].set_ylabel(row_name, fontsize=7)
fig.tight_layout(pad=0.5)
for path in FIGURES:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
print("wrote both Figure 5 assets")
