#!/usr/bin/env python3
"""Create the explicitly illustrative RGB-D reconstruction comparison layout."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bao_cao" / "figures" / "rgbd_reconstruction_mockup.pdf"


def draw_image_placeholder(ax, cx, cy, color="#9AA0A6"):
    """Small neutral image icon to signal that this panel has no data yet."""
    w, h = 0.15, 0.11
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                           fill=False, lw=0.9, edgecolor=color))
    ax.add_patch(plt.Circle((cx - 0.035, cy + 0.018), 0.009,
                            fill=False, lw=0.8, edgecolor=color))
    ax.plot([cx - 0.055, cx - 0.015, cx + 0.005, cx + 0.055],
            [cy - 0.035, cy + 0.002, cy - 0.015, cy + 0.03],
            color=color, lw=0.8)


def main():
    labels = [
        "GT RGB", "Error-only", "OURS", "FULL",
        "GT Depth", "OURS Depth", "Depth Error", "Coverage Mask",
    ]
    fig = plt.figure(figsize=(8.2, 4.35), facecolor="white")
    fig.text(0.055, 0.95, "Qualitative RGB-D Reconstruction Comparison",
             ha="left", va="top", fontsize=12, fontweight="semibold", color="#222222")

    # A soft container and eight intentionally empty panels.
    container = FancyBboxPatch((0.045, 0.17), 0.91, 0.72,
                               boxstyle="round,pad=0.008,rounding_size=0.018",
                               transform=fig.transFigure, facecolor="#FCFCFC",
                               edgecolor="#D8D8D8", linewidth=0.8, zorder=-1)
    fig.add_artist(container)
    left, right = 0.065, 0.935
    top, bottom = 0.82, 0.28
    gap_x, gap_y = 0.025, 0.055
    cell_w = (right - left - 3 * gap_x) / 4
    cell_h = (top - bottom - gap_y) / 2

    for i, label in enumerate(labels):
        row, col = divmod(i, 4)
        x = left + col * (cell_w + gap_x)
        y = top - (row + 1) * cell_h - row * gap_y
        ax = fig.add_axes([x, y, cell_w, cell_h])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                    boxstyle="round,pad=0.006,rounding_size=0.025",
                                    facecolor="white", edgecolor="#E2E2E2", linewidth=0.75))
        ax.text(0.5, 0.12, label, ha="center", va="center", fontsize=8, color="#333333")
        if row == 0:
            draw_image_placeholder(ax, 0.5, 0.58)
        else:
            # Corner marks echo an empty depth/mask preview without suggesting data.
            ax.plot([0.43, 0.43], [0.58, 0.64], color="#9AA0A6", lw=0.9)
            ax.plot([0.43, 0.48], [0.64, 0.64], color="#9AA0A6", lw=0.9)
            ax.plot([0.57, 0.57], [0.58, 0.64], color="#9AA0A6", lw=0.9)
            ax.plot([0.52, 0.57], [0.64, 0.64], color="#9AA0A6", lw=0.9)
            ax.plot([0.43, 0.43], [0.48, 0.54], color="#9AA0A6", lw=0.9)
            ax.plot([0.43, 0.48], [0.48, 0.48], color="#9AA0A6", lw=0.9)
            ax.plot([0.57, 0.57], [0.48, 0.54], color="#9AA0A6", lw=0.9)
            ax.plot([0.52, 0.57], [0.48, 0.48], color="#9AA0A6", lw=0.9)

    fig.text(0.06, 0.205,
             "Bố cục minh họa; các ô chưa chứa dữ liệu thực nghiệm. Cần xuất ảnh từ checkpoint và viewpoint xác định.",
             ha="left", va="center", fontsize=8, color="#666666")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
