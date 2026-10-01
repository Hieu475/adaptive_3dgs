#!/usr/bin/env python3
"""Build supplementary video from frozen artifacts (no reruns).

Segments: title -> fr2_xyz A/B stills -> office0 A/B stills -> demo progress
GIFs (fresh ours runs). Output: docs/paper/supp_video.mp4 (H.264, yuv420p).
"""
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "docs" / "paper" / "supp_frames"
WORK.mkdir(parents=True, exist_ok=True)


def title_card(lines, name, hold=90):
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.62, lines[0], ha="center", fontsize=34, weight="bold")
    for i, ln in enumerate(lines[1:]):
        ax.text(0.5, 0.45 - 0.09 * i, ln, ha="center", fontsize=18)
    fig.savefig(WORK / name, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return [name] * hold


def still_row(d, v, cols, name, hold=150):
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.9))
    for ax, pol in zip(axes, cols):
        img = np.asarray(Image.open(REPO / d / f"{pol}_view{v:02d}.png"))
        ax.imshow(img); ax.axis("off")
        ax.set_title("GT" if pol == "gt" else pol, fontsize=14)
    fig.suptitle(name.replace("_", " "), fontsize=16)
    fig.tight_layout()
    out = WORK / f"still_{d.split('/')[-1]}_{v:02d}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return [out.name] * hold


def gif_frames(gif_path, every=3, cap=60):
    im = Image.open(REPO / gif_path)
    outs = []
    i = 0
    try:
        while len(outs) < cap:
            f = im.convert("RGB").resize((1280, 720))
            p = WORK / f"gif_{len(outs):03d}_{Path(gif_path).parent.name}.png"
            f.save(p)
            outs.append(p.name)
            i += every
            im.seek(i)
    except EOFError:
        pass
    return outs


def main():
    seq = []
    seq += title_card(["Adaptive 3DGS: Dual Throttling + Sensor-Adaptive Routing",
                       "TUM fr2_xyz: OURS 27.4 dB vs error_only 18.9 (view 149)",
                       "Replica office0: error_only 28.4 vs OURS 25.9 (view 98)",
                       "Same fixed policies, opposite winners"], "t0.png")
    seq += still_row("results/qualitative_fr2", 149, ["gt", "ours", "error_only", "full"],
                     "fr2_xyz view149: throttling wins")
    seq += still_row("results/qualitative_office0", 98, ["gt", "ours", "error_only", "full"],
                     "office0 view98: raw-error wins")
    for g in ["results/demo_output/tum_fr2_xyz/demo_recon_progress.gif",
              "results/demo_output/replica_office0_ours_fresh/demo_recon_progress.gif"]:
        p = REPO / g
        if p.exists():
            seq += gif_frames(g)
    lst = WORK / "list.txt"
    lst.write_text("".join(f"file '{n}'\nduration 0.2\n" for n in seq))
    mp4 = REPO / "docs" / "paper" / "supp_video.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-framerate", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(mp4)], check=True, capture_output=True)
    print("wrote", mp4, mp4.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
