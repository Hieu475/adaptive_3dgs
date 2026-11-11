#!/usr/bin/env python3
"""Build Figure 7 from the corrected-pose, eight-seed dilution traces."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "dilution" / "dilution_factorial"
OUTPUTS = (
    ROOT / "docs" / "paper" / "iccv_official" / "figs" / "fig7_dilution.pdf",
    ROOT / "docs" / "paper" / "figs" / "fig7_dilution.pdf",
)
SCENES = ("tum_fr2_xyz", "tum_fr1_xyz")
CONDITIONS = (
    ("error_off", "Error, no throttle", "#2878B5", "-"),
    ("error_coverage", "Error + coverage", "#E07A1F", "--"),
    ("temporal_coverage", "Temporal + coverage (ours)", "#2A9D69", "-."),
)
SEEDS = set(range(42, 50))


def load_traces(scene, condition, field):
    runs = []
    for path in sorted((DATA / scene).glob(f"{condition}_seed*.json")):
        record = json.loads(path.read_text())
        if record["seed"] not in SEEDS or record["condition"] != condition:
            continue
        runs.append({frame["frame"]: frame[field] for frame in record["frames"]})
    if len(runs) != 8:
        raise RuntimeError(f"Expected 8 seed traces for {scene}/{condition}; found {len(runs)}")
    frames = sorted(set.intersection(*(set(run) for run in runs)))
    values = np.asarray([[run[frame] for frame in frames] for run in runs], dtype=float)
    return np.asarray(frames), values.mean(axis=0), values.std(axis=0, ddof=1)


def main():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.35), sharex="col")
    for col, scene in enumerate(SCENES):
        short = scene.removeprefix("tum_")
        for row, (field, ylabel) in enumerate((("n_gaussians", "Map size (K)"), ("psnr", "PSNR (dB)"))):
            ax = axes[row, col]
            for condition, label, color, linestyle in CONDITIONS:
                frames, mean, sd = load_traces(scene, condition, field)
                if field == "n_gaussians":
                    mean, sd = mean / 1000.0, sd / 1000.0
                ax.plot(frames, mean, color=color, linestyle=linestyle, linewidth=1.35, label=label)
                ax.fill_between(frames, mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0)
            ax.grid(True, color="#D9DDE2", linewidth=0.55, alpha=0.8)
            ax.set_xlim(1, 150)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(short.replace("_", " "))
            if row == 1:
                ax.set_xlabel("Frame")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.01), columnspacing=1.5, handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.5, w_pad=1.2)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
