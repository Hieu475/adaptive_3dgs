#!/usr/bin/env python3
"""Long-horizon (500-frame) analysis: does the gap to unthrottled raw-error grow with horizon?

For each scene and seed, per-frame PSNR of the three policies is averaged in windows
[1,150], [151,300], [301,500]; the table reports the seed-mean difference to the unthrottled
baseline in each window, plus the final map sizes. With two seeds per scene the numbers are
descriptive (no intervals). Replica scenes are excluded: the local copies have only 100-120 frames.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "results" / "dilution" / "dilution_long"
WINDOWS = [(1, 150), (151, 300), (301, 500)]
CONDS = ("error_off", "error_coverage", "error_budget_prop")


def main():
    out, lines = {}, ["# Long-horizon (500 frames)", ""]
    for sd in sorted(p for p in D.iterdir() if p.is_dir()):
        runs = {}
        for f in sd.glob("*_seed*.json"):
            r = json.loads(f.read_text())
            runs[(r["condition"], r["seed"])] = r
        seeds = sorted({s for _, s in runs if all((c, s) in runs for c in CONDS)})
        if not seeds:
            continue
        n_frames = len(runs[(CONDS[0], seeds[0])]["frames"])
        if n_frames < 400:
            continue  # short sequence (e.g. Replica): not a long-horizon test
        def win(c, s, a, b):
            fr = runs[(c, s)]["frames"]
            return float(np.mean([f["psnr"] for f in fr if a <= f["frame"] <= b]))
        res = {"n_frames": n_frames, "seeds": seeds, "windows": {}, "final_map_k": {}, "mean_budget_scale": {}}
        for c in CONDS:
            res["final_map_k"][c] = float(np.mean([runs[(c, s)]["summary"]["N_final"] for s in seeds]) / 1000)
        res["mean_budget_scale"] = float(np.mean([runs[("error_budget_prop", s)]["summary"].get("mean_budget_scale", 1.0) for s in seeds]))
        lines += [f"## {sd.name} ({n_frames} frames, seeds {seeds}, mean budget scale {res['mean_budget_scale']:.2f})", "",
                  "| window | unthrottled PSNR | coverage - base | proportional - base | per-seed prop. gains |", "|---|--:|--:|--:|---|"]
        for a, b in WINDOWS:
            base = [win("error_off", s, a, b) for s in seeds]
            cov = [win("error_coverage", s, a, b) - x for s, x in zip(seeds, base)]
            prop = [win("error_budget_prop", s, a, b) - x for s, x in zip(seeds, base)]
            res["windows"][f"{a}-{b}"] = {"base": float(np.mean(base)), "cov": float(np.mean(cov)),
                                          "prop": float(np.mean(prop)), "prop_per_seed": [float(x) for x in prop]}
            lines.append(f"| {a}-{b} | {np.mean(base):.2f} | {np.mean(cov):+.2f} | {np.mean(prop):+.2f} | "
                         f"{', '.join(f'{x:+.2f}' for x in prop)} |")
        lines += ["", "Final map (k): " + ", ".join(f"{c} {v:.0f}" for c, v in res["final_map_k"].items()), ""]
        out[sd.name] = res
    (D / "analysis.json").write_text(json.dumps(out, indent=2))
    (D / "analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
