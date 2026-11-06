#!/usr/bin/env python3
"""Evaluate value-gated growth (VGG) on held-out test scenes (results/vgg_test).

Per scene (seed-paired against the unthrottled baseline): PSNR change of the fixed spawn cap,
the map-proportional budget, VGG and VGG + proportional budget; map size and frame time. Across
scenes: sign count, worst case, mean, and regret against the per-scene best of {no cap, fixed cap}.
Hyperparameters were frozen on development scenes (results/vgg_dev) before this run.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run_final_confirmation import bootstrap_ci_95  # noqa: E402

D = ROOT / "results" / "vgg_test"
M = {"spawn_cap_always20": "fixed cap", "error_budget_prop": "prop. budget", "vgg": "VGG", "vgg_prop": "VGG + prop."}


def main():
    scenes, L = {}, ["# VGG on test scenes", "", "| scene | n | base PSNR | " + " | ".join(M.values()) + " | map k (base/cap/VGG) | frame ms (base/cap/VGG) |", "|---|--:|--:|" + "--:|" * len(M) + "---|---|"]
    for sd in sorted(p for p in D.iterdir() if p.is_dir()):
        runs = {}
        for f in sd.glob("*_seed*.json"):
            r = json.loads(f.read_text())
            runs[(r["condition"], r["seed"])] = r["summary"]
        seeds = sorted({s for c, s in runs if c == "error_off" and all((k, s) in runs for k in M)})
        if len(seeds) < 2:
            continue
        g = lambda c, key: float(np.mean([runs[(c, s)][key] for s in seeds]))
        row = {"n": len(seeds), "base": g("error_off", "mean_psnr")}
        for c in M:
            d = np.array([runs[(c, s)]["mean_psnr"] - runs[("error_off", s)]["mean_psnr"] for s in seeds])
            lo, hi = bootstrap_ci_95(d)
            row[c] = {"gain": float(d.mean()), "ci95": [float(lo), float(hi)]}
        row["map_k"] = {c: g(c, "N_final") / 1000 for c in ("error_off", "spawn_cap_always20", "vgg")}
        row["frame_ms"] = {c: g(c, "mean_frame_ms") for c in ("error_off", "spawn_cap_always20", "vgg")}
        scenes[sd.name] = row
        L.append(f"| {sd.name} | {len(seeds)} | {row['base']:.2f} | " + " | ".join(f"{row[c]['gain']:+.2f}" for c in M) +
                 " | " + "/".join(f"{row['map_k'][c]:.0f}" for c in row["map_k"]) + " | " + "/".join(f"{row['frame_ms'][c]:.0f}" for c in row["frame_ms"]) + " |")
    summ = {}
    for c in M:
        v = np.array([s[c]["gain"] for s in scenes.values()])
        best = np.array([max(0.0, s["spawn_cap_always20"]["gain"]) for s in scenes.values()])
        regret = best - v if c in ("spawn_cap_always20", "vgg") else None
        summ[c] = {"n_scenes": len(v), "n_positive": int((v > 0).sum()), "min": float(v.min()), "median": float(np.median(v)),
                   "mean": float(v.mean()), "max": float(v.max()),
                   "wilcoxon_p": float(wilcoxon(v).pvalue) if len(v) >= 6 and np.any(v != 0) else None}
        if regret is not None:
            summ[c]["mean_regret_vs_best_of_off_cap"] = float(regret.mean())
            summ[c]["max_regret_vs_best_of_off_cap"] = float(regret.max())
    L += ["", "| method | scenes | positive | worst | median | mean | max | mean regret | max regret |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for c, x in summ.items():
        mr = x.get("mean_regret_vs_best_of_off_cap")
        L.append(f"| {M[c]} | {x['n_scenes']} | {x['n_positive']} | {x['min']:+.2f} | {x['median']:+.2f} | {x['mean']:+.2f} | {x['max']:+.2f} | "
                 + (f"{mr:.2f} | {x['max_regret_vs_best_of_off_cap']:.2f} |" if mr is not None else "- | - |"))
    out = {"scenes": scenes, "summary": summ}
    (D / "analysis.json").write_text(json.dumps(out, indent=2))
    (D / "analysis.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
