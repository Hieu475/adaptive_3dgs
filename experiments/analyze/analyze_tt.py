#!/usr/bin/env python3
"""Final evaluation of the time-targeted budget (TT) and value-gated growth (VGG).

Usage: analyze_tt.py <results dir> [--fresh]   (default results/controllers/vgg_test; --fresh flattens
results/controllers/vgg_fresh/<scene>/<scene>/...). Hyperparameters were frozen on development scenes
(results/controllers/vgg_dev, results/controllers/tt_dev) before these runs.

Per scene (seed-paired vs. the unthrottled baseline at the modeled 15 ms budget): PSNR change and
mean frame time of the fixed cap, the map-proportional budget, VGG alone, TT, TT+VGG and TT+cap.
Across scenes: sign count, worst case, median, mean; the contribution of VGG at equal controller
(TT+VGG minus TT); and the fraction of scenes where TT+VGG is better *and* not slower than baseline.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run.run_final_confirmation import bootstrap_ci_95  # noqa: E402

METHODS = {"spawn_cap_always20": "fixed cap", "error_budget_prop": "prop. budget", "vgg": "VGG", "vgg_prop": "VGG+prop.",
           "tt": "TT", "tt_vgg": "TT+VGG", "tt_cap": "TT+cap"}


def load(root, fresh):
    scenes = {}
    for sd in sorted(p for p in root.iterdir() if p.is_dir()):
        dirs = [sd / sd.name] if fresh and (sd / sd.name).is_dir() else [sd]
        runs = {}
        for d in dirs:
            for f in d.glob("*_seed*.json"):
                r = json.loads(f.read_text())
                runs[(r["condition"], r["seed"])] = r["summary"]
        scenes[sd.name] = runs
    return scenes


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ROOT / "results" / "controllers" / "vgg_test"
    fresh = "--fresh" in sys.argv
    L, out = [f"# {root.name}", ""], {}
    shown = [m for m in ("tt", "tt_vgg", "tt_cap", "spawn_cap_always20", "error_budget_prop", "vgg") ]
    L += ["| scene | n | base PSNR (ms) | " + " | ".join(METHODS[m] for m in shown) + " |", "|---|--:|--|" + "--|" * len(shown)]
    for name, runs in load(root, fresh).items():
        seeds = sorted({s for c, s in runs if c == "error_off" and all((k, s) in runs for k in ("tt", "tt_vgg"))})
        if len(seeds) < 2:
            continue
        g = lambda c, k: float(np.mean([runs[(c, s)][k] for s in seeds]))
        row = {"n": len(seeds), "base_psnr": g("error_off", "mean_psnr"), "base_ms": g("error_off", "mean_frame_ms"), "methods": {}}
        for m in METHODS:
            if all((m, s) in runs for s in seeds):
                d = np.array([runs[(m, s)]["mean_psnr"] - runs[("error_off", s)]["mean_psnr"] for s in seeds])
                lo, hi = bootstrap_ci_95(d)
                row["methods"][m] = {"gain": float(d.mean()), "ci95": [float(lo), float(hi)], "ms": g(m, "mean_frame_ms"),
                                     "map_k": g(m, "N_final") / 1000}
        d = np.array([runs[("tt_vgg", s)]["mean_psnr"] - runs[("tt", s)]["mean_psnr"] for s in seeds])
        row["vgg_over_tt"] = float(d.mean())
        out[name] = row
        L.append(f"| {name} | {len(seeds)} | {row['base_psnr']:.2f} ({row['base_ms']:.0f}) | " +
                 " | ".join((f"{row['methods'][m]['gain']:+.2f} ({row['methods'][m]['ms']:.0f})" if m in row["methods"] else "-") for m in shown) + " |")
    summ = {}
    for m in METHODS:
        v = np.array([s["methods"][m]["gain"] for s in out.values() if m in s["methods"]])
        if len(v):
            ms = np.array([s["methods"][m]["ms"] - s["base_ms"] for s in out.values() if m in s["methods"]])
            summ[m] = {"n_scenes": len(v), "n_positive": int((v > 0).sum()), "min": float(v.min()), "median": float(np.median(v)),
                       "mean": float(v.mean()), "max": float(v.max()), "mean_time_delta_ms": float(ms.mean()),
                       "wilcoxon_p": float(wilcoxon(v).pvalue) if len(v) >= 6 and np.any(v != 0) else None}
    vv = np.array([s["vgg_over_tt"] for s in out.values()])
    summ["vgg_over_tt"] = {"n_scenes": len(vv), "n_positive": int((vv > 0).sum()), "min": float(vv.min()), "median": float(np.median(vv)), "mean": float(vv.mean())}
    dom = [s["methods"]["tt_vgg"]["gain"] > 0 and s["methods"]["tt_vgg"]["ms"] <= s["base_ms"] * 1.15 for s in out.values()]
    summ["tt_vgg_better_and_within_15pct_time"] = {"n": int(sum(dom)), "of": len(dom)}
    L += ["", "| method | scenes | positive | worst | median | mean | best | mean dt (ms) |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for m in METHODS:
        if m in summ:
            x = summ[m]
            L.append(f"| {METHODS[m]} | {x['n_scenes']} | {x['n_positive']} | {x['min']:+.2f} | {x['median']:+.2f} | {x['mean']:+.2f} | {x['max']:+.2f} | {x['mean_time_delta_ms']:+.0f} |")
    x = summ["vgg_over_tt"]
    L += ["", f"TT+VGG minus TT (same controller): positive in {x['n_positive']}/{x['n_scenes']} scenes, worst {x['min']:+.2f}, median {x['median']:+.2f}, mean {x['mean']:+.2f} dB.",
          f"TT+VGG better than baseline and within +15% of its frame time: {summ['tt_vgg_better_and_within_15pct_time']['n']}/{summ['tt_vgg_better_and_within_15pct_time']['of']} scenes."]
    (root / "analysis_tt.json").write_text(json.dumps({"scenes": out, "summary": summ}, indent=2))
    (root / "analysis_tt.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
