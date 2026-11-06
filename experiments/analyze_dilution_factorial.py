#!/usr/bin/env python3
"""Summarize paired corrected-substrate factorial runs and frame-65 traces."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
CONTRASTS = (
    ("coverage on error", "error_coverage", "error_off"),
    ("coverage on temporal", "temporal_coverage", "temporal_off"),
    ("backlog on error", "error_coverage_backlog", "error_coverage"),
    ("backlog on temporal", "temporal_coverage_backlog", "temporal_coverage"),
    ("selector without throttle", "temporal_off", "error_off"),
    ("selector with coverage", "temporal_coverage", "error_coverage"),
    ("error budget x4", "error_budget4", "error_off"),
    ("error cap 23k", "error_cap23k", "error_off"),
    ("error cap 39k", "error_cap39k", "error_off"),
)


def ci_paired(values, seed=2026):
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(x), size=(10000, len(x)))
    return [float(v) for v in np.percentile(x[idx].mean(axis=1), [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results" / "dilution_factorial")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    output = args.output or args.input / "analysis.json"
    records = [json.loads(p.read_text()) for p in args.input.glob("*/*.json")]
    by_scene = {}
    for rec in records:
        by_scene.setdefault(rec["scene"], {})[(rec["seed"], rec["condition"])] = rec
    analysis = {}
    lines = ["# Corrected-pose selector × throttle factorial", "",
             "Mean PSNR uses 150 evaluated frames. CI is paired seed bootstrap; p is two-sided paired Wilcoxon.", ""]
    for scene, data in by_scene.items():
        lines += [f"## {scene}", "", "| Condition | n | PSNR | Depth L1 | Map N | Young budget share | Old selected / old population | Spawn/frame |",
                  "|:--|--:|--:|--:|--:|--:|--:|--:|"]
        conditions = sorted({c for (_, c) in data})
        rows = {}
        for cond in conditions:
            runs = [v for (seed, c), v in data.items() if c == cond]
            rows[cond] = {"n": len(runs)}
            for key in ("mean_psnr", "mean_depth_l1", "N_final", "young_budget_share", "mean_opt_time_ms"):
                rows[cond][key] = float(np.mean([r["summary"][key] for r in runs]))
            for key in ("selected_young", "selected_old", "population_young", "population_old", "n_spawned", "loss", "mean_opacity"):
                rows[cond][f"mean_{key}"] = float(np.mean([np.mean([f[key] for f in r["frames"]]) for r in runs]))
            rows[cond]["backlog_trigger_frames"] = int(sum(
                sum(f["n_backlog"] > 500 for f in r["frames"]) for r in runs))
            old_selected = sum(f["selected_old"] for r in runs for f in r["frames"])
            old_population = sum(f["population_old"] for r in runs for f in r["frames"])
            rows[cond]["old_selection_rate"] = old_selected / old_population if old_population else 0.0
            a = rows[cond]
            lines.append(f"| {cond} | {a['n']} | {a['mean_psnr']:.3f} | "
                         f"{a['mean_depth_l1']:.4f} | {a['N_final']:.0f} | "
                         f"{a['young_budget_share']:.3f} | {a['old_selection_rate']:.3f} | "
                         f"{a['mean_n_spawned']:.0f} |")
        lines += ["", "| Contrast | n | Δ PSNR | 95% CI | p |", "|:--|--:|--:|:--|--:|"]
        contrasts = {}
        for name, a, b in CONTRASTS:
            seeds = sorted({seed for seed, c in data if c == a and (seed, b) in data})
            if not seeds:
                continue
            diff = np.asarray([data[(s, a)]["summary"]["mean_psnr"] -
                               data[(s, b)]["summary"]["mean_psnr"] for s in seeds])
            p = float(wilcoxon(diff).pvalue) if len(diff) > 1 and np.any(diff) else None
            ci = ci_paired(diff)
            contrasts[name] = {"seeds": seeds, "mean_db": float(diff.mean()),
                               "ci95_db": ci, "p": p, "values_db": diff.tolist()}
            lines.append(f"| {name} | {len(seeds)} | {diff.mean():+.3f} | "
                         f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {p if p is not None else '—'} |")
        required = ("error_off", "error_coverage", "temporal_off", "temporal_coverage")
        seeds = sorted({s for s, _ in data if all((s, c) in data for c in required)})
        if seeds:
            did = np.asarray([
                (data[(s, "temporal_coverage")]["summary"]["mean_psnr"] -
                 data[(s, "temporal_off")]["summary"]["mean_psnr"]) -
                (data[(s, "error_coverage")]["summary"]["mean_psnr"] -
                 data[(s, "error_off")]["summary"]["mean_psnr"])
                for s in seeds])
            ci = ci_paired(did)
            p = float(wilcoxon(did).pvalue) if len(did) > 1 and np.any(did) else None
            contrasts["selector x coverage interaction"] = {
                "seeds": seeds, "mean_db": float(did.mean()),
                "ci95_db": ci, "p": p, "values_db": did.tolist()}
            lines.append(f"| selector × coverage interaction | {len(seeds)} | {did.mean():+.3f} | "
                         f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {p if p is not None else '—'} |")
        primary = ("coverage on error", "coverage on temporal", "error budget x4", "error cap 23k")
        indexed = sorted(((name, contrasts[name]["p"]) for name in primary
                          if name in contrasts and contrasts[name]["p"] is not None),
                         key=lambda x: x[1])
        running = 0.0
        for rank, (name, p) in enumerate(indexed):
            running = max(running, min(1.0, p * (len(indexed) - rank)))
            contrasts[name]["holm_p_primary_family"] = running
        # The frame-65 diagnosis compares direct observations, not a causal model.
        traces = {}
        for cond in ("error_off", "error_coverage", "temporal_coverage"):
            candidates = [r for (seed, c), r in data.items() if c == cond and seed == 42]
            if not candidates:
                continue
            f = candidates[0]["frames"]
            window = [r for r in f if 55 <= r["frame"] <= 75]
            drops = sorted(((f[i]["psnr"] - f[i-1]["psnr"], f[i]["frame"])
                            for i in range(1, len(f))), key=lambda x: x[0])[:5]
            traces[cond] = {"window": window, "largest_psnr_drops": drops}
        analysis[scene] = {"conditions": rows, "contrasts": contrasts, "frame65_seed42": traces}
        lines += ["", "Largest PSNR drops for seed 42 (frame, Δ dB): " +
                  "; ".join(f"{c}: {[(frame, round(d, 2)) for d, frame in t['largest_psnr_drops']]}"
                            for c, t in traces.items()), ""]
    output.write_text(json.dumps(analysis, indent=2))
    output.with_suffix(".md").write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
