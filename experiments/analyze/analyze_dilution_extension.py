#!/usr/bin/env python3
"""Paired contrasts for results/dilution/dilution_extension (Replica factorial + proportional budget).

Writes analysis.json and analysis.md next to the runs. Paired by seed within scene;
CIs are seed-bootstrap (repeatability on a fixed trajectory, not scene generalization).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run.run_final_confirmation import bootstrap_ci_95, safe_wilcoxon  # noqa: E402

OUT = ROOT / "results" / "dilution" / "dilution_extension"
CONTRASTS = {
    "coverage on error": ("error_coverage", "error_off"),
    "coverage on temporal": ("temporal_coverage", "temporal_off"),
    "backlog on error": ("error_coverage_backlog", "error_coverage"),
    "backlog on temporal": ("temporal_coverage_backlog", "temporal_coverage"),
    "selector without throttle": ("temporal_off", "error_off"),
    "selector with coverage": ("temporal_coverage", "error_coverage"),
    "error budget x4": ("error_budget4", "error_off"),
    "error budget proportional to map": ("error_budget_prop", "error_off"),
    "proportional vs coverage (error)": ("error_budget_prop", "error_coverage"),
}
FIELDS = ("mean_psnr", "mean_depth_l1", "N_final", "mature_update_coverage",
          "young_budget_share", "mean_budget_scale", "mean_opt_time_ms")


def load(scene_dir):
    runs = {}
    for f in sorted(scene_dir.glob("*_seed*.json")):
        d = json.loads(f.read_text())
        runs[(d["condition"], d["seed"])] = d["summary"]
    return runs


def main():
    result = {}
    lines = ["# Dilution extension", ""]
    for scene_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        runs = load(scene_dir)
        if not runs:
            continue
        conds = sorted({c for c, _ in runs})
        seeds_by_cond = {c: sorted(s for cc, s in runs if cc == c) for c in conds}
        table = {}
        for c in conds:
            row = {"n": len(seeds_by_cond[c])}
            for fld in FIELDS:
                vals = [runs[(c, s)][fld] for s in seeds_by_cond[c] if fld in runs[(c, s)]]
                if vals:
                    row[fld] = float(np.mean(vals))
            table[c] = row
        contrasts = {}
        for name, (a, b) in CONTRASTS.items():
            common = sorted(set(seeds_by_cond.get(a, [])) & set(seeds_by_cond.get(b, [])))
            if len(common) < 3:
                continue
            d = np.array([runs[(a, s)]["mean_psnr"] - runs[(b, s)]["mean_psnr"] for s in common])
            dd = np.array([runs[(a, s)]["mean_depth_l1"] - runs[(b, s)]["mean_depth_l1"] for s in common])
            lo, hi = bootstrap_ci_95(d)
            contrasts[name] = {"seeds": common, "mean_db": float(d.mean()), "ci95_db": [float(lo), float(hi)],
                               "p": float(safe_wilcoxon(d, np.zeros_like(d))),
                               "depth_l1_delta": float(dd.mean()), "values_db": d.tolist()}
        result[scene_dir.name] = {"conditions": table, "contrasts": contrasts}
        lines += [f"## {scene_dir.name}", "", "| condition | n | PSNR | depth L1 | map | mature cov. | young share | budget scale |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for c in conds:
            r = table[c]
            lines.append(f"| {c} | {r['n']} | {r['mean_psnr']:.2f} | {r['mean_depth_l1']:.4f} | {r['N_final']:.0f} | "
                         f"{r['mature_update_coverage']:.3f} | {r['young_budget_share']:.3f} | "
                         f"{r.get('mean_budget_scale', 1.0):.2f} |")
        lines += ["", "| contrast | dPSNR (dB) | 95% CI | p | dDepth L1 | n |", "|---|--:|---|--:|--:|--:|"]
        for n, c in contrasts.items():
            lines.append(f"| {n} | {c['mean_db']:+.2f} | [{c['ci95_db'][0]:+.2f}, {c['ci95_db'][1]:+.2f}] | "
                         f"{c['p']:.4f} | {c['depth_l1_delta']:+.4f} | {len(c['seeds'])} |")
        lines.append("")
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    (OUT / "analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
