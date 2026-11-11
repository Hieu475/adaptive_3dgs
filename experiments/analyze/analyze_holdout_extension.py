#!/usr/bin/env python3
"""Train-vs-held-out PSNR changes for the budget and throttling controls (results/holdout/holdout_extension).

Held-out frames are every 8th frame, rendered but never mapped (same-trajectory proxy). Deltas
are seed-paired relative to unthrottled raw-error selection.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run.run_final_confirmation import bootstrap_ci_95  # noqa: E402

D = ROOT / "results" / "holdout" / "holdout_extension"
CONDS = ("error_coverage", "spawn_cap_always20", "error_budget_prop")


def main():
    out, L = {}, ["# Held-out views", "", "| scene | condition | train dPSNR | held-out dPSNR [95% CI] | gap (train-holdout) | base gap |", "|---|---|--:|---|--:|--:|"]
    for sd in sorted(p for p in D.iterdir() if p.is_dir()):
        runs = {}
        for f in sd.glob("*_seed*.json"):
            r = json.loads(f.read_text())
            runs[(r["condition"], r["seed"])] = r
        seeds = sorted({s for c, s in runs if c == "error_off"})
        base_gap = float(np.mean([runs[("error_off", s)]["train_psnr"] - runs[("error_off", s)]["holdout_psnr"] for s in seeds]))
        out[sd.name] = {"n": len(seeds), "base_gap": base_gap, "base_holdout": float(np.mean([runs[("error_off", s)]["holdout_psnr"] for s in seeds]))}
        for c in CONDS:
            tr = np.array([runs[(c, s)]["train_psnr"] - runs[("error_off", s)]["train_psnr"] for s in seeds])
            ho = np.array([runs[(c, s)]["holdout_psnr"] - runs[("error_off", s)]["holdout_psnr"] for s in seeds])
            gap = float(np.mean([runs[(c, s)]["train_psnr"] - runs[(c, s)]["holdout_psnr"] for s in seeds]))
            lo, hi = bootstrap_ci_95(ho)
            out[sd.name][c] = {"train_delta": float(tr.mean()), "holdout_delta": float(ho.mean()),
                               "holdout_ci95": [float(lo), float(hi)], "gap": gap,
                               "retained": float(ho.mean() / tr.mean()) if tr.mean() > 0 else None}
            L.append(f"| {sd.name} | {c} | {tr.mean():+.2f} | {ho.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {gap:+.2f} | {base_gap:+.2f} |")
    (D / "analysis.json").write_text(json.dumps(out, indent=2))
    (D / "analysis.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
