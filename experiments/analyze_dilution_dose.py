#!/usr/bin/env python3
"""Does quality collapse onto budget-per-Gaussian across free and capped maps?

Per scene and per seed we fit  PSNR ~ a + b*log10(B/N) + c*[capped]  over the six cells
(3 budget multipliers x {free, cap}). b is the dose-response slope; c is the residual
offset of the capped regime once budget-per-Gaussian is accounted for. Intervals are
seed-bootstrap (repeatability on a fixed trajectory, not scene generalization).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run_final_confirmation import bootstrap_ci_95  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "dilution_dose"


def main():
    result, lines = {}, ["# Budget-per-Gaussian dose-response", ""]
    for sd in sorted(p for p in OUT.iterdir() if p.is_dir()):
        runs = {}
        for f in sd.glob("*_seed*.json"):
            d = json.loads(f.read_text())
            runs[(d["condition"], d["seed"])] = d["summary"]
        seeds = sorted({s for _, s in runs})
        conds = [f"{r}_x{m}" for r in ("free", "cap") for m in (1, 2, 4)]
        full = [s for s in seeds if all((c, s) in runs for c in conds)]
        if len(full) < 3:
            continue
        lines += [f"## {sd.name} (n={len(full)} complete seeds)", "",
                  "| cell | PSNR | depth L1 | map | B/N (us) | mature cov. |", "|---|--:|--:|--:|--:|--:|"]
        table = {}
        for c in conds:
            g = lambda k: float(np.mean([runs[(c, s)][k] for s in full]))
            table[c] = {k: g(k) for k in ("mean_psnr", "mean_depth_l1", "N_final",
                                          "mean_budget_per_gaussian_us", "mature_update_coverage")}
            t = table[c]
            lines.append(f"| {c} | {t['mean_psnr']:.2f} | {t['mean_depth_l1']:.4f} | {t['N_final']:.0f} | "
                         f"{t['mean_budget_per_gaussian_us']:.3f} | {t['mature_update_coverage']:.3f} |")
        slopes, offsets, slopes_cov = [], [], []
        for s in full:
            x = np.array([np.log10(runs[(c, s)]["mean_budget_per_gaussian_us"]) for c in conds])
            y = np.array([runs[(c, s)]["mean_psnr"] for c in conds])
            cap = np.array([1.0 if c.startswith("cap") else 0.0 for c in conds])
            A = np.stack([np.ones_like(x), x, cap], axis=1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            slopes.append(coef[1])
            offsets.append(coef[2])
            # slope within each regime separately (does the dose-response bend differently?)
            for reg in (0.0, 1.0):
                m = cap == reg
                if m.sum() >= 3:
                    slopes_cov.append((reg, np.polyfit(x[m], y[m], 1)[0]))
        sl, of = np.array(slopes), np.array(offsets)
        sl_ci, of_ci = bootstrap_ci_95(sl), bootstrap_ci_95(of)
        free_sl = np.mean([v for r, v in slopes_cov if r == 0.0])
        cap_sl = np.mean([v for r, v in slopes_cov if r == 1.0])
        res = {"cells": table, "pooled_slope_db_per_decade": float(sl.mean()),
               "pooled_slope_ci95": [float(sl_ci[0]), float(sl_ci[1])],
               "cap_offset_db": float(of.mean()), "cap_offset_ci95": [float(of_ci[0]), float(of_ci[1])],
               "free_slope_db_per_decade": float(free_sl), "cap_slope_db_per_decade": float(cap_sl),
               "seeds": full}
        result[sd.name] = res
        lines += ["", f"Pooled slope: {res['pooled_slope_db_per_decade']:+.2f} dB per decade of B/N "
                      f"[{sl_ci[0]:+.2f}, {sl_ci[1]:+.2f}]; capped-regime offset after accounting for B/N: "
                      f"{res['cap_offset_db']:+.2f} dB [{of_ci[0]:+.2f}, {of_ci[1]:+.2f}]; "
                      f"within-regime slopes free {free_sl:+.2f} / cap {cap_sl:+.2f}.", ""]
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    (OUT / "analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
