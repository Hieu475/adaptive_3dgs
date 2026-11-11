#!/usr/bin/env python3
"""Seed-paired ATE / PSNR contrasts vs error_off for results/tracking/closed_loop (map-based tracking)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "results" / "tracking" / "closed_loop"


def main():
    out, L = {}, ["| scene | cond | dPSNR | dATE (cm) | ATE base (cm) | track ms | map ms |", "|---|---|--:|--:|--:|--:|--:|"]
    for sd in sorted(p for p in D.iterdir() if p.is_dir()):
        r = {}
        for f in sd.glob("*_seed*.json"):
            j = json.loads(f.read_text()); r[(j["condition"], j["seed"])] = j
        seeds = sorted({s for c, s in r if c == "error_off"})
        for c in sorted({c for c, _ in r} - {"error_off"}):
            ss = [s for s in seeds if (c, s) in r]
            dp = np.mean([r[(c, s)]["psnr_tracked"] - r[("error_off", s)]["psnr_tracked"] for s in ss])
            da = np.mean([100 * (r[(c, s)]["ate_rmse_m"] - r[("error_off", s)]["ate_rmse_m"]) for s in ss])
            base = np.mean([100 * r[("error_off", s)]["ate_rmse_m"] for s in ss])
            out[f"{sd.name}/{c}"] = {"dpsnr": float(dp), "date_cm": float(da), "base_ate_cm": float(base), "n": len(ss)}
            L.append(f"| {sd.name} | {c} | {dp:+.2f} | {da:+.2f} | {base:.2f} | "
                     f"{np.mean([r[(c, s)]['track_ms'] for s in ss]):.0f} | {np.mean([r[(c, s)]['map_ms'] for s in ss]):.0f} |")
    summ = {}
    for c in ("tt_vgg", "tt", "error_budget_prop"):
        v = [x for k, x in out.items() if k.endswith("/" + c) and x["base_ate_cm"] < 50]  # exclude diverged tracking
        if v:
            summ[c] = {"n_scenes": len(v), "mean_dpsnr": float(np.mean([x["dpsnr"] for x in v])),
                       "median_date_cm": float(np.median([x["date_cm"] for x in v])),
                       "max_abs_date_cm": float(max(abs(x["date_cm"]) for x in v)),
                       "n_psnr_pos": int(sum(x["dpsnr"] > 0 for x in v))}
            L.append(f"\n{c} (scenes with converged tracking, n={len(v)}): mean dPSNR {summ[c]['mean_dpsnr']:+.2f}, "
                     f"median dATE {summ[c]['median_date_cm']:+.2f} cm, max |dATE| {summ[c]['max_abs_date_cm']:.2f} cm")
    out["_summary"] = summ
    (D / "analysis.json").write_text(json.dumps(out, indent=1))
    (D / "analysis.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
