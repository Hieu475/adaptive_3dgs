#!/usr/bin/env python3
"""Paired effects under estimated poses (results/tracked_poses)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run_final_confirmation import bootstrap_ci_95  # noqa: E402

D = ROOT / "results" / "tracked_poses"
CONDS = ("error_coverage", "spawn_cap_always20", "error_budget_prop")


def main():
    out, L = {}, ["# Estimated poses", "", "| scene | condition | tracked-pose dPSNR | GT-pose dPSNR [95% CI] | ATE (cm) |", "|---|---|--:|---|--:|"]
    for sd in sorted(p for p in D.iterdir() if p.is_dir()):
        runs = {}
        for f in sd.glob("*_seed*.json"):
            r = json.loads(f.read_text())
            runs[(r["condition"], r["seed"])] = r
        seeds = sorted({s for c, s in runs if c == "error_off"})
        base = {"psnr_tracked": float(np.mean([runs[("error_off", s)]["psnr_tracked"] for s in seeds])),
                "psnr_at_gt_pose": float(np.mean([runs[("error_off", s)]["psnr_at_gt_pose"] for s in seeds])),
                "ate_cm": float(np.mean([runs[("error_off", s)]["ate_rmse_m"] for s in seeds]) * 100)}
        out[sd.name] = {"n": len(seeds), "base": base}
        for c in CONDS:
            ss = [s for s in seeds if (c, s) in runs]
            tr = np.array([runs[(c, s)]["psnr_tracked"] - runs[("error_off", s)]["psnr_tracked"] for s in ss])
            gt = np.array([runs[(c, s)]["psnr_at_gt_pose"] - runs[("error_off", s)]["psnr_at_gt_pose"] for s in ss])
            lo, hi = bootstrap_ci_95(gt)
            ate = float(np.mean([runs[(c, s)]["ate_rmse_m"] for s in ss]) * 100)
            out[sd.name][c] = {"tracked_delta": float(tr.mean()), "gt_delta": float(gt.mean()), "gt_ci95": [float(lo), float(hi)], "ate_cm": ate}
            L.append(f"| {sd.name} | {c} | {tr.mean():+.2f} | {gt.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {ate:.2f} |")
        L.append(f"| {sd.name} | error_off (base) | {base['psnr_tracked']:.2f} | {base['psnr_at_gt_pose']:.2f} | {base['ate_cm']:.2f} |")
    (D / "analysis.json").write_text(json.dumps(out, indent=2))
    (D / "analysis.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
