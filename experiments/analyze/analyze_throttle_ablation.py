#!/usr/bin/env python3
"""Paired contrasts for results/dilution/throttle_ablation (decomposition of coverage throttling).

Per scene: seed-paired change in mean PSNR (and depth L1, map size) of every condition relative
to the unthrottled raw-error baseline, plus full-throttle vs each component. Four seeds per
scene, so intervals are repeatability on a fixed trajectory, not scene generalization.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run.run_final_confirmation import bootstrap_ci_95  # noqa: E402

D = ROOT / "results" / "dilution" / "throttle_ablation"
CONDS = ("error_coverage", "cov_spawn_only", "cov_thresh_only", "spawn_cap_always20", "spawn_cap_always50",
         "error_budget_prop", "cov_plus_prop")


def load(scene_dir):
    runs = {}
    for f in scene_dir.glob("*_seed*.json"):
        r = json.loads(f.read_text())
        runs[(r["condition"], r["seed"])] = r["summary"]
    return runs


def main():
    out, L = {}, ["# Coverage-throttling decomposition", ""]
    for sd in sorted(p for p in D.iterdir() if p.is_dir()):
        runs = load(sd)
        seeds = sorted({s for c, s in runs if c == "error_off"})
        res = {"seeds": seeds, "base_psnr": float(np.mean([runs[("error_off", s)]["mean_psnr"] for s in seeds])),
               "contrasts": {}}
        L += [f"## {sd.name} (n={len(seeds)})", "", f"Baseline (error_off) PSNR {res['base_psnr']:.2f}", "",
              "| condition | dPSNR [95% CI] | dDepth L1 | map (k) | spawn/frame |", "|---|---|--:|--:|--:|"]
        for c in CONDS:
            ss = [s for s in seeds if (c, s) in runs]
            if len(ss) < 2:
                continue
            d = np.array([runs[(c, s)]["mean_psnr"] - runs[("error_off", s)]["mean_psnr"] for s in ss])
            dd = np.array([runs[(c, s)]["mean_depth_l1"] - runs[("error_off", s)]["mean_depth_l1"] for s in ss])
            lo, hi = bootstrap_ci_95(d)
            m = float(np.mean([runs[(c, s)]["N_final"] for s in ss]) / 1000)
            sp = float(np.mean([runs[(c, s)]["mean_n_spawned"] for s in ss]))
            res["contrasts"][c] = {"mean_db": float(d.mean()), "ci95": [float(lo), float(hi)],
                                   "depth_l1_delta": float(dd.mean()), "map_k": m, "spawn_per_frame": sp,
                                   "n": len(ss)}
            L.append(f"| {c} | {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {dd.mean():+.4f} | {m:.1f} | {sp:.0f} |")
        out[sd.name] = res
        L.append("")
    (D / "analysis.json").write_text(json.dumps(out, indent=2))
    (D / "analysis.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
