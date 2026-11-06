#!/usr/bin/env python3
"""Scene-level analysis of the proportional budget and coverage throttling.

Unit of analysis is the scene. For each scene we compute the seed-paired change in mean PSNR
of (a) the map-proportional budget and (b) coverage throttling, both relative to the
unthrottled raw-error baseline from the same batch. Across scenes we report the sign count,
an exact binomial sign test and a Wilcoxon signed-rank test (two-sided).

Sources (all from the commit tagged paper-rerun-1 or later runs of the same code):
  results/dilution_scenes/      11 additional scenes, 3 seeds, 99 frames (all three conditions)
  results/dilution_extension/   Replica office0/office1 (8 seeds, 150 frames), TUM fr2/fr1 (prop. budget)
  results/rerun_paper1/dilution_factorial/   TUM fr2/fr1 coverage contrast (8 seeds, 150 frames)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run_final_confirmation import bootstrap_ci_95  # noqa: E402

R = ROOT / "results"


def psnr_by_seed(d, cond):
    out = {}
    for f in sorted(Path(d).glob(f"{cond}_seed*.json")):
        r = json.loads(f.read_text())
        out[r["seed"]] = (r["summary"]["mean_psnr"], r["summary"]["mean_depth_l1"], r["summary"]["N_final"])
    return out


def contrast(a, b):
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    d = np.array([a[s][0] - b[s][0] for s in seeds])
    dd = np.array([a[s][1] - b[s][1] for s in seeds])
    lo, hi = bootstrap_ci_95(d)
    return {"n": len(seeds), "mean_db": float(d.mean()), "ci95": [float(lo), float(hi)],
            "depth_l1_delta": float(dd.mean()), "all_positive": bool((d > 0).all())}


def main():
    scenes = {}
    for sd in sorted((R / "dilution_scenes").glob("*/")):
        if not sd.is_dir():
            continue
        off, cov, prop = (psnr_by_seed(sd, c) for c in ("error_off", "error_coverage", "error_budget_prop"))
        scenes[sd.name] = {"prop": contrast(prop, off), "cov": contrast(cov, off), "frames": 99,
                           "off_psnr": float(np.mean([v[0] for v in off.values()])) if off else None,
                           "off_map": float(np.mean([v[2] for v in off.values()])) if off else None}
    for sc in ("replica_office0", "replica_office1"):
        d = R / "dilution_extension" / sc
        off, cov, prop = (psnr_by_seed(d, c) for c in ("error_off", "error_coverage", "error_budget_prop"))
        scenes[sc] = {"prop": contrast(prop, off), "cov": contrast(cov, off), "frames": 150,
                      "off_psnr": float(np.mean([v[0] for v in off.values()])),
                      "off_map": float(np.mean([v[2] for v in off.values()]))}
    for sc in ("tum_fr2_xyz", "tum_fr1_xyz"):
        off_x = psnr_by_seed(R / "dilution_extension" / sc, "error_off")
        prop = psnr_by_seed(R / "dilution_extension" / sc, "error_budget_prop")
        fd = R / "rerun_paper1/dilution_factorial" / sc
        off_f, cov = psnr_by_seed(fd, "error_off"), psnr_by_seed(fd, "error_coverage")
        scenes[sc] = {"prop": contrast(prop, off_x), "cov": contrast(cov, off_f), "frames": 150,
                      "off_psnr": float(np.mean([v[0] for v in off_x.values()])),
                      "off_map": float(np.mean([v[2] for v in off_x.values()]))}
    summary = {}
    for key in ("prop", "cov"):
        vals = [(s, v[key]["mean_db"]) for s, v in scenes.items() if v[key]]
        d = np.array([x for _, x in vals])
        pos = int((d > 0).sum())
        summary[key] = {"n_scenes": len(d), "n_positive": pos, "median_db": float(np.median(d)),
                        "min_db": float(d.min()), "max_db": float(d.max()), "mean_db": float(d.mean()),
                        "sign_test_p": float(binomtest(pos, len(d), 0.5).pvalue),
                        "wilcoxon_p": float(wilcoxon(d).pvalue) if len(d) >= 6 else None,
                        "worst": min(vals, key=lambda t: t[1])[0], "best": max(vals, key=lambda t: t[1])[0]}
    from scipy.stats import spearmanr
    xs = [(v["off_map"], v["prop"]["mean_db"], v["cov"]["mean_db"]) for v in scenes.values() if v["prop"] and v["cov"]]
    rho_p = spearmanr([x[0] for x in xs], [x[1] for x in xs])
    rho_c = spearmanr([x[0] for x in xs], [x[2] for x in xs])
    summary["map_size_vs_prop_gain"] = {"spearman": float(rho_p[0]), "p": float(rho_p[1]), "n": len(xs)}
    summary["map_size_vs_cov_gain"] = {"spearman": float(rho_c[0]), "p": float(rho_c[1]), "n": len(xs)}
    out = {"scenes": scenes, "summary": summary}
    (R / "dilution_scenes" / "analysis.json").write_text(json.dumps(out, indent=2))
    L = ["# Scene-level proportional budget and coverage throttling", "",
         "| scene | frames | n seeds | base PSNR | prop. budget dPSNR [95% CI] | coverage dPSNR [95% CI] |", "|---|--:|--:|--:|---|---|"]
    for s, v in scenes.items():
        p, c = v["prop"], v["cov"]
        fmt = lambda x: f"{x['mean_db']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]" if x else "n/a"
        L.append(f"| {s} | {v['frames']} | {p['n'] if p else 0} | {v['off_psnr']:.2f} | {fmt(p)} | {fmt(c)} |")
    L += ["", "| contrast | scenes | positive | median | range | sign-test p | Wilcoxon p |", "|---|--:|--:|--:|---|--:|--:|"]
    for k, nm in (("prop", "proportional budget"), ("cov", "coverage throttling")):
        x = summary[k]
        wp = f"{x['wilcoxon_p']:.4f}" if x["wilcoxon_p"] is not None else "n/a"
        L.append(f"| {nm} | {x['n_scenes']} | {x['n_positive']} | {x['median_db']:+.2f} | "
                 f"[{x['min_db']:+.2f}, {x['max_db']:+.2f}] | {x['sign_test_p']:.4f} | {wp} |")
    m = summary["map_size_vs_prop_gain"]
    L += ["", f"Spearman(unthrottled final map size, proportional-budget gain) = {m['spearman']:+.2f} (p={m['p']:.3f}, n={m['n']})."]
    (R / "dilution_scenes" / "analysis.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
