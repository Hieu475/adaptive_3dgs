#!/usr/bin/env python3
"""Quality at equal measured time: TT+VGG (time targets T) vs fixed-budget free growth (x1/x2/x4).

For each scene, TT+VGG points come from results/tt_pareto (T = 70, 85, 120, 140 ms) and
results/vgg_test (T = 100 ms); the baseline curve is the free-growth cells of
results/compute_matched (modeled budget x1, x2, x4). For each TT+VGG point inside the baseline's
measured-time range, the baseline PSNR is linearly interpolated at the same mean frame time.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
SCENES = ["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0", "replica_office1"]
SEEDS = (42, 43, 44, 45)


def mean_cell(files):
    rs = [json.loads(f.read_text())["summary"] for f in files]
    return float(np.mean([r["mean_psnr"] for r in rs])), float(np.mean([r["mean_frame_ms"] for r in rs])), float(np.mean([r["N_final"] for r in rs]) / 1000), len(rs)


def main():
    out, L = {}, ["# TT+VGG vs fixed budget at equal measured time", ""]
    for sc in SCENES:
        base = {}
        for m in (1, 2, 4):
            fs = [R / f"compute_matched/{sc}/free_x{m}_seed{s}.json" for s in SEEDS]
            base[m] = mean_cell([f for f in fs if f.exists()])
        pts = {}
        for T in (70, 85, 120, 140):
            fs = sorted((R / f"tt_pareto/{sc}").glob(f"tt_vgg_T{T}_seed*.json"))
            if fs:
                pts[T] = mean_cell(fs)
        fs = sorted((R / f"vgg_test/{sc}").glob("tt_vgg_seed*.json"))
        pts[100] = mean_cell(fs)
        xs = np.array([base[m][1] for m in base]); ys = np.array([base[m][0] for m in base])
        order = np.argsort(xs); xs, ys = xs[order], ys[order]
        rows = {}
        L += [f"## {sc}", "", "baseline (free growth): " + ", ".join(f"x{m}: {v[0]:.2f} dB @ {v[1]:.0f} ms" for m, v in base.items()), "",
              "| T (ms) | TT+VGG PSNR | frame ms | map (k) | baseline PSNR at same time | gain |", "|--:|--:|--:|--:|--:|--:|"]
        for T in sorted(pts):
            psnr, ms, mk, n = pts[T]
            if xs[0] <= ms <= xs[-1]:
                b = float(np.interp(ms, xs, ys)); g = psnr - b
                bs, gs = f"{b:.2f}", f"{g:+.2f}"
            else:
                b, g = None, None
                # below the fastest baseline cell: compare with its x1 cell (the baseline cannot run faster)
                bs, gs = (f"{ys[0]:.2f} (x1 @ {xs[0]:.0f} ms)", f"{psnr - ys[0]:+.2f} and {xs[0] - ms:.0f} ms faster") if ms < xs[0] else ("beyond range", "n/a")
            rows[T] = {"psnr": psnr, "ms": ms, "map_k": mk, "baseline_psnr_at_time": b, "gain": g, "n": n}
            L.append(f"| {T} | {psnr:.2f} | {ms:.0f} | {mk:.0f} | {bs} | {gs} |")
        out[sc] = {"baseline": {m: {"psnr": v[0], "ms": v[1]} for m, v in base.items()}, "points": rows}
        L.append("")
    (R / "tt_pareto").mkdir(exist_ok=True)
    (R / "tt_pareto" / "analysis.json").write_text(json.dumps(out, indent=2))
    (R / "tt_pareto" / "analysis.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
