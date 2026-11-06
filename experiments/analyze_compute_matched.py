#!/usr/bin/env python3
"""Quality at equal measured wall-clock: free growth vs spawn cap x0.2 across budget multipliers.

Per scene, cells (regime, multiplier) are averaged over seeds. For each spawn-cap cell we
linearly interpolate the free-growth curve (PSNR vs mean end-to-end frame time) at the same
time, and report the difference (positive: the cap is better at equal time). The same is done for
optimization time. Interpolation is only reported inside the free-growth time range.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results" / "compute_matched"
MULTS = (1, 2, 4)


def interp(x, xs, ys):
    order = np.argsort(xs)
    xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def main():
    out, L = {}, ["# Equal measured time", ""]
    for sd in sorted(p for p in D.iterdir() if p.is_dir()):
        cells = {}
        for r in ("free", "spawn"):
            for m in MULTS:
                rs = [json.loads(f.read_text())["summary"] for f in sd.glob(f"{r}_x{m}_seed*.json")]
                if rs:
                    cells[f"{r}_x{m}"] = {k: float(np.mean([x[k] for x in rs])) for k in
                                          ("mean_psnr", "mean_frame_ms", "mean_opt_ms", "N_final", "mature_update_coverage")}
                    cells[f"{r}_x{m}"]["n"] = len(rs)
        if len(cells) < 6:
            continue
        free = [cells[f"free_x{m}"] for m in MULTS]
        matched = {}
        for m in MULTS:
            c = cells[f"spawn_x{m}"]
            e_frame = interp(c["mean_frame_ms"], [f["mean_frame_ms"] for f in free], [f["mean_psnr"] for f in free])
            e_opt = interp(c["mean_opt_ms"], [f["mean_opt_ms"] for f in free], [f["mean_psnr"] for f in free])
            matched[f"spawn_x{m}"] = {"free_psnr_at_same_frame_ms": e_frame, "gain_vs_free_frame": None if e_frame is None else c["mean_psnr"] - e_frame,
                                      "free_psnr_at_same_opt_ms": e_opt, "gain_vs_free_opt": None if e_opt is None else c["mean_psnr"] - e_opt}
        out[sd.name] = {"cells": cells, "matched": matched}
        L += [f"## {sd.name}", "", "| cell | PSNR | frame ms | opt ms | map (k) | mature cov. |", "|---|--:|--:|--:|--:|--:|"]
        for k, c in cells.items():
            L.append(f"| {k} | {c['mean_psnr']:.2f} | {c['mean_frame_ms']:.0f} | {c['mean_opt_ms']:.0f} | {c['N_final'] / 1000:.0f} | {c['mature_update_coverage']:.2f} |")
        L += ["", "| spawn cell | free PSNR at same frame time | gain | free PSNR at same opt time | gain |", "|---|--:|--:|--:|--:|"]
        f = lambda v: "out of range" if v is None else f"{v:.2f}"
        g = lambda v: "n/a" if v is None else f"{v:+.2f}"
        for k, v in matched.items():
            L.append(f"| {k} | {f(v['free_psnr_at_same_frame_ms'])} | {g(v['gain_vs_free_frame'])} | {f(v['free_psnr_at_same_opt_ms'])} | {g(v['gain_vs_free_opt'])} |")
        L.append("")
    (D / "analysis.json").write_text(json.dumps(out, indent=2))
    (D / "analysis.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
