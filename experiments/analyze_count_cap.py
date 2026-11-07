#!/usr/bin/env python3
"""TT+VGG vs matched-size hard count caps (results/count_cap vs results/vgg_test), seed-paired."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
C, V = ROOT / "results" / "count_cap", ROOT / "results" / "vgg_test"


def ld(p):
    return json.loads(p.read_text())["summary"]


def main():
    out, L = {}, ["| scene | TT+VGG | TT+countcap | countcap | (all vs error_off, dB) | frame ms (VGG / TT+cap) |", "|---|--:|--:|--:|---|---|"]
    for sd in sorted(p for p in C.iterdir() if p.is_dir()):
        seeds = sorted(int(f.stem.split("seed")[1]) for f in sd.glob("tt_countcap_seed*.json"))
        g = lambda base, c: [ld(base / sd.name / f"{c}_seed{s}.json") for s in seeds]
        off, vgg = g(V, "error_off"), g(V, "tt_vgg")
        tcc, cc = g(C, "tt_countcap"), g(C, "countcap")
        d = lambda x: float(np.mean([a["mean_psnr"] - o["mean_psnr"] for a, o in zip(x, off)]))
        r = {"tt_vgg": d(vgg), "tt_countcap": d(tcc), "countcap": d(cc),
             "ms_vgg": float(np.mean([a["mean_frame_ms"] for a in vgg])), "ms_tcc": float(np.mean([a["mean_frame_ms"] for a in tcc])),
             "n": len(seeds)}
        out[sd.name] = r
        L.append(f"| {sd.name} | {r['tt_vgg']:+.2f} | {r['tt_countcap']:+.2f} | {r['countcap']:+.2f} | | {r['ms_vgg']:.0f} / {r['ms_tcc']:.0f} |")
    v = list(out.values())
    out["_summary"] = {"n_scenes": len(v), "mean_tt_vgg": float(np.mean([x["tt_vgg"] for x in v])),
                       "mean_tt_countcap": float(np.mean([x["tt_countcap"] for x in v])),
                       "mean_countcap": float(np.mean([x["countcap"] for x in v])),
                       "vgg_better_than_tcc": int(sum(x["tt_vgg"] > x["tt_countcap"] for x in v)),
                       "median_vgg_minus_tcc": float(np.median([x["tt_vgg"] - x["tt_countcap"] for x in v]))}
    L.append("\n" + json.dumps(out["_summary"], indent=1))
    (C / "analysis.json").write_text(json.dumps(out, indent=1)); (C / "analysis.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
