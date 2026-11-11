#!/usr/bin/env python3
"""Analysis of the steps-regime evaluation (results/controllers/engine/{test,fresh}).

Per scene (or fresh window) and condition: seed-mean PSNR and measured frame time. Contrasts:
  gain vs error_off (the original budgeted system on the same core),
  equal-time gain vs the fixed-K curve (uniform K steps per frame, K in 3/6/10/15), interpolated
  linearly in measured time at the method's own mean frame time.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R = ROOT / "results" / "controllers" / "engine"
FIXED = ("tts_fixK3", "tts_fixK6", "tts_fixK10", "tts_fixK15")
METHODS = ("tt_vgg", "error_off_opt", "tt_vgg_opt", "tts_T60", "tts_vgg_T45", "tts_vgg_T60", "tts_vgg_T75", "tts_vgg_T90")


def load_unit(d):
    out = {}
    for f in Path(d).glob("*_seed*.json"):
        j = json.loads(f.read_text())
        out.setdefault(j["condition"], []).append((j["summary"]["mean_psnr"], j["summary"]["mean_frame_ms"], j["summary"].get("mean_depth_l1", np.nan), j["summary"]["N_final"]))
    return {c: tuple(float(np.mean([v[i] for v in vs])) for i in range(4)) for c, vs in out.items()}


def curve(u):
    pts = sorted((u[c][1], u[c][0]) for c in FIXED if c in u)
    return np.array(pts)


def interp(u, ms):
    pts = curve(u)
    if len(pts) < 2:
        return np.nan
    return float(np.interp(ms, pts[:, 0], pts[:, 1]))  # clamps outside the range


def main():
    units = {}
    for split in ("test", "fresh"):
        base = R / split
        if not base.exists():
            continue
        for sd in sorted(p for p in base.iterdir() if p.is_dir()):
            d = sd / sd.name if (sd / sd.name).is_dir() else sd
            u = load_unit(d)
            if "error_off" in u and all(c in u for c in FIXED):
                units[f"{split}/{sd.name}"] = u
    out = {"units": {}, "summary": {}}
    L = ["| condition | split | n | mean ms | dPSNR vs orig. (mean / median / worst) | equal-time vs fixed-K (mean / median / worst) |", "|---|---|--:|--:|---|---|"]
    for split in ("test", "fresh"):
        us = {k: v for k, v in units.items() if k.startswith(split)}
        if not us:
            continue
        for m in ("error_off",) + FIXED + METHODS:
            rows = [(u[m][0] - u["error_off"][0], u[m][1], u[m][0] - interp(u, u[m][1])) for u in us.values() if m in u]
            if not rows:
                continue
            g = np.array([r[0] for r in rows]); t = np.mean([r[1] for r in rows]); e = np.array([r[2] for r in rows])
            out["summary"][f"{split}/{m}"] = {"n": len(rows), "ms": float(t), "gain_mean": float(g.mean()), "gain_median": float(np.median(g)),
                                              "gain_min": float(g.min()), "eq_mean": float(e.mean()), "eq_median": float(np.median(e)), "eq_min": float(e.min()),
                                              "eq_pos": int((e > 0).sum())}
            L.append(f"| {m} | {split} | {len(rows)} | {t:.0f} | {g.mean():+.2f} / {np.median(g):+.2f} / {g.min():+.2f} | {e.mean():+.2f} / {np.median(e):+.2f} / {e.min():+.2f} |")
    for k, u in units.items():
        out["units"][k] = {c: {"psnr": v[0], "ms": v[1], "N": v[3]} for c, v in u.items()}
    (R / "analysis.json").write_text(json.dumps(out, indent=1))
    (R / "analysis.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n{len(units)} complete units")


if __name__ == "__main__":
    main()
