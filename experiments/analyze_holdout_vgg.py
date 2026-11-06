#!/usr/bin/env python3
"""Train vs held-out PSNR changes for TT and TT+VGG (results/holdout_vgg)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results" / "holdout_vgg"


def main():
    out = {}
    for sd in sorted(p for p in D.iterdir() if p.is_dir()):
        runs = {}
        for f in sd.glob("*_seed*.json"):
            r = json.loads(f.read_text())
            runs[(r["condition"], r["seed"])] = r
        seeds = sorted({s for c, s in runs if c == "error_off" and all((k, s) in runs for k in ("tt", "tt_vgg"))})
        out[sd.name] = {"n": len(seeds)}
        for c in ("tt", "tt_vgg"):
            tr = float(np.mean([runs[(c, s)]["train_psnr"] - runs[("error_off", s)]["train_psnr"] for s in seeds]))
            ho = float(np.mean([runs[(c, s)]["holdout_psnr"] - runs[("error_off", s)]["holdout_psnr"] for s in seeds]))
            out[sd.name][c] = {"train_delta": tr, "holdout_delta": ho, "retained": ho / tr if tr > 0.05 else None}
    (D / "analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
