#!/usr/bin/env python3
"""Held-out-view check of the steps regime (every 8th frame rendered, never mapped).

Conditions: error_off (original budgeted system), fixK10 (uniform 10 steps per frame, steps-regime
optimizer), tts_vgg_T75 (time-targeted steps with value-gated growth, T = 75 ms).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_vgg as vgg  # noqa: E402  (installs the VGG and time-target wrappers)
from experiments.run_frozen_benchmark import load_benchmark_sequence  # noqa: E402
from experiments.run_holdout_benchmark import run_holdout_trajectory  # noqa: E402

CONDS = {
    "error_off": ("error_off", {}),
    "fixK10": ("tts", {"T_ms": 1.0, "n_min": 10, "n_max": 10, "n_init": 10}),
    "fixK3": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}),
    "tts_vgg_T75": ("tts_vgg", {"T_ms": 75.0}),
    "fixK6": ("tts", {"T_ms": 1.0, "n_min": 6, "n_max": 6, "n_init": 6}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0", "replica_office1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--conditions", nargs="+", choices=list(CONDS), default=list(CONDS))
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--holdout_every", type=int, default=8)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. losses.weight_depth=1.0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "holdout_engine")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    for scene in a.scenes:
        frames, intr = load_benchmark_sequence(scene_name=scene, n_frames=a.n_frames + 1, H=240, W=320, device="cuda")
        for seed in a.seeds:
            for name in a.conditions:
                target = a.out_dir / scene / f"{name}{a.tag}_seed{seed}.json"
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                cond, tt = CONDS[name]
                policy, cfg = vgg.config_for(cond, seed, 15.0, "cuda", {}, tt)
                for kv in a.set:
                    path, val = kv.split("=")
                    node, keys = cfg, path.split(".")
                    for k in keys[:-1]:
                        node = node.setdefault(k, {})
                    node[keys[-1]] = float(val)
                print(f"RUN {scene} {name} seed={seed}", flush=True)
                s = run_holdout_trajectory(policy, seed, frames, intr, 15.0, "cuda", 320, 240, a.holdout_every, cfg)
                s.update({"scene": scene, "condition": name + a.tag, "n_frames_loaded": len(frames), "holdout_every": a.holdout_every})
                tmp = target.with_suffix(".tmp"); tmp.write_text(json.dumps(s, indent=1)); tmp.replace(target)
                print(f"DONE {scene} {name} seed={seed}: train={s['train_psnr']:.2f} holdout={s['holdout_psnr']:.2f} "
                      f"gap={s['gen_gap_db']:+.2f} N={s['N_final']}", flush=True)


if __name__ == "__main__":
    main()
