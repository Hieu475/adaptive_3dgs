#!/usr/bin/env python3
"""Long-term retention of the map: final-map PSNR on frames that were never mapped or mapped long ago.

Frames with index % 5 == 0 are never mapped (MonoGS' evaluation grid). After the last frame the final map
is rendered from those frames' ground-truth poses ("held-out, final map"); the PSNR of the same held-out
frames right when they arrive ("held-out, causal") and of mapped frames of the first and last thirds of the
sequence ("mapped, early / late", final map) are also reported. A map that only fits the recent views shows
a large gap between causal and final-map PSNR.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_vgg as vgg  # noqa: E402  (installs the time-target / VGG wrappers)
from experiments.run_frozen_benchmark import load_benchmark_sequence  # noqa: E402
from src.core.pipeline import OnlineReconstructionPipeline  # noqa: E402
from src.evaluation.reproducibility import set_seed  # noqa: E402

CONDS = {
    "orig": ("error_off", {}, {}),
    "k3": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {}),
    "k10": ("tts", {"T_ms": 1.0, "n_min": 10, "n_max": 10, "n_init": 10}, {}),
    # replay constants chosen on the four development scenes: every 3rd step, 32-frame reservoir, every 2nd mapped frame
    "k3_rp": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"replay": {"every": 3, "size": 32, "stride": 2}}),
    # joint replay (chosen on the four development scenes): steps 2..K optimize the current frame jointly with 2 replayed frames
    "k3_jr": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"replay": {"every": 1, "size": 32, "stride": 2, "joint": 2}}),
    # joint replay drawing the replayed frames from the m stored frames closest in pose (keyframe window)
    "k3_jn4": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"replay": {"every": 1, "size": 32, "stride": 2, "joint": 2, "near": 4}}),
    "k3_jn8": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"replay": {"every": 1, "size": 32, "stride": 2, "joint": 2, "near": 8}}),
    "k3_c1": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"consolidation": {"beta": 1.0, "decay": 1.0}}),
    "k3_c4": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"consolidation": {"beta": 4.0, "decay": 1.0}}),
    "k3_c16": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"consolidation": {"beta": 16.0, "decay": 1.0}}),
    "k3_jr_c4": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"replay": {"every": 1, "size": 32, "stride": 2, "joint": 2}, "consolidation": {"beta": 4.0, "decay": 1.0}}),
    "k3_jr_s2": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"replay": {"every": 1, "size": 32, "stride": 2, "joint": 2, "scale": 0.5}}),
    "k3_jr_s4": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"replay": {"every": 1, "size": 32, "stride": 2, "joint": 2, "scale": 0.25}}),
    "k3_jr_s4x4": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"replay": {"every": 1, "size": 128, "stride": 2, "joint": 2, "scale": 0.25}}),
    "k3_jr_s8x16": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}, {"replay": {"every": 1, "size": 256, "stride": 2, "joint": 2, "scale": 0.125}}),
    "k10_rp": ("tts", {"T_ms": 1.0, "n_min": 10, "n_max": 10, "n_init": 10}, {"replay": {"every": 3, "size": 32, "stride": 2}}),
}


def parse_cond(name):
    """kN (N uniform steps), kN_rp (plus keyframe replay), kN_jt (plus joint two-view steps; set replay.every)."""
    import re
    m = re.fullmatch(r"k(\d+)(_rp|_jt)?", name)
    n = int(m.group(1))
    tt = {"T_ms": 1.0, "n_min": n, "n_max": n, "n_init": n}
    tr = {}
    if m.group(2) == "_rp":
        tr = {"replay": {"every": 3, "size": 32, "stride": 2}}
    elif m.group(2) == "_jt":
        tr = {"replay": {"every": 3, "size": 32, "stride": 2, "joint": 1}}
    return "tts", tt, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--offset", type=int, default=0, help="start the sequence at this frame (fresh window)")
    ap.add_argument("--set", nargs="*", default=[], help="training overrides, e.g. replay.every=3 replay.size=8")
    ap.add_argument("--cfg", nargs="*", default=[], help="overrides relative to the whole config, e.g. densification.disable_pruning=1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "retention")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    for sc in a.scenes:
        frames, intr = load_benchmark_sequence(scene_name=sc, n_frames=a.offset + a.n_frames + 1, H=240, W=320, device="cuda")
        frames = frames[a.offset:]
        n = len(frames)
        for seed in a.seeds:
            for name in a.conds:
                target = a.out_dir / sc / f"{name}{a.tag}_seed{seed}.json"
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                set_seed(seed)
                cond, tt, tr = CONDS[name] if name in CONDS else parse_cond(name)
                _, cfg = vgg.config_for(cond, seed, 15.0, "cuda", {}, tt)
                cfg["training"].update(copy.deepcopy(tr))
                for kv in a.set:
                    path, val = kv.split("=")
                    node, keys = cfg["training"], path.split(".")
                    for k in keys[:-1]:
                        node = node.setdefault(k, {})
                    try:
                        node[keys[-1]] = float(val) if "." in val else int(val)
                    except ValueError:
                        node[keys[-1]] = val
                for kv in a.cfg:
                    path, val = kv.split("=")
                    node, keys = cfg, path.split(".")
                    for k in keys[:-1]:
                        node = node.setdefault(k, {})
                    node[keys[-1]] = float(val) if "." in val else int(val)
                vgg.STATES.clear(); vgg.TT_STATES.clear()
                pipe = OnlineReconstructionPipeline(config=cfg, device="cuda")
                pipe.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"], intrinsics=intr, pose=frames[0]["pose"])
                held, causal, mapped_idx, t_ms = [], [], [], []
                for t in range(1, n):
                    f = frames[t]
                    if t % 5 == 0:
                        held.append(t)
                        causal.append(float(pipe.eval_frame(rgb=f["rgb"], depth=f["depth"], pose=f["pose"])["psnr"]))
                    else:
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        pipe.process_frame(rgb=f["rgb"], depth=f["depth"], gt_pose=f["pose"])
                        torch.cuda.synchronize(); t_ms.append(1000 * (time.perf_counter() - t0))
                        mapped_idx.append(t)
                ev = lambda t: float(pipe.eval_frame(rgb=frames[t]["rgb"], depth=frames[t]["depth"], pose=frames[t]["pose"])["psnr"])
                final_held = [ev(t) for t in held]
                early = [ev(t) for t in mapped_idx[: len(mapped_idx) // 3]][::3]
                late = [ev(t) for t in mapped_idx[-(len(mapped_idx) // 3):]][::3]
                res = {"scene": sc, "cond": name + a.tag, "seed": seed, "n_frames": n, "n_heldout": len(held),
                       "heldout_causal": float(np.mean(causal)), "heldout_final": float(np.mean(final_held)),
                       "mapped_early_final": float(np.mean(early)), "mapped_late_final": float(np.mean(late)),
                       "N_final": int(pipe.gaussian_model.num_gaussians), "mean_frame_ms": float(np.mean(t_ms[5:]))}
                target.write_text(json.dumps(res, indent=1))
                print("DONE", sc, name + a.tag, seed, {k: round(v, 2) for k, v in res.items() if isinstance(v, float)}, flush=True)
                del pipe; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
