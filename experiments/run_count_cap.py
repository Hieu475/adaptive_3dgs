#!/usr/bin/env python3
"""Does value-gated growth beat a plain map-size budget of the same size?

Taming-3DGS / Mini-Splatting style methods control reconstruction through a fixed Gaussian-count
budget. This is that control inside our pipeline (hard cap ``gaussian.max_gaussians``), matched per scene
and seed to the final map size of TT+VGG (read from the frozen bundles), under the same time-targeted
optimization budget. Conditions:
  tt_countcap   TT budget + hard cap at TT+VGG's final map size
  countcap      baseline budget + the same hard cap
The comparison is against tt_vgg and error_off from the same bundle. These are budget-matched controls
inside our pipeline, not the published methods.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_vgg as rv  # noqa: E402
from experiments.run_dilution_factorial import summarize  # noqa: E402
from experiments.run_frozen_benchmark import load_benchmark_sequence, run_policy_trajectory  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--ref", type=Path, default=ROOT / "results" / "vgg_test", help="bundle holding tt_vgg runs")
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "count_cap")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    for sc in a.scenes:
        frames, intr = load_benchmark_sequence(scene_name=sc, n_frames=a.offset + a.n_frames + 1, H=240, W=320, device="cuda")
        frames = frames[a.offset:]
        for seed in a.seeds:
            ref = a.ref / sc / f"tt_vgg_seed{seed}.json"
            if not ref.exists():
                continue
            ncap = int(json.loads(ref.read_text())["summary"]["N_final"])
            for cond in ("tt_countcap", "countcap"):
                target = a.out_dir / sc / f"{cond}_seed{seed}.json"
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                policy, cfg = rv.config_for("tt" if cond == "tt_countcap" else "error_off", seed, 15.0, "cuda", {}, {"T_ms": 100.0})
                cfg["gaussian"]["max_gaussians"] = ncap
                rv.abl.ext.BUDGET_SCALES.clear(); rv.STATES.clear(); rv.TT_STATES.clear()
                print(f"RUN {sc} {cond} seed={seed} cap={ncap}", flush=True)
                summ, logs = run_policy_trajectory(policy=policy, seed=seed, frames=frames, intrinsics=intr,
                                                   budget_ms=15.0, device="cuda", W=320, H=240, custom_config=cfg)
                res = summarize(summ, logs)
                res["mean_frame_ms"] = float(sum(r["frame_time_ms"] for r in logs) / len(logs))
                rec = {"scene": sc, "seed": seed, "condition": cond, "summary": res, "cap": ncap, "offset": a.offset}
                tmp = target.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); tmp.replace(target)
                print(f"DONE {sc} {cond} seed={seed}: PSNR={summ['mean_psnr']:.3f} map={summ['N_final']} "
                      f"frame={res['mean_frame_ms']:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
