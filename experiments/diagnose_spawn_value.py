#!/usr/bin/env python3
"""Does the counterfactual spawn value separate TUM-like from Replica-like scenes?

Runs the unthrottled raw-error baseline and, every ``--every`` frames, measures the value of the
Gaussians spawned in the last ``--window`` frames on the incoming frame (research.growth_value).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_throttle_ablation as abl  # noqa: E402
from experiments.run_phase13_frozen_benchmark import load_phase10_sequence, run_policy_trajectory  # noqa: E402
from research.growth_value import spawn_value  # noqa: E402
from research.pipeline import OnlineReconstructionPipeline  # noqa: E402

LOG = []
_orig = OnlineReconstructionPipeline.process_frame


def _patched(self, rgb, depth, gt_pose=None, *a, **k):
    cfg = getattr(self, "_diag", None)
    if cfg and gt_pose is not None and self.frame_count >= cfg["start"] and self.frame_count % cfg["every"] == 0:
        v = spawn_value(self, rgb, depth, gt_pose, cfg["window"])
        if v:
            v["frame"] = int(self.frame_count)
            LOG.append(v)
    return _orig(self, rgb, depth, gt_pose, *a, **k)


OnlineReconstructionPipeline.process_frame = _patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0", "replica_office1"])
    ap.add_argument("--condition", default="error_off")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "spawn_value_diag.json")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    out = {}
    orig_init = OnlineReconstructionPipeline.__init__

    def init(self, *args, **kw):
        orig_init(self, *args, **kw)
        self._diag = {"start": 10, "every": a.every, "window": a.window}
    OnlineReconstructionPipeline.__init__ = init
    for sc in a.scenes:
        frames, intr = load_phase10_sequence(scene_name=sc, n_frames=a.n_frames + 1, H=240, W=320, device="cuda")
        policy, cfg = abl.config_for(a.condition, a.seed, 15.0, "cuda")
        LOG.clear()
        run_policy_trajectory(policy=policy, seed=a.seed, frames=frames, intrinsics=intr, budget_ms=15.0,
                              device="cuda", W=320, H=240, custom_config=cfg)
        out[sc] = list(LOG)
        v = [x["value_db"] for x in LOG]
        per = [x["value_db"] / max(x["n_recent"], 1) * 1000 for x in LOG]
        print(f"{sc}: n={len(v)} mean value {sum(v) / len(v):+.3f} dB, per 1000 spawned {sum(per) / len(per):+.4f} dB; "
              f"late(>=80) mean {sum(x['value_db'] for x in LOG if x['frame'] >= 80) / max(1, sum(1 for x in LOG if x['frame'] >= 80)):+.3f}", flush=True)
    a.out.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
