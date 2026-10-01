#!/usr/bin/env python3
"""Mixed-regime per-frame routing test (causal switch evidence).

Takes Replica office0 (clean: routes error_only), injects 30% random depth
holes into the second half of the trajectory, and runs the per-frame adaptive
router. Expected: routing flips at the injection boundary, proving the
controller responds to within-sequence observable shifts (no scene labels).

Compares end-to-end PSNR vs fixed ours / fixed error_only on the same
corrupted trajectory. Saves per-frame route trace + means.

Usage:
  python experiments/run_mixed_regime.py --seed 42 --n_frames 99
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_phase13_frozen_benchmark import load_phase10_sequence, build_pipeline_config
from research.pipeline import OnlineReconstructionPipeline
from research.noise_adaptive import frame_noise_score_v3
from research.reproducibility import set_seed


def run_one(policy, seed, frames, intr, budget_ms=15.0, device="cuda",
            W=320, H=240, adaptive=False):
    set_seed(seed)
    cfg = build_pipeline_config(policy="ours", seed=seed, budget_ms=budget_ms,
                                W=W, H=H, device=device)
    if adaptive:
        cfg["adaptive_routing"] = {"enabled": True}
    elif policy == "error_only":
        cfg = build_pipeline_config(policy="error_only", seed=seed, budget_ms=budget_ms,
                                    W=W, H=H, device=device)
    pipe = OnlineReconstructionPipeline(config=cfg, device=device)
    pipe.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                    intrinsics=intr, pose=frames[0].get("pose", torch.eye(4)))
    psnrs, routes = [], []
    for t in range(1, len(frames)):
        f = frames[t]
        kw = {}
        if adaptive:
            nz = frame_noise_score_v3(f.get("raw_holes", 0.0),
                                      f.get("raw_edge", 0.01))["noise"]
            kw = {"frame_noise": nz}
        m = pipe.process_frame(rgb=f["rgb"], depth=f["depth"],
                               gt_pose=f.get("pose"), **kw)
        psnrs.append(m["psnr"])
        if adaptive:
            routes.append(m.get("routed_policy", "?"))
    return float(np.mean(psnrs)), routes


def main():
    ap = argparse.ArgumentParser(description="Mixed-regime routing test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_frames", type=int, default=99)
    ap.add_argument("--hole_frac", type=float, default=0.30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="results/phase14_adaptive_perframe")
    a = ap.parse_args()
    if a.device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.70, 0)
        except Exception:
            pass
    out = REPO_ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240
    frames, intr = load_phase10_sequence(scene_name="replica_office0",
                                         n_frames=a.n_frames + 1, H=H, W=W,
                                         device=a.device)
    # Inject holes into second half only (regime shift mid-trajectory).
    rng = np.random.default_rng(a.seed)
    half = len(frames) // 2
    for f in frames[half:]:
        d = f["depth"]
        mask = torch.from_numpy(rng.random(d.shape) < a.hole_frac).to(d.device)
        d2 = d.clone()
        d2[mask] = 0.0
        f["depth"] = d2
        f["raw_holes"] = float(a.hole_frac)
    from collections import Counter
    res = {}
    adapt_routes = None
    for pol, adapt in [("adaptive", True), ("ours", False), ("error_only", False)]:
        mean, routes = run_one(pol, a.seed, frames, intr, device=a.device, W=W, H=H,
                               adaptive=adapt)
        res[pol] = mean
        print(f"{pol}: mean={mean:.2f}", flush=True)
        if adapt:
            adapt_routes = routes
    # Route analysis from the same adaptive run (no recompute).
    first = adapt_routes[:half - 1]
    second = adapt_routes[half - 1:]
    res["route_first_half"] = dict(Counter(first))
    res["route_second_half"] = dict(Counter(second))
    res["n_frames"] = len(adapt_routes)
    print("first half routes:", res["route_first_half"])
    print("second half routes:", res["route_second_half"])
    json.dump(res, open(out / "mixed_regime.json", "w"), indent=1)
    print("wrote", out / "mixed_regime.json")


if __name__ == "__main__":
    main()
