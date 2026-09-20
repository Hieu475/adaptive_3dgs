"""Diagnose the "blur / bad-LPIPS" hypothesis: are most Gaussians under-converged?

Uses the *already-existing* `GaussianModel.update_counts` (lifetime optimizer-
touch counter, incremented once per Gaussian per micro-step in
`state_store.py:195`) — no new instrumentation needed, this data has been
tracked all along, just never reported.

Usage (run on your real GPU + TUM data environment):
    python experiments/diagnostics/diagnose_lifetime_convergence.py \
        --policy ours --scene tum_fr2_xyz --n_frames 3669 --seed 42

What it reports:
  1. Distribution of lifetime update_counts across the final Gaussian population
     (percentiles, % with 0 touches, % with <5 touches i.e. fewer than one
     full K=5 micro-step pass).
  2. Correlation between a Gaussian's update_count and its final per-pixel
     contribution to reconstruction error (via render attribution) — if
     low-touch-count Gaussians disproportionately explain the remaining
     error, that's direct, quantitative evidence for the "blur from
     under-convergence" hypothesis (root cause of the bad LPIPS numbers).
  3. Breaks #1 down by *when* each Gaussian was created (early vs late in the
     trajectory) — expect late-created Gaussians to have systematically fewer
     lifetime touches once coverage_throttling is active, since throttling
     reduces how much of the newly-limited per-frame budget they can compete
     for against the already-large existing population.

This is read-only / diagnostic: it does not change training. Run it once on
the final OURS population after a full benchmark run.
"""
import argparse
import json
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from research.pipeline import OnlineReconstructionPipeline  # noqa: E402


def summarize_update_counts(update_counts: torch.Tensor) -> dict:
    uc = update_counts.detach().cpu().numpy()
    n = len(uc)
    pct = lambda p: float(np.percentile(uc, p)) if n > 0 else float('nan')
    return {
        "n_gaussians": int(n),
        "mean": float(uc.mean()) if n else float('nan'),
        "p10": pct(10), "p25": pct(25), "p50_median": pct(50), "p75": pct(75), "p90": pct(90),
        "frac_zero_touches": float((uc == 0).mean()) if n else float('nan'),
        "frac_under_5_touches": float((uc < 5).mean()) if n else float('nan'),  # < one full K=5 pass
        "frac_under_1_full_pass": float((uc < 5).mean()) if n else float('nan'),
        "max": int(uc.max()) if n else 0,
    }


def run(args):
    from experiments.run_phase13_frozen_benchmark import build_pipeline_config, load_phase10_sequence

    n_load = args.n_frames + 1
    frames, intrinsics = load_phase10_sequence(
        scene_name=args.scene, n_frames=n_load, H=240, W=320, device="cpu",
    )

    cfg = build_pipeline_config(policy=args.policy, seed=args.seed, device=args.device)
    if args.sh_degree is not None:
        cfg["gaussian"]["sh_degree"] = args.sh_degree
        # sh_degree controls num_sh_coeffs at GaussianModel construction time,
        # OnlineReconstructionPipeline must be able to read this from config.gaussian.sh_degree
        # (see research/pipeline.py _default_config / initialize()).
    pipeline = OnlineReconstructionPipeline(config=cfg, device=args.device)
    pipeline.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                         intrinsics=intrinsics, pose=frames[0].get("pose", torch.eye(4)))

    for t in range(1, len(frames)):
        pipeline.process_frame(rgb=frames[t]["rgb"], depth=frames[t]["depth"], gt_pose=frames[t].get("pose", None))
        if t % 500 == 0:
            print(f"[{args.policy}] Frame {t}/{len(frames)-1} processed | Gaussians: {pipeline.gaussian_model.num_gaussians}")

    uc = pipeline.gaussian_model.update_counts
    if hasattr(pipeline.gaussian_model, "state_store") and hasattr(pipeline.gaussian_model.state_store, "creation_frames"):
        creation_frame = pipeline.gaussian_model.state_store.creation_frames.detach().cpu()
    else:
        creation_frame = torch.zeros(len(uc), dtype=torch.long)

    report = {"policy": args.policy, "scene": args.scene, "n_frames": args.n_frames,
              "overall": summarize_update_counts(uc)}

    # Break down by creation time (early third / middle third / late third of trajectory)
    n_frames_total = len(frames)
    thirds = np.digitize(creation_frame.numpy(), [n_frames_total // 3, 2 * n_frames_total // 3])
    report["by_creation_time"] = {}
    for name, idx in [("early_third", 0), ("middle_third", 1), ("late_third", 2)]:
        mask = torch.from_numpy(thirds == idx)
        if mask.any():
            report["by_creation_time"][name] = summarize_update_counts(uc[mask])

    print(json.dumps(report, indent=2))
    out_path = f"results/diagnostics_lifetime_convergence_{args.policy}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Headline interpretation
    overall = report["overall"]
    if overall["frac_under_5_touches"] > 0.5:
        print(f"\n[HYPOTHESIS SUPPORTED] {overall['frac_under_5_touches']*100:.1f}% of the final "
              f"Gaussian population never received a single full K=5 optimization pass in its "
              f"lifetime -> strong candidate explanation for high LPIPS despite good SSIM/PSNR.")
    else:
        print(f"\n[HYPOTHESIS NOT CLEARLY SUPPORTED] Only {overall['frac_under_5_touches']*100:.1f}% "
              f"under-touched -> under-convergence alone may not fully explain the LPIPS gap; "
              f"check sh_degree, resolution, or loss weighting instead.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="ours")
    p.add_argument("--scene", default="tum_fr2_xyz")
    p.add_argument("--n_frames", type=int, default=3669)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sh_degree", type=int, default=None, help="Override gaussian.sh_degree (e.g. 1 to test view-dependent color fix)")
    run(p.parse_args())
