#!/usr/bin/env python3
"""Summarize the preregistered Replica office1 E1/E2 diagnostic run."""
import json
import math
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIR_DIR = ROOT / "results/replica_office1_E1E2_seed42/replica_office1"


def finite_records(arm):
    keys = ("psnr", "ssim", "depth_l1", "psnr_pre", "psnr_post", "depth_l1_pre", "depth_l1_post")
    return all(math.isfinite(float(frame[k])) for frame in arm["frames"] for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_PAIR_DIR,
                    help="Directory containing E1_seed*.json and E2_seed*.json")
    args = ap.parse_args()
    pair_dir = args.input if args.input.is_absolute() else ROOT / args.input
    e1_paths = sorted(pair_dir.glob("E1_seed*.json"))
    if len(e1_paths) != 1:
        raise ValueError(f"Expected exactly one E1 pair in {pair_dir}, found {len(e1_paths)}")
    seed = int(e1_paths[0].stem.split("seed")[-1])
    e2_path = pair_dir / f"E2_seed{seed}.json"
    e1 = json.loads(e1_paths[0].read_text())
    e2 = json.loads(e2_path.read_text())
    no_throttle, coverage = e1["arms"]
    early, late = e2["arms"]
    a, b = no_throttle["summary"]["matched_final_map"], coverage["summary"]["matched_final_map"]
    s0, s1 = early["summary"], late["summary"]
    e1_delta = {"psnr": b["psnr"] - a["psnr"], "ssim": b["ssim"] - a["ssim"],
                "depth_l1": b["depth_l1"] - a["depth_l1"]}
    e2_delta = {k: s1[k] - s0[k] for k in ("mean_psnr", "final_psnr", "mean_ssim", "final_ssim",
                                               "mean_depth_l1", "final_depth_l1")}
    early_frames = [int(f["frame"]) for f in early["frames"] if int(f["n_spawned"]) > 0]
    late_frames = [int(f["frame"]) for f in late["frames"] if int(f["n_spawned"]) > 0]
    off_error = e2["validation"]["fallback_spawned"]
    n_spawn = e2["target_total_spawned"]
    result = {
        "scene": "replica_office1", "seed": seed,
        "E1": {"valid": e1["valid"], "target_final_gaussians": e1["target_final_gaussians"],
               "matched_map_metrics": {"no_throttle": a, "coverage_throttle": b},
               "coverage_minus_no_throttle": e1_delta,
               "natural_trajectory_summary": {
                   name: {k: arm["summary"][k] for k in ("mean_psnr", "mean_depth_l1", "N_final_active")}
                   for name, arm in (("no_throttle", no_throttle), ("coverage_throttle", coverage))}},
        "E2": {"valid": e2["valid"], "target_total_spawned": n_spawn,
               "windows": e2["validation"], "first_last_spawn_frames": {
                   "early": [min(early_frames), max(early_frames)] if early_frames else None,
                   "late": [min(late_frames), max(late_frames)] if late_frames else None},
               "off_error_mask_candidate_fraction": [x / max(n_spawn, 1) for x in off_error],
               "arms": {"early": s0, "late": s1}, "late_minus_early": e2_delta},
        "all_frame_metrics_finite": all(finite_records(arm) for pair in (e1, e2) for arm in pair["arms"]),
        "interpretation": "One seed on a post-hoc selected Replica scene. E1 matches only the final rendered map count; map identity/history differ. E2 fixes total births and time windows under uniform-depth proposals, but fallback births form a large share and mean/final depth metrics disagree. Exploratory only.",
    }

    lines = [
        "# Replica office1 E1/E2 matched-control follow-up", "",
        f"Seed {seed}, 119 online frames, ground-truth Replica poses, 320x240. office1 was selected after the cross-scene run because it had the largest observed depth L1 regression for `ours`; this makes the follow-up exploratory.", "",
        "## E1 — same final Gaussian count", "",
        f"Exact map-count gate: **{'PASS' if e1['valid'] else 'FAIL'}** at {e1['target_final_gaussians']:,} Gaussians in both final renders.", "",
        "| Final-map arm | PSNR | SSIM | Depth L1 (m) |",
        "|:--|--:|--:|--:|",
        f"| No throttle | {a['psnr']:.3f} | {a['ssim']:.4f} | {a['depth_l1']:.5f} |",
        f"| Coverage throttle | {b['psnr']:.3f} | {b['ssim']:.4f} | {b['depth_l1']:.5f} |",
        f"| Throttle − no throttle | {e1_delta['psnr']:+.3f} | {e1_delta['ssim']:+.4f} | {e1_delta['depth_l1']:+.5f} |", "",
        f"At equal final map count, the coverage-throttled final snapshot is +{e1_delta['psnr']:.3f} dB PSNR and {e1_delta['depth_l1']:+.5f} m depth L1 versus the unthrottled snapshot. This indicates that count alone does not determine final-frame quality in this run. Primitive identities, parameter state, and optimization histories still differ. The natural full-trajectory means move in the opposite direction: PSNR {no_throttle['summary']['mean_psnr']:.3f}→{coverage['summary']['mean_psnr']:.3f} dB and depth L1 {no_throttle['summary']['mean_depth_l1']:.5f}→{coverage['summary']['mean_depth_l1']:.5f} m.", "",
        "## E2 — same total births, different timing", "",
        f"Exact spawn-count gate: **{'PASS' if e2['valid'] else 'FAIL'}**; both arms spawned {n_spawn:,} Gaussians. Early births occurred in frames {min(early_frames)}–{max(early_frames)}; late births in frames {min(late_frames)}–{max(late_frames)}.", "",
        "| Metric | Early | Late | Late − early |",
        "|:--|--:|--:|--:|",
        f"| Mean PSNR (dB) | {s0['mean_psnr']:.3f} | {s1['mean_psnr']:.3f} | {e2_delta['mean_psnr']:+.3f} |",
        f"| Final PSNR (dB) | {s0['final_psnr']:.3f} | {s1['final_psnr']:.3f} | {e2_delta['final_psnr']:+.3f} |",
        f"| Mean depth L1 (m) | {s0['mean_depth_l1']:.5f} | {s1['mean_depth_l1']:.5f} | {e2_delta['mean_depth_l1']:+.5f} |",
        f"| Final depth L1 (m) | {s0['final_depth_l1']:.5f} | {s1['final_depth_l1']:.5f} | {e2_delta['final_depth_l1']:+.5f} |", "",
        f"The late arm improves mean and final PSNR, but mean depth L1 worsens while final-frame depth L1 improves. Of the uniform-depth candidates, {100*off_error[0]/n_spawn:.1f}% early and {100*off_error[1]/n_spawn:.1f}% late lie outside the current error mask; this is expected for the configured proposal distribution, not quota fallback. E2 therefore tests timing under uniform valid-depth proposals, not the normal error-mask-driven spawn rule. Candidate pixels still belong to different observed frames and poses in the two time windows.", "",
        "## Limits", "",
        "Single seed and one post-hoc selected scene; no uncertainty estimate. E1 matches the final snapshot count, not map population at each frame. E2 does not match final map size, and its mean/final depth results disagree. Do not generalize these contrasts to all Replica scenes or treat them as evidence that a controller should use late spawning.", "",
    ]
    out_json = pair_dir.parent / "analysis.json"
    out_md = pair_dir.parent / "analysis.md"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    out_md.write_text("\n".join(lines))
    print(out_md)


if __name__ == "__main__":
    main()
