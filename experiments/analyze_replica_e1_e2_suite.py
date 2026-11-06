#!/usr/bin/env python3
"""Analyze all-scene Replica E1/E2 matched-control artifacts for one seed."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES = ("office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2")


def all_finite(pair):
    keys = ("psnr", "ssim", "depth_l1", "psnr_pre", "psnr_post", "depth_l1_pre", "depth_l1_post")
    return all(math.isfinite(float(frame[k])) for arm in pair["arms"] for frame in arm["frames"] for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results/replica_E1E2_seed44_capacity_fixed")
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()
    base = args.input if args.input.is_absolute() else ROOT / args.input
    rows = []
    for scene in SCENES:
        folder = base / f"replica_{scene}"
        e1 = json.loads((folder / f"E1_seed{args.seed}.json").read_text())
        e2 = json.loads((folder / f"E2_seed{args.seed}.json").read_text())
        no_thr, cov = e1["arms"]
        early, late = e2["arms"]
        m0, m1 = no_thr["summary"]["matched_final_map"], cov["summary"]["matched_final_map"]
        s0, s1 = early["summary"], late["summary"]
        row = {
            "scene": scene,
            "n_online_frames": int(e2["validation"]["realized_online_frames"]),
            "E1_valid": bool(e1["valid"]), "E2_valid": bool(e2["valid"]),
            "all_frame_metrics_finite": all_finite(e1) and all_finite(e2),
            "E1_target_gaussians": int(e1["target_final_gaussians"]),
            "E1_matched_map": {"no_throttle": m0, "coverage_throttle": m1},
            "E1_delta_coverage_minus_no_throttle": {
                "psnr": float(m1["psnr"] - m0["psnr"]),
                "ssim": float(m1["ssim"] - m0["ssim"]),
                "depth_l1": float(m1["depth_l1"] - m0["depth_l1"]),
            },
            "E1_natural_trajectory_delta": {
                "mean_psnr": float(cov["summary"]["mean_psnr"] - no_thr["summary"]["mean_psnr"]),
                "mean_depth_l1": float(cov["summary"]["mean_depth_l1"] - no_thr["summary"]["mean_depth_l1"]),
            },
            "E2_target_births": int(e2["target_total_spawned"]),
            "E2_shared_map_capacity": int(e2["shared_gaussian_capacity"]),
            "E2_windows": e2["validation"],
            "E2_off_error_mask_candidate_fraction": [
                float(x / max(e2["target_total_spawned"], 1))
                for x in e2["validation"]["fallback_spawned"]
            ],
            "E2_late_minus_early": {
                "mean_psnr": float(s1["mean_psnr"] - s0["mean_psnr"]),
                "final_psnr": float(s1["final_psnr"] - s0["final_psnr"]),
                "mean_depth_l1": float(s1["mean_depth_l1"] - s0["mean_depth_l1"]),
                "final_depth_l1": float(s1["final_depth_l1"] - s0["final_depth_l1"]),
            },
        }
        rows.append(row)

    paired_summary = {
        "E1_matched_final_map_psnr_wins_coverage": sum(r["E1_delta_coverage_minus_no_throttle"]["psnr"] > 0 for r in rows),
        "E1_matched_final_map_depth_wins_coverage": sum(r["E1_delta_coverage_minus_no_throttle"]["depth_l1"] < 0 for r in rows),
        "E1_natural_mean_psnr_wins_coverage": sum(r["E1_natural_trajectory_delta"]["mean_psnr"] > 0 for r in rows),
        "E2_late_mean_psnr_wins": sum(r["E2_late_minus_early"]["mean_psnr"] > 0 for r in rows),
        "E2_late_mean_depth_wins": sum(r["E2_late_minus_early"]["mean_depth_l1"] < 0 for r in rows),
        "E2_late_final_depth_wins": sum(r["E2_late_minus_early"]["final_depth_l1"] < 0 for r in rows),
    }
    result = {
        "protocol_version": "matched-dilution-controls-replica-all-scenes-v2",
        "seed": args.seed, "scenes": rows,
        "all_count_and_finiteness_gates_pass": all(
            r["E1_valid"] and r["E2_valid"] and r["all_frame_metrics_finite"] for r in rows),
        "scene_win_counts": paired_summary,
        "interpretation": "Single seed across eight short Replica prefixes; E1 only matches final snapshot count, and E2 uses uniform valid-depth proposals with a shared capacity sized for the target births. Exploratory, not confirmatory.",
    }
    out_json = base / "analysis.json"
    out_md = base / "analysis.md"
    lines = [
        f"# Replica E1/E2 all-scene matched controls — seed {args.seed}", "",
        "All eight available scenes; 99 online frames for office0/room0 and 119 for the other scenes. E1 matches final snapshot size by common lowest-opacity pruning. E2 matches total births in early/late halves, uses uniform valid-depth proposals, and raises both arms to the same capacity sufficient for the pilot target.", "",
        f"Count and finite-metric gates: **{'PASS' if result['all_count_and_finiteness_gates_pass'] else 'FAIL'}**.", "",
        "| Scene | T | E1 N | E1 Δ final PSNR | E1 Δ final depth L1 | E1 Δ natural mean PSNR | E2 births | E2 Δ mean PSNR | E2 Δ final PSNR | E2 Δ mean depth L1 | E2 Δ final depth L1 | Off-mask candidates early/late |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|",
    ]
    for r in rows:
        e1d, nat, e2d = r["E1_delta_coverage_minus_no_throttle"], r["E1_natural_trajectory_delta"], r["E2_late_minus_early"]
        off = r["E2_off_error_mask_candidate_fraction"]
        lines.append(
            f"| {r['scene']} | {r['n_online_frames']} | {r['E1_target_gaussians']:,} | {e1d['psnr']:+.3f} dB | {e1d['depth_l1']:+.5f} m | {nat['mean_psnr']:+.3f} dB | {r['E2_target_births']:,} | {e2d['mean_psnr']:+.3f} dB | {e2d['final_psnr']:+.3f} dB | {e2d['mean_depth_l1']:+.5f} m | {e2d['final_depth_l1']:+.5f} m | {100*off[0]:.1f}% / {100*off[1]:.1f}% |"
        )
    lines += ["", "## Counts across scenes", "",
              f"E1 coverage-throttle wins at matched final-map PSNR: {paired_summary['E1_matched_final_map_psnr_wins_coverage']}/8; at matched depth L1: {paired_summary['E1_matched_final_map_depth_wins_coverage']}/8. On natural trajectories it wins mean PSNR in {paired_summary['E1_natural_mean_psnr_wins_coverage']}/8.",
              f"E2 late-spawn wins mean PSNR in {paired_summary['E2_late_mean_psnr_wins']}/8, mean depth L1 in {paired_summary['E2_late_mean_depth_wins']}/8, and final depth L1 in {paired_summary['E2_late_final_depth_wins']}/8.", "",
              "One seed, no uncertainty intervals. E1 compares final renders after pruning and does not match the map history or Gaussian identities. E2 matches total births but not final map size or frame-specific candidate pixels; its outside-error-mask fractions are expected for uniform-depth proposals. Do not interpret this as a general controller rule.", ""]
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    out_md.write_text("\n".join(lines))
    print(out_md)


if __name__ == "__main__":
    main()
