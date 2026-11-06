#!/usr/bin/env python3
"""Summarize endogenous matched-map replications and frozen-schedule transfer gates."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNS = {
    44: ("calibration", "replica_e1_dynamic_matched_map_office0_seed44_calibration"),
    45: ("calibration", "replica_e1_dynamic_matched_map_office0_seed45_calibration"),
    57: ("held-out", "replica_e1_dynamic_matched_map_office0_seed57_heldout"),
    58: ("held-out", "replica_e1_dynamic_matched_map_office0_seed58_heldout"),
}
SCHEDULE_RUNS = (
    ("natural floor, margin 2k", "replica_e1_native_floor_v1_office0_seed57"),
    ("natural floor, margin 4k", "replica_e1_native_floor_v2_office0_seed57"),
    ("natural floor + growth guard 50", "replica_e1_native_floor_v3_office0_seed57"),
    ("natural floor + growth guard 300", "replica_e1_native_floor_v4_office0_seed57"),
    ("endogenous calibration schedule", "replica_e1_office0_dynamic_calibrated_v1_seed57"),
)


def load_run(folder):
    path = RESULTS / folder / "dynamic_matched_map_results.json"
    d = json.loads(path.read_text())
    s = d["summary"]
    return {
        "seed": int(d["seed"]), "folder": folder,
        "frames": int(s["processed_frames"]), "requested": int(s["requested_online_frames"]),
        "gate": bool(s["matched_count_gate_passed_every_frame"]),
        "mean_n": s.get("mean_matched_gaussians"),
        "psnr": s.get("mean_delta_coverage_minus_no_throttle_psnr"),
        "depth": s.get("mean_delta_coverage_minus_no_throttle_depth_l1"),
        "ssim": s.get("mean_delta_coverage_minus_no_throttle_ssim"),
        "mature": s.get("mean_delta_coverage_minus_no_throttle_mature_update_coverage"),
        "pre_psnr": s.get("mean_pre_match_delta_coverage_minus_no_throttle_psnr"),
        "prune_off": s.get("mean_matching_prune_delta_no_throttle_psnr"),
        "prune_cov": s.get("mean_matching_prune_delta_coverage_throttle_psnr"),
        "wins": s.get("paired_frames_coverage_psnr_wins"),
        "losses": s.get("paired_frames_coverage_psnr_losses"),
    }


def main():
    rows = []
    for seed, (split, folder) in RUNS.items():
        row = load_run(folder)
        row["split"] = split
        rows.append(row)
    schedule_rows = []
    for label, folder in SCHEDULE_RUNS:
        path = RESULTS / folder / "dynamic_matched_map_results.json"
        d = json.loads(path.read_text())
        failure = d.get("count_schedule_gate_failure")
        schedule_rows.append((label, d["summary"], failure))

    lines = [
        "# Replica office0 E1 dynamic matching: calibration transfer", "",
        "Dynamic matched-map runs match cardinality after each frame by pruning the larger arm to the lower realized count. This keeps the pair matched but makes the count path endogenous and changes Gaussian identity/spatial support.", "",
        "| Split | Seed | Frames | Gate | Mean N | PSNR Δ after match | PSNR Δ before match | Depth L1 Δ | Mature coverage Δ | PSNR wins/losses |", "|:--|--:|--:|:--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(f"| {r['split']} | {r['seed']} | {r['frames']}/{r['requested']} | {'PASS' if r['gate'] else 'FAIL'} | {r['mean_n']:.0f} | {r['psnr']:+.3f} dB | {r['pre_psnr']:+.3f} dB | {r['depth']:+.5f} m | {r['mature']:+.5f} | {r['wins']}/{r['losses']} |")
    lines += ["", "## Calibration versus held-out seeds", ""]
    for split in ("calibration", "held-out"):
        group = [r for r in rows if r["split"] == split]
        lines.append(f"- {split}: mean PSNR Δ {sum(r['psnr'] for r in group)/len(group):+.3f} dB; mean depth L1 Δ {sum(r['depth'] for r in group)/len(group):+.5f} m; mean pre-match PSNR Δ {sum(r['pre_psnr'] for r in group)/len(group):+.3f} dB.")
    lines += ["", "## Frozen schedule transfer attempts", "",
              "All schedules were frozen before seed 57. Prefix metrics from gate failures are not complete-trajectory outcomes.", "",
              "| Frozen schedule | Frames | Gate | First failure |", "|:--|--:|:--:|:--|"]
    for label, s, failure in schedule_rows:
        where = "—" if not failure else f"frame {failure['frame']}: target {failure['requested_target']}, realized {failure['realized_counts']}"
        lines.append(f"| {label} | {s['processed_frames']}/{s['requested_online_frames']} | {'PASS' if s['matched_count_gate_passed_every_frame'] else 'FAIL'} | {where} |")
    lines += ["", "The first two natural-floor schedules missed at frame 4 by one Gaussian because their target increment exceeded the held-out realized growth by one. Adding growth-rate margins moved failure later but did not eliminate it: the coverage arm fell below target at frames 55 and 57. A schedule derived from endogenous calibration also failed at frame 57 (465 Gaussian short). These are feasibility failures, not evidence about relative quality.", "",
              "## Interpretation", "",
              "Endogenous matching completed on all four seeds, but the held-out effect must be read with the pruning intervention. Compare pre-match and post-match PSNR deltas and each arm's pruning effect; the no-throttle map can lose quality when pruned to a lower coverage-arm count. This analysis is one scene only. The frozen schedules did not transfer, so do not report their prefix metric means as matched-map E1 results. A robust exogenous target path is not established.", ""]
    out = RESULTS / "replica_e1_office0_dynamic_calibration_transfer.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
