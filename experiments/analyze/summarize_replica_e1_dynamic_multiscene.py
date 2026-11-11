#!/usr/bin/env python3
"""Summarize paired dynamic matched-map E1 runs for two Replica office scenes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RUNS = {
    "office0": {
        "calibration": {44: "replica_e1_dynamic_matched_map_office0_seed44_calibration",
                        45: "replica_e1_dynamic_matched_map_office0_seed45_calibration"},
        "held-out": {57: "replica_e1_dynamic_matched_map_office0_seed57_heldout",
                     58: "replica_e1_dynamic_matched_map_office0_seed58_heldout"},
    },
    "office1": {
        "calibration": {44: "replica_e1_dynamic_matched_map_office1_seed44_v2",
                        45: "replica_e1_dynamic_matched_map_office1_seed45_v2"},
        "held-out": {57: "replica_e1_dynamic_matched_map_office1_seed57_heldout",
                     58: "replica_e1_dynamic_matched_map_office1_seed58_heldout"},
    },
}


def read(folder):
    path = RESULTS / "replica" / folder / "dynamic_matched_map_results.json"
    d = json.loads(path.read_text())
    s = d["summary"]
    return {
        "folder": folder, "seed": int(d["seed"]),
        "frames": int(s["processed_frames"]), "requested": int(s.get("requested_online_frames", d["online_frames"])),
        "gate": bool(s["matched_count_gate_passed_every_frame"]),
        "mean_n": s["mean_matched_gaussians"],
        "psnr": s["mean_delta_coverage_minus_no_throttle_psnr"],
        "pre_psnr": s["mean_pre_match_delta_coverage_minus_no_throttle_psnr"],
        "depth": s["mean_delta_coverage_minus_no_throttle_depth_l1"],
        "mature": s["mean_delta_coverage_minus_no_throttle_mature_update_coverage"],
        "prune_no": s["mean_matching_prune_delta_no_throttle_psnr"],
        "prune_cov": s["mean_matching_prune_delta_coverage_throttle_psnr"],
        "wins": s["paired_frames_coverage_psnr_wins"],
        "losses": s["paired_frames_coverage_psnr_losses"],
    }


def mean(rows, key):
    return sum(float(r[key]) for r in rows) / len(rows)


def main():
    data = {scene: {split: [read(folder) for folder in seeds.values()]
                    for split, seeds in splits.items()} for scene, splits in RUNS.items()}
    all_pass = all(r["gate"] and r["frames"] == r["requested"]
                   for scene in data.values() for split in scene.values() for r in split)
    lines = [
        "# Replica dynamic matched-map E1 transfer: office0 and office1", "",
        f"All paired count gates complete: **{'yes' if all_pass else 'no'}**.", "",
        "Each frame, both arms were pruned to the lower realized Gaussian count, then continued from that state. Coverage-minus-no-throttle is the sign convention. Depth L1 positive means coverage is worse.", "",
        "| Scene | Split | Seeds | Frames per run | Mean N | PSNR Δ pre/post match | No-throttle prune effect | Coverage prune effect | Depth L1 Δ | Mature coverage Δ |", "|:--|:--|:--|:--|--:|:--|:--|:--|:--|:--|",
    ]
    for scene, splits in data.items():
        for split, rows in splits.items():
            seed_text = ",".join(str(r["seed"]) for r in rows)
            frame_text = ",".join(f"{r['frames']}/{r['requested']}" for r in rows)
            lines.append(f"| {scene} | {split} | {seed_text} | {frame_text} | {mean(rows, 'mean_n'):.0f} | {mean(rows, 'pre_psnr'):+.3f}/{mean(rows, 'psnr'):+.3f} dB | {mean(rows, 'prune_no'):+.3f} dB | {mean(rows, 'prune_cov'):+.3f} dB | {mean(rows, 'depth'):+.5f} m | {mean(rows, 'mature'):+.5f} |")
    lines += ["", "## Held-out consistency", ""]
    for scene, splits in data.items():
        cal, held = splits["calibration"], splits["held-out"]
        direction = "same sign" if mean(cal, "psnr") * mean(held, "psnr") > 0 else "sign changed"
        lines.append(f"- {scene}: calibration post-match PSNR Δ {mean(cal, 'psnr'):+.3f} dB; held-out {mean(held, 'psnr'):+.3f} dB ({direction}). Held-out depth L1 Δ {mean(held, 'depth'):+.5f} m.")
    lines += ["", "## Interpretation", "",
              "This extends the endogenous matched-count diagnostic to a second Replica office scene, with two held-out seeds per scene. It remains a two-scene study, and all four held-out runs share the same dataset family and mapping protocol. Endogenous matching is not a fixed-target causal intervention: count choice depends on the lower realized arm, and opacity pruning changes primitive identity and spatial support. Interpret post-match deltas alongside pre-match deltas and each arm's pruning effect. Paired frames are repeated measures, not independent samples.", "",
              "The office0 fixed-schedule transfer failures are documented separately in `replica_e1_office0_dynamic_calibration_transfer.md`; they remain gate failures and their prefix metrics are excluded here.", ""]
    out = RESULTS / "replica" / "replica_e1_dynamic_multiscene_transfer.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
