#!/usr/bin/env python3
"""Summarize fixed-schedule E1 gate outcomes without pooling partial runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "replica_e1_fixed_schedule_calibration_summary.md"


def main():
    paths = sorted(RESULTS.glob("replica_e1_fixed_*_office*_seed*/dynamic_matched_map_results.json"))
    runs = []
    for path in paths:
        data = json.loads(path.read_text())
        summary = data["summary"]
        failure = data.get("count_schedule_gate_failure")
        runs.append({
            "folder": path.parent.name,
            "scene": data["scene"].removeprefix("replica_"),
            "seed": int(data["seed"]),
            "schedule": data.get("target_schedule_id"),
            "processed": int(summary["processed_frames"]),
            "requested": int(summary["requested_online_frames"]),
            "gate": bool(summary["matched_count_gate_passed_every_frame"]),
            "mean_n": summary.get("mean_matched_gaussians"),
            "delta_psnr": summary.get("mean_delta_coverage_minus_no_throttle_psnr"),
            "pre_delta_psnr": summary.get("mean_pre_match_delta_coverage_minus_no_throttle_psnr"),
            "delta_depth": summary.get("mean_delta_coverage_minus_no_throttle_depth_l1"),
            "fail": failure,
            "path": str(path.parent.relative_to(ROOT)),
        })
    valid = [r for r in runs if r["gate"]]
    lines = [
        "# Replica E1 fixed-count schedule calibration and evaluation", "",
        "The schedule was derived from existing seed 44/45 natural E1 trajectories. Failed schedules are preserved as gate failures; their partial metric means are not pooled as full-trajectory results.", "",
        "| Schedule | Scene | Seed | Frames matched/requested | Count gate | Mean matched N | PSNR Δ after match | PSNR Δ before match | Depth L1 Δ after match | Gate failure |",
        "|:--|:--|--:|--:|:--|--:|--:|--:|--:|:--|",
    ]
    for r in runs:
        psnr = "—" if r["delta_psnr"] is None else f"{r['delta_psnr']:+.3f} dB"
        pre = "—" if r["pre_delta_psnr"] is None else f"{r['pre_delta_psnr']:+.3f} dB"
        depth = "—" if r["delta_depth"] is None else f"{r['delta_depth']:+.5f} m"
        n = "—" if r["mean_n"] is None else f"{r['mean_n']:.0f}"
        failure = "—" if not r["fail"] else f"frame {r['fail']['frame']} short: " + ", ".join(
            f"{k}={v}" for k, v in r["fail"]["realized_counts"].items()) + f" < target {r['fail']['requested_target']}"
        lines.append(f"| {r['schedule']} | {r['scene']} | {r['seed']} | {r['processed']}/{r['requested']} | "
                     f"{'PASS' if r['gate'] else 'FAIL'} | {n} | {psnr} | {pre} | {depth} | {failure} |")
    lines += ["", "## Interpretation", ""]
    if valid:
        lines.append(f"Full-schedule count gates passed in {len(valid)}/{len(runs)} runs. Only gate-passing runs contribute complete-trajectory contrasts.")
        by_scene = {}
        for r in valid:
            by_scene.setdefault((r["schedule"], r["scene"]), []).append(r)
        for (schedule, scene), rows in sorted(by_scene.items()):
            if len(rows) > 1:
                vals = [r["delta_psnr"] for r in rows]
                lines.append(f"- {schedule}, {scene}: matched-count mean PSNR delta {sum(vals)/len(vals):+.3f} dB across {len(rows)} seeds.")
    else:
        lines.append("No run passed the complete fixed schedule gate.")
    lines += [
        "", "Gate failures show that schedules calibrated on seeds 44/45 do not transfer reliably. Progressive schedules v1–v4 failed before the end of office0; v5 passed office0 seeds 53/54 but failed on office1 seed 53. The fixed-initial schedule passed office0 seeds 55/56 but failed reproducibly on office1 (both seeds at frame 54) and office4 (both seeds at frame 51), because native pruning reduced the population below the fixed target.",
        "", "Complete runs are exploratory and must not be pooled across scenes or different target levels. The office0 fixed-initial run has a severe capacity restriction; v5 uses a higher target. In both cases, opacity pruning changes Gaussian identities and spatial coverage, so equal cardinality alone is not a clean causal intervention. Failed runs contain prefix-only metrics and are not full-trajectory results.",
        "", "Next, calibrate scene-specific target trajectories against the post-native-pruning lower envelope using independent calibration seeds, then freeze them before evaluation on untouched seeds. This batch does not establish a cross-scene effect or isolate the causal mechanism.", "",
    ]
    OUTPUT.write_text("\n".join(lines))
    print(OUTPUT)


if __name__ == "__main__":
    main()
