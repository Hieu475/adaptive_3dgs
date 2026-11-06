#!/usr/bin/env python3
"""Freeze matched-count targets from complete endogenous matched-map calibrations."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower_quantile(values, q):
    values = sorted(int(v) for v in values)
    return values[max(0, math.ceil(q * len(values)) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="Scene key, e.g. office0")
    ap.add_argument("--calibration-files", nargs="+", type=Path, required=True)
    ap.add_argument("--evaluation-seeds", nargs="+", type=int, required=True)
    ap.add_argument("--lower-quantile", type=float, default=0.0)
    ap.add_argument("--margin", type=int, default=1000)
    ap.add_argument("--growth-margin-per-frame", type=int, default=100)
    ap.add_argument("--schedule-id", default="replica-e1-dynamic-calibrated-v1")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 0 <= args.lower_quantile <= 1 or min(args.margin, args.growth_margin_per_frame) < 0:
        raise ValueError("quantile must be in [0,1] and margins must be non-negative")
    files = [p if p.is_absolute() else ROOT / p for p in args.calibration_files]
    records = []
    for path in files:
        d = json.loads(path.read_text())
        if d.get("scene", "").removeprefix("replica_") != args.scene:
            raise ValueError(f"Wrong scene in calibration file: {path}")
        summary = d["summary"]
        if not summary.get("matched_count_gate_passed_every_frame"):
            raise ValueError(f"Incomplete or gate-failed calibration: {path}")
        if d.get("target_schedule_id") != "endogenous_lower_of_pair" or d.get("target_schedule_sha256"):
            raise ValueError(f"Calibration must use endogenous lower-of-pair matching: {path}")
        records.append((int(d["seed"]), path, d))
    if len({seed for seed, _, _ in records}) != len(records):
        raise ValueError("Calibration seeds must be unique")
    lengths = {len(d["frames"]) for _, _, d in records}
    if len(lengths) != 1:
        raise ValueError(f"Calibration frame lengths differ: {sorted(lengths)}")
    initial_counts = set()
    for _, _, d in records:
        first = d["frames"][0]
        policy = first["policies"]["no_throttle"]
        initial_counts.add(int(first["pre_match_counts"]["no_throttle"]) - int(policy["n_spawned"]))
    if len(initial_counts) != 1:
        raise ValueError(f"Initialization counts differ: {sorted(initial_counts)}")

    per_frame = []
    previous = None
    n = next(iter(lengths))
    for i in range(n):
        counts = [int(d["frames"][i]["matched_gaussian_count"]) for _, _, d in records]
        q_count = lower_quantile(counts, args.lower_quantile)
        deltas = ([int(d["frames"][i]["matched_gaussian_count"])
                   - int(d["frames"][i - 1]["matched_gaussian_count"])
                   for _, _, d in records] if i else [])
        q_delta = lower_quantile(deltas, args.lower_quantile) if deltas else None
        raw = max(1, q_count - args.margin)
        if previous is None or raw < previous:
            target = raw
        else:
            target = min(raw, previous + max(0, int(q_delta) - args.growth_margin_per_frame))
        per_frame.append({"frame": int(records[0][2]["frames"][i]["online_frame"]),
                          "calibration_counts": counts, "lower_quantile_count": q_count,
                          "calibration_count_deltas": deltas,
                          "lower_quantile_count_delta": q_delta,
                          "raw_target": raw, "target": target})
        previous = target

    scene_schedule = {
        "online_frames": n,
        "initial_gaussians": next(iter(initial_counts)),
        "calibration_seeds": [seed for seed, _, _ in records],
        "calibration_runs": len(records),
        "lower_quantile": args.lower_quantile,
        "margin_gaussians": args.margin,
        "growth_margin_per_frame_gaussians": args.growth_margin_per_frame,
        "endogenous_matched_count_lower_quantile": [x["lower_quantile_count"] for x in per_frame],
        "targets": [x["target"] for x in per_frame],
        "calibration_samples": per_frame,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = {
        "schedule_id": args.schedule_id,
        "schedule_definition": "For each frame, take the empirical lower quantile of the endogenous matched Gaussian counts from complete calibration runs, subtract a fixed margin, and cap positive target increments by the lower quantile of per-frame calibration count increments minus a fixed growth margin. The target path is frozen before held-out evaluation; the runner records a gate failure without adapting if either arm falls short.",
        "calibration_source": "complete endogenous lower-of-pair dynamic matched-map runs",
        "calibration_seeds": [seed for seed, _, _ in records],
        "evaluation_seed_plan": args.evaluation_seeds,
        "lower_quantile": args.lower_quantile,
        "margin_gaussians": args.margin,
        "growth_margin_per_frame_gaussians": args.growth_margin_per_frame,
        "scenes": {args.scene: scene_schedule},
        "calibration_source_sha256": {str(path.relative_to(ROOT)): sha(path) for _, path, _ in records},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
