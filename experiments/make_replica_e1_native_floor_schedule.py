#!/usr/bin/env python3
"""Freeze per-frame E1 count targets below a calibration native-pruning envelope."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower_quantile(values, q):
    """Empirical lower quantile using the conservative nearest-rank index."""
    ordered = sorted(int(v) for v in values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="+", default=["office0", "office1", "office4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[44, 45])
    ap.add_argument("--evaluation-seeds", nargs="+", type=int, default=[57, 58])
    ap.add_argument("--input-root", type=Path, default=ROOT / "results")
    ap.add_argument("--lower-quantile", type=float, default=0.0,
                    help="Empirical quantile across calibration seed-arm counts; 0 selects the minimum")
    ap.add_argument("--margin", type=int, default=2000,
                    help="Subtract this many Gaussians from the per-frame calibration quantile")
    ap.add_argument("--growth-margin-per-frame", type=int, default=50,
                    help="Reduce each positive target increment by this many Gaussians to tolerate seed variation")
    ap.add_argument("--schedule-id", default="replica-e1-native-floor-schedule-v1")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/replica_e1_native_floor_schedule_v1.json")
    args = ap.parse_args()
    if not 0 <= args.lower_quantile <= 1 or args.margin < 0 or args.growth_margin_per_frame < 0:
        raise ValueError("lower quantile must be in [0,1] and margins must be non-negative")
    source_root = args.input_root if args.input_root.is_absolute() else ROOT / args.input_root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    source_hashes, scene_schedules = {}, {}
    for scene in args.scenes:
        runs = []
        for seed in args.seeds:
            path = source_root / f"replica_E1E2_seed{seed}_capacity_fixed" / f"replica_{scene}" / f"E1_seed{seed}.json"
            record = json.loads(path.read_text())
            if not record.get("valid"):
                raise ValueError(f"Invalid calibration artifact: {path}")
            if len(record["arms"]) != 2:
                raise ValueError(f"Expected paired E1 calibration arms: {path}")
            source_hashes[str(path.relative_to(ROOT))] = sha256(path)
            for arm in record["arms"]:
                runs.append((seed, arm["condition"], arm["frames"]))
        lengths = {len(frames) for _, _, frames in runs}
        if len(lengths) != 1:
            raise ValueError(f"Calibration frame counts differ for {scene}: {sorted(lengths)}")
        initial_counts = {int(frames[0]["n_gaussians"]) - int(frames[0].get("n_spawned", 0))
                          for _, _, frames in runs}
        if len(initial_counts) != 1:
            raise ValueError(f"Calibration initialization counts differ for {scene}: {sorted(initial_counts)}")
        per_frame = []
        previous_target = None
        for i in range(next(iter(lengths))):
            samples = [int(frames[i]["n_gaussians"]) for _, _, frames in runs]
            q_count = lower_quantile(samples, args.lower_quantile)
            sample_deltas = ([int(frames[i]["n_gaussians"]) - int(frames[i - 1]["n_gaussians"])
                              for _, _, frames in runs] if i else [])
            q_delta = lower_quantile(sample_deltas, args.lower_quantile) if sample_deltas else None
            raw_target = max(1, q_count - args.margin)
            if previous_target is None:
                target = raw_target
            elif raw_target < previous_target:
                target = raw_target
            else:
                safe_increment = max(0, int(q_delta) - args.growth_margin_per_frame)
                target = min(raw_target, previous_target + safe_increment)
            per_frame.append({"frame": int(runs[0][2][i]["frame"]), "sample_counts": samples,
                              "sample_count_deltas": sample_deltas,
                              "lower_quantile_count": q_count,
                              "lower_quantile_count_delta": q_delta,
                              "raw_target_before_growth_cap": raw_target,
                              "target": target})
            previous_target = target
        scene_schedules[scene] = {
            "online_frames": len(per_frame),
            "initial_gaussians": next(iter(initial_counts)),
            "calibration_seeds": args.seeds,
            "calibration_samples_per_frame": len(runs),
            "calibration_arms": [condition for _, condition, _ in runs[:2]],
            "lower_quantile": args.lower_quantile,
            "margin_gaussians": args.margin,
            "growth_margin_per_frame_gaussians": args.growth_margin_per_frame,
            "native_pruning_lower_quantile": [x["lower_quantile_count"] for x in per_frame],
            "targets": [x["target"] for x in per_frame],
            "calibration_samples": per_frame,
        }
    result = {
        "schedule_id": args.schedule_id,
        "schedule_definition": "At each online frame, compute the empirical lower quantile of naturally realized E1 Gaussian counts over calibration seeds and both arms, subtract the fixed count margin, and cap positive target increments by the empirical lower quantile of per-frame count increments minus the fixed per-frame growth margin. Downward target changes follow the count floor. Targets are frozen before evaluation; the runner reports a gate failure instead of adapting if either test arm falls below target.",
        "calibration_source": "paired E1 capacity-fixed natural trajectories; no dynamic count matching applied within the trajectory",
        "calibration_seeds": args.seeds,
        "evaluation_seed_plan": args.evaluation_seeds,
        "lower_quantile": args.lower_quantile,
        "margin_gaussians": args.margin,
        "growth_margin_per_frame_gaussians": args.growth_margin_per_frame,
        "scenes": scene_schedules,
        "calibration_source_sha256": source_hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
