#!/usr/bin/env python3
"""Freeze exogenous per-frame Gaussian-count schedules from prior E1 pilots."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=["office0", "office1", "office4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[44, 45])
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, default=[46, 47])
    parser.add_argument("--schedule-id", default="replica-e1-fixed-count-schedule-v1")
    parser.add_argument("--input-root", type=Path, default=ROOT / "results")
    parser.add_argument("--growth-fraction", type=float, default=0.70)
    parser.add_argument("--absolute-margin", type=int, default=0,
                        help="Additional fixed Gaussian margin below the fraction-scaled calibration envelope")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/replica_e1_fixed_count_schedule_v1.json")
    args = parser.parse_args()
    root = args.input_root if args.input_root.is_absolute() else ROOT / args.input_root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not 0.0 <= args.growth_fraction < 1.0:
        raise ValueError("growth fraction must be in [0, 1)")

    scenes = {}
    input_hashes = {}
    for scene in args.scenes:
        runs = []
        for seed in args.seeds:
            path = root / f"replica_E1E2_seed{seed}_capacity_fixed" / f"replica_{scene}" / f"E1_seed{seed}.json"
            record = json.loads(path.read_text())
            if not record.get("valid"):
                raise ValueError(f"Invalid calibration E1 count gate: {path}")
            input_hashes[str(path.relative_to(ROOT))] = sha(path)
            for arm in record["arms"]:
                runs.append(arm["frames"])
        n = min(len(run) for run in runs)
        if len({len(run) for run in runs}) != 1:
            raise ValueError(f"Calibration run lengths differ for {scene}")
        init_counts = [int(run[0]["n_gaussians"]) - int(run[0].get("n_spawned", 0)) for run in runs]
        if len(set(init_counts)) != 1:
            raise ValueError(f"Initialization counts differ for {scene}: {init_counts}")
        initial = init_counts[0]
        lower_envelope = [min(int(run[i]["n_gaussians"]) for run in runs) for i in range(n)]
        targets = [max(initial, initial + int(args.growth_fraction * max(0, count - initial))
                       - args.absolute_margin)
                   for count in lower_envelope]
        scenes[scene] = {
            "online_frames": n,
            "initial_gaussians": initial,
            "growth_fraction_of_calibration_lower_envelope": args.growth_fraction,
            "absolute_margin_gaussians": args.absolute_margin,
            "calibration_seeds": args.seeds,
            "calibration_arms": ["no_throttle", "coverage_throttle"],
            "calibration_lower_envelope": lower_envelope,
            "targets": targets,
        }
    result = {
        "schedule_id": args.schedule_id,
        "schedule_definition": "For each online frame, target = max(initialization_count, initialization_count + floor(growth_fraction * max(0, minimum Gaussian count across both policies and calibration seeds at that frame - initialization_count)) - absolute_margin). The full per-frame target arrays in this file are frozen before the listed evaluation seeds are run.",
        "growth_fraction": args.growth_fraction,
        "absolute_margin_gaussians": args.absolute_margin,
        "calibration_seeds": args.seeds,
        "calibration_source": "existing all-scene E1 capacity-fixed trajectory artifacts",
        "evaluation_seed_plan": args.evaluation_seeds,
        "scenes": scenes,
        "calibration_source_sha256": input_hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
