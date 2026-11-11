#!/usr/bin/env python3
"""Freeze an exact matched-count path from one complete dynamic E1 run."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True,
                    help="Complete dynamic matched-map JSON whose per-frame counts define the replay path")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--schedule-id", default="replica-e1-same-seed-replay-count-path-v1")
    ap.add_argument("--evaluation-seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    d = json.loads(source.read_text())
    if not d["summary"].get("matched_count_gate_passed_every_frame"):
        raise ValueError("Source trajectory must pass the full matched-count gate")
    if d.get("target_schedule_id") != "endogenous_lower_of_pair":
        raise ValueError("Source must be an endogenous lower-of-pair trajectory")
    rows = d["frames"]
    if len(rows) != int(d["summary"]["requested_online_frames"]):
        raise ValueError("Source trajectory is incomplete")
    first = rows[0]
    init_n = (int(first["pre_match_counts"]["no_throttle"])
              - int(first["policies"]["no_throttle"]["n_spawned"]))
    targets = [int(r["matched_gaussian_count"]) for r in rows]
    scene_key = d["scene"].removeprefix("replica_")
    record = {
        "schedule_id": args.schedule_id,
        "schedule_definition": "Exact per-frame count path copied from a complete endogenous lower-of-pair source run. Used only as a same-seed selector sensitivity replay; not an independent evaluation schedule.",
        "calibration_source": "same-seed lowest-opacity endogenous matched-map run",
        "calibration_seeds": [int(d["seed"])],
        "evaluation_seed_plan": args.evaluation_seeds,
        "source_run_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scenes": {scene_key: {
            "online_frames": len(rows),
            "initial_gaussians": init_n,
            "targets": targets,
            "source_prune_strategy": d.get("prune_strategy", "lowest_opacity"),
            "source_seed": int(d["seed"]),
        }},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
