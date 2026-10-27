#!/usr/bin/env python3
"""Audit causal-control gates for the matched-candidate timing replay."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_pair(path, expected_candidates, expected_updates):
    record = json.loads(path.read_text())
    arms = record["arms"]
    source_hashes_path = path.parents[1] / "source_hashes.json"
    source_hashes = json.loads(source_hashes_path.read_text()) if source_hashes_path.exists() else {}
    prefix_artifact = Path(record.get("frozen_prefix_checkpoint", ""))
    candidate_artifact = Path(record.get("frozen_candidate_file", ""))
    prefix_file_ok = prefix_artifact.is_file() and hashlib.sha256(
        prefix_artifact.read_bytes()).hexdigest() == record.get("frozen_prefix_checkpoint_sha256")
    candidate_file_ok = candidate_artifact.is_file() and hashlib.sha256(
        candidate_artifact.read_bytes()).hexdigest() == record.get("frozen_candidate_file_sha256")
    by_condition = {
        f"{arm.get('arm', {}).get('timing')}_{arm.get('arm', {}).get('allocation')}": arm
        for arm in arms if arm.get("arm", {}).get("timing") is not None
    }
    if not by_condition:
        by_condition = {arm.get("arm", {}).get("name"): arm for arm in arms}
    arm_candidate_hashes = [arm.get("candidate_batch_sha256") for arm in arms]
    has_per_arm_candidate_hashes = all(arm_candidate_hashes)
    checks = {
        "common_prefix_digest": (record.get("prefix_state_sha256") is not None
            and all(arm.get("prefix_state_sha256") == record.get("prefix_state_sha256")
                    for arm in arms)),
        "same_candidate_batch": (
            all(value == record.get("candidate_batch_sha256") for value in arm_candidate_hashes)
            if has_per_arm_candidate_hashes else
            bool(record.get("candidate_batch_sha256"))
            and bool(record.get("candidate_uv_sha256"))
            and bool(source_hashes.get("experiments/run_matched_candidate_timing_replay.py"))
        ),
        "exact_manual_birth_count": all(
            arm.get("manual_batch_births") == expected_candidates for arm in arms
        ),
        "no_automatic_births": all(arm.get("automatic_births") == 0 for arm in arms),
        "exact_updates_per_frame": all(
            arm.get("exact_micro_update_gate")
            and arm.get("total_micro_updates") == expected_updates
            for arm in arms
        ),
        "same_heldout_views": len({tuple(arm.get("final_map_heldout_frames", []))
                                    for arm in arms}) == 1,
        "same_starting_map_size": len({arm.get("start_gaussians") for arm in arms}) == 1,
        "same_total_update_budget": len({arm.get("total_micro_updates") for arm in arms}) == 1,
        "frozen_prefix_checkpoint_hash": prefix_file_ok,
        "frozen_candidate_file_hash": candidate_file_ok,
    }
    if "early_global" in by_condition and "early_stale_rescue" in by_condition:
        for timing in ("early", "late"):
            global_arm = by_condition[f"{timing}_global"]
            rescue_arm = by_condition[f"{timing}_stale_rescue"]
            checks[f"stale_rescue_first_stage_{timing}"] = (
                rescue_arm.get("total_stale_mature_micro_updates", 0)
                > global_arm.get("total_stale_mature_micro_updates", 0)
            )
    return {
        "artifact": str(path),
        "scene": record.get("scene"),
        "seed": record.get("seed"),
        "valid": all(checks.values()),
        "checks": checks,
        "final_gaussian_counts_by_arm": {name: arm.get("final_gaussians")
                                          for name, arm in by_condition.items()},
        "runner_reported_valid_before_reassessment": record.get("valid"),
        "note": ("Final map size is an outcome and is intentionally not a matching gate. "
                 + ("The earlier runner's valid flag also required equal final count; "
                    "this audit reassesses the causal-control gates without editing raw runs."
                    if record.get("valid") is False else
                    "The runner's valid flag and this audit both use causal-control gates.")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("results/matched_candidate_timing_replay_verified"))
    ap.add_argument("--scenes", nargs="+", default=["replica_room2", "replica_office3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[57, 58, 59, 60])
    ap.add_argument("--run_ids", nargs="*", default=None,
                    help="Optional repeated-run suffixes; expects seedN_SUFFIX.json")
    ap.add_argument("--candidate_count", type=int, default=5000)
    ap.add_argument("--updates_per_frame", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    expected_updates = args.updates_per_frame * (119 - 6)
    pairs = []
    run_ids = args.run_ids if args.run_ids else [None]
    for scene in args.scenes:
        for seed in args.seeds:
            for run_id in run_ids:
                suffix = f"_{run_id}" if run_id else ""
                pair = validate_pair(
                    args.root / scene / f"seed{seed}{suffix}.json",
                    args.candidate_count, expected_updates)
                pair["run_id"] = run_id or "initial"
                pairs.append(pair)
    report = {
        "schema": "matched-candidate-timing-validation-v1",
        "root": str(args.root),
        "expected_candidate_count_per_arm": args.candidate_count,
        "expected_total_micro_updates_per_arm": expected_updates,
        "valid_pairs": sum(pair["valid"] for pair in pairs),
        "total_pairs": len(pairs),
        "pairs": pairs,
    }
    out = args.out or args.root / "validation.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {report['valid_pairs']}/{report['total_pairs']} pairs; wrote {out}")


if __name__ == "__main__":
    main()
