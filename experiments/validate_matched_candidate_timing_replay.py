#!/usr/bin/env python3
"""Audit causal-control gates for the matched-candidate timing replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def validate_pair(path, expected_candidates, expected_updates):
    record = json.loads(path.read_text())
    arms = record["arms"]
    source_hashes_path = path.parents[1] / "source_hashes.json"
    source_hashes = json.loads(source_hashes_path.read_text()) if source_hashes_path.exists() else {}
    recorded_source_hashes = record.get("source_hashes", {})
    source_inventory_matches = bool(source_hashes) and recorded_source_hashes == source_hashes
    source_checkout_matches = bool(source_hashes) and all(
        (Path(relpath).is_file()
         and hashlib.sha256(Path(relpath).read_bytes()).hexdigest() == expected)
        for relpath, expected in source_hashes.items()
    )
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
        "source_inventory_matches_run": source_inventory_matches,
        "source_files_match_current_checkout": source_checkout_matches,
    }
    metric_required = {
        "psnr_valid_gt", "ssim_standard", "ssim_legacy", "depth_l1_gt_valid",
        "depth_rmse_gt_valid", "depth_l1_overlap", "depth_rmse_overlap",
        "pred_depth_coverage", "valid_gt_pixel_count", "overlap_pixel_count",
    }
    metric_rows = [view for arm in arms for view in arm.get("fixed_heldout_views", [])]
    per_frame_rows = [frame for arm in arms for frame in arm.get("frames", [])]
    metric_fields_complete = bool(metric_rows) and all(
        metric_required.issubset(view) for view in metric_rows
    ) and all(metric_required.issubset(frame) for frame in per_frame_rows)
    metric_values_valid = bool(metric_rows) and all(
        math.isfinite(float(view["pred_depth_coverage"]))
        and 0.0 <= float(view["pred_depth_coverage"]) <= 1.0
        and int(view["overlap_pixel_count"]) <= int(view["valid_gt_pixel_count"])
        and int(view["valid_gt_pixel_count"]) > 0
        and math.isfinite(float(view["ssim_standard"]))
        and (int(view["overlap_pixel_count"]) == 0
             or math.isfinite(float(view["depth_l1_overlap"])))
        for view in metric_rows
    )
    checks["rgbd_metric_v2_fields_complete"] = metric_fields_complete
    checks["rgbd_metric_v2_values_valid"] = metric_values_valid
    if str(record.get("protocol", {}).get("device", "")).startswith("cuda"):
        checks["cuda_stage_timing_recorded"] = bool(per_frame_rows) and all(
            frame.get("time_opt_cuda_ms") is not None
            and frame.get("time_opt_stage_wall_ms") is not None
            for frame in per_frame_rows
        )
    else:
        checks["cuda_stage_timing_recorded"] = True
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
