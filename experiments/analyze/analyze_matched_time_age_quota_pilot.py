#!/usr/bin/env python3
"""Summarize calibration-only matched-time age-quota pilot replays."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BASE = ROOT / "results/legacy/phase14_generalization_postfix"
RUNS = {
    "initial": ("matched_time_age_quota_pilot_replica_office1", (5000, 4775), False),
    "recalibration2": ("matched_time_age_quota_pilot_replica_office1_recalibration2", (5000, 4494), False),
    "warm_order31": ("matched_time_age_quota_pilot_replica_office1_warm_ordered", (5000, 4494), True),
    "warm_order53": ("matched_time_age_quota_pilot_replica_office1_order53", (5000, 4494), True),
    "warm_order79": ("matched_time_age_quota_pilot_replica_office1_order79", (5000, 4494), True),
}
TOLERANCE = 0.05


def load(folder, expected_updates):
    path = BASE / folder / "replica_office1/seed42_frame90.json"
    record = json.loads(path.read_text())
    if not record["all_count_and_snapshot_gates_passed"]:
        raise RuntimeError(f"Snapshot/quota validity gate failed: {path}")
    arms = {arm["action"]["name"]: arm for arm in record["arms"]}
    for dose in (0, 128):
        for fraction, updates in zip((25, 75), expected_updates):
            arm = arms[f"spawn{dose}_mature{fraction}"]
            if not arm["valid"] or arm["n_micro_updates"] != updates:
                raise RuntimeError(f"Update-count/quota mismatch: {path}, {arm['action']['name']}")
    return record, arms


def pair(record, arms, dose):
    pre = record["pre_action_snapshot_eval"]["psnr"]
    low, high = arms[f"spawn{dose}_mature25"], arms[f"spawn{dose}_mature75"]
    mean_time = (low["opt_stage_sync_ms"] + high["opt_stage_sync_ms"]) / 2
    rel_time = abs(high["opt_stage_sync_ms"] - low["opt_stage_sync_ms"]) / mean_time
    return {
        "time_25_ms": low["opt_stage_sync_ms"],
        "time_75_ms": high["opt_stage_sync_ms"],
        "relative_time_difference": rel_time,
        "matched_time_gate": rel_time <= TOLERANCE,
        "psnr_gain_25_db": low["final_map_eval_psnr"] - pre,
        "psnr_gain_75_db": high["final_map_eval_psnr"] - pre,
        "psnr_75_minus_25_db": high["final_map_eval_psnr"] - low["final_map_eval_psnr"],
        "depth_l1_75_minus_25_m": high["final_map_eval_depth_l1"] - low["final_map_eval_depth_l1"],
    }


def main():
    runs, hashes, warm = {}, set(), []
    for label, (folder, counts, warmed) in RUNS.items():
        record, arms = load(folder, counts)
        hashes.add(record["base_snapshot_hash"])
        runs[label] = {"snapshot_hash": record["base_snapshot_hash"],
                       "execution_order": record.get("action_execution_order"),
                       "warmup": record.get("warmup_record"),
                       "comparisons": {str(d): pair(record, arms, d) for d in (0, 128)}}
        if warmed:
            warm.append(runs[label])
    if len(hashes) != 1:
        raise RuntimeError("Calibration replays do not share the same map snapshot")
    pooled = {}
    for dose in (0, 128):
        t25 = [r["comparisons"][str(dose)]["time_25_ms"] for r in warm]
        t75 = [r["comparisons"][str(dose)]["time_75_ms"] for r in warm]
        m25, m75 = statistics.mean(t25), statistics.mean(t75)
        gap = abs(m75 - m25) / ((m25 + m75) / 2)
        pooled[str(dose)] = {
            "n_repeated_timings_same_snapshot": len(warm),
            "mean_time_25_ms": m25, "sd_time_25_ms": statistics.stdev(t25),
            "mean_time_75_ms": m75, "sd_time_75_ms": statistics.stdev(t75),
            "relative_difference_of_means": gap,
            "matched_time_gate_on_means": gap <= TOLERANCE,
            "mean_psnr_75_minus_25_db": statistics.mean(
                r["comparisons"][str(dose)]["psnr_75_minus_25_db"] for r in warm),
            "mean_depth_l1_75_minus_25_m": statistics.mean(
                r["comparisons"][str(dose)]["depth_l1_75_minus_25_m"] for r in warm),
        }
    out = BASE / "matched_time_age_quota_pilot_replica_office1"
    (out / "analysis.json").write_text(json.dumps({
        "status": "calibration pilot only; one seed, no inferential claim",
        "snapshot_hash": next(iter(hashes)), "time_match_tolerance": TOLERANCE,
        "runs": runs, "pooled_warm_runs": pooled,
    }, indent=2) + "\n")
    lines = [
        "# Matched-time age-quota pilot: Replica office1", "",
        "All five replays branch from the same frame-90 seed-42 full pipeline checkpoint. The primary outcome is zero-spawn; spawn-128 is secondary. A pair passes only if synchronized optimizer-stage time differs by at most 5%. The last three runs include a discarded 1,000-update warm-up and seeded action order. These are repeated timing measurements from one trajectory, not independent seeds.", "",
        "| Replay | Spawn | Updates (25%,75%) | Time (25%,75%) ms | Gap | Gate | PSNR 75%-25% dB | Depth-L1 75%-25% m |", "|---|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for label, result in runs.items():
        folder, counts, _ = RUNS[label]
        for dose in ("0", "128"):
            r = result["comparisons"][dose]
            lines.append(f"| {label} | {dose} | {counts[0]},{counts[1]} | {r['time_25_ms']:.3f},{r['time_75_ms']:.3f} | {r['relative_time_difference']*100:.2f}% | {'PASS' if r['matched_time_gate'] else 'FAIL'} | {r['psnr_75_minus_25_db']:+.4f} | {r['depth_l1_75_minus_25_m']:+.6f} |")
    lines += ["", "## Pooled warmed timing repeats", "",
              "| Spawn | Mean time 25% (SD) ms | Mean time 75% (SD) ms | Gap between means | 5% gate | PSNR difference dB | Depth-L1 difference m |",
              "|---:|---:|---:|---:|:---:|---:|---:|"]
    for dose, r in pooled.items():
        lines.append(f"| {dose} | {r['mean_time_25_ms']:.3f} ({r['sd_time_25_ms']:.3f}) | {r['mean_time_75_ms']:.3f} ({r['sd_time_75_ms']:.3f}) | {r['relative_difference_of_means']*100:.2f}% | {'PASS' if r['matched_time_gate_on_means'] else 'FAIL'} | {r['mean_psnr_75_minus_25_db']:+.4f} | {r['mean_depth_l1_75_minus_25_m']:+.6f} |")
    lines += [
        "", "## Interpretation", "",
        "The first two calibrations were sensitive to action order, including a 196.630 ms first-action outlier. Warm-up and shuffled order reduced that extreme, but the three warmed runs still failed the 5% gate on pooled mean times: the gaps were 5.12% at zero spawn and 7.09% at spawn 128. Within-run timing variation also remained visible. Across all branches, the quota-75% maps had higher PSNR and lower depth L1, but unequal measured optimizer time prevents a compute-matched causal claim. The warm-run PSNR contrast repeats the same checkpoint intervention and does not increase trajectory-level sample size.",
        "",
        "A reliable matched-time protocol needs a lower-variance stage timer or a closed-loop time-budgeted optimizer. Do not scale this pilot to a confirmatory scene/seed claim until the measured-time gate is consistently attainable.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print(out / "report.md")


if __name__ == "__main__":
    main()
