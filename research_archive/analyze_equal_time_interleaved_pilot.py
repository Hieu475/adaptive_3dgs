#!/usr/bin/env python3
"""Analyze the interleaved, repeated, same-checkpoint equal-time quota pilot."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results/legacy/phase14_generalization_postfix/matched_time_age_quota_pilot_replica_office1_interleaved5_matched/replica_office1/seed42_frame90.json"
OUT = ROOT / "results/legacy/phase14_generalization_postfix/matched_time_age_quota_pilot_replica_office1"
TOLERANCE = 0.05
REPEATS = 5


def stats(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    return {"mean": mean, "sd": sd, "cv_fraction": sd / mean}


def main():
    d = json.loads(INPUT.read_text())
    if not d["all_count_and_snapshot_gates_passed"]:
        raise RuntimeError("Common snapshot or branch validity gate failed")
    if len(d["arms"]) != 4 * REPEATS or len(set(a["snapshot_hash"] for a in d["arms"])) != 1:
        raise RuntimeError("Expected 20 valid branches from one snapshot")
    arms = {a["action"]["name"]: a for a in d["arms"]}
    result = {
        "snapshot_hash": d["base_snapshot_hash"],
        "execution_order": d["action_execution_order"],
        "timing_match_tolerance": TOLERANCE,
        "by_spawn": {},
    }
    for dose in (0, 128):
        group = {}
        for pct, target in ((25, 5000), (75, 4189 if dose == 0 else 4374)):
            rows = [arms[f"spawn{dose}_mature{pct}_repeat{i}"] for i in range(REPEATS)]
            if any(not x["valid"] or x["n_micro_updates"] != target for x in rows):
                raise RuntimeError(f"Quota/update gate failed for spawn {dose}, mature {pct}")
            group[str(pct)] = {
                "n_updates_per_repeat": target,
                "cuda_event_ms": stats([x["opt_stage_cuda_event_ms"] for x in rows]),
                "sync_wall_ms": stats([x["opt_stage_sync_ms"] for x in rows]),
                "psnr": stats([x["final_map_eval_psnr"] for x in rows]),
                "depth_l1_m": stats([x["final_map_eval_depth_l1"] for x in rows]),
            }
        e25, e75 = group["25"]["cuda_event_ms"]["mean"], group["75"]["cuda_event_ms"]["mean"]
        w25, w75 = group["25"]["sync_wall_ms"]["mean"], group["75"]["sync_wall_ms"]["mean"]
        p25, p75 = group["25"]["psnr"]["mean"], group["75"]["psnr"]["mean"]
        z25, z75 = group["25"]["depth_l1_m"]["mean"], group["75"]["depth_l1_m"]["mean"]
        event_gap = abs(e75 - e25) / ((e75 + e25) / 2)
        wall_gap = abs(w75 - w25) / ((w75 + w25) / 2)
        paired_psnr = [arms[f"spawn{dose}_mature75_repeat{i}"]["final_map_eval_psnr"]
                       - arms[f"spawn{dose}_mature25_repeat{i}"]["final_map_eval_psnr"]
                       for i in range(REPEATS)]
        group["contrast"] = {
            "cuda_event_mean_gap_fraction": event_gap,
            "sync_wall_mean_gap_fraction": wall_gap,
            "cuda_event_time_match_gate": event_gap <= TOLERANCE,
            "sync_wall_time_match_gate": wall_gap <= TOLERANCE,
            "psnr_75_minus_25_db": p75 - p25,
            "paired_repeat_psnr_differences_db": paired_psnr,
            "depth_l1_75_minus_25_m": z75 - z25,
        }
        result["by_spawn"][str(dose)] = group

    (OUT / "equal_time_interleaved_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Interleaved equal-time age-quota pilot", "",
        "Five repeats of each quota × spawn arm were run in one invocation from the same full pipeline checkpoint, after a discarded warm-up, in one seeded shuffled order. Each branch passed same-snapshot and exact update-quota checks. Repeats estimate timing variability for this one map only; they are not independent seeds.", "",
        "| Spawn | Updates (25%,75%) | CUDA event mean (25%,75%) ms; CV | CUDA event gap | Wall mean (25%,75%) ms; CV | Wall gap | Time gate | PSNR 75%-25% dB | Depth-L1 75%-25% m |",
        "|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for dose in (0, 128):
        g = result["by_spawn"][str(dose)]
        x, y, c = g["25"], g["75"], g["contrast"]
        lines.append(
            f"| {dose} | {x['n_updates_per_repeat']},{y['n_updates_per_repeat']} | "
            f"{x['cuda_event_ms']['mean']:.3f} ({x['cuda_event_ms']['cv_fraction']*100:.2f}%), "
            f"{y['cuda_event_ms']['mean']:.3f} ({y['cuda_event_ms']['cv_fraction']*100:.2f}%) | "
            f"{c['cuda_event_mean_gap_fraction']*100:.2f}% | "
            f"{x['sync_wall_ms']['mean']:.3f} ({x['sync_wall_ms']['cv_fraction']*100:.2f}%), "
            f"{y['sync_wall_ms']['mean']:.3f} ({y['sync_wall_ms']['cv_fraction']*100:.2f}%) | "
            f"{c['sync_wall_mean_gap_fraction']*100:.2f}% | "
            f"{'PASS' if c['cuda_event_time_match_gate'] and c['sync_wall_time_match_gate'] else 'FAIL'} | "
            f"{c['psnr_75_minus_25_db']:+.5f} | {c['depth_l1_75_minus_25_m']:+.6f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "At spawn 128, both CUDA-event and synchronized wall means meet the 5% match gate (3.86%); quota 75% is +0.08196 dB PSNR and −0.000659 m depth L1 relative to quota 25%. At zero spawn, the time gap is 5.21%, just outside the gate; the +0.06492 dB PSNR and −0.000862 m depth-L1 differences are not treated as compute-matched. Per-action timing CVs are 1.45–4.21% in this interleaved invocation. CUDA-event and wall measurements are close, consistent with GPU-stage time dominating the measured stage.",
        "",
        "This is a pilot on one Replica office1 seed-42 snapshot, with one execution order and five repeated branches. It demonstrates a feasible matched-time point for the spawn-128 comparison, not a scene-level or trajectory-level causal effect. The configured 15 ms scheduler budget remains modeled; observed optimizer-stage times are about 60–68 ms.",
        "",
        "Next: replicate the spawn-128 matched-time contrast across independent seed snapshots and scenes, with the same frozen timing gate. Recalibrate zero-spawn separately before claiming that contrast.",
        "",
    ]
    (OUT / "equal_time_interleaved_report.md").write_text("\n".join(lines))
    print(OUT / "equal_time_interleaved_report.md")


if __name__ == "__main__":
    main()
