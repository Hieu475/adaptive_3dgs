#!/usr/bin/env python3
"""Analyze ten-repeat timing calibration on saved Replica office1 checkpoints."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/legacy/phase14_generalization_postfix/replica_office1_matched_time_repeatability_v2"
DATA = OUT / "replica_office1"
SOURCE = ROOT / "results/legacy/phase14_generalization_postfix/matched_time_spawn128_cross_scene_replication_v1/replica_office1"
TOL = 0.05


def summary(xs):
    m = statistics.mean(xs)
    sd = statistics.stdev(xs)
    return {"mean": m, "sd": sd, "cv_fraction": sd / m}


def main():
    seeds = []
    for seed in range(260, 264):
        d = json.loads((DATA / f"seed{seed}_frame90.json").read_text())
        prior = json.loads((SOURCE / f"seed{seed}_frame90.json").read_text())
        if not d["all_count_and_snapshot_gates_passed"]:
            raise RuntimeError(f"Snapshot/quota validity failed for seed {seed}")
        if d["base_snapshot_hash"] != prior["base_snapshot_hash"]:
            raise RuntimeError(f"Snapshot changed for seed {seed}")
        arms = {a["action"]["name"]: a for a in d["arms"]}
        result = {"seed": seed, "snapshot_hash": d["base_snapshot_hash"], "quota": {}}
        for pct, updates in ((25, 5000), (75, 4374)):
            rows = [arms[f"spawn128_mature{pct}_repeat{i}"] for i in range(10)]
            if any(not a["valid"] or a["n_micro_updates"] != updates or a["n_spawned"] != 128 for a in rows):
                raise RuntimeError(f"Action quota failed for seed {seed}, {pct}%")
            result["quota"][str(pct)] = {
                "cuda_event_ms": summary([a["opt_stage_cuda_event_ms"] for a in rows]),
                "sync_wall_ms": summary([a["opt_stage_sync_ms"] for a in rows]),
                "frame_wall_ms": summary([a["frame_time_ms"] for a in rows]),
                "psnr": summary([a["final_map_eval_psnr"] for a in rows]),
                "depth_l1_m": summary([a["final_map_eval_depth_l1"] for a in rows]),
            }
        low, high = result["quota"]["25"], result["quota"]["75"]
        event_gap = abs(high["cuda_event_ms"]["mean"] - low["cuda_event_ms"]["mean"]) / statistics.mean(
            [high["cuda_event_ms"]["mean"], low["cuda_event_ms"]["mean"]])
        wall_gap = abs(high["sync_wall_ms"]["mean"] - low["sync_wall_ms"]["mean"]) / statistics.mean(
            [high["sync_wall_ms"]["mean"], low["sync_wall_ms"]["mean"]])
        repeatability = all(result["quota"][str(p)]["cuda_event_ms"]["cv_fraction"] <= TOL and
                            result["quota"][str(p)]["sync_wall_ms"]["cv_fraction"] <= TOL for p in (25, 75))
        time_match = event_gap <= TOL and wall_gap <= TOL
        result["cuda_event_mean_gap_fraction"] = event_gap
        result["sync_wall_mean_gap_fraction"] = wall_gap
        result["repeatability_gate"] = repeatability
        result["mean_time_match_gate"] = time_match
        result["matched_time_gate"] = repeatability and time_match
        result["psnr_75_minus_25_db"] = high["psnr"]["mean"] - low["psnr"]["mean"]
        result["depth_l1_75_minus_25_m"] = high["depth_l1_m"]["mean"] - low["depth_l1_m"]["mean"]
        seeds.append(result)

    report = {
        "status": "same-checkpoint timing calibration; no new trajectory replication",
        "cv_and_pair_gap_threshold": TOL,
        "repeats_per_arm": 10,
        "seed_results": seeds,
        "matched_time_snapshots": sum(x["matched_time_gate"] for x in seeds),
    }
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Replica office1 matched-time repeatability v2", "",
        "Ten interleaved repeats of each quota arm were run from each of four exact saved Replica office1 frame-90 checkpoints. Snapshot/map, spawn=128, and exact update/quota checks passed for all 80 branches. Timing gate: each arm CV ≤5% for both CUDA-event and synchronized wall time, and quota mean gaps ≤5% for both endpoints.", "",
        "| Seed | Event ms (25%/75%) | Event CV (25%/75%) | Wall CV (25%/75%) | Event gap | Wall gap | Gate | PSNR Δ dB | Depth-L1 Δ m |",
        "|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for x in seeds:
        a, b = x["quota"]["25"], x["quota"]["75"]
        lines.append(
            f"| {x['seed']} | {a['cuda_event_ms']['mean']:.2f}/{b['cuda_event_ms']['mean']:.2f} | "
            f"{a['cuda_event_ms']['cv_fraction']*100:.2f}%/{b['cuda_event_ms']['cv_fraction']*100:.2f}% | "
            f"{a['sync_wall_ms']['cv_fraction']*100:.2f}%/{b['sync_wall_ms']['cv_fraction']*100:.2f}% | "
            f"{x['cuda_event_mean_gap_fraction']*100:.2f}% | {x['sync_wall_mean_gap_fraction']*100:.2f}% | "
            f"{'PASS' if x['matched_time_gate'] else 'FAIL'} | "
            f"{x['psnr_75_minus_25_db']:+.5f} | {x['depth_l1_75_minus_25_m']:+.6f} |")
    lines += [
        "", f"## Result: {report['matched_time_snapshots']}/4 snapshots pass", "",
        "Only seed 263 passes the full repeatability and matched-mean gate. Seeds 260 and 261 have large within-action timing variation (CUDA-event CV 42–58%); seed 262 is close but misses at 5.06% CV for the 75% arm. Increasing from five to ten repeats did not make the timing endpoint reliable across Replica snapshots. The timing values are measured optimizer-stage duration; they are not the modeled 15 ms scheduler budget.", "",
        "The quota-75% PSNR and depth differences are retained per seed, but only seed 263 (+0.03798 dB PSNR, −0.000653 m Depth L1) passes the time gate. This single passing snapshot is not a scene-level matched-compute estimate. These are repeated branches of four previously observed trajectories, not new seed samples.", "",
        "Decision: stop count tuning for these checkpoints. Before further Replica matched-compute claims, identify why same-checkpoint GPU-stage timing varies by seed/action execution and establish a stable measurement/compute budget protocol.", "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
