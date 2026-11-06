#!/usr/bin/env python3
"""Analyze cross-scene multi-seed spawn-128 matched-time quota replication."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/matched_time_spawn128_cross_scene_replication_v1"
SCENES = {
    "replica_office1": (BASE, (260, 261, 262, 263)),
    "tum_fr1_rpy": (BASE / "tum_seeds264_267", (264, 265, 266, 267)),
}
OUT = BASE
TOLERANCE = 0.05
REPEATS = 5


def mean_sd_cv(xs):
    m = statistics.mean(xs)
    sd = statistics.stdev(xs)
    return {"mean": m, "sd": sd, "cv_fraction": sd / m}


def analyze_record(path, scene, seed):
    rec = json.loads(path.read_text())
    if (rec["scene"], rec["seed"], rec["snapshot_frame"]) != (scene, seed, 90):
        raise RuntimeError(f"Wrong record identity: {path}")
    if not rec["all_count_and_snapshot_gates_passed"]:
        raise RuntimeError(f"Map/spawn/update validity gate failed: {path}")
    arms = {x["action"]["name"]: x for x in rec["arms"]}
    stats = {}
    for pct, updates in ((25, 5000), (75, 4374)):
        rows = [arms[f"spawn128_mature{pct}_repeat{i}"] for i in range(REPEATS)]
        if any(not row["valid"] or row["n_micro_updates"] != updates or row["n_spawned"] != 128 for row in rows):
            raise RuntimeError(f"Exact spawn/quota gate failed: {path}, quota={pct}")
        stats[pct] = {
            "cuda_event_ms": mean_sd_cv([r["opt_stage_cuda_event_ms"] for r in rows]),
            "sync_wall_ms": mean_sd_cv([r["opt_stage_sync_ms"] for r in rows]),
            "frame_wall_ms": mean_sd_cv([r["frame_time_ms"] for r in rows]),
            "psnr": mean_sd_cv([r["final_map_eval_psnr"] for r in rows]),
            "depth_l1_m": mean_sd_cv([r["final_map_eval_depth_l1"] for r in rows]),
        }
    lo, hi = stats[25], stats[75]
    event_gap = abs(hi["cuda_event_ms"]["mean"] - lo["cuda_event_ms"]["mean"]) / statistics.mean(
        [hi["cuda_event_ms"]["mean"], lo["cuda_event_ms"]["mean"]])
    wall_gap = abs(hi["sync_wall_ms"]["mean"] - lo["sync_wall_ms"]["mean"]) / statistics.mean(
        [hi["sync_wall_ms"]["mean"], lo["sync_wall_ms"]["mean"]])
    cv_pass = all(stats[p]["cuda_event_ms"]["cv_fraction"] <= TOLERANCE and
                  stats[p]["sync_wall_ms"]["cv_fraction"] <= TOLERANCE for p in (25, 75))
    time_pass = event_gap <= TOLERANCE and wall_gap <= TOLERANCE
    return {
        "scene": scene, "seed": seed, "snapshot_hash": rec["base_snapshot_hash"],
        "all_count_and_snapshot_gates_passed": True,
        "per_quota": stats,
        "cuda_event_mean_gap_fraction": event_gap,
        "sync_wall_mean_gap_fraction": wall_gap,
        "within_arm_cv_gate": cv_pass,
        "between_quota_time_gate": time_pass,
        "matched_time_snapshot_gate": cv_pass and time_pass,
        "psnr_75_minus_25_db": hi["psnr"]["mean"] - lo["psnr"]["mean"],
        "depth_l1_75_minus_25_m": hi["depth_l1_m"]["mean"] - lo["depth_l1_m"]["mean"],
    }


def main():
    scene_results = {}
    all_records = []
    for scene, (folder, seeds) in SCENES.items():
        rows = []
        for seed in seeds:
            row = analyze_record(folder / scene / f"seed{seed}_frame90.json", scene, seed)
            rows.append(row)
            all_records.append(row)
        valid = [x for x in rows if x["matched_time_snapshot_gate"]]
        scene_results[scene] = {
            "n_seed_snapshots": len(rows),
            "n_valid_mapping_quota_snapshots": sum(x["all_count_and_snapshot_gates_passed"] for x in rows),
            "n_matched_time_snapshots": len(valid),
            "per_seed": rows,
            "matched_time_effect_summary": ({
                "n": len(valid),
                "mean_psnr_75_minus_25_db": statistics.mean(x["psnr_75_minus_25_db"] for x in valid),
                "sd_across_seed_snapshots_db": statistics.stdev(x["psnr_75_minus_25_db"] for x in valid) if len(valid) > 1 else None,
                "mean_depth_l1_75_minus_25_m": statistics.mean(x["depth_l1_75_minus_25_m"] for x in valid),
                "sd_depth_across_seed_snapshots_m": statistics.stdev(x["depth_l1_75_minus_25_m"] for x in valid) if len(valid) > 1 else None,
                "positive_psnr_seeds": sum(x["psnr_75_minus_25_db"] > 0 for x in valid),
                "negative_depth_l1_seeds": sum(x["depth_l1_75_minus_25_m"] < 0 for x in valid),
            } if valid else None),
            "descriptive_all_seed_mean_psnr_db": statistics.mean(x["psnr_75_minus_25_db"] for x in rows),
            "descriptive_all_seed_mean_depth_l1_m": statistics.mean(x["depth_l1_75_minus_25_m"] for x in rows),
        }
    out_json = OUT / "analysis.json"
    out_json.write_text(json.dumps({
        "status": "exploratory replication; two development scenes; no cross-scene pooled inference",
        "timing_gate": "within-arm CUDA event and sync wall CV <=5%; between-quota means <=5% for both endpoints",
        "n_repeats_per_arm": REPEATS,
        "scenes": scene_results,
        "protocol_deviation": "The first combined invocation also ran TUM seeds 260-263 because the runner applies the seed list to every scene. Those TUM records are excluded; corrected TUM seeds 264-267 were run separately with their own manifest.",
    }, indent=2) + "\n")

    lines = [
        "# Matched-time spawn-128 quota replication", "",
        "Four independent seed snapshots per scene; five same-snapshot branches per arm. Each arm spawned 128 Gaussians. The 25%-mature arm used 5,000 updates; the 75%-mature arm used 4,374 updates. Every snapshot passed exact map/spawn/quota checks. The matched-time gate additionally requires each arm's CUDA-event and synchronized wall CV ≤5% and both between-quota mean time gaps ≤5%.", "",
        "| Scene | Seed | Event times 25%/75% ms | Event gap | Event CV 25%/75% | Wall gap | Time gate | PSNR Δ dB | Depth-L1 Δ m |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for scene, data in scene_results.items():
        for x in data["per_seed"]:
            a, b = x["per_quota"][25], x["per_quota"][75]
            lines.append(
                f"| {scene} | {x['seed']} | {a['cuda_event_ms']['mean']:.2f}/{b['cuda_event_ms']['mean']:.2f} | "
                f"{x['cuda_event_mean_gap_fraction']*100:.2f}% | "
                f"{a['cuda_event_ms']['cv_fraction']*100:.2f}%/{b['cuda_event_ms']['cv_fraction']*100:.2f}% | "
                f"{x['sync_wall_mean_gap_fraction']*100:.2f}% | "
                f"{'PASS' if x['matched_time_snapshot_gate'] else 'FAIL'} | "
                f"{x['psnr_75_minus_25_db']:+.5f} | {x['depth_l1_75_minus_25_m']:+.6f} |")
        s = data["matched_time_effect_summary"]
        if s:
            lines += ["", f"**{scene} matched-time subset:** {s['n']}/4 snapshots; mean PSNR difference {s['mean_psnr_75_minus_25_db']:+.5f} dB (seed SD {s['sd_across_seed_snapshots_db']:.5f}); mean Depth-L1 difference {s['mean_depth_l1_75_minus_25_m']:+.6f} m; positive PSNR in {s['positive_psnr_seeds']}/{s['n']} and lower depth L1 in {s['negative_depth_l1_seeds']}/{s['n']}.", ""]
        else:
            lines += ["", f"**{scene}:** 0/4 snapshots passed the matched-time gate; no compute-matched quality estimate is reported. Descriptive all-seed mean PSNR difference {data['descriptive_all_seed_mean_psnr_db']:+.5f} dB and depth-L1 difference {data['descriptive_all_seed_mean_depth_l1_m']:+.6f} m are not treated as matched-compute results.", ""]
    lines += [
        "## Interpretation and limits", "",
        "All eight snapshots passed mapping, spawn, and exact update-quota checks. The timing gate passed in 3/4 TUM snapshots and 0/4 Replica snapshots. Thus the calibrated 4,374-update quota-75 dose transferred inconsistently: it was time-matched for most TUM seeds but not Replica office1. The three passing TUM snapshots show a positive fixed-view PSNR contrast; the sample is small and the scene was used during development. Both scenes are development scenes, not held-out generalization tests. The initially mis-seeded TUM records 260–263 are explicitly excluded.", "",
        "No pooled cross-scene estimate is reported. A 15 ms configuration remains a modeled scheduler budget; all timing values here are measured synchronized optimizer-stage or full-frame time. See per-seed details and manifests in the raw result directories.", "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
