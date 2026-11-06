#!/usr/bin/env python3
"""Analyze early/mid capacity-balanced age-allocation replays by seed cluster."""
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/capacity_balanced_early_mid"
FRAMES = (45, 70)
SEEDS = tuple(range(42, 50))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantile(xs, q):
    xs = sorted(xs)
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] + (xs[hi] - xs[lo]) * (p - lo)


def bootstrap_ci(xs, seed, n=50000):
    rng = random.Random(seed)
    draws = [sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n)]
    return [quantile(draws, .025), quantile(draws, .975)]


def sign_test_p(xs):
    xs = [x for x in xs if x != 0]
    n = len(xs)
    if not n:
        return 1.0
    k = min(sum(x > 0 for x in xs), sum(x < 0 for x in xs))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def main():
    prereg_path = BASE / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    records, hashes = {}, {}
    for frame in FRAMES:
        for seed in SEEDS:
            path = BASE / "replica_office1" / f"seed{seed}_frame{frame}.json"
            rec = json.loads(path.read_text())
            if (rec["scene"], rec["seed"], rec["snapshot_frame"]) != ("replica_office1", seed, frame):
                raise ValueError(f"Record identity mismatch: {path}")
            if not rec["all_count_and_snapshot_gates_passed"]:
                raise ValueError(f"Validity gate failed: {path}")
            arms = {a["action"]["name"]: a for a in rec["arms"]}
            targets = {"spawn0_mature96": (4800, 200, 0), "spawn0_mature100": (5000, 0, 0),
                       "spawn128_mature96": (4800, 200, 128), "spawn128_mature100": (5000, 0, 128)}
            for name, (mature, young, spawned) in targets.items():
                a = arms[name]
                if (a["realized_mature_micro_updates"], a["realized_young_micro_updates"], a["n_spawned"]) != (mature, young, spawned):
                    raise ValueError(f"Quota/spawn mismatch: {path} {name}")
            records[(seed, frame)] = (rec, arms)
            hashes[str(path.relative_to(ROOT))] = sha256(path)

    per_seed = []
    frame_summary = []
    for frame in FRAMES:
        d_psnr, d_depth, d_time = [], [], []
        for seed in SEEDS:
            _, arms = records[(seed, frame)]
            a, b = arms["spawn0_mature96"], arms["spawn0_mature100"]
            d_psnr.append(b["final_map_eval_psnr"] - a["final_map_eval_psnr"])
            d_depth.append(b["final_map_eval_depth_l1"] - a["final_map_eval_depth_l1"])
            d_time.append(b["opt_stage_sync_ms"] - a["opt_stage_sync_ms"])
        frame_summary.append({
            "frame": frame, "valid_snapshots": 8,
            "mean_psnr_effect_db": sum(d_psnr) / 8,
            "seed_bootstrap_95_ci_db": bootstrap_ci(d_psnr, 55000 + frame),
            "positive_seeds": sum(x > 0 for x in d_psnr),
            "mean_depth_l1_change_m": sum(d_depth) / 8,
            "mean_sync_optimization_time_change_ms": sum(d_time) / 8,
        })
    for seed in SEEDS:
        per_frame = []
        for frame in FRAMES:
            rec, arms = records[(seed, frame)]
            a, b = arms["spawn0_mature96"], arms["spawn0_mature100"]
            c, d = arms["spawn128_mature96"], arms["spawn128_mature100"]
            per_frame.append({
                "frame": frame,
                "psnr_effect_db": b["final_map_eval_psnr"] - a["final_map_eval_psnr"],
                "depth_l1_change_m": b["final_map_eval_depth_l1"] - a["final_map_eval_depth_l1"],
                "sync_opt_time_change_ms": b["opt_stage_sync_ms"] - a["opt_stage_sync_ms"],
                "quota_by_spawn_interaction_db": (d["final_map_eval_psnr"] - c["final_map_eval_psnr"])
                    - (b["final_map_eval_psnr"] - a["final_map_eval_psnr"]),
                "snapshot_sha256": rec["base_snapshot_hash"],
            })
        per_seed.append({
            "seed": seed,
            "mean_psnr_effect_db": sum(x["psnr_effect_db"] for x in per_frame) / len(FRAMES),
            "mean_depth_l1_change_m": sum(x["depth_l1_change_m"] for x in per_frame) / len(FRAMES),
            "mean_sync_opt_time_change_ms": sum(x["sync_opt_time_change_ms"] for x in per_frame) / len(FRAMES),
            "mean_quota_by_spawn_interaction_db": sum(x["quota_by_spawn_interaction_db"] for x in per_frame) / len(FRAMES),
            "frames": per_frame,
        })

    cluster_effects = [x["mean_psnr_effect_db"] for x in per_seed]
    cluster_depth = [x["mean_depth_l1_change_m"] for x in per_seed]
    cluster_time = [x["mean_sync_opt_time_change_ms"] for x in per_seed]
    cluster_interaction = [x["mean_quota_by_spawn_interaction_db"] for x in per_seed]
    result = {
        "status": "all preregistered gates passed",
        "preregistration": prereg,
        "valid_snapshot_records": 16,
        "primary": {
            "estimand": "Per-seed equal-weight mean across frames 45 and 70 of zero-spawn PSNR (100% mature minus 96% mature)",
            "mean_effect_db": sum(cluster_effects) / 8,
            "seed_cluster_bootstrap_95_ci_db": bootstrap_ci(cluster_effects, 55670),
            "positive_seed_clusters": sum(x > 0 for x in cluster_effects),
            "n_seed_clusters": 8,
            "exact_two_sided_sign_test_p": sign_test_p(cluster_effects),
            "seed_cluster_effects_db": {str(x["seed"]): x["mean_psnr_effect_db"] for x in per_seed},
        },
        "per_frame_descriptive": frame_summary,
        "secondary": {
            "mean_depth_l1_change_m": sum(cluster_depth) / 8,
            "depth_l1_seed_cluster_bootstrap_95_ci_m": bootstrap_ci(cluster_depth, 55671),
            "mean_sync_opt_time_change_ms": sum(cluster_time) / 8,
            "sync_opt_time_seed_cluster_bootstrap_95_ci_ms": bootstrap_ci(cluster_time, 55672),
            "mean_quota_by_spawn_interaction_db": sum(cluster_interaction) / 8,
            "interaction_seed_cluster_bootstrap_95_ci_db": bootstrap_ci(cluster_interaction, 55673),
        },
        "seed_records": per_seed,
        "provenance": {
            "manifest_sha256": sha256(BASE / "manifest.json"),
            "preregistration_sha256": sha256(prereg_path),
            "runner_sha256": sha256(ROOT / "experiments/run_joint_spawn_refine_replay.py"),
            "analyzer_sha256": sha256(Path(__file__)),
            "record_sha256": hashes,
        },
        "interpretation_limit": prereg["interpretation_limit"],
    }
    (BASE / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    primary, secondary = result["primary"], result["secondary"]
    ci, tci = primary["seed_cluster_bootstrap_95_ci_db"], secondary["sync_opt_time_seed_cluster_bootstrap_95_ci_ms"]
    lines = [
        "# Early/mid capacity-balanced age-allocation replay",
        "",
        "Eight seed clusters, each averaging frame-45 and frame-70 same-snapshot interventions. Every arm used exactly 5,000 Gaussian micro-updates; the primary contrast moves 200 updates from young to mature while holding map size and total update count fixed.",
        "",
        "| Valid snapshots | Primary PSNR effect (dB; 95% seed-cluster CI) | Positive clusters | Exact sign p | Depth-L1 change (m) | Sync optimization-time change (ms; 95% CI) |",
        "|--:|--:|--:|--:|--:|--:|",
        f"| 16/16 | {primary['mean_effect_db']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {primary['positive_seed_clusters']}/8 | {primary['exact_two_sided_sign_test_p']:.4f} | {secondary['mean_depth_l1_change_m']:+.5f} | {secondary['mean_sync_opt_time_change_ms']:+.3f} [{tci[0]:+.3f}, {tci[1]:+.3f}] |",
        "",
        "## By snapshot",
        "",
        "| Frame | Valid | Mean PSNR effect (dB; seed CI) | Positive seeds | Mean depth-L1 change (m) | Mean sync-time change (ms) |",
        "|--:|--:|--:|--:|--:|--:|",
    ]
    for row in frame_summary:
        c = row["seed_bootstrap_95_ci_db"]
        lines.append(f"| {row['frame']} | 8/8 | {row['mean_psnr_effect_db']:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}] | {row['positive_seeds']}/8 | {row['mean_depth_l1_change_m']:+.5f} | {row['mean_sync_optimization_time_change_ms']:+.3f} |")
    lines += [
        "",
        "All quota, snapshot, candidate, and map-count gates passed. The primary analysis treats seed as the independent cluster; frame-level values are descriptive.",
        "",
        "This evaluates immediate local effects at two snapshots in one known regression scene. It is not a continuously applied policy or evidence of held-out generalization. Frame-115 results from the separate capacity-matched replay are reported separately, not pooled into this estimate.",
        "",
        "Per-seed records, provenance hashes, and spawn interaction are in `analysis.json`.",
        "",
    ]
    (BASE / "report.md").write_text("\n".join(lines))
    print(BASE / "report.md")


if __name__ == "__main__":
    main()
