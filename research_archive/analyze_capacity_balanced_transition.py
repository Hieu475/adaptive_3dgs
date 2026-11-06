#!/usr/bin/env python3
"""Analyze the preregistered frame-90/100 capacity-balanced replay."""
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/capacity_balanced_transition"
FRAMES, SEEDS = (90, 100), tuple(range(42, 50))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantile(xs, q):
    xs = sorted(xs); p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] + (xs[hi] - xs[lo]) * (p - lo)


def bootstrap_ci(xs, seed, n=50000):
    rng = random.Random(seed)
    vals = [sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n)]
    return [quantile(vals, .025), quantile(vals, .975)]


def sign_test_p(xs):
    xs = [x for x in xs if x != 0]
    if not xs: return 1.0
    n = len(xs); k = min(sum(x > 0 for x in xs), sum(x < 0 for x in xs))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    prereg_path = BASE / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    records, hashes = {}, {}
    for f in FRAMES:
        for seed in SEEDS:
            p = BASE / "replica_office1" / f"seed{seed}_frame{f}.json"
            r = json.loads(p.read_text())
            if (r["scene"], r["seed"], r["snapshot_frame"]) != ("replica_office1", seed, f):
                raise ValueError(f"Record identity mismatch: {p}")
            if not r["all_count_and_snapshot_gates_passed"]:
                raise ValueError(f"Gate failed: {p}")
            arms = {x["action"]["name"]: x for x in r["arms"]}
            for name, expected in {"spawn0_mature96": (4800, 200, 0), "spawn0_mature100": (5000, 0, 0),
                                   "spawn128_mature96": (4800, 200, 128), "spawn128_mature100": (5000, 0, 128)}.items():
                a = arms[name]
                actual = (a["realized_mature_micro_updates"], a["realized_young_micro_updates"], a["n_spawned"])
                if actual != expected: raise ValueError(f"Quota/spawn mismatch: {p} {name} {actual}")
            records[(seed, f)] = (r, arms)
            hashes[str(p.relative_to(ROOT))] = sha256(p)

    per_frame, seeds = [], []
    for f in FRAMES:
        effects, depths, times = [], [], []
        for seed in SEEDS:
            _, a = records[(seed, f)]
            x, y = a["spawn0_mature96"], a["spawn0_mature100"]
            effects.append(y["final_map_eval_psnr"] - x["final_map_eval_psnr"])
            depths.append(y["final_map_eval_depth_l1"] - x["final_map_eval_depth_l1"])
            times.append(y["opt_stage_sync_ms"] - x["opt_stage_sync_ms"])
        per_frame.append({"frame": f, "valid_snapshots": 8,
            "mean_psnr_effect_db": sum(effects)/8, "seed_bootstrap_95_ci_db": bootstrap_ci(effects, 99000+f),
            "positive_seeds": sum(x > 0 for x in effects), "mean_depth_l1_change_m": sum(depths)/8,
            "mean_sync_optimization_time_change_ms": sum(times)/8})
    for seed in SEEDS:
        fs, ds, ts, ints = [], [], [], []
        for f in FRAMES:
            _, a = records[(seed, f)]
            x, y = a["spawn0_mature96"], a["spawn0_mature100"]
            z, w = a["spawn128_mature96"], a["spawn128_mature100"]
            fs.append(y["final_map_eval_psnr"] - x["final_map_eval_psnr"])
            ds.append(y["final_map_eval_depth_l1"] - x["final_map_eval_depth_l1"])
            ts.append(y["opt_stage_sync_ms"] - x["opt_stage_sync_ms"])
            ints.append((w["final_map_eval_psnr"]-z["final_map_eval_psnr"])-fs[-1])
        seeds.append({"seed": seed, "mean_psnr_effect_db": sum(fs)/2,
                      "mean_depth_l1_change_m": sum(ds)/2, "mean_sync_opt_time_change_ms": sum(ts)/2,
                      "mean_quota_by_spawn_interaction_db": sum(ints)/2})

    es = [x["mean_psnr_effect_db"] for x in seeds]
    ds = [x["mean_depth_l1_change_m"] for x in seeds]
    ts = [x["mean_sync_opt_time_change_ms"] for x in seeds]
    it = [x["mean_quota_by_spawn_interaction_db"] for x in seeds]
    result = {"status": "all preregistered gates passed", "preregistration": prereg,
        "valid_snapshot_records": 16,
        "primary": {"estimand": "Equal-weight per-seed mean over frames 90/100 of zero-spawn PSNR (100% mature minus 96% mature)",
            "mean_effect_db": sum(es)/8, "seed_cluster_bootstrap_95_ci_db": bootstrap_ci(es,99100),
            "positive_seed_clusters": sum(x>0 for x in es), "n_seed_clusters": 8,
            "exact_two_sided_sign_test_p": sign_test_p(es),
            "seed_cluster_effects_db": {str(x['seed']):x['mean_psnr_effect_db'] for x in seeds}},
        "per_frame_descriptive": per_frame,
        "secondary": {"mean_depth_l1_change_m": sum(ds)/8, "depth_l1_seed_cluster_bootstrap_95_ci_m": bootstrap_ci(ds,99101),
            "mean_sync_opt_time_change_ms": sum(ts)/8, "sync_opt_time_seed_cluster_bootstrap_95_ci_ms": bootstrap_ci(ts,99102),
            "mean_quota_by_spawn_interaction_db": sum(it)/8, "interaction_seed_cluster_bootstrap_95_ci_db": bootstrap_ci(it,99103)},
        "seed_records": seeds,
        "provenance": {"manifest_sha256": sha256(BASE/"manifest.json"), "preregistration_sha256": sha256(prereg_path),
            "runner_sha256": sha256(ROOT/"experiments/run_joint_spawn_refine_replay.py"),
            "analyzer_sha256": sha256(Path(__file__)), "record_sha256": hashes},
        "interpretation_limit": prereg["interpretation_limit"]}
    (BASE/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    c=result["primary"]["seed_cluster_bootstrap_95_ci_db"]
    t=result["secondary"]["sync_opt_time_seed_cluster_bootstrap_95_ci_ms"]
    lines=["# Transition-window capacity-balanced age-allocation replay","",
        "Eight seed clusters; each averages two same-snapshot interventions at frames 90 and 100. The primary contrast moves 200 of 5,000 updates from young to mature (96% to 100%) at zero spawn.","",
        "| Valid snapshots | Primary PSNR effect (dB; 95% seed-cluster CI) | Positive clusters | Exact sign p | Depth-L1 change (m) | Sync optimization-time change (ms; 95% CI) |",
        "|--:|--:|--:|--:|--:|--:|",
        f"| 16/16 | {sum(es)/8:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}] | {sum(x>0 for x in es)}/8 | {sign_test_p(es):.4f} | {sum(ds)/8:+.5f} | {sum(ts)/8:+.3f} [{t[0]:+.3f}, {t[1]:+.3f}] |","",
        "## By snapshot","","| Frame | Valid | Mean PSNR effect (dB; seed CI) | Positive seeds | Mean depth-L1 change (m) | Mean sync-time change (ms) |","|--:|--:|--:|--:|--:|--:|"]
    for r in per_frame:
        c=r["seed_bootstrap_95_ci_db"]
        lines.append(f"| {r['frame']} | 8/8 | {r['mean_psnr_effect_db']:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}] | {r['positive_seeds']}/8 | {r['mean_depth_l1_change_m']:+.5f} | {r['mean_sync_optimization_time_change_ms']:+.3f} |")
    lines += ["","All snapshot, quota, candidate-pool, spawn-count, and map-cardinality gates passed. Seed is the independent cluster; per-frame results are descriptive.","",
        "This estimates immediate local effects near the observed stage-dependent sign change. It is not a continuously applied policy test. Other preregistered snapshot groups remain separate and are not pooled into this primary estimate.","",
        "Per-seed values and provenance hashes are in `analysis.json`.",""]
    (BASE/"report.md").write_text("\n".join(lines))
    print(BASE/"report.md")


if __name__ == "__main__": main()
