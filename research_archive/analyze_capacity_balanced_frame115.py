#!/usr/bin/env python3
"""Analyze the preregistered late-frame capacity-balanced age-quota replay."""
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/capacity_balanced_frame115"
SEEDS = tuple(range(42, 50))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantile(values, q):
    xs = sorted(values)
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] + (xs[hi] - xs[lo]) * (p - lo)


def bootstrap_ci(values, seed=77115, n=50000):
    rng = random.Random(seed)
    draws = [sum(values[rng.randrange(len(values))] for _ in values) / len(values)
             for _ in range(n)]
    return [quantile(draws, .025), quantile(draws, .975)]


def sign_test_p(values):
    nz = [v for v in values if v != 0]
    if not nz:
        return 1.0
    n = len(nz)
    k = min(sum(v > 0 for v in nz), sum(v < 0 for v in nz))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def main():
    prereg_path = BASE / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    record_hashes = {}
    records = []
    for seed in SEEDS:
        path = BASE / "replica_office1" / f"seed{seed}_frame115.json"
        rec = json.loads(path.read_text())
        if (rec["scene"], rec["seed"], rec["snapshot_frame"]) != ("replica_office1", seed, 115):
            raise ValueError(f"Record identity mismatch: {path}")
        if not rec["all_count_and_snapshot_gates_passed"]:
            raise ValueError(f"Preregistered gate failed for seed {seed}")
        arms = {a["action"]["name"]: a for a in rec["arms"]}
        expected = {
            "spawn0_mature96": (4800, 200, 0),
            "spawn128_mature96": (4800, 200, 128),
            "spawn0_mature100": (5000, 0, 0),
            "spawn128_mature100": (5000, 0, 128),
        }
        for name, (mature, young, spawned) in expected.items():
            a = arms[name]
            if (a["realized_mature_micro_updates"], a["realized_young_micro_updates"], a["n_spawned"]) != (mature, young, spawned):
                raise ValueError(f"Quota/spawn mismatch in seed {seed}, arm {name}")
        record_hashes[str(path.relative_to(ROOT))] = sha256(path)
        records.append((rec, arms))

    per_seed = []
    for rec, arms in records:
        a96, a100 = arms["spawn0_mature96"], arms["spawn0_mature100"]
        b96, b100 = arms["spawn128_mature96"], arms["spawn128_mature100"]
        per_seed.append({
            "seed": rec["seed"],
            "primary_psnr_db_100_minus_96": a100["final_map_eval_psnr"] - a96["final_map_eval_psnr"],
            "primary_depth_l1_m_100_minus_96": a100["final_map_eval_depth_l1"] - a96["final_map_eval_depth_l1"],
            "primary_sync_opt_ms_100_minus_96": a100["opt_stage_sync_ms"] - a96["opt_stage_sync_ms"],
            "psnr_spawn128_db_100_minus_96": b100["final_map_eval_psnr"] - b96["final_map_eval_psnr"],
            "quota_by_spawn_interaction_db": (b100["final_map_eval_psnr"] - b96["final_map_eval_psnr"])
                - (a100["final_map_eval_psnr"] - a96["final_map_eval_psnr"]),
            "pre_psnr": rec["pre_action_snapshot_eval"]["psnr"],
            "snapshot_sha256": rec["base_snapshot_hash"],
            "gaussian_count_spawn0": a96["n_gaussians_after_action"],
            "gaussian_count_spawn128": b96["n_gaussians_after_action"],
        })

    effects = [x["primary_psnr_db_100_minus_96"] for x in per_seed]
    depth = [x["primary_depth_l1_m_100_minus_96"] for x in per_seed]
    time = [x["primary_sync_opt_ms_100_minus_96"] for x in per_seed]
    interaction = [x["quota_by_spawn_interaction_db"] for x in per_seed]
    result = {
        "status": "all preregistered validity gates passed",
        "preregistration": prereg,
        "valid_snapshots": len(records),
        "primary": {
            "estimand": "zero-spawn final-map PSNR, 100% mature minus 96% mature; 200 of 5000 updates move from young to mature",
            "mean_psnr_effect_db": sum(effects) / len(effects),
            "seed_bootstrap_95_ci_db": bootstrap_ci(effects),
            "positive_seeds": sum(x > 0 for x in effects),
            "exact_two_sided_sign_test_p": sign_test_p(effects),
            "per_seed_effect_db": {str(x["seed"]): x["primary_psnr_db_100_minus_96"] for x in per_seed},
        },
        "secondary": {
            "mean_depth_l1_change_m": sum(depth) / len(depth),
            "depth_l1_seed_bootstrap_95_ci_m": bootstrap_ci(depth, 77116),
            "mean_sync_opt_time_change_ms": sum(time) / len(time),
            "sync_opt_time_seed_bootstrap_95_ci_ms": bootstrap_ci(time, 77117),
            "mean_quota_by_spawn_interaction_db": sum(interaction) / len(interaction),
            "interaction_seed_bootstrap_95_ci_db": bootstrap_ci(interaction, 77118),
        },
        "seed_records": per_seed,
        "provenance": {
            "manifest_sha256": sha256(BASE / "manifest.json"),
            "preregistration_sha256": sha256(prereg_path),
            "runner_sha256": sha256(ROOT / "experiments/run_joint_spawn_refine_replay.py"),
            "analyzer_sha256": sha256(Path(__file__)),
            "record_sha256": record_hashes,
        },
        "interpretation_limit": prereg["interpretation_limit"],
    }
    (BASE / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    ci = result["primary"]["seed_bootstrap_95_ci_db"]
    tci = result["secondary"]["sync_opt_time_seed_bootstrap_95_ci_ms"]
    lines = [
        "# Capacity-balanced late-frame age-allocation replay",
        "",
        "Eight same-snapshot seed replays at Replica office1 frame 115. Every arm performed exactly 5,000 Gaussian micro-updates. The primary contrast moves 200 updates between age groups while preserving total update count and map cardinality.",
        "",
        "| Valid snapshots | PSNR change, 100% vs 96% mature (dB; 95% seed CI) | Positive seeds | Exact sign p | Depth-L1 change (m) | Sync optimization-time change (ms; 95% seed CI) |",
        "|--:|--:|--:|--:|--:|--:|",
        f"| {len(records)}/8 | {sum(effects)/len(effects):+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {sum(x>0 for x in effects)}/8 | {sign_test_p(effects):.4f} | {sum(depth)/len(depth):+.5f} | {sum(time)/len(time):+.3f} [{tci[0]:+.3f}, {tci[1]:+.3f}] |",
        "",
        "## Checks and interpretation",
        "",
        "All eight seed snapshots passed the preregistered hash, render, spawn, quota, and map-count gates. The 96%-mature arms realized exactly 4,800 mature and 200 young updates; 100%-mature arms realized 5,000 and zero. Within each spawn dose, final Gaussian count matched between quota arms.",
        "",
        "The estimate is an immediate one-frame effect of reallocating only 200 updates from young to mature at one late snapshot in a known regression scene. It tests feasible quota compliance; it does not show that a continuously applied quota prevents long-horizon regression. The update count is fixed, while measured optimization wall time may differ.",
        "",
        "The spawn-128 interaction and per-seed effects are in `analysis.json`.",
        "",
    ]
    (BASE / "report.md").write_text("\n".join(lines))
    print(BASE / "report.md")


if __name__ == "__main__":
    main()
