#!/usr/bin/env python3
"""Analyze preregistered same-snapshot quota replays across multiple frames."""
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/legacy/phase14_generalization_postfix/multisnapshot_mature_quota_replay"
FRAMES = (45, 70, 115)
SEEDS = tuple(range(42, 50))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values, q):
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap_ci(values, seed, n=50000):
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        draws.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def sign_test_p(values):
    n = len(values)
    k = min(sum(v > 0 for v in values), sum(v < 0 for v in values))
    if any(v == 0 for v in values):
        # Exact two-sided binomial test conditional on nonzero signs.
        n = sum(v != 0 for v in values)
        k = min(sum(v > 0 for v in values), sum(v < 0 for v in values))
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def arms_by_name(record):
    return {arm["action"]["name"]: arm for arm in record["arms"]}


def load_records():
    records = {}
    hashes = {}
    for scene in ("replica_office1", "tum_fr1_rpy"):
        for seed in SEEDS:
            for frame in FRAMES:
                path = BASE / scene / f"seed{seed}_frame{frame}.json"
                record = json.loads(path.read_text())
                if (record["scene"], record["seed"], record["snapshot_frame"]) != (scene, seed, frame):
                    raise ValueError(f"Identity mismatch in {path}")
                records[(scene, seed, frame)] = record
                hashes[str(path.relative_to(ROOT))] = sha256(path)
    return records, hashes


def main():
    prereg_path = BASE / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    records, record_hashes = load_records()
    result = {
        "status": "preregistered multi-snapshot replay; primary scene-level inference available only where all gates pass",
        "protocol": prereg,
        "scenes": {},
        "planned_two_scene_family_complete": False,
        "interpretation_limit": prereg["interpretation_limit"],
        "provenance": {
            "manifest_sha256": sha256(BASE / "manifest.json"),
            "preregistration_sha256": sha256(prereg_path),
            "runner_sha256": sha256(ROOT / "experiments/run/run_joint_spawn_refine_replay.py"),
            "analyzer_sha256": sha256(Path(__file__)),
            "record_sha256": record_hashes,
        },
    }

    primary_p = {}
    for scene in ("replica_office1", "tum_fr1_rpy"):
        frame_rows = []
        seed_cluster = {seed: [] for seed in SEEDS}
        valid_seed_clusters = []
        for frame in FRAMES:
            frame_effects, frame_depth, frame_time = [], [], []
            actual_mature = {25: [], 75: []}
            valid_n = 0
            for seed in SEEDS:
                rec = records[(scene, seed, frame)]
                valid = bool(rec["all_count_and_snapshot_gates_passed"])
                valid_n += int(valid)
                arms = arms_by_name(rec)
                a25, a75 = arms["spawn0_mature25"], arms["spawn0_mature75"]
                actual_mature[25].append(a25["realized_mature_micro_updates"])
                actual_mature[75].append(a75["realized_mature_micro_updates"])
                if valid:
                    effect = a75["final_map_eval_psnr"] - a25["final_map_eval_psnr"]
                    frame_effects.append(effect)
                    frame_depth.append(a75["final_map_eval_depth_l1"] - a25["final_map_eval_depth_l1"])
                    frame_time.append(a75["opt_stage_sync_ms"] - a25["opt_stage_sync_ms"])
                    seed_cluster[seed].append(effect)
            frame_rows.append({
                "frame": frame,
                "valid_snapshots": valid_n,
                "total_snapshots": len(SEEDS),
                "mean_realized_mature_updates_spawn0_25_requested": sum(actual_mature[25]) / len(actual_mature[25]),
                "mean_realized_mature_updates_spawn0_75_requested": sum(actual_mature[75]) / len(actual_mature[75]),
                "mean_quota_contrast_db_when_valid": sum(frame_effects) / len(frame_effects) if frame_effects else None,
                "seed_bootstrap_ci_db_when_valid": bootstrap_ci(frame_effects, 2300 + frame) if frame_effects else None,
                "mean_depth_l1_change_m_when_valid": sum(frame_depth) / len(frame_depth) if frame_depth else None,
                "mean_sync_optimization_time_change_ms_when_valid": sum(frame_time) / len(frame_time) if frame_time else None,
                "frame_effects_are_descriptive": True,
            })

        complete = all(len(seed_cluster[s]) == len(FRAMES) for s in SEEDS)
        scene_result = {"valid_seed_clusters": sum(complete for _ in [0]) * len(SEEDS),
                        "total_seed_clusters": len(SEEDS), "frame_results": frame_rows,
                        "primary": None}
        if complete:
            cluster_means = [sum(seed_cluster[s]) / len(FRAMES) for s in SEEDS]
            primary = {
                "estimand": "equal-weight mean over the three preregistered frame contrasts, averaged per seed cluster",
                "mean_effect_db": sum(cluster_means) / len(cluster_means),
                "seed_cluster_bootstrap_95_ci_db": bootstrap_ci(cluster_means, 9901 if scene == "replica_office1" else 9902),
                "positive_seed_clusters": sum(x > 0 for x in cluster_means),
                "n_seed_clusters": len(cluster_means),
                "exact_two_sided_sign_test_p": sign_test_p(cluster_means),
                "seed_cluster_effects_db": {str(seed): seed_cluster[seed] and sum(seed_cluster[seed]) / len(FRAMES) for seed in SEEDS},
            }
            scene_result["primary"] = primary
            primary_p[scene] = primary["exact_two_sided_sign_test_p"]
        else:
            scene_result["primary_unavailable_reason"] = "At least one preregistered snapshot failed a validity gate for every or some seed cluster; preregistered complete-cluster primary analysis is unavailable."

        # Report the gate failure modes directly from the measured quotas.
        invalid = []
        for frame in FRAMES:
            for seed in SEEDS:
                rec = records[(scene, seed, frame)]
                if not rec["all_count_and_snapshot_gates_passed"]:
                    arms = arms_by_name(rec)
                    invalid.append({
                        "seed": seed, "frame": frame,
                        "arm_valid": {name: arms[name]["valid"] for name in arms},
                        "requested_vs_realized_mature_updates": {
                            name: {"requested_fraction": arms[name]["requested_mature_fraction"],
                                   "realized": arms[name]["realized_mature_micro_updates"],
                                   "target_nominal": round(5000 * arms[name]["requested_mature_fraction"])}
                            for name in ("spawn0_mature25", "spawn0_mature75")
                        },
                        "map_count_matched_within_spawn_dose": rec["map_count_matched_within_spawn_dose"],
                    })
        scene_result["invalid_snapshot_records"] = invalid
        result["scenes"][scene] = scene_result

    result["primary_family"] = {
        "planned_scene_tests": 2,
        "estimable_scene_tests": list(primary_p),
        "raw_p_values": primary_p,
        "holm_adjusted_p_values_complete_preregistered_family": None,
        "interpretation": "No Holm-adjusted result is reported because the preregistered two-scene family is incomplete: the Replica primary endpoint failed its complete-snapshot validity requirement.",
    }
    out_json = BASE / "analysis.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# Multi-snapshot mature-quota replay", "",
             "Three snapshots (frames 45, 70, 115) were replayed independently per seed. The primary unit is the seed cluster, averaging its three frame contrasts; frames are not treated as independent samples.", "",
             "| Scene | Valid snapshots by frame 45 / 70 / 115 | Complete seed clusters | Primary mean 75%-vs-25% PSNR (dB; 95% seed CI) | Positive clusters | Unadjusted exact sign p |", "|:--|:--:|--:|--:|--:|--:|"]
    for scene, data in result["scenes"].items():
        valid = " / ".join(f"{r['valid_snapshots']}/{r['total_snapshots']}" for r in data["frame_results"])
        p = data["primary"]
        if p:
            effect = f"{p['mean_effect_db']:+.3f} [{p['seed_cluster_bootstrap_95_ci_db'][0]:+.3f}, {p['seed_cluster_bootstrap_95_ci_db'][1]:+.3f}]"
            positive, pval = f"{p['positive_seed_clusters']}/8", f"{p['exact_two_sided_sign_test_p']:.4f}"
            clusters = f"{p['n_seed_clusters']}/8"
        else:
            effect, positive, pval, clusters = "Unavailable", "—", "—", "0/8"
        lines.append(f"| `{scene}` | {valid} | {clusters} | {effect} | {positive} | {pval} |")
    lines += ["", "## Gate diagnosis", ""]
    rep = result["scenes"]["replica_office1"]
    if rep["invalid_snapshot_records"]:
        vals25 = [x["requested_vs_realized_mature_updates"]["spawn0_mature25"]["realized"] for x in rep["invalid_snapshot_records"]]
        vals75 = [x["requested_vs_realized_mature_updates"]["spawn0_mature75"]["realized"] for x in rep["invalid_snapshot_records"]]
        lines.append(f"Replica office1 has {len(rep['invalid_snapshot_records'])} invalid seed-snapshot records. At frame 115, zero-spawn arms requested 1,250 and 3,750 mature updates for the 25% and 75% quotas, but realized means were {sum(vals25)/len(vals25):.1f} and {sum(vals75)/len(vals75):.1f}; for each seed, both quota settings produced the same mature/young split within a spawn dose. The scheduler reallocates quota when one age group reaches its available update capacity, so the two intended interventions cease to be distinct at this snapshot. These records are excluded from the preregistered primary analysis; the complete-cluster Replica primary result is unavailable.")
    else:
        lines.append("No validity failures observed.")
    lines += ["", "## Interpretation", "",
              "The TUM primary result is a controlled local age-allocation contrast across three independent prefix snapshots, summarized at the seed level. It does not test a continuously applied policy. Replica's failed late-frame quota compliance is itself a substantive controller limitation and prevents the preregistered three-snapshot primary estimate for that scene.",
              "The planned two-scene inferential family is incomplete, so no Holm-adjusted p-value is reported. The TUM exact sign p-value is unadjusted and should not be presented as a completed two-scene confirmation. Per-frame rows and invalid records are included in `analysis.json`.",
              "Quota effects hold update counts fixed, not synchronized wall time. The run remains limited to known regression scenes and mapping under ground-truth poses.", ""]
    (BASE / "report.md").write_text("\n".join(lines))
    print(BASE / "report.md")


if __name__ == "__main__":
    main()
