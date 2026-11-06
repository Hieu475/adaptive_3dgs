#!/usr/bin/env python3
"""Analyze the preregistered same-snapshot Replica office1 quota replay."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import subprocess
import argparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/office1_mature_quota_replay"
RUNS = BASE / "replica_office1"
SEEDS = tuple(range(42, 50))
ARMS = ("spawn0_mature25", "spawn0_mature75", "spawn128_mature25", "spawn128_mature75")


def bootstrap_ci(values: list[float], seed: int = 20261017) -> list[float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(x, size=(50000, len(x)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(sampled, [0.025, 0.975])]


def exact_sign_test(values: list[float]) -> float:
    x = np.asarray(values, dtype=float)
    x = x[x != 0]
    n, k = len(x), int(np.sum(x > 0))
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n-k) + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def main() -> None:
    global BASE, RUNS
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("replica_office1", "tum_fr1_rpy"),
                        default="replica_office1")
    scene = parser.parse_args().scene
    folder = "office1_mature_quota_replay" if scene == "replica_office1" else "tum_fr1_rpy_mature_quota_replay"
    BASE = ROOT / "results/phase14_generalization_postfix" / folder
    RUNS = BASE / scene
    records = {}
    effects = {"primary_spawn0_mature75_minus_mature25_db": [],
               "mature75_minus_mature25_spawn128_db": [],
               "spawn_by_quota_interaction_db": [],
               "depth_l1_effect_spawn0_mature75_minus_mature25_m": [],
               "opt_stage_time_mature75_minus_mature25_spawn0_ms": []}
    arm_rows = {name: [] for name in ARMS}
    gate_records = []
    for seed in SEEDS:
        path = RUNS / f"seed{seed}_frame90.json"
        rec = json.loads(path.read_text())
        if rec["seed"] != seed or rec["scene"] != scene:
            raise ValueError(f"wrong unit in {path}")
        if rec["prefix_policy"] != "ours":
            raise ValueError(f"non-OURS prefix in {path}")
        arms = {a["action"]["name"]: a for a in rec["arms"]}
        if set(arms) != set(ARMS):
            raise ValueError(f"unexpected arms in {path}: {sorted(arms)}")
        count_match = all(
            arms[f"spawn{dose}_mature25"]["n_gaussians_after_action"]
            == arms[f"spawn{dose}_mature75"]["n_gaussians_after_action"]
            for dose in (0, 128)
        )
        exact_work = all(a["n_micro_updates"] == 5000 and a["valid"] for a in arms.values())
        valid = (rec["all_count_and_snapshot_gates_passed"] and
                 rec["map_count_matched_within_spawn_dose"] and count_match and exact_work)
        gate_records.append({"seed": seed, "valid": bool(valid), "same_snapshot_hash": len({a["snapshot_hash"] for a in arms.values()}) == 1,
                             "pre_action_psnr_range": rec["pre_action_reference_psnr_range"],
                             "same_count_by_spawn_dose": count_match,
                             "exact_work_and_quota": exact_work,
                             "candidate_pool_nested": rec["spawn_doses_are_nested"]})
        if not valid:
            raise RuntimeError(f"seed {seed} failed a preregistered gate")
        pre = float(rec["pre_action_snapshot_eval"]["psnr"])
        gains = {name: float(arms[name]["final_map_eval_psnr"] - pre) for name in ARMS}
        depth = {name: float(arms[name]["final_map_eval_depth_l1"]) for name in ARMS}
        effects["primary_spawn0_mature75_minus_mature25_db"].append(gains["spawn0_mature75"] - gains["spawn0_mature25"])
        effects["mature75_minus_mature25_spawn128_db"].append(gains["spawn128_mature75"] - gains["spawn128_mature25"])
        effects["spawn_by_quota_interaction_db"].append(
            (gains["spawn128_mature75"] - gains["spawn0_mature75"])
            - (gains["spawn128_mature25"] - gains["spawn0_mature25"]))
        effects["depth_l1_effect_spawn0_mature75_minus_mature25_m"].append(
            depth["spawn0_mature75"] - depth["spawn0_mature25"])
        for name in ARMS:
            arm = arms[name]
            arm_rows[name].append({
                "seed": seed,
                "final_map_psnr_gain_db": gains[name],
                "final_map_eval_psnr": float(arm["final_map_eval_psnr"]),
                "final_map_depth_l1_m": depth[name],
                "mature_update_coverage": float(arm["mature_update_coverage"]),
                "mature_micro_updates": int(arm["realized_mature_micro_updates"]),
                "young_micro_updates": int(arm["realized_young_micro_updates"]),
                "n_gaussians_after_action": int(arm["n_gaussians_after_action"]),
                "n_spawned": int(arm["n_spawned"]),
                "opt_stage_sync_ms": float(arm["opt_stage_sync_ms"]),
                "frame_time_ms": float(arm["frame_time_ms"]),
                "compute_return_db_per_opt_ms": float(arm["compute_return_db_per_opt_ms"]),
            })
        effects["opt_stage_time_mature75_minus_mature25_spawn0_ms"].append(
            float(arms["spawn0_mature75"]["opt_stage_sync_ms"]
                  - arms["spawn0_mature25"]["opt_stage_sync_ms"]))
        records[str(seed)] = {"pre_action_snapshot_eval": rec["pre_action_snapshot_eval"],
                              "base_snapshot_hash": rec["base_snapshot_hash"],
                              "candidate_pool_sha256": rec["common_candidate_pool_sha256"],
                              "gains_db": gains}

    def stats(x: list[float], sign_test: bool = False) -> dict:
        return {"mean": float(np.mean(x)), "std_seed": float(np.std(x, ddof=1)),
                "bootstrap_ci95_seed": bootstrap_ci(x),
                "positive_seeds": int(np.sum(np.asarray(x) > 0)),
                "negative_seeds": int(np.sum(np.asarray(x) < 0)),
                "exact_two_sided_sign_test_p": exact_sign_test(x) if sign_test else None,
                "per_seed": x}

    analysis = {
        "experiment": f"{scene}_mature_quota_same_snapshot_replay",
        "status": "exploratory preregistered one-frame intervention",
        "n_seed_snapshots": len(SEEDS),
        "valid_seed_snapshots": sum(x["valid"] for x in gate_records),
        "gate_results": gate_records,
        "arm_summaries": {
            name: {
                "mean_final_map_psnr_gain_db": float(np.mean([r["final_map_psnr_gain_db"] for r in rows])),
                "mean_final_map_depth_l1_m": float(np.mean([r["final_map_depth_l1_m"] for r in rows])),
                "mean_mature_update_coverage": float(np.mean([r["mature_update_coverage"] for r in rows])),
                "mean_n_gaussians_after_action": float(np.mean([r["n_gaussians_after_action"] for r in rows])),
                "mean_opt_stage_sync_ms": float(np.mean([r["opt_stage_sync_ms"] for r in rows])),
                "mean_frame_time_ms": float(np.mean([r["frame_time_ms"] for r in rows])),
                "per_seed": rows,
            } for name, rows in arm_rows.items()
        },
        "contrasts": {
            "primary_zero_spawn_mature75_minus_mature25_db": stats(effects["primary_spawn0_mature75_minus_mature25_db"], sign_test=True),
            "secondary_spawn128_mature75_minus_mature25_db": stats(effects["mature75_minus_mature25_spawn128_db"]),
            "secondary_spawn_by_quota_interaction_db": stats(effects["spawn_by_quota_interaction_db"]),
            "secondary_depth_l1_mature75_minus_mature25_m_at_zero_spawn": stats(effects["depth_l1_effect_spawn0_mature75_minus_mature25_m"]),
            "secondary_opt_stage_time_mature75_minus_mature25_ms_at_zero_spawn": stats(effects["opt_stage_time_mature75_minus_mature25_spawn0_ms"]),
        },
        "interpretation_limit": "One-step local treatment effect only; does not establish long-horizon mediation or cross-scene transfer.",
    }

    primary = effects["primary_spawn0_mature75_minus_mature25_db"]
    ci = bootstrap_ci(primary)
    interaction = effects["spawn_by_quota_interaction_db"]
    int_ci = bootstrap_ci(interaction, seed=20261018)
    depth_effect = effects["depth_l1_effect_spawn0_mature75_minus_mature25_m"]
    depth_ci = bootstrap_ci(depth_effect, seed=20261019)
    time_effect = effects["opt_stage_time_mature75_minus_mature25_spawn0_ms"]
    time_ci = bootstrap_ci(time_effect, seed=20261020)
    lines = [
        f"# {scene} mature-update quota: same-snapshot replay",
        "",
        f"This is a preregistered one-frame replay at frame 90 on `{scene}`. Each of eight seeds produces one map snapshot using the post-fix OURS policy. Four deep-copied branches share that exact map hash, RNG state and pre-action reference render. Every branch performs exactly 5,000 Gaussian micro-updates. Within each spawn dose, the 25% and 75% mature-quota branches have exactly the same final Gaussian count; spawn candidates are shared and nested.",
        "",
        "Mature means age ≥10 frames. The quality endpoint is a read-only final-map render at the same reference view, minus the common pre-action snapshot PSNR. The primary contrast is the 75%-minus-25% mature quota effect at zero spawn, which isolates age allocation at fixed map cardinality and update count.",
        "",
        f"**Validity:** {analysis['valid_seed_snapshots']}/{analysis['n_seed_snapshots']} seed snapshots passed every preregistered gate.",
        "",
        "## Arm summaries",
        "",
        "| Spawn dose | Mature quota | Mean PSNR gain over snapshot (dB) | Mean final depth L1 (m) | Mean mature update coverage | Mean final Gaussian count | Mean synchronized optimization time (ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARMS:
        spawn, quota = name.removeprefix("spawn").split("_mature")
        s = analysis["arm_summaries"][name]
        lines.append(f"| {spawn} | {quota}% | {s['mean_final_map_psnr_gain_db']:+.4f} | {s['mean_final_map_depth_l1_m']:.5f} | {s['mean_mature_update_coverage']:.4f} | {s['mean_n_gaussians_after_action']:.0f} | {s['mean_opt_stage_sync_ms']:.3f} |")
    lines += [
        "",
        "## Primary paired result",
        "",
        f"At zero spawn, raising the mature allocation from 25% to 75% changed final-map PSNR gain by **{np.mean(primary):+.4f} dB** (seed-bootstrap 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]); {sum(x > 0 for x in primary)}/8 seeds favored the 75% allocation. Exact two-sided sign-test p={exact_sign_test(primary):.4f}. The mature-update coverage changed from {analysis['arm_summaries']['spawn0_mature25']['mean_mature_update_coverage']:.4f} to {analysis['arm_summaries']['spawn0_mature75']['mean_mature_update_coverage']:.4f}, with identical map cardinality within each seed.",
        "",
        f"At spawn dose 128, the 75%-minus-25% mature-allocation effect was {np.mean(effects['mature75_minus_mature25_spawn128_db']):+.4f} dB (95% CI [{bootstrap_ci(effects['mature75_minus_mature25_spawn128_db'])[0]:+.4f}, {bootstrap_ci(effects['mature75_minus_mature25_spawn128_db'])[1]:+.4f}]). The spawn-by-quota interaction was {np.mean(interaction):+.4f} dB (95% CI [{int_ci[0]:+.4f}, {int_ci[1]:+.4f}]). These are secondary contrasts.",
        "",
        f"At zero spawn, mature 75% changed final-map depth L1 by {np.mean(depth_effect):+.5f} m versus 25% (95% CI [{depth_ci[0]:+.5f}, {depth_ci[1]:+.5f}]).",
        "",
        f"The synchronized optimization-stage time was {np.mean(time_effect):+.3f} ms higher on average for 75% versus 25% mature allocation at zero spawn (95% CI [{time_ci[0]:+.3f}, {time_ci[1]:+.3f}]). The intervention holds the number of Gaussian micro-updates fixed, not measured wall time.",
        "",
        "## Conclusion and limit",
        "",
        f"For this `{scene}` OURS snapshot and immediate next-frame render, the controlled contrast estimates the local effect of assigning more of the same optimizer update count to mature Gaussians. Because map, spawn dose, candidates, update count, seed and reference render are controlled, this isolates an immediate age-allocation effect. It does not show that mature-update starvation caused the multi-frame PSNR regression: this is one local action on eight snapshots at one frame. The dose-by-quota interaction should not by itself motivate an adaptive spawn controller.",
        "",
        f"Source commit: `{json.loads((BASE / 'manifest.json').read_text())['git_sha']}`. Frozen protocol: [`preregistration.md`](preregistration.md). Per-seed replay records and gate results are preserved next to this report.",
        "",
    ]
    (BASE / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (BASE / "report.md").write_text("\n".join(lines))
    source_paths = [
        "experiments/run_joint_spawn_refine_replay.py",
        "experiments/run_dilution_factorial.py",
        "experiments/run_phase13_frozen_benchmark.py",
        "research/pipeline.py", "research/scheduler.py", "research/reproducibility.py",
    ]
    source_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                     for p in source_paths}
    input_hashes = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(RUNS.glob("*.json"))}
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                         text=True).strip().splitlines()
    except Exception:
        status = ["unavailable"]
    provenance = {
        "experiment": f"{scene}_mature_quota_same_snapshot_replay",
        "source_revision_from_run_manifest": json.loads((BASE / "manifest.json").read_text())["git_sha"],
        "source_status_in_main_workspace_at_analysis": "unrelated pre-existing workspace changes; see source hashes",
        "source_sha256": source_hashes,
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((BASE / "preregistration.json").read_bytes()).hexdigest(),
        "protocol_markdown_sha256": hashlib.sha256((BASE / "preregistration.md").read_bytes()).hexdigest(),
        "run_artifact_sha256": input_hashes,
        "analysis_sha256": hashlib.sha256((BASE / "analysis.json").read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256((BASE / "report.md").read_bytes()).hexdigest(),
        "analysis_time_git_status_entry_count": len(status),
        "dataset_path_in_run_checkout": ("/tmp/adaptive_3dgs_phase14_postfix_clean_20261017/datasets/Replica/office1"
                                          if scene.startswith("replica_") else
                                          "/tmp/adaptive_3dgs_phase14_postfix_clean_20261017/datasets/TUM/rgbd_dataset_freiburg1_rpy"),
        "runtime": {k: json.loads((BASE / "manifest.json").read_text()).get(k)
                    for k in ("renderer_backend", "torch_version", "cuda_version", "gpu", "vram_gb")},
    }
    (BASE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(BASE / "report.md")


if __name__ == "__main__":
    main()
