#!/usr/bin/env python3
"""Summarize matched-state growth × stale-rescue horizon replays."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize_arm(arm):
    rows = arm["horizon_frame_metrics"]
    mature = sum(int(r["mature_eligible"]) for r in rows)
    stale = sum(int(r["stale_mature_eligible"]) for r in rows)
    stale_updated = sum(int(r["stale_mature_updated"]) for r in rows)
    total_updates = sum(int(r["n_micro_updates"]) for r in rows)
    stale_micro_updates = sum(int(r["stale_mature_micro_updates"]) for r in rows)
    is_rescue = arm["action"]["allocation"] == "stale_rescue"
    requested_fraction = float(arm.get("requested_stale_update_fraction") or 0.0)
    quota_exact_frames = sum(
        int(r["stale_mature_micro_updates"]) == int(round(
            int(r["n_micro_updates"]) * requested_fraction)) for r in rows)
    return {
        "fixed_eval_psnr_gain_db": float(arm["fixed_eval_psnr_gain_db"]),
        "fixed_eval_depth_l1_change": float(arm["fixed_eval_depth_l1_change"]),
        "starvation_prevalence": stale / mature if mature else 0.0,
        "stale_candidate_update_coverage": stale_updated / stale if stale else 0.0,
        "stale_micro_update_share": stale_micro_updates / total_updates if total_updates else 0.0,
        "stale_quota_exact_frame_fraction": (quota_exact_frames / len(rows)
                                               if rows and is_rescue else None),
        "mean_mature_coverage": float(np.mean([r["mature_update_coverage"] for r in rows])),
        "mean_stale_update_coverage": float(np.mean([r["stale_mature_update_coverage"] for r in rows])),
        "mean_opt_stage_ms": float(np.mean([r["opt_stage_sync_ms"] for r in rows])),
        "p50_opt_stage_ms": float(np.percentile([r["opt_stage_sync_ms"] for r in rows], 50)),
        "p95_opt_stage_ms": float(np.percentile([r["opt_stage_sync_ms"] for r in rows], 95)),
        "mean_frame_wall_ms": float(np.mean([r["frame_wall_sync_ms"] for r in rows])),
        "p95_frame_wall_ms": float(np.percentile([r["frame_wall_sync_ms"] for r in rows], 95)),
        "mean_micro_updates_per_horizon_frame": float(np.mean([r["n_micro_updates"] for r in rows])),
        "target_opt_stage_ms": float(arm["target_opt_stage_ms"])
            if arm.get("target_opt_stage_ms") is not None else None,
        "budget_mode": arm.get("budget_mode", "fixed_micro_updates"),
        "horizon_spawns": int(sum(r["n_spawned"] for r in rows)),
        "horizon_updates": int(sum(r["n_micro_updates"] for r in rows)),
        "horizon_frames": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--compare_input", type=Path,
                    help="Optional shorter-horizon run with matching seed/snapshot arms")
    args = ap.parse_args()
    records = [json.loads(p.read_text()) for p in sorted(args.input.glob("*/seed*_frame*.json"))]
    if not records:
        raise SystemExit(f"No same-snapshot replay records found in {args.input}")
    invalid = []
    by_seed = {}
    for rec in records:
        seed = int(rec["seed"])
        arms = {(int(a["action"]["spawn_count"]), a["action"]["allocation"]): a
                for a in rec["arms"]}
        arm_valid = all(a.get("valid") for a in rec["arms"])
        no_horizon_spawns = all(
            sum(int(row["n_spawned"]) for row in a["horizon_frame_metrics"]) == 0
            for a in rec["arms"])
        measured_time = rec.get("budget_mode") == "measured_opt_stage_time"
        exact_updates = all(
            a["n_micro_updates"] == rec["target_micro_updates_per_arm"] and
            len(a["horizon_frame_metrics"]) == rec["outcome_horizon_frames"] and
            (all(int(row["n_micro_updates"]) == rec["target_micro_updates_per_arm"]
                 for row in a["horizon_frame_metrics"]) if not measured_time else
             all(int(row["n_micro_updates"]) > 0 and
                 np.isfinite(float(row["opt_stage_sync_ms"])) and
                 float(row["opt_stage_sync_ms"]) > 0
                 for row in a["horizon_frame_metrics"]) and
             bool(a.get("measured_time_budget_gate")))
            for a in rec["arms"])
        common_hash = len({a["snapshot_hash"] for a in rec["arms"]}) == 1
        update_budget_gate = exact_updates
        passed = (rec["all_count_and_snapshot_gates_passed"] and arm_valid and
                  no_horizon_spawns and update_budget_gate and common_hash)
        if not passed:
            invalid.append({"seed": seed, "snapshot_frame": rec["snapshot_frame"],
                            "arm_valid": arm_valid, "no_horizon_spawns": no_horizon_spawns,
                            "update_budget_gate": update_budget_gate, "common_hash": common_hash})
        if passed:
            by_seed[seed] = arms
    if not by_seed:
        raise SystemExit("No records passed the full horizon validity gates")

    doses = sorted({dose for arms in by_seed.values() for dose, _ in arms})
    scene_names = sorted({str(rec["scene"]) for rec in records})
    snapshot_frames = sorted({int(rec["snapshot_frame"]) for rec in records})
    if len(doses) != 2 or len(by_seed) < 1:
        raise SystemExit("Expected exactly two spawn doses")
    low, high = doses
    cells = {}
    for dose in doses:
        for allocation in ("global", "stale_rescue"):
            values = [summarize_arm(arms[(dose, allocation)]) for arms in by_seed.values()]
            cells[f"spawn{dose}_{allocation}"] = {}
            for key in values[0]:
                column = [v[key] for v in values]
                if isinstance(column[0], str):
                    cells[f"spawn{dose}_{allocation}"][key] = (
                        column[0] if all(x == column[0] for x in column) else "mixed")
                else:
                    cells[f"spawn{dose}_{allocation}"][key] = (
                        float(np.mean(column)) if all(x is not None for x in column)
                        else None)
    contrasts_by_seed = {}
    for seed, arms in by_seed.items():
        s = {key: summarize_arm(arm) for key, arm in arms.items()}
        def diff(a, b, metric):
            return s[a][metric] - s[b][metric]
        contrasts_by_seed[seed] = {
            "growth_effect_global_fixed_psnr_db": diff((high, "global"), (low, "global"), "fixed_eval_psnr_gain_db"),
            "growth_effect_rescue_fixed_psnr_db": diff((high, "stale_rescue"), (low, "stale_rescue"), "fixed_eval_psnr_gain_db"),
            "rescue_effect_low_growth_fixed_psnr_db": diff((low, "stale_rescue"), (low, "global"), "fixed_eval_psnr_gain_db"),
            "rescue_effect_high_growth_fixed_psnr_db": diff((high, "stale_rescue"), (high, "global"), "fixed_eval_psnr_gain_db"),
            "growth_by_rescue_interaction_fixed_psnr_db": diff((high, "stale_rescue"), (high, "global"), "fixed_eval_psnr_gain_db") - diff((low, "stale_rescue"), (low, "global"), "fixed_eval_psnr_gain_db"),
            "growth_effect_global_starvation": diff((high, "global"), (low, "global"), "starvation_prevalence"),
            "growth_effect_global_mature_coverage": diff((high, "global"), (low, "global"), "mean_mature_coverage"),
            "rescue_effect_low_growth_starvation": diff((low, "stale_rescue"), (low, "global"), "starvation_prevalence"),
            "rescue_effect_high_growth_starvation": diff((high, "stale_rescue"), (high, "global"), "starvation_prevalence"),
        }
    contrasts = {name: float(np.mean([row[name] for row in contrasts_by_seed.values()]))
                 for name in next(iter(contrasts_by_seed.values()))}
    rng = np.random.default_rng(20261021)
    contrast_ci95 = {}
    seed_ids = sorted(contrasts_by_seed)
    for name in contrasts:
        samples = np.empty(10000, dtype=float)
        values = np.asarray([contrasts_by_seed[s][name] for s in seed_ids], dtype=float)
        for idx in range(len(samples)):
            samples[idx] = np.mean(rng.choice(values, size=len(values), replace=True))
        contrast_ci95[name] = [float(v) for v in np.percentile(samples, [2.5, 97.5])]
    horizon_comparison = None
    if args.compare_input:
        compare_records = [json.loads(p.read_text()) for p in
                           sorted(args.compare_input.glob("*/seed*_frame*.json"))]
        compare_by_seed = {}
        for rec in compare_records:
            if rec.get("all_count_and_snapshot_gates_passed") and all(a.get("valid") for a in rec["arms"]):
                compare_by_seed[int(rec["seed"])] = {
                    (int(a["action"]["spawn_count"]), a["action"]["allocation"]): a
                    for a in rec["arms"]}
        matched = sorted(set(by_seed) & set(compare_by_seed))
        horizon_comparison = {"compare_input": str(args.compare_input), "matched_seeds": matched,
                              "cells": {}}
        for dose in doses:
            for allocation in ("global", "stale_rescue"):
                key = f"spawn{dose}_{allocation}"
                differences = []
                for seed in matched:
                    current = summarize_arm(by_seed[seed][(dose, allocation)])
                    shorter = summarize_arm(compare_by_seed[seed][(dose, allocation)])
                    differences.append(current["fixed_eval_psnr_gain_db"] -
                                       shorter["fixed_eval_psnr_gain_db"])
                if differences:
                    arr = np.asarray(differences)
                    boot = np.asarray([np.mean(rng.choice(arr, size=len(arr), replace=True))
                                       for _ in range(10000)])
                    horizon_comparison["cells"][key] = {
                        "longer_minus_shorter_fixed_psnr_gain_db": float(np.mean(arr)),
                        "seed_bootstrap_ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
                        "per_seed": {str(seed): float(value)
                                     for seed, value in zip(matched, differences)},
                    }
    analysis = {
        "status": "exploratory matched-state horizon replication; exposed sequence; no held-out claim",
        "scenes": scene_names, "snapshot_frames": snapshot_frames,
        "records_total": len(records), "records_valid": len(by_seed), "invalid_records": invalid,
        "independent_seed_count": len(by_seed), "seed_ids": sorted(by_seed),
        "spawn_doses": doses, "cells_mean_over_seeds": cells,
        "seed_level_contrasts": contrasts_by_seed, "mean_seed_contrasts": contrasts,
        "seed_bootstrap_ci95": contrast_ci95,
        "paired_horizon_comparison": horizon_comparison,
        "budget_mode": records[0].get("budget_mode", "fixed_micro_updates"),
        "target_opt_stage_ms": records[0].get("target_opt_stage_ms"),
        "interpretation_limit": "These exposed sequences establish only exploratory, scene-conditional evidence. Growth may alter representation support; the intervention and fixed-prefix endpoint do not establish a universal starvation mediation mechanism. Measured optimization-stage time is not full-pipeline real-time latency.",
    }
    lines = [
        "# Matched-state growth × starvation horizon pilot", "",
        f"Exploratory, sequence-conditional result: one common snapshot per seed on {', '.join(f'`{s}`' for s in scene_names)} at frame(s) {snapshot_frames}, a one-time spawn intervention followed by a {records[0].get('budget_mode', 'fixed_micro_updates')} horizon with no subsequent births, and fixed pre-intervention prefix views evaluated before and after. The configured optimization-stage target is {records[0].get('target_opt_stage_ms')} ms; this target does not constrain whole-frame latency. Seed-bootstrap intervals quantify repeatability across these initializations on the listed scene/snapshot only; they do not establish cross-scene generalization.", "",
        f"Valid seed units: {len(by_seed)}/{len(records)}; invalid units: {len(invalid)}; common snapshot/hash, configured update-budget mode, completed horizon, and zero-additional-spawn gates are required.", "",
        "## Arm means across valid seeds", "",
        "| Arm | Fixed-view PSNR change (dB) | Fixed-view depth-L1 change | Starvation prevalence | Stale candidate coverage | Stale update share | Quota exact frames | Mature coverage | Mean updates/frame | Opt ms p50/p95 | Full-frame wall ms mean/p95 | Horizon births |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name, c in cells.items():
        exact = (f"{c['stale_quota_exact_frame_fraction']:.0%}"
                 if c["stale_quota_exact_frame_fraction"] is not None else "—")
        lines.append(f"| {name} | {c['fixed_eval_psnr_gain_db']:+.3f} | {c['fixed_eval_depth_l1_change']:+.5f} | {c['starvation_prevalence']:.3f} | {c['stale_candidate_update_coverage']:.3f} | {c['stale_micro_update_share']:.3f} | {exact} | {c['mean_mature_coverage']:.3f} | {c['mean_micro_updates_per_horizon_frame']:.0f} | {c['p50_opt_stage_ms']:.1f}/{c['p95_opt_stage_ms']:.1f} | {c['mean_frame_wall_ms']:.1f}/{c['p95_frame_wall_ms']:.1f} | {c['horizon_spawns']:.0f} |")
    lines += ["", "## Matched contrasts (descriptive means over seeds)", "",
              "| Contrast | Mean | Seed-bootstrap 95% CI | Per-seed values |", "|:--|--:|:--|:--|"]
    for name, value in contrasts.items():
        vals = ", ".join(f"seed {seed}: {row[name]:+.4f}" for seed, row in contrasts_by_seed.items())
        ci = contrast_ci95[name]
        lines.append(f"| {name} | {value:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {vals} |")
    if horizon_comparison:
        lines += ["", "## Paired horizon extension", "",
                  f"Longer-minus-shorter change in fixed-view PSNR gain, matched by seed: {horizon_comparison['matched_seeds']}. Positive means the longer horizon retains/improves the fixed prefix views more; bootstrap resamples these matched seeds.",
                  "", "| Arm | Mean longer−shorter (dB) | Seed-bootstrap 95% CI | Per-seed values |",
                  "|:--|--:|:--|:--|"]
        for name, row in horizon_comparison["cells"].items():
            vals = ", ".join(f"{seed}: {value:+.4f}" for seed, value in row["per_seed"].items())
            ci = row["seed_bootstrap_ci95"]
            lines.append(f"| {name} | {row['longer_minus_shorter_fixed_psnr_gain_db']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {vals} |")
    lines += ["", "## Interpretation", "",
              f"At the descriptive seed mean, the 3,584-birth intervention changed global-allocation mature update coverage by {contrasts['growth_effect_global_mature_coverage']:+.3f}, stale/mature eligible population ratio by {contrasts['growth_effect_global_starvation']:+.3f}, and fixed-view PSNR change by {contrasts['growth_effect_global_fixed_psnr_db']:+.3f} dB. Rescue changed the stale/mature ratio by {contrasts['rescue_effect_low_growth_starvation']:+.3f} at zero growth and {contrasts['rescue_effect_high_growth_starvation']:+.3f} at high growth; its fixed-view PSNR effects were {contrasts['rescue_effect_low_growth_fixed_psnr_db']:+.3f} and {contrasts['rescue_effect_high_growth_fixed_psnr_db']:+.3f} dB. The growth treatment altered exposure, but rescue did not attenuate a growth-associated fixed-view quality loss in this pilot (growth-by-rescue interaction {contrasts['growth_by_rescue_interaction_fixed_psnr_db']:+.3f} dB). The result supports a scene/stage-conditional dilution signal, not a demonstrated quality mediation mechanism. Spawn can also add useful representation support, so its quality effects cannot be attributed to starvation alone.",
              "",
              "Quota realization: stale-rescue arms only realize the requested per-frame fraction when enough stale candidates exist. The table reports the realized work share and exact-quota frame fraction; the nominal 50% request must not be described as 50% realized allocation.", "",
              "All contrasts are paired within seed; bootstrap intervals resample seeds only. The estimates are conditional on the listed exposed scene and snapshot. In measured-time mode, the controller matches optimization-stage wall time approximately; this does not match whole-frame time. Full-frame latency is reported separately and includes rendering and other pipeline work. Fixed prefix views measure retention rather than unseen-view generalization.", ""]
    (args.input / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (args.input / "report.md").write_text("\n".join(lines))
    import hashlib
    import subprocess
    root = Path(__file__).resolve().parent.parent
    source_paths = ["experiments/run_growth_starvation_factorial.py",
                    "experiments/analyze_growth_starvation_horizon.py",
                    "experiments/analyze_sequence_pose_motion.py",
                    "experiments/run_phase13_frozen_benchmark.py",
                    "experiments/run_dilution_factorial.py",
                    "research/pipeline.py", "research/state_store.py",
                    "research/gaussian_repr.py", "research/reproducibility.py"]
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).strip().splitlines()
    except Exception:
        head, status = None, ["unavailable"]
    import torch
    provenance = {
        "experiment": "matched_state_growth_starvation_horizon_pilot",
        "git_head": head, "git_dirty": bool(status), "git_status_entry_count": len(status),
        "source_sha256": {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
                           for rel in source_paths},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "protocol": {"valid_seed_count": len(by_seed), "seed_ids": sorted(by_seed),
                     "spawn_doses": doses,
                     "outcome_horizon_frames": int(records[0]["outcome_horizon_frames"]),
                     "fixed_micro_updates_per_frame": int(records[0]["target_micro_updates_per_arm"])},
    }
    (args.input / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(args.input / "report.md")


if __name__ == "__main__":
    main()
