#!/usr/bin/env python3
"""Analyze preregistered same-snapshot spawn-dose x age-quota replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def bootstrap_mean(x, rng, n=10000):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return [None, None]
    idx = rng.integers(0, len(x), size=(n, len(x)))
    return [float(v) for v in np.percentile(x[idx].mean(axis=1), [2.5, 97.5])]


def hierarchical_scene_bootstrap(units, value_by_unit, rng, n=10000):
    """Resample seeds, then snapshot frames within each sampled seed."""
    by_seed = {}
    for unit in units:
        by_seed.setdefault(int(unit["seed"]), []).append(value_by_unit[id(unit)])
    seeds = sorted(by_seed)
    draws = []
    for _ in range(n):
        sampled = []
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            values = by_seed[int(seed)]
            sampled.extend(rng.choice(values, size=len(values), replace=True).tolist())
        draws.append(float(np.mean(sampled)))
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def summarize_units(units):
    lookup = {}
    for rec in units:
        for arm in rec["arms"]:
            key = (int(arm["action"]["spawn_count"]),
                   round(float(arm["action"]["mature_fraction"]), 5))
            lookup[key] = arm
    def gain(dose, frac):
        a = lookup[(dose, frac)]
        return a["post_psnr"] - a["pre_psnr"]
    fractions = sorted({k[1] for k in lookup})
    doses = sorted({k[0] for k in lookup})
    contrast = (gain(max(doses), max(fractions)) - gain(min(doses), max(fractions))) - (
        gain(max(doses), min(fractions)) - gain(min(doses), min(fractions)))
    effects = {}
    for frac in fractions:
        effects[f"spawn_gain_{min(doses)}_to_{max(doses)}_mature{int(frac*100)}"] = float(
            gain(max(doses), frac) - gain(min(doses), frac))
    return float(contrast), effects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results" / "joint_spawn_refine_replay" / "expanded_confirmation")
    args = ap.parse_args()
    files = sorted(args.input.glob("*/seed*_frame*.json"))
    records = [json.loads(p.read_text()) for p in files]
    valid = [r for r in records if r.get("all_count_and_snapshot_gates_passed")]
    failed = [r for r in records if not r.get("all_count_and_snapshot_gates_passed")]
    by_scene = {}
    for rec in valid:
        by_scene.setdefault(rec["scene"], []).append(rec)

    rng = np.random.default_rng(20261016)
    scene_results = {}
    lines = ["# Expanded same-snapshot spawn-dose × mature-quota replay", "",
             "Exploratory immediate one-step effects. Unit = scene × seed × snapshot frame. The primary contrast was fixed before the expanded batch.", "",
             f"Valid experimental units: {len(valid)}/{len(records)}; failed units: {len(failed)}; arms: {sum(len(r['arms']) for r in records)}.", "",
             "Primary contrast per unit: (PSNR gain 256 vs 0 at 75% mature updates) − (PSNR gain 256 vs 0 at 25% mature updates).",
             "Positive values mean the spawn-dose effect is more favorable under the higher mature quota.", "",
             "| Scene | Units | Primary contrast (dB) | 95% seed/frame hierarchical CI | spawn effect 0→256 @25% | spawn effect 0→256 @75% |",
             "|:--|--:|--:|:--|--:|--:|"]
    unit_contrasts = []
    for scene, units in sorted(by_scene.items()):
        contrasts, effects_by_unit = [], []
        for unit in units:
            c, e = summarize_units([unit])
            contrasts.append(c)
            effects_by_unit.append(e)
        contrast_by_id = {id(unit): c for unit, c in zip(units, contrasts)}
        ci = hierarchical_scene_bootstrap(units, contrast_by_id, rng)
        e25 = [e["spawn_gain_0_to_256_mature25"] for e in effects_by_unit]
        e75 = [e["spawn_gain_0_to_256_mature75"] for e in effects_by_unit]
        scene_results[scene] = {
            "n_units": len(units), "primary_contrast_mean_db": float(np.mean(contrasts)),
            "primary_contrast_ci95_seed_frame_hierarchical": ci,
            "spawn_gain_0_to_256_mature25_db": float(np.mean(e25)),
            "spawn_gain_0_to_256_mature75_db": float(np.mean(e75)),
            "unit_contrasts_db": contrasts,
        }
        lines.append(f"| {scene} | {len(units)} | {np.mean(contrasts):+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {np.mean(e25):+.4f} | {np.mean(e75):+.4f} |")
        unit_contrasts.extend((scene, c) for c in contrasts)

    # Cell summaries retain the dose curve and measured resource signals.
    cell = {}
    for dose in (0, 128, 256):
        for frac in (0.25, 0.75):
            arms = [a for rec in valid for a in rec["arms"]
                    if int(a["action"]["spawn_count"]) == dose
                    and abs(float(a["action"]["mature_fraction"]) - frac) < 1e-6]
            cell[f"spawn{dose}_mature{int(frac*100)}"] = {
                "n_arms": len(arms),
                "mean_psnr_gain_db": float(np.mean([a["post_psnr"] - a["pre_psnr"] for a in arms])),
                "mean_final_map_eval_psnr_gain_db": float(np.mean([
                    a["final_map_eval_psnr"] - rec["pre_action_snapshot_eval"]["psnr"]
                    for rec in valid for a in rec["arms"]
                    if int(a["action"]["spawn_count"]) == dose
                    and abs(float(a["action"]["mature_fraction"]) - frac) < 1e-6])),
                "mean_final_map_eval_depth_l1": float(np.mean([a["final_map_eval_depth_l1"] for a in arms])),
                "mean_final_map_depth_l1_gain": float(np.mean([
                    rec["pre_action_snapshot_eval"]["depth_l1"] - a["final_map_eval_depth_l1"]
                    for rec in valid for a in rec["arms"]
                    if int(a["action"]["spawn_count"]) == dose
                    and abs(float(a["action"]["mature_fraction"]) - frac) < 1e-6])),
                "mean_process_reported_depth_l1_preoptimization": float(np.mean([
                    a.get("process_reported_depth_l1_preoptimization",
                          a.get("pipeline_depth_l1_pre", a.get("pipeline_depth_l1", 0.0)))
                    for a in arms])),
                "mean_mature_update_coverage": float(np.mean([a["mature_update_coverage"] for a in arms])),
                "mean_growth_pressure": float(np.mean([a["growth_pressure"] for a in arms])),
                "mean_opt_stage_sync_ms": float(np.mean([a["opt_stage_sync_ms"] for a in arms])),
                "mean_frame_time_ms": float(np.mean([a["frame_time_ms"] for a in arms])),
                "mean_compute_return_db_per_opt_ms": float(np.mean([a["compute_return_db_per_opt_ms"] for a in arms])),
            }

    # Hierarchical bootstrap: resample scenes, then seeds, then snapshot frames.
    scene_names = sorted(by_scene)
    cluster_boot = []
    for _ in range(10000):
        chosen_scenes = rng.choice(scene_names, size=len(scene_names), replace=True)
        sampled = []
        for s in chosen_scenes:
            scene_units = by_scene[s]
            seed_ids = sorted({int(u["seed"]) for u in scene_units})
            for seed in rng.choice(seed_ids, size=len(seed_ids), replace=True):
                vals = [summarize_units([u])[0] for u in scene_units if int(u["seed"]) == int(seed)]
                sampled.extend(rng.choice(vals, size=len(vals), replace=True).tolist())
        cluster_boot.append(float(np.mean(sampled)))
    overall_contrasts = [v for _, v in unit_contrasts]
    overall_ci = [float(v) for v in np.percentile(cluster_boot, [2.5, 97.5])] if cluster_boot else [None, None]
    analysis = {
        "preregistration": "preregistration.json",
        "n_units_total": len(records), "n_units_valid": len(valid), "n_units_failed": len(failed),
        "n_arms_total": sum(len(r["arms"]) for r in records),
        "failed_units": [{"scene": r["scene"], "seed": r["seed"], "snapshot_frame": r["snapshot_frame"]} for r in failed],
        "scene_results": scene_results,
        "overall_primary_contrast_mean_db": float(np.mean(overall_contrasts)) if overall_contrasts else None,
        "overall_primary_contrast_scene_seed_frame_hierarchical_ci95_db": overall_ci,
        "cells": cell,
        "interpretation_limit": "One-step exploratory evidence only; no long-horizon, held-out, or causal-mediation claim.",
    }
    lines += ["", f"**Overall mean primary contrast:** {analysis['overall_primary_contrast_mean_db']:+.4f} dB; scene/seed/frame hierarchical bootstrap 95% interval [{overall_ci[0]:+.4f}, {overall_ci[1]:+.4f}].", "",
              "## Corrected final-map evaluation diagnostics", "",
              "The pipeline's `depth_l1` is computed from a pre-optimization render. The following metrics re-render the final map read-only at the same view; gains compare against the common pre-action snapshot map.", "",
              "| Arm | n | Final-map PSNR gain | Final-map depth L1 | Depth L1 gain | Pipeline pre-update depth L1 | Mature coverage | Growth pressure | Opt stage ms | Frame ms | Compute return |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, d in cell.items():
        lines.append(f"| {name} | {d['n_arms']} | {d['mean_final_map_eval_psnr_gain_db']:+.4f} | {d['mean_final_map_eval_depth_l1']:.5f} | {d['mean_final_map_depth_l1_gain']:+.5f} | {d['mean_process_reported_depth_l1_preoptimization']:.5f} | {d['mean_mature_update_coverage']:.4f} | {d['mean_growth_pressure']:.5f} | {d['mean_opt_stage_sync_ms']:.3f} | {d['mean_frame_time_ms']:.3f} | {d['mean_compute_return_db_per_opt_ms']:+.6f} |")
    lines += ["", "Failed gate records: " + (", ".join(f"{r['scene']}/seed{r['seed']}/frame{r['snapshot_frame']}" for r in failed) if failed else "none."), "",
              "Do not infer policy selection from this immediate endpoint. Follow-on work should target long-horizon runs and independent scene-level prediction."]
    (args.input / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (args.input / "report.md").write_text("\n".join(lines) + "\n")
    source_paths = [
        "experiments/run_joint_spawn_refine_replay.py",
        "experiments/analyze_joint_spawn_refine_replay.py",
        "experiments/run_dilution_factorial.py",
        "experiments/run_frozen_benchmark.py",
        "src/core/pipeline.py", "src/core/scheduler.py", "src/core/densification.py",
        "src/core/gaussian_repr.py", "src/evaluation/reproducibility.py",
    ]
    source_hashes = {}
    for rel in source_paths:
        path = ROOT / rel
        source_hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip().splitlines()
    except Exception:
        git_sha, git_status = None, ["unavailable"]
    try:
        import torch
        runtime = {"torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as exc:
        runtime = {"error": str(exc)}
    provenance = {
        "experiment": "joint_spawn_refine_same_snapshot_replay",
        "status": "exploratory; workspace was dirty; relevant file hashes recorded",
        "git_head": git_sha,
        "git_dirty": bool(git_status),
        "git_status_entry_count": len(git_status),
        "source_sha256": source_hashes,
        "preregistration_file": ("preregistration_addendum.json" if (args.input / "preregistration_addendum.json").exists() else "preregistration.json"),
        "preregistration_sha256": hashlib.sha256((args.input / ("preregistration_addendum.json" if (args.input / "preregistration_addendum.json").exists() else "preregistration.json")).read_bytes()).hexdigest(),
        "runtime": runtime,
        "dataset_scenes": sorted(by_scene),
        "protocol": {"seeds": sorted({int(r["seed"]) for r in records}),
                     "snapshot_frames": sorted({int(r["snapshot_frame"]) for r in records}),
                     "spawn_counts": [0, 128, 256], "mature_fractions": [0.25, 0.75],
                     "micro_updates_per_arm": (int(records[0]["target_micro_updates_per_arm"]) if records else None),
                     "map_snapshots_persisted": bool(records) and all(r.get("map_snapshot_sha256") for r in records),
                     "units": len(records)},
    }
    (args.input / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(args.input / "report.md")


if __name__ == "__main__":
    main()
