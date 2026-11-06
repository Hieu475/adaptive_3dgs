#!/usr/bin/env python3
"""Summarize same-snapshot growth x stale-mature rescue replay units."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def bootstrap_seed_frame_ci(units, values, rng, n_boot=10000):
    """Resample seed clusters, then frames nested within each sampled seed."""
    by_seed = {}
    for unit, value in zip(units, values):
        by_seed.setdefault(int(unit["seed"]), []).append(float(value))
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        return [None, None]
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        selected = rng.choice(seeds, size=len(seeds), replace=True)
        vals = []
        for seed in selected:
            frame_vals = by_seed[int(seed)]
            vals.extend(rng.choice(frame_vals, size=len(frame_vals), replace=True).tolist())
        samples[b] = np.mean(vals)
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def lookup_arms(unit):
    return {(int(a["action"]["spawn_count"]), a["action"]["allocation"]): a
            for a in unit["arms"]}


def gain(arm):
    return float(arm["final_map_eval_psnr"] - arm["pre_psnr"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.input.glob("*/seed*_frame*.json"))
    records = [json.loads(path.read_text()) for path in files]
    valid = [r for r in records if r.get("all_count_and_snapshot_gates_passed")]
    failed = [r for r in records if not r.get("all_count_and_snapshot_gates_passed")]
    if not records:
        raise SystemExit(f"No replay records found under {args.input}")

    rng = np.random.default_rng(20261021)
    doses = sorted({int(d) for r in records for d in r.get("target_spawn_doses", [])})
    if len(doses) < 2:
        raise SystemExit("Need at least two spawn doses for the growth contrast")
    low, high = doses[0], doses[-1]

    interactions, global_growth_effects, rescue_deltas = [], [], []
    for rec in valid:
        arms = lookup_arms(rec)
        get_gain = lambda dose, mode: gain(arms[(dose, mode)])
        interactions.append((get_gain(high, "stale_rescue") - get_gain(high, "global")) -
                            (get_gain(low, "stale_rescue") - get_gain(low, "global")))
        global_growth_effects.append(get_gain(high, "global") - get_gain(low, "global"))
        rescue_deltas.append(get_gain(high, "stale_rescue") - get_gain(high, "global"))

    interaction_ci = bootstrap_seed_frame_ci(valid, interactions, rng)
    growth_ci = bootstrap_seed_frame_ci(valid, global_growth_effects, rng)
    rescue_ci = bootstrap_seed_frame_ci(valid, rescue_deltas, rng)

    cells = {}
    for dose in doses:
        for mode in ("global", "stale_rescue"):
            arms = [a for rec in valid for a in rec["arms"]
                    if int(a["action"]["spawn_count"]) == dose
                    and a["action"]["allocation"] == mode]
            cells[f"spawn{dose}_{mode}"] = {
                "n_arms": len(arms),
                "mean_final_map_psnr_gain_db": float(np.mean([gain(a) for a in arms])) if arms else None,
                "mean_mature_update_coverage": float(np.mean([a["mature_update_coverage"] for a in arms])) if arms else None,
                "mean_stale_mature_update_coverage": float(np.mean([a["stale_mature_update_coverage"] for a in arms])) if arms else None,
                "mean_stale_mature_eligible": float(np.mean([a["stale_mature_eligible"] for a in arms])) if arms else None,
                "mean_stale_mature_micro_updates": float(np.mean([a["stale_mature_micro_updates"] for a in arms])) if arms else None,
                "mean_n_spawned": float(np.mean([a["n_spawned"] for a in arms])) if arms else None,
                "mean_final_gaussian_count": float(np.mean([a["n_gaussians_after_action"] for a in arms])) if arms else None,
                "mean_optimization_stage_ms": float(np.mean([a["opt_stage_sync_ms"] for a in arms])) if arms else None,
                "mean_frame_ms": float(np.mean([a["frame_time_ms"] for a in arms])) if arms else None,
            }

    analysis = {
        "experiment": "growth_starvation_same_snapshot_factorial",
        "status": "exploratory one-step replay; sequences/snapshots were previously exposed",
        "n_units_total": len(records), "n_units_valid": len(valid), "n_units_failed": len(failed),
        "failed_units": [{"scene": r["scene"], "seed": r["seed"],
                          "snapshot_frame": r["snapshot_frame"]} for r in failed],
        "spawn_low": low, "spawn_high": high,
        "global_growth_effect_high_minus_low_db": float(np.mean(global_growth_effects)) if valid else None,
        "global_growth_effect_ci95_seed_frame": growth_ci,
        "rescue_effect_at_high_growth_db": float(np.mean(rescue_deltas)) if valid else None,
        "rescue_effect_at_high_growth_ci95_seed_frame": rescue_ci,
        "growth_by_rescue_interaction_db": float(np.mean(interactions)) if valid else None,
        "growth_by_rescue_interaction_ci95_seed_frame": interaction_ci,
        "cells": cells,
        "interpretation_limit": "One-step same-snapshot mechanism pilot; cannot establish long-horizon mediation, generalization, or equal-wall-time benefit.",
    }
    lines = [
        "# Same-snapshot growth × stale-mature rescue factorial", "",
        "Exploratory one-step causal pilot. Every arm starts from one common map snapshot and reference view; global and rescue arms use the same total micro-update count. This is not a held-out or long-horizon result.", "",
        f"Valid units: {len(valid)}/{len(records)}; failed units: {len(failed)}; doses: {doses}.", "",
        "Primary diagnostic: does higher spawn dose change stale-mature update coverage under global allocation, and does the stale-rescue intervention attenuate any resulting PSNR loss?",
        "",
        "| Arm | n | PSNR gain (dB) | Mature coverage | Stale-mature coverage | Stale eligible | Stale updates | Spawned | Final map | Opt ms | Frame ms |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name, cell in cells.items():
        lines.append(f"| {name} | {cell['n_arms']} | {cell['mean_final_map_psnr_gain_db']:+.4f} | "
                     f"{cell['mean_mature_update_coverage']:.4f} | {cell['mean_stale_mature_update_coverage']:.4f} | "
                     f"{cell['mean_stale_mature_eligible']:.1f} | {cell['mean_stale_mature_micro_updates']:.1f} | "
                     f"{cell['mean_n_spawned']:.1f} | {cell['mean_final_gaussian_count']:.1f} | "
                     f"{cell['mean_optimization_stage_ms']:.2f} | {cell['mean_frame_ms']:.2f} |")
    lines += [
        "", f"**Global allocation: high-minus-low spawn PSNR effect:** {analysis['global_growth_effect_high_minus_low_db']:+.4f} dB; seed/frame 95% CI {growth_ci}.",
        f"**Stale-rescue effect at high growth:** {analysis['rescue_effect_at_high_growth_db']:+.4f} dB; seed/frame 95% CI {rescue_ci}.",
        f"**Growth × rescue interaction:** {analysis['growth_by_rescue_interaction_db']:+.4f} dB; seed/frame 95% CI {interaction_ci}.",
        "", "Failed units: " + (", ".join(f"{r['scene']}/seed{r['seed']}/frame{r['snapshot_frame']}" for r in failed) if failed else "none."),
        "", "Do not infer that Gaussian growth causes long-horizon quality loss from this one-step experiment. The growth dose also changes representation support. A positive rescue interaction is consistent with starvation mediation but needs long-horizon replication and a measured-time comparison.",
    ]
    (args.input / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (args.input / "report.md").write_text("\n".join(lines) + "\n")

    source_paths = ["experiments/run_growth_starvation_factorial.py",
                    "experiments/analyze_growth_starvation_factorial.py",
                    "research/pipeline.py", "research/state_store.py",
                    "research/gaussian_repr.py", "research/reproducibility.py"]
    source_hashes = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                     for rel in source_paths}
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip().splitlines()
    except Exception:
        git_head, git_status = None, ["unavailable"]
    provenance = {
        "experiment": "growth_starvation_same_snapshot_factorial",
        "git_head": git_head, "git_dirty": bool(git_status),
        "git_status_entry_count": len(git_status), "source_sha256": source_hashes,
        "run_manifest": "manifest.json",
        "runtime": {"torch": __import__("torch").__version__,
                    "cuda": __import__("torch").version.cuda,
                    "gpu": (__import__("torch").cuda.get_device_name(0)
                            if __import__("torch").cuda.is_available() else None)},
        "protocol": {"units": len(records), "valid_units": len(valid),
                     "spawn_doses": doses,
                     "allocation_modes": ["global", "stale_rescue"],
                     "fixed_micro_updates_per_arm": int(records[0]["target_micro_updates_per_arm"]),
                     "stale_after_frames": int(records[0]["stale_after_frames"])},
    }
    (args.input / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(args.input / "report.md")


if __name__ == "__main__":
    main()
