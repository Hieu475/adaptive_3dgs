#!/usr/bin/env python3
"""Analyze same-snapshot growth x stale-rescue dose-response horizons."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize_arm(arm):
    rows = arm["horizon_frame_metrics"]
    fraction = arm.get("requested_stale_update_fraction")
    stale_updates = sum(int(row["stale_mature_micro_updates"]) for row in rows)
    total_updates = sum(int(row["n_micro_updates"]) for row in rows)
    quota_exact = None
    if fraction is not None:
        quota_exact = float(np.mean([
            int(row["stale_mature_micro_updates"]) ==
            round(int(row["n_micro_updates"]) * float(fraction)) for row in rows]))
    strata_acc = {}
    for row in rows:
        diag = row.get("age_stratified_diagnostics") or {}
        for name, stat in (diag.get("mature_revisit_strata") or {}).items():
            acc = strata_acc.setdefault(name, {
                "population": 0, "visible": 0, "staleness_population_sum": 0.0,
                "updates_population_sum": 0.0, "color_visible_sum": 0.0,
                "depth_visible_sum": 0.0,
            })
            population = int(stat.get("population_attributed", 0))
            visible = int(stat.get("visible_attributed", 0))
            acc["population"] += population
            acc["visible"] += visible
            acc["staleness_population_sum"] += float(stat.get("mean_staleness_frames", 0.0)) * population
            acc["updates_population_sum"] += float(stat.get("mean_update_count", 0.0)) * population
            acc["color_visible_sum"] += float(stat.get("visible_color_error_mean", 0.0)) * visible
            acc["depth_visible_sum"] += float(stat.get("visible_depth_error_mean", 0.0)) * visible
    revisit_strata = {}
    for name, acc in strata_acc.items():
        pop, visible = acc["population"], acc["visible"]
        revisit_strata[name] = {
            "mean_population_frame_observations": pop,
            "visible_frame_observations": visible,
            "visible_fraction": visible / pop if pop else 0.0,
            "mean_staleness_frames": acc["staleness_population_sum"] / pop if pop else 0.0,
            "mean_update_count": acc["updates_population_sum"] / pop if pop else 0.0,
            "visible_color_error_mean": acc["color_visible_sum"] / visible if visible else 0.0,
            "visible_depth_error_mean": acc["depth_visible_sum"] / visible if visible else 0.0,
        }
    return {
        "fixed_view_psnr_change_db": float(arm["fixed_eval_psnr_gain_db"]),
        "fixed_view_depth_l1_change": float(arm["fixed_eval_depth_l1_change"]),
        "stale_update_share": stale_updates / total_updates if total_updates else 0.0,
        "quota_exact_frame_fraction": quota_exact,
        "mean_mature_update_coverage": float(np.mean([r["mature_update_coverage"] for r in rows])),
        "mean_stale_update_coverage": float(np.mean([r["stale_mature_update_coverage"] for r in rows])),
        "mean_micro_updates": float(np.mean([r["n_micro_updates"] for r in rows])),
        "p50_opt_stage_ms": float(np.percentile([r["opt_stage_sync_ms"] for r in rows], 50)),
        "p95_opt_stage_ms": float(np.percentile([r["opt_stage_sync_ms"] for r in rows], 95)),
        "mean_frame_wall_ms": float(np.mean([r["frame_wall_sync_ms"] for r in rows])),
        "p95_frame_wall_ms": float(np.percentile([r["frame_wall_sync_ms"] for r in rows], 95)),
        "horizon_frames": len(rows),
        "mature_revisit_strata": revisit_strata,
    }


def mean_nested(values):
    first = values[0]
    if isinstance(first, dict):
        return {key: mean_nested([value[key] for value in values]) for key in first}
    if first is None:
        return None
    return float(np.mean(values))


def bootstrap(values, seed=20261021, n=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (n, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    args = ap.parse_args()
    files = sorted(args.input.glob("*/seed*_frame*.json"))
    if not files:
        raise SystemExit(f"No records found in {args.input}")
    records = [json.loads(path.read_text()) for path in files]
    valid, invalid = {}, []
    for rec in records:
        arms = rec["arms"]
        ok = (rec.get("all_count_and_snapshot_gates_passed") and
              all(a.get("valid") for a in arms) and
              len({a["snapshot_hash"] for a in arms}) == 1 and
              all(sum(int(r["n_spawned"]) for r in a["horizon_frame_metrics"]) == 0
                  for a in arms))
        if not ok:
            invalid.append({"scene": rec["scene"], "seed": rec["seed"], "snapshot_frame": rec["snapshot_frame"]})
            continue
        scene_seed = (str(rec["scene"]), int(rec["seed"]), int(rec["snapshot_frame"]))
        indexed = {}
        for arm in arms:
            action = arm["action"]
            dose = int(action["spawn_count"])
            mode = action["allocation"]
            fraction = action.get("stale_fraction") if mode == "stale_rescue" else None
            key = (dose, mode, None if fraction is None else float(fraction))
            indexed[key] = summarize_arm(arm)
        valid[scene_seed] = indexed
    if not valid:
        raise SystemExit("No seed blocks passed snapshot/arm/horizon gates")
    scenes = sorted({key[0] for key in valid})
    snapshots = sorted({key[2] for key in valid})
    doses = sorted({key[0] for arms in valid.values() for key in arms})
    fractions = sorted({key[2] for arms in valid.values() for key in arms if key[1] == "stale_rescue"})
    if len(doses) != 2 or not fractions:
        raise SystemExit("Expected two spawn doses and at least one stale-rescue fraction")

    seed_metrics = {}
    cells = {}
    for dose in doses:
        for mode, fraction in [("global", None)] + [("stale_rescue", f) for f in fractions]:
            key = (dose, mode, fraction)
            values = [arms[key] for arms in valid.values() if key in arms]
            if len(values) != len(valid):
                raise SystemExit(f"Missing paired arm {key} in one or more seed blocks")
            cell = {metric: mean_nested([v[metric] for v in values])
                    for metric in values[0]}
            for metric in values[0]:
                xs = [v[metric] for v in values]
                if isinstance(xs[0], (float, int)):
                    cell[metric + "_seed_bootstrap_ci95"] = bootstrap(xs)
            cells[f"spawn{dose}_{mode}" + (f"_{int(round(fraction*100))}" if fraction is not None else "")] = cell

    contrasts = {}
    for seed_key, arms in valid.items():
        scene_seed_name = f"{seed_key[0]}_seed{seed_key[1]}_frame{seed_key[2]}"
        seed_metrics[scene_seed_name] = {}
        for dose in doses:
            global_psnr = arms[(dose, "global", None)]["fixed_view_psnr_change_db"]
            for fraction in fractions:
                rescue = arms[(dose, "stale_rescue", fraction)]["fixed_view_psnr_change_db"]
                name = f"spawn{dose}_rescue{int(round(fraction*100))}_minus_global_db"
                contrasts.setdefault(name, []).append(rescue - global_psnr)
                seed_metrics[scene_seed_name][name] = rescue - global_psnr
        low, high = doses
        for mode, fraction, label in [("global", None, "global")] + [
                ("stale_rescue", f, f"rescue{int(round(f*100))}") for f in fractions]:
            low_value = arms[(low, mode, fraction)]["fixed_view_psnr_change_db"]
            high_value = arms[(high, mode, fraction)]["fixed_view_psnr_change_db"]
            name = f"growth_effect_{label}_db"
            contrasts.setdefault(name, []).append(high_value - low_value)
            seed_metrics[scene_seed_name][name] = high_value - low_value
    contrast_summary = {
        name: {"mean_db": float(np.mean(values)), "seed_bootstrap_ci95": bootstrap(values),
               "per_seed": {f"{k[0]}_seed{k[1]}": float(v)
                            for k, v in zip(valid, values)}}
        for name, values in contrasts.items()
    }
    analysis = {
        "status": "exploratory post hoc same-snapshot stale-rescue dose response; sequences are exposed",
        "scenes": scenes, "snapshot_frames": snapshots, "valid_seed_blocks": len(valid),
        "invalid_seed_blocks": invalid, "seed_ids_by_scene": {
            scene: sorted(k[1] for k in valid if k[0] == scene) for scene in scenes},
        "spawn_doses": doses, "stale_rescue_fractions": fractions,
        "target_opt_stage_ms": sorted({float(r["target_opt_stage_ms"]) for r in records
                                       if r.get("target_opt_stage_ms") is not None}),
        "cell_means_across_seed_blocks": cells,
        "seed_level_contrasts": seed_metrics, "paired_contrast_summary": contrast_summary,
        "interpretation_limit": "Exploratory seed blocks on previously exposed scenes. Quota is an update fraction request, not guaranteed compute share; measured optimization-stage time does not constrain full-frame latency. Fixed prefix views measure retention, not held-out generalization.",
    }
    (args.input / "dose_response_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    lines = ["# Same-snapshot stale-rescue dose response", "",
        f"Exploratory, exposed scenes {scenes}; snapshots {snapshots}. Valid paired seed blocks: {len(valid)}; invalid: {len(invalid)}. Growth doses: {doses}; stale-rescue requests: {fractions}.",
        "The stale fraction is a requested share of micro-updates. Candidate capacity can prevent the requested quota from being fully realized; the table reports realized stale work share and exact-quota frame frequency. Optimization-stage time targets are reported separately from full-frame wall latency.", "",
        "## Arm means", "",
        "| Arm | Fixed-view PSNR Δ dB | Depth-L1 Δ | Stale update share | Exact quota frames | Mature coverage | Updates/frame | Opt stage p50/p95 ms | Full-frame mean/p95 ms |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, c in cells.items():
        exact = "—" if c["quota_exact_frame_fraction"] is None else f"{c['quota_exact_frame_fraction']:.0%}"
        lines.append(f"| {name} | {c['fixed_view_psnr_change_db']:+.3f} | {c['fixed_view_depth_l1_change']:+.5f} | {c['stale_update_share']:.3f} | {exact} | {c['mean_mature_update_coverage']:.3f} | {c['mean_micro_updates']:.0f} | {c['p50_opt_stage_ms']:.1f}/{c['p95_opt_stage_ms']:.1f} | {c['mean_frame_wall_ms']:.1f}/{c['p95_frame_wall_ms']:.1f} |")
    lines += ["", "## Mature Gaussian residual and revisit strata", "",
        "Statistics are aggregated within each seed across pre-optimization horizon frames, then averaged across seeds. Staleness is current frame minus last update frame; residuals are measured only for Gaussians visible in that frame. These descriptive frame-level measurements are not treated as independent observations for confidence intervals.", "",
        "| Arm | Last-update age bin | Population-frame observations | Visible fraction | Mean staleness frames | Mean update count | Visible color residual | Visible depth residual |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for name, cell in cells.items():
        for bin_name, stats in cell["mature_revisit_strata"].items():
            lines.append(f"| {name} | {bin_name} | {stats['mean_population_frame_observations']:.0f} | {stats['visible_fraction']:.3f} | {stats['mean_staleness_frames']:.2f} | {stats['mean_update_count']:.2f} | {stats['visible_color_error_mean']:.5f} | {stats['visible_depth_error_mean']:.5f} |")
    lines += ["", "## Paired contrasts", "", "| Contrast | Mean dB | Seed bootstrap 95% CI | Per seed |", "|:--|--:|:--|:--|"]
    for name, row in contrast_summary.items():
        ci = row["seed_bootstrap_ci95"]
        per_seed = ", ".join(f"{seed}: {value:+.3f}" for seed, value in row["per_seed"].items())
        lines.append(f"| {name} | {row['mean_db']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {per_seed} |")
    lines += ["", "Interpretation is conditional on the tested snapshot and scene. Dose-response patterns can reveal whether stale-directed work has a monotonic, saturating, or non-monotonic retention effect, but this design alone does not establish mediation, long-horizon benefit, or generalization.", "",
        "Per-seed outputs, validity fields, update counts, stage timing, and provenance are in the adjacent JSON records and `dose_response_analysis.json`.", ""]
    (args.input / "dose_response_report.md").write_text("\n".join(lines))
    print(args.input / "dose_response_report.md")


if __name__ == "__main__":
    main()
