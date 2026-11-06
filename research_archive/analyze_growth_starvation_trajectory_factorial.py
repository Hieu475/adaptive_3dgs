#!/usr/bin/env python3
"""Analyze the continuous growth x stale-mature rescue factorial pilot."""
import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_seed_ci(values, n=10000, seed=20261021):
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    args = ap.parse_args()
    files = sorted(args.input.glob("*_seed*_*.json"))
    records = [json.loads(p.read_text()) for p in files]
    records = [r for r in records if isinstance(r, dict) and "summary" in r]
    if not records:
        raise SystemExit("No trajectory arm records found")
    by_key = {(r["summary"]["seed"], r["summary"]["growth"], r["summary"]["allocation"]): r
              for r in records}
    seeds = sorted({int(r["summary"]["seed"]) for r in records})
    complete = [s for s in seeds if all((s, g, a) in by_key
                for g in ("unthrottled", "throttled")
                for a in ("global", "stale_rescue"))]
    records_complete = [by_key[(s, g, a)] for s in complete
                        for g in ("unthrottled", "throttled")
                        for a in ("global", "stale_rescue")]

    def summary(seed, growth, allocation):
        return by_key[(seed, growth, allocation)]["summary"]

    def contrast(field, function):
        return [function(summary(s, "unthrottled", "global"),
                         summary(s, "throttled", "global"),
                         summary(s, "unthrottled", "stale_rescue"),
                         summary(s, "throttled", "stale_rescue")) for s in complete]

    global_growth = contrast("mean_psnr", lambda ug, tg, ur, tr: ug["mean_psnr"] - tg["mean_psnr"])
    rescue_unthrottled = contrast("mean_psnr", lambda ug, tg, ur, tr: ur["mean_psnr"] - ug["mean_psnr"])
    rescue_throttled = contrast("mean_psnr", lambda ug, tg, ur, tr: tr["mean_psnr"] - tg["mean_psnr"])
    interaction = [a - b for a, b in zip(rescue_unthrottled, rescue_throttled)]
    fixed_rescue_unthrottled = contrast("fixed_eval_mean_psnr", lambda ug, tg, ur, tr: ur["fixed_eval_mean_psnr"] - ug["fixed_eval_mean_psnr"])
    fixed_rescue_throttled = contrast("fixed_eval_mean_psnr", lambda ug, tg, ur, tr: tr["fixed_eval_mean_psnr"] - tg["fixed_eval_mean_psnr"])
    fixed_interaction = [a - b for a, b in zip(fixed_rescue_unthrottled, fixed_rescue_throttled)]
    contrast_summary = {
        "unthrottled_minus_throttled_psnr_global_db": global_growth,
        "stale_rescue_effect_unthrottled_psnr_db": rescue_unthrottled,
        "stale_rescue_effect_throttled_psnr_db": rescue_throttled,
        "growth_by_rescue_interaction_psnr_db": interaction,
        "fixed_view_rescue_effect_unthrottled_psnr_db": fixed_rescue_unthrottled,
        "fixed_view_rescue_effect_throttled_psnr_db": fixed_rescue_throttled,
        "fixed_view_growth_by_rescue_interaction_psnr_db": fixed_interaction,
    }
    contrasts = {}
    for name, values in contrast_summary.items():
        contrasts[name] = {"seed_values": values,
                           "mean": float(np.mean(values)) if values else None,
                           "seed_bootstrap_ci95": bootstrap_seed_ci(values) if values else [None, None]}

    arm_rows = []
    for r in sorted(records_complete, key=lambda x:(x["summary"]["growth"], x["summary"]["allocation"], x["summary"]["seed"])):
        s = r["summary"]
        logs = r["frames"]
        mature_eligible_sum = sum(int(x["mature_eligible"]) for x in logs)
        stale_eligible_sum = sum(int(x["stale_mature_eligible"]) for x in logs)
        stale_selected_sum = sum(int(x["stale_mature_updated"]) for x in logs)
        mature_selected_sum = sum(int(x["mature_updated"]) for x in logs)
        row = {k: s.get(k) for k in (
            "seed", "growth", "allocation", "mean_psnr", "final_psnr", "fixed_eval_mean_psnr",
            "fixed_eval_mean_depth_l1", "mean_stale_mature_coverage", "mean_mature_coverage",
            "total_spawned", "final_gaussians", "mean_opt_stage_sync_ms",
            "mean_frame_wall_sync_ms", "p95_frame_wall_sync_ms", "peak_vram_mb",
            "stale_quota_compliance")}
        row["weighted_mature_update_coverage"] = mature_selected_sum / mature_eligible_sum if mature_eligible_sum else 0.0
        row["starvation_prevalence"] = stale_eligible_sum / mature_eligible_sum if mature_eligible_sum else 0.0
        row["stale_candidate_update_coverage"] = stale_selected_sum / stale_eligible_sum if stale_eligible_sum else 0.0
        arm_rows.append(row)

    windows = []
    for s in complete:
        for growth in ("unthrottled", "throttled"):
            for allocation in ("global", "stale_rescue"):
                logs = by_key[(s, growth, allocation)]["frames"]
                last_frame = max(int(x["frame"]) for x in logs) if logs else 0
                for lo, hi in ((1, 30), (31, 60), (61, 90), (91, last_frame)):
                    rows = [x for x in logs if lo <= x["frame"] <= hi]
                    if rows:
                        eligible_n = sum(int(x["mature_eligible"]) for x in rows)
                        stale_n = sum(int(x["stale_mature_eligible"]) for x in rows)
                        windows.append({"seed": s, "growth": growth, "allocation": allocation,
                            "frames": [lo, hi],
                            "mean_psnr": float(np.mean([x["psnr"] for x in rows])),
                            "mean_stale_mature_coverage": float(np.mean([x["stale_mature_update_coverage"] for x in rows])),
                            "starvation_prevalence": stale_n / eligible_n if eligible_n else 0.0,
                            "mean_growth_pressure": float(np.mean([x["growth_pressure"] for x in rows])),
                            "mean_gaussians": float(np.mean([x["n_gaussians"] for x in rows]))})

    invalid = []
    for r in records_complete:
        s = r["summary"]
        bad = [x["frame"] for x in r["frames"]
               if not np.isfinite(x["psnr"]) or int(x["n_micro_updates"]) != 5000
               or x["frame_wall_sync_ms"] <= 0 or x["opt_stage_sync_ms"] <= 0]
        expected_n = int(s["n_processed_frames"])
        observed_frames = [int(x["frame"]) for x in r["frames"]]
        if len(r["frames"]) != expected_n or observed_frames != list(range(1, expected_n + 1)) or bad:
            invalid.append({"seed": s["seed"], "growth": s["growth"],
                            "allocation": s["allocation"], "frames": len(r["frames"]),
                            "bad_frames": bad})

    analysis = {"status": "exploratory one-seed feasibility pilot; exposed sequence",
                "n_complete_seeds": len(complete), "complete_seeds": complete,
                "n_arms": len(records_complete), "invalid_runs": invalid,
                "contrasts": contrasts, "arm_summaries": arm_rows,
                "window_summaries": windows,
                "interpretation_limit": "A trajectory factorial can show whether throttling changes stale exposure and whether rescue modifies quality, but this one-scene pilot is not a mediation or generalization test."}
    (args.input / "analysis.json").write_text(json.dumps(analysis, indent=2))

    lines = ["# Continuous growth × stale-mature rescue factorial pilot", "",
             "Exploratory feasibility result on an exposed sequence (119 online frames after initialization). Equal-work arms use 5,000 micro-updates/frame; `frame wall` uses explicit CUDA synchronization.", "",
             f"Complete seed clusters: {len(complete)}; arms analyzed: {len(records_complete)}; protocol-invalid runs: {len(invalid)}.", "",
             "## Arm results", "",
             "| Seed | Growth | Allocation | Mean PSNR | Final PSNR | Fixed-view PSNR | Starvation prevalence | Stale-candidate update coverage | Spawned | Final map | Opt stage ms | Frame ms | p95 frame ms | VRAM MB | Quota compliance |",
             "|--:|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for x in arm_rows:
        lines.append(f"| {x['seed']} | {x['growth']} | {x['allocation']} | {x['mean_psnr']:.4f} | {x['final_psnr']:.4f} | {x['fixed_eval_mean_psnr']:.4f} | {x['starvation_prevalence']:.4f} | {x['stale_candidate_update_coverage']:.4f} | {x['total_spawned']} | {x['final_gaussians']} | {x['mean_opt_stage_sync_ms']:.2f} | {x['mean_frame_wall_sync_ms']:.2f} | {x['p95_frame_wall_sync_ms']:.2f} | {x['peak_vram_mb']:.1f} | {x['stale_quota_compliance']} |")
    lines += ["", "## Seed-level contrasts", "",
              "Positive `unthrottled − throttled` means unthrottled growth improved mean PSNR under global selection. Positive rescue interaction means stale rescue helped more under unthrottled growth.", "",
              "| Contrast | Mean (dB) | Seed-bootstrap 95% CI |", "|:--|--:|:--|"]
    for name, values in contrasts.items():
        label = name.replace("_", " ")
        ci = values["seed_bootstrap_ci95"]
        ci_text = "not estimable (one seed)" if ci[0] is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        mean_text = "NA" if values["mean"] is None else f"{values['mean']:+.4f}"
        lines.append(f"| {label} | {mean_text} | {ci_text} |")
    lines += ["", "Fixed-view quality contrasts:", ""]
    for name in ("fixed_view_rescue_effect_unthrottled_psnr_db",
                 "fixed_view_rescue_effect_throttled_psnr_db",
                 "fixed_view_growth_by_rescue_interaction_psnr_db"):
        values = contrasts[name]
        ci = values["seed_bootstrap_ci95"]
        ci_text = "not estimable (one seed)" if ci[0] is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        lines.append(f"- {name.replace('_', ' ')}: {values['mean']:+.4f} dB; {ci_text}.")
    lines += ["", "## 30-frame trajectory windows", "",
              "| Seed | Growth | Allocation | Frames | Mean PSNR | Starvation prevalence | Stale-candidate update coverage | Growth pressure | Mean map size |",
              "|--:|:--|:--|:--|--:|--:|--:|--:|"]
    for x in windows:
        lines.append(f"| {x['seed']} | {x['growth']} | {x['allocation']} | {x['frames'][0]}–{x['frames'][1]} | {x['mean_psnr']:.4f} | {x['starvation_prevalence']:.4f} | {x['mean_stale_mature_coverage']:.4f} | {x['mean_growth_pressure']:.5f} | {x['mean_gaussians']:.0f} |")
    lines += ["", "This pilot is descriptive. A positive effect of stale rescue does not establish mediation unless growth changes stale exposure and the rescue attenuates the growth-associated quality change. No held-out, statistical, or real-time claim follows from one seed and one scene."]
    (args.input / "report.md").write_text("\n".join(lines) + "\n")
    print(args.input / "report.md")


if __name__ == "__main__":
    main()
