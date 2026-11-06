#!/usr/bin/env python3
"""Seed-level analysis of within-frame mature revisit/residual strata."""
import argparse
import json
from pathlib import Path

import numpy as np


BINS = ("staleness_0_2_frames", "staleness_3_5_frames",
        "staleness_6_9_frames", "staleness_10_19_frames",
        "staleness_20_plus_frames")


def bootstrap(values, seed=20261021, n=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (n, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def load_blocks(root):
    blocks = []
    for path in sorted(root.glob("*/seed*_frame*.json")):
        record = json.loads(path.read_text())
        if not record.get("all_count_and_snapshot_gates_passed"):
            continue
        for arm in record["arms"]:
            if not arm.get("valid"):
                continue
            action = arm["action"]
            frame_rows = []
            for row in arm["horizon_frame_metrics"]:
                strata = ((row.get("age_stratified_diagnostics") or {})
                          .get("mature_revisit_strata") or {})
                if not strata:
                    continue
                current = {name: strata.get(name) for name in BINS}
                low = current[BINS[0]]
                high = current[BINS[-1]]
                if not low or not high:
                    continue
                if int(low.get("visible_attributed", 0)) == 0 or int(high.get("visible_attributed", 0)) == 0:
                    continue
                frame_rows.append({
                    "frame": int(row["frame"]),
                    "rgb_residual_old_minus_recent": float(high["visible_color_error_mean"]) - float(low["visible_color_error_mean"]),
                    "depth_residual_old_minus_recent": float(high["visible_depth_error_mean"]) - float(low["visible_depth_error_mean"]),
                    "visible_fraction_old_minus_recent": float(high["visible_fraction_attributed"]) - float(low["visible_fraction_attributed"]),
                    "old_staleness_minus_recent": float(high["mean_staleness_frames"]) - float(low["mean_staleness_frames"]),
                    "bins": current,
                })
            if len(frame_rows) < 10:
                continue
            blocks.append({
                "scene": record["scene"], "seed": int(record["seed"]),
                "snapshot_frame": int(record["snapshot_frame"]),
                "spawn_count": int(action["spawn_count"]),
                "allocation": action["allocation"],
                "stale_fraction": action.get("stale_fraction") if action["allocation"] == "stale_rescue" else None,
                "target_opt_stage_ms": arm.get("target_opt_stage_ms"),
                "frames": frame_rows,
            })
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    args = ap.parse_args()
    blocks = load_blocks(args.input)
    if not blocks:
        raise SystemExit(f"No valid blocks with revisit strata found in {args.input}")
    scene_names = sorted({b["scene"] for b in blocks})
    summaries = {}
    seed_rows = {}
    for block in blocks:
        label = (f"spawn{block['spawn_count']}_" +
                 ("global" if block["allocation"] == "global" else
                  f"stale{int(round(float(block['stale_fraction'])*100))}"))
        key = (block["scene"], block["snapshot_frame"], label)
        values = {}
        for metric in ("rgb_residual_old_minus_recent", "depth_residual_old_minus_recent",
                       "visible_fraction_old_minus_recent", "old_staleness_minus_recent"):
            values[metric] = float(np.mean([r[metric] for r in block["frames"]]))
        values["valid_frames"] = len(block["frames"])
        seed_rows.setdefault(key, {})[str(block["seed"])] = values
    for key, rows in seed_rows.items():
        summaries["/".join(map(str, key))] = {}
        for metric in ("rgb_residual_old_minus_recent", "depth_residual_old_minus_recent",
                       "visible_fraction_old_minus_recent", "old_staleness_minus_recent"):
            vals = [row[metric] for row in rows.values()]
            summaries["/".join(map(str, key))][metric] = {
                "seed_mean": float(np.mean(vals)),
                "seed_bootstrap_ci95": bootstrap(vals),
                "per_seed": dict(zip(rows.keys(), vals)),
            }

    paired = {}
    groupings = {}
    for key in seed_rows:
        groupings.setdefault((key[0], key[1], key[2].split("_")[0]), []).append(key)
    for (scene, frame, growth_label), keys in groupings.items():
        dose_keys = {key[2]: key for key in keys}
        global_key = next((key for key in keys if key[2].endswith("global")), None)
        if global_key is None:
            continue
        for key in keys:
            if key == global_key:
                continue
            common_seeds = sorted(set(seed_rows[global_key]) & set(seed_rows[key]))
            contrast_name = f"{scene}_frame{frame}_{key[2]}_minus_global"
            metric_contrasts = {}
            for metric in ("rgb_residual_old_minus_recent", "depth_residual_old_minus_recent",
                           "visible_fraction_old_minus_recent"):
                diffs = [seed_rows[key][s][metric] - seed_rows[global_key][s][metric]
                         for s in common_seeds]
                metric_contrasts[metric] = {
                    "mean_paired_difference": float(np.mean(diffs)),
                    "seed_bootstrap_ci95": bootstrap(diffs),
                    "per_seed": dict(zip(common_seeds, diffs)),
                }
            paired[contrast_name] = metric_contrasts

    result = {
        "status": "exploratory seed-level association; scene/snapshot exposed",
        "scenes": scene_names,
        "valid_arm_seed_blocks": len(blocks),
        "aggregation": "Within each seed-arm, average paired within-frame bin contrasts across eligible horizon frames; bootstrap seed blocks only.",
        "definition": "Old-minus-recent contrasts compare mature visible Gaussians last updated 20+ frames ago with mature visible Gaussians updated 0–2 frames ago in the same frame.",
        "arm_summaries": summaries,
        "paired_dose_minus_global": paired,
        "interpretation_limit": "This controls camera/frame by comparing strata within the same frame, but identities and scene content differ across strata and arms. The association is not a per-Gaussian causal effect; residual is measured before current-frame optimization.",
    }
    (args.input / "revisit_residual_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Same-frame mature revisit and residual association", "",
        f"Exploratory, exposed sequences: {scene_names}. Valid arm-seed blocks with at least 10 paired frames: {len(blocks)}.",
        "Within every frame, compare visible mature Gaussians last updated 20+ frames ago against visible mature Gaussians updated 0–2 frames ago. Then average within seed-arm; confidence intervals resample seeds, not frames. This pairs scene/camera/time within the contrast.", "",
        "## Arm summaries", "",
        "| Scene/snapshot/arm | Old−recent RGB residual | Old−recent depth residual | Old−recent visible fraction | Old−recent staleness (frames) |",
        "|:--|--:|--:|--:|--:|"]
    for key, values in summaries.items():
        rgb = values["rgb_residual_old_minus_recent"]
        depth = values["depth_residual_old_minus_recent"]
        vis = values["visible_fraction_old_minus_recent"]
        stale = values["old_staleness_minus_recent"]
        lines.append(f"| {key} | {rgb['seed_mean']:+.5f} [{rgb['seed_bootstrap_ci95'][0]:+.5f}, {rgb['seed_bootstrap_ci95'][1]:+.5f}] | {depth['seed_mean']:+.5f} [{depth['seed_bootstrap_ci95'][0]:+.5f}, {depth['seed_bootstrap_ci95'][1]:+.5f}] | {vis['seed_mean']:+.3f} | {stale['seed_mean']:.1f} |")
    lines += ["", "## Dose minus global paired changes in the age-residual gradient", "",
        "Negative residual differences mean the 20+ frame-visible group has lower residual relative to the 0–2 frame group after the dose than under global allocation.", "",
        "| Contrast | RGB residual gradient change | Depth residual gradient change | Visibility gradient change |",
        "|:--|--:|--:|--:|"]
    for key, metrics in paired.items():
        vals = []
        for metric in ("rgb_residual_old_minus_recent", "depth_residual_old_minus_recent",
                       "visible_fraction_old_minus_recent"):
            item = metrics[metric]
            vals.append(f"{item['mean_paired_difference']:+.5f} [{item['seed_bootstrap_ci95'][0]:+.5f}, {item['seed_bootstrap_ci95'][1]:+.5f}]")
        lines.append(f"| {key} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines += ["", "Interpretation: staleness is not a direct proxy for residual magnitude. Within-frame matching removes camera/frame confounding, but strata remain different Gaussian populations, and treatment changes which identities stay visible in each bin. The association is descriptive and cannot identify the causal quality return of updating a particular Gaussian.", ""]
    (args.input / "revisit_residual_report.md").write_text("\n".join(lines))
    print(args.input / "revisit_residual_report.md")


if __name__ == "__main__":
    main()
