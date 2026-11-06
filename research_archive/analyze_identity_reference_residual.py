#!/usr/bin/env python3
"""Aggregate fixed-view persistent-ID residual endpoints by seed.

IDs, frames, and reference views are repeated measurements; uncertainty is
computed by resampling seed blocks only.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def endpoint_cells(arm):
    before = {int(x["frame"]): x.get("identity_reference_residual")
              for x in arm.get("fixed_eval_before_action", [])}
    after = {int(x["frame"]): x.get("identity_reference_residual")
             for x in arm.get("fixed_eval_after_horizon", [])}
    cells = {}
    n_pre = n_post = n_pair = 0
    for frame in sorted(set(before) & set(after)):
        pre, post = before[frame], after[frame]
        if not pre or not post:
            continue
        if pre["persistent_ids"] != post["persistent_ids"]:
            raise ValueError(f"persistent-ID order mismatch at reference frame {frame}")
        for i, pid in enumerate(pre["persistent_ids"]):
            pre_vis = bool(pre["present"][i] and pre["visible"][i])
            post_vis = bool(post["present"][i] and post["visible"][i])
            n_pre += pre_vis
            n_post += post_vis
            key = (frame, int(pid))
            if pre_vis and post_vis:
                n_pair += 1
                cells[key] = {
                    "rgb": float(post["rgb_center_abs_error"][i] - pre["rgb_center_abs_error"][i]),
                    "depth": float(post["depth_center_abs_error"][i] - pre["depth_center_abs_error"][i]),
                }
    return cells, {"visible_before_id_view": n_pre,
                   "visible_after_id_view": n_post,
                   "visible_paired_within_arm": n_pair}


def mean_field(cells, field):
    values = [v[field] for v in cells.values()]
    return float(np.mean(values)) if values else None


def bootstrap_mean(values, seed=20261021, n_boot=10000):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {"mean": None, "ci95": [None, None], "n_seeds": 0}
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return {"mean": float(x.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "n_seeds": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="Root directory containing per-scene seed*_frame*.json records")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    records = []
    for path in sorted(args.input.rglob("seed*_frame*.json")):
        data = json.loads(path.read_text())
        arms = {}
        for arm in data["arms"]:
            if not arm.get("valid"):
                continue
            key = (int(arm["action"]["spawn_count"]), arm["action"]["allocation"])
            cells, counts = endpoint_cells(arm)
            arms[key] = {"cells": cells, "counts": counts,
                         "psnr_change": float(arm["fixed_eval_psnr_gain_db"]),
                         "depth_l1_change": float(arm["fixed_eval_depth_l1_change"])}
        valid = bool(data.get("all_count_and_snapshot_gates_passed")) and len(arms) == 4
        records.append({"path": str(path), "scene": data["scene"], "seed": int(data["seed"]),
                        "snapshot_frame": int(data["snapshot_frame"]),
                        "valid_block": valid, "arms": arms})

    by_scene = {}
    for scene in sorted({r["scene"] for r in records}):
        scene_records = [r for r in records if r["scene"] == scene and r["valid_block"]]
        scene_out = {"valid_seed_blocks": sorted(r["seed"] for r in scene_records),
                     "invalid_records": [r["path"] for r in records
                                         if r["scene"] == scene and not r["valid_block"]],
                     "growth_conditions": {}}
        for growth in (0, 3584):
            paired_rows = []
            contrast_rgb, contrast_depth, contrast_psnr = [], [], []
            for rec in scene_records:
                global_arm = rec["arms"].get((growth, "global"))
                rescue_arm = rec["arms"].get((growth, "stale_rescue"))
                if global_arm is None or rescue_arm is None:
                    continue
                common = set(global_arm["cells"]) & set(rescue_arm["cells"])
                rgb_delta = [rescue_arm["cells"][k]["rgb"] - global_arm["cells"][k]["rgb"]
                             for k in common]
                depth_delta = [rescue_arm["cells"][k]["depth"] - global_arm["cells"][k]["depth"]
                               for k in common]
                paired_rows.append({
                    "seed": rec["seed"],
                    "common_visible_id_views": len(common),
                    "global_visible_paired_within_arm": global_arm["counts"]["visible_paired_within_arm"],
                    "stale_visible_paired_within_arm": rescue_arm["counts"]["visible_paired_within_arm"],
                    "global_rgb_change": mean_field(global_arm["cells"], "rgb"),
                    "stale_rgb_change": mean_field(rescue_arm["cells"], "rgb"),
                    "global_depth_change": mean_field(global_arm["cells"], "depth"),
                    "stale_depth_change": mean_field(rescue_arm["cells"], "depth"),
                    "common_support_rgb_rescue_minus_global": float(np.mean(rgb_delta)) if rgb_delta else None,
                    "common_support_depth_rescue_minus_global": float(np.mean(depth_delta)) if depth_delta else None,
                    "psnr_change_global_db": global_arm["psnr_change"],
                    "psnr_change_stale_db": rescue_arm["psnr_change"],
                    "psnr_rescue_minus_global_db": rescue_arm["psnr_change"] - global_arm["psnr_change"],
                })
                if rgb_delta:
                    contrast_rgb.append(float(np.mean(rgb_delta)))
                    contrast_depth.append(float(np.mean(depth_delta)))
                contrast_psnr.append(rescue_arm["psnr_change"] - global_arm["psnr_change"])
            scene_out["growth_conditions"][str(growth)] = {
                "per_seed": paired_rows,
                "common_support_rgb_rescue_minus_global": bootstrap_mean(contrast_rgb, seed=20261021 + growth),
                "common_support_depth_rescue_minus_global": bootstrap_mean(contrast_depth, seed=20262021 + growth),
                "fixed_view_psnr_rescue_minus_global_db": bootstrap_mean(contrast_psnr, seed=20263021 + growth),
            }
        by_scene[scene] = scene_out

    output = args.output or args.input / "identity_reference_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"analysis": "fixed_view_persistent_id_center_residual",
              "claim_boundary": "center-sampled residual; descriptive; common-support contrast conditions on post-treatment visibility; seed is the resampling unit",
              "scenes": by_scene}
    output.write_text(json.dumps(result, indent=2))
    lines = ["# Fixed-view persistent-ID residual analysis", "",
             "Exploratory descriptive analysis. Residual is sampled at each Gaussian's projected center on the same fixed GT camera view. It is not contribution-weighted attribution. Common-support contrasts condition on post-treatment visibility. Seed blocks, not IDs/views, are resampled.", ""]
    for scene, sdata in by_scene.items():
        lines += [f"## {scene}", "", f"Valid seed blocks: {sdata['valid_seed_blocks']}", ""]
        for growth, gdata in sdata["growth_conditions"].items():
            def fmt(item):
                mean = item["mean"]
                ci = item["ci95"]
                return "n/a" if mean is None else f"{mean:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}], n={item['n_seeds']}"
            lines += [f"### Births: {growth}", "",
                      f"- Stale-rescue minus global RGB residual change, common visible ID-view support: {fmt(gdata['common_support_rgb_rescue_minus_global'])}",
                      f"- Stale-rescue minus global depth residual change, common visible ID-view support: {fmt(gdata['common_support_depth_rescue_minus_global'])}",
                      f"- Fixed-view PSNR change, stale-rescue minus global: {fmt(gdata['fixed_view_psnr_rescue_minus_global_db'])}",
                      "- Per-seed denominators and arm-level changes are in the companion JSON.", ""]
    report = output.with_suffix(".md")
    report.write_text("\n".join(lines))
    print(report)
    print(output)


if __name__ == "__main__":
    main()
