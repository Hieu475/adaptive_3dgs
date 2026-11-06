#!/usr/bin/env python3
"""Post-hoc, descriptive frame diagnostics for the exposed walking-half-sphere runs.

This analysis does not retune the novelty policy. It compares paired per-frame
metrics by fixed temporal windows and by median splits of input-only scene
features. The eight seeds are retained as clusters and no frame-level
significance tests are performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/phase14_generalization_postfix/continuous_age_quota/novelty_rule_transfer/fr3_walking_halfsphere/confirmatory"
SCENE = "tum_fr3_walking_halfsphere"
SEEDS = tuple(range(140, 148))
ARMS = ("global_importance", "mature75", "novelty_rule")
CONTRASTS = {
    "novelty_minus_global": ("novelty_rule", "global_importance"),
    "mature75_minus_global": ("mature75", "global_importance"),
    "novelty_minus_mature75": ("novelty_rule", "mature75"),
}
WINDOWS = tuple((lo, min(lo + 49, 499)) for lo in range(1, 500, 50))
METRICS = ("psnr", "ssim", "depth_l1", "psnr_pre", "psnr_post", "depth_l1_pre", "depth_l1_post", "psnr_update_gain", "depth_l1_update_change", "frame_wall_ms", "opt_stage_sync_ms")
FEATURES = (
    "camera_translation_m_per_frame",
    "camera_angular_deg_per_frame",
    "new_surface_ratio_voxel_5cm",
    "depth_valid_ratio_sampled",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mean(xs) -> float:
    return float(np.mean(np.asarray(xs, dtype=np.float64))) if xs else float("nan")


def metric_value(row: dict, metric: str) -> float:
    if metric == "psnr_update_gain":
        return float(row["psnr_post"]) - float(row["psnr_pre"])
    if metric == "depth_l1_update_change":
        return float(row["depth_l1_post"]) - float(row["depth_l1_pre"])
    return float(row[metric])


def main() -> None:
    manifest_path = RUN / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("scenes") != [SCENE] or manifest.get("seeds") != list(SEEDS):
        raise ValueError("Run manifest scene/seed set differs from expected study")
    records = {}
    hashes = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = RUN / SCENE / f"seed{seed}_{arm}.json"
            data = json.loads(path.read_text())
            if (data["summary"]["scene"], data["summary"]["seed"], data["summary"]["arm"]) != (SCENE, seed, arm):
                raise ValueError(f"Run identity mismatch: {path}")
            if len(data["frames"]) != 499 or not data["summary"]["update_count_gate_pass"]:
                raise ValueError(f"Run gate mismatch: {path}")
            records[seed, arm] = {int(row["frame"]): row for row in data["frames"]}
            hashes[str(path.relative_to(ROOT))] = sha256(path)

    # Features are policy-independent input/pose diagnostics. Use one copy per
    # frame, checking repeated seed logs agree before making temporal bins.
    feature_by_frame = {}
    for frame in range(1, 500):
        rows = [records[seed, "novelty_rule"][frame] for seed in SEEDS]
        feature_by_frame[frame] = {}
        for key in FEATURES:
            vals = [float(row[key]) for row in rows]
            if max(vals) - min(vals) > 1e-8:
                raise ValueError(f"Input feature {key} differs across seeds at frame {frame}")
            feature_by_frame[frame][key] = vals[0]

    frame_deltas = {}
    seed_effects = {}
    for contrast, (treatment, control) in CONTRASTS.items():
        frame_deltas[contrast] = {}
        seed_effects[contrast] = {}
        for metric in METRICS:
            frame_deltas[contrast][metric] = {}
            seed_effects[contrast][metric] = {}
        for seed in SEEDS:
            per_frame = {metric: {} for metric in METRICS}
            for frame in range(1, 500):
                tr = records[seed, treatment][frame]
                co = records[seed, control][frame]
                for metric in METRICS:
                    per_frame[metric][frame] = metric_value(tr, metric) - metric_value(co, metric)
                    frame_deltas[contrast][metric].setdefault(frame, []).append(per_frame[metric][frame])
            for metric in METRICS:
                seed_effects[contrast][metric][str(seed)] = mean(list(per_frame[metric].values()))

    temporal = []
    for lo, hi in WINDOWS:
        item = {"frames": [lo, hi], "n_frames": hi - lo + 1}
        for contrast in frame_deltas:
            item[contrast] = {}
            for metric in METRICS:
                by_seed = []
                for seed in SEEDS:
                    by_seed.append(mean([
                        metric_value(records[seed, CONTRASTS[contrast][0]][f], metric)
                        - metric_value(records[seed, CONTRASTS[contrast][1]][f], metric)
                        for f in range(lo, hi + 1)
                    ]))
                item[contrast][metric] = {
                    "mean_across_seed_trajectories": mean(by_seed),
                    "seed_values": dict(zip(map(str, SEEDS), by_seed)),
                    "positive_seed_count": sum(x > 0 for x in by_seed),
                }
        item["input_features_global_reference"] = {
            key: mean([feature_by_frame[f][key] for f in range(lo, hi + 1)]) for key in FEATURES
        }
        item["novelty_route_fraction_25pct"] = mean([
            float(records[seed, "novelty_rule"][f]["requested_mature_fraction"] == 0.25)
            for seed in SEEDS for f in range(lo, hi + 1)
        ])
        item["novelty_realized_mature_coverage"] = mean([
            float(records[seed, "novelty_rule"][f]["mature_update_coverage"])
            for seed in SEEDS for f in range(lo, hi + 1)
        ])
        temporal.append(item)

    stratified = []
    thresholds = {key: float(np.median([feature_by_frame[f][key] for f in range(1, 500)])) for key in FEATURES}
    for key in FEATURES:
        cut = thresholds[key]
        low_frames = [f for f in range(1, 500) if feature_by_frame[f][key] <= cut]
        high_frames = [f for f in range(1, 500) if feature_by_frame[f][key] > cut]
        # Depth validity is adverse when low, so label bins by observed values.
        bins = [("lower_or_equal_median", low_frames), ("above_median", high_frames)]
        for label, frames in bins:
            if not frames:
                continue
            row = {"feature": key, "median_cut": cut, "bin": label, "n_frames": len(frames)}
            row["feature_mean"] = mean([feature_by_frame[f][key] for f in frames])
            for contrast in frame_deltas:
                row[contrast] = {}
                treatment, control = CONTRASTS[contrast]
                for metric in METRICS:
                    by_seed = [mean([
                        metric_value(records[seed, treatment][f], metric) - metric_value(records[seed, control][f], metric)
                        for f in frames
                    ]) for seed in SEEDS]
                    row[contrast][metric] = {
                        "mean_across_seed_trajectories": mean(by_seed),
                        "seed_values": dict(zip(map(str, SEEDS), by_seed)),
                        "positive_seed_count": sum(x > 0 for x in by_seed),
                    }
            row["novelty_route_fraction_25pct"] = mean([
                float(records[seed, "novelty_rule"][f]["requested_mature_fraction"] == 0.25)
                for seed in SEEDS for f in frames
            ])
            stratified.append(row)

    out = {
        "status": "post-hoc descriptive diagnostics; no retuning and no frame-level inference",
        "scene": SCENE,
        "seeds": list(SEEDS),
        "windows": temporal,
        "feature_median_splits": stratified,
        "feature_medians": thresholds,
        "seed_level_full_trajectory_effects": seed_effects,
        "provenance": {
            "manifest_sha256": sha256(manifest_path),
            "record_sha256": hashes,
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
        },
        "interpretation_limits": [
            "The sequence was selected after observing another sequence outcome; this is post-hoc exploratory analysis.",
            "Frames are repeated measurements within eight trajectories; no frame-level significance tests are used.",
            "Motion, novelty, depth validity, map state and outcomes are associated observations, not causal explanations.",
            "Input features use ground-truth poses; they do not show an online feature computation budget.",
            "Map-dependent coverage and growth metrics are policy-dependent after the trajectories diverge.",
        ],
    }
    out_path = RUN / "frame_diagnostics.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    lines = [
        "# TUM fr3 walking halfsphere: post-hoc frame diagnostics", "",
        "Descriptive analysis of the exposed 500-frame, eight-seed study. No thresholds were changed. Motion/novelty/depth features come from the logged GT-pose, input-only diagnostics; outcomes are paired by seed and frame. Frames are repeated measures, so tables report seed-trajectory means without frame-level inference.", "",
        "## Fixed temporal windows", "",
        "| Frames | Novelty mean | Translation m/frame | Rotation deg/frame | Valid depth | Novelty→25% route | Rule−global PSNR dB (positive seeds) | Rule−global SSIM | Rule−global depth L1 | Rule−global update PSNR gain | Rule−global update depth change | Fixed75−global PSNR dB | Rule−fixed75 PSNR dB |",
        "|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for w in temporal:
        d = w["novelty_minus_global"]
        feat = w["input_features_global_reference"]
        lines.append(
            f"| {w['frames'][0]}–{w['frames'][1]} | {feat['new_surface_ratio_voxel_5cm']:.3f} | "
            f"{feat['camera_translation_m_per_frame']:.4f} | {feat['camera_angular_deg_per_frame']:.2f} | "
            f"{feat['depth_valid_ratio_sampled']:.3f} | {w['novelty_route_fraction_25pct']:.1%} | "
            f"{d['psnr']['mean_across_seed_trajectories']:+.3f} ({d['psnr']['positive_seed_count']}/8) | "
            f"{d['ssim']['mean_across_seed_trajectories']:+.4f} | {d['depth_l1']['mean_across_seed_trajectories']:+.4f} | "
            f"{d['psnr_update_gain']['mean_across_seed_trajectories']:+.4f} | "
            f"{d['depth_l1_update_change']['mean_across_seed_trajectories']:+.5f} | "
            f"{w['mature75_minus_global']['psnr']['mean_across_seed_trajectories']:+.3f} | "
            f"{w['novelty_minus_mature75']['psnr']['mean_across_seed_trajectories']:+.3f} |"
        )
    lines += ["", "## Feature median splits", "",
              "The split thresholds are descriptive medians over this sequence; they are not candidate policy thresholds. Each cell is the mean of eight seed-specific window means.", "",
              "| Feature | Bin | Frames | Feature mean | Rule−global PSNR dB (positive seeds) | Rule−global SSIM | Rule−global depth L1 | Fixed75−global PSNR dB | Rule−fixed75 PSNR dB | Rule route to 25% |",
              "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for row in stratified:
        d = row["novelty_minus_global"]
        lines.append(
            f"| `{row['feature']}` | {row['bin']} | {row['n_frames']} | {row['feature_mean']:.4f} | "
            f"{d['psnr']['mean_across_seed_trajectories']:+.3f} ({d['psnr']['positive_seed_count']}/8) | "
            f"{d['ssim']['mean_across_seed_trajectories']:+.4f} | {d['depth_l1']['mean_across_seed_trajectories']:+.4f} | "
            f"{row['mature75_minus_global']['psnr']['mean_across_seed_trajectories']:+.3f} | "
            f"{row['novelty_minus_mature75']['psnr']['mean_across_seed_trajectories']:+.3f} | {row['novelty_route_fraction_25pct']:.1%} |"
        )
    ng = seed_effects["novelty_minus_global"]
    lines += ["", "## Pre/post update decomposition", "",
              "Across complete trajectories, novelty minus global PSNR was "
              f"{mean(list(ng['psnr_pre'].values())):+.3f} dB before the current frame's optimization and "
              f"{mean(list(ng['psnr_post'].values())):+.3f} dB after it. The within-frame optimization gain difference was "
              f"{mean(list(ng['psnr_update_gain'].values())):+.4f} dB, slightly narrowing the average PSNR gap. "
              "For depth L1, the pre-update difference was "
              f"{mean(list(ng['depth_l1_pre'].values())):+.4f} and the post-update difference was "
              f"{mean(list(ng['depth_l1_post'].values())):+.4f}; the current update slightly narrowed that gap too. "
              "This indicates the average quality gap is already present before the current frame's refine step, while that step is not on average making the paired gap worse. It does not identify which earlier spawn/refine decisions produced the state difference.", "",
              "## Reading and limits", "",
              "Read these tables as localization clues only. Within-window feature averages and policy deltas can move together because sequence motion and mapping state evolve over time; they do not establish that motion, novelty, depth validity, or Gaussian growth caused the observed differences. In particular, novelty features are computed offline using ground-truth poses, and map-dependent metrics are post-treatment after each arm's path diverges. The seed-level values and hashes are in `frame_diagnostics.json`; do not use this analysis to retune the 0.06 rule.", ""]
    report_path = RUN / "frame_diagnostics.md"
    report_path.write_text("\n".join(lines))
    print(report_path)


if __name__ == "__main__":
    main()
