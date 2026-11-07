#!/usr/bin/env python3
"""Exploratory frame-level diagnostics for the room1 mature-quota trajectory run."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_frozen_benchmark import load_benchmark_sequence
from experiments.run_postfix_frame_diagnostics import scene_features


RUN = ROOT / "results/phase14_generalization_postfix/room1_trajectory_age_quota_seeds270_272"
SCENE = "replica_room1"
SEEDS = (270, 271, 272)
W, H = 320, 240


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    return float(statistics.mean(values)) if values else None


def pearson(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


def interval_stats(rows, lo, hi):
    selected = rows[lo:hi]
    return {
        "frame_first": selected[0]["frame"],
        "frame_last": selected[-1]["frame"],
        "n_frames": len(selected),
        "delta_psnr_mean": mean([r["delta_psnr_75_minus_25"] for r in selected]),
        "delta_depth_l1_mean": mean([r["delta_depth_l1_75_minus_25"] for r in selected]),
        "delta_mature_coverage_mean": mean([r["delta_mature_coverage_75_minus_25"] for r in selected]),
    }


def main():
    frames, intrinsics = load_benchmark_sequence(
        scene_name=SCENE, n_frames=120, H=H, W=W, device="cpu"
    )
    static = scene_features(frames, intrinsics)
    all_rows = []
    per_seed = {}

    for seed in SEEDS:
        arms = {}
        for arm in ("mature25", "mature75"):
            path = RUN / SCENE / f"seed{seed}_{arm}.json"
            arms[arm] = json.loads(path.read_text())
        lo, hi = arms["mature25"]["frames"], arms["mature75"]["frames"]
        if [r["frame"] for r in lo] != [r["frame"] for r in hi]:
            raise RuntimeError(f"Frame index mismatch for seed {seed}")
        if len(lo) != len(frames) - 1:
            raise RuntimeError(f"Expected {len(frames)-1} replay frames for seed {seed}, got {len(lo)}")

        rows = []
        for a, b in zip(lo, hi):
            idx = int(a["frame"])
            row = {
                "seed": seed,
                "frame": idx,
                "delta_psnr_75_minus_25": float(b["psnr"] - a["psnr"]),
                "delta_ssim_75_minus_25": float(b["ssim"] - a["ssim"]),
                "delta_depth_l1_75_minus_25": float(b["depth_l1"] - a["depth_l1"]),
                "mature25_psnr": float(a["psnr"]),
                "mature75_psnr": float(b["psnr"]),
                "mature25_depth_l1": float(a["depth_l1"]),
                "mature75_depth_l1": float(b["depth_l1"]),
                "mature25_update_coverage": float(a["mature_update_coverage"]),
                "mature75_update_coverage": float(b["mature_update_coverage"]),
                "delta_mature_coverage_75_minus_25": float(
                    b["mature_update_coverage"] - a["mature_update_coverage"]
                ),
                "mature25_realized_update_share": float(a["realized_mature_updates"] / max(1, a["n_micro_updates"])),
                "mature75_realized_update_share": float(b["realized_mature_updates"] / max(1, b["n_micro_updates"])),
                "delta_growth_pressure_75_minus_25": float(b["growth_pressure"] - a["growth_pressure"]),
                "delta_gaussian_count_75_minus_25": int(b["n_gaussians"] - a["n_gaussians"]),
                "mature25_opt_stage_sync_ms": float(a["opt_stage_sync_ms"]),
                "mature75_opt_stage_sync_ms": float(b["opt_stage_sync_ms"]),
            }
            row.update(static[idx])
            rows.append(row)
            all_rows.append(row)

        per_seed[seed] = {
            "arm_summaries": {arm: arms[arm]["summary"] for arm in ("mature25", "mature75")},
            "intervals": {
                "first_10": interval_stats(rows, 0, 10),
                "frames_11_to_99": interval_stats(rows, 10, 99),
                "last_20": interval_stats(rows, 99, 119),
            },
            "realized_mature_share_after_frame_10": {
                "mature25": mean([r["mature25_realized_update_share"] for r in rows if r["frame"] >= 10]),
                "mature75": mean([r["mature75_realized_update_share"] for r in rows if r["frame"] >= 10]),
            },
            "correlations_across_frames_descriptive_only": {
                feature: pearson(
                    [r[feature] for r in rows],
                    [r["delta_psnr_75_minus_25"] for r in rows],
                )
                for feature in (
                    "new_surface_ratio_voxel_5cm",
                    "camera_translation_m_per_frame",
                    "camera_angular_deg_per_frame",
                    "depth_valid_ratio_sampled",
                    "delta_mature_coverage_75_minus_25",
                    "delta_growth_pressure_75_minus_25",
                    "delta_gaussian_count_75_minus_25",
                )
            },
        }

    novelty = np.asarray([r["new_surface_ratio_voxel_5cm"] for r in all_rows], dtype=float)
    q1, q2 = [float(x) for x in np.quantile(novelty, [1/3, 2/3])]
    bins = {"low": [], "mid": [], "high": []}
    for r in all_rows:
        v = r["new_surface_ratio_voxel_5cm"]
        bins["low" if v <= q1 else "mid" if v <= q2 else "high"].append(r)
    novelty_results = {
        "definition": "Tertiles of policy-independent 5cm new-surface ratio, pooled across the three seed-aligned input trajectories.",
        "cutpoints": [q1, q2],
        "bins": {},
    }
    for name, items in bins.items():
        by_seed = {}
        for seed in SEEDS:
            local = [r for r in items if r["seed"] == seed]
            by_seed[seed] = {
                "n_frames": len(local),
                "delta_psnr_mean": mean([r["delta_psnr_75_minus_25"] for r in local]),
                "delta_depth_l1_mean": mean([r["delta_depth_l1_75_minus_25"] for r in local]),
                "delta_mature_coverage_mean": mean([r["delta_mature_coverage_75_minus_25"] for r in local]),
            }
        novelty_results["bins"][name] = {
            "n_rows": len(items),
            "mean_seed_delta_psnr": mean([x["delta_psnr_mean"] for x in by_seed.values() if x["delta_psnr_mean"] is not None]),
            "mean_seed_delta_depth_l1": mean([x["delta_depth_l1_mean"] for x in by_seed.values() if x["delta_depth_l1_mean"] is not None]),
            "per_seed": by_seed,
        }

    csv_path = RUN / "frame_diagnostics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    payload = {
        "status": "exploratory descriptive analysis; frame-level rows are correlated repeated measures",
        "protocol": "phase14-room1-age-quota-frame-diagnostics-v1",
        "scene": SCENE,
        "seeds": list(SEEDS),
        "input_pose_convention": "Phase 10 Replica ground-truth extrinsics using the pipeline convention",
        "scene_feature_source": "experiments/run_postfix_frame_diagnostics.py::scene_features",
        "scene_features_are_policy_independent": True,
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "run_manifest_sha256": sha256(RUN / "manifest.json"),
        "preregistration_sha256": sha256(RUN / "preregistration.json"),
        "per_seed": per_seed,
        "novelty_tertile_analysis": novelty_results,
        "frame_csv": csv_path.name,
    }
    json_path = RUN / "trajectory_diagnostics.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    lines = [
        "# Exploratory frame diagnostics: Replica room1 age-quota trajectory", "",
        "Input-only camera, depth, and 5cm surface-novelty features were joined to the already completed mature25/mature75 trajectories. No mapping run was repeated. This analysis is descriptive: the policies produce diverging maps, and frame observations are serially correlated.", "",
        "## Per-seed timing and quality summary", "",
        "| Seed | Mean PSNR Δ | Final PSNR Δ | First 10 PSNR Δ | Frames 11–99 PSNR Δ | Last 20 PSNR Δ | Mature share after frame 10 (25%, 75%) |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        x = per_seed[seed]
        s25 = x["arm_summaries"]["mature25"]
        s75 = x["arm_summaries"]["mature75"]
        lines.append(
            f"| {seed} | {s75['mean_psnr']-s25['mean_psnr']:+.4f} dB | "
            f"{s75['final_psnr']-s25['final_psnr']:+.4f} dB | "
            f"{x['intervals']['first_10']['delta_psnr_mean']:+.4f} dB | "
            f"{x['intervals']['frames_11_to_99']['delta_psnr_mean']:+.4f} dB | "
            f"{x['intervals']['last_20']['delta_psnr_mean']:+.4f} dB | "
            f"{x['realized_mature_share_after_frame_10']['mature25']:.1%}, "
            f"{x['realized_mature_share_after_frame_10']['mature75']:.1%} |"
        )
    lines += ["", "## Quality by policy-independent surface novelty", "",
              f"Novelty tertile cutpoints were {q1:.4f} and {q2:.4f}. The table averages per-seed frame means so no seed is given extra weight.", "",
              "| Novelty bin | Frames | Mean per-seed PSNR Δ | Mean per-seed depth L1 Δ |",
              "|---|---:|---:|---:|"]
    for name in ("low", "mid", "high"):
        b = novelty_results["bins"][name]
        lines.append(f"| {name} | {b['n_rows']} | {b['mean_seed_delta_psnr']:+.4f} dB | {b['mean_seed_delta_depth_l1']:+.6f} m |")
    lines += ["", "## Within-seed correlations", "",
              "Pearson correlations below are computed separately within each 119-frame seed trajectory between frame PSNR difference (mature75−mature25) and the listed feature difference/value. They are descriptive only; no frame-level p-values are used.", "",
              "| Feature | Seed 270 | Seed 271 | Seed 272 | Mean across seeds |",
              "|---|---:|---:|---:|---:|"]
    feature_labels = {
        "new_surface_ratio_voxel_5cm": "Input new-surface ratio",
        "camera_translation_m_per_frame": "Camera translation",
        "camera_angular_deg_per_frame": "Camera angular change",
        "depth_valid_ratio_sampled": "Valid-depth ratio",
        "delta_mature_coverage_75_minus_25": "Difference in mature update coverage",
        "delta_growth_pressure_75_minus_25": "Difference in growth pressure",
        "delta_gaussian_count_75_minus_25": "Difference in map size",
    }
    for feature, label in feature_labels.items():
        vals = [per_seed[s]["correlations_across_frames_descriptive_only"][feature] for s in SEEDS]
        shown = ["NA" if v is None else f"{v:+.3f}" for v in vals]
        valid = [v for v in vals if v is not None]
        average = "NA" if not valid else f"{statistics.mean(valid):+.3f}"
        lines.append(f"| {label} | {shown[0]} | {shown[1]} | {shown[2]} | {average} |")
    lines += ["", "## Interpretation", "",
              "Use these summaries to nominate snapshots for a controlled replay. Do not interpret the correlations as causal: each policy's map state has already diverged, scene features are temporally correlated, and novelty tertiles summarize one fixed path. The observed trajectory benefit is concentrated near the end; this analysis checks whether that timing aligns with measured novelty, motion, coverage, or map-growth differences.", "",
              "Artifacts: `trajectory_diagnostics.json` and `frame_diagnostics.csv`."]
    (RUN / "trajectory_diagnostics_report.md").write_text("\n".join(lines) + "\n")
    print(RUN / "trajectory_diagnostics_report.md")


if __name__ == "__main__":
    main()
