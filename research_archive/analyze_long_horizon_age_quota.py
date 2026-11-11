#!/usr/bin/env python3
"""Analyze the preregistered exploratory 500-frame age-quota run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/legacy/phase14_generalization_postfix/continuous_age_quota/long_horizon_tum_fr1_desk/run_seed42"
OUT = RUN / "analysis.json"
REPORT = RUN / "report.md"


def read(arm):
    return json.loads((RUN / "tum_fr1_desk" / f"seed42_{arm}.json").read_text())


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def main():
    g, q = read("global_importance"), read("mature75")
    gf, qf = g["frames"], q["frames"]
    assert len(gf) == len(qf) == 499
    assert all(a["frame"] == b["frame"] and a["n_micro_updates"] == b["n_micro_updates"] == 5000
               for a, b in zip(gf, qf))
    assert all(np.isfinite(r[k]) for rows in (gf, qf) for r in rows
               for k in ("psnr", "ssim", "depth_l1", "frame_wall_ms"))
    bounds = [(1, 30), (31, 60), (61, 90), (91, 120), (121, 200),
              (201, 300), (301, 400), (401, 499)]
    by_g = {r["frame"]: r for r in gf}
    by_q = {r["frame"]: r for r in qf}
    windows = []
    for lo, hi in bounds:
        frames = [i for i in range(lo, hi + 1) if i in by_g and i in by_q]
        if not frames:
            continue
        windows.append({
            "start": lo, "end": hi, "n_frames": len(frames),
            "mean_psnr_delta_mature75_minus_global": float(np.mean([
                by_q[i]["psnr"] - by_g[i]["psnr"] for i in frames])),
            "mean_global_psnr": float(np.mean([by_g[i]["psnr"] for i in frames])),
            "mean_mature75_psnr": float(np.mean([by_q[i]["psnr"] for i in frames])),
            "mean_global_depth_l1": float(np.mean([by_g[i]["depth_l1"] for i in frames])),
            "mean_mature75_depth_l1": float(np.mean([by_q[i]["depth_l1"] for i in frames])),
            "mean_global_gaussians": float(np.mean([by_g[i]["n_gaussians"] for i in frames])),
            "mean_mature75_gaussians": float(np.mean([by_q[i]["n_gaussians"] for i in frames])),
        })
    quota_exact = [
        r["realized_mature_updates"] == round(5000 * r["requested_mature_fraction"])
        and r["realized_young_updates"] == 5000 - round(5000 * r["requested_mature_fraction"])
        for r in qf
    ]
    dynamics = []
    for lo in range(1, 500, 20):
        hi = min(lo + 19, 499)
        frames = list(range(lo, hi + 1))
        exact = [
            by_q[i]["realized_mature_updates"] == round(5000 * by_q[i]["requested_mature_fraction"])
            and by_q[i]["realized_young_updates"] == 5000 - round(5000 * by_q[i]["requested_mature_fraction"])
            for i in frames
        ]
        dynamics.append({
            "start": lo, "end": hi,
            "mean_psnr_delta_mature75_minus_global": float(np.mean([
                by_q[i]["psnr"] - by_g[i]["psnr"] for i in frames])),
            "mean_depth_l1_delta_mature75_minus_global": float(np.mean([
                by_q[i]["depth_l1"] - by_g[i]["depth_l1"] for i in frames])),
            "mean_gaussian_count_delta_mature75_minus_global": float(np.mean([
                by_q[i]["n_gaussians"] - by_g[i]["n_gaussians"] for i in frames])),
            "exact_quota_fraction": float(np.mean(exact)),
        })
    data = {
        "study": "exploratory_long_horizon_mature_quota",
        "scene": "tum_fr1_desk", "seed": 42,
        "n_frames_including_init": 500, "processed_frames": len(gf),
        "updates_per_frame": 5000, "update_count_gate_pass": True,
        "exact_age_split_fraction": float(np.mean(quota_exact)),
        "global_summary": g["summary"], "mature75_summary": q["summary"],
        "windows": windows, "dynamics_20_frame": dynamics,
        "limitations": [
            "single seed and previously studied scene; exploratory only",
            "ground-truth poses; mapping-only",
            "equal update count, not equal wall-clock",
            "dirty workspace; clean-checkout replication required before paper use",
        ],
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "# Exploratory 500-frame mature-quota trajectory",
        "",
        "TUM fr1 desk, seed 42, 500 frames including initialization (499 processed), GT W2C, 320×240, 5,000 updates per processed frame. This is a previously studied scene and a single-seed descriptive extension, not a held-out or confirmatory test.",
        "",
        "## Trajectory summaries",
        "",
        "| Arm | Mean PSNR (dB) | Final PSNR | Mean depth L1 | Final Gaussians | Mean frame wall time (ms) | p95 wall time (ms) | Mean synchronized optimization (ms) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for label, obj in (("Global importance", g["summary"]), ("Mature 75%", q["summary"])):
        lines.append(f"| {label} | {obj['mean_psnr']:.3f} | {obj['final_psnr']:.3f} | {obj['mean_depth_l1']:.5f} | {obj['final_n_gaussians']} | {obj['mean_frame_wall_ms']:.1f} | {obj['p95_frame_wall_ms']:.1f} | {obj['mean_opt_stage_sync_ms']:.1f} |")
    lines += [
        "",
        f"Exact requested age split was realized on {np.mean(quota_exact):.1%} of mature75 frames. Both arms passed the exact 5,000-update/frame gate. Quota infeasibility on other frames is retained in the trajectory; it is not filtered out.",
        "",
        "## Fixed-window paired PSNR differences",
        "",
        "| Frames | N | Mature75 − global PSNR (dB) | Mean depth L1: global / mature75 | Mean Gaussians: global / mature75 |",
        "|:--|--:|--:|--:|--:|",
    ]
    for w in windows:
        lines.append(f"| {w['start']}–{w['end']} | {w['n_frames']} | {w['mean_psnr_delta_mature75_minus_global']:+.3f} | {w['mean_global_depth_l1']:.5f} / {w['mean_mature75_depth_l1']:.5f} | {w['mean_global_gaussians']:.0f} / {w['mean_mature75_gaussians']:.0f} |")
    lines += [
        "",
        "## Post hoc 20-frame dynamics",
        "",
        "This finer breakdown was added after inspecting the preregistered coarse windows; it is exploratory and descriptive.",
        "",
        "| Frames | ΔPSNR (dB) | Δdepth L1 | ΔGaussian count | Exact quota fraction |",
        "|:--|--:|--:|--:|--:|",
    ]
    for w in dynamics:
        lines.append(f"| {w['start']}–{w['end']} | {w['mean_psnr_delta_mature75_minus_global']:+.3f} | {w['mean_depth_l1_delta_mature75_minus_global']:+.5f} | {w['mean_gaussian_count_delta_mature75_minus_global']:+.0f} | {w['exact_quota_fraction']:.0%} |")
    lines += [
        "",
        "In frames 401–420 the quota arm's PSNR difference is already strongly negative (−4.251 dB per-frame window mean) while its map is still smaller by about 5,881 Gaussians on average. The quota map becomes larger than global around frames 441–460, after that quality decline has begun. This temporal ordering argues against the simple explanation that a larger quota map initiated the decline; it does not identify another cause. The 20-frame partition was chosen post hoc, and the pronounced single-seed deltas need independent replication.",
        "",
        "## Interpretation limits",
        "",
        "Window deltas describe one paired trajectory only; no p-values or seed-level uncertainty are appropriate here. Equal update count does not imply equal wall-clock: measured latency is reported separately. The run uses GT poses and does not evaluate SLAM tracking. The trajectory extends only as far as frame 499 and does not answer revisit/forgetting behavior beyond that prefix. Source files were dirty during execution, so these numbers require clean-checkout replication before paper use.",
        "",
        "See the [preregistration](../preregistration.md) and run manifest for frozen protocol and provenance.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
