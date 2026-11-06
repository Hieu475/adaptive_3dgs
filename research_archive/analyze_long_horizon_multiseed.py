#!/usr/bin/env python3
"""Seed-level analysis for the preregistered 500-frame quota study."""
from __future__ import annotations

import json
import math
import argparse
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/phase14_generalization_postfix/continuous_age_quota/long_horizon_tum_fr1_desk/multiseed"
SCENE = "tum_fr1_desk"
OUT = RUN / "analysis.json"
REPORT = RUN / "report.md"
SEEDS = tuple(range(90, 98))


def read(seed, arm):
    p = RUN / SCENE / f"seed{seed}_{arm}.json"
    return json.loads(p.read_text())


def exact_sign_p(diffs):
    wins = sum(x > 0 for x in diffs)
    losses = sum(x < 0 for x in diffs)
    n = wins + losses
    if n == 0:
        return 1.0, wins, losses, len(diffs) - n
    k = min(wins, losses)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p), wins, losses, len(diffs) - n


def bootstrap_mean_ci(values, draws=10000, seed=20261018):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(a, size=(draws, len(a)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main():
    global RUN, SCENE, OUT, REPORT, SEEDS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", default=str(RUN.relative_to(ROOT)))
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    a = ap.parse_args()
    RUN = (ROOT / a.run_dir).resolve()
    SCENE = a.scene
    OUT, REPORT = RUN / "analysis.json", RUN / "report.md"
    SEEDS = tuple(a.seeds)
    manifest_path = RUN / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    source_snapshot_clean = bool(manifest.get("source_snapshot_clean_at_launch", False))
    preregistration = manifest.get("preregistration_path")
    preregistration_rel = (Path(os.path.relpath(ROOT / preregistration, RUN)).as_posix()
                           if preregistration else "preregistration.md")
    paired = {}
    for seed in SEEDS:
        g = read(seed, "global_importance")
        q = read(seed, "mature75")
        gs, qs = g["summary"], q["summary"]
        assert gs["n_frames_including_init"] == qs["n_frames_including_init"] == 500
        assert gs["update_count_gate_pass"] and qs["update_count_gate_pass"]
        gf, qf = g["frames"], q["frames"]
        assert len(gf) == len(qf) == 499
        assert all(a["frame"] == b["frame"] and a["n_micro_updates"] == b["n_micro_updates"] == 5000
                   for a, b in zip(gf, qf))
        assert all(np.isfinite(r[k]) for rows in (gf, qf) for r in rows
                   for k in ("psnr", "ssim", "depth_l1", "frame_wall_ms"))
        paired[seed] = {"global": g, "mature75": q}

    primary = [paired[s]["mature75"]["summary"]["mean_psnr"] -
               paired[s]["global"]["summary"]["mean_psnr"] for s in SEEDS]
    ci = bootstrap_mean_ci(primary)
    p_sign, wins, losses, ties = exact_sign_p(primary)
    rows = []
    for seed, d in zip(SEEDS, primary):
        g = paired[seed]["global"]["summary"]
        q = paired[seed]["mature75"]["summary"]
        rows.append({
            "seed": seed,
            "trajectory_mean_psnr_global": g["mean_psnr"],
            "trajectory_mean_psnr_mature75": q["mean_psnr"],
            "mean_psnr_delta_mature75_minus_global": d,
            "mean_depth_l1_delta_mature75_minus_global": q["mean_depth_l1"] - g["mean_depth_l1"],
            "final_psnr_delta_mature75_minus_global": q["final_psnr"] - g["final_psnr"],
            "final_depth_l1_delta_mature75_minus_global": q["final_depth_l1"] - g["final_depth_l1"],
            "final_gaussian_count_delta_mature75_minus_global": q["final_n_gaussians"] - g["final_n_gaussians"],
            "mean_wall_ms_delta_mature75_minus_global": q["mean_frame_wall_ms"] - g["mean_frame_wall_ms"],
            "quota_exact_fraction": q["exact_age_split_fraction"],
        })

    window_specs = [(1, 120), (121, 200), (201, 300), (301, 400), (401, 499)]
    windows = []
    for lo, hi in window_specs:
        seed_diffs = []
        depth_diffs = []
        for seed in SEEDS:
            g = {r["frame"]: r for r in paired[seed]["global"]["frames"]}
            q = {r["frame"]: r for r in paired[seed]["mature75"]["frames"]}
            frames = range(lo, hi + 1)
            seed_diffs.append(float(np.mean([q[i]["psnr"] - g[i]["psnr"] for i in frames])))
            depth_diffs.append(float(np.mean([q[i]["depth_l1"] - g[i]["depth_l1"] for i in frames])))
        w_p, w_wins, w_losses, w_ties = exact_sign_p(seed_diffs)
        windows.append({
            "start": lo, "end": hi,
            "mean_of_seed_mean_psnr_deltas": float(np.mean(seed_diffs)),
            "psnr_delta_seed_bootstrap_ci95": bootstrap_mean_ci(seed_diffs, seed=20261018 + lo),
            "positive_seeds": w_wins, "negative_seeds": w_losses, "ties": w_ties,
            "sign_test_two_sided_p_descriptive": w_p,
            "mean_depth_l1_delta": float(np.mean(depth_diffs)),
            "per_seed_psnr_deltas": seed_diffs,
        })
    latency_summary = {}
    for arm, key in (("global_importance", "global"), ("mature75", "mature75")):
        summaries = [paired[s][key]["summary"] for s in SEEDS]
        latency_summary[arm] = {
            metric: float(np.mean([summary[field] for summary in summaries]))
            for metric, field in (
                ("mean_frame_wall_ms", "mean_frame_wall_ms"),
                ("mean_p50_frame_wall_ms", "p50_frame_wall_ms"),
                ("mean_p95_frame_wall_ms", "p95_frame_wall_ms"),
                ("mean_p99_frame_wall_ms", "p99_frame_wall_ms"),
                ("mean_opt_stage_sync_ms", "mean_opt_stage_sync_ms"),
            )
        }
    data = {
        "study": f"exploratory_500_frame_{SCENE}_age_quota_multiseed",
        "scene": SCENE, "seeds": list(SEEDS),
        "n_frames_including_init": 500, "processed_frames_per_run": 499,
        "updates_per_frame": 5000, "all_gates_pass": True,
        "primary": {
            "endpoint": "per-seed trajectory mean PSNR difference, mature75 minus global importance",
            "seed_differences": primary,
            "mean_difference": float(np.mean(primary)),
            "seed_bootstrap_ci95": ci,
            "exact_sign_test_two_sided_p": p_sign,
            "positive_seeds": wins, "negative_seeds": losses, "ties": ties,
            "minimum_seed_effect": float(min(primary)),
        },
        "seed_rows": rows, "windows": windows,
        "latency_summary_mean_over_seeds": latency_summary,
        "limitations": [
            "one previously studied scene; exploratory only",
            "seed is uncertainty unit; 8 pairs are limited",
            "same-seed arms are not bitwise matched because gsplat backward is nondeterministic",
            "equal updates do not imply equal wall-clock; pose is ground truth",
            ("source snapshot was clean at run launch; outputs were generated after the source commit"
             if source_snapshot_clean else
             "dirty workspace; clean-checkout replication required before paper use"),
        ],
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        f"# Exploratory 500-frame quota comparison across {len(SEEDS)} seeds",
        "",
        f"{SCENE}, seeds {SEEDS[0]}–{SEEDS[-1]}, 500 frames including initialization, 5,000 updates per processed frame, GT W2C, 320×240. This is within-scene long-horizon evidence, not a held-out generalization test.",
        (f"Source snapshot revision: `{manifest.get('source_revision', 'not recorded')}` (clean at run launch)."
         if source_snapshot_clean else "Source workspace was dirty at run launch; clean-source replication is required."),
        "",
        "## Primary seed-level result",
        "",
        "Endpoint: each seed's trajectory mean PSNR difference, mature75 minus global importance. Seeds, not frames, are the uncertainty units.",
        "",
        f"Mean effect: **{np.mean(primary):+.3f} dB**; seed bootstrap 95% interval **[{ci[0]:+.3f}, {ci[1]:+.3f}] dB**; exact two-sided sign test **p={p_sign:.4f}**; {wins} positive, {losses} negative, {ties} tied seeds; smallest seed effect **{min(primary):+.3f} dB**.",
        "",
        "| Seed | Global mean PSNR | Mature75 mean PSNR | Δ PSNR | Δ depth L1 | Δ final Gaussians | Δ mean wall time (ms) | Exact quota fraction |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['trajectory_mean_psnr_global']:.3f} | {r['trajectory_mean_psnr_mature75']:.3f} | {r['mean_psnr_delta_mature75_minus_global']:+.3f} | {r['mean_depth_l1_delta_mature75_minus_global']:+.5f} | {r['final_gaussian_count_delta_mature75_minus_global']:+d} | {r['mean_wall_ms_delta_mature75_minus_global']:+.2f} | {r['quota_exact_fraction']:.1%} |")
    lines += [
        "",
        "## Frozen trajectory windows",
        "",
        "| Frames | Mean of seed PSNR deltas (dB) | Seed bootstrap 95% interval | Positive / negative seeds | Descriptive sign p | Mean depth-L1 delta |",
        "|:--|--:|--:|:--:|--:|--:|",
    ]
    for w in windows:
        ciw = w["psnr_delta_seed_bootstrap_ci95"]
        lines.append(f"| {w['start']}–{w['end']} | {w['mean_of_seed_mean_psnr_deltas']:+.3f} | [{ciw[0]:+.3f}, {ciw[1]:+.3f}] | {w['positive_seeds']} / {w['negative_seeds']} | {w['sign_test_two_sided_p_descriptive']:.4f} | {w['mean_depth_l1_delta']:+.5f} |")
    lines += [
        "",
        "## Measured latency (descriptive)",
        "",
        "Values below are averages of the per-seed measured frame latency summaries; they include the mapping frame pipeline represented by this runner. The 15 ms modeled optimization budget is not a whole-frame latency target.",
        "",
        "| Arm | Mean frame time (ms) | Mean p50 (ms) | Mean p95 (ms) | Mean p99 (ms) | Mean synchronized optimization stage (ms) |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for arm in ("global_importance", "mature75"):
        latency = latency_summary[arm]
        lines.append(f"| {arm} | {latency['mean_frame_wall_ms']:.2f} | {latency['mean_p50_frame_wall_ms']:.2f} | {latency['mean_p95_frame_wall_ms']:.2f} | {latency['mean_p99_frame_wall_ms']:.2f} | {latency['mean_opt_stage_sync_ms']:.2f} |")
    lines += [
        "",
        "Window sign tests are descriptive and not a confirmatory family; no frame-level inferential tests were run.",
        "",
        "## Limits",
        "",
        "This result covers one previously studied scene and an available 500-frame prefix. Same-seed arms do not share an identical numerical trajectory because gsplat backward accumulation is nondeterministic; the paired-seed contrast is therefore a seed-clustered comparison, not a framewise matched counterfactual. Latency summaries are descriptive and do not imply equal wall-clock budgets. GT poses do not test SLAM tracking. Source cleanliness is reported above; the dataset is symlinked from the original workspace.",
        "",
        f"See the [preregistration]({preregistration_rel}) and run manifest for protocol and provenance.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
