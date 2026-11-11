#!/usr/bin/env python3
"""Summarize independently preregistered mature-quota snapshot replays."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/legacy/phase14_generalization_postfix"
SCENES = {
    "replica_office1": "office1_mature_quota_replay",
    "tum_fr1_rpy": "tum_fr1_rpy_mature_quota_replay",
}


def main() -> None:
    analyses = {scene: json.loads((BASE / folder / "analysis.json").read_text())
                for scene, folder in SCENES.items()}
    raw_p = {scene: analyses[scene]["contrasts"][
        "primary_zero_spawn_mature75_minus_mature25_db"]["exact_two_sided_sign_test_p"]
        for scene in SCENES}
    holm = {}
    running = 0.0
    ranked = sorted(raw_p.items(), key=lambda x: x[1])
    for i, (scene, p) in enumerate(ranked):
        running = max(running, min(1.0, p * (len(ranked) - i)))
        holm[scene] = running

    rows = []
    for scene, folder in SCENES.items():
        data = analyses[scene]
        c = data["contrasts"]["primary_zero_spawn_mature75_minus_mature25_db"]
        t = data["contrasts"]["secondary_opt_stage_time_mature75_minus_mature25_ms_at_zero_spawn"]
        d = data["contrasts"]["secondary_depth_l1_mature75_minus_mature25_m_at_zero_spawn"]
        low = data["arm_summaries"]["spawn0_mature25"]
        high = data["arm_summaries"]["spawn0_mature75"]
        rows.append({
            "scene": scene,
            "n_valid": data["valid_seed_snapshots"],
            "gain25_db": low["mean_final_map_psnr_gain_db"],
            "gain75_db": high["mean_final_map_psnr_gain_db"],
            "delta_db": c["mean"],
            "ci_low": c["bootstrap_ci95_seed"][0],
            "ci_high": c["bootstrap_ci95_seed"][1],
            "positive_seeds": c["positive_seeds"],
            "raw_p": raw_p[scene],
            "holm_p": holm[scene],
            "delta_depth_m": d["mean"],
            "delta_sync_ms": t["mean"],
            "sync_ci_low": t["bootstrap_ci95_seed"][0],
            "sync_ci_high": t["bootstrap_ci95_seed"][1],
            "muc25": low["mean_mature_update_coverage"],
            "muc75": high["mean_mature_update_coverage"],
            "gaussians25": low["mean_n_gaussians_after_action"],
            "gaussians75": high["mean_n_gaussians_after_action"],
        })
    lines = [
        "# Mature-update quota replay: cross-scene result",
        "",
        "The two scenes were evaluated in separate preregistered eight-seed same-snapshot replays. Each primary contrast is 75% minus 25% mature allocation at zero spawn; the map snapshot, Gaussian count, RNG state, reference render and 5,000 Gaussian micro-updates are paired within seed. Holm correction covers the two scene-level primary tests.",
        "",
        "| Scene | Valid snapshots | PSNR gain at 25% (dB) | PSNR gain at 75% (dB) | Paired quota effect (dB; seed CI) | Positive seeds | Raw p | Holm p | Depth-L1 change (m) | Sync time change (ms; seed CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['scene']}` | {r['n_valid']}/8 | {r['gain25_db']:+.3f} | {r['gain75_db']:+.3f} | "
            f"{r['delta_db']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
            f"{r['positive_seeds']}/8 | {r['raw_p']:.4f} | {r['holm_p']:.4f} | "
            f"{r['delta_depth_m']:+.5f} | {r['delta_sync_ms']:+.3f} "
            f"[{r['sync_ci_low']:+.3f}, {r['sync_ci_high']:+.3f}] |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "At these two frame-90 OURS snapshots, a higher mature quota improves the next-frame final-map render under both Replica and TUM, and the effect is positive in all eight seeds per scene. Exact Gaussian cardinality and update count are held fixed within each spawn-dose pair. The scene-level Holm-adjusted sign-test p-values remain below 0.05.",
        "",
        "The higher quota also costs additional synchronized optimization time. The intervention therefore establishes an immediate quality-versus-time tradeoff at fixed update count, not at fixed measured latency. The snapshots were selected from known regression scenes, and the replay covers only one action/frame per seed. This does not yet establish that continuously increasing mature updates prevents long-horizon regression or generalizes to unseen scenes.",
        "",
        "Per-scene preregistrations, unit records, analyses and provenance are in `office1_mature_quota_replay/` and `tum_fr1_rpy_mature_quota_replay/`.",
        "",
    ]
    (BASE / "mature_quota_cross_scene_report.md").write_text("\n".join(lines))
    (BASE / "mature_quota_cross_scene_summary.json").write_text(json.dumps({
        "status": "exploratory; two preregistered local replay tests",
        "primary_test_family": "two scene-level exact sign tests; Holm correction",
        "scenes": rows,
        "interpretation_limit": "Immediate one-frame effects at frame 90 only; not long-horizon mediation or generalization evidence.",
    }, indent=2) + "\n")
    print(BASE / "mature_quota_cross_scene_report.md")


if __name__ == "__main__":
    main()
