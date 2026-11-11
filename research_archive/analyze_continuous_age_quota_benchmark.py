#!/usr/bin/env python3
"""Analyze the preregistered continuous mature-quota trajectory cohort."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/legacy/phase14_generalization_postfix/continuous_age_quota/confirmatory"
SCENES = ("replica_room2", "tum_fr1_desk")
SEEDS = tuple(range(42, 50))
ARMS = ("global_importance", "mature75")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quantile(xs, q):
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap_ci(xs, rng_seed, n=50000):
    rng = random.Random(rng_seed)
    boot = [sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n)]
    return [quantile(boot, .025), quantile(boot, .975)]


def sign_test_p(xs):
    xs = [x for x in xs if x != 0]
    n = len(xs)
    if n == 0:
        return 1.0
    k = min(sum(x > 0 for x in xs), sum(x < 0 for x in xs))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def main():
    manifest_path = BASE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["scenes"] != list(SCENES) or manifest["seeds"] != list(SEEDS):
        raise ValueError("Manifest scene/seed set does not match preregistration")
    records = {}
    record_hashes = {}
    for scene in SCENES:
        for seed in SEEDS:
            for arm in ARMS:
                path = BASE / scene / f"seed{seed}_{arm}.json"
                rec = json.loads(path.read_text())
                summary = rec["summary"]
                if (summary["scene"], summary["seed"], summary["arm"]) != (scene, seed, arm):
                    raise ValueError(f"Identity mismatch: {path}")
                if not summary["update_count_gate_pass"]:
                    raise ValueError(f"Update count gate failed: {path}")
                if len(rec["frames"]) != summary["n_frames_including_init"] - 1:
                    raise ValueError(f"Frame count mismatch: {path}")
                if any(x["n_micro_updates"] != 5000 for x in rec["frames"]):
                    raise ValueError(f"Per-frame update count mismatch: {path}")
                records[(scene, seed, arm)] = rec
                record_hashes[str(path.relative_to(ROOT))] = sha256(path)

    scene_rows = []
    trajectory_windows = [(1, 30), (31, 60), (61, 90), (91, 119)]
    window_rows = []
    raw_ps = []
    for scene in SCENES:
        diffs = {key: [] for key in ("mean_psnr", "mean_ssim", "mean_depth_l1",
                                     "final_psnr", "final_depth_l1", "final_n_gaussians",
                                     "mean_frame_wall_ms", "p95_frame_wall_ms", "p99_frame_wall_ms",
                                     "mean_opt_stage_sync_ms")}
        seed_rows = []
        quota_fractions = []
        for seed in SEEDS:
            base = records[(scene, seed, "global_importance")]["summary"]
            treat = records[(scene, seed, "mature75")]["summary"]
            row = {"seed": seed}
            for key in diffs:
                val = treat[key] - base[key]
                diffs[key].append(val)
                row[f"delta_{key}"] = val
            row["quota_exact_fraction"] = treat["exact_age_split_fraction"]
            quota_fractions.append(treat["exact_age_split_fraction"])
            seed_rows.append(row)
        raw = sign_test_p(diffs["mean_psnr"])
        raw_ps.append(raw)
        scene_rows.append({
            "scene": scene,
            "n_seed_pairs": len(SEEDS),
            "mean_psnr_delta_db": sum(diffs["mean_psnr"]) / len(SEEDS),
            "mean_psnr_delta_95_seed_bootstrap_ci_db": bootstrap_ci(diffs["mean_psnr"], 12042 + len(scene)),
            "positive_seed_pairs": sum(x > 0 for x in diffs["mean_psnr"]),
            "negative_seed_pairs": sum(x < 0 for x in diffs["mean_psnr"]),
            "worst_seed_psnr_regression_db": min(diffs["mean_psnr"]),
            "raw_exact_two_sided_sign_p": raw,
            "secondary_mean_deltas": {
                key: {"mean": sum(v) / len(v), "95_seed_bootstrap_ci": bootstrap_ci(v, 13000 + i + len(scene))}
                for i, (key, v) in enumerate(diffs.items()) if key != "mean_psnr"
            },
            "mean_exact_age_quota_fraction": sum(quota_fractions) / len(quota_fractions),
            "seed_pairs": seed_rows,
        })
        for lo, hi in trajectory_windows:
            seed_window_deltas, seed_depth_deltas, compliance = [], [], []
            for seed in SEEDS:
                ref_frames = records[(scene, seed, "global_importance")]["frames"]
                age_frames = records[(scene, seed, "mature75")]["frames"]
                paired = [(a, b) for a, b in zip(ref_frames, age_frames)
                          if lo <= a["frame"] <= hi]
                if not paired:
                    raise ValueError(f"No frames in window {lo}-{hi} for {scene}/{seed}")
                seed_window_deltas.append(sum(b["psnr"] - a["psnr"] for a, b in paired) / len(paired))
                seed_depth_deltas.append(sum(b["depth_l1"] - a["depth_l1"] for a, b in paired) / len(paired))
                exact = [b["realized_mature_updates"] == 3750 and b["realized_young_updates"] == 1250
                         for _, b in paired]
                compliance.append(sum(exact) / len(exact))
            window_rows.append({
                "scene": scene, "frame_window": [lo, hi], "n_seed_clusters": len(SEEDS),
                "mean_psnr_delta_db": sum(seed_window_deltas) / len(SEEDS),
                "seed_cluster_bootstrap_95_ci_db": bootstrap_ci(seed_window_deltas, 24000 + lo + len(scene)),
                "positive_seed_clusters": sum(x > 0 for x in seed_window_deltas),
                "mean_depth_l1_delta_m": sum(seed_depth_deltas) / len(SEEDS),
                "mean_exact_quota_compliance": sum(compliance) / len(SEEDS),
                "seed_psnr_deltas_db": dict(zip(map(str, SEEDS), seed_window_deltas)),
            })
    adjusted = holm(raw_ps)
    for row, p_adj in zip(scene_rows, adjusted):
        row["holm_adjusted_p"] = p_adj
    result = {
        "status": "complete; preregistered gates passed",
        "interpretation": "continuous constant age-quota effect under matched update count; not equal wall-clock budget and not a router test",
        "pilot_excluded": True,
        "scene_results": scene_rows,
        "descriptive_trajectory_windows": window_rows,
        "valid_trajectory_pairs": len(SCENES) * len(SEEDS),
        "arms_per_pair": 2,
        "frames_per_arm": {scene: records[(scene, SEEDS[0], "global_importance")]["summary"]["n_frames_including_init"] for scene in SCENES},
        "provenance": {
            "manifest_sha256": sha256(manifest_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "source_revision": manifest["source_revision"],
            "runner_sha256": manifest["runner_sha256"],
            "source_file_sha256": manifest["source_file_sha256"],
            "record_sha256": record_hashes,
        },
    }
    (BASE / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Continuous mature-quota trajectory result", "",
        "Eight paired seeds per scene; 120 frames including initialization (119 processed frames). Both arms performed exactly 5,000 Gaussian micro-updates per processed frame. Pilot runs are excluded.", "",
        "| Scene | Valid seed pairs | Mean trajectory PSNR delta, 75%-mature minus global (dB; 95% seed CI) | Positive / negative seeds | Worst seed delta | Exact sign p | Holm p | Mean exact quota compliance | Mean depth-L1 delta (m) | Mean frame-wall delta (ms) |",
        "|:--|--:|--:|:--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in scene_rows:
        ci = row["mean_psnr_delta_95_seed_bootstrap_ci_db"]
        sec = row["secondary_mean_deltas"]
        lines.append(
            f"| `{row['scene']}` | {row['n_seed_pairs']}/8 | {row['mean_psnr_delta_db']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{row['positive_seed_pairs']}/{row['negative_seed_pairs']} | {row['worst_seed_psnr_regression_db']:+.3f} dB | "
            f"{row['raw_exact_two_sided_sign_p']:.4f} | {row['holm_adjusted_p']:.4f} | "
            f"{row['mean_exact_age_quota_fraction']:.1%} | {sec['mean_depth_l1']['mean']:+.5f} | "
            f"{sec['mean_frame_wall_ms']['mean']:+.2f} |"
        )
    lines += [
        "", "## Descriptive trajectory windows", "",
        "Seed is still the aggregation unit. These windows are exploratory descriptions; no frame-wise or window-wise p-values are reported.", "",
        "| Scene | Frames | Mean PSNR delta, treatment minus reference (dB; 95% seed CI) | Positive seed clusters | Mean depth-L1 delta (m) | Exact quota compliance |",
        "|:--|:--:|--:|:--:|--:|--:|",
    ]
    for row in window_rows:
        ci = row["seed_cluster_bootstrap_95_ci_db"]
        lo, hi = row["frame_window"]
        lines.append(
            f"| `{row['scene']}` | {lo}–{hi} | {row['mean_psnr_delta_db']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{row['positive_seed_clusters']}/8 | {row['mean_depth_l1_delta_m']:+.5f} | "
            f"{row['mean_exact_quota_compliance']:.1%} |"
        )
    lines += [
        "", "## Secondary trajectory metrics (treatment minus reference)", "",
        "| Scene | Mean SSIM delta | Mean depth-L1 delta (m) | Final PSNR delta (dB) | Final Gaussian count delta | Mean synchronized optimization-time delta (ms/frame) | Mean p95 frame-wall delta (ms) |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for row in scene_rows:
        sec = row["secondary_mean_deltas"]
        lines.append(
            f"| `{row['scene']}` | {sec['mean_ssim']['mean']:+.5f} | {sec['mean_depth_l1']['mean']:+.5f} | "
            f"{sec['final_psnr']['mean']:+.3f} | {sec['final_n_gaussians']['mean']:+.1f} | "
            f"{sec['mean_opt_stage_sync_ms']['mean']:+.3f} | {sec['p95_frame_wall_ms']['mean']:+.2f} |"
        )
    lines += [
        "", "## Interpretation", "",
        "The estimand is the continuous trajectory effect of requesting a 75% mature / 25% young split with capacity transfer, compared with global importance allocation, at the same number of optimizer updates. The exact age split was not feasible on every frame; compliance is reported above and per seed in `analysis.json`.", "",
        "The fixed update count does not imply matched wall-clock cost. Read the paired frame-wall and synchronized optimization-time deltas as measured system outcomes. Scene results remain separate; two scenes do not establish broad generalization. This is not evidence for a state-conditioned controller or for equal-latency performance.", "",
        "All seed-level effects, secondary endpoints, input hashes, and source hashes are in `analysis.json`.", "",
    ]
    (BASE / "report.md").write_text("\n".join(lines))
    print(BASE / "report.md")


if __name__ == "__main__":
    main()
