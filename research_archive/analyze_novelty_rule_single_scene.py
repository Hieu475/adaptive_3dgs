#!/usr/bin/env python3
"""Analyze one preregistered novelty-rule sequence as a scene-level test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sign_test_p(values: list[float]) -> float:
    nonzero = [x for x in values if x != 0]
    if not nonzero:
        return 1.0
    n = len(nonzero)
    k = min(sum(x > 0 for x in nonzero), sum(x < 0 for x in nonzero))
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def bootstrap_ci(values: list[float], draws: int, seed: int) -> list[float]:
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(draws, len(a)), replace=True).mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", default="results/legacy/phase14_generalization_postfix/continuous_age_quota/novelty_rule_transfer/fr3_structure_texture_near/confirmatory")
    ap.add_argument("--scene", default="tum_fr3_structure_texture_near")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(130, 138)))
    ap.add_argument("--expected_frames", type=int, default=500)
    ap.add_argument("--draws", type=int, default=50000)
    ap.add_argument("--bootstrap_seed", type=int, default=76127)
    a = ap.parse_args()
    run = (ROOT / a.run_dir).resolve()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("scenes") != [a.scene] or manifest.get("seeds") != a.seeds:
        raise ValueError("Scene or seeds differ from the frozen protocol")
    if manifest.get("arms") != ["global_importance", "mature75", "novelty_rule"]:
        raise ValueError("Arm set/order differs from the frozen protocol")

    records: dict[tuple[int, str], dict] = {}
    run_hashes: dict[str, str] = {}
    for seed in a.seeds:
        for arm in manifest["arms"]:
            path = run / a.scene / f"seed{seed}_{arm}.json"
            payload = json.loads(path.read_text())
            summary, rows = payload["summary"], payload["frames"]
            if (summary["scene"], summary["seed"], summary["arm"]) != (a.scene, seed, arm):
                raise ValueError(f"Run identity mismatch: {path}")
            if summary["n_frames_including_init"] != a.expected_frames or len(rows) != a.expected_frames - 1:
                raise ValueError(f"Frame-count gate failed: {path}")
            if not summary["update_count_gate_pass"] or any(row["n_micro_updates"] != 5000 for row in rows):
                raise ValueError(f"Update-count gate failed: {path}")
            if any(not all(math.isfinite(float(row[key])) for key in ("psnr", "ssim", "depth_l1", "frame_wall_ms")) for row in rows):
                raise ValueError(f"Nonfinite metric: {path}")
            if arm == "novelty_rule":
                for row in rows:
                    expected = 0.25 if row["new_surface_ratio_voxel_5cm"] < 0.06 else 0.75
                    if abs(row["requested_mature_fraction"] - expected) > 1e-9:
                        raise ValueError(f"Frozen rule mismatch: {path}, frame {row['frame']}")
            records[(seed, arm)] = payload
            run_hashes[str(path.relative_to(ROOT))] = sha256(path)

    primary, fixed75, mature_global = [], [], []
    seed_rows = []
    for seed in a.seeds:
        g = records[(seed, "global_importance")]["summary"]
        f = records[(seed, "mature75")]["summary"]
        r = records[(seed, "novelty_rule")]["summary"]
        for arm in ("mature75", "novelty_rule"):
            other = records[(seed, arm)]["frames"]
            base = records[(seed, "global_importance")]["frames"]
            if [x["frame"] for x in other] != [x["frame"] for x in base]:
                raise ValueError(f"Paired frame-index mismatch for seed {seed}")
        p = r["mean_psnr"] - g["mean_psnr"]
        q = r["mean_psnr"] - f["mean_psnr"]
        m = f["mean_psnr"] - g["mean_psnr"]
        primary.append(p)
        fixed75.append(q)
        mature_global.append(m)
        novelty_rows = records[(seed, "novelty_rule")]["frames"]
        seed_rows.append({
            "seed": seed,
            "novelty_minus_global_mean_psnr_db": p,
            "novelty_minus_fixed75_mean_psnr_db": q,
            "fixed75_minus_global_mean_psnr_db": m,
            "novelty_minus_global_mean_depth_l1": r["mean_depth_l1"] - g["mean_depth_l1"],
            "novelty_minus_global_mean_ssim": r["mean_ssim"] - g["mean_ssim"],
            "novelty_minus_global_final_psnr_db": r["final_psnr"] - g["final_psnr"],
            "novelty_minus_global_final_depth_l1": r["final_depth_l1"] - g["final_depth_l1"],
            "novelty_minus_global_final_gaussian_count": r["final_n_gaussians"] - g["final_n_gaussians"],
            "novelty_minus_global_mean_frame_wall_ms": r["mean_frame_wall_ms"] - g["mean_frame_wall_ms"],
            "novelty_exact_quota_compliance": r["exact_age_split_fraction"],
            "novelty_mean_requested_mature_fraction": r["mean_requested_mature_fraction"],
            "novelty_fraction_routed_to_25_percent": float(np.mean([x["requested_mature_fraction"] == 0.25 for x in novelty_rows])),
        })

    primary_ci = bootstrap_ci(primary, a.draws, a.bootstrap_seed)
    result = {
        "status": "complete; all preregistered gates passed",
        "scene": a.scene,
        "seeds": a.seeds,
        "primary_estimand": "per-seed trajectory mean post-update PSNR: novelty_rule minus global_importance",
        "primary": {
            "mean_effect_db": float(np.mean(primary)),
            "seed_bootstrap_95_ci_db": primary_ci,
            "exact_two_sided_sign_test_p": sign_test_p(primary),
            "positive_seed_pairs": sum(x > 0 for x in primary),
            "negative_seed_pairs": sum(x < 0 for x in primary),
            "ties": sum(x == 0 for x in primary),
            "worst_seed_effect_db": float(min(primary)),
            "seed_effects_db": primary,
        },
        "secondary_descriptive": {
            "novelty_rule_minus_mature75_mean_psnr_db": float(np.mean(fixed75)),
            "mature75_minus_global_mean_psnr_db": float(np.mean(mature_global)),
            "mean_novelty_minus_global_depth_l1": float(np.mean([x["novelty_minus_global_mean_depth_l1"] for x in seed_rows])),
            "mean_novelty_minus_global_ssim": float(np.mean([x["novelty_minus_global_mean_ssim"] for x in seed_rows])),
            "mean_fraction_routed_to_25_percent": float(np.mean([x["novelty_fraction_routed_to_25_percent"] for x in seed_rows])),
            "mean_exact_quota_compliance": float(np.mean([x["novelty_exact_quota_compliance"] for x in seed_rows])),
        },
        "seed_rows": seed_rows,
        "provenance": {
            "manifest_sha256": sha256(manifest_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "source_revision": manifest.get("source_revision"),
            "source_snapshot_clean_at_launch": manifest.get("source_snapshot_clean_at_launch"),
            "snapshot_commit_clean_at_launch": manifest.get("snapshot_commit_clean_at_launch"),
            "runner_sha256": manifest.get("runner_sha256"),
            "source_file_sha256": manifest.get("source_file_sha256"),
            "dataset_archive_sha256": manifest.get("dataset_archive_sha256"),
            "run_sha256": run_hashes,
        },
        "limitations": [
            "one newly accessed TUM sequence; scene is the generalization unit",
            "ground-truth poses and offline input-only novelty features; no estimated-pose or real-time routing claim",
            "equal update counts do not establish equal-wall-clock benefit",
            "gsplat CUDA backward is nondeterministic, so seeds are clustered comparisons rather than bitwise matched paths",
        ],
    }
    (run / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Preregistered frozen-novelty-rule test: TUM fr3 structure-texture-near", "",
        f"Sequence `{a.scene}`, {a.expected_frames} frames including initialization, {len(a.seeds)} seed clusters, 5,000 updates per processed frame, GT W2C, gsplat CUDA, 320×240.",
        f"Clean source revision: `{manifest.get('source_revision')}`; source clean at launch: `{manifest.get('source_snapshot_clean_at_launch')}`.", "",
        "## Primary result", "",
        "Contrast: per-seed trajectory mean PSNR, novelty rule minus global importance. The seed is the uncertainty unit.", "",
        f"Mean effect: **{np.mean(primary):+.3f} dB**; seed-bootstrap 95% CI **[{primary_ci[0]:+.3f}, {primary_ci[1]:+.3f}] dB**; exact two-sided sign test **p={sign_test_p(primary):.4f}**; {sum(x > 0 for x in primary)} positive, {sum(x < 0 for x in primary)} negative; worst seed effect **{min(primary):+.3f} dB**.", "",
        "| Seed | Rule − global PSNR (dB) | Rule − fixed75 PSNR (dB) | Rule − global depth L1 | Routed to 25% | Exact quota compliance |",
        "|--:|--:|--:|--:|--:|--:|",
    ]
    for row in seed_rows:
        lines.append(f"| {row['seed']} | {row['novelty_minus_global_mean_psnr_db']:+.3f} | {row['novelty_minus_fixed75_mean_psnr_db']:+.3f} | {row['novelty_minus_global_mean_depth_l1']:+.5f} | {row['novelty_fraction_routed_to_25_percent']:.1%} | {row['novelty_exact_quota_compliance']:.1%} |")
    lines += [
        "", "## Limits", "",
        "This is one held-out sequence, not broad scene-level generalization. Input novelty was computed offline from the complete GT-pose trajectory and does not establish online feature cost. Equal update counts are not equal-wall-clock allocation. See `manifest.json`, `analysis.json`, and the preregistration for hashes, latency summaries, and protocol.", "",
    ]
    (run / "report.md").write_text("\n".join(lines))
    print(run / "report.md")


if __name__ == "__main__":
    main()
