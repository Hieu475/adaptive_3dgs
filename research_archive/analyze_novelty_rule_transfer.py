#!/usr/bin/env python3
"""Analyze the preregistered frozen-novelty-rule held-out trajectory cohort."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/continuous_age_quota/novelty_rule_transfer/confirmatory"
SCENES = ("replica_room0", "tum_fr1_room")
SEEDS = tuple(range(42, 50))
ARMS = ("global_importance", "mature75", "novelty_rule")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mean(xs):
    return sum(xs) / len(xs)


def quantile(xs, q):
    xs = sorted(xs)
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] + (xs[hi] - xs[lo]) * (p - lo)


def bootstrap_ci(xs, seed, n=50000):
    rng = random.Random(seed)
    vals = [mean([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n)]
    return [quantile(vals, .025), quantile(vals, .975)]


def sign_test_p(xs):
    xs = [x for x in xs if x != 0]
    if not xs:
        return 1.0
    k = min(sum(x > 0 for x in xs), sum(x < 0 for x in xs))
    return min(1.0, 2 * sum(math.comb(len(xs), i) for i in range(k + 1)) / (2 ** len(xs)))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, running)
    return out


def main():
    manifest_path = BASE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["scenes"] != list(SCENES) or manifest["seeds"] != list(SEEDS):
        raise ValueError("Scene or seed set differs from preregistered cohort")
    if manifest["arms"] != list(ARMS):
        raise ValueError("Arm set differs from preregistered cohort")
    records, hashes = {}, {}
    for scene in SCENES:
        for seed in SEEDS:
            for arm in ARMS:
                p = BASE / scene / f"seed{seed}_{arm}.json"
                rec = json.loads(p.read_text())
                s = rec["summary"]
                if (s["scene"], s["seed"], s["arm"]) != (scene, seed, arm):
                    raise ValueError(f"Identity mismatch: {p}")
                if not s["update_count_gate_pass"]:
                    raise ValueError(f"Update-count gate failed: {p}")
                rows = rec["frames"]
                if not rows or any(r["n_micro_updates"] != 5000 for r in rows):
                    raise ValueError(f"Per-frame update mismatch: {p}")
                if arm == "novelty_rule":
                    for r in rows:
                        expected = .25 if r["new_surface_ratio_voxel_5cm"] < .06 else .75
                        if abs(r["requested_mature_fraction"] - expected) > 1e-9:
                            raise ValueError(f"Frozen-rule compliance failure: {p}, frame {r['frame']}")
                records[(scene, seed, arm)] = rec
                hashes[str(p.relative_to(ROOT))] = sha256(p)

    scene_results, primary_ps = [], []
    for scene in SCENES:
        primary = {k: [] for k in ("mean_psnr", "mean_depth_l1", "mean_ssim", "final_psnr",
                                    "final_depth_l1", "final_n_gaussians", "mean_frame_wall_ms",
                                    "p95_frame_wall_ms", "p99_frame_wall_ms", "mean_opt_stage_sync_ms")}
        secondary = {k: [] for k in ("mean_psnr", "mean_depth_l1", "mean_ssim", "final_psnr",
                                      "final_depth_l1", "final_n_gaussians", "mean_frame_wall_ms",
                                      "mean_opt_stage_sync_ms")}
        seed_rows = []
        rule_compliance, fixed_compliance, mean_requested = [], [], []
        novelty_branch_fractions = []
        for seed in SEEDS:
            base = records[(scene, seed, "global_importance")]["summary"]
            fixed = records[(scene, seed, "mature75")]["summary"]
            rule = records[(scene, seed, "novelty_rule")]["summary"]
            sr = {"seed": seed}
            for k in primary:
                primary[k].append(rule[k] - base[k])
                sr[f"rule_minus_global_{k}"] = rule[k] - base[k]
            for k in secondary:
                secondary[k].append(rule[k] - fixed[k])
                sr[f"rule_minus_fixed75_{k}"] = rule[k] - fixed[k]
            rule_compliance.append(rule["exact_age_split_fraction"])
            fixed_compliance.append(fixed["exact_age_split_fraction"])
            mean_requested.append(rule["mean_requested_mature_fraction"])
            rows = records[(scene, seed, "novelty_rule")]["frames"]
            novelty_branch_fractions.append(sum(r["requested_mature_fraction"] == .25 for r in rows) / len(rows))
            sr["rule_quota_exact_fraction"] = rule["exact_age_split_fraction"]
            sr["rule_mean_requested_mature_fraction"] = rule["mean_requested_mature_fraction"]
            seed_rows.append(sr)
        p = sign_test_p(primary["mean_psnr"])
        primary_ps.append(p)
        scene_results.append({
            "scene": scene,
            "actual_frames_including_initialization": records[(scene, SEEDS[0], "global_importance")]["summary"]["n_frames_including_init"],
            "n_seed_clusters": len(SEEDS),
            "primary_rule_minus_global": {
                "mean_psnr_delta_db": mean(primary["mean_psnr"]),
                "seed_bootstrap_95_ci_db": bootstrap_ci(primary["mean_psnr"], 76001 + len(scene)),
                "positive_seed_pairs": sum(x > 0 for x in primary["mean_psnr"]),
                "negative_seed_pairs": sum(x < 0 for x in primary["mean_psnr"]),
                "worst_seed_psnr_delta_db": min(primary["mean_psnr"]),
                "exact_two_sided_sign_p": p,
                "secondary_mean_deltas": {
                    k: {"mean": mean(v), "seed_bootstrap_95_ci": bootstrap_ci(v, 76100 + i + len(scene))}
                    for i, (k, v) in enumerate(primary.items()) if k != "mean_psnr"
                },
            },
            "secondary_descriptive_rule_minus_fixed75": {
                k: {"mean": mean(v), "seed_bootstrap_95_ci": bootstrap_ci(v, 77100 + i + len(scene))}
                for i, (k, v) in enumerate(secondary.items())
            },
            "mean_rule_exact_quota_compliance": mean(rule_compliance),
            "mean_fixed75_exact_quota_compliance": mean(fixed_compliance),
            "mean_requested_mature_fraction_under_rule": mean(mean_requested),
            "mean_fraction_frames_routed_to_25_percent": mean(novelty_branch_fractions),
            "seed_pairs": seed_rows,
        })
    adjusted = holm(primary_ps)
    for r, adj in zip(scene_results, adjusted):
        r["primary_rule_minus_global"]["holm_adjusted_p"] = adj

    result = {
        "status": "complete; preregistered gates passed",
        "pilot_excluded": True,
        "primary_estimand": "continuous frozen novelty rule minus global-importance trajectory mean PSNR under exactly matched 5000 micro-updates per frame",
        "frozen_rule": manifest["novelty_rule"],
        "scene_results": scene_results,
        "provenance": {
            "manifest_sha256": sha256(manifest_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "source_revision": manifest["source_revision"],
            "runner_sha256": manifest["runner_sha256"],
            "source_file_sha256": manifest["source_file_sha256"],
            "record_sha256": hashes,
        },
        "limitations": [
            "Only two held-out mapping scenes; no broad generalization claim.",
            "Novel-surface ratio uses ground-truth poses and offline input-only point accumulation; not tested with estimated poses or real-time feature computation.",
            "Equal micro-update counts do not imply equal wall-clock cost.",
            "The calibration TUM fr1 desk results are not pooled with the held-out evaluation.",
        ],
    }
    (BASE / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Frozen novelty-rule transfer result", "",
        "The rule was calibrated on TUM fr1 desk, then frozen before these held-out scene runs. Each scene has eight seed clusters; all arms used exactly 5,000 updates per processed frame. Replica room0 contains 100 available frames, while TUM fr1 room uses 120. The engineering pilots are excluded.", "",
        "| Scene | Frames incl. init | Rule − global mean trajectory PSNR (dB; 95% seed CI) | Positive / negative seeds | Worst seed delta | Exact sign p | Holm p | Rule quota compliance | Rule routed to 25% | Rule − fixed 75% PSNR (descriptive) | Rule − global depth L1 (m) |",
        "|:--|--:|--:|:--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in scene_results:
        p = r["primary_rule_minus_global"]
        ci = p["seed_bootstrap_95_ci_db"]
        fixed_ci = r["secondary_descriptive_rule_minus_fixed75"]["mean_psnr"]["seed_bootstrap_95_ci"]
        lines.append(
            f"| `{r['scene']}` | {r['actual_frames_including_initialization']} | {p['mean_psnr_delta_db']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{p['positive_seed_pairs']}/{p['negative_seed_pairs']} | {p['worst_seed_psnr_delta_db']:+.3f} | "
            f"{p['exact_two_sided_sign_p']:.4f} | {p['holm_adjusted_p']:.4f} | "
            f"{r['mean_rule_exact_quota_compliance']:.1%} | {r['mean_fraction_frames_routed_to_25_percent']:.1%} | "
            f"{r['secondary_descriptive_rule_minus_fixed75']['mean_psnr']['mean']:+.3f} [{fixed_ci[0]:+.3f}, {fixed_ci[1]:+.3f}] | "
            f"{p['secondary_mean_deltas']['mean_depth_l1']['mean']:+.5f} |"
        )
    lines += [
        "", "## Interpretation", "",
        "Primary inference is only the frozen rule versus global importance, separately by scene, with Holm correction across two tests. The fixed-75% comparison is descriptive. Compliance is intent-to-treat: if one age group lacks capacity, the scheduler transfers unused quota while preserving the update count.", "",
        "The rule routes only 28.3% of Replica room0 frames and 3.4% of TUM fr1 room frames to 25% mature; in practice, it behaves mostly like the fixed-75% arm. Rule-minus-fixed-75% PSNR is +0.010 dB and -0.038 dB, respectively, as descriptive contrasts. The primary gains over global importance therefore do not show that the novelty switch adds value over a constant quota.", "",
        "The rule uses policy-independent input novelty computed from ground-truth poses. This checks mapping-only transfer; it does not establish a deployable real-time router, estimated-pose performance, or equal-wall-time benefit. Do not revise the threshold from these held-out outcomes.", "",
        "Per-seed effects, measured runtime, quota compliance, exact hashes, and limits are in `analysis.json`.", "",
    ]
    (BASE / "report.md").write_text("\n".join(lines))
    print(BASE / "report.md")


if __name__ == "__main__":
    main()
