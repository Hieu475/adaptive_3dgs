#!/usr/bin/env python3
"""Summarize 50k Replica E2 timing contrasts across proposal policies."""
import json
import random
import statistics
from pathlib import Path


BASE = Path("results/legacy/phase14_generalization_postfix")
SCENES = ("replica_room2", "replica_office3")
SEEDS = (57, 58, 59, 60)
ROOTS = {
    "uniform_depth": {
        57: "replica_e2_timing_clean_checkout_fixed50k_room2_office3_seeds57_58",
        58: "replica_e2_timing_clean_checkout_fixed50k_room2_office3_seeds57_58",
        59: "replica_e2_timing_clean_checkout_uniform50k_seeds59_60",
        60: "replica_e2_timing_clean_checkout_uniform50k_seeds59_60",
    },
    "error_first": {
        57: "replica_e2_timing_clean_checkout_errorfirst50k_room2_office3_seeds57_58",
        58: "replica_e2_timing_clean_checkout_errorfirst50k_room2_office3_seeds57_58",
        59: "replica_e2_timing_clean_checkout_errorfirst50k_seeds59_60",
        60: "replica_e2_timing_clean_checkout_errorfirst50k_seeds59_60",
    },
}
METRICS = {
    "mean_psnr": "mean_psnr",
    "final_psnr": "final_psnr",
    "mean_ssim": "mean_ssim",
    "mean_depth_l1": "mean_depth_l1",
}


def avg(values):
    return statistics.mean(values)


def bootstrap_mean_ci(values, *, seed, samples=20000):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(avg([values[rng.randrange(n)] for _ in range(n)]) for _ in range(samples))
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def load_pair(mode, scene, seed):
    path = BASE / ROOTS[mode][seed] / scene / f"E2_seed{seed}.json"
    pair = json.loads(path.read_text())
    if not pair.get("valid") or pair.get("target_total_spawned") != 50000:
        raise ValueError(f"Invalid or wrong-dose pair: {path}")
    if pair.get("validation", {}).get("candidate_mode") != mode:
        raise ValueError(f"Candidate mode mismatch: {path}")
    if pair["validation"].get("realized_total_spawned") != [50000, 50000]:
        raise ValueError(f"Exact spawn count failed: {path}")
    arms = {arm["condition"]: arm for arm in pair["arms"]}
    early, late = arms["early_spawn"], arms["late_spawn"]
    e, l = early["frames"], late["frames"]
    if len(e) != 119 or len(l) != 119:
        raise ValueError(f"Expected 119 frames per arm: {path}")
    delta = {key: late["summary"][field] - early["summary"][field]
             for key, field in METRICS.items()}
    fallback = {
        "early": pair["validation"]["fallback_spawned"][0] / 50000,
        "late": pair["validation"]["fallback_spawned"][1] / 50000,
    }
    split = 59
    half = {
        "early_first": avg([row["psnr"] for row in e[:split]]),
        "early_second": avg([row["psnr"] for row in e[split:]]),
        "late_first": avg([row["psnr"] for row in l[:split]]),
        "late_second": avg([row["psnr"] for row in l[split:]]),
    }
    return {
        "scene": scene, "seed": seed, "candidate_mode": mode,
        "source_commit": pair["unthrottled_spawn_pilot"]["manifest"]["git_sha"],
        "delta_late_minus_early": delta,
        "fallback_fraction": fallback,
        "half_window_mean_psnr": half,
        "artifact": str(path),
    }


def main():
    records = [load_pair(mode, scene, seed)
               for mode in ROOTS for scene in SCENES for seed in SEEDS]
    effects = {}
    for mode in ROOTS:
        effects[mode] = {}
        for scene in SCENES:
            cohort = [r for r in records if r["candidate_mode"] == mode and r["scene"] == scene]
            effects[mode][scene] = {
                "n": len(cohort),
                "mean_delta": {},
                "bootstrap_95_ci": {},
                "positive_negative_psnr": {
                    "positive": sum(r["delta_late_minus_early"]["mean_psnr"] > 0 for r in cohort),
                    "negative": sum(r["delta_late_minus_early"]["mean_psnr"] < 0 for r in cohort),
                },
                "mean_fallback_fraction": {
                    side: avg([r["fallback_fraction"][side] for r in cohort])
                    for side in ("early", "late")
                },
                "mean_half_window_psnr": {
                    k: avg([r["half_window_mean_psnr"][k] for r in cohort])
                    for k in ("early_first", "early_second", "late_first", "late_second")
                },
            }
            for metric in METRICS:
                vals = [r["delta_late_minus_early"][metric] for r in cohort]
                effects[mode][scene]["mean_delta"][metric] = avg(vals)
                effects[mode][scene]["bootstrap_95_ci"][metric] = bootstrap_mean_ci(
                    vals, seed=1700 + sum(map(ord, f"{mode}/{scene}/{metric}")))

    interactions = {}
    for scene in SCENES:
        interactions[scene] = {"n": len(SEEDS), "mean_errorfirst_minus_uniform": {}, "bootstrap_95_ci": {}}
        for metric in METRICS:
            by_seed = []
            for seed in SEEDS:
                u = next(r for r in records if r["scene"] == scene and r["seed"] == seed
                         and r["candidate_mode"] == "uniform_depth")
                e = next(r for r in records if r["scene"] == scene and r["seed"] == seed
                         and r["candidate_mode"] == "error_first")
                by_seed.append(e["delta_late_minus_early"][metric] - u["delta_late_minus_early"][metric])
            interactions[scene]["mean_errorfirst_minus_uniform"][metric] = avg(by_seed)
            interactions[scene]["bootstrap_95_ci"][metric] = bootstrap_mean_ci(
                by_seed, seed=2900 + sum(map(ord, f"{scene}/{metric}")))

    output = {
        "schema": "replica-e2-candidate-transfer-analysis-v1",
        "status": "exploratory",
        "contrast": "late spawn minus early spawn; fixed 50,000 births per pair",
        "inference_note": "Descriptive seed-level bootstrap with n=4; not preregistered and not confirmatory.",
        "effects_by_mode_and_scene": effects,
        "paired_proposal_by_timing_interaction": interactions,
        "records": records,
    }
    out_json = BASE / "replica_e2_candidate_transfer_analysis.json"
    out_md = BASE / "replica_e2_candidate_transfer_analysis.md"
    out_json.write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# Replica E2 candidate-policy transfer summary",
        "",
        "Fixed 50k births per pair; early frames 1–59 versus late frames 60–119. Deltas are late minus early. Intervals bootstrap four seed-level contrasts and are descriptive only.",
        "",
        "| Candidate mode | Scene | n | Mean ΔPSNR [95% seed bootstrap CI] | Mean Δfinal PSNR | Mean ΔDepth L1 (m) | PSNR signs +/− | Fallback births early/late |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ROOTS:
        for scene in SCENES:
            d = effects[mode][scene]
            ci = d["bootstrap_95_ci"]["mean_psnr"]
            signs = d["positive_negative_psnr"]
            fb = d["mean_fallback_fraction"]
            lines.append(
                f"| {mode} | {scene} | {d['n']} | {d['mean_delta']['mean_psnr']:+.4f} "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] dB | {d['mean_delta']['final_psnr']:+.4f} dB | "
                f"{d['mean_delta']['mean_depth_l1']:+.5f} | {signs['positive']}/{signs['negative']} | "
                f"{fb['early']:.1%}/{fb['late']:.1%} |"
            )
    lines += ["", "## Paired candidate-policy interaction", "",
              "Error-first minus uniform-depth in the per-seed late-minus-early contrast; positive values mean the timing effect is more positive under error-first.",
              "", "| Scene | Metric | Mean interaction [95% seed bootstrap CI] |", "|---|---|---:|"]
    for scene in SCENES:
        for metric in METRICS:
            d = interactions[scene]
            ci = d["bootstrap_95_ci"][metric]
            lines.append(f"| {scene} | {metric} | {d['mean_errorfirst_minus_uniform'][metric]:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
    lines += ["", "## Interpretation", "",
              "Read mean trajectory PSNR, final-frame PSNR, and Depth L1 together. Candidate mode changes the size of the late-versus-early contrast and the fallback rate; it is a material part of the E2 treatment. A positive final-frame PSNR contrast does not erase a negative trajectory-mean contrast or depth regression.",
              "",
              "There are four seeds per scene, the bootstrap intervals are descriptive, and the source snapshots differ between the initial and added cohorts. The runs establish exact-count, clean-source replications for this protocol; they do not establish a universal timing rule or compute-dilution mediation.",
              "",
              "Raw pairs and source commits are recorded in the JSON. Initial clean cohorts: [uniform 57–58](replica_e2_timing_clean_checkout_fixed50k_room2_office3_seeds57_58/), [error-first 57–58](replica_e2_timing_clean_checkout_errorfirst50k_room2_office3_seeds57_58/). Added clean cohorts: [uniform 59–60](replica_e2_timing_clean_checkout_uniform50k_seeds59_60/), [error-first 59–60](replica_e2_timing_clean_checkout_errorfirst50k_seeds59_60/)."]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
