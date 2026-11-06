#!/usr/bin/env python3
"""Summarize frozen-prefix timing replays by independent seed."""
import json
import random
import statistics
from pathlib import Path


METRICS = {
    "fixed_heldout_mean_psnr": "Final held-out PSNR (dB)",
    "fixed_heldout_mean_depth_l1": "Final held-out Depth L1 (m)",
    "fixed_heldout_mean_ssim": "Final held-out SSIM",
    "trajectory_mean_psnr": "Trajectory mean PSNR (dB)",
    "trajectory_mean_depth_l1": "Trajectory mean Depth L1 (m)",
    "mean_mature_update_coverage": "Mean mature update coverage",
    "mean_opt_stage_sync_ms": "Mean optimization-stage time (ms)",
}
ROOT = Path("results/matched_candidate_timing_replay_fixed_snapshot")
SCENES = ("replica_room2", "replica_office3")
SEEDS = (57, 58, 59, 60)
RUN_IDS = ("repeat1", "repeat2", "repeat3")


def bootstrap_mean_ci(values, seed, samples=20000):
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(statistics.mean(values[rng.randrange(n)] for _ in range(n))
                       for _ in range(samples))
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def main():
    validation = json.loads((ROOT / "validation.json").read_text())
    if validation["valid_pairs"] != len(SCENES) * len(SEEDS) * len(RUN_IDS):
        raise RuntimeError("Not all expected scene/seed/repeat pairs passed validation")
    valid_keys = {(p["scene"], int(p["seed"]), p["run_id"])
                  for p in validation["pairs"] if p["valid"]}
    result = {"schema": "frozen-timing-cohort-v1",
              "status": "exploratory; two scenes, four independent seed prefixes per scene",
              "contrast": "same frozen 5k candidate batch inserted at frame 90 versus frame 30",
              "technical_repeats_per_seed": len(RUN_IDS),
              "inference_note": "Technical repeats are averaged within each seed; bootstrap resampling is over four independent seed/prefix realizations per scene. Intervals remain descriptive with n=4 seeds.",
              "scenes": {}}
    lines = [
        "# Frozen-prefix matched-candidate timing cohort", "",
        "Two Replica scenes × four independently built seed prefixes × three technical replay pairs per prefix. Each pair starts from a serialized checkpoint, inserts the same frozen 5,000-Gaussian batch at frame 30 or 90, disables other births, and uses the same update quota and held-out views.",
        "",
        "All 24 pairs passed checksum and treatment gates. The three replays within a seed are averaged before cross-seed inference; they are not counted as independent seeds. Intervals bootstrap four seed-level means and are descriptive.",
        "",
        "## Seed-level contrasts", "",
        "Each cell is the mean of three replay repeats. Negative Depth L1 means lower late-arm error.",
        "",
    ]
    for scene in SCENES:
        lines += ["", f"## {scene}: seed-level contrasts", "",
                  "Each row averages three replay repeats for one frozen prefix.",
                  "", "| Seed | Held-out ΔPSNR (dB) | Held-out ΔDepth L1 (m) | Held-out ΔSSIM | Trajectory ΔPSNR (dB) | Δmature coverage |",
                  "|---:|---:|---:|---:|---:|---:|"]
        by_seed = {}
        for seed in SEEDS:
            records = []
            prefix_hashes, candidate_hashes = set(), set()
            for run_id in RUN_IDS:
                if (scene, seed, run_id) not in valid_keys:
                    raise RuntimeError(f"Missing valid audit for {(scene, seed, run_id)}")
                path = ROOT / scene / f"seed{seed}_{run_id}.json"
                record = json.loads(path.read_text())
                records.append(record)
                prefix_hashes.add(record["prefix_state_sha256"])
                candidate_hashes.add(record["candidate_batch_sha256"])
            if len(prefix_hashes) != 1 or len(candidate_hashes) != 1:
                raise RuntimeError(f"Seed repeats do not share frozen artifacts: {scene} seed {seed}")
            means = {}
            run_values = {}
            for metric in (*METRICS.keys(), "final_gaussian_count_delta"):
                values = [float(r["final_gaussian_count_delta"] if metric == "final_gaussian_count_delta"
                                else r["paired_late_minus_early"][metric]) for r in records]
                means[metric] = statistics.mean(values)
                run_values[metric] = values
            by_seed[seed] = {"mean": means, "technical_repeat_values": run_values,
                             "prefix_sha256": next(iter(prefix_hashes)),
                             "candidate_batch_sha256": next(iter(candidate_hashes))}
            lines.append(
                f"| {seed} | {means['fixed_heldout_mean_psnr']:+.4f} | "
                f"{means['fixed_heldout_mean_depth_l1']:+.5f} | "
                f"{means['fixed_heldout_mean_ssim']:+.5f} | "
                f"{means['trajectory_mean_psnr']:+.4f} | "
                f"{means['mean_mature_update_coverage']:+.5f} |"
            )
        result["scenes"][scene] = {"seeds": by_seed, "mean_across_seeds": {},
                                   "bootstrap_95_ci_across_seeds": {}, "seed_signs": {}}
        lines += ["", f"### {scene}: across-seed summary", "",
                  "| Outcome | Mean across seeds [95% seed bootstrap interval] | Seed signs (+/−) |",
                  "|---|---:|---:|"]
        for metric, label in METRICS.items():
            values = [by_seed[s]["mean"][metric] for s in SEEDS]
            ci = bootstrap_mean_ci(values, 8301 + sum(map(ord, scene + metric)))
            signs = {"positive": sum(v > 0 for v in values),
                     "negative": sum(v < 0 for v in values),
                     "zero": sum(v == 0 for v in values)}
            result["scenes"][scene]["mean_across_seeds"][metric] = statistics.mean(values)
            result["scenes"][scene]["bootstrap_95_ci_across_seeds"][metric] = list(ci)
            result["scenes"][scene]["seed_signs"][metric] = signs
            lines.append(f"| {label} | {statistics.mean(values):+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}] | {signs['positive']}/{signs['negative']} |")

    result["interpretation"] = (
        "Timing effects are scene- and seed-state-dependent. Room2 tends to show "
        "positive held-out PSNR contrast but has a depth regression on seed60; office3 "
        "shows negative held-out PSNR on seeds59–60 and positive PSNR on seed57. "
        "This is not a universal late-spawn benefit. Fixed-prefix repeats are stable "
        "within most seed-scene cells, while the contrast varies across prefix states."
    )
    lines += ["", "## Interpretation and limits", "", result["interpretation"], "",
              "This experiment isolates insertion timing for a fixed candidate batch. It does not model natural discovery of later surfaces, does not prove mature starvation mediates the quality changes, uses a 5k batch rather than the prior 50k dose, and covers only two scenes. Per-scene seed count is four, so seed-bootstrap intervals are descriptive. PSNR, SSIM, depth, and trajectory-average outcomes can disagree; retain all of them in any downstream claim.",
              "", "Per-seed replay ranges and exact prefix/candidate hashes are in the adjacent `repeatability_<scene>_seed<seed>.md` reports. Pair audits are in `validation.json`."]
    out_md, out_json = ROOT / "cohort_analysis.md", ROOT / "cohort_analysis.json"
    out_md.write_text("\n".join(lines) + "\n")
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
