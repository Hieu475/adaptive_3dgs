#!/usr/bin/env python3
"""Aggregate matched-candidate timing replay pairs; intervals are descriptive."""
import argparse
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
    "final_gaussian_count_delta": "Final Gaussian count (late − early)",
}


def bootstrap_mean_ci(values, seed, samples=20000):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(values[rng.randrange(n)] for _ in range(n))
                   for _ in range(samples))
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("results/legacy/matched_candidate_timing_replay_verified"))
    ap.add_argument("--scenes", nargs="+", default=["replica_room2", "replica_office3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[57, 58, 59, 60])
    ap.add_argument("--same_seed_repeat_roots", nargs="*", default=[
        "results/legacy/matched_candidate_timing_replay_pilot",
        "results/legacy/matched_candidate_timing_replay_full",
        "results/legacy/matched_candidate_timing_replay_verified",
        "results/legacy/matched_candidate_timing_replay_final_smoke",
    ])
    ap.add_argument("--frozen_repeat_root", type=Path,
                    default=Path("results/legacy/matched_candidate_timing_replay_fixed_snapshot"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    validation_by_key = {}
    if validation is not None:
        validation_by_key = {
            (pair["scene"], int(pair["seed"])): pair
            for pair in validation["pairs"]
        }
    rows = []
    for scene in args.scenes:
        for seed in args.seeds:
            path = root / scene / f"seed{seed}.json"
            record = json.loads(path.read_text())
            audit = validation_by_key.get((scene, seed))
            if audit is not None and not audit["valid"]:
                raise ValueError(f"Causal-control gates failed: {path}: {audit['checks']}")
            if record["arms"][0]["prefix_state_sha256"] != record["arms"][1]["prefix_state_sha256"]:
                raise ValueError(f"Prefix state mismatch: {path}")
            delta = record["paired_late_minus_early"]
            rows.append({"scene": scene, "seed": seed, "path": str(path),
                         "final_gaussian_count_delta": record.get(
                             "final_gaussian_count_delta_late_minus_early",
                             record["arms"][1]["final_gaussians"] - record["arms"][0]["final_gaussians"]),
                         "validity_audited": audit["valid"] if audit else record.get("valid"),
                         **delta})

    result = {"schema": "matched-candidate-timing-replay-summary-v1",
              "status": "exploratory pilot",
              "contrast": "late insertion at frame 90 minus early insertion at frame 30; same frozen 5k Gaussian batch",
              "inference_note": "Percentile bootstrap over four seeds within each scene; descriptive only. Two scenes are insufficient for population-level scene inference.",
              "scenes": {}}
    lines = [
        "# Matched-candidate timing replay: pilot report", "",
        "Two Replica scenes, four seeds per scene. Each pair starts from the same prefix snapshot, inserts an identical frozen batch of 5,000 Gaussian parameters at frame 30 or 90, disables all other births, and uses 5,000 micro-updates per mapped frame. Six held-out frames are omitted from mapping and scored after the final frame.",
        "",
        "Final validity is reassessed using causal-control gates: identical prefix digest within pair, identical candidate batch, exactly 5,000 manual additions per arm, zero automatic births, 565,000 optimizer micro-updates per arm, and the same held-out view list. Final map size is an outcome because timing can change pruning.",
        "",
        "## Late minus early contrasts", "",
        "Negative Depth L1 means the late arm has lower depth error. Intervals bootstrap the four seed-level contrasts within each scene and are descriptive only.",
        "",
        "| Scene | Outcome | Mean Δ [95% seed bootstrap interval] | Seed signs (+/−) |",
        "|---|---|---:|---:|",
    ]
    for scene in args.scenes:
        cohort = [row for row in rows if row["scene"] == scene]
        if len(cohort) != len(args.seeds) or any(row["validity_audited"] is False for row in cohort):
            raise ValueError(f"Missing or invalid audited pairs for {scene}")
        result["scenes"][scene] = {"n_seeds": len(cohort), "mean": {},
                                   "bootstrap_95_ci": {}, "signs": {}}
        for key, label in METRICS.items():
            vals = [float(row["final_gaussian_count_delta"] if key == "final_gaussian_count_delta"
                          else row[key]) for row in cohort]
            ci = bootstrap_mean_ci(vals, 2401 + sum(map(ord, scene + key)))
            mean = statistics.mean(vals)
            signs = {"positive": sum(v > 0 for v in vals),
                     "negative": sum(v < 0 for v in vals),
                     "zero": sum(v == 0 for v in vals)}
            result["scenes"][scene]["mean"][key] = mean
            result["scenes"][scene]["bootstrap_95_ci"][key] = list(ci)
            result["scenes"][scene]["signs"][key] = signs
            lines.append(f"| {scene} | {label} | {mean:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}] | {signs['positive']}/{signs['negative']} |")
    result["seed_level_records"] = rows
    repeat_records = []
    for repeat_root in args.same_seed_repeat_roots:
        repeat_path = Path(repeat_root) / "replica_room2" / "seed57.json"
        if not repeat_path.exists():
            continue
        repeat = json.loads(repeat_path.read_text())
        repeat_records.append({
            "artifact": str(repeat_path),
            "candidate_batch_sha256": repeat.get("candidate_batch_sha256"),
            "prefix_state_sha256": repeat.get("prefix_state_sha256"),
            "heldout_psnr_delta": repeat["paired_late_minus_early"]["fixed_heldout_mean_psnr"],
            "heldout_depth_l1_delta": repeat["paired_late_minus_early"]["fixed_heldout_mean_depth_l1"],
        })
    if repeat_records:
        repeat_psnr = [r["heldout_psnr_delta"] for r in repeat_records]
        prefix_hashes = sorted({r["prefix_state_sha256"] for r in repeat_records
                                if r["prefix_state_sha256"]})
        result["same_seed_repeatability"] = {
            "scene": "replica_room2", "seed": 57,
            "candidate_hashes_identical": len({r["candidate_batch_sha256"] for r in repeat_records}) == 1,
            "n_runs": len(repeat_records),
            "heldout_psnr_delta_min": min(repeat_psnr),
            "heldout_psnr_delta_max": max(repeat_psnr),
            "heldout_psnr_signs": {"positive": sum(v > 0 for v in repeat_psnr),
                                    "negative": sum(v < 0 for v in repeat_psnr)},
            "n_distinct_recorded_prefix_hashes": len(prefix_hashes),
            "runs": repeat_records,
        }
    frozen_path = args.frozen_repeat_root / "cohort_analysis.json"
    frozen_repeatability = json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    result["frozen_prefix_repeatability"] = frozen_repeatability
    result["interpretation"] = (
        "Across frozen prefixes, room2 shows positive held-out PSNR timing contrasts "
        "but Depth L1 and SSIM mostly worsen; office3 shows a mixed PSNR contrast, "
        "improved held-out Depth L1, but lower SSIM and trajectory-mean PSNR. Effects "
        "therefore depend on scene and metric. The timing intervention does not establish "
        "starvation as the mediator or show that natural online late spawning is beneficial."
    )
    lines += ["", "## Interpretation and limits", "",
              "### Same-seed repeatability diagnostic", "",
              "Replica room2 seed57 was rerun four times with identical candidate-batch hashes. Held-out PSNR late-minus-early ranged from "
              f"{min(r['heldout_psnr_delta'] for r in repeat_records):+.4f} to "
              f"{max(r['heldout_psnr_delta'] for r in repeat_records):+.4f} dB, with "
              f"{sum(r['heldout_psnr_delta'] > 0 for r in repeat_records)} positive and "
              f"{sum(r['heldout_psnr_delta'] < 0 for r in repeat_records)} negative contrasts. "
              f"The two reruns that recorded prefix hashes have {len({r['prefix_state_sha256'] for r in repeat_records if r['prefix_state_sha256']})} distinct prefix states. "
              "The different prefix hashes show that rebuilding the same-seed prefix does not reproduce the same state. The frozen-checkpoint repeats below are much tighter, so the earlier spread is associated with prefix regeneration rather than ordinary replay jitter. Seed-level intervals still do not include uncertainty over prefix realizations.",
              "",
              "### Frozen prefix cohort", ""]
    for scene, scene_summary in frozen_repeatability.get("scenes", {}).items():
        psnr = scene_summary["mean_across_seeds"]["fixed_heldout_mean_psnr"]
        psnr_ci = scene_summary["bootstrap_95_ci_across_seeds"]["fixed_heldout_mean_psnr"]
        psnr_signs = scene_summary["seed_signs"]["fixed_heldout_mean_psnr"]
        depth = scene_summary["mean_across_seeds"]["fixed_heldout_mean_depth_l1"]
        depth_ci = scene_summary["bootstrap_95_ci_across_seeds"]["fixed_heldout_mean_depth_l1"]
        depth_signs = scene_summary["seed_signs"]["fixed_heldout_mean_depth_l1"]
        lines.append(
            f"Across four frozen seed prefixes on {scene}, held-out PSNR contrast averaged {psnr:+.4f} dB [{psnr_ci[0]:+.4f}, {psnr_ci[1]:+.4f}] with seed signs {psnr_signs['positive']}/{psnr_signs['negative']}; Depth L1 averaged {depth:+.5f} m [{depth_ci[0]:+.5f}, {depth_ci[1]:+.5f}] with seed signs {depth_signs['positive']}/{depth_signs['negative']}. Each seed mean averages three technical replay repeats."
        )
    lines += ["",
              result["interpretation"], "",
              "The batch is generated once from the frame-30 RGB-D observation and the same batch object is passed to both replay arms. This controls candidate identity, but it does not model natural discovery of surfaces that only become visible later. The dose is 5k, not the 50k dose used in the prior E2 transfer. The result remains exploratory: two scenes and four frozen seed prefixes per scene. Each prefix has three technical replay repeats; those repeats do not count as independent seeds. The room2 contrast changed under earlier prefix regeneration, so future replications should preserve prefix checkpoints and report distinct prefix variation separately.",
              "", "Artifacts: per-seed JSON files, `validation.json`, and source hashes under `matched_candidate_timing_replay_verified/`; protocol in `research/matched_candidate_timing_protocol.md`; runner in `experiments/run/run_matched_candidate_timing_replay.py`."]
    out = args.out or root / "analysis.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    (out.with_suffix(".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
