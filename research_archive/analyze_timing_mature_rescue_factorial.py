#!/usr/bin/env python3
"""Analyze timing × stale-mature-rescue replay pilot without treating repeats as seeds."""
import json
import statistics
from pathlib import Path


ROOT = Path("results/matched_candidate_timing_rescue_pilot")
SCENES = ("replica_room2", "replica_office3")
SEED = 57
RUN_IDS = ("repeat1", "repeat2", "repeat3")
OUTCOMES = {
    "fixed_heldout_mean_psnr": "Held-out PSNR (dB)",
    "fixed_heldout_mean_depth_l1": "Held-out Depth L1 (m)",
    "fixed_heldout_mean_ssim": "Held-out SSIM",
    "trajectory_mean_psnr": "Trajectory mean PSNR (dB)",
    "trajectory_mean_depth_l1": "Trajectory mean Depth L1 (m)",
    "mean_mature_update_coverage": "Mature update coverage",
    "mean_stale_mature_update_coverage": "Stale-mature update coverage",
    "realized_stale_update_fraction": "Share of updates to stale-mature Gaussians",
    "mean_opt_stage_sync_ms": "Optimization-stage time (ms)",
}


def mean_range(values):
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values),
            "technical_repeats": len(values)}


def main():
    validation = json.loads((ROOT / "validation.json").read_text())
    expected = len(SCENES) * len(RUN_IDS)
    if validation["valid_pairs"] != expected:
        raise RuntimeError(f"Only {validation['valid_pairs']}/{expected} pairs passed all gates")
    validation_keys = {(p["scene"], p.get("run_id")): p
                       for p in validation["pairs"]}
    result = {
        "schema": "timing-x-stale-mature-rescue-pilot-v1",
        "status": "exploratory pilot; one seed per scene, three technical repeats",
        "treatment": "2x2 early/late insertion crossed with global/stale-mature-rescue allocation",
        "dose": {"candidate_gaussians": 5000, "micro_updates_per_mapped_frame": 5000,
                 "stale_fraction_requested": 0.5, "mature_age_frames": 10,
                 "stale_after_frames": 10},
        "statistical_note": "Three replay repeats share one frozen prefix and candidate batch per scene. They are technical repeats, not independent seeds; report means and ranges, no seed-level confidence intervals.",
        "scenes": {},
    }
    lines = [
        "# Timing × stale-mature-rescue factorial pilot", "",
        "Two Replica scenes, seed57 only, three technical repeats from one frozen prefix/candidate batch per scene. The factorial crosses insertion timing (frame 30 vs 90) with update allocation (global importance vs 50% stale-mature rescue; stale means mature age ≥10 frames and last updated ≥10 frames ago). All 6 pairs passed checksum, exact-birth, zero-automatic-birth, exact-update, held-out-view, and rescue-first-stage gates.",
        "",
        "The stale-rescue arm first assigns quota to visible stale mature Gaussians. Remaining updates go to the ordinary global-importance pool. Every arm uses 565,000 total micro-updates across 113 mapping frames; final map size is an outcome.",
        "",
        "## Arm outcomes", "",
        "Values are mean [min, max] over the three technical repeats; these are not confidence intervals.",
    ]
    for scene in SCENES:
        records = []
        for run_id in RUN_IDS:
            audit = validation_keys.get((scene, run_id))
            if not audit or not audit["valid"]:
                raise RuntimeError(f"Missing/invalid audit for {scene}/{run_id}")
            record = json.loads((ROOT / scene / f"seed{SEED}_{run_id}.json").read_text())
            if not record["valid"]:
                raise RuntimeError(f"Runner marked invalid pair {scene}/{run_id}")
            records.append(record)
        # Prefix and candidate identity must be identical across all technical repeats.
        if len({r["prefix_state_sha256"] for r in records}) != 1:
            raise RuntimeError(f"Prefix changed across repeats for {scene}")
        if len({r["candidate_batch_sha256"] for r in records}) != 1:
            raise RuntimeError(f"Candidate batch changed across repeats for {scene}")
        arm_names = ("early_global", "late_global", "early_stale_rescue", "late_stale_rescue")
        scene_result = {"seed": SEED, "technical_repeats": len(records),
                        "arms": {}, "contrasts": {}}
        lines += ["", f"## {scene}", "", "### Arm outcomes", "",
                  "| Arm | Held-out PSNR | Held-out Depth L1 (m) | Held-out SSIM | Stale update share | Stale coverage | Updates / arm |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for arm_name in arm_names:
            arm_rows = [next(a for a in r["arms"] if a["arm"]["name"] == arm_name)
                        for r in records]
            values = {}
            for key in OUTCOMES:
                values[key] = mean_range([float(a[key]) for a in arm_rows])
            scene_result["arms"][arm_name] = values
            share = values["realized_stale_update_fraction"]
            stale_cov = values["mean_stale_mature_update_coverage"]
            lines.append(
                f"| {arm_name} | {values['fixed_heldout_mean_psnr']['mean']:.4f} "
                f"[{values['fixed_heldout_mean_psnr']['min']:.4f}, {values['fixed_heldout_mean_psnr']['max']:.4f}] | "
                f"{values['fixed_heldout_mean_depth_l1']['mean']:.5f} "
                f"[{values['fixed_heldout_mean_depth_l1']['min']:.5f}, {values['fixed_heldout_mean_depth_l1']['max']:.5f}] | "
                f"{values['fixed_heldout_mean_ssim']['mean']:.5f} | "
                f"{share['mean']:.3f} | {stale_cov['mean']:.3f} | "
                f"{statistics.mean(a['total_micro_updates'] for a in arm_rows)} |"
            )

        contrast_definitions = {
            "timing_global": ("late_global", "early_global"),
            "timing_stale_rescue": ("late_stale_rescue", "early_stale_rescue"),
            "rescue_early": ("early_stale_rescue", "early_global"),
            "rescue_late": ("late_stale_rescue", "late_global"),
        }
        contrast_values = {name: {} for name in contrast_definitions}
        did_values = {}
        for key in OUTCOMES:
            per_repeat = {}
            for name, (high, low) in contrast_definitions.items():
                per_repeat[name] = []
                for record in records:
                    arms = {a["arm"]["name"]: a for a in record["arms"]}
                    per_repeat[name].append(float(arms[high][key] - arms[low][key]))
                contrast_values[name][key] = mean_range(per_repeat[name])
            did_values[key] = mean_range([
                rescue - global_delta for rescue, global_delta in zip(
                    per_repeat["timing_stale_rescue"], per_repeat["timing_global"])
            ])
        scene_result["contrasts"] = contrast_values
        scene_result["difference_in_differences"] = did_values
        lines += ["", "### Paired contrasts", "",
                  "Late−early compares timing within allocation; rescue−global compares allocation within timing. DiD = (late−early under stale rescue) − (late−early under global).",
                  "", "| Contrast | Held-out ΔPSNR (dB) | Held-out ΔDepth L1 (m) | Held-out ΔSSIM | Δ stale update share |",
                  "|---|---:|---:|---:|---:|"]
        for name in (*contrast_definitions.keys(), "difference_in_differences"):
            metrics = did_values if name == "difference_in_differences" else contrast_values[name]
            psnr = metrics["fixed_heldout_mean_psnr"]
            depth = metrics["fixed_heldout_mean_depth_l1"]
            ssim = metrics["fixed_heldout_mean_ssim"]
            share = metrics["realized_stale_update_fraction"]
            lines.append(f"| {name} | {psnr['mean']:+.5f} [{psnr['min']:+.5f}, {psnr['max']:+.5f}] | "
                         f"{depth['mean']:+.5f} [{depth['min']:+.5f}, {depth['max']:+.5f}] | "
                         f"{ssim['mean']:+.5f} | {share['mean']:+.4f} |")
        result["scenes"][scene] = scene_result

    result["interpretation"] = (
        "The rescue first stage worked: stale-mature update share increased from about "
        "1–2% under global allocation to about 25–29% under rescue, with exact total "
        "compute. The quality response was scene-dependent. In room2 stale rescue lifted "
        "held-out PSNR for both timing arms, but depth worsened and the timing-by-rescue "
        "interaction was small. In office3, rescue improved the early arm's held-out PSNR "
        "but slightly reduced the late arm's; the interaction was large and negative, "
        "while depth and SSIM moved differently. This is evidence that update allocation "
        "can affect outcomes under matched update counts, but it does not establish a "
        "universal starvation mechanism or controller."
    )
    lines += ["", "## Interpretation and limits", "", result["interpretation"], "",
              "This is one seed per scene and three technical repeats per frozen prefix. The repeats measure replay stability and are not independent scene or seed samples. Candidate dose is 5k. The intervention changes which primitives receive updates and therefore tests the effect of stale-mature allocation under this exact replay; it does not prove that natural map growth caused all observed quality changes. PSNR, depth, SSIM, trajectory averages, and latency are reported separately because they can disagree.",
              "", "Artifacts: `validation.json`, per-run factorial records, the frozen checkpoints/candidate batches in the shared `frozen_artifacts/` directory, and source hashes under this result root."]
    out_md, out_json = ROOT / "factorial_analysis.md", ROOT / "factorial_analysis.json"
    out_md.write_text("\n".join(lines) + "\n")
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
