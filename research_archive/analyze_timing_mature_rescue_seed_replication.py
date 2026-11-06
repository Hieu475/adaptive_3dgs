#!/usr/bin/env python3
"""Summarize stale-mature rescue factorial across independent seeds.

Seed 57 is collapsed across its three technical repeats before it is counted as
one seed. Seeds 58-60 each contribute one independently built prefix/batch.
"""
import json
import statistics
from pathlib import Path


ROOT = Path("results")
PILOT = ROOT / "matched_candidate_timing_rescue_pilot"
REPLICATION = ROOT / "matched_candidate_timing_rescue_replication"
SCENES = ("replica_room2", "replica_office3")
METRICS = {
    "fixed_heldout_mean_psnr": "Held-out PSNR (dB)",
    "fixed_heldout_mean_depth_l1": "Held-out Depth L1 (m)",
    "fixed_heldout_mean_ssim": "Held-out SSIM",
}
CONTRASTS = {
    "timing_global": ("late_global", "early_global"),
    "timing_stale_rescue": ("late_stale_rescue", "early_stale_rescue"),
    "rescue_early": ("early_stale_rescue", "early_global"),
    "rescue_late": ("late_stale_rescue", "late_global"),
}


def avg(values):
    return statistics.mean(values)


def load_records(scene):
    pilot_validation = json.loads((PILOT / "validation.json").read_text())
    rep_validation = json.loads((REPLICATION / "validation.json").read_text())
    pilot_ok = {(p["scene"], p["run_id"]): p["valid"] for p in pilot_validation["pairs"]}
    rep_ok = {(p["scene"], p["seed"], p["run_id"]): p["valid"]
              for p in rep_validation["pairs"]}
    if rep_validation["valid_pairs"] != 6 or rep_validation["total_pairs"] != 6:
        raise RuntimeError("Replication validation is not 6/6")
    result = {}
    pilot_records = []
    for run_id in ("repeat1", "repeat2", "repeat3"):
        if not pilot_ok.get((scene, run_id), False):
            raise RuntimeError(f"Invalid seed57 pilot: {scene}/{run_id}")
        pilot_records.append(json.loads(
            (PILOT / scene / f"seed57_{run_id}.json").read_text()))
    result[57] = pilot_records
    for seed in (58, 59, 60):
        if not rep_ok.get((scene, seed, "independent_seed"), False):
            raise RuntimeError(f"Invalid replication: {scene}/{seed}")
        result[seed] = [json.loads(
            (REPLICATION / scene / f"seed{seed}_independent_seed.json").read_text())]
    return result


def arm_map(record):
    return {row["arm"]["name"]: row for row in record["arms"]}


def summarize_seed(records):
    if len({r["prefix_state_sha256"] for r in records}) != 1:
        raise RuntimeError("Technical repeats within a seed do not share one prefix")
    if len({r["candidate_batch_sha256"] for r in records}) != 1:
        raise RuntimeError("Technical repeats within a seed do not share one candidate batch")
    arm_values = {}
    for arm_name in ("early_global", "late_global", "early_stale_rescue", "late_stale_rescue"):
        rows = [arm_map(r)[arm_name] for r in records]
        arm_values[arm_name] = {m: avg([float(row[m]) for row in rows]) for m in METRICS}
        arm_values[arm_name]["realized_stale_update_fraction"] = avg(
            [float(row["realized_stale_update_fraction"]) for row in rows])
        arm_values[arm_name]["total_micro_updates"] = avg(
            [float(row["total_micro_updates"]) for row in rows])
    contrasts = {}
    for name, (high, low) in CONTRASTS.items():
        contrasts[name] = {
            metric: arm_values[high][metric] - arm_values[low][metric]
            for metric in METRICS
        }
    did = {metric: contrasts["timing_stale_rescue"][metric]
           - contrasts["timing_global"][metric] for metric in METRICS}
    return {"arms": arm_values, "contrasts": contrasts, "difference_in_differences": did,
            "technical_repeats": len(records)}


def aggregate(values):
    mean = statistics.mean(values)
    return {"mean": mean, "sd_across_seeds": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values), "n_seeds": len(values),
            "positive_seeds": sum(v > 0 for v in values),
            "negative_seeds": sum(v < 0 for v in values)}


def main():
    report = {
        "schema": "timing-x-stale-mature-rescue-seed-replication-v1",
        "seed_design": "Seed57 is the mean of three technical replays from one frozen prefix; seeds58-60 have independently rebuilt prefixes and candidate batches.",
        "validation": {"pilot": str(PILOT / "validation.json"),
                       "replication": str(REPLICATION / "validation.json")},
        "scenes": {},
    }
    md = [
        "# Stale-mature rescue factorial: independent-seed replication", "",
        "Seed57 contributes one seed-level estimate formed by averaging its three technical repeats. Seeds58–60 each contribute one independently initialized prefix and candidate batch. Thus n=4 seeds per scene; technical repeats are never counted as independent samples.",
        "", "All factorial arms used 5,000 candidate Gaussians and exactly 565,000 micro-updates. Rescue requested a 50% stale-mature quota; the realized share is reported separately. Validation passed 6/6 new seed-scene records.",
        "", "## Seed-level difference-in-differences", "",
        "DiD = (late−early under stale rescue) − (late−early under global selection). Positive values mean rescue makes the late-vs-early timing effect more positive for that metric.",
    ]
    for scene in SCENES:
        per_seed = {seed: summarize_seed(records) for seed, records in load_records(scene).items()}
        all_records = load_records(scene)
        prefix_hashes = {seed: records[0]["prefix_state_sha256"]
                         for seed, records in all_records.items()}
        candidate_hashes = {seed: records[0]["candidate_batch_sha256"]
                            for seed, records in all_records.items()}
        if len(set(prefix_hashes.values())) != len(prefix_hashes):
            raise RuntimeError(f"Independent seeds unexpectedly share a prefix in {scene}")
        if len(set(candidate_hashes.values())) != len(candidate_hashes):
            raise RuntimeError(f"Independent seeds unexpectedly share a candidate batch in {scene}")
        scene_json = {"per_seed": {str(k): v for k, v in per_seed.items()}, "aggregate": {},
                      "prefix_hashes_by_seed": {str(k): v for k, v in prefix_hashes.items()},
                      "candidate_hashes_by_seed": {str(k): v for k, v in candidate_hashes.items()},
                      "distinct_prefixes_and_candidate_batches_across_seeds": True}
        md += ["", f"## {scene}", "", "| Seed | DiD PSNR (dB) | DiD Depth L1 (m) | DiD SSIM | Rescue−global PSNR early | Rescue−global PSNR late | Rescue stale share (early / late) |",
               "|---:|---:|---:|---:|---:|---:|---:|"]
        for seed in sorted(per_seed):
            d = per_seed[seed]["difference_in_differences"]
            arms = per_seed[seed]["arms"]
            md.append(f"| {seed} | {d['fixed_heldout_mean_psnr']:+.4f} | {d['fixed_heldout_mean_depth_l1']:+.5f} | {d['fixed_heldout_mean_ssim']:+.5f} | {per_seed[seed]['contrasts']['rescue_early']['fixed_heldout_mean_psnr']:+.4f} | {per_seed[seed]['contrasts']['rescue_late']['fixed_heldout_mean_psnr']:+.4f} | {arms['early_stale_rescue']['realized_stale_update_fraction']:.3f} / {arms['late_stale_rescue']['realized_stale_update_fraction']:.3f} |")
        for metric, label in METRICS.items():
            vals = [per_seed[s]["difference_in_differences"][metric] for s in sorted(per_seed)]
            scene_json["aggregate"][metric] = aggregate(vals)
        for contrast in CONTRASTS:
            scene_json["aggregate"][contrast] = {}
            for metric in METRICS:
                vals = [per_seed[s]["contrasts"][contrast][metric] for s in sorted(per_seed)]
                scene_json["aggregate"][contrast][metric] = aggregate(vals)
        did_psnr = scene_json["aggregate"]["fixed_heldout_mean_psnr"]
        did_depth = scene_json["aggregate"]["fixed_heldout_mean_depth_l1"]
        early_psnr = scene_json["aggregate"]["rescue_early"]["fixed_heldout_mean_psnr"]
        late_psnr = scene_json["aggregate"]["rescue_late"]["fixed_heldout_mean_psnr"]
        md += ["", "Seed-level mean ± SD across n=4 seeds (ranges are across seeds, not CIs):",
               f"- DiD PSNR: {did_psnr['mean']:+.4f} ± {did_psnr['sd_across_seeds']:.4f} dB; range [{did_psnr['min']:+.4f}, {did_psnr['max']:+.4f}], sign {did_psnr['positive_seeds']} positive / {did_psnr['negative_seeds']} negative.",
               f"- DiD Depth L1: {did_depth['mean']:+.5f} ± {did_depth['sd_across_seeds']:.5f} m; range [{did_depth['min']:+.5f}, {did_depth['max']:+.5f}].",
               f"- Rescue−global held-out PSNR at early insertion: {early_psnr['mean']:+.4f} dB ({early_psnr['positive_seeds']}/4 seeds positive); at late insertion: {late_psnr['mean']:+.4f} dB ({late_psnr['positive_seeds']}/4 positive)."]
        report["scenes"][scene] = scene_json
    md += ["", "## Interpretation", "",
           "This replication tests whether stale-mature reallocation changes the timing effect under a frozen-batch replay, not whether natural map growth is the sole cause of quality loss. Read PSNR, SSIM, and depth together: an improvement in one metric can coincide with regression in another. Four seeds per scene are useful replication evidence but remain too few for a strong generalization claim; do not interpret these summaries as a paper-ready significance test.",
           "", "Raw runs and frozen artifacts are in `matched_candidate_timing_rescue_pilot/` and `matched_candidate_timing_rescue_replication/` under results."]
    out_md = REPLICATION / "seed_replication_analysis.md"
    out_json = REPLICATION / "seed_replication_analysis.json"
    out_md.write_text("\n".join(md) + "\n")
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
