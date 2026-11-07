#!/usr/bin/env python3
"""Merge clean six-policy Phase-14 output with the separate clean RTG run."""
import json
from pathlib import Path

import numpy as np

from experiments.run_final_confirmation import (
    bootstrap_ci_95,
    cohens_dz,
    holm_bonferroni,
    safe_wilcoxon,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "phase14_projected_area_fix_clean_8seeds"
RTG_SOURCE = RESULTS / "phase14_projected_area_fix_clean_rtg8"
OUT = RESULTS / "phase14_projected_area_fix_clean_7policy_8seeds"
POLICIES = ("no_op", "error_only", "error_influence", "error_influence_temporal", "rtg_slam_reimpl", "ours", "full")
SEEDS = list(range(42, 50))
METRICS = ("mean_psnr", "final_psnr", "mean_ssim", "final_ssim", "N_final", "fps")


def load(path):
    return json.loads(path.read_text())


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    base = load(SOURCE / "final_confirmation_results.json")
    rtg = load(RTG_SOURCE / "final_confirmation_results.json")
    raw = base["raw_runs"] + rtg["raw_runs"]
    keyed = {(r["policy"], r["seed"]): r for r in raw}
    expected = {(policy, seed) for policy in POLICIES for seed in SEEDS}
    if set(keyed) != expected or len(raw) != len(expected):
        raise ValueError("Expected one finite run per policy and seed")
    for row in raw:
        for metric in METRICS:
            if not np.isfinite(float(row[metric])):
                raise ValueError(f"Non-finite {metric}: {row['policy']} seed {row['seed']}")
    raw.sort(key=lambda r: (POLICIES.index(r["policy"]), r["seed"]))

    policy_stats = {}
    for policy in POLICIES:
        rows = [r for r in raw if r["policy"] == policy]
        policy_stats[policy] = {}
        for metric in METRICS:
            vals = np.asarray([r[metric] for r in rows], dtype=float)
            policy_stats[policy][metric] = {
                "mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()
            }

    comparisons = []
    p_mean, p_final = [], []
    for baseline in POLICIES:
        if baseline == "ours":
            continue
        ours = np.asarray(policy_stats["ours"]["mean_psnr"]["values"])
        base_vals = np.asarray(policy_stats[baseline]["mean_psnr"]["values"])
        delta = ours - base_vals
        ours_final = np.asarray(policy_stats["ours"]["final_psnr"]["values"])
        base_final = np.asarray(policy_stats[baseline]["final_psnr"]["values"])
        delta_final = ours_final - base_final
        delta_ssim = (np.asarray(policy_stats["ours"]["mean_ssim"]["values"])
                      - np.asarray(policy_stats[baseline]["mean_ssim"]["values"]))
        p, pf = safe_wilcoxon(ours, base_vals), safe_wilcoxon(ours_final, base_final)
        p_mean.append(p)
        p_final.append(pf)
        comparisons.append({
            "baseline": baseline,
            "delta_mean_psnr": float(delta.mean()),
            "ci_mean_psnr": bootstrap_ci_95(delta),
            "p_mean_psnr": p,
            "cohens_dz_mean_psnr": cohens_dz(ours, base_vals),
            "delta_final_psnr": float(delta_final.mean()),
            "ci_final_psnr": bootstrap_ci_95(delta_final),
            "p_final_psnr": pf,
            "cohens_dz_final_psnr": cohens_dz(ours_final, base_final),
            "delta_mean_ssim": float(delta_ssim.mean()),
            "p_mean_ssim": safe_wilcoxon(
                np.asarray(policy_stats["ours"]["mean_ssim"]["values"]),
                np.asarray(policy_stats[baseline]["mean_ssim"]["values"])),
        })
    holm_mean, holm_final = holm_bonferroni(p_mean), holm_bonferroni(p_final)
    for index, comp in enumerate(comparisons):
        comp["p_holm_mean_psnr"] = holm_mean[index]
        comp["p_holm_final_psnr"] = holm_final[index]

    metadata = dict(base["metadata"])
    metadata["policies"] = list(POLICIES)
    metadata["n_seeds"] = len(SEEDS)
    metadata["total_wall_sec"] = float(metadata.get("total_wall_sec", 0.0)
                                        + rtg["metadata"].get("total_wall_sec", 0.0))
    metadata["min_wilcoxon_p"] = float(2 / 2**len(SEEDS))
    combined = {
        "metadata": metadata,
        "policy_stats": policy_stats,
        "paired_comparisons": comparisons,
        "raw_runs": raw,
        "source_runs": [
            "phase14_projected_area_fix_clean_8seeds",
            "phase14_projected_area_fix_clean_rtg8",
        ],
    }

    out_report = [
        "# Clean Phase-14 confirmation: seven policies", "",
        "**Scene:** `tum_fr2_xyz` | **Frames:** 150 | **Seeds:** 42–49 | **Budget:** 15 ms modeled optimization budget | **Backend:** gsplat | **Pose:** ground-truth W2C", "",
        "The six-policy run and the RTG-SLAM reimplementation run were executed separately from the same clean source commit. Seeds are the replicate unit.", "",
        "| Policy | Mean PSNR | Final PSNR | Mean SSIM | Mean depth L1 | FPS | Final map N |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for policy in POLICIES:
        s = policy_stats[policy]
        out_report.append(
            f"| {policy} | {s['mean_psnr']['mean']:.2f} ± {s['mean_psnr']['std']:.2f} | "
            f"{s['final_psnr']['mean']:.2f} ± {s['final_psnr']['std']:.2f} | "
            f"{s['mean_ssim']['mean']:.4f} | {float(np.mean([r['mean_depth_l1'] for r in raw if r['policy']==policy])):.5f} | "
            f"{s['fps']['mean']:.1f} | {s['N_final']['mean']:,.0f} |"
        )
    out_report += ["", "## Paired contrasts with OURS", "",
                   "Wilcoxon tests use eight per-seed trajectory summaries. Holm correction is applied across the six baseline contrasts; bootstrap intervals use the harness implementation.", "",
                   "| Baseline | Δ mean PSNR | 95% CI | Raw p | Holm p | Δ final PSNR | Holm p final |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for comp in comparisons:
        out_report.append(
            f"| {comp['baseline']} | {comp['delta_mean_psnr']:+.2f} dB | "
            f"[{comp['ci_mean_psnr'][0]:+.2f}, {comp['ci_mean_psnr'][1]:+.2f}] | "
            f"{comp['p_mean_psnr']:.4f} | {comp['p_holm_mean_psnr']:.4f} | "
            f"{comp['delta_final_psnr']:+.2f} dB | {comp['p_holm_final_psnr']:.4f} |"
        )
    q_noop = policy_stats["no_op"]["mean_psnr"]["mean"]
    q_full = policy_stats["full"]["mean_psnr"]["mean"]
    q_ours = policy_stats["ours"]["mean_psnr"]["mean"]
    ratio = (q_ours - q_noop) / (q_full - q_noop) if abs(q_full - q_noop) > 1e-9 else float("nan")
    latency = float(np.mean([r["mean_frame_time_ms"] for r in raw if r["policy"] == "ours"]))
    out_report += ["", "## System interpretation", "",
                   f"Headroom recovery above no-op is {ratio*100:.1f}%; OURS is {q_full-q_ours:.2f} dB below FULL with {policy_stats['full']['N_final']['mean']/policy_stats['ours']['N_final']['mean']:.2f}× fewer Gaussians.",
                   f"OURS mean end-to-end frame time is {latency:.1f} ms. The 15 ms value is only the modeled optimization budget, so this run does not establish real-time operation.", "",
                   "## Protocol notes", "",
                   "Ground-truth poses isolate mapping; no ATE/RPE claim is made. This bundle combines two runs from the same source commit and records both raw run sets; the complete merged table is generated from per-seed outputs without treating frames as independent replicates.", ""]

    OUT.mkdir(parents=True)
    (OUT / "final_confirmation_results.json").write_text(json.dumps(combined, indent=2) + "\n")
    (OUT / "final_confirmation_report.md").write_text("\n".join(out_report))
    print(OUT)


if __name__ == "__main__":
    main()
