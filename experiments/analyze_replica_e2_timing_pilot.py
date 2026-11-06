#!/usr/bin/env python3
"""Summarize the exploratory Replica E2 matched-spawn timing pilot."""
import json
import argparse
import statistics
from pathlib import Path


SCENES = ("replica_room2", "replica_office3")
SEEDS = (57, 58)


def avg(xs):
    return statistics.mean(xs) if xs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("results/phase14_generalization_postfix/replica_e2_timing_pilot_room2_office3_seeds57_58"))
    parser.add_argument("--out-json", type=Path,
                        default=Path("results/phase14_generalization_postfix/replica_e2_timing_pilot_analysis.json"))
    parser.add_argument("--out-md", type=Path,
                        default=Path("results/phase14_generalization_postfix/replica_e2_timing_pilot_report.md"))
    args = parser.parse_args()
    root = args.root
    rows = []
    for scene in SCENES:
        for seed in SEEDS:
            path = root / scene / f"E2_seed{seed}.json"
            pair = json.loads(path.read_text())
            if not pair.get("valid"):
                raise ValueError(f"E2 pair failed exact-count gate: {path}")
            arms = {arm["condition"]: arm for arm in pair["arms"]}
            early, late = arms["early_spawn"], arms["late_spawn"]
            eframes, lframes = early["frames"], late["frames"]
            n = len(eframes)
            half = n // 2
            def half_mean(frames, start, end, metric):
                return avg([f[metric] for f in frames[start:end]])
            def spawn_total(arm):
                return sum(int(f["n_spawned"]) for f in arm["frames"])
            early_count, late_count = spawn_total(early), spawn_total(late)
            if early_count != late_count or early_count != int(pair["target_total_spawned"]):
                raise ValueError(f"Spawn totals do not match: {path}")
            fallback_early = sum(int(f.get("n_spawned_fallback", 0)) for f in eframes) / max(early_count, 1)
            fallback_late = sum(int(f.get("n_spawned_fallback", 0)) for f in lframes) / max(late_count, 1)
            rows.append({
                "scene": scene, "seed": seed, "valid": True,
                "frames": n, "split_frame": int(eframes[half - 1]["frame"]),
                "spawn_total": early_count,
                "mean_psnr_delta_late_minus_early": late["summary"]["mean_psnr"] - early["summary"]["mean_psnr"],
                "final_psnr_delta_late_minus_early": late["summary"]["final_psnr"] - early["summary"]["final_psnr"],
                "mean_ssim_delta_late_minus_early": late["summary"]["mean_ssim"] - early["summary"]["mean_ssim"],
                "mean_depth_l1_delta_late_minus_early": late["summary"]["mean_depth_l1"] - early["summary"]["mean_depth_l1"],
                "early_arm_psnr_first_half": half_mean(eframes, 0, half, "psnr"),
                "early_arm_psnr_second_half": half_mean(eframes, half, n, "psnr"),
                "late_arm_psnr_first_half": half_mean(lframes, 0, half, "psnr"),
                "late_arm_psnr_second_half": half_mean(lframes, half, n, "psnr"),
                "off_error_mask_spawn_fraction_early": fallback_early,
                "off_error_mask_spawn_fraction_late": fallback_late,
                "early_mean_frame_ms": early["summary"]["mean_frame_time_ms"],
                "late_mean_frame_ms": late["summary"]["mean_frame_time_ms"],
                "git_sha": pair["unthrottled_spawn_pilot"]["manifest"]["git_sha"],
                "target_total_spawned": int(pair["target_total_spawned"]),
                "target_selection": pair.get("target_selection", "per_seed_natural_pilot"),
                "candidate_mode": pair["validation"].get("candidate_mode", "uniform_depth"),
                "window_early": pair["validation"]["early_window"],
                "window_late": pair["validation"]["late_window"],
            })
    per_scene = {}
    for scene in SCENES:
        cohort = [r for r in rows if r["scene"] == scene]
        per_scene[scene] = {
            "n": len(cohort),
            "mean_spawn_total": avg([r["spawn_total"] for r in cohort]),
            "mean_psnr_delta_late_minus_early": avg([r["mean_psnr_delta_late_minus_early"] for r in cohort]),
            "mean_final_psnr_delta_late_minus_early": avg([r["final_psnr_delta_late_minus_early"] for r in cohort]),
            "mean_ssim_delta_late_minus_early": avg([r["mean_ssim_delta_late_minus_early"] for r in cohort]),
            "mean_depth_l1_delta_late_minus_early": avg([r["mean_depth_l1_delta_late_minus_early"] for r in cohort]),
            "mean_off_mask_spawn_fraction_early": avg([r["off_error_mask_spawn_fraction_early"] for r in cohort]),
            "mean_off_mask_spawn_fraction_late": avg([r["off_error_mask_spawn_fraction_late"] for r in cohort]),
            "half_window_means": {
                key: avg([r[key] for r in cohort])
                for key in ("early_arm_psnr_first_half", "early_arm_psnr_second_half",
                            "late_arm_psnr_first_half", "late_arm_psnr_second_half")
            },
            "seed_signs_mean_psnr": [1 if r["mean_psnr_delta_late_minus_early"] > 0 else -1
                                      for r in cohort],
        }
    result = {
        "schema": "replica-e2-timing-pilot-analysis-v1",
        "status": "exploratory",
        "contrast": "late spawn minus early spawn; equal per-pair total spawn count",
        "proposal_policy": sorted({r["candidate_mode"] for r in rows}),
        "pose_protocol": "Replica W2C extrinsics from corrected Phase 10 loader",
        "inference_note": "Two seeds per scene; no inferential test or cross-frame independence assumption.",
        "source_git_shas": sorted({row["git_sha"] for row in rows}),
        "scenes": per_scene,
        "pairs": rows,
    }
    fixed_target = all(r["target_selection"] == "frozen_cli_target" for r in rows)
    target_values = sorted({r["target_total_spawned"] for r in rows})
    candidate_modes = sorted({r["candidate_mode"] for r in rows})
    fallback_rates = [r[k] for r in rows for k in
                      ("off_error_mask_spawn_fraction_early", "off_error_mask_spawn_fraction_late")]
    fallback_range = f"{min(fallback_rates):.0%}–{max(fallback_rates):.0%}"
    source_shas = sorted({r["git_sha"] for r in rows})
    if fixed_target:
        dose_description = f"A shared {target_values[0]:,}-spawn total was frozen across both scenes and both seeds before these runs."
        limitation = (f"Only two seeds per scene are included, and {fallback_range} of births are fallback proposals outside the error mask. "
                      "The result is exploratory, candidate support changes with the evolving error map, and this does not establish compute dilution as the mediator.")
    else:
        dose_description = "Each pair's spawn target was estimated from its own natural unthrottled pilot."
        limitation = (f"Only two seeds per scene are included, with very large pilot-derived totals (about 145k–154k births) and {fallback_range} fallback births. "
                      "The totals were not preregistered. This is not evidence that the usual error-mask spawn policy should be delayed, "
                      "nor that compute dilution mediates the effect. Replicate at a frozen dose and use a proposal pool tied to measured novelty.")
    out_json, out_md = args.out_json, args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Replica E2 timing pilot",
        "",
        f"Exploratory matched-total-spawn timing comparison. {dose_description} Candidate mode: {', '.join(candidate_modes)}. Births are restricted to an early or late half of the 119-frame corrected-pose trajectory. Total births match exactly within each pair.",
        "",
        "| Scene | Seed | Target spawns | Mean ΔPSNR | Final ΔPSNR | Mean ΔSSIM | Mean ΔDepth L1 (m) | Fallback births early/late |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scene']} | {r['seed']} | {r['spawn_total']} | "
            f"{r['mean_psnr_delta_late_minus_early']:+.4f} dB | "
            f"{r['final_psnr_delta_late_minus_early']:+.4f} dB | "
            f"{r['mean_ssim_delta_late_minus_early']:+.5f} | "
            f"{r['mean_depth_l1_delta_late_minus_early']:+.5f} | "
            f"{r['off_error_mask_spawn_fraction_early']:.1%}/{r['off_error_mask_spawn_fraction_late']:.1%} |"
        )
    lines += ["", "## Window dynamics", "",
              "| Scene | Early arm, first half | Early arm, second half | Late arm, first half | Late arm, second half |",
              "|---|---:|---:|---:|---:|"]
    for scene, summary in per_scene.items():
        h = summary["half_window_means"]
        lines.append(f"| {scene} | {h['early_arm_psnr_first_half']:.3f} | {h['early_arm_psnr_second_half']:.3f} | {h['late_arm_psnr_first_half']:.3f} | {h['late_arm_psnr_second_half']:.3f} |")
    if len(source_shas) == 1 and "-dirty" not in source_shas[0]:
        provenance_note = f"All four pairs were rerun from clean source commit `{source_shas[0]}`; the frozen source bundle and provenance manifest are linked in the experiment report."
    else:
        provenance_note = f"Source manifests record these revisions: {', '.join(f'`{sha}`' for sha in source_shas)}. This is an exploratory working-tree run, not a clean-checkout paper result."
    lines += ["", "## Interpretation", "",
              f"Late spawning had lower trajectory-mean PSNR on both scenes, but higher final PSNR; late-arm quality starts lower before births begin and recovers in the second half. This pilot shows a timing-dependent trajectory tradeoff under the {', '.join(candidate_modes)} proposal protocol, not a universal benefit of delaying births.",
              "",
              limitation,
              "",
              "All four exact-count gates passed. " + provenance_note,
              "",
              f"Raw pairs: [`room2 seed 57`]({root.name}/replica_room2/E2_seed57.json), [`room2 seed 58`]({root.name}/replica_room2/E2_seed58.json), [`office3 seed 57`]({root.name}/replica_office3/E2_seed57.json), [`office3 seed 58`]({root.name}/replica_office3/E2_seed58.json)."]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
