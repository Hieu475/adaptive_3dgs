#!/usr/bin/env python3
"""Paired analysis for E1/E2 matched dilution control artifacts."""
import argparse
import json
import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SCENES_DEFAULT = ("tum_fr2_xyz", "tum_fr1_xyz")
SCENES_E1_E2 = ("tum_fr2_xyz", "tum_fr1_xyz")
SEEDS_DEFAULT = tuple(range(42, 50))


def read_pair(root, scene, experiment, seed):
    path = root / scene / f"{experiment}_seed{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def paired_stats(deltas, seed=42, n_boot=10000, positive_is_win=True):
    values = np.asarray(deltas, dtype=float)
    if not len(values):
        return {"n": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    p_value = exact_wilcoxon_p(values)
    return {
        "n": int(len(values)),
        "mean_delta": float(values.mean()),
        "std_delta": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci95_bootstrap_mean": [float(np.percentile(samples, 2.5)),
                                float(np.percentile(samples, 97.5))],
        "wins": int((values > 0).sum()) if positive_is_win else int((values < 0).sum()),
        "ties": int((values == 0).sum()),
        "losses": int((values < 0).sum()) if positive_is_win else int((values > 0).sum()),
        "worst_delta": float(values.min()) if positive_is_win else float(values.max()),
        "best_delta": float(values.max()) if positive_is_win else float(values.min()),
        "wilcoxon_p_unadjusted": p_value,
        "deltas": values.tolist(),
    }


def exact_wilcoxon_p(values):
    """Exact two-sided signed-rank p-value by enumerating sign assignments."""
    values = np.asarray(values, dtype=float)
    values = values[values != 0]
    if not len(values):
        return 1.0
    order = np.argsort(np.abs(values))
    ranks = np.empty(len(values), dtype=float)
    sorted_abs = np.abs(values)[order]
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_abs[j] == sorted_abs[i]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = average_rank
        i = j
    observed_positive = float(ranks[values > 0].sum())
    total = float(ranks.sum())
    observed_min = min(observed_positive, total - observed_positive)
    extreme = 0
    for signs in itertools.product((0, 1), repeat=len(values)):
        positive = sum(rank for rank, bit in zip(ranks, signs) if bit)
        if min(positive, total - positive) <= observed_min + 1e-12:
            extreme += 1
    return extreme / (2 ** len(values))


def analyze_e1(root, scenes, seeds):
    rows = []
    for scene in scenes:
        for seed in seeds:
            pair = read_pair(root, scene, "E1", seed)
            if not pair:
                continue
            arms = {arm["condition"]: arm["summary"].get("matched_final_map", {})
                    for arm in pair["arms"]}
            off = arms.get("no_throttle_matched_final", {})
            cov = arms.get("coverage_throttle_matched_final", {})
            rows.append({"scene": scene, "seed": seed,
                         "valid": bool(pair.get("valid") and off.get("valid") and cov.get("valid")),
                         "target_gaussians": pair.get("target_final_gaussians"),
                         "delta_psnr": cov.get("psnr", 0.0) - off.get("psnr", 0.0),
                         "delta_ssim": cov.get("ssim", 0.0) - off.get("ssim", 0.0),
                         "delta_depth_l1": cov.get("depth_l1", 0.0) - off.get("depth_l1", 0.0)})
    valid = [r for r in rows if r["valid"]]
    per_scene = {}
    for scene in scenes:
        rs = [r for r in valid if r["scene"] == scene]
        per_scene[scene] = {metric: paired_stats([r[metric] for r in rs], seed=43+i,
                                                 positive_is_win=(metric != "delta_depth_l1"))
                            for i, metric in enumerate(("delta_psnr", "delta_ssim", "delta_depth_l1"))}
        per_scene[scene]["valid_pairs"] = len(rs)
        per_scene[scene]["invalid_or_missing_pairs"] = len(seeds) - len(rs)
    all_valid = [r for r in valid]
    overall = {metric: paired_stats([r[metric] for r in all_valid], seed=47+i,
                                    positive_is_win=(metric != "delta_depth_l1"))
               for i, metric in enumerate(("delta_psnr", "delta_ssim", "delta_depth_l1"))}
    return {"per_scene": per_scene, "overall": overall,
            "valid_pairs": len(valid), "invalid_or_missing_pairs": len(scenes)*len(seeds)-len(valid),
            "rows": rows}


def analyze_e2(root, scenes, seeds):
    rows = []
    for scene in scenes:
        for seed in seeds:
            pair = read_pair(root, scene, "E2", seed)
            if not pair:
                continue
            arms = {arm["condition"]: arm for arm in pair["arms"]}
            early, late = arms.get("early_spawn"), arms.get("late_spawn")
            if not early or not late:
                continue
            count_early = sum(f["n_spawned"] for f in early["frames"])
            count_late = sum(f["n_spawned"] for f in late["frames"])
            rows.append({"scene": scene, "seed": seed,
                         "valid": bool(pair.get("valid") and count_early == count_late),
                         "target_spawned": pair.get("target_total_spawned"),
                         "delta_mean_psnr": late["summary"]["mean_psnr"] - early["summary"]["mean_psnr"],
                         "delta_final_psnr": late["summary"]["final_psnr"] - early["summary"]["final_psnr"],
                         "delta_mean_depth_l1": late["summary"]["mean_depth_l1"] - early["summary"]["mean_depth_l1"],
                         "early_off_mask_fraction": sum(f.get("n_spawned_fallback", 0) for f in early["frames"]) / max(count_early, 1),
                         "late_off_mask_fraction": sum(f.get("n_spawned_fallback", 0) for f in late["frames"]) / max(count_late, 1)})
    valid = [r for r in rows if r["valid"]]
    per_scene = {}
    metrics = ("delta_mean_psnr", "delta_final_psnr", "delta_mean_depth_l1")
    for scene in scenes:
        rs = [r for r in valid if r["scene"] == scene]
        per_scene[scene] = {metric: paired_stats([r[metric] for r in rs], seed=53+i,
                                                 positive_is_win=(metric != "delta_mean_depth_l1"))
                            for i, metric in enumerate(metrics)}
        per_scene[scene]["mean_early_off_mask_fraction"] = float(np.mean([r["early_off_mask_fraction"] for r in rs])) if rs else None
        per_scene[scene]["mean_late_off_mask_fraction"] = float(np.mean([r["late_off_mask_fraction"] for r in rs])) if rs else None
        per_scene[scene]["valid_pairs"] = len(rs)
        per_scene[scene]["invalid_or_missing_pairs"] = len(seeds) - len(rs)
    overall = {metric: paired_stats([r[metric] for r in valid], seed=59+i,
                                    positive_is_win=(metric != "delta_mean_depth_l1"))
               for i, metric in enumerate(metrics)}
    return {"per_scene": per_scene, "overall": overall,
            "valid_pairs": len(valid), "invalid_or_missing_pairs": len(scenes)*len(seeds)-len(valid),
            "rows": rows}


def analyze_e3(root, scenes, seeds):
    rows = []
    for scene in scenes:
        for seed in seeds:
            pair = read_pair(root, scene, "E3", seed)
            if not pair:
                continue
            arms = {arm["condition"]: arm for arm in pair.get("arms", [])}
            off, cov = arms.get("E3_no_throttle"), arms.get("E3_coverage_throttle")
            if not off or not cov:
                continue
            exact = all(int(f.get("n_micro_updates", -1)) == pair["target_micro_updates_per_frame"]
                        for arm in (off, cov) for f in arm["frames"])
            rows.append({"scene": scene, "seed": seed,
                         "valid": bool(pair.get("valid") and exact),
                         "delta_mean_psnr": cov["summary"]["mean_psnr"] - off["summary"]["mean_psnr"],
                         "delta_final_psnr": cov["summary"]["final_psnr"] - off["summary"]["final_psnr"],
                         "delta_mean_depth_l1": cov["summary"]["mean_depth_l1"] - off["summary"]["mean_depth_l1"],
                         "delta_mean_opt_time_ms": cov["summary"]["mean_opt_time_ms"] - off["summary"]["mean_opt_time_ms"],
                         "target_micro_updates_per_frame": pair["target_micro_updates_per_frame"]})
    valid = [r for r in rows if r["valid"]]
    metrics = ("delta_mean_psnr", "delta_final_psnr", "delta_mean_depth_l1", "delta_mean_opt_time_ms")
    per_scene = {}
    for scene in scenes:
        rs = [r for r in valid if r["scene"] == scene]
        per_scene[scene] = {metric: paired_stats([r[metric] for r in rs], seed=61+i,
            positive_is_win=(metric not in ("delta_mean_depth_l1", "delta_mean_opt_time_ms")))
            for i, metric in enumerate(metrics)}
        per_scene[scene]["valid_pairs"] = len(rs)
        per_scene[scene]["invalid_or_missing_pairs"] = len(seeds) - len(rs)
    return {"per_scene": per_scene, "valid_pairs": len(valid),
            "invalid_or_missing_pairs": len(scenes)*len(seeds)-len(valid), "rows": rows}


def summarize_age_arm(arm):
    frames = arm["frames"]
    n_updates = sum(int(f.get("n_micro_updates", 0)) for f in frames)
    n_mature_eligible = sum(int(f.get("mature_eligible", 0)) for f in frames)
    n_mature_updated = sum(int(f.get("mature_updated", 0)) for f in frames)
    mature_quota = sum(int(f.get("age_quota_mature_micro_updates") or 0) for f in frames)
    synchronized_frames = [f for f in frames if float(f.get("opt_stage_sync_ms", 0.0)) > 0]
    sync_returns = [(float(f.get("psnr_post", f.get("psnr", 0.0))) -
                     float(f.get("psnr_pre", f.get("psnr", 0.0)))) /
                    float(f["opt_stage_sync_ms"]) for f in synchronized_frames]
    return {
        "mean_psnr": float(arm["summary"]["mean_psnr"]),
        "final_psnr": float(arm["summary"]["final_psnr"]),
        "mean_depth_l1": float(arm["summary"]["mean_depth_l1"]),
        "mean_mature_update_coverage": n_mature_updated / n_mature_eligible if n_mature_eligible else 0.0,
        "mature_update_coverage_eligible": n_mature_eligible,
        "mature_update_coverage_updated": n_mature_updated,
        "growth_pressure_mean": float(np.mean([f.get("growth_pressure", 0.0) for f in frames])),
        "compute_return_db_per_opt_ms_mean": float(np.mean(sync_returns)) if sync_returns else None,
        "synchronized_timing_frames": len(synchronized_frames),
        "timing_frames": len(frames),
        "realized_mature_micro_update_share": mature_quota / n_updates if n_updates else 0.0,
        "total_micro_updates": n_updates,
        "mean_opt_time_ms": float(arm["summary"]["mean_opt_time_ms"]),
    }


def analyze_e5(root, scenes, seeds):
    fractions = (25, 50, 75)
    rows = []
    for scene in scenes:
        for seed in seeds:
            pair = read_pair(root, scene, "E5", seed)
            baseline_pair = read_pair(root, scene, "E3", seed)
            if not pair or not baseline_pair or not pair.get("valid"):
                continue
            baseline = next((a for a in baseline_pair.get("arms", [])
                             if a.get("condition") == "E3_no_throttle"), None)
            if baseline is None:
                continue
            base_summary = summarize_age_arm(baseline)
            target_updates = int(pair["target_micro_updates_per_frame"])
            baseline_exact = (base_summary["total_micro_updates"] ==
                              target_updates * len(baseline["frames"]))
            for fraction, arm in zip(fractions, pair.get("arms", [])):
                summary = summarize_age_arm(arm)
                exact = (summary["total_micro_updates"] ==
                         target_updates * len(arm["frames"]))
                rows.append({"scene": scene, "seed": seed, "requested_mature_fraction": fraction,
                    "valid": bool(pair.get("valid") and exact and baseline_exact),
                    "target_micro_updates_per_frame": target_updates, "arm": summary,
                    "delta_mean_psnr": summary["mean_psnr"] - base_summary["mean_psnr"],
                    "delta_final_psnr": summary["final_psnr"] - base_summary["final_psnr"],
                    "delta_mean_depth_l1": summary["mean_depth_l1"] - base_summary["mean_depth_l1"],
                    "delta_compute_return": (summary["compute_return_db_per_opt_ms_mean"] - base_summary["compute_return_db_per_opt_ms_mean"]
                        if summary["compute_return_db_per_opt_ms_mean"] is not None and
                           base_summary["compute_return_db_per_opt_ms_mean"] is not None else None)})
    valid = [r for r in rows if r["valid"]]
    per_scene = {}
    for scene in scenes:
        per_scene[scene] = {}
        for fraction in fractions:
            rs = [r for r in valid if r["scene"] == scene and r["requested_mature_fraction"] == fraction]
            compute_returns = [r["arm"]["compute_return_db_per_opt_ms_mean"] for r in rs
                               if r["arm"]["compute_return_db_per_opt_ms_mean"] is not None]
            per_scene[scene][str(fraction)] = {
                "valid_pairs": len(rs),
                "micro_updates_per_frame": int(rs[0]["target_micro_updates_per_frame"]) if rs else None,
                "diagnostics": {**{key: float(np.mean([r["arm"][key] for r in rs])) if rs else None
                    for key in ("mean_mature_update_coverage", "growth_pressure_mean",
                                "realized_mature_micro_update_share", "mean_opt_time_ms")},
                    "compute_return_db_per_opt_ms_mean": float(np.mean(compute_returns)) if compute_returns else None,
                    "synchronized_timing_seed_pairs": len(compute_returns)},
                "contrasts_vs_unconstrained": {
                    metric: paired_stats([r[metric] for r in rs if r[metric] is not None], seed=71+i,
                        positive_is_win=(metric != "delta_mean_depth_l1"))
                    for i, metric in enumerate(("delta_mean_psnr", "delta_final_psnr",
                                                "delta_mean_depth_l1", "delta_compute_return"))},
            }
    domain_summaries = {}
    quota_marginals = {}
    for domain, domain_scenes in (
        ("TUM", [s for s in scenes if s.startswith("tum_")]),
        ("Replica", [s for s in scenes if s.startswith("replica_")]),
    ):
        domain_summaries[domain] = {"scenes": domain_scenes, "by_quota": {}}
        for fraction in fractions:
            scene_blocks = [per_scene[s][str(fraction)] for s in domain_scenes
                            if per_scene[s][str(fraction)]["valid_pairs"]]
            returns = [b["diagnostics"]["compute_return_db_per_opt_ms_mean"]
                       for b in scene_blocks
                       if b["diagnostics"]["compute_return_db_per_opt_ms_mean"] is not None]
            domain_summaries[domain]["by_quota"][str(fraction)] = {
                "scene_count": len(scene_blocks),
                "macro_mean_delta_psnr": float(np.mean([
                    b["contrasts_vs_unconstrained"]["delta_mean_psnr"]["mean_delta"]
                    for b in scene_blocks])) if scene_blocks else None,
                "macro_final_delta_psnr": float(np.mean([
                    b["contrasts_vs_unconstrained"]["delta_final_psnr"]["mean_delta"]
                    for b in scene_blocks])) if scene_blocks else None,
                "macro_delta_depth_l1": float(np.mean([
                    b["contrasts_vs_unconstrained"]["delta_mean_depth_l1"]["mean_delta"]
                    for b in scene_blocks])) if scene_blocks else None,
                "macro_mature_update_coverage": float(np.mean([
                    b["diagnostics"]["mean_mature_update_coverage"]
                    for b in scene_blocks])) if scene_blocks else None,
                "macro_growth_pressure": float(np.mean([
                    b["diagnostics"]["growth_pressure_mean"]
                    for b in scene_blocks])) if scene_blocks else None,
                "macro_compute_return_db_per_opt_ms": float(np.mean(returns)) if returns else None,
            }
        quota_marginals[domain] = {}
        for lower, upper in ((25, 50), (50, 75)):
            scene_deltas = [
                per_scene[s][str(upper)]["contrasts_vs_unconstrained"]["delta_mean_psnr"]["mean_delta"] -
                per_scene[s][str(lower)]["contrasts_vs_unconstrained"]["delta_mean_psnr"]["mean_delta"]
                for s in domain_scenes
                if per_scene[s][str(lower)]["valid_pairs"] and
                   per_scene[s][str(upper)]["valid_pairs"]
            ]
            quota_marginals[domain][f"{lower}_to_{upper}"] = {
                "scene_deltas_mean_psnr": scene_deltas,
                "macro_delta_mean_psnr": float(np.mean(scene_deltas)) if scene_deltas else None,
                "scenes_improved": sum(x > 0 for x in scene_deltas),
                "scenes_worsened": sum(x < 0 for x in scene_deltas),
                "scenes_tied": sum(x == 0 for x in scene_deltas),
                "scene_count": len(scene_deltas),
            }
    return {"mature_age_frames": 10,
            "micro_updates_per_frame_by_scene": {scene: sorted({r["target_micro_updates_per_frame"] for r in valid if r["scene"] == scene}) for scene in scenes},
            "valid_arm_pairs": len(valid), "invalid_or_missing_arm_pairs": len(scenes)*len(seeds)*len(fractions)-len(valid),
            "reference": "E3 no-throttle unconstrained-importance arm",
            "per_scene": per_scene, "domain_macro_summary": domain_summaries,
            "quota_marginals": quota_marginals, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "results" / "dilution" / "matched_dilution_controls")
    ap.add_argument("--scenes", nargs="+", default=list(SCENES_DEFAULT))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    args = ap.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    # E1/E2 were only run under their corrected-pose TUM protocol. Do not
    # count Replica scenes requested for E3/E5 as missing E1/E2 observations.
    e1_e2_scenes = [scene for scene in SCENES_E1_E2 if scene in args.scenes]
    result = {"status": "exploratory", "scenes": args.scenes, "seeds": args.seeds,
              "E1_E2_scenes": e1_e2_scenes,
              "E1": analyze_e1(root, e1_e2_scenes, args.seeds),
              "E2": analyze_e2(root, e1_e2_scenes, args.seeds),
              "E3": analyze_e3(root, args.scenes, args.seeds),
              "E5": analyze_e5(root, args.scenes, args.seeds),
              "multiple_comparison_correction": "none; report p-values descriptively"}
    (root / "analysis.json").write_text(json.dumps(result, indent=2))
    lines = ["# Matched dilution control analysis", "",
             "Exploratory paired analysis; only exact-count pairs are included.",
             "Bootstrap intervals are over paired seed deltas. Wilcoxon p-values are unadjusted and descriptive.", ""]
    for key, label in (("E1", "Coverage minus no-throttle at matched final map size"),
                       ("E2", "Late minus early spawn timing at matched spawn count"),
                       ("E3", "Coverage minus no-throttle at matched micro-update count")):
        block = result[key]
        lines += [f"## {key}: {label}", "", f"Valid pairs: {block['valid_pairs']}; invalid/missing: {block['invalid_or_missing_pairs']}", "",
                  "| Scene | Metric | n | Mean delta | 95% paired bootstrap CI | Wins/losses | Worst | p (unadjusted) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        metrics = (("delta_psnr", "delta_ssim", "delta_depth_l1") if key == "E1" else
                   ("delta_mean_psnr", "delta_final_psnr", "delta_mean_depth_l1") if key == "E2" else
                   ("delta_mean_psnr", "delta_final_psnr", "delta_mean_depth_l1", "delta_mean_opt_time_ms"))
        for scene in block["per_scene"]:
            for metric in metrics:
                stat = block["per_scene"][scene][metric]
                ci = stat.get("ci95_bootstrap_mean", [None, None])
                n_stat = stat.get("n", 0)
                mean_text = f"{stat['mean_delta']:.4f}" if stat.get("mean_delta") is not None else "NA"
                ci_text = (f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None and ci[1] is not None else "NA")
                worst = stat.get("worst_delta")
                worst_text = f"{worst:.4f}" if worst is not None else "NA"
                p = stat.get("wilcoxon_p_unadjusted")
                p_text = f"{p:.4g}" if p is not None else "NA"
                lines.append(f"| {scene} | {metric} | {n_stat} | {mean_text} | {ci_text} | {stat.get('wins', 0)}/{stat.get('losses', 0)} | {worst_text} | {p_text} |")
        lines.append("")
    if result["E2"]["per_scene"]:
        lines += ["E2 off-error-mask spawn fractions by scene: " + "; ".join(
            f"{scene}: early {block['mean_early_off_mask_fraction']:.1%}, late {block['mean_late_off_mask_fraction']:.1%}"
            for scene, block in result["E2"]["per_scene"].items() if block["valid_pairs"]), ""]
    lines += ["## E5: fixed-work age allocation versus unconstrained importance", "",
              "Mature means age ≥10 frames; each scene uses a fixed per-frame update quota shared by its E5 arms and E3 reference. The reference is the no-throttle unconstrained-importance arm. Mature coverage is updated visible mature Gaussians divided by visible mature eligible Gaussians.", "",
              "| Scene | Updates/frame | Mature quota | n | Δ mean PSNR | Δ final PSNR | Δ depth L1 | Mature coverage | Growth pressure | Synchronized ComputeReturn (dB/ms) | Realized mature update share |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scene in args.scenes:
        for fraction, block in result["E5"]["per_scene"].get(scene, {}).items():
            d = block["diagnostics"]
            c = block["contrasts_vs_unconstrained"]
            vals = [c["delta_mean_psnr"].get("mean_delta", float("nan")),
                    c["delta_final_psnr"].get("mean_delta", float("nan")),
                    c["delta_mean_depth_l1"].get("mean_delta", float("nan"))]
            cr = d["compute_return_db_per_opt_ms_mean"]
            cr_text = f"{cr:.6g}" if cr is not None else "NA"
            lines.append(f"| {scene} | {block['micro_updates_per_frame'] or 'NA'} | {fraction}% | {block['valid_pairs']} | {vals[0]:+.4f} | {vals[1]:+.4f} | {vals[2]:+.5f} | {d['mean_mature_update_coverage']:.4f} | {d['growth_pressure_mean']:.5f} | {cr_text} | {d['realized_mature_micro_update_share']:.3f} |")
    lines += ["", "### E5 domain-level mechanism diagnostics", "",
              "Macro means weight scenes equally; they are descriptive, not pooled seed-level estimates.",
              "ComputeReturn is available only for synchronized Replica runs.", "",
              "| Domain | Quota | Scenes | Mean PSNR gain | Final PSNR gain | Depth L1 delta | Mature coverage | Growth pressure | ComputeReturn |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for domain, summary in result["E5"]["domain_macro_summary"].items():
        for fraction, d in summary["by_quota"].items():
            cr = d["macro_compute_return_db_per_opt_ms"]
            cr_text = f"{cr:.5f}" if cr is not None else "NA"
            lines.append(f"| {domain} | {fraction}% | {d['scene_count']} | {d['macro_mean_delta_psnr']:+.4f} | {d['macro_final_delta_psnr']:+.4f} | {d['macro_delta_depth_l1']:+.5f} | {d['macro_mature_update_coverage']:.4f} | {d['macro_growth_pressure']:.5f} | {cr_text} |")
    lines += ["", "Adjacent quota response in macro mean PSNR (scene means are paired by scene):", "",
              "| Domain | Quota step | Mean change | Scenes improved / worsened |",
              "|---|---|---:|---:|"]
    for domain, steps in result["E5"]["quota_marginals"].items():
        for step, d in steps.items():
            lines.append(f"| {domain} | {step.replace('_to_', '→')}% | {d['macro_delta_mean_psnr']:+.4f} dB | {d['scenes_improved']} / {d['scenes_worsened']} of {d['scene_count']} |")
    lines += ["", "E5 intervals use paired seed bootstrap and are available in `analysis.json`. ComputeReturn is reported only for runs with CUDA-synchronized stage wall time (cache + optimization + post-render); earlier TUM runs without synchronized timing are omitted from this metric. ComputeReturn remains an observational per-frame diagnostic, not an isolated marginal action effect.", ""]
    (root / "analysis.md").write_text("\n".join(lines))
    print(f"Wrote {root / 'analysis.md'} and {root / 'analysis.json'}")


if __name__ == "__main__":
    main()
