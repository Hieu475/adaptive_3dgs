#!/usr/bin/env python3
"""Summarize independent randomized assignments from a shared frozen snapshot."""

import argparse
import glob
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


METRICS = {
    "rgb_attributed_error_per_pixel": ("rgb_attributed_abs_error_per_pixel", "rgb"),
    "depth_attributed_error_per_pixel": ("depth_attributed_abs_error_per_pixel", "depth"),
    "rgb_mean_contribution_weighted_error": ("rgb_contribution_weighted_error", "rgb"),
    "depth_mean_contribution_weighted_error": ("depth_contribution_weighted_error", "depth"),
    "topk_contribution_mass": ("contribution_mass_topk", "rgb"),
}


def aggregate(rows, key):
    by_id = defaultdict(list)
    for row in rows:
        residual = row["identity_contribution_residual"]
        for pid, value in zip(residual["persistent_ids"], residual[key]):
            by_id[int(pid)].append(float(value))
    return {pid: sum(v) / len(v) for pid, v in by_id.items()}


def contrast(units, outcomes):
    strata = defaultdict(lambda: {False: [], True: []})
    for u in units:
        pid = u["persistent_id"]
        if pid in outcomes:
            strata[u["stratum"]][u["treated"]].append(outcomes[pid])
    parts = {}
    numerator = denominator = 0.0
    for name, cls in strata.items():
        if not cls[False] or not cls[True]:
            continue
        n = len(cls[False]) + len(cls[True])
        delta = sum(cls[True]) / len(cls[True]) - sum(cls[False]) / len(cls[False])
        parts[name] = {"treated_n": len(cls[True]), "control_n": len(cls[False]), "effect": delta}
        numerator += n * delta
        denominator += n
    return (numerator / denominator if denominator else None), parts


def randomization_p(units, outcomes, observed, seed=1001, n_perm=5000):
    by_stratum = defaultdict(list)
    treated_n = defaultdict(int)
    for u in units:
        pid = u["persistent_id"]
        if pid in outcomes:
            by_stratum[u["stratum"]].append(pid)
            treated_n[u["stratum"]] += int(u["treated"])
    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        fake = []
        for s, ids in by_stratum.items():
            selected = set(rng.sample(ids, treated_n[s]))
            fake.extend({"persistent_id": pid, "stratum": s, "treated": pid in selected} for pid in ids)
        value, _ = contrast(fake, outcomes)
        if value is not None:
            null.append(value)
    if observed is None or not null:
        return None
    return (1 + sum(abs(x) >= abs(observed) for x in null)) / (len(null) + 1)


def analyze(path):
    data = json.loads(path.read_text())
    arm = data["arms"][0]
    assignment = arm["randomized_assignment"]
    units = [{"persistent_id": int(u["persistent_id"]), "stratum": u["stratum"], "treated": bool(u["treated"])}
             for u in assignment["assignment_units"] if not u["stratum"].endswith("not_randomizable")]
    result = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "valid": bool(data["all_count_and_snapshot_gates_passed"] and arm["valid"]),
              "snapshot_hash": data["base_snapshot_hash"], "randomization_seed": assignment["seed"],
              "cohort_present_visible": assignment["cohort_present_and_visible"],
              "randomizable_n": assignment["randomizable_ids"], "treated_n": assignment["treated_ids"],
              "control_n": assignment["control_ids"], "effects": {},
              "revisit_support": data.get("identity_revisit_support", []),
              "fixed_view_psnr_deltas": [b["psnr"] - a["psnr"] for a, b in zip(arm["fixed_eval_before_action"], arm["fixed_eval_after_horizon"])],
              "fixed_view_depth_l1_deltas": [b["depth_l1"] - a["depth_l1"] for a, b in zip(arm["fixed_eval_before_action"], arm["fixed_eval_after_horizon"])],
              "fixed_view_frames": [r["frame"] for r in arm["fixed_eval_before_action"]]}
    before = arm["fixed_eval_before_action"]
    after = arm["fixed_eval_after_horizon"]
    for name, (key, metric) in METRICS.items():
        pre, post = aggregate(before, key), aggregate(after, key)
        outcome = {pid: post[pid] - pre[pid] for pid in pre.keys() & post.keys()}
        effect, strata = contrast(units, outcome)
        result["effects"][name] = {"effect_treated_minus_control": effect, "strata": strata,
                                    "randomization_p_sharp_null_no_interference": randomization_p(
                                        units, outcome, effect, seed=1001 + assignment["seed"] + len(name))}
    trace = arm["identity_trace"]
    ti = {int(pid): i for i, pid in enumerate(trace["persistent_ids"])}
    compliance = {"treated_n": 0, "treated_exactly_one_update": 0,
                  "control_n": 0, "control_zero_updates": 0}
    for u in units:
        pid = u["persistent_id"]
        if pid not in ti:
            continue
        i = ti[pid]
        delta = int(trace["update_count_post"][i]) - int(trace["update_count_pre"][i])
        if u["treated"]:
            compliance["treated_n"] += 1
            compliance["treated_exactly_one_update"] += int(delta == 1)
        else:
            compliance["control_n"] += 1
            compliance["control_zero_updates"] += int(delta == 0)
    result["compliance"] = compliance
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/phase14_generalization_postfix/mechanism_growth_starvation/randomized_local_trial"))
    args = ap.parse_args()
    paths = [Path(x) for x in sorted(glob.glob(str(args.root / "replicate*" / "*" / "seed*_frame*.json")))]
    if not paths:
        raise SystemExit(f"No randomized trial records under {args.root}")
    trials = [analyze(path) for path in paths]
    if len({x["snapshot_hash"] for x in trials}) != 1:
        raise RuntimeError("Trial records do not share the same snapshot hash")
    output = {"design": "independent within-stratum randomized assignments from one frozen snapshot",
              "interpretation_unit": "randomization replicate; not independent scene/seed replication",
              "trials": trials, "summary": {}}
    for name in METRICS:
        vals = [x["effects"][name]["effect_treated_minus_control"] for x in trials
                if x["effects"][name]["effect_treated_minus_control"] is not None]
        output["summary"][name] = {"n_assignment_replicates": len(vals),
            "mean_effect": sum(vals) / len(vals) if vals else None,
            "sd_between_assignments": ((sum((x - sum(vals)/len(vals))**2 for x in vals)/(len(vals)-1))**0.5 if len(vals)>1 else None),
            "positive_replicates": sum(x > 0 for x in vals), "replicate_effects": vals,
            "trial_randomization_p_values": [x["effects"][name]["randomization_p_sharp_null_no_interference"] for x in trials]}
    out_dir = args.root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = ["# Randomized persistent-ID update trial", "",
        f"**Snapshot:** `{trials[0]['snapshot_hash'][:12]}`. **Assignments:** {len(trials)} independent randomization draws from that same snapshot. These are randomization replicates, not independent scene/seed replications.", "",
        "Across each trial, eligible visible cohort IDs were assigned approximately 1:1 within age × staleness × current-view RGB-residual median blocks. Treated IDs received exactly one optimizer update and controls zero. We compare pre-to-post changes in top-8 contribution-weighted residuals over the same fixed future views, chosen where cohort IDs became invisible and later reappeared. The treatment estimate is a candidate-count-weighted mean of within-stratum treated-minus-control contrasts.", "",
        "| Outcome | Mean effect over assignment draws | Between-assignment SD | Sign consistency | Per-draw randomization p* range |",
        "|---|---:|---:|---:|---:|"]
    for name, label in (("rgb_attributed_error_per_pixel", "RGB attributed absolute error / pixel"),
                        ("depth_attributed_error_per_pixel", "Depth attributed absolute error / pixel"),
                        ("rgb_mean_contribution_weighted_error", "RGB contribution-weighted mean residual"),
                        ("depth_mean_contribution_weighted_error", "Depth contribution-weighted mean residual"),
                        ("topk_contribution_mass", "Top-k contribution mass")):
        s = output["summary"][name]
        sd = "—" if s["sd_between_assignments"] is None else f"{s['sd_between_assignments']:+.4g}"
        mean = "—" if s["mean_effect"] is None else f"{s['mean_effect']:+.4g}"
        pvals = s["trial_randomization_p_values"]
        p = "—" if not pvals else f"{min(pvals):.4f}–{max(pvals):.4f}"
        lines.append(f"| {label} | {mean} | {sd} | {s['positive_replicates']}/{s['n_assignment_replicates']} positive | {p} |")
    psnr_means = [sum(t["fixed_view_psnr_deltas"]) / len(t["fixed_view_psnr_deltas"]) for t in trials]
    support = trials[0]["revisit_support"]
    lines += ["", "Actual revisit support by fixed view: " + "; ".join(f"frame {x['frame']}: {x['revisited_visible_ids']} cohort IDs reappeared (of {x['source_visible_ids']} source-visible)" for x in support) + ".", "",
        f"Mean global fixed-view PSNR delta per assignment: {sum(psnr_means)/len(psnr_means):+.5f} dB; assignment-specific values: " + ", ".join(f"{x:+.5f}" for x in psnr_means) + ".", "",
        "*The within-trial randomization p-value tests a sharp no-effect null and assumes no interference between Gaussian updates. Shared rendering and optimization can violate that assumption; treat p-values as exploratory. These repeated assignments share one map, scene, seed, and frame, so the between-assignment SD measures assignment sensitivity only. The gsplat mapping path is evaluated with the reference renderer's top-8 attribution endpoint; this differs from the gsplat RGB-D PSNR metric. This is a local mechanism pilot, not evidence of scene generalization or long-horizon benefit.", ""]
    (out_dir / "report.md").write_text("\n".join(lines))
    print(out_dir / "report.md")


if __name__ == "__main__":
    main()
