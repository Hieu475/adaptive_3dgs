#!/usr/bin/env python3
"""Analyze the first randomized persistent-ID update-allocation trial."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def _aggregate_by_id(rows, metric_key):
    values = defaultdict(list)
    visible = defaultdict(list)
    for row in rows:
        residual = row["identity_contribution_residual"]
        for pid, value, is_visible in zip(residual["persistent_ids"],
                residual[metric_key], residual["visible"]):
            values[int(pid)].append(float(value))
            visible[int(pid)].append(bool(is_visible))
    return ({pid: sum(v) / len(v) for pid, v in values.items()},
            {pid: sum(v) / len(v) for pid, v in visible.items()})


def _contrast(units, outcomes):
    per_stratum = {}
    numerator = denominator = 0.0
    for stratum in sorted({u["stratum"] for u in units}):
        members = [u for u in units if u["stratum"] == stratum and u["persistent_id"] in outcomes]
        treated = [outcomes[u["persistent_id"]] for u in members if u["treated"]]
        controls = [outcomes[u["persistent_id"]] for u in members if not u["treated"]]
        if not treated or not controls:
            continue
        estimate = sum(treated) / len(treated) - sum(controls) / len(controls)
        n = len(treated) + len(controls)
        per_stratum[stratum] = {"n_treated": len(treated), "n_control": len(controls),
                                "effect": estimate, "weight_n": n}
        numerator += n * estimate
        denominator += n
    return (numerator / denominator if denominator else None), per_stratum


def _randomization_test(units, outcomes, observed, n_perm=10000, seed=7321):
    strata = defaultdict(list)
    treated_counts = defaultdict(int)
    for u in units:
        pid = u["persistent_id"]
        if pid in outcomes:
            strata[u["stratum"]].append(pid)
            treated_counts[u["stratum"]] += int(u["treated"])
    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        permuted = []
        for stratum, pids in strata.items():
            selected = set(rng.sample(pids, treated_counts[stratum]))
            permuted.extend({"persistent_id": pid, "stratum": stratum,
                             "treated": pid in selected} for pid in pids)
        statistic, _ = _contrast(permuted, outcomes)
        if statistic is not None:
            null.append(statistic)
    if observed is None or not null:
        return None
    p = (1 + sum(abs(x) >= abs(observed) for x in null)) / (len(null) + 1)
    return {"two_sided_p_sharp_null_no_interference_assumption": p,
            "permutations": len(null), "null_95_interval": [
                sorted(null)[int(0.025 * (len(null)-1))],
                sorted(null)[int(0.975 * (len(null)-1))]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, nargs="?", default=Path(
        "results/legacy/phase14_generalization_postfix/mechanism_growth_starvation/randomized_local_trial/replicate0/tum_fr1_rpy/seed42_frame11.json"))
    ap.add_argument("--output-dir", type=Path, default=Path(
        "results/legacy/phase14_generalization_postfix/mechanism_growth_starvation/randomized_local_trial/replicate0"))
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    arm = data["arms"][0]
    assignment = arm["randomized_assignment"]
    units = [u for u in assignment["assignment_units"]
             if not u["stratum"].endswith("not_randomizable")]
    units = [{"persistent_id": int(u["persistent_id"]), "stratum": u["stratum"],
              "treated": bool(u["treated"])} for u in units]
    pre_rgb, pre_vis = _aggregate_by_id(arm["fixed_eval_before_action"],
                                        "rgb_attributed_abs_error_per_pixel")
    post_rgb, post_vis = _aggregate_by_id(arm["fixed_eval_after_horizon"],
                                          "rgb_attributed_abs_error_per_pixel")
    pre_depth, _ = _aggregate_by_id(arm["fixed_eval_before_action"],
                                    "depth_attributed_abs_error_per_pixel")
    post_depth, _ = _aggregate_by_id(arm["fixed_eval_after_horizon"],
                                     "depth_attributed_abs_error_per_pixel")
    outcomes = {}
    for metric, pre, post in (("rgb", pre_rgb, post_rgb), ("depth", pre_depth, post_depth)):
        outcomes[metric] = {pid: post[pid] - pre[pid] for pid in pre.keys() & post.keys()}
    result = {"source_file": str(args.input),
              "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "scene": data["scene"], "seed": data["seed"],
              "snapshot_frame": data["snapshot_frame"],
              "snapshot_hash": data["base_snapshot_hash"],
              "valid": bool(data["all_count_and_snapshot_gates_passed"] and arm["valid"]),
              "assignment": {k: assignment[k] for k in (
                  "seed", "cohort_requested", "cohort_present_and_visible",
                  "randomizable_ids", "treated_ids", "control_ids", "strata")},
              "fixed_view_frames": [x["frame"] for x in arm["fixed_eval_before_action"]],
              "fixed_view_psnr_deltas_db": {
                  str(before["frame"]): after["psnr"] - before["psnr"]
                  for before, after in zip(arm["fixed_eval_before_action"], arm["fixed_eval_after_horizon"])},
              "fixed_view_depth_l1_deltas": {
                  str(before["frame"]): after["depth_l1"] - before["depth_l1"]
                  for before, after in zip(arm["fixed_eval_before_action"], arm["fixed_eval_after_horizon"])},
              "per_id_visibility_fraction": {
                  "treated_pre": sum(pre_vis[u["persistent_id"]] for u in units if u["treated"])/max(1, sum(u["treated"] for u in units)),
                  "treated_post": sum(post_vis[u["persistent_id"]] for u in units if u["treated"])/max(1, sum(u["treated"] for u in units)),
                  "control_pre": sum(pre_vis[u["persistent_id"]] for u in units if not u["treated"])/max(1, sum(not u["treated"] for u in units)),
                  "control_post": sum(post_vis[u["persistent_id"]] for u in units if not u["treated"])/max(1, sum(not u["treated"] for u in units))},
              "effects": {}, "assignment_compliance": {}}
    trace = arm["identity_trace"]
    trace_index = {int(pid): i for i, pid in enumerate(trace["persistent_ids"])}
    compliance = {"treated_assigned": 0, "treated_updated_once": 0,
                  "control_assigned": 0, "control_not_updated": 0}
    for u in units:
        pid = u["persistent_id"]
        if pid not in trace_index:
            continue
        i = trace_index[pid]
        delta = int(trace["update_count_post"][i]) - int(trace["update_count_pre"][i])
        if u["treated"]:
            compliance["treated_assigned"] += 1
            compliance["treated_updated_once"] += int(delta == 1)
        else:
            compliance["control_assigned"] += 1
            compliance["control_not_updated"] += int(delta == 0)
    result["assignment_compliance"] = compliance
    for metric in ("rgb", "depth"):
        effect, strata = _contrast(units, outcomes[metric])
        result["effects"][metric] = {"weighted_stratified_updated_minus_control_change_per_pixel": effect,
                                     "strata": strata,
                                     "randomization_test": _randomization_test(units, outcomes[metric], effect,
                                         seed=7321 + (metric == "depth"))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Randomized persistent-ID update trial: first pilot", "",
        f"**Validity:** {'PASS' if result['valid'] else 'FAIL'} — shared snapshot `{result['snapshot_hash'][:12]}`, {result['assignment']['randomizable_ids']} randomized IDs ({result['assignment']['treated_ids']} update, {result['assignment']['control_ids']} control), no births.", "",
        "Within each prespecified age × staleness × current-view RGB-residual median stratum, eligible visible IDs were assigned approximately 1:1. Treated IDs received one optimizer update; controls received none. Outcomes are the change in top-8 alpha-contribution-weighted absolute residual per image pixel, averaged over the fixed future views. The mapping/update path used gsplat; the local attribution endpoint used the repository reference renderer.", "",
        "| Metric | Stratified treated-minus-control change | Randomization p (sharp null; no-interference assumption) |",
        "|---|---:|---:|"]
    for metric, label in (("rgb", "RGB"), ("depth", "Depth")):
        row = result["effects"][metric]
        p = row["randomization_test"]["two_sided_p_sharp_null_no_interference_assumption"]
        lines.append(f"| {label} attributed absolute error / pixel | {row['weighted_stratified_updated_minus_control_change_per_pixel']:+.8g} | {p:.4f} |")
    lines += ["", f"Fixed-view PSNR changes (frames {', '.join(map(str, result['fixed_view_frames']))}): " + ", ".join(f"{frame}: {delta:+.4f} dB" for frame, delta in result["fixed_view_psnr_deltas_db"].items()) + ".", "",
        f"Assignment compliance: {compliance['treated_updated_once']}/{compliance['treated_assigned']} treated IDs received exactly one recorded update; {compliance['control_not_updated']}/{compliance['control_assigned']} controls received none.", "",
        "This is one assignment replicate on one exposed TUM snapshot, so it is a mechanism pilot, not generalizable evidence. Randomization p-values assume no interference between Gaussians; 3DGS compositing and shared optimizer state can violate that assumption. The contribution attribution is top-8 truncated and uses a reference renderer, so its magnitude is diagnostic and is not numerically interchangeable with gsplat's rendered PSNR. Repeat the assignment from the same full checkpoint with independent randomization seeds, then replicate across snapshots/scenes.", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
