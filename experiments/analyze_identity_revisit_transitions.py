#!/usr/bin/env python3
"""Descriptive matched-stratum analysis of persistent-ID revisit residuals.

The allocation is observational: updated IDs were selected by the scheduler.
This script matches update/non-update events within seed, arm, source frame,
revisit lag, and pre-action staleness bin. Statistical resampling is by seed.
"""

from __future__ import annotations

import argparse
import hashlib
import glob
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path("results/phase14_generalization_postfix/mechanism_growth_starvation/late_snapshot_fixedref_identity_multiseed")
SCENES = {
    "tum_fr1_rpy": ("tum_run/tum_fr1_rpy/*.json", "TUM fr1/rpy"),
    "replica_office1": ("replica_run/replica_office1/*.json", "Replica office1"),
}


def stale_bin(value: float) -> str:
    if value <= 2:
        return "0–2"
    if value <= 9:
        return "3–9"
    if value <= 19:
        return "10–19"
    return "20+"


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] * (hi - x) + values[hi] * (x - lo)


def bootstrap_ci(seed_values: list[float], seed: int = 20261021, n_boot: int = 20000):
    rng = random.Random(seed)
    n = len(seed_values)
    means = [sum(seed_values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    return percentile(means, 0.025), percentile(means, 0.975)


def finite_float(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "revisit_transition_analysis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out = {"analysis": "matched persistent-ID revisit transition association",
           "matching": ["seed", "arm", "source_frame", "revisit_lag", "pre-update staleness bin"],
           "inference_unit": "seed", "input_sha256": {}, "scenes": {}}
    report = ["# Persistent-ID revisit transition analysis", "",
              "**Status:** descriptive, post hoc association; not a causal estimate.", "",
              "For each cohort ID, the analysis pairs consecutive visible observations only when at least one sampled frame in between was not visible (a revisit). It measures the change in current-view pre-optimizer residual from the source observation to the next visible observation. Within each seed, arm, source frame, revisit lag, and source-frame staleness bin, it contrasts events whose ID received at least one micro-update after the source observation with events that received none. The stratum contrasts are averaged equally within seed. Confidence intervals resample seeds, not ID-frame events.", "",
              "This matching reduces frame/camera and revisit-interval differences. It does not remove scheduler selection bias, changing view content, residual-dependent selection, or post-treatment map changes. Residuals are current-view center-sampled attribution values, not contribution-weighted fixed-view outcomes. Interpret as association only.", ""]

    for scene_key, (pattern, scene_name) in SCENES.items():
        records = []
        for filename in sorted(glob.glob(str(ROOT / pattern))):
            out["input_sha256"][str(Path(filename))] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            data = json.loads(Path(filename).read_text())
            seed = int(data["seed"])
            for arm in data["arms"]:
                action = arm["action"]
                arm_name = action["name"]
                spawn_count = int(action["spawn_count"])
                allocation = action["allocation"]
                traces = [arm["identity_trace"]] + [m["identity_trace"] for m in arm["horizon_frame_metrics"]]
                frames = [int(arm["frame"])] + [int(m["frame"]) for m in arm["horizon_frame_metrics"]]
                strata = defaultdict(lambda: {False: [], True: []})
                visible_transitions = 0
                revisit_transitions = 0
                positions = []
                for trace in traces:
                    positions.append({int(pid): i for i, pid in enumerate(trace["persistent_ids"])})
                all_ids = sorted(set().union(*(set(p) for p in positions)))
                for pid in all_ids:
                    visible_at = []
                    for t, trace in enumerate(traces):
                        i = positions[t].get(pid)
                        if i is not None and trace["present_pre"][i] and trace["visible_pre"][i]:
                            visible_at.append((t, i))
                    for (t, i), (u, j) in zip(visible_at, visible_at[1:]):
                        if u - t < 2:
                            continue
                        current, nxt = traces[t], traces[u]
                        lag = frames[u] - frames[t]
                        before_rgb = finite_float(current["rgb_residual_pre"][i])
                        after_rgb = finite_float(nxt["rgb_residual_pre"][j])
                        if before_rgb is None or after_rgb is None:
                            continue
                        visible_transitions += 1
                        revisit_transitions += 1
                        updated = int(current["update_count_post"][i]) > int(current["update_count_pre"][i])
                        key = (seed, arm_name, frames[t], lag, stale_bin(float(current["staleness_frames_pre"][i])))
                        val = {"rgb": after_rgb - before_rgb, "depth": None}
                        before_depth = finite_float(current["depth_residual_pre"][i])
                        after_depth = finite_float(nxt["depth_residual_pre"][j])
                        if before_depth is not None and after_depth is not None:
                            val["depth"] = after_depth - before_depth
                        strata[key][updated].append(val)

                grouped = defaultdict(list)
                matched_strata = 0
                for key, classes in strata.items():
                    if not classes[False] or not classes[True]:
                        continue
                    matched_strata += 1
                    for metric in ("rgb", "depth"):
                        a = [v[metric] for v in classes[True] if v[metric] is not None]
                        b = [v[metric] for v in classes[False] if v[metric] is not None]
                        if a and b:
                            grouped[(key[0], metric)].append(sum(a) / len(a) - sum(b) / len(b))
                by_metric_seed = defaultdict(dict)
                for (s, metric), contrasts in grouped.items():
                    by_metric_seed[metric][s] = sum(contrasts) / len(contrasts)
                records.append({"seed": seed, "arm": arm_name, "spawn_count": spawn_count,
                                "allocation": allocation, "visible_transitions": visible_transitions,
                                "revisit_transitions": revisit_transitions,
                                "matched_strata": matched_strata,
                                "seed_contrast": {metric: {str(s): v for s, v in values.items()}
                                                  for metric, values in by_metric_seed.items()}})

        summaries = []
        arm_keys = sorted({(r["arm"], r["spawn_count"], r["allocation"]) for r in records})
        for arm_name, spawn_count, allocation in arm_keys:
            arm_records = [r for r in records if r["arm"] == arm_name]
            row = {"arm": arm_name, "spawn_count": spawn_count, "allocation": allocation,
                   "n_seed_files": len(arm_records),
                   "visible_transitions": sum(r["visible_transitions"] for r in arm_records),
                   "revisit_transitions": sum(r["revisit_transitions"] for r in arm_records),
                   "matched_strata": sum(r["matched_strata"] for r in arm_records),
                   "metrics": {}}
            for metric in ("rgb", "depth"):
                values = []
                for rec in arm_records:
                    sv = rec["seed_contrast"].get(metric, {})
                    if str(rec["seed"]) in sv:
                        values.append((rec["seed"], float(sv[str(rec["seed"])])))
                if values:
                    vals = [v for _, v in values]
                    lo, hi = bootstrap_ci(vals, seed=20261021 + spawn_count + (0 if metric == "rgb" else 1))
                    row["metrics"][metric] = {"n_seeds": len(vals), "mean_updated_minus_not_updated_delta": sum(vals)/len(vals),
                                               "seed_bootstrap_95_ci": [lo, hi],
                                               "seed_values": {str(s): v for s, v in values}}
            summaries.append(row)
        out["scenes"][scene_key] = {"display_name": scene_name, "arm_summaries": summaries, "seed_arm_details": records}
        report += [f"## {scene_name}", "", "Contrast is (next-visible residual − source residual) for updated IDs minus the same change for non-updated IDs; negative means residual fell more in the updated class.", "",
                   "| Arm | Seed n | Revisit events | Matched strata | RGB contrast (95% seed bootstrap CI) | Depth contrast (95% seed bootstrap CI) |",
                   "|---|---:|---:|---:|---:|---:|"]
        for row in summaries:
            rgb = row["metrics"].get("rgb")
            dep = row["metrics"].get("depth")
            fmt = lambda x: "—" if x is None else f"{x['mean_updated_minus_not_updated_delta']:+.6f} [{x['seed_bootstrap_95_ci'][0]:+.6f}, {x['seed_bootstrap_95_ci'][1]:+.6f}]"
            report.append(f"| {row['arm']} | {rgb['n_seeds'] if rgb else 0} | {row['revisit_transitions']} | {row['matched_strata']} | {fmt(rgb)} | {fmt(dep)} |")
        report += [""]

    report += ["## Interpretation", "",
               "The strict within-source-frame matched comparison is not supported by adequate overlap in these traces. TUM contains revisit observations but no matched strata with both update classes. Replica has only sparse overlap, concentrated in the stale-rescue arms; global-allocation arms have no matched strata. Therefore this dataset cannot tell us whether an individual update reduced that same Gaussian's residual on revisit after controlling frame, revisit lag, and staleness. The missing support is a design limitation, not evidence of no effect.", "",
               "The prior fixed-view cohort endpoint remains the stronger individual-ID diagnostic, but it is center-sampled and has low common-visible support. A next experiment should predefine repeated-revisit cohorts and ensure both treated and untreated IDs occur in matched strata, or randomize updates within a common candidate pool and measure contribution-aware fixed-view outcomes. Do not interpret this analysis as evidence that starvation causes quality loss.", ""]

    (args.output_dir / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
