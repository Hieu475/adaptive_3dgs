#!/usr/bin/env python3
"""Compare repeated exploratory TUM fr1 desk long-horizon runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/continuous_age_quota/long_horizon_tum_fr1_desk"
RUNS = {
    "plain_run_1": "run_seed42",
    "diagnostic_run": "diagnostics_seed42",
    "plain_run_2": "repeat_seed42",
}
OUT = BASE / "repeatability_analysis.json"
REPORT = BASE / "repeatability_report.md"


def load(run, arm):
    path = BASE / run / "tum_fr1_desk" / f"seed42_{arm}.json"
    return json.loads(path.read_text())


def main():
    data = {run: {arm: load(folder, arm) for arm in ("global_importance", "mature75")}
            for run, folder in RUNS.items()}
    out = {"study": "exploratory_single_seed_long_horizon_repeatability",
           "scene": "tum_fr1_desk", "seed": 42, "runs": {}}
    lines = [
        "# Exploratory 500-frame trajectory repeatability audit",
        "",
        "Three executions of TUM fr1 desk, seed 42, with identical 500-frame input, 5,000 updates per frame and the same two arms. Run 1 and run 2 did not capture age diagnostics; the diagnostic run enabled read-only age summaries. All are single-seed exploratory executions, not independent samples.",
        "",
        "## Run summaries",
        "",
        "| Run | Diagnostics | Arm | Mean PSNR | Final PSNR | Mean depth L1 | Final Gaussians | Mean frame wall (ms) | Exact quota fraction |",
        "|:--|:--:|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for run, arms in data.items():
        for arm, obj in arms.items():
            s = obj["summary"]
            lines.append(f"| {run} | {'yes' if run == 'diagnostic_run' else 'no'} | {arm} | {s['mean_psnr']:.3f} | {s['final_psnr']:.3f} | {s['mean_depth_l1']:.5f} | {s['final_n_gaussians']} | {s['mean_frame_wall_ms']:.1f} | {s['exact_age_split_fraction'] if s['exact_age_split_fraction'] is not None else '—'} |")
            assert s["update_count_gate_pass"] and s["n_frames_including_init"] == 500
            out["runs"].setdefault(run, {})[arm] = s
    lines += [
        "",
        "## Per-frame deviation from the first plain run",
        "",
        "| Arm | Compared run | Metric | First differing frame | Mean absolute difference | Maximum absolute difference |",
        "|:--|:--|:--|--:|--:|--:|",
    ]
    deviations = {}
    for arm in ("global_importance", "mature75"):
        ref = data["plain_run_1"][arm]["frames"]
        for other in ("diagnostic_run", "plain_run_2"):
            cur = data[other][arm]["frames"]
            assert len(ref) == len(cur) == 499
            assert all(a["frame"] == b["frame"] and a["n_micro_updates"] == b["n_micro_updates"] == 5000
                       for a, b in zip(ref, cur))
            for key in ("psnr", "depth_l1", "n_gaussians"):
                diffs = np.array([float(b[key]) - float(a[key]) for a, b in zip(ref, cur)])
                first = next((i + 1 for i, v in enumerate(diffs) if abs(v) > 1e-6), None)
                entry = {"arm": arm, "compared_run": other, "metric": key,
                         "first_differing_frame": first,
                         "mean_absolute_difference": float(np.mean(np.abs(diffs))),
                         "maximum_absolute_difference": float(np.max(np.abs(diffs)))}
                deviations[f"{arm}:{other}:{key}"] = entry
                lines.append(f"| {arm} | {other} | {key} | {first} | {entry['mean_absolute_difference']:.5f} | {entry['maximum_absolute_difference']:.5f} |")
    out["deviations_from_first_plain_run"] = deviations
    lines += [
        "",
        "## Policy-effect trajectory across executions",
        "",
        "| Run | Frames | Mean paired PSNR delta (mature75 − global) | Final-frame delta | Mean paired depth-L1 delta |",
        "|:--|:--:|--:|--:|--:|",
    ]
    effect_rows = {}
    for run, arms in data.items():
        g = arms["global_importance"]["frames"]
        q = arms["mature75"]["frames"]
        psnr_delta = np.array([b["psnr"] - a["psnr"] for a, b in zip(g, q)])
        depth_delta = np.array([b["depth_l1"] - a["depth_l1"] for a, b in zip(g, q)])
        row = {"mean_psnr_delta": float(np.mean(psnr_delta)),
               "final_frame_psnr_delta": float(psnr_delta[-1]),
               "mean_depth_l1_delta": float(np.mean(depth_delta))}
        effect_rows[run] = row
        lines.append(f"| {run} | {len(g)} | {row['mean_psnr_delta']:+.3f} | {row['final_frame_psnr_delta']:+.3f} | {row['mean_depth_l1_delta']:+.5f} |")
    out["policy_effects"] = effect_rows
    lines += [
        "",
        "## Interpretation",
        "",
        "The repeated plain run itself differs from the first plain run despite identical seed and protocol. The diagnostic-enabled run also differs, but these three executions cannot separate instrumentation effects from GPU/backend nondeterminism, random tie behavior, or trajectory sensitivity. Small early metric deviations can compound into substantially different map and quality trajectories. The age-stratified diagnostics therefore cannot be compared directly with the first plain trajectory as though they were a matched counterfactual.",
        "",
        "Do not use the instrumented run to explain the original late regression yet. First establish run-to-run variability with multiple paired seeds and/or a deterministic replay/snapshot protocol, and test whether diagnostics leave the trajectory unchanged. These data are one seed on one scene; no p-values are reported. Clean-checkout replication remains required.",
        "",
        "See the [repeatability preregistration](repeatability_preregistration.md), [instrumented diagnostics preregistration](diagnostics_preregistration.md), and [original 500-frame report](run_seed42/report.md).",
    ]
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
