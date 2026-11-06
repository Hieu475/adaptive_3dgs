#!/usr/bin/env python3
"""Validate and summarize the preregistered coverage-saturating snapshot replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean(xs):
    return statistics.mean(xs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    prereg = json.loads((root / "preregistration.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    scene_name = prereg["scene"]
    total_updates = int(prereg["replay"]["total_micro_updates_per_action"])
    cap = float(prereg["replay"].get("coverage_saturating_cap", 0.75))
    cells = sorted((root / scene_name).glob("seed*_frame*.json"))
    expected = {(seed, frame) for seed in prereg["seeds"] for frame in prereg["snapshot_frames"]}
    records = {}
    checkpoint_hashes = {}
    map_hashes = {}
    for path in cells:
        cell = json.loads(path.read_text())
        key = (cell["seed"], cell["snapshot_frame"])
        assert key not in records, f"duplicate cell {key}"
        records[key] = cell
        assert cell["all_count_and_snapshot_gates_passed"], f"cell gates failed: {key}"
        assert len(cell["arms"]) == prereg["replay"]["repeats_per_arm"] * 3, f"wrong arm count: {key}"
        for field, dest in (("replay_checkpoint_path", checkpoint_hashes), ("map_snapshot_path", map_hashes)):
            rel = cell[field]
            actual = sha256(root / rel)
            expected_hash = cell[field.replace("_path", "_sha256")]
            assert actual == expected_hash, f"artifact hash mismatch: {key} {rel}"
            dest[rel] = actual
        for arm in cell["arms"]:
            assert arm["valid"], f"invalid arm: {key} {arm['action']['name']}"
            assert arm["n_spawned"] == arm["target_spawned"] == 128
            assert arm["n_micro_updates"] == arm["target_micro_updates"] == total_updates
            assert arm["snapshot_hash"] == cell["base_snapshot_hash"]
            for metric in ("final_map_eval_psnr", "final_map_eval_depth_l1", "opt_stage_sync_ms", "mature_update_coverage", "realized_mature_fraction"):
                assert math.isfinite(arm[metric]), f"nonfinite {metric}: {key}"
            policy = arm["quota_policy"]
            mature = arm["realized_mature_micro_updates"]
            if policy == "fixed_fraction":
                fraction = arm["action"]["mature_fraction"]
                assert mature == round(fraction * total_updates), f"fixed age quota mismatch: {key} {arm['action']['name']}"
            elif policy == "coverage_saturating":
                expected_mature = min(round(cap * total_updates), arm["mature_eligible"])
                assert mature == expected_mature, f"adaptive rule mismatch: {key} {arm['action']['name']}"
            else:
                raise AssertionError(f"unknown quota policy: {policy}")
    assert set(records) == expected, f"cell mismatch; missing={expected-set(records)}, extra={set(records)-expected}"

    by_cell_policy = defaultdict(list)
    for (seed, frame), cell in records.items():
        for arm in cell["arms"]:
            name = arm["action"]["name"]
            policy = "adaptive" if "coverage_saturating" in name else ("mature25" if "mature25" in name else "mature75")
            by_cell_policy[(frame, seed, policy)].append(arm)
    for key, arms in by_cell_policy.items():
        assert len(arms) == 10, f"repeat count mismatch: {key}"

    summary = {"experiment": prereg["title"], "manifest_sha256": sha256(root / "manifest.json"), "preregistration_sha256": sha256(root / "preregistration.json"), "cells": len(records), "arms": sum(len(c["arms"]) for c in records.values()), "valid_arms": sum(bool(a["valid"]) for c in records.values() for a in c["arms"]), "checkpoint_sha256": checkpoint_hashes, "map_snapshot_sha256": map_hashes, "by_frame": {}}
    report = ["# Coverage-saturating allocation: held-out snapshot replay", "", "## Protocol and validity", "", f"- Scene: `{prereg['scene']}`; seeds: {', '.join(map(str, prereg['seeds']))}; snapshot frames: {', '.join(map(str, prereg['snapshot_frames']))}.", f"- Same snapshot and candidate pool within each seed/frame cell; each arm added exactly 128 Gaussians and performed exactly {total_updates:,} micro-updates. Ten repeats per arm.", f"- Fixed controls used exact 25% and 75% mature-update quotas. Adaptive rule (preregistered): `min(round({cap:g} × total_updates), visible_mature_eligible_count)` mature updates, with remainder allocated to young candidates.", f"- Validity: {len(records)} / {len(expected)} cells passed; {summary['valid_arms']} / {summary['arms']} branches passed. Snapshot and replay-checkpoint SHA-256 values match their records.", f"- Manifest Git revision: `{manifest['git_sha']}`; backend: `{manifest['renderer_backend']}`; GPU: `{manifest['gpu']}`; pose protocol: {prereg['pose_protocol']}", "", "## Results", "", "Values below average ten action repeats in each seed/frame cell; Δ is adaptive minus the fixed arm. Positive PSNR and negative depth L1 differences favor adaptive. Seeds are the independent held-out units; repeats are not treated as independent scenes.", ""]
    for frame in prereg["snapshot_frames"]:
        per_seed = []
        for seed in prereg["seeds"]:
            row = {"seed": seed}
            for policy in ("mature25", "mature75", "adaptive"):
                arms = by_cell_policy[(frame, seed, policy)]
                row[policy] = {k: mean([a[k] for a in arms]) for k in ("final_map_eval_psnr", "final_map_eval_depth_l1", "opt_stage_sync_ms", "realized_mature_fraction", "mature_update_coverage", "mature_eligible", "realized_mature_micro_updates")}
            row["delta_vs_25"] = {"psnr": row["adaptive"]["final_map_eval_psnr"] - row["mature25"]["final_map_eval_psnr"], "depth_l1": row["adaptive"]["final_map_eval_depth_l1"] - row["mature25"]["final_map_eval_depth_l1"], "opt_ms_pct": 100 * (row["adaptive"]["opt_stage_sync_ms"] / row["mature25"]["opt_stage_sync_ms"] - 1)}
            row["delta_vs_75"] = {"psnr": row["adaptive"]["final_map_eval_psnr"] - row["mature75"]["final_map_eval_psnr"], "depth_l1": row["adaptive"]["final_map_eval_depth_l1"] - row["mature75"]["final_map_eval_depth_l1"], "opt_ms_pct": 100 * (row["adaptive"]["opt_stage_sync_ms"] / row["mature75"]["opt_stage_sync_ms"] - 1)}
            per_seed.append(row)
        summary["by_frame"][str(frame)] = per_seed
        report += [f"### Frame {frame}", "", "| Seed | Mature eligible | Adaptive mature share | Mature coverage (25 / 75 / adaptive) | ΔPSNR vs 25% | Δdepth L1 vs 25% | ΔPSNR vs 75% | Δdepth L1 vs 75% | Adaptive opt time vs 25 / 75 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in per_seed:
            vals = [row["seed"], round(row["adaptive"]["mature_eligible"]), f"{row['adaptive']['realized_mature_fraction']:.3f}", f"{row['mature25']['mature_update_coverage']:.3f} / {row['mature75']['mature_update_coverage']:.3f} / {row['adaptive']['mature_update_coverage']:.3f}", f"{row['delta_vs_25']['psnr']:+.4f}", f"{row['delta_vs_25']['depth_l1']:+.5f}", f"{row['delta_vs_75']['psnr']:+.4f}", f"{row['delta_vs_75']['depth_l1']:+.5f}", f"{row['delta_vs_25']['opt_ms_pct']:+.1f}% / {row['delta_vs_75']['opt_ms_pct']:+.1f}%"]
            report.append("| " + " | ".join(map(str, vals)) + " |")
        for control, label in (("mature25", "25%"), ("mature75", "75%")):
            dps = [r[f"delta_vs_{'25' if control=='mature25' else '75'}"]["psnr"] for r in per_seed]
            ddepth = [r[f"delta_vs_{'25' if control=='mature25' else '75'}"]["depth_l1"] for r in per_seed]
            dtime = [r[f"delta_vs_{'25' if control=='mature25' else '75'}"]["opt_ms_pct"] for r in per_seed]
            report.append(f"- Across the three seeds, adaptive − {label}: mean ΔPSNR **{mean(dps):+.4f} dB**, mean Δdepth L1 **{mean(ddepth):+.5f} m**, mean optimizer-stage synchronized time **{mean(dtime):+.2f}%**; direction by seed (|ΔPSNR| ≤ 0.0001 dB counted as tied): {sum(v>1e-4 for v in dps)} positive, {sum(v< -1e-4 for v in dps)} negative, {sum(abs(v)<=1e-4 for v in dps)} tied.")
        report.append("")
    report += ["## Interpretation", ""]
    for frame in prereg["snapshot_frames"]:
        rows = summary["by_frame"][str(frame)]
        eligible = mean([r["adaptive"]["mature_eligible"] for r in rows])
        share = mean([r["adaptive"]["realized_mature_fraction"] for r in rows])
        d25 = mean([r["delta_vs_25"]["psnr"] for r in rows])
        d75 = mean([r["delta_vs_75"]["psnr"] for r in rows])
        d75_l1 = mean([r["delta_vs_75"]["depth_l1"] for r in rows])
        report.append(f"- Frame {frame}: mean mature-eligible pool {eligible:.0f}; adaptive mature share {share:.1%}; mean ΔPSNR adaptive−25% {d25:+.4f} dB, adaptive−75% {d75:+.4f} dB (Δdepth L1 vs 75% {d75_l1:+.5f} m).")
        age_diags = [records[(seed, frame)]["arms"][0].get("age_stratified_diagnostics") for seed in prereg["seeds"]]
        if all(age_diags):
            young_color = mean([d["young_lt_10_frames"]["visible_color_error_mean"] for d in age_diags])
            mature_color = mean([d["mature_ge_10_frames"]["visible_color_error_mean"] for d in age_diags])
            young_depth = mean([d["young_lt_10_frames"]["visible_depth_error_mean"] for d in age_diags])
            mature_depth = mean([d["mature_ge_10_frames"]["visible_depth_error_mean"] for d in age_diags])
            report.append(f"- Pre-action attributed residuals averaged across seeds: young/mature visible color error {young_color:.5f}/{mature_color:.5f} ({young_color/mature_color:.2f}×), depth error {young_depth:.5f}/{mature_depth:.5f} ({young_depth/mature_depth:.2f}×). Descriptive state diagnostics; not a randomized test of residual-based allocation.")
    scope = prereg.get("development_and_scope", {})
    limitations = prereg.get("interpretation_limits", [])
    if scope:
        report.append(f"- Scope: {scope.get('prior_scene_use', '')} {scope.get('rule_holdout', '')} {scope.get('selection_data', '')}")
        report.append(f"- Limitation: {scope.get('limitation', '')}")
    if limitations:
        report.append("- Preregistered limits: " + " ".join(limitations))
    report += ["", "This is same-snapshot, matched-update evidence conditional on the tested scene state. It does not establish full-trajectory gains, cross-scene generalization, or geometry completeness. Ten repeats reduce action-level noise but do not increase the number of independent scenes. The comparison matches micro-update count, not wall time: synchronized optimizer-stage timing is reported descriptively. No novel-view, full-pipeline real-time, or long-horizon outcome is measured.", "", "Git provenance is marked dirty in the generated manifest; do not present this artifact as a clean-checkout reproduction.", "", "## Provenance", "", f"- Preregistration SHA-256: `{summary['preregistration_sha256']}`", f"- Manifest SHA-256: `{summary['manifest_sha256']}`", f"- Commit field: `{manifest['git_sha']}`", "- Exact input snapshot and replay-checkpoint hashes are recorded in `launch_config.json` and `summary.json`.", ""]
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "report.md").write_text("\n".join(report))
    launch = {"experiment": prereg["title"], "protocol_version": prereg["protocol_version"], "preregistration_sha256": summary["preregistration_sha256"], "manifest_sha256": summary["manifest_sha256"], "git_sha": manifest["git_sha"], "dirty_worktree": manifest["git_sha"].endswith("-dirty"), "config_sha256": manifest["config_sha256"], "checkpoint_sha256": checkpoint_hashes, "map_snapshot_sha256": map_hashes}
    (root / "launch_config.json").write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {root/'report.md'}; {len(records)} valid cells, {summary['valid_arms']} valid arms")
    for frame, rows in summary["by_frame"].items():
        for control in ("25", "75"):
            p = [r[f"delta_vs_{control}"]["psnr"] for r in rows]
            d = [r[f"delta_vs_{control}"]["depth_l1"] for r in rows]
            print(f"frame={frame} adaptive-{control}: PSNR {mean(p):+.5f} dB; depth {mean(d):+.6f} m; seed signs {sum(x>0 for x in p)}/3")


if __name__ == "__main__":
    main()
