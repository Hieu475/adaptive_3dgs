#!/usr/bin/env python3
"""Attach policy-independent office4 input features to the frame-70 quota replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_phase13_frozen_benchmark import load_phase10_sequence
from experiments.run_postfix_frame_diagnostics import scene_features

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mean(xs):
    return statistics.mean(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "results/phase14_generalization_postfix/office4_quota_dose_response_frame70_seeds291_293")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.root.resolve()
    prereg = json.loads((out / "preregistration.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    frame_id = prereg["snapshot_frames"][0]
    frames, intrinsics = load_phase10_sequence(scene_name=prereg["scene"], n_frames=frame_id + 1,
                                                H=240, W=320, device=args.device)
    input_features = scene_features(frames, intrinsics)[frame_id]
    seed_rows, diagnostics, source_hashes = [], [], {}
    policies = ["fixed25", "fixed40", "fixed49", "fixed60", "fixed75", "adaptive"]
    for seed in prereg["seeds"]:
        cell_path = out / prereg["scene"] / f"seed{seed}_frame{frame_id}.json"
        cell = json.loads(cell_path.read_text())
        assert cell["all_count_and_snapshot_gates_passed"] and len(cell["arms"]) == 60
        source_hashes[str(seed)] = {"snapshot_hash": cell["base_snapshot_hash"],
                                    "map_snapshot_sha256": cell["map_snapshot_sha256"],
                                    "replay_checkpoint_sha256": cell["replay_checkpoint_sha256"],
                                    "candidate_pool_sha256": cell["common_candidate_pool_sha256"]}
        by_policy = {p: [] for p in policies}
        for arm in cell["arms"]:
            assert arm["valid"] and arm["snapshot_hash"] == cell["base_snapshot_hash"]
            assert arm["n_spawned"] == 128 and arm["n_micro_updates"] == 5000
            if arm["quota_policy"] == "coverage_saturating":
                label = "adaptive"
                expected = min(3750, arm["mature_eligible"])
            else:
                pct = round(100 * arm["action"]["mature_fraction"])
                label = f"fixed{pct}"
                expected = round(5000 * arm["action"]["mature_fraction"])
            assert arm["realized_mature_micro_updates"] == expected
            by_policy[label].append(arm)
        assert all(len(by_policy[p]) == 10 for p in policies)
        for label, arms in by_policy.items():
            seed_rows.append({"seed": seed, "policy": label,
                              "psnr": mean([a["final_map_eval_psnr"] for a in arms]),
                              "depth_l1": mean([a["final_map_eval_depth_l1"] for a in arms]),
                              "mature_share": mean([a["realized_mature_fraction"] for a in arms]),
                              "mature_coverage": mean([a["mature_update_coverage"] for a in arms]),
                              "opt_stage_sync_ms": mean([a["opt_stage_sync_ms"] for a in arms])})
        diag = cell["arms"][0]["age_stratified_diagnostics"]
        assert all(a["age_stratified_diagnostics"] == diag for a in cell["arms"])
        diagnostics.append(diag)

    means = {}
    for policy in policies:
        rows = [r for r in seed_rows if r["policy"] == policy]
        means[policy] = {k: mean([r[k] for r in rows]) for k in ("psnr", "depth_l1", "mature_share", "mature_coverage", "opt_stage_sync_ms")}
    base = means["fixed25"]
    young = {k: mean([d["young_lt_10_frames"][k] for d in diagnostics])
             for k in ("visible_color_error_mean", "visible_depth_error_mean", "visible_attributed")}
    mature = {k: mean([d["mature_ge_10_frames"][k] for d in diagnostics])
              for k in ("visible_color_error_mean", "visible_depth_error_mean", "visible_attributed")}
    stale = [d["mature_revisit_strata"]["staleness_20_plus_frames"] for d in diagnostics]
    stale_mean = {k: mean([d[k] for d in stale]) for k in ("visible_attributed", "mean_staleness_frames", "mean_update_count", "visible_color_error_mean", "visible_depth_error_mean")}

    fixed_policies = [p for p in policies if p.startswith("fixed")]
    best_fixed = max(fixed_policies, key=lambda p: means[p]["psnr"])
    summary = {"scene": prereg["scene"], "frame": frame_id, "seeds": prereg["seeds"],
               "status": "exploratory cross-scene state diagnostic", "input_feature_definition": "5cm world-space voxel novelty at 4-pixel stride, using valid RGB-D and Phase 10 ground-truth poses; policy independent",
               "input_features": input_features, "policy_means": means,
               "best_fixed_policy_by_psnr": best_fixed,
               "delta_vs_fixed25": {p: {"psnr": means[p]["psnr"]-base["psnr"], "depth_l1": means[p]["depth_l1"]-base["depth_l1"]} for p in policies},
               "age_diagnostics": {"young": young, "mature": mature,
                                   "young_to_mature_color_ratio": young["visible_color_error_mean"]/mature["visible_color_error_mean"],
                                   "young_to_mature_depth_ratio": young["visible_depth_error_mean"]/mature["visible_depth_error_mean"],
                                   "mature_staleness_20plus": stale_mean},
               "valid_cells": len(prereg["seeds"]), "valid_branches": len(seed_rows)*10,
               "source_hashes": source_hashes,
               "preregistration_sha256": sha256(out/"preregistration.json"),
               "manifest_sha256": sha256(out/"manifest.json"),
               "config_sha256": manifest["config_sha256"],
               "feature_code_sha256": sha256(Path(__file__).with_name("run_postfix_frame_diagnostics.py")),
               "analysis_code_sha256": sha256(Path(__file__).resolve())}

    lines = [f"# {prereg['scene']} frame-{frame_id} quota/state diagnostics", "", "## Validity and paired protocol", "", f"- Three new seed snapshots; {len(seed_rows)*10} measured branches. Each branch passed exact 128 spawn, 5000 update, quota, snapshot and candidate-pool gates.", "- Scene-state inputs are computed from RGB-D and ground-truth pose only; map-dependent visibility, age residual and staleness diagnostics come from the exact replay snapshots.", f"- Backend `{manifest['renderer_backend']}`; GPU `{manifest['gpu']}`; revision `{manifest['git_sha']}`.", "", "## Input and map state", "", "| Translation m/frame | Rotation deg/frame | 5cm novelty | Valid depth | Map view coverage | Young/mature color residual | Ratio | Young/mature depth residual | Ratio |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"| {input_features['camera_translation_m_per_frame']:.4f} | {input_features['camera_angular_deg_per_frame']:.2f} | {input_features['new_surface_ratio_voxel_5cm']:.4f} | {input_features['depth_valid_ratio_sampled']:.3f} | {mean([d['view_coverage_valid_depth'] for d in diagnostics]):.3f} | {young['visible_color_error_mean']:.5f} / {mature['visible_color_error_mean']:.5f} | {summary['age_diagnostics']['young_to_mature_color_ratio']:.2f}× | {young['visible_depth_error_mean']:.5f} / {mature['visible_depth_error_mean']:.5f} | {summary['age_diagnostics']['young_to_mature_depth_ratio']:.2f}× |")
    lines += ["", "| Policy | Mature share | Mature coverage | ΔPSNR vs 25% | Δdepth L1 vs 25% | Sync optimizer ms |", "|---|---:|---:|---:|---:|---:|"]
    for p in policies:
        m=means[p]; delta=summary["delta_vs_fixed25"][p]
        lines.append(f"| {p} | {m['mature_share']:.3f} | {m['mature_coverage']:.3f} | {delta['psnr']:+.4f} dB | {delta['depth_l1']:+.5f} m | {m['opt_stage_sync_ms']:.2f} |")
    best_delta = summary["delta_vs_fixed25"][best_fixed]
    lines += ["", f"In the 20+ frame mature staleness stratum, mean visible count was {stale_mean['visible_attributed']:.0f}, mean staleness {stale_mean['mean_staleness_frames']:.1f} frames, mean prior update count {stale_mean['mean_update_count']:.2f}, color residual {stale_mean['visible_color_error_mean']:.5f}, and depth residual {stale_mean['visible_depth_error_mean']:.5f}.", "", "## Interpretation", "", f"The best tested fixed quota by mean PSNR was {best_fixed} ({best_delta['psnr']:+.4f} dB vs fixed25%). The fixed-75% arm changed PSNR by {summary['delta_vs_fixed25']['fixed75']['psnr']:+.4f} dB and depth L1 by {summary['delta_vs_fixed25']['fixed75']['depth_l1']:+.5f} m. At this state the young/mature residual channels differ: young/mature color ratio is {summary['age_diagnostics']['young_to_mature_color_ratio']:.2f}× and depth ratio is {summary['age_diagnostics']['young_to_mature_depth_ratio']:.2f}×. These aggregate residual ratios do not by themselves identify the best quota.", "", "This is one snapshot in a quota-tested scene, with three seeds. It is exploratory transfer, not broad generalization. The comparison matches update count, not wall time; metrics are same-view RGB-D, with no geometry completeness, novel-view, or trajectory-level outcome. The manifest includes a dirty-worktree marker.", "", "## Provenance", "", f"- Preregistration SHA-256: `{summary['preregistration_sha256']}`", f"- Run manifest SHA-256: `{summary['manifest_sha256']}`", f"- Feature code SHA-256: `{summary['feature_code_sha256']}`", "- Per-seed snapshot/map/checkpoint/candidate hashes: `summary.json`", ""]
    (out/"state_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (out/"state_report.md").write_text("\n".join(lines))
    launch={"protocol_version":prereg["protocol_version"],"preregistration_sha256":summary["preregistration_sha256"],"manifest_sha256":summary["manifest_sha256"],"state_summary_sha256":sha256(out/"state_summary.json"),"analysis_code_sha256":summary["analysis_code_sha256"]}
    (out/"state_launch_config.json").write_text(json.dumps(launch,indent=2,sort_keys=True)+"\n")
    print(f"features {input_features}; best fixed policy {best_fixed}; adaptive-vs25 {summary['delta_vs_fixed25']['adaptive']}")


if __name__ == "__main__":
    main()
