#!/usr/bin/env python3
"""Join quota replay outcomes to policy-independent frame input diagnostics.

This is a descriptive four-cell analysis. It deliberately does not fit a router
or pool the separate scene/frame studies.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/phase14_generalization_postfix"
CELLS = [
    ("replica_office1", 70, "quota_response_frame70"),
    ("replica_office1", 90, "quota_response_frame90"),
    ("tum_fr1_rpy", 70, "quota_response_tum_fr1_rpy_frame70"),
    ("tum_fr1_rpy", 90, "quota_response_tum_fr1_rpy_frame90"),
]
STATIC = (
    "camera_translation_m_per_frame",
    "camera_angular_deg_per_frame",
    "new_surface_ratio_voxel_5cm",
    "depth_valid_ratio_sampled",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(values):
    return statistics.mean(values)


def main() -> None:
    rows = []
    for scene, frame, folder in CELLS:
        study = RESULTS / folder
        analysis_path = study / "analysis.json"
        manifest_path = study / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        analysis = json.loads(analysis_path.read_text())
        if analysis["scene"] != scene or analysis["snapshot_frame"] != frame:
            raise ValueError(f"Study identity mismatch: {folder}")
        if analysis["valid_seed_snapshots"] != 8:
            raise ValueError(f"Incomplete quota gates: {folder}")

        diagnostics = RESULTS / "frame_diagnostics" / scene / "ours"
        input_rows = []
        for path in sorted(diagnostics.glob("seed*.json")):
            payload = json.loads(path.read_text())
            found = [x for x in payload["frames"] if int(x["frame"]) == frame]
            if len(found) != 1:
                raise ValueError(f"Expected one diagnostic row at {scene}/{frame}: {path}")
            input_rows.append(found[0])
        if len(input_rows) != 8:
            raise ValueError(f"Expected 8 trajectory diagnostics for {scene}/{frame}")
        static = {}
        for key in STATIC:
            distinct = {round(float(x[key]), 12) for x in input_rows}
            if len(distinct) != 1:
                raise ValueError(f"Policy-independent input feature differs by seed: {scene}/{frame}/{key}")
            static[key] = float(input_rows[0][key])

        seed_rows = []
        for path in sorted((study / scene).glob(f"*_frame{frame}.json")):
            rec = json.loads(path.read_text())
            if rec["scene"] != scene or rec["snapshot_frame"] != frame:
                raise ValueError(f"Record identity mismatch: {path}")
            if not rec["all_count_and_snapshot_gates_passed"]:
                raise ValueError(f"Invalid replay gate: {path}")
            arms = {
                int(round(100 * a["action"]["mature_fraction"])): a
                for a in rec["arms"] if a["action"]["spawn_count"] == 0
            }
            if set(arms) != {25, 75, 96, 100}:
                raise ValueError(f"Unexpected zero-spawn quota arms: {path}")
            eligible = {int(a["mature_eligible"]) for a in arms.values()}
            if len(eligible) != 1:
                raise ValueError(f"Mature-eligible count changes by quota arm: {path}")
            base = arms[25]["final_map_eval_psnr"]
            seed_rows.append({
                "seed": rec["seed"],
                "pre_psnr_db": float(rec["pre_action_snapshot_eval"]["psnr"]),
                "pre_depth_l1": float(rec["pre_action_snapshot_eval"]["depth_l1"]),
                "mature_eligible": eligible.pop(),
                "contrasts_db_vs_25": {
                    str(q): float(arms[q]["final_map_eval_psnr"] - base)
                    for q in (75, 96, 100)
                },
                "realized_mature_update_coverage": {
                    str(q): float(arms[q]["mature_update_coverage"])
                    for q in (25, 75, 96, 100)
                },
                "compute_return_db_per_opt_ms": {
                    str(q): float(arms[q]["compute_return_db_per_opt_ms"])
                    for q in (25, 75, 96, 100)
                },
            })
        if len(seed_rows) != 8:
            raise ValueError(f"Expected 8 quota replay seeds for {scene}/{frame}")
        rows.append({
            "scene": scene,
            "snapshot_frame": frame,
            "study": folder,
            "status": manifest.get("status"),
            "protocol_version": manifest.get("protocol_version"),
            "git_sha": manifest.get("git_sha"),
            "diagnostic_policy": "input-only motion/depth/novelty; identical across 8 seeds",
            "static_input_features": static,
            "pre_action_psnr_db_mean": mean([x["pre_psnr_db"] for x in seed_rows]),
            "pre_action_depth_l1_mean": mean([x["pre_depth_l1"] for x in seed_rows]),
            "mature_eligible_range": [min(x["mature_eligible"] for x in seed_rows),
                                       max(x["mature_eligible"] for x in seed_rows)],
            "mature_eligible_mean": mean([x["mature_eligible"] for x in seed_rows]),
            "mean_psnr_contrast_db_vs_25": {
                q: mean([x["contrasts_db_vs_25"][q] for x in seed_rows])
                for q in ("75", "96", "100")
            },
            "mean_realized_mature_update_coverage": {
                q: mean([x["realized_mature_update_coverage"][q] for x in seed_rows])
                for q in ("25", "75", "96", "100")
            },
            "seed_rows": seed_rows,
            "analysis_sha256": sha256(analysis_path),
            "manifest_sha256": sha256(manifest_path),
        })

    result = {
        "status": "descriptive scene-state linkage; no pooled model or router claim",
        "n_study_cells": len(rows),
        "n_seed_replays_per_cell": 8,
        "static_features_are_policy_independent": True,
        "map_dependent_features_from_independent_trajectory_reruns_joined": False,
        "cells": rows,
        "interpretation_limits": [
            "Only two scenes and two snapshots per scene were measured; static feature values identify four cells, not 32 independent scene states.",
            "Mature-update coverage is a realized treatment outcome, not a pre-action router input.",
            "Coverage and residual-by-map diagnostics were not saved in these replay records and were not joined from separate map reruns.",
            "The results do not establish which feature explains quota response or support a held-out policy selector.",
        ],
    }
    out_json = RESULTS / "quota_response_scene_state_diagnostics.json"
    out_md = RESULTS / "quota_response_scene_state_diagnostics.md"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Scene-state diagnostics for mature-quota response",
        "",
        "This links four same-snapshot dose-response studies to policy-independent pose/depth input features. Eight-seed replay outcomes are summarized per study cell. Separate map-dependent trajectory reruns are not used as substitutes for replay state.",
        "",
        "| Scene | Frame | Translation m/frame | Rotation deg/frame | 5cm new-surface ratio | Valid depth ratio | Pre-action PSNR | Mature-eligible Gaussians | ΔPSNR 75−25 | ΔPSNR 96−25 | ΔPSNR 100−25 |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        f = row["static_input_features"]
        d = row["mean_psnr_contrast_db_vs_25"]
        elo, ehi = row["mature_eligible_range"]
        lines.append(
            f"| `{row['scene']}` | {row['snapshot_frame']} | {f['camera_translation_m_per_frame']:.5f} | "
            f"{f['camera_angular_deg_per_frame']:.3f} | {f['new_surface_ratio_voxel_5cm']:.4f} | "
            f"{f['depth_valid_ratio_sampled']:.3f} | {row['pre_action_psnr_db_mean']:.3f} dB | "
            f"{row['mature_eligible_mean']:.0f} [{elo}–{ehi}] | {d['75']:+.3f} dB | {d['96']:+.3f} dB | {d['100']:+.3f} dB |"
        )
    lines += [
        "",
        "## What the diagnostics show",
        "",
        "The 75%-versus-25% response is negative at Replica office1 frame 70, but positive in the other three cells. The sampled motion/novelty/depth values therefore do not yet explain the response consistently: TUM has much lower translation and lower valid-depth ratios than Replica, yet its 75%-versus-25% response is positive at both snapshots. Within Replica, the response changes sign between frames 70 and 90 as novelty and rotation also change, but there are only two points and several covariates move together.",
        "",
        "Mature-eligible population is quota invariant within each seed snapshot (gate checked), but it varies only modestly by seed within each cell. Realized mature-update coverage rises with the assigned quota, as intended; it is an intervention/compliance measure and cannot be treated as an independent scene-state predictor. Pre-action PSNR also differs substantially across cells, so response magnitude may depend on map maturity, but four cells cannot separate that from scene and stage.",
        "",
        "## Decision",
        "",
        "These data do not support a router or a causal explanation based on motion, novelty, depth validity, or mature-eligible count. They do show that a single quota is unreliable across the measured conditions. The replay runner now has opt-in same-frame summaries for view coverage plus visible counts and color/depth residual means by Gaussian age, using the shared pre-spawn attribution pass. Historical records summarized here predate that instrumentation; a new replay is required to populate it. Keep any selector as a hypothesis until it beats a constant-quota baseline on held-out scenes under matched compute.",
        "",
        "Limitations: four scene-stage cells, one-frame interventions, exploratory cross-cell comparison, no pooled inference. Each cell records hashes of its analysis and manifest in the JSON file.",
        "",
    ]
    out_md.write_text("\n".join(lines))
    print(out_md)


if __name__ == "__main__":
    main()
