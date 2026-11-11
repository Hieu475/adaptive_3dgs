#!/usr/bin/env python3
"""Validate and summarize the instrumented one-seed quota replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/legacy/phase14_generalization_postfix"
RUN = RESULTS / "quota_age_diagnostic_seed42"
PREVIOUS = {
    ("replica_office1", 70): "quota_response_frame70",
    ("replica_office1", 90): "quota_response_frame90",
    ("tum_fr1_rpy", 70): "quota_response_tum_fr1_rpy_frame70",
    ("tum_fr1_rpy", 90): "quota_response_tum_fr1_rpy_frame90",
}
CORE_DIAGNOSTIC_KEYS = (
    "stage", "mature_age_frames", "n_gaussians_attributed",
    "view_coverage_valid_depth", "young_lt_10_frames", "mature_ge_10_frames",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cells = []
    for (scene, frame), old_folder in PREVIOUS.items():
        path = RUN / scene / f"seed42_frame{frame}.json"
        record = json.loads(path.read_text())
        if record["scene"] != scene or record["snapshot_frame"] != frame:
            raise ValueError(f"Wrong scene/frame in {path}")
        if not record["all_count_and_snapshot_gates_passed"]:
            raise ValueError(f"Replay gates failed in {path}")
        arms = record["arms"]
        if len(arms) != 8 or any(not arm["valid"] for arm in arms):
            raise ValueError(f"Expected 8 valid arms in {path}")
        hashes = {arm["snapshot_hash"] for arm in arms}
        if hashes != {record["base_snapshot_hash"]}:
            raise ValueError(f"Action arms do not share the recorded snapshot: {path}")
        diagnostics = [arm.get("age_stratified_diagnostics") for arm in arms]
        if any(x is None for x in diagnostics):
            raise ValueError(f"Missing age diagnostics in {path}")
        canonical_core = {
            json.dumps({k: d[k] for k in CORE_DIAGNOSTIC_KEYS}, sort_keys=True)
            for d in diagnostics
        }
        if len(canonical_core) != 1:
            raise ValueError(f"Pre-spawn diagnostics differ across arms: {path}")
        if any(d["n_gaussians_attributed"] != diagnostics[0]["n_gaussians_attributed"]
               for d in diagnostics):
            raise ValueError(f"Attribution population differs by arm: {path}")

        zero_spawn = {
            int(round(100 * arm["action"]["mature_fraction"])): arm
            for arm in arms if int(arm["action"]["spawn_count"]) == 0
        }
        if set(zero_spawn) != {25, 75, 96, 100}:
            raise ValueError(f"Unexpected quota arms: {path}")
        base_psnr = zero_spawn[25]["final_map_eval_psnr"]
        contrasts = {
            str(q): float(zero_spawn[q]["final_map_eval_psnr"] - base_psnr)
            for q in (75, 96, 100)
        }

        old_path = RESULTS / old_folder / scene / f"seed42_frame{frame}.json"
        old = json.loads(old_path.read_text())
        same_snapshot_as_old_study = old["base_snapshot_hash"] == record["base_snapshot_hash"]
        d = diagnostics[0]
        cells.append({
            "scene": scene,
            "snapshot_frame": frame,
            "seed": 42,
            "replay_valid": True,
            "all_arms_share_snapshot": True,
            "pre_spawn_diagnostics_identical_across_8_arms": True,
            "same_snapshot_as_archived_dose_study": same_snapshot_as_old_study,
            "current_snapshot_hash": record["base_snapshot_hash"],
            "archived_snapshot_hash": old["base_snapshot_hash"],
            "arm_diagnostics": d,
            "zero_spawn_psnr_contrasts_vs_25_db": contrasts,
            "current_record_sha256": sha256(path),
            "archived_record_sha256": sha256(old_path),
        })

    output = {
        "status": "one-seed instrumentation validation; descriptive only",
        "run_path": str(RUN.relative_to(ROOT)),
        "seed": 42,
        "n_cells": len(cells),
        "n_arms_per_cell": 8,
        "cells": cells,
        "limitations": [
            "One seed validates logging and cannot update the eight-seed dose-response estimates.",
            "Each newly generated snapshot hash is compared with the archived study; mismatches mean outcomes must not be pooled or treated as exact replays.",
            "Instrumentation runs include diagnostic aggregation overhead in frame wall time; frame_time_ms is not a production latency measurement here.",
        ],
    }
    json_path = RESULTS / "quota_age_diagnostic_seed42_summary.json"
    md_path = RESULTS / "quota_age_diagnostic_seed42_summary.md"
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# One-seed validation of age-stratified replay diagnostics",
        "",
        "All four snapshot replays passed the existing count/snapshot gates. The pre-spawn diagnostic fields were identical across eight action arms per snapshot. This validates within-run logging only; the one seed is not a new dose-response estimate.",
        "",
        "| Scene | Frame | Coverage | Young visible / population | Young color / depth error | Mature visible / population | Mature color / depth error | ΔPSNR 75−25 | ΔPSNR 96−25 | ΔPSNR 100−25 | Archived snapshot hash match? |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|",
    ]
    for cell in cells:
        d = cell["arm_diagnostics"]
        y, m = d["young_lt_10_frames"], d["mature_ge_10_frames"]
        c = cell["zero_spawn_psnr_contrasts_vs_25_db"]
        lines.append(
            f"| `{cell['scene']}` | {cell['snapshot_frame']} | {d['view_coverage_valid_depth']:.4f} | "
            f"{y['visible_attributed']} / {y['population_attributed']} ({y['visible_fraction_attributed']:.3f}) | "
            f"{y['visible_color_error_mean']:.4f} / {y['visible_depth_error_mean']:.4f} | "
            f"{m['visible_attributed']} / {m['population_attributed']} ({m['visible_fraction_attributed']:.3f}) | "
            f"{m['visible_color_error_mean']:.4f} / {m['visible_depth_error_mean']:.4f} | "
            f"{c['75']:+.3f} dB | {c['96']:+.3f} dB | {c['100']:+.3f} dB | "
            f"{'yes' if cell['same_snapshot_as_archived_dose_study'] else '**no**'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "In these seed-42 snapshots, mature visible fractions are lower than young fractions at Replica office1 frame 90 and TUM fr1_rpy frame 90; at frame 70 the gap is smaller or reversed depending on scene. Residual levels also vary by scene and age. These are contemporaneous attribution summaries and do not establish that residual or visibility imbalance causes a quota to win.",
        "",
        "The newly generated map snapshot hashes do not match the archived seed-42 dose-response records. This run is therefore a logging validation on the current workspace state, not an exact replay of the earlier eight-seed experiments. Do not combine its PSNR contrasts with those studies.",
        "",
        "Instrumentation adds aggregation work before optimization, so `frame_time_ms` from this run is not suitable as a production latency result. The synchronized optimization-stage timer remains a separate quantity.",
        "",
        "The next useful experiment is to rerun all eight seeds with this instrumentation from one frozen checkout, at a prespecified set of held-out snapshots, then test whether the measured pre-action features predict quota response under matched compute.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    print(md_path)


if __name__ == "__main__":
    main()
