#!/usr/bin/env python3
"""Report valid dose-response cells and quota-capacity failures in cohort 2."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/quota_response_heldout_cohort2_full8"
SCENES = ("replica_office0", "tum_fr1_xyz")
FRAMES = (70, 90)
SEEDS = tuple(range(42, 50))
QUOTAS = (25, 75, 96, 100)
CORE_KEYS = (
    "stage", "mature_age_frames", "n_gaussians_attributed",
    "view_coverage_valid_depth", "young_lt_10_frames", "mature_ge_10_frames",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quantile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    p = (len(ys) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return ys[lo] + (ys[hi] - ys[lo]) * (p - lo)


def bootstrap_ci(xs: list[float], seed: int, n: int = 50000) -> list[float]:
    rng = random.Random(seed)
    draws = [sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n)]
    return [quantile(draws, 0.025), quantile(draws, 0.975)]


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    dx, dy = [x - mx for x in rx], [y - my for y in ry]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denom if denom else None


def main() -> None:
    manifest_path = BASE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prereg_path = BASE / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    cells = []
    for scene_index, scene in enumerate(SCENES):
        for frame in FRAMES:
            records = []
            invalid_details = []
            for seed in SEEDS:
                path = BASE / scene / f"seed{seed}_frame{frame}.json"
                rec = json.loads(path.read_text())
                if (rec["scene"], rec["seed"], rec["snapshot_frame"]) != (scene, seed, frame):
                    raise ValueError(f"Identity mismatch: {path}")
                arms = {a["action"]["name"]: a for a in rec["arms"]}
                if len(arms) != 8:
                    raise ValueError(f"Expected 8 action arms: {path}")
                diagnostics = [a.get("age_stratified_diagnostics") for a in rec["arms"]]
                if any(d is None for d in diagnostics):
                    raise ValueError(f"Missing diagnostics: {path}")
                diag_core = {json.dumps({k: d[k] for k in CORE_KEYS}, sort_keys=True)
                             for d in diagnostics}
                if len(diag_core) != 1:
                    raise ValueError(f"Pre-spawn diagnostics differ across arms: {path}")
                for q in QUOTAS:
                    for spawn in (0, 128):
                        arm = arms[f"spawn{spawn}_mature{q}"]
                        expected_mature = round(5000 * q / 100)
                        expected_young = 5000 - expected_mature
                        if arm["n_micro_updates"] != 5000 or arm["n_spawned"] != spawn:
                            raise ValueError(f"Wrong total updates/spawn dose: {path}")
                        if arm["valid"] and (
                            arm["realized_mature_micro_updates"] != expected_mature or
                            arm["realized_young_micro_updates"] != expected_young
                        ):
                            raise ValueError(f"Runner marks wrong quota valid: {path}")
                failed = [
                    {"action": a["action"]["name"],
                     "requested_mature_updates": round(5000 * a["action"]["mature_fraction"]),
                     "requested_young_updates": 5000 - round(5000 * a["action"]["mature_fraction"]),
                     "realized_mature_updates": a["realized_mature_micro_updates"],
                     "realized_young_updates": a["realized_young_micro_updates"],
                     "mature_eligible": a["mature_eligible"],
                     "young_visible_attributed": a["age_stratified_diagnostics"]["young_lt_10_frames"]["visible_attributed"],
                     "young_quota_shortfall": max(0, (5000 - round(5000 * a["action"]["mature_fraction"])) - a["realized_young_micro_updates"])}
                    for a in rec["arms"] if not a["valid"]
                ]
                if failed:
                    invalid_details.append({"seed": seed, "failed_arms": failed})
                rec["_arms"] = arms
                rec["_diagnostics"] = diagnostics[0]
                rec["_path"] = path
                records.append(rec)

            valid = [r for r in records if r["all_count_and_snapshot_gates_passed"]]
            cell = {"scene": scene, "snapshot_frame": frame,
                    "valid_seed_snapshots": len(valid), "total_seed_snapshots": len(records),
                    "pre_spawn_diagnostics_identical_across_arms": True,
                    "invalid_snapshot_details": invalid_details,
                    "record_sha256": {str(r["seed"]): sha256(r["_path"]) for r in records}}
            if len(valid) == 8:
                contrasts = {"75_minus_25": [], "100_minus_96": []}
                gains = {q: [] for q in QUOTAS}
                pre_psnr, feat = [], {
                    "pre_action_psnr": [], "coverage": [], "mature_visible_fraction": [],
                    "young_visible_fraction": [], "mature_minus_young_depth_residual": [],
                    "mature_minus_young_color_residual": [], "mature_eligible": [],
                }
                diags = []
                for r in valid:
                    arms = r["_arms"]
                    p0 = float(r["pre_action_snapshot_eval"]["psnr"])
                    pre_psnr.append(p0)
                    for q in QUOTAS:
                        gains[q].append(float(arms[f"spawn0_mature{q}"]["final_map_eval_psnr"] - p0))
                    contrasts["75_minus_25"].append(float(arms["spawn0_mature75"]["final_map_eval_psnr"] - arms["spawn0_mature25"]["final_map_eval_psnr"]))
                    contrasts["100_minus_96"].append(float(arms["spawn0_mature100"]["final_map_eval_psnr"] - arms["spawn0_mature96"]["final_map_eval_psnr"]))
                    d = r["_diagnostics"]
                    y, m = d["young_lt_10_frames"], d["mature_ge_10_frames"]
                    feat["pre_action_psnr"].append(p0)
                    feat["coverage"].append(float(d["view_coverage_valid_depth"]))
                    feat["mature_visible_fraction"].append(float(m["visible_fraction_attributed"]))
                    feat["young_visible_fraction"].append(float(y["visible_fraction_attributed"]))
                    feat["mature_minus_young_depth_residual"].append(float(m["visible_depth_error_mean"] - y["visible_depth_error_mean"]))
                    feat["mature_minus_young_color_residual"].append(float(m["visible_color_error_mean"] - y["visible_color_error_mean"]))
                    feat["mature_eligible"].append(float(arms["spawn0_mature25"]["mature_eligible"]))
                    diags.append(d)
                cell["quota_curve_zero_spawn"] = {
                    str(q): {"mean_psnr_gain_db": sum(gains[q]) / 8,
                             "seed_bootstrap_95_ci_db": bootstrap_ci(gains[q], 98200 + scene_index * 100 + frame + q)}
                    for q in QUOTAS
                }
                cell["primary_contrasts"] = {
                    key: {"mean_effect_db": sum(xs) / 8,
                          "seed_bootstrap_95_ci_db": bootstrap_ci(xs, 98300 + scene_index * 100 + frame + i),
                          "positive_seeds": sum(x > 0 for x in xs), "n_seeds": 8,
                          "per_seed_effect_db": {str(SEEDS[j]): xs[j] for j in range(8)}}
                    for i, (key, xs) in enumerate(contrasts.items())
                }
                cell["mean_scene_state_diagnostics"] = {
                    "view_coverage": sum(float(d["view_coverage_valid_depth"]) for d in diags) / 8,
                    "young_visible_fraction": sum(float(d["young_lt_10_frames"]["visible_fraction_attributed"]) for d in diags) / 8,
                    "mature_visible_fraction": sum(float(d["mature_ge_10_frames"]["visible_fraction_attributed"]) for d in diags) / 8,
                    "young_depth_residual": sum(float(d["young_lt_10_frames"]["visible_depth_error_mean"]) for d in diags) / 8,
                    "mature_depth_residual": sum(float(d["mature_ge_10_frames"]["visible_depth_error_mean"]) for d in diags) / 8,
                    "young_color_residual": sum(float(d["young_lt_10_frames"]["visible_color_error_mean"]) for d in diags) / 8,
                    "mature_color_residual": sum(float(d["mature_ge_10_frames"]["visible_color_error_mean"]) for d in diags) / 8,
                }
                cell["exploratory_spearman_with_75_minus_25"] = {
                    key: spearman(vals, contrasts["75_minus_25"]) for key, vals in feat.items()
                }
                cell["inference_status"] = "full 8/8 cell; descriptive paired estimates available"
            else:
                # Do not estimate quota effects when any seed snapshot in the cell
                # fails the preregistered treatment-capacity gate.
                bad = [a for r in records for a in r["_arms"].values() if not a["valid"]]
                for arm in bad:
                    arm["_failure_young_shortfall"] = max(
                        0, (5000 - round(5000 * arm["action"]["mature_fraction"])) -
                        arm["realized_young_micro_updates"])
                shortfalls = [a["_failure_young_shortfall"] for a in bad]
                cell["inference_status"] = "not estimable under preregistered exact-quota gate"
                cell["invalid_arm_count"] = len(bad)
                cell["failed_arm_names"] = sorted({a["action"]["name"] for a in bad})
                cell["mean_failed_arm_young_quota_shortfall"] = sum(shortfalls) / max(len(shortfalls), 1)
                zero25 = [r["_arms"]["spawn0_mature25"] for r in records]
                spawn25 = [r["_arms"]["spawn128_mature25"] for r in records]
                cell["zero_spawn_25_mature_realized_updates"] = {
                    "mean_mature": sum(a["realized_mature_micro_updates"] for a in zero25) / 8,
                    "mean_young": sum(a["realized_young_micro_updates"] for a in zero25) / 8,
                    "mean_young_shortfall": sum(max(0, 3750 - a["realized_young_micro_updates"]) for a in zero25) / 8,
                }
                cell["spawn128_25_mature_realized_updates"] = {
                    "mean_mature": sum(a["realized_mature_micro_updates"] for a in spawn25) / 8,
                    "mean_young": sum(a["realized_young_micro_updates"] for a in spawn25) / 8,
                    "mean_young_shortfall": sum(max(0, 3750 - a["realized_young_micro_updates"]) for a in spawn25) / 8,
                }
            cells.append(cell)

    output = {
        "status": "cohort completed; two of four scene-frame cells failed exact-quota feasibility",
        "source_commit": manifest["git_sha"],
        "manifest_sha256": sha256(manifest_path),
        "preregistration_sha256": sha256(prereg_path),
        "runner_sha256": sha256(ROOT / "experiments/run_joint_spawn_refine_replay.py"),
        "pipeline_sha256": sha256(ROOT / "research/pipeline.py"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "cells": cells,
        "inference_policy": "No confirmatory p-values are reported because half the preregistered scene-frame cells were not estimable; the eight-test family is incomplete.",
        "limitations": [
            "No effects are computed for cells that failed the exact mature/young split gate.",
            "In the invalid cells, the 25% mature arm could not allocate its full 3750 young updates from visible young candidates; excess updates transferred to the mature group.",
            "The analysis shows quota feasibility, not that spawning is beneficial; spawn-dose effect comparisons require a separately specified estimand.",
            "Feature correlations are exploratory and n=8 in valid cells; they do not justify a router.",
            "This cohort is kept separate from prior held-out quota experiments.",
        ],
    }
    json_path = BASE / "analysis.json"
    md_path = BASE / "report.md"
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# Second held-out quota cohort: treatment feasibility and valid effects",
        "",
        f"Source commit `{manifest['git_sha']}`. This cohort has 32 seed snapshots. All action arms completed, but only two of four scene-frame cells met the preregistered exact-quota gate for all eight seeds. No confirmatory p-values are reported because the preregistered eight-test family is incomplete.",
        "",
        "| Scene | Frame | Valid seed snapshots | Mean 25% arm young-update shortfall (spawn 0 / 128) | 75−25 effect (95% seed CI) | Seeds + | 100−96 effect (95% seed CI) | Seeds + | Mature / young visibility | Mature / young depth residual |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for cell in cells:
        if cell["inference_status"].startswith("full"):
            c, dcontrast = cell["primary_contrasts"]["75_minus_25"], cell["primary_contrasts"]["100_minus_96"]
            ci, dci = c["seed_bootstrap_95_ci_db"], dcontrast["seed_bootstrap_95_ci_db"]
            d = cell["mean_scene_state_diagnostics"]
            sign = f"{c['positive_seeds']}/8"
            dsign = f"{dcontrast['positive_seeds']}/8"
            vis = f"{d['mature_visible_fraction']:.3f} / {d['young_visible_fraction']:.3f}"
            resid = f"{d['mature_depth_residual']:.4f} / {d['young_depth_residual']:.4f}"
            shortfall = "0 / 0"
            effect = f"{c['mean_effect_db']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
            deffect = f"{dcontrast['mean_effect_db']:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]"
        else:
            a, b = cell["zero_spawn_25_mature_realized_updates"], cell["spawn128_25_mature_realized_updates"]
            shortfall = f"{a['mean_young_shortfall']:.1f} / {b['mean_young_shortfall']:.1f}"
            effect, sign, deffect, dsign = "not estimable", "—", "not estimable", "—"
            vis, resid = "see seed diagnostics", "see seed diagnostics"
        lines.append(
            f"| `{cell['scene']}` | {cell['snapshot_frame']} | {cell['valid_seed_snapshots']}/8 | "
            f"{shortfall} | {effect} | {sign} | {deffect} | {dsign} | {vis} | {resid} |"
        )
    lines += [
        "",
        "## Capacity failure diagnosis",
        "",
        "The two invalid cells fail because a 25%-mature arm requests 3,750 young updates, but too few visible young Gaussians are available at five updates each. The scheduler transfers the unused young quota to mature Gaussians, preserving 5,000 total updates while violating the intended age split. At Replica office0 frame 90, all eight zero-spawn 25% arms failed this split; at TUM fr1_xyz frame 70, all eight zero-spawn arms failed and one spawn-128 arm also failed. The preregistered validity rule therefore excludes the whole cell from quota-effect estimation.",
        "",
        "This is evidence that fixed age quotas can be infeasible under a fixed update budget when one age group's visible capacity is too small. The spawn-128 25% arms often restore young-update capacity, but this replay was not preregistered to estimate a spawn-by-quota causal effect, so no such claim is made.",
        "",
        "## Interpretation of the two valid cells",
        "",
        "The two complete cells can be reported with paired effect estimates and confidence intervals, but they do not complete the preregistered eight-test family. Treat them as descriptive follow-up only. The invalid cells should guide a separate capacity-aware protocol, not be repaired by reallocating quotas after observing outcomes.",
        "",
        "Per-seed invalid arms, exact realized splits, diagnostics, record hashes, and source provenance are in `analysis.json`. The preregistration remains unchanged.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    print(md_path)


if __name__ == "__main__":
    main()
