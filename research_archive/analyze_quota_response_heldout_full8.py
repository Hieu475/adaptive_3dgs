#!/usr/bin/env python3
"""Analyze preregistered held-out mature-quota response and diagnostics."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/quota_response_heldout_full8"
SCENES = ("replica_room1", "tum_fr2_xyz")
FRAMES = (70, 90)
SEEDS = tuple(range(42, 50))
QUOTAS = (25, 75, 96, 100)
PRIMARY = [
    ("75_minus_25", 75, 25),
    ("100_minus_96", 100, 96),
]
CORE_DIAGNOSTIC_KEYS = (
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


def sign_test_p(xs: list[float]) -> float:
    values = [x for x in xs if x != 0]
    if not values:
        return 1.0
    n = len(values)
    tail = min(sum(x > 0 for x in values), sum(x < 0 for x in values))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(tail + 1)) / (2 ** n))


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


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
    prereg_path = BASE / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    if prereg["status"] != "preregistered before execution":
        raise ValueError("Preregistration is not marked as pre-run")
    manifest_path = BASE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["git_sha"] != "d1ad5c0fff1bd0e72c137a60aa6610ffb7b257b9":
        raise ValueError(f"Unexpected frozen source revision: {manifest['git_sha']}")

    cells = []
    test_rows = []
    cell_seed_data = {}
    for scene in SCENES:
        for frame in FRAMES:
            records = []
            for seed in SEEDS:
                path = BASE / scene / f"seed{seed}_frame{frame}.json"
                record = json.loads(path.read_text())
                if (record["scene"], record["seed"], record["snapshot_frame"]) != (scene, seed, frame):
                    raise ValueError(f"Identity mismatch: {path}")
                if not record["all_count_and_snapshot_gates_passed"]:
                    raise ValueError(f"Snapshot gate failed: {path}")
                if len(record["arms"]) != 8 or any(not arm["valid"] for arm in record["arms"]):
                    raise ValueError(f"Expected eight valid action arms: {path}")
                arms = {a["action"]["name"]: a for a in record["arms"]}
                for q in QUOTAS:
                    for spawn in (0, 128):
                        arm = arms[f"spawn{spawn}_mature{q}"]
                        want_mature = round(5000 * q / 100)
                        if (arm["n_micro_updates"], arm["realized_mature_micro_updates"],
                            arm["realized_young_micro_updates"], arm["n_spawned"]) != (
                                5000, want_mature, 5000 - want_mature, spawn):
                            raise ValueError(f"Action quota mismatch: {path}")
                if {a["snapshot_hash"] for a in record["arms"]} != {record["base_snapshot_hash"]}:
                    raise ValueError(f"Arms do not share the base map: {path}")
                diagnostics = [a.get("age_stratified_diagnostics") for a in record["arms"]]
                if any(d is None for d in diagnostics):
                    raise ValueError(f"Missing diagnostics: {path}")
                core = {json.dumps({k: d[k] for k in CORE_DIAGNOSTIC_KEYS}, sort_keys=True)
                        for d in diagnostics}
                if len(core) != 1:
                    raise ValueError(f"Pre-spawn diagnostics vary by arm: {path}")
                record["_arms"] = arms
                record["_diagnostics"] = diagnostics[0]
                record["_path"] = path
                records.append(record)

            effects = {name: [] for name, _, _ in PRIMARY}
            gains = {q: [] for q in QUOTAS}
            per_seed = []
            feature_values = {
                "pre_action_psnr_db": [],
                "view_coverage_valid_depth": [],
                "young_visible_fraction": [],
                "mature_visible_fraction": [],
                "mature_minus_young_depth_residual": [],
                "mature_minus_young_color_residual": [],
                "mature_eligible_count": [],
            }
            diag_rows = []
            for rec in records:
                arms = rec["_arms"]
                pre = float(rec["pre_action_snapshot_eval"]["psnr"])
                arm_gains = {}
                for q in QUOTAS:
                    arm = arms[f"spawn0_mature{q}"]
                    gain = float(arm["final_map_eval_psnr"] - pre)
                    gains[q].append(gain)
                    arm_gains[str(q)] = gain
                contrasts = {}
                for name, hi, lo in PRIMARY:
                    effect = float(arms[f"spawn0_mature{hi}"]["final_map_eval_psnr"] -
                                   arms[f"spawn0_mature{lo}"]["final_map_eval_psnr"])
                    effects[name].append(effect)
                    contrasts[name] = effect
                d = rec["_diagnostics"]
                y, m = d["young_lt_10_frames"], d["mature_ge_10_frames"]
                feature_values["pre_action_psnr_db"].append(pre)
                feature_values["view_coverage_valid_depth"].append(float(d["view_coverage_valid_depth"]))
                feature_values["young_visible_fraction"].append(float(y["visible_fraction_attributed"]))
                feature_values["mature_visible_fraction"].append(float(m["visible_fraction_attributed"]))
                feature_values["mature_minus_young_depth_residual"].append(float(m["visible_depth_error_mean"] - y["visible_depth_error_mean"]))
                feature_values["mature_minus_young_color_residual"].append(float(m["visible_color_error_mean"] - y["visible_color_error_mean"]))
                feature_values["mature_eligible_count"].append(float(arms["spawn0_mature25"]["mature_eligible"]))
                diag_rows.append(d)
                per_seed.append({"seed": rec["seed"], "pre_action_psnr_db": pre,
                                 "snapshot_hash": rec["base_snapshot_hash"],
                                 "quota_gain_db": arm_gains, "primary_effect_db": contrasts,
                                 "scene_state_diagnostics": d})

            for name, _, _ in PRIMARY:
                p = sign_test_p(effects[name])
                test_rows.append({"scene": scene, "frame": frame, "contrast": name,
                                  "p_raw": p, "effects": effects[name]})

            rho = {
                name: {feature: spearman(vals, effects[name])
                       for feature, vals in feature_values.items()}
                for name, _, _ in PRIMARY
            }
            diagnostics_mean = {}
            for path_keys, name in [
                (("view_coverage_valid_depth",), "view_coverage"),
                (("young_lt_10_frames", "visible_fraction_attributed"), "young_visible_fraction"),
                (("mature_ge_10_frames", "visible_fraction_attributed"), "mature_visible_fraction"),
                (("young_lt_10_frames", "visible_color_error_mean"), "young_color_residual"),
                (("mature_ge_10_frames", "visible_color_error_mean"), "mature_color_residual"),
                (("young_lt_10_frames", "visible_depth_error_mean"), "young_depth_residual"),
                (("mature_ge_10_frames", "visible_depth_error_mean"), "mature_depth_residual"),
                (("mature_ge_10_frames", "visible_attributed"), "mature_visible_count"),
            ]:
                vv = []
                for d in diag_rows:
                    v = d
                    for key in path_keys:
                        v = v[key]
                    vv.append(float(v))
                diagnostics_mean[name] = sum(vv) / len(vv)
            quota_stats = {
                str(q): {"mean_psnr_gain_db": sum(gains[q]) / len(gains[q]),
                         "seed_bootstrap_95_ci_db": bootstrap_ci(gains[q], 97000 + frame + q)}
                for q in QUOTAS
            }
            cell = {"scene": scene, "snapshot_frame": frame, "n_valid_seeds": len(records),
                    "quota_curve_zero_spawn": quota_stats,
                    "primary_contrasts": {
                        name: {"mean_effect_db": sum(effects[name]) / len(effects[name]),
                               "seed_bootstrap_95_ci_db": bootstrap_ci(effects[name], 97100 + frame + i),
                               "positive_seeds": sum(v > 0 for v in effects[name]),
                               "n_seeds": len(effects[name]),
                               "per_seed_effect_db": {str(SEEDS[j]): effects[name][j] for j in range(8)}}
                        for i, (name, _, _) in enumerate(PRIMARY)
                    },
                    "mean_scene_state_diagnostics": diagnostics_mean,
                    "exploratory_seed_level_spearman_rho": rho,
                    "per_seed": per_seed,
                    "record_sha256": {str(rec["seed"]): sha256(rec["_path"]) for rec in records}}
            cells.append(cell)
            cell_seed_data[(scene, frame)] = {"features": feature_values, "effects": effects}

    adjusted = holm([row["p_raw"] for row in test_rows])
    for row, p_adj in zip(test_rows, adjusted):
        row["holm_adjusted_p"] = p_adj
    test_lookup = {(x["scene"], x["frame"], x["contrast"]): x for x in test_rows}
    for cell in cells:
        for name, _, _ in PRIMARY:
            cell["primary_contrasts"][name]["exact_two_sided_sign_test_raw_p"] = test_lookup[(cell["scene"], cell["snapshot_frame"], name)]["p_raw"]
            cell["primary_contrasts"][name]["holm_adjusted_p_across_8_tests"] = test_lookup[(cell["scene"], cell["snapshot_frame"], name)]["holm_adjusted_p"]

    output = {
        "status": "all preregistered held-out snapshot and action gates passed",
        "source_commit": manifest["git_sha"],
        "manifest_sha256": sha256(manifest_path),
        "preregistration_sha256": sha256(prereg_path),
        "runner_sha256": sha256(ROOT / "experiments/run_joint_spawn_refine_replay.py"),
        "pipeline_sha256": sha256(ROOT / "research/pipeline.py"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "confirmatory_test_family": test_rows,
        "cells": cells,
        "interpretation_limits": [
            "The four held-out scene-frame cells are reported separately; no pooled quota effect is estimated.",
            "This confirms one-frame local quota response only; it does not establish continuous policy or long-horizon behavior.",
            "Seed-level feature correlations are exploratory at n=8 per cell and were not used to select a rule.",
            "Age-stratified residual and visibility summaries are measured on the same pre-spawn render and are associations, not causal mediation evidence.",
            "The recorded frame_time_ms includes diagnostic aggregation and is not an end-to-end latency result.",
        ],
    }
    out_json = BASE / "analysis.json"
    out_md = BASE / "report.md"
    out_json.write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# Held-out mature-quota replay with age diagnostics",
        "",
        f"Preregistered protocol `{prereg['protocol_version']}`; source commit `{manifest['git_sha']}`. All 32 seed snapshots and 256 action arms passed their within-snapshot validity gates. Primary inference uses eight contrasts with Holm correction across the full family.",
        "",
        "| Scene | Frame | 75−25 PSNR (95% CI) | Seeds + | Holm p | 100−96 PSNR (95% CI) | Seeds + | Holm p | Mature / young visible fraction | Mature / young depth residual |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for cell in cells:
        a, b = cell["primary_contrasts"]["75_minus_25"], cell["primary_contrasts"]["100_minus_96"]
        aci, bci = a["seed_bootstrap_95_ci_db"], b["seed_bootstrap_95_ci_db"]
        d = cell["mean_scene_state_diagnostics"]
        lines.append(
            f"| `{cell['scene']}` | {cell['snapshot_frame']} | {a['mean_effect_db']:+.3f} [{aci[0]:+.3f}, {aci[1]:+.3f}] dB | "
            f"{a['positive_seeds']}/8 | {a['holm_adjusted_p_across_8_tests']:.4f} | "
            f"{b['mean_effect_db']:+.3f} [{bci[0]:+.3f}, {bci[1]:+.3f}] dB | {b['positive_seeds']}/8 | "
            f"{b['holm_adjusted_p_across_8_tests']:.4f} | {d['mature_visible_fraction']:.3f} / {d['young_visible_fraction']:.3f} | "
            f"{d['mature_depth_residual']:.4f} / {d['young_depth_residual']:.4f} |"
        )
    lines += [
        "",
        "## Exploratory feature associations",
        "",
        "Spearman rho with the preregistered 75%-versus-25% contrast (8 seeds within each cell):",
        "",
        "| Scene | Frame | Pre PSNR | Coverage | Mature visibility | Young visibility | Mature−young depth residual | Mature−young color residual | Mature eligible count |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    def fmt(v):
        return "NA" if v is None else f"{v:+.2f}"
    for cell in cells:
        r = cell["exploratory_seed_level_spearman_rho"]["75_minus_25"]
        lines.append(
            f"| `{cell['scene']}` | {cell['snapshot_frame']} | {fmt(r['pre_action_psnr_db'])} | {fmt(r['view_coverage_valid_depth'])} | "
            f"{fmt(r['mature_visible_fraction'])} | {fmt(r['young_visible_fraction'])} | "
            f"{fmt(r['mature_minus_young_depth_residual'])} | {fmt(r['mature_minus_young_color_residual'])} | "
            f"{fmt(r['mature_eligible_count'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The 75%-versus-25% effect is positive in three of four held-out cells; it is near zero and split 4/8 in Replica room1 at frame 70. The 100%-versus-96% effect is negative in Replica room1 frame 70, near zero at Replica frame 90, and positive in both TUM fr2_xyz cells. After the preregistered Holm correction across eight contrasts, none is below 0.05 (the smallest adjusted p is 0.0625), so treat the directions and magnitudes as evidence for follow-up rather than a final confirmatory claim.",
        "",
        "The diagnostics do not yield a single obvious quota rule. Replica room1 frame 90 has very low mature visibility (0.020) but a positive 75%-versus-25% effect; TUM fr2_xyz has high mature visibility (0.89–0.92) and also a positive effect. The seed-level feature correlations change direction across cells and remain exploratory.",
        "",
        "Feature correlations are descriptive and must not be used to fit a router from this sample.",
        "",
        "All action comparisons are paired within one snapshot per seed. CUDA runs may differ across independent replays because gsplat backward uses nondeterministic floating-point atomics; this does not break the within-record pairing. The synchronized optimization-stage timer is distinct from frame wall time, which includes added diagnostic aggregation.",
        "",
        "Per-seed outcomes, diagnostics, adjusted p-values, and source hashes are in `analysis.json`; the preregistration is in `preregistration.json` and `.md`.",
        "",
    ]
    out_md.write_text("\n".join(lines))
    print(out_md)


if __name__ == "__main__":
    main()
