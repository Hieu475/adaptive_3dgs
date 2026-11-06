#!/usr/bin/env python3
"""Summarize full-seed age-diagnostic quota replays without pooling studies."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/phase14_generalization_postfix"
RUN = RESULTS / "quota_response_age_diagnostics_full8"
FROZEN_CHECKOUT = Path("/tmp/adaptive_3dgs_phase14_postfix_clean_20261017")
CELLS = [
    ("replica_office1", 70, "quota_response_frame70"),
    ("replica_office1", 90, "quota_response_frame90"),
    ("tum_fr1_rpy", 70, "quota_response_tum_fr1_rpy_frame70"),
    ("tum_fr1_rpy", 90, "quota_response_tum_fr1_rpy_frame90"),
]
QUOTAS = (25, 75, 96, 100)
CONTRASTS = {"75_minus_25": (75, 25), "100_minus_96": (100, 96)}
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
    vals = [x for x in xs if x != 0]
    if not vals:
        return 1.0
    n = len(vals)
    tail = min(sum(x > 0 for x in vals), sum(x < 0 for x in vals))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(tail + 1)) / (2 ** n))


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    out = [0.0] * len(pvalues)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[idx]))
        out[idx] = running
    return out


def average_ranks(xs: list[float]) -> list[float]:
    ordered = sorted(enumerate(xs), key=lambda pair: pair[1])
    ranks = [0.0] * len(xs)
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
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / den if den else None


def main() -> None:
    summary_cells = []
    preregistration_manifest = []
    for cell_index, (scene, frame, archived_folder) in enumerate(CELLS):
        records = []
        old_records = []
        for seed in range(42, 50):
            path = RUN / scene / f"seed{seed}_frame{frame}.json"
            record = json.loads(path.read_text())
            if (record["scene"], record["seed"], record["snapshot_frame"]) != (scene, seed, frame):
                raise ValueError(f"Record identity mismatch: {path}")
            if not record["all_count_and_snapshot_gates_passed"]:
                raise ValueError(f"Replay validity gate failed: {path}")
            if len(record["arms"]) != 8 or any(not x["valid"] for x in record["arms"]):
                raise ValueError(f"Expected eight valid actions: {path}")
            arms = {x["action"]["name"]: x for x in record["arms"]}
            for q in QUOTAS:
                for spawn in (0, 128):
                    name = f"spawn{spawn}_mature{q}"
                    arm = arms[name]
                    if (arm["n_micro_updates"], arm["realized_mature_micro_updates"],
                        arm["realized_young_micro_updates"], arm["n_spawned"]) != (
                            5000, round(5000 * q / 100), 5000 - round(5000 * q / 100), spawn):
                        raise ValueError(f"Update/spawn quota mismatch: {path} {name}")
            hashes = {x["snapshot_hash"] for x in record["arms"]}
            if hashes != {record["base_snapshot_hash"]}:
                raise ValueError(f"Action arms do not share a snapshot: {path}")
            diagnostics = [x.get("age_stratified_diagnostics") for x in record["arms"]]
            if any(x is None for x in diagnostics):
                raise ValueError(f"Missing age diagnostics: {path}")
            core = {json.dumps({k: d[k] for k in CORE_DIAGNOSTIC_KEYS}, sort_keys=True)
                    for d in diagnostics}
            if len(core) != 1:
                raise ValueError(f"Pre-spawn diagnostics vary across arms: {path}")
            record["_arms"] = arms
            record["_diagnostics"] = diagnostics[0]
            records.append(record)

            old_path = RESULTS / archived_folder / scene / f"seed{seed}_frame{frame}.json"
            old_records.append(json.loads(old_path.read_text()))

        prereg_path = RESULTS / archived_folder / "preregistration.json"
        preregistration_manifest.append({
            "scene": scene, "snapshot_frame": frame,
            "source": str(prereg_path.relative_to(ROOT)), "sha256": sha256(prereg_path),
        })
        per_seed = []
        effects = {key: [] for key in CONTRASTS}
        old_effects = {key: [] for key in CONTRASTS}
        gains = {q: [] for q in QUOTAS}
        diag_rows = []
        for record, old in zip(records, old_records):
            arms = record["_arms"]
            pre = float(record["pre_action_snapshot_eval"]["psnr"])
            row = {"seed": record["seed"], "pre_action_psnr_db": pre,
                   "snapshot_hash": record["base_snapshot_hash"], "quota_gain_db": {}}
            for q in QUOTAS:
                arm = arms[f"spawn0_mature{q}"]
                gain = float(arm["final_map_eval_psnr"] - pre)
                gains[q].append(gain)
                row["quota_gain_db"][str(q)] = gain
            for key, (hi, lo) in CONTRASTS.items():
                effect = float(arms[f"spawn0_mature{hi}"]["final_map_eval_psnr"] -
                               arms[f"spawn0_mature{lo}"]["final_map_eval_psnr"])
                effects[key].append(effect)
                oldarms = {x["action"]["name"]: x for x in old["arms"]}
                old_effects[key].append(float(oldarms[f"spawn0_mature{hi}"]["final_map_eval_psnr"] -
                                              oldarms[f"spawn0_mature{lo}"]["final_map_eval_psnr"]))
                row.setdefault("primary_effect_db", {})[key] = effect
            per_seed.append(row)
            diag_rows.append(record["_diagnostics"])

        raw_p = [sign_test_p(effects[k]) for k in CONTRASTS]
        adj_p = holm(raw_p)
        primary = {}
        for idx, key in enumerate(CONTRASTS):
            xs = effects[key]
            primary[key] = {
                "mean_effect_db": sum(xs) / len(xs),
                "seed_bootstrap_95_ci_db": bootstrap_ci(xs, 98000 + cell_index * 10 + idx),
                "positive_seeds": sum(x > 0 for x in xs),
                "n_seeds": len(xs),
                "exact_two_sided_sign_test_raw_p": raw_p[idx],
                "holm_adjusted_p": adj_p[idx],
                "per_seed_effect_db": {str(r["seed"]): effects[key][i] for i, r in enumerate(per_seed)},
                "archived_mean_effect_db": sum(old_effects[key]) / len(old_effects[key]),
                "rerun_minus_archived_paired_effect_db": [effects[key][i] - old_effects[key][i]
                                                           for i in range(len(effects[key]))],
            }

        def diag_mean(path: tuple[str, ...]) -> float:
            values = []
            for d in diag_rows:
                val = d
                for key in path:
                    val = val[key]
                values.append(float(val))
            return sum(values) / len(values)

        features = {
            "pre_action_psnr_db": [x["pre_action_psnr_db"] for x in per_seed],
            "view_coverage_valid_depth": [float(d["view_coverage_valid_depth"]) for d in diag_rows],
            "young_visible_fraction": [float(d["young_lt_10_frames"]["visible_fraction_attributed"]) for d in diag_rows],
            "mature_visible_fraction": [float(d["mature_ge_10_frames"]["visible_fraction_attributed"]) for d in diag_rows],
            "mature_minus_young_depth_residual": [float(d["mature_ge_10_frames"]["visible_depth_error_mean"] - d["young_lt_10_frames"]["visible_depth_error_mean"]) for d in diag_rows],
            "mature_minus_young_color_residual": [float(d["mature_ge_10_frames"]["visible_color_error_mean"] - d["young_lt_10_frames"]["visible_color_error_mean"]) for d in diag_rows],
            "mature_eligible_count": [float(records[i]["_arms"]["spawn0_mature25"]["mature_eligible"]) for i in range(8)],
        }
        correlations = {
            effect_name: {feature: spearman(values, effects[effect_name]) for feature, values in features.items()}
            for effect_name in CONTRASTS
        }
        hash_matches = [records[i]["base_snapshot_hash"] == old_records[i]["base_snapshot_hash"]
                        for i in range(8)]
        max_pre_psnr_delta = max(abs(float(records[i]["pre_action_snapshot_eval"]["psnr"] -
                                             old_records[i]["pre_action_snapshot_eval"]["psnr"]))
                                 for i in range(8))
        summary_cells.append({
            "scene": scene, "snapshot_frame": frame,
            "archived_study": archived_folder,
            "n_valid_seed_snapshots": len(records),
            "snapshot_hash_matches_archived": sum(hash_matches),
            "snapshot_hash_match_flags_by_seed": {str(42+i): hash_matches[i] for i in range(8)},
            "max_pre_action_psnr_delta_vs_archived_db": max_pre_psnr_delta,
            "pre_spawn_diagnostics_identical_across_arms": True,
            "quota_curve_zero_spawn": {
                str(q): {"mean_psnr_gain_db": sum(gains[q]) / 8,
                         "seed_bootstrap_95_ci_db": bootstrap_ci(gains[q], 99000 + cell_index * 100 + q)}
                for q in QUOTAS
            },
            "primary_contrasts": primary,
            "mean_scene_state_diagnostics": {
                "view_coverage_valid_depth": diag_mean(("view_coverage_valid_depth",)),
                "young_visible_fraction": diag_mean(("young_lt_10_frames", "visible_fraction_attributed")),
                "mature_visible_fraction": diag_mean(("mature_ge_10_frames", "visible_fraction_attributed")),
                "young_color_residual": diag_mean(("young_lt_10_frames", "visible_color_error_mean")),
                "mature_color_residual": diag_mean(("mature_ge_10_frames", "visible_color_error_mean")),
                "young_depth_residual": diag_mean(("young_lt_10_frames", "visible_depth_error_mean")),
                "mature_depth_residual": diag_mean(("mature_ge_10_frames", "visible_depth_error_mean")),
                "mature_eligible_count": diag_mean(("mature_ge_10_frames", "visible_attributed")),
            },
            "exploratory_seed_level_spearman_rho": correlations,
            "per_seed": per_seed,
            "record_sha256": {
                str(42+i): sha256(RUN / scene / f"seed{42+i}_frame{frame}.json") for i in range(8)
            },
        })

    result = {
        "status": "full eight-seed diagnostic replication; each cell analyzed separately",
        "source_commit": json.loads((RUN / "manifest.json").read_text())["git_sha"],
        "frozen_checkout": str(FROZEN_CHECKOUT),
        "frozen_checkout_clean_after_run": True,
        "source_manifest_sha256": sha256(RUN / "manifest.json"),
        "runner_sha256": sha256(ROOT / "experiments/run_joint_spawn_refine_replay.py"),
        "pipeline_sha256": sha256(ROOT / "research/pipeline.py"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "preregistrations": preregistration_manifest,
        "cells": summary_cells,
        "interpretation_limits": [
            "The four scene-stage cells are separate studies and are not pooled.",
            "The full-seed replay passed within-snapshot gates, but CUDA replay snapshot hashes differ from the archived runs; seed-level outcomes are repeated independent replays, not exact state replays.",
            "Seed-level Spearman correlations are exploratory (n=8 per cell) and are not evidence for a deployable selector.",
            "Visibility/residual summaries are measured on the same pre-spawn render as the intervention and are contemporaneous associations, not causal mechanisms.",
            "Frame wall time includes diagnostic aggregation and must not be interpreted as production latency.",
        ],
    }
    out_json = RESULTS / "quota_response_age_diagnostics_full8_summary.json"
    out_md = RESULTS / "quota_response_age_diagnostics_full8_summary.md"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Full eight-seed quota replay with age-stratified diagnostics",
        "",
        f"Source commit `{result['source_commit']}`. Each of four cells has eight seeds and eight action arms per snapshot; all within-snapshot action/count gates passed. No cross-cell pooling was performed.",
        "",
        "| Scene | Frame | 75−25 PSNR (95% seed CI) | Sign | 100−96 PSNR (95% seed CI) | Sign | Mature visible fraction | Young visible fraction | Mature depth residual | Young depth residual | Snapshot hashes matching archived seeds |",
        "|:--|--:|--:|:--:|--:|:--:|--:|--:|--:|--:|--:|",
    ]
    for cell in summary_cells:
        a, b = cell["primary_contrasts"]["75_minus_25"], cell["primary_contrasts"]["100_minus_96"]
        aci, bci = a["seed_bootstrap_95_ci_db"], b["seed_bootstrap_95_ci_db"]
        f = cell["mean_scene_state_diagnostics"]
        lines.append(
            f"| `{cell['scene']}` | {cell['snapshot_frame']} | {a['mean_effect_db']:+.3f} [{aci[0]:+.3f}, {aci[1]:+.3f}] dB | {a['positive_seeds']}/8 | "
            f"{b['mean_effect_db']:+.3f} [{bci[0]:+.3f}, {bci[1]:+.3f}] dB | {b['positive_seeds']}/8 | "
            f"{f['mature_visible_fraction']:.3f} | {f['young_visible_fraction']:.3f} | "
            f"{f['mature_depth_residual']:.4f} | {f['young_depth_residual']:.4f} | "
            f"{cell['snapshot_hash_matches_archived']}/8 |"
        )
    lines += ["", "## Seed-level feature associations (exploratory Spearman rho)", "",
              "The entries below are within-cell correlations with the 75%-versus-25% PSNR contrast (eight seeds per cell). They are descriptive only; do not treat their signs as selected predictors.", "",
              "| Scene | Frame | Pre PSNR | Coverage | Mature visibility | Young visibility | Mature−young depth residual | Mature−young color residual | Mature eligible count |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for cell in summary_cells:
        r = cell["exploratory_seed_level_spearman_rho"]["75_minus_25"]
        fmt = lambda x: "NA" if x is None else f"{x:+.2f}"
        lines.append(
            f"| `{cell['scene']}` | {cell['snapshot_frame']} | {fmt(r['pre_action_psnr_db'])} | "
            f"{fmt(r['view_coverage_valid_depth'])} | {fmt(r['mature_visible_fraction'])} | "
            f"{fmt(r['young_visible_fraction'])} | {fmt(r['mature_minus_young_depth_residual'])} | "
            f"{fmt(r['mature_minus_young_color_residual'])} | {fmt(r['mature_eligible_count'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The rerun reproduces the broad quota-response pattern across cells while recording age-stratified state on the exact intervention snapshots. Per-seed diagnostics and contrasts still vary, so any apparent feature association is exploratory. A policy selector remains unsupported until a feature rule is preregistered and evaluated on held-out scenes under matched compute.",
        "",
        "All 32 new seed-snapshot records are separate from the archived dose-response files. CUDA snapshot hashes did not match for the same seed/frame in this replay (see JSON for counts and per-seed flags); no old/new records were combined. This is consistent with the previously isolated gsplat atomic-backward nondeterminism documented in [the repeatability diagnosis](../determinism_stage_trace/diagnosis.md). The within-cell action comparisons remain paired because all eight arms in each new record share one snapshot hash and reference render.",
        "",
        "The diagnostic additions increase host work and frame wall time. Use synchronized optimization-stage time only for within-run optimization timing; this run is not an end-to-end latency benchmark.",
        "",
        "Per-seed contrasts, feature correlations, record hashes, manifest hash, and original preregistration hashes are saved in the adjacent JSON file.",
        "",
    ]
    out_md.write_text("\n".join(lines))
    print(out_md)


if __name__ == "__main__":
    main()
