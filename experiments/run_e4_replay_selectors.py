#!/usr/bin/env python3
"""E4: replay selector rankings on identical frozen oracle candidate pools.

Uses validation and cross-scene-test rows only. The oracle dataset records
pre-intervention features and single-Gaussian counterfactual labels; this is a
ranking-fidelity study, not a joint-update simulation.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.reproducibility import create_provenance_manifest

DATA = ROOT / "results/oracle_dataset/oracle_dataset.json"
OUT = ROOT / "results/matched_dilution_controls"
EVAL_SPLITS = {"validation", "cross_scene_test"}


def mean_ci_by_seed(rows, field, seed=2026, n_boot=20000):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row[field])
    values = np.asarray([np.mean(by_seed[s]) for s in sorted(by_seed)], dtype=float)
    if not len(values):
        return {"n_seeds": 0, "mean": None, "ci95_cluster_bootstrap": [None, None]}
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return {"n_seeds": len(values), "mean": float(values.mean()),
            "ci95_cluster_bootstrap": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "per_seed_mean": {str(s): float(np.mean(by_seed[s])) for s in sorted(by_seed)}}


def main():
    source = json.loads(DATA.read_text())
    rows = [r for r in source if r.get("split") in EVAL_SPLITS
            and r.get("visible", False) and not r.get("filtered", True)
            and r.get("features") is not None]

    rows.sort(key=lambda r: (r["seed"], r["scene"], r["frame"], r["persistent_id"]))
    candidate_frames = defaultdict(set)
    for r in rows:
        f = r["features"]
        err = float(f.get("rgb_error", 0.0)) + float(f.get("depth_error", 0.0))
        r["_error"] = err
        candidate_frames[(r["seed"], r["scene"], r["persistent_id"])].add(r["frame"])
    temporal_repeated_ids = sum(len(frames) > 1 for frames in candidate_frames.values())

    groups = defaultdict(list)
    for r in rows:
        groups[(r["seed"], r["scene"], r["frame"], r["split"])].append(r)

    methods = ("error_only", "error_x_influence")
    snapshot_results = []
    for (seed, scene, frame, split), candidates in sorted(groups.items()):
        n = len(candidates)
        k = max(1, math.ceil(0.20 * n))
        scores = {
            "error_only": [r["_error"] for r in candidates],
            "error_x_influence": [r["_error"] * float(r.get("influence_mass", 0.0)) for r in candidates],
        }
        oracle = np.asarray([float(r.get("delta_quality_global", r.get("delta_quality", 0.0)))
                             for r in candidates])
        oracle_order = np.argsort(-oracle, kind="stable")[:k]
        oracle_set = set(int(i) for i in oracle_order)
        oracle_gain = float(oracle[oracle_order].sum())
        for method in methods:
            selected = np.argsort(-np.asarray(scores[method]), kind="stable")[:k]
            selected_set = set(int(i) for i in selected)
            selected_utility = np.asarray([float(candidates[i].get("oracle_utility_joint", 0.0))
                                           for i in selected])
            selected_gain = float(oracle[selected].sum())
            snapshot_results.append({
                "seed": int(seed), "scene": scene, "frame": int(frame), "split": split,
                "method": method, "candidate_count": n, "k": k,
                "oracle_topk_overlap": len(selected_set & oracle_set) / k,
                "selected_mean_oracle_utility": float(selected_utility.mean()),
                "selected_negative_utility_fraction": float(np.mean(selected_utility < 0.0)),
                "additive_selected_delta_quality": selected_gain,
                "additive_oracle_regret": oracle_gain - selected_gain,
                "oracle_additive_gain": oracle_gain,
            })

    summaries = {}
    for method in methods:
        mr = [r for r in snapshot_results if r["method"] == method]
        summaries[method] = {
            field: mean_ci_by_seed(mr, field, seed=seed)
            for seed, field in enumerate(("oracle_topk_overlap", "selected_mean_oracle_utility",
                                          "selected_negative_utility_fraction",
                                          "additive_selected_delta_quality", "additive_oracle_regret"), 91)
        }

    contrasts = {}
    for method in methods[1:]:
        paired = []
        for group_key in sorted(groups):
            lookup = {(r["seed"], r["scene"], r["frame"], r["split"], r["method"]): r
                      for r in snapshot_results}
            a = lookup.get((*group_key, method))
            b = lookup.get((*group_key, "error_only"))
            if a and b:
                paired.append({"seed": int(group_key[0]), "scene": group_key[1],
                               "frame": int(group_key[2]), "split": group_key[3],
                               "delta_additive_regret": a["additive_oracle_regret"] - b["additive_oracle_regret"],
                               "delta_topk_overlap": a["oracle_topk_overlap"] - b["oracle_topk_overlap"],
                               "delta_selected_utility": a["selected_mean_oracle_utility"] - b["selected_mean_oracle_utility"]})
        contrasts[method + "_minus_error_only"] = {
            field: mean_ci_by_seed(paired, field, seed=111+i)
            for i, field in enumerate(("delta_additive_regret", "delta_topk_overlap", "delta_selected_utility"))
        }

    result = {
        "experiment": "E4_same_snapshot_candidate_replay",
        "status": "exploratory",
        "source": str(DATA.relative_to(ROOT)),
        "protocol": {
            "evaluation_splits": sorted(EVAL_SPLITS),
            "candidate_pool": "visible, unfiltered oracle rows grouped by seed/scene/frame/split",
            "quota": "top 20 percent of each identical candidate pool",
            "temporal_selector_evaluable": False,
            "temporal_selector_reason": "zero persistent-ID overlap across the evaluation snapshots; historical error cannot be replayed",
            "persistent_ids_repeated_across_evaluation_frames": temporal_repeated_ids,
            "primary_oracle": "single-action global delta_quality; top-k additive regret is not joint gain",
            "influence_proxy": "saved influence_mass; production score multiplies current error by screen area/contribution mass",
        },
        "n_candidate_rows": len(rows), "n_snapshots": len(groups),
        "snapshot_results": snapshot_results,
        "summaries_by_seed_cluster": summaries,
        "paired_contrasts": contrasts,
        "known_limitations": [
            "Oracle labels are single-Gaussian interventions; interactions make group utility non-additive.",
            "The frozen candidate set is geometry-stratified and sampled, not the full online visible population.",
            "Temporal influence selector is not evaluated: none of the sampled persistent IDs repeats across evaluation frames.",
            "Influence mass is an offline proxy for the production screen-area term.",
        ],
    }
    result["manifest"] = create_provenance_manifest(
        config=result["protocol"], dataset_name="oracle_dataset validation + cross_scene_test",
        renderer_backend="gsplat (source oracle artifacts)",
        extra_metadata={"experiment": "E4_same_snapshot_candidate_replay",
                        "source_artifact": "results/oracle_dataset/oracle_dataset.json",
                        "status": "exploratory", "seed_list": sorted({r["seed"] for r in rows}),
                        "pose_convention": "inherited from source oracle protocol; no new pose processing"},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E4_replay.json").write_text(json.dumps(result, indent=2))
    lines = ["# E4 — same-snapshot selector replay", "",
             "Validation and cross-scene test candidate pools; top 20% selected per identical snapshot pool.",
             "Intervals bootstrap seeds as clusters. Utility is measured from single-candidate interventions; additive regret is not joint-update gain.", "",
             f"Candidate rows: {len(rows)}; snapshot groups: {len(groups)}.", "",
             "| Selector | Top-k oracle overlap (95% CI) | Mean selected oracle utility (95% CI) | Negative utility fraction (95% CI) | Additive quality delta (95% CI) | Additive regret (95% CI) |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        s = summaries[method]
        keys = ("oracle_topk_overlap", "selected_mean_oracle_utility",
                "selected_negative_utility_fraction", "additive_selected_delta_quality",
                "additive_oracle_regret")
        vals = [f"{s[k]['mean']:.6g} [{s[k]['ci95_cluster_bootstrap'][0]:.6g}, {s[k]['ci95_cluster_bootstrap'][1]:.6g}]"
                for k in keys]
        lines.append(f"| {method} | " + " | ".join(vals) + " |")
    lines += ["", "## Paired contrast versus error-only", "",
              "Negative regret delta and positive overlap/utility delta favor the compared selector.", "",
              "| Selector | Δ additive regret (95% CI) | Δ top-k overlap (95% CI) | Δ selected utility (95% CI) |",
              "|---|---:|---:|---:|"]
    for method in methods[1:]:
        s = contrasts[method + "_minus_error_only"]
        keys = ("delta_additive_regret", "delta_topk_overlap", "delta_selected_utility")
        vals = [f"{s[k]['mean']:+.6g} [{s[k]['ci95_cluster_bootstrap'][0]:+.6g}, {s[k]['ci95_cluster_bootstrap'][1]:+.6g}]"
                for k in keys]
        lines.append(f"| {method} | " + " | ".join(vals) + " |")
    lines += ["", "Temporal selector replay was not estimable because no sampled persistent ID repeated across evaluation frames.",
              "", "## Limits", ""] + [f"- {x}" for x in result["known_limitations"]]
    (OUT / "E4_replay.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT / 'E4_replay.md'} and {OUT / 'E4_replay.json'}")


if __name__ == "__main__":
    main()
