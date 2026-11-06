#!/usr/bin/env python3
"""Validate and summarize the preregistered office1 quota dose-response."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    prereg = json.loads((root / "preregistration.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    source = (root.parent / prereg["snapshot_source_root"]
              if prereg.get("snapshot_source_root") else None)
    source_hashes = prereg.get("source_checkpoint_sha256", {})
    fractions = prereg["intervention"]["fixed_mature_fractions"]
    total = int(prereg["intervention"]["total_micro_updates_per_action"])
    cap = float(prereg["intervention"]["coverage_saturating_cap"])
    expected_keys = {(s, f) for s in prereg["seeds"] for f in prereg["snapshot_frames"]}
    cells = sorted((root / prereg["scene"]).glob("seed*_frame*.json"))
    records = {}
    grouped = defaultdict(list)
    new_checkpoint_hashes, map_hashes = {}, {}

    for path in cells:
        cell = json.loads(path.read_text())
        key = (cell["seed"], cell["snapshot_frame"])
        assert key not in records
        records[key] = cell
        assert cell["all_count_and_snapshot_gates_passed"], f"cell failed: {key}"
        assert len(cell["arms"]) == (len(fractions) + 1) * prereg["intervention"]["repeats_per_arm"]
        if source is not None:
            src_json = json.loads((source / prereg["scene"] / path.name).read_text())
            assert cell["base_snapshot_hash"] == src_json["base_snapshot_hash"], f"base snapshot differs: {key}"
            assert cell["common_candidate_pool_sha256"] == src_json["common_candidate_pool_sha256"], f"candidate pool differs: {key}"

        cp_rel, map_rel = cell["replay_checkpoint_path"], cell["map_snapshot_path"]
        assert sha256(root / cp_rel) == cell["replay_checkpoint_sha256"]
        assert sha256(root / map_rel) == cell["map_snapshot_sha256"]
        if source is not None:
            source_cp = source / "snapshots" / prereg["scene"] / f"seed{key[0]}_frame{key[1]}.checkpoint.pt"
            assert sha256(source_cp) == source_hashes[str(key[0])], f"preregistered source checkpoint hash mismatch: {key}"
            assert Path(cell["replay_checkpoint_source"]).resolve() == source_cp.resolve()
        new_checkpoint_hashes[str(key[0])] = cell["replay_checkpoint_sha256"]
        map_hashes[str(key[0])] = cell["map_snapshot_sha256"]

        for arm in cell["arms"]:
            assert arm["valid"] and arm["snapshot_hash"] == cell["base_snapshot_hash"]
            assert arm["n_spawned"] == arm["target_spawned"] == prereg["intervention"]["spawn_count"]
            assert arm["n_micro_updates"] == arm["target_micro_updates"] == total
            name = arm["action"]["name"]
            if arm["quota_policy"] == "coverage_saturating":
                policy = "adaptive"
                expected_mature = min(round(cap * total), arm["mature_eligible"])
            else:
                frac = float(arm["action"]["mature_fraction"])
                policy = f"fixed{int(round(frac * 100))}"
                assert frac in fractions, f"unregistered fraction {frac} in {name}"
                expected_mature = round(frac * total)
            assert arm["realized_mature_micro_updates"] == expected_mature, f"quota mismatch: {name}"
            for metric in ("final_map_eval_psnr", "final_map_eval_depth_l1", "opt_stage_sync_ms", "mature_update_coverage"):
                assert math.isfinite(arm[metric]), f"nonfinite {metric}: {key}/{name}"
            grouped[(key[0], key[1], policy)].append(arm)

    assert set(records) == expected_keys, f"missing/extra cells: {expected_keys-set(records)}, {set(records)-expected_keys}"
    policies = [f"fixed{int(round(x*100))}" for x in fractions] + ["adaptive"]
    for key in expected_keys:
        for policy in policies:
            assert len(grouped[(key[0], key[1], policy)]) == prereg["intervention"]["repeats_per_arm"], f"repeat count mismatch: {key}/{policy}"

    per_seed = []
    for seed, frame in sorted(expected_keys):
        row = {"seed": seed, "frame": frame, "policies": {}}
        for policy in policies:
            arms = grouped[(seed, frame, policy)]
            row["policies"][policy] = {k: mean([a[k] for a in arms]) for k in ("final_map_eval_psnr", "final_map_eval_depth_l1", "opt_stage_sync_ms", "realized_mature_fraction", "mature_update_coverage", "realized_mature_micro_updates", "mature_eligible")}
        diag = records[(seed, frame)]["arms"][0].get("age_stratified_diagnostics")
        if diag:
            y, m = diag["young_lt_10_frames"], diag["mature_ge_10_frames"]
            row["pre_action_age_residuals"] = {"young_color": y["visible_color_error_mean"], "mature_color": m["visible_color_error_mean"], "young_depth": y["visible_depth_error_mean"], "mature_depth": m["visible_depth_error_mean"], "young_visible": y["visible_attributed"], "mature_visible": m["visible_attributed"]}
        per_seed.append(row)

    pooled = {}
    for policy in policies:
        pooled[policy] = {key: mean([r["policies"][policy][key] for r in per_seed]) for key in ("final_map_eval_psnr", "final_map_eval_depth_l1", "opt_stage_sync_ms", "realized_mature_fraction", "mature_update_coverage", "realized_mature_micro_updates")}
    mean_psnr_best = max(pooled, key=lambda p: pooled[p]["final_map_eval_psnr"])
    adaptive_share = pooled["adaptive"]["realized_mature_fraction"]
    nearest_fixed_dose = min(
        (f"fixed{int(round(x*100))}" for x in fractions),
        key=lambda p: abs(pooled[p]["realized_mature_fraction"] - adaptive_share),
    )
    summary = {"protocol_version": prereg["protocol_version"], "preregistration_sha256": sha256(root / "preregistration.json"), "manifest_sha256": sha256(root / "manifest.json"), "config_sha256": manifest["config_sha256"], "git_sha": manifest["git_sha"], "source_checkpoint_sha256": source_hashes, "replayed_checkpoint_sha256": new_checkpoint_hashes, "map_snapshot_sha256": map_hashes, "cell_count": len(records), "branch_count": sum(len(c["arms"]) for c in records.values()), "valid_branch_count": sum(bool(a["valid"]) for c in records.values() for a in c["arms"]), "policies": policies, "per_seed": per_seed, "mean_across_seed_means": pooled, "best_mean_psnr_policy": mean_psnr_best}

    source_text = ("Replayed map digest and common candidate-pool hash matched the preregistered source run for every seed; source checkpoint hashes match preregistration." if source is not None else "The saved map snapshots and replay checkpoints were generated from each registered prefix; their hashes match the records.")
    lines = [f"# {prereg['scene']} quota dose-response", "", "## Protocol / validity", "", f"- {prereg['status']}.", f"- Scene `{prereg['scene']}`; seeds {', '.join(map(str, prereg['seeds']))}, frame {', '.join(map(str, prereg['snapshot_frames']))}; {prereg['intervention']['spawn_count']} spawns and {total:,} updates per arm; {prereg['intervention']['repeats_per_arm']} repeats per quota.", f"- Fixed quotas: {', '.join(f'{x:.0%}' for x in fractions)}; coverage-saturating arm retains cap {cap:.0%}.", f"- Validity: {len(records)}/{len(expected_keys)} cells and {summary['valid_branch_count']}/{summary['branch_count']} branches passed exact count/quota checks.", f"- {source_text}", f"- Backend `{manifest['renderer_backend']}`, GPU `{manifest['gpu']}`, revision `{manifest['git_sha']}`.", "", "## Results", "", f"Each cell below averages {prereg['intervention']['repeats_per_arm']} repeats. Seeds are the independent units; the last row equally averages the {len(prereg['seeds'])} seed means. Higher PSNR and lower depth L1 are favorable.", "", "| Seed | Fixed 25% | Fixed 40% | Fixed 49% | Fixed 60% | Fixed 75% | Adaptive |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_seed:
        cells_out = [str(r["seed"])] + [f"{r['policies'][p]['final_map_eval_psnr']:.4f} / {r['policies'][p]['final_map_eval_depth_l1']:.5f}" for p in policies]
        lines.append("| " + " | ".join(cells_out) + " |")
    cells_out = ["Mean"] + [f"{pooled[p]['final_map_eval_psnr']:.4f} / {pooled[p]['final_map_eval_depth_l1']:.5f}" for p in policies]
    lines.append("| " + " | ".join(cells_out) + " |")
    lines += ["", "Cell format: PSNR (dB) / depth L1 (m).", "", "| Policy | Mean mature updates | Mean mature share | Mean mature coverage | Mean optimizer-stage sync (ms) | Mean PSNR gain vs 25% (dB) | Mean depth L1 delta vs 25% (m) |", "|---|---:|---:|---:|---:|---:|---:|"]
    base = pooled["fixed25"]
    for p in policies:
        q = pooled[p]
        lines.append(f"| {p} | {q['realized_mature_micro_updates']:.1f} | {q['realized_mature_fraction']:.3f} | {q['mature_update_coverage']:.3f} | {q['opt_stage_sync_ms']:.2f} | {q['final_map_eval_psnr']-base['final_map_eval_psnr']:+.4f} | {q['final_map_eval_depth_l1']-base['final_map_eval_depth_l1']:+.5f} |")
    lines += ["", "## Pre-action age residuals", "", "The following are frame-level attribution diagnostics measured before the quota action; they are descriptive and do not establish residual-guided policy value.", "", "| Seed | Young/mature visible color error | Ratio | Young/mature visible depth error | Ratio |", "|---:|---:|---:|---:|---:|"]
    if all("pre_action_age_residuals" in r for r in per_seed):
        for r in per_seed:
            d = r["pre_action_age_residuals"]
            lines.append(f"| {r['seed']} | {d['young_color']:.5f} / {d['mature_color']:.5f} | {d['young_color']/d['mature_color']:.2f}× | {d['young_depth']:.5f} / {d['mature_depth']:.5f} | {d['young_depth']/d['mature_depth']:.2f}× |")
        y_color = mean([r["pre_action_age_residuals"]["young_color"] for r in per_seed])
        m_color = mean([r["pre_action_age_residuals"]["mature_color"] for r in per_seed])
        y_depth = mean([r["pre_action_age_residuals"]["young_depth"] for r in per_seed])
        m_depth = mean([r["pre_action_age_residuals"]["mature_depth"] for r in per_seed])
        lines.append(f"| Mean | {y_color:.5f} / {m_color:.5f} | {y_color/m_color:.2f}× | {y_depth:.5f} / {m_depth:.5f} | {y_depth/m_depth:.2f}× |")
    best_gain = pooled[mean_psnr_best]["final_map_eval_psnr"] - pooled["fixed25"]["final_map_eval_psnr"]
    adaptive_gain = pooled["adaptive"]["final_map_eval_psnr"] - pooled["fixed25"]["final_map_eval_psnr"]
    checkpoint_source_line = ("- Reused source checkpoint SHA-256 values are recorded in `preregistration.json`." if source_hashes else "- Prefix and replay-checkpoint hashes are recorded in `summary.json`; no prior snapshot checkpoint was reused.")
    lines += ["", "## Interpretation", "", f"The highest numerical mean PSNR among tested quotas was `{mean_psnr_best}` (Δ vs fixed-25%: {best_gain:+.4f} dB). The quota curve is conditional on these three frame-{prereg['snapshot_frames'][0]} maps and one current RGB-D view; it does not establish a globally optimal fraction or trajectory benefit. Adaptive realized {adaptive_share:.1%} mature updates and its nearest fixed-dose control is `{nearest_fixed_dose}`; adaptive ΔPSNR vs fixed-25% is {adaptive_gain:+.4f} dB. Their comparison tests whether eligibility-based rounding adds value beyond a fixed quota at this state.", "", "This is an exploratory dose-response, not independent confirmation. Update count is matched; optimizer wall time is reported but not controlled. No geometry completeness, novel-view, or long-horizon metric is included.", "", "## Provenance", "", f"- Preregistration SHA-256: `{summary['preregistration_sha256']}`", f"- Manifest SHA-256: `{summary['manifest_sha256']}`", checkpoint_source_line, "- Per-run checkpoint/map hashes and per-seed outcomes: `summary.json`", ""]
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "report.md").write_text("\n".join(lines))
    launch = {"protocol_version": prereg["protocol_version"], "preregistration_sha256": summary["preregistration_sha256"], "manifest_sha256": summary["manifest_sha256"], "config_sha256": summary["config_sha256"], "git_sha": manifest["git_sha"], "dirty_worktree": manifest["git_sha"].endswith("-dirty"), "source_checkpoint_sha256": source_hashes, "replayed_checkpoint_sha256": new_checkpoint_hashes, "map_snapshot_sha256": map_hashes}
    (root / "launch_config.json").write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
    print(f"Validated {len(records)} cells / {summary['valid_branch_count']} branches. Best mean-PSNR policy: {mean_psnr_best}")
    for p in policies:
        q = pooled[p]
        print(f"{p:>8}: PSNR={q['final_map_eval_psnr']:.5f}, depth={q['final_map_eval_depth_l1']:.6f}, mature={q['realized_mature_micro_updates']:.1f} updates")


if __name__ == "__main__":
    main()
