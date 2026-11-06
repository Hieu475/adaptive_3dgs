#!/usr/bin/env python3
"""Summarize same-snapshot, no-feedback pruning counterfactuals."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(seed):
    path = RESULTS / f"replica_e1_snapshot_prune_diagnostics_office1_seed{seed}" / "snapshot_prune_ablation.json"
    d = json.loads(path.read_text())
    rows = d["frames"]
    s = d["summary"]
    if not s["all_frame_targets_exact"] or len(rows) != s["requested_online_frames"]:
        raise ValueError(f"Incomplete count gate in {path}")
    for row in rows:
        for policy in ("no_throttle", "coverage_throttle"):
            finite_state = row.get("finite_state", {}).get(policy, {})
            for state_name in ("confidence", "screen_area"):
                state = finite_state.get(state_name)
                if state is None or state["nonfinite_count"] != 0:
                    raise ValueError(
                        f"Non-finite {state_name} in {path}, frame {row['online_frame']}, {policy}"
                    )
            for selector in ("lowest_opacity", "uniform_random"):
                metrics = row["counterfactuals"][policy][selector]
                if metrics["n_gaussians"] != row["snapshot_target"]:
                    raise ValueError(f"Count mismatch in {path}, frame {row['online_frame']}")
                if not all(math.isfinite(float(metrics[k])) for k in ("psnr", "ssim", "depth_l1")):
                    raise ValueError(f"Non-finite counterfactual metric in {path}, frame {row['online_frame']}")
    natural_delta = sum(r["natural_metrics"]["coverage_throttle"]["psnr"]
                        - r["natural_metrics"]["no_throttle"]["psnr"] for r in rows) / len(rows)
    diagnostics = {}
    for selector in ("lowest_opacity", "uniform_random"):
        samples = [(r["counterfactuals"]["no_throttle"][selector]["removed_gaussian_diagnostics"])
                   for r in rows]
        total = sum(x["count"] for x in samples)
        keys = ("mean_opacity", "mean_age_frames",
                "mature_fraction_age_ge_10", "stable_or_frozen_fraction")
        diagnostics[selector] = {
            "removed_count": total,
            **{k: (sum(x[k] * x["count"] for x in samples) / total if total else 0.0) for k in keys},
            "mean_confidence": (sum(x["mean_confidence"] * x["confidence_finite_count"]
                                    for x in samples if x["mean_confidence"] is not None)
                                / max(sum(x["confidence_finite_count"] for x in samples), 1)),
            "confidence_finite_fraction": (sum(x["confidence_finite_count"] for x in samples)
                                           / max(total, 1)),
            "mean_spatial_occupancy_4cube": (sum(x["spatial_occupancy_4cube"] for x in samples if x["count"])
                                               / max(sum(x["count"] > 0 for x in samples), 1)),
        }
    return d, natural_delta, diagnostics


def main():
    data = {seed: load(seed) for seed in (57, 58)}
    lines = [
        "# Same-snapshot E1 pruning counterfactuals: office1", "",
        "Two held-out seeds, 119 frames each. The natural mapping trajectories remain untouched; opacity and random pruning are rendered as immediate counterfactuals on the same per-policy snapshot, at the same per-frame target count.", "",
        "| Seed | Natural PSNR Δ coverage−no-throttle | Selector | Post-prune PSNR Δ | Post-prune Depth L1 Δ | No-throttle prune PSNR effect | Coverage prune PSNR effect | Mean target N | Gate |", "|--:|--:|:--|--:|--:|--:|--:|--:|:--:|",
    ]
    for seed, (d, natural_delta, _) in data.items():
        s = d["summary"]
        for selector in ("lowest_opacity", "uniform_random"):
            delta = s["mean_counterfactual_delta_coverage_minus_no_throttle"][selector]
            effects = s["mean_immediate_prune_effect_by_policy"]
            lines.append(f"| {seed} | {natural_delta:+.3f} dB | {selector} | {delta['psnr']:+.3f} dB | {delta['depth_l1']:+.5f} m | {effects['no_throttle'][selector]['psnr']:+.3f} dB | {effects['coverage_throttle'][selector]['psnr']:+.3f} dB | {s['mean_snapshot_target']:.0f} | PASS |")
    selectors = ("lowest_opacity", "uniform_random")
    means = {}
    for selector in selectors:
        means[selector] = {
            metric: sum(d[0]["summary"]["mean_counterfactual_delta_coverage_minus_no_throttle"][selector][metric]
                        for d in data.values()) / len(data)
            for metric in ("psnr", "ssim", "depth_l1")
        }
        means[selector]["no_prune_psnr"] = sum(
            d[0]["summary"]["mean_immediate_prune_effect_by_policy"]["no_throttle"][selector]["psnr"]
            for d in data.values()) / len(data)
    opacity_damage = means["lowest_opacity"]["no_prune_psnr"]
    random_damage = means["uniform_random"]["no_prune_psnr"]
    lines += ["", "## What was removed from the larger no-throttle map", "",
              "Means are weighted by frame-level removed-Gaussian events (the same Gaussian can count again on later frames); spatial occupancy is averaged over frames with nonzero removal. Confidence mean uses finite values only; coverage is reported.", "",
              "| Seed | Selector | Removal events | Mean opacity | Mean confidence (finite coverage) | Mean age | Mature fraction (age ≥ 10) | Stable/frozen fraction | 4³-bin occupancy |",
              "|--:|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for seed, (_, _, diag_by_selector) in data.items():
        for selector in selectors:
            x = diag_by_selector[selector]
            conf = (f"{x['mean_confidence']:.4f} ({x['confidence_finite_fraction']:.1%})"
                    if x["confidence_finite_fraction"] > 0 else "NA (0%)")
            lines.append(f"| {seed} | {selector} | {x['removed_count']:,} | {x['mean_opacity']:.4f} | {conf} | {x['mean_age_frames']:.1f} | {x['mature_fraction_age_ge_10']:.3f} | {x['stable_or_frozen_fraction']:.3f} | {x['mean_spatial_occupancy_4cube']:.3f} |")
    lines += ["", "## Two-seed mean", "",
              f"- Natural, unpruned PSNR delta: {sum(x[1] for x in data.values())/len(data):+.3f} dB.",
              f"- Immediate opacity-prune counterfactual: PSNR delta {means['lowest_opacity']['psnr']:+.3f} dB; Depth L1 delta {means['lowest_opacity']['depth_l1']:+.5f} m; no-throttle arm changes {opacity_damage:+.3f} dB.",
              f"- Immediate random-prune counterfactual: PSNR delta {means['uniform_random']['psnr']:+.3f} dB; Depth L1 delta {means['uniform_random']['depth_l1']:+.5f} m; no-throttle arm changes {random_damage:+.3f} dB.",
              f"- The no-throttle immediate prune damage differs by {abs(opacity_damage-random_damage):.3f} dB between selectors.", "",
              "## Projected-area NaN fix validation", "",
              "Fast attribution now sets projected area to zero for Gaussians at camera depth ≤ 0.1 or with non-finite projected area. This prevents behind/near-camera projections from contaminating importance normalization and confidence.", "",
              "| Seed | Frames | Policy traces with non-finite confidence | Policy traces with non-finite screen area | Confidence coverage among removed events |", "|--:|--:|--:|--:|--:|"]
    for seed, (d, _, diag_by_selector) in data.items():
        rows = d["frames"]
        bad_conf = sum(row["finite_state"][policy]["confidence"]["nonfinite_count"] > 0
                       for row in rows for policy in ("no_throttle", "coverage_throttle"))
        bad_area = sum(row["finite_state"][policy]["screen_area"]["nonfinite_count"] > 0
                       for row in rows for policy in ("no_throttle", "coverage_throttle"))
        removed = sum(diag_by_selector[s]["removed_count"] for s in selectors)
        finite_removed = sum(diag_by_selector[s]["removed_count"] * diag_by_selector[s]["confidence_finite_fraction"] for s in selectors)
        coverage = finite_removed / max(removed, 1)
        lines.append(f"| {seed} | {len(rows)} | {bad_conf} | {bad_area} | {coverage:.1%} |")
    lines += ["", "This validates finite values on the two office1 diagnostic traces only. Because projected area also feeds importance and scheduling, metrics from runs performed before this fix are not equivalent and need rerunning before paper use.", "",
              "## Interpretation", "",
              "This is a one-step snapshot ablation: each pruning selector is applied to the same map state for a render-only counterfactual, and the live pipeline then continues without pruning. It identifies immediate rendering damage from selector choice conditional on these snapshots. The positive post-prune coverage-minus-no-throttle PSNR delta is largely produced by pruning the larger no-throttle map, not by an online policy effect. This does not measure how either pruned map would optimize on later frames, and it is limited to one scene and two seeds.", ""]
    out = RESULTS / "replica_e1_snapshot_prune_ablation_office1.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
