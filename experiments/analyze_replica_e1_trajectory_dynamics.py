#!/usr/bin/env python3
"""Describe when Replica E1 natural-trajectory differences emerge.

This is a temporal decomposition of existing E1 runs, not a new causal
contrast: map size and map identity are not matched during the trajectory.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("office0", "office1", "office2", "office3", "office4",
          "room0", "room1", "room2")
FIELDS = ("psnr", "n_gaussians", "growth_pressure", "mature_update_coverage",
          "n_spawned")


def mean(rows, field):
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-a", type=int, default=44)
    parser.add_argument("--seed-b", type=int, default=45)
    parser.add_argument("--root-a", type=Path, default=ROOT / "results/replica_E1E2_seed44_capacity_fixed")
    parser.add_argument("--root-b", type=Path, default=ROOT / "results/replica_E1E2_seed45_capacity_fixed")
    parser.add_argument("--output", type=Path, default=ROOT / "results/replica_E1E2_seed_pair_comparison/e1_trajectory_dynamics")
    args = parser.parse_args()
    roots = [p if p.is_absolute() else ROOT / p for p in (args.root_a, args.root_b)]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    seeds = [args.seed_a, args.seed_b]
    units = []

    for root, seed in zip(roots, seeds):
        for scene in SCENES:
            path = root / f"replica_{scene}" / f"E1_seed{seed}.json"
            pair = json.loads(path.read_text())
            if not pair.get("valid"):
                raise ValueError(f"Invalid E1 count gate: {path}")
            by_condition = {arm["condition"]: arm for arm in pair["arms"]}
            no_throttle = by_condition["no_throttle_matched_final"]["frames"]
            coverage = by_condition["coverage_throttle_matched_final"]["frames"]
            n = min(len(no_throttle), len(coverage))
            if len(no_throttle) != len(coverage) or n < 3:
                raise ValueError(f"Unmatched or too-short trajectory logs: {path}")
            bounds = (0, n // 3, 2 * n // 3, n)
            for phase, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
                a, b = no_throttle[lo:hi], coverage[lo:hi]
                row = {"scene": scene, "seed": seed, "phase": phase,
                       "frames": [int(a[0]["frame"]), int(a[-1]["frame"])],
                       "n": len(a)}
                for field in FIELDS:
                    row[f"delta_{field}"] = mean(b, field) - mean(a, field)
                units.append(row)

    phases = []
    for phase in (1, 2, 3):
        rows = [r for r in units if r["phase"] == phase]
        phases.append({
            "phase": phase,
            "n_scene_seed_units": len(rows),
            "deltas": {field: mean(rows, f"delta_{field}") for field in FIELDS},
            "psnr_wins_coverage": sum(r["delta_psnr"] > 0 for r in rows),
            "psnr_losses_coverage": sum(r["delta_psnr"] < 0 for r in rows),
        })
    result = {
        "experiment": "replica_e1_natural_trajectory_temporal_decomposition",
        "status": "descriptive_exploratory",
        "seeds": seeds,
        "scenes": list(SCENES),
        "phase_definition": "Each trajectory divided into three equal contiguous frame-index thirds.",
        "delta_definition": "Coverage-throttle natural trajectory minus no-throttle natural trajectory; frame metrics averaged within each phase before paired differences are summarized.",
        "units": units,
        "phase_summary": phases,
        "limitations": [
            "E1 matches Gaussian count only after the natural trajectory at the final snapshot; intermediate map size and identity differ.",
            "Temporal phase summaries are descriptive; frames are not independent samples.",
            "Association between map growth and quality does not establish mediation or causality.",
            "Only two seeds and eight short Replica prefixes are included.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Replica E1 natural-trajectory temporal decomposition", "",
        f"Seeds {seeds[0]} and {seeds[1]}, eight scenes. Delta = coverage-throttle minus no-throttle on the natural trajectories.",
        "Each trajectory is split into three equal frame-index thirds; values are paired scene-seed means. This is a descriptive re-analysis, not a new causal experiment.", "",
        "| Phase | Units | PSNR Δ (dB) | PSNR wins/losses | Gaussian count Δ | Growth pressure Δ | Mature update coverage Δ | Spawned/frame Δ |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for p in phases:
        d = p["deltas"]
        lines.append(
            f"| {('first third' if p['phase'] == 1 else 'middle third' if p['phase'] == 2 else 'final third')} | {p['n_scene_seed_units']} | "
            f"{d['psnr']:+.4f} | {p['psnr_wins_coverage']}/{p['psnr_losses_coverage']} | "
            f"{d['n_gaussians']:+.0f} | {d['growth_pressure']:+.5f} | "
            f"{d['mature_update_coverage']:+.4f} | {d['n_spawned']:+.1f} |"
        )
    lines += ["", "## Limits", ""] + [f"- {x}" for x in result["limitations"]]
    lines.append("")
    (output / "analysis.md").write_text("\n".join(lines))
    print(output / "analysis.md")


if __name__ == "__main__":
    main()
