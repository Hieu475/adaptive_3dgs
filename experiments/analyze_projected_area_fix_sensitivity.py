#!/usr/bin/env python3
"""Compare retained office1 snapshot traces from before/after area masking."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = (57, 58)
POLICIES = ("no_throttle", "coverage_throttle")


def read(folder, seed):
    path = RESULTS / f"{folder}{seed}" / "snapshot_prune_ablation.json"
    data = json.loads(path.read_text())
    if data["scene"] != "replica_office1" or data["seed"] != seed:
        raise ValueError(f"Unexpected scene/seed in {path}")
    if len(data["frames"]) != 119:
        raise ValueError(f"Expected 119 frames in {path}")
    return data


def means(data):
    rows = data["frames"]
    quality = {
        p: sum(row["natural_metrics"][p]["psnr"] for row in rows) / len(rows)
        for p in POLICIES
    }
    counts = {
        p: sum(row["natural_counts"][p] for row in rows) / len(rows)
        for p in POLICIES
    }
    return quality, counts


def main():
    pairs = {}
    for seed in SEEDS:
        before = read("replica_e1_snapshot_prune_office1_seed", seed)
        after = read("replica_e1_snapshot_prune_diagnostics_office1_seed", seed)
        pairs[seed] = (before, after, means(before), means(after))

    lines = [
        "# Projected-area fix sensitivity: office1", "",
        "Paired descriptive comparison of retained 119-frame natural trajectories for seeds 57 and 58. The before files are `replica_e1_snapshot_prune_office1_seed{57,58}`; after files are the diagnostic reruns generated after the projected-area validity mask was added.", "",
        "| Seed | Policy | Mean PSNR before | Mean PSNR after | Change | Mean map N before | Mean map N after | Change |", "|--:|:--|--:|--:|--:|--:|--:|--:|",
    ]
    delta_changes = []
    for seed, (_, _, (qb, cb), (qa, ca)) in pairs.items():
        for policy in POLICIES:
            lines.append(
                f"| {seed} | {policy} | {qb[policy]:.4f} | {qa[policy]:.4f} | "
                f"{qa[policy]-qb[policy]:+.4f} dB | {cb[policy]:.1f} | {ca[policy]:.1f} | "
                f"{ca[policy]-cb[policy]:+.1f} |"
            )
        before_delta = qb["coverage_throttle"] - qb["no_throttle"]
        after_delta = qa["coverage_throttle"] - qa["no_throttle"]
        delta_changes.append(after_delta - before_delta)
    mean_shift = sum(delta_changes) / len(delta_changes)
    lines += [
        "", "## Paired policy gap", "",
        f"Mean coverage−no-throttle PSNR gap changed by {mean_shift:+.4f} dB across the two seeds. This descriptive change does not reverse the observed office1 gap. It is not a multi-scene estimate.", "",
        "## Provenance limitation", "",
        "Both artifact generations report the same git commit SHA with a `-dirty` suffix. The earlier JSON was retained before the source edit in this work session, but the dirty working-tree diff was not hashed into the manifest. Treat this as a useful paired sensitivity check, not a clean-checkout reproducibility result. Re-run the paper-facing confirmation from a clean, recorded source state before publication.", "",
    ]
    out = RESULTS / "replica_e1_projected_area_fix_sensitivity.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
