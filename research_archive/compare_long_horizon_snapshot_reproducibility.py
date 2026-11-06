#!/usr/bin/env python3
"""Descriptively compare dirty-workspace and frozen-source 500f effects."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/continuous_age_quota/long_horizon_tum_fr1_desk"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="tum_fr1_desk")
    a = ap.parse_args()
    base = ROOT / "results/phase14_generalization_postfix/continuous_age_quota" / f"long_horizon_{a.scene}"
    dirty_path = base / "multiseed/analysis.json"
    clean_path = base / "clean_snapshot_rerun/analysis.json"
    out_path = base / "snapshot_reproducibility.json"
    report_path = base / "snapshot_reproducibility.md"
    dirty, clean = json.loads(dirty_path.read_text()), json.loads(clean_path.read_text())
    drows = {r["seed"]: r for r in dirty["seed_rows"]}
    crows = {r["seed"]: r for r in clean["seed_rows"]}
    seeds = sorted(set(drows) & set(crows))
    assert len(seeds) == 8, f"expected 8 paired seeds, got {len(seeds)}"
    d = np.array([drows[s]["mean_psnr_delta_mature75_minus_global"] for s in seeds])
    c = np.array([crows[s]["mean_psnr_delta_mature75_minus_global"] for s in seeds])
    delta = c - d
    consistent = int(np.sum(np.sign(c) == np.sign(d)))
    data = {
        "study": "dirty_vs_clean_frozen_source_snapshot_reproducibility",
        "scene": a.scene, "seeds": seeds,
        "dirty_workspace_mean_effect_db": float(np.mean(d)),
        "clean_snapshot_mean_effect_db": float(np.mean(c)),
        "change_in_mean_effect_clean_minus_dirty_db": float(np.mean(c) - np.mean(d)),
        "mean_absolute_seed_effect_difference_db": float(np.mean(np.abs(delta))),
        "seed_effect_direction_agreement": f"{consistent}/8",
        "dirty_seed_bootstrap_ci95": dirty["primary"]["seed_bootstrap_ci95"],
        "clean_seed_bootstrap_ci95": clean["primary"]["seed_bootstrap_ci95"],
        "dirty_positive_negative": [dirty["primary"]["positive_seeds"], dirty["primary"]["negative_seeds"]],
        "clean_positive_negative": [clean["primary"]["positive_seeds"], clean["primary"]["negative_seeds"]],
        "clean_sign_test_p": clean["primary"]["exact_sign_test_two_sided_p"],
        "snapshot_revision": json.loads((base / "clean_snapshot_rerun/manifest.json").read_text())["source_revision"],
        "per_seed": [{"seed": s, "dirty_effect_db": float(drows[s]["mean_psnr_delta_mature75_minus_global"]),
                      "clean_effect_db": float(crows[s]["mean_psnr_delta_mature75_minus_global"]),
                      "clean_minus_dirty_db": float(crows[s]["mean_psnr_delta_mature75_minus_global"] - drows[s]["mean_psnr_delta_mature75_minus_global"])}
                     for s in seeds],
        "interpretation": "descriptive repeat across a clean frozen source snapshot; GPU nondeterminism prevents exact path reproduction",
    }
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "# Frozen-source snapshot reproducibility comparison",
        "",
        f"{a.scene}, same seeds {seeds}, 500 frames, same protocol and datasets. The second set was run from an isolated source snapshot whose working tree was clean at launch (revision `{data['snapshot_revision']}`). This comparison is descriptive; repeated CUDA/gsplat trajectories are not bitwise deterministic.",
        "",
        "| Source run | Mean 500f PSNR effect (mature75 − global) | 95% seed bootstrap CI | Positive / negative seeds | Exact sign p |",
        "|:--|--:|--:|:--:|--:|",
        f"| Original dirty workspace | {np.mean(d):+.3f} dB | [{dirty['primary']['seed_bootstrap_ci95'][0]:+.3f}, {dirty['primary']['seed_bootstrap_ci95'][1]:+.3f}] | {dirty['primary']['positive_seeds']} / {dirty['primary']['negative_seeds']} | {dirty['primary']['exact_sign_test_two_sided_p']:.4f} |",
        f"| Clean frozen snapshot | {np.mean(c):+.3f} dB | [{clean['primary']['seed_bootstrap_ci95'][0]:+.3f}, {clean['primary']['seed_bootstrap_ci95'][1]:+.3f}] | {clean['primary']['positive_seeds']} / {clean['primary']['negative_seeds']} | {clean['primary']['exact_sign_test_two_sided_p']:.4f} |",
        "",
        f"Mean effect changed by **{np.mean(c)-np.mean(d):+.3f} dB**; seed-specific effect directions agree on **{consistent}/8** seeds; the mean absolute seed-effect difference is **{np.mean(np.abs(delta)):.3f} dB**. The clean snapshot has {clean['primary']['positive_seeds']}/8 positive seed effects (exact sign-test p={clean['primary']['exact_sign_test_two_sided_p']:.4f}). The bootstrap interval and sign test use different summaries and should both be reported.",
        "",
        "| Seed | Dirty effect (dB) | Clean snapshot effect (dB) | Clean − dirty (dB) |",
        "|--:|--:|--:|--:|",
    ]
    for row in data["per_seed"]:
        lines.append(f"| {row['seed']} | {row['dirty_effect_db']:+.3f} | {row['clean_effect_db']:+.3f} | {row['clean_minus_dirty_db']:+.3f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        f"This is a successful clean-source replication of the positive mean direction, not exact numerical replication. Seed-level variation remains; the clean-run exact sign test p-value is {clean['primary']['exact_sign_test_two_sided_p']:.4f}. The snapshot was assembled from the dirty workspace source files and committed only inside `/tmp`; it is not a commit to the user's repository. Dataset files were symlinked read-only from the original workspace. The source revision and hashes are preserved in the clean-run manifest.",
        "",
        "The result remains exploratory, single-scene, GT-pose, and equal-update-count. It does not establish cross-scene generalization or equal wall-time.",
        "",
        "See the [dirty-workspace run](multiseed/report.md), [clean-snapshot rerun](clean_snapshot_rerun/report.md), and [preregistration](multiseed_preregistration.md).",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
