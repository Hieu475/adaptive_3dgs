#!/usr/bin/env python3
"""Check that headline numbers in the paper and README match the frozen result bundles.

CPU-only, no GPU needed. Exit code 1 on any mismatch.

    python paper_repro/check_numbers.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MAIN = RES / "phase14_projected_area_fix_clean_7policy_8seeds" / "final_confirmation_results.json"
FACT = RES / "dilution_factorial" / "analysis.json"
DEPTH = RES / "depth_intervention" / "analysis.json"
PAPERS = [ROOT / "docs/paper/iccv_official/main.tex"]
README = ROOT / "README.md"
# Numbers from superseded bundles that must not reappear in current text.
STALE = ["27.67", "98.0\\%", "98.0%", "93.6\\%", "93.6%", "12.34", "7005"]

failures = []


def need(text, needle, where, label):
    if needle not in text:
        failures.append(f"{where}: missing {label} = {needle!r}")


def main():
    main_d = json.loads(MAIN.read_text())
    stats = main_d["policy_stats"]
    ours = stats["ours"]["mean_psnr"]["mean"]
    noop = stats["no_op"]["mean_psnr"]["mean"]
    full = stats["full"]["mean_psnr"]["mean"]
    budgeted = {p: stats[p]["mean_psnr"]["mean"] for p in
                ("error_only", "error_influence", "error_influence_temporal", "rtg_slam_reimpl")}
    best = max(budgeted.values())
    expected = {
        "OURS mean PSNR": f"{ours:.2f}",
        "gain vs strongest budgeted": f"{ours - best:.2f}",
        "gain vs RTG reimpl": f"{ours - budgeted['rtg_slam_reimpl']:.2f}",
        "gain vs error_only": f"{ours - budgeted['error_only']:.2f}",
        "headroom vs no-op %": f"{100 * (ours - noop) / (full - noop):.1f}",
        "headroom vs best budgeted %": f"{100 * (ours - best) / (full - best):.1f}",
        "FULL minus OURS": f"{full - ours:.2f}",
        "map ratio FULL/OURS": f"{stats['full']['N_final']['mean'] / stats['ours']['N_final']['mean']:.1f}",
    }
    fact = json.loads(FACT.read_text())
    for scene in ("tum_fr2_xyz", "tum_fr1_xyz"):
        c = fact[scene]["contrasts"]
        expected[f"{scene} coverage|error"] = f"{c['coverage on error']['mean_db']:.2f}"
        expected[f"{scene} budget x4"] = f"{c['error budget x4']['mean_db']:.2f}"
        expected[f"{scene} cap23k"] = f"{c['error cap 23k']['mean_db']:.2f}"
    dep = json.loads(DEPTH.read_text())
    assert dep, "empty depth analysis"

    for paper in PAPERS:
        if not paper.exists():
            continue
        text = paper.read_text()
        rel = paper.relative_to(ROOT)
        for label, val in expected.items():
            need(text, val, str(rel), label)
        for s in STALE:
            if s in text:
                failures.append(f"{rel}: contains stale value {s!r}")
    rd = README.read_text()
    for label in ("OURS mean PSNR", "gain vs strongest budgeted", "headroom vs no-op %"):
        need(rd, expected[label], "README.md", label)
    for s in STALE:
        if re.search(re.escape(s), rd):
            failures.append(f"README.md: contains stale value {s!r}")

    print("Canonical bundle:", MAIN.parent.name)
    for k, v in expected.items():
        print(f"  {k:32s} {v}")
    if failures:
        print("\nMISMATCHES:")
        print("\n".join("  - " + f for f in failures))
        return 1
    print("\nOK: paper(s) and README agree with frozen bundles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
