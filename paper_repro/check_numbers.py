#!/usr/bin/env python3
"""Check that paper, generated tables and README agree with the frozen result bundles.

CPU-only. Exit code 1 on any mismatch. Source of truth:
  results/rerun_paper1/            main benchmark + factorial (git tag paper-rerun-1)
  results/dilution_extension/      Replica factorial, proportional budget
  results/dilution_dose*/          dose-response
The paper takes its headline numbers from docs/paper/iccv_official/tables/macros.tex, which
`make_main_tables.py` generates from the bundles; this script regenerates the expected values
and verifies (1) the generated files are fresh, (2) the paper uses them, (3) no superseded
values remain in the paper or README.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "docs/paper/iccv_official/tables"
MAIN_TEX = ROOT / "docs/paper/iccv_official/main.tex"
SUPP_TEX = ROOT / "docs/paper/iccv_official/supp.tex"
README = ROOT / "README.md"
# Values from superseded bundles / evaluators that must not reappear.
STALE = ["27.64", "27.67", "98.0\\%", "98.0%", "93.6", "12.34", "7005", "97.6\\%", "92.5\\%", "3.8\\times", "$4.48$"]
failures = []


def fail(msg):
    failures.append(msg)


def main():
    # (1) regenerate and compare with the committed outputs
    before = {p.name: p.read_text() for p in TABLES.glob("*") if p.suffix in (".tex", ".json")}
    for script in ("make_main_tables.py", "make_tables.py"):
        subprocess.run([sys.executable, str(ROOT / "paper_repro" / script)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for name, text in before.items():
        if (TABLES / name).read_text() != text:
            fail(f"tables/{name} was stale (regenerated from JSON; commit the update)")
    macros = json.loads((TABLES / "macros.json").read_text())
    nums = json.loads((TABLES / "numbers.json").read_text())

    # (2) the paper uses generated content
    main = MAIN_TEX.read_text()
    for token in ("tables/macros.tex", "tables/tab_main.tex", "tables/tab_factorial.tex", "tables/par_core.tex",
                  "tables/tab_replica.tex", "tables/tab_dose.tex"):
        if token not in main:
            fail(f"main.tex does not input {token}")
    for key in ("oursPSNR", "gainBest", "gapFull", "mapRatio", "covErrA", "covErrB"):
        if f"\\{key}" not in main:
            fail(f"main.tex does not use macro \\{key}")
    for sc in ("replica_office0", "replica_office1"):
        for key in ("coverage on error", "error budget x4", "error budget proportional to map",
                    "selector without throttle"):
            if f"{abs(nums[sc][key]):.2f}" not in main:
                fail(f"main.tex missing {sc} {key} = {abs(nums[sc][key]):.2f}")
    for sc in ("tum_fr2_xyz", "tum_fr1_xyz"):
        if f"{nums[sc]['error budget proportional to map']:.2f}" not in main:
            fail(f"main.tex missing {sc} proportional budget")
    for sc, v in nums["dose"].items():
        if f"{abs(v['rho_all']):.2f}" not in main:
            fail(f"main.tex missing {sc} rho_all = {v['rho_all']}")
        if v["rho_free"] != 1.0 or v["n_seeds"] != 8:
            fail(f"{sc}: rho_free/n_seeds changed ({v['rho_free']}, {v['n_seeds']}); revisit the +1.00 claim")

    sl = nums["scene_level"]
    for key, needle in (("prop_pos", f"all {sl['n']} scenes"), ("prop_median", f"{sl['prop_median']:.2f}"),
                        ("prop_max", f"{sl['prop_max']:.2f}"), ("prop_min", f"{sl['prop_min']:.2f}"),
                        ("cov_pos", f"{sl['cov_pos']} of {sl['n']}"), ("cov_median", f"{abs(sl['cov_median']):.2f}"),
                        ("cov_min", f"{abs(sl['cov_min']):.2f}"), ("cov_max", f"{sl['cov_max']:.2f}"),
                        ("rho", f"{sl['rho']:.2f}"), ("rho_p", f"{sl['rho_p']:.3f}")):
        if needle not in main:
            fail(f"main.tex missing scene-level {key}: {needle!r}")
    if sl["prop_pos"] != sl["n"]:
        fail("not all scenes positive for the proportional budget; revise the 'all 15 scenes' claims")

    # (3) README + stale values
    rd = README.read_text()
    for key in ("oursPSNR", "gainBest", "headNoop"):
        if macros[key] not in rd:
            fail(f"README.md missing {key} = {macros[key]}")
    supp = SUPP_TEX.read_text()
    if "before the evaluator correction" not in supp:
        fail("supp.tex lacks the earlier-revision notice for Tabs. S4-S6")
    texts = {"main.tex": main, "README.md": rd}
    for name, text in texts.items():
        for s in STALE:
            if s in text:
                fail(f"{name}: contains superseded value {s!r}")

    print("Canonical bundles: results/rerun_paper1 (tag paper-rerun-1), dilution_extension, dilution_dose*")
    for k, v in macros.items():
        print(f"  {k:12s} {v}")
    if failures:
        print("\nMISMATCHES:")
        print("\n".join("  - " + f for f in failures))
        return 1
    print("\nOK: paper, generated tables and README agree with the frozen bundles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
