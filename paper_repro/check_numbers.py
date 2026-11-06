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

    for sc, wins in nums.get("long", {}).items():
        for w, v in wins.items():
            for k in ("base", "cov", "prop"):
                if f"{abs(v[k]):.2f}" not in main and not (w == "151-300"):
                    fail(f"main.tex missing long-horizon {sc} {w} {k} = {v[k]:.2f}")

    ab = nums.get("ablation", {})
    cited = {"error_coverage": [0, 1, 2], "cov_spawn_only": [0, 1, 2], "cov_thresh_only": [0, 1, 2],
             "spawn_cap_always20": [0, 1, 2, 3], "spawn_cap_always50": [0, 1, 2, 3],
             "cov_plus_prop": [0, 2, 3], "error_budget_prop": [0, 2, 3]}
    for cond, idxs in cited.items():
        for i in idxs:
            if f"{abs(ab[cond][i]):.2f}" not in main:
                fail(f"main.tex missing ablation {cond}[{i}] = {ab[cond][i]:.2f}")

    ho = nums.get("holdout")
    if ho:
        prop = ho["deltas"]["error_budget_prop"]
        for tr, h in prop:
            for v in (tr, h):
                if f"{abs(v):.2f}" not in main:
                    fail(f"main.tex missing held-out proportional-budget value {v:.2f}")
        for tr, h in ho["deltas"]["error_coverage"]:
            if f"{abs(h):.2f}" not in main:
                fail(f"main.tex missing held-out coverage value {h:.2f}")
        ret = [h / tr for tr, h in prop]
        if not (0.465 <= min(ret) and max(ret) <= 0.765):
            fail(f"retained fraction range changed: {min(ret):.2f}-{max(ret):.2f}; update the 47-76% text")
        if f"{ho['gap_base'][2]:.2f}" not in main or f"{ho['gap_prop'][2]:.2f}" not in main:
            fail("main.tex missing office0 gap values")

    for sc, cells in nums.get("compute", {}).items():
        for k in ("free_x1", "free_x4", "spawn_x1", "spawn_x2"):
            pass
    comp = nums.get("compute", {})
    need = [("tum_fr2_xyz", "spawn_x1", 0), ("tum_fr2_xyz", "spawn_x1", 1), ("tum_fr2_xyz", "free_x1", 0), ("tum_fr2_xyz", "free_x1", 1),
            ("tum_fr2_xyz", "free_x4", 0), ("tum_fr2_xyz", "free_x4", 1), ("tum_fr1_xyz", "spawn_x2", 0), ("tum_fr1_xyz", "spawn_x2", 1),
            ("tum_fr1_xyz", "free_x4", 0), ("tum_fr1_xyz", "free_x4", 1), ("replica_office0", "free_x1", 1), ("replica_office0", "free_x4", 1)]
    for sc, cell, i in need:
        v = comp[sc][cell][i]
        if (f"{v:.2f}" if i == 0 else f"{v:d}") not in main:
            fail(f"main.tex missing compute-matched {sc} {cell}[{i}] = {v}")
    cm = nums.get("compute_matched", {})
    for sc, keys in (("replica_office0", ("spawn_x2", "spawn_x4")), ("replica_office1", ("spawn_x2", "spawn_x4"))):
        for k in keys:
            if f"{abs(cm[sc][k]):.2f}" not in main:
                fail(f"main.tex missing equal-time gain {sc} {k} = {cm[sc][k]}")
    for sc, v in nums.get("tracked", {}).items():
        for key in ("base_tracked", "base_gt"):
            if f"{v[key]:.2f}" not in main:
                fail(f"main.tex missing tracked {sc} {key} = {v[key]}")
        conds = ("error_coverage", "spawn_cap_always20", "error_budget_prop")
        for idx in (0, 1):  # 0: tracked-pose delta, 1: GT-pose delta; the text quotes the min and max
            vals = [v[c][idx] for c in conds]
            for x in (min(vals), max(vals)):
                if f"{abs(x):.2f}" not in main:
                    fail(f"main.tex missing tracked {sc} range endpoint {x}")

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
