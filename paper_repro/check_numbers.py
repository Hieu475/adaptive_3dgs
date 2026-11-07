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
OLD_TEX = ROOT / "docs/paper/iccv_official/old_analysis.tex"   # earlier analysis (moved from the main paper)
README = ROOT / "README.md"
# Values from superseded bundles / evaluators that must not reappear.
STALE = ["27.64", "27.67", "98.0\\%", "98.0%", "93.6", "12.34", "7005", "97.6\\%", "92.5\\%", "3.8\\times", "$4.48$"]
failures = []


def fail(msg):
    failures.append(msg)


def main():
    # (1) regenerate and compare with the committed outputs
    before = {p.name: p.read_text() for p in TABLES.glob("*") if p.suffix in (".tex", ".json")}
    for script in ("make_main_tables.py", "make_tables.py", "make_engine.py"):
        subprocess.run([sys.executable, str(ROOT / "paper_repro" / script)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for name, text in before.items():
        if (TABLES / name).read_text() != text:
            fail(f"tables/{name} was stale (regenerated from JSON; commit the update)")
    macros = json.loads((TABLES / "macros.json").read_text())
    nums = json.loads((TABLES / "numbers.json").read_text())

    # (2) the paper uses generated content
    main_only = MAIN_TEX.read_text() + "\n" + OLD_TEX.read_text()   # earlier analysis now lives in old_analysis.tex
    main = main_only + "\n" + SUPP_TEX.read_text()   # moved diagnostics live in the supplement
    for token in ("tables/macros.tex", "tables/tab_main.tex", "tables/tab_factorial.tex", "tables/par_core.tex",
                  "tables/tab_replica.tex", "tables/tab_dose.tex"):
        if token not in main_only and token not in ("tables/tab_dose.tex", "tables/tab_replica.tex") :
            fail(f"main.tex does not input {token}")
    import re
    eng_defined = set(re.findall(r"\\newcommand\{\\(eng\w+)\}", (TABLES / "engine_macros.tex").read_text()))
    eng_used = set(re.findall(r"\\(eng[A-Za-z]+)", MAIN_TEX.read_text() + SUPP_TEX.read_text()))
    for k in sorted(eng_used - eng_defined):
        fail(f"text uses undefined engine macro \\{k}")
    if not eng_used:
        fail("paper does not use the generated engine macros")
    en = json.loads((TABLES / "engine_numbers.json").read_text())
    readme = README.read_text()
    for key, fmt in (("engTestTtsVggTFourFiveGainMean", "{:.2f}"), ("engFreshTtsVggTFourFiveGainMean", "{:.2f}"), ("engAblNooptDeltaAbs", "{:.2f}"),
                     ("engHitMaxPct", "{:.1f}"), ("engCostSlope", "{:.1f}")):
        if fmt.format(en[key]) not in readme:
            fail(f"README missing engine number {key} = {fmt.format(en[key])}")
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

    supp = SUPP_TEX.read_text()
    cc = nums.get("countcap")
    if cc:
        for lab, txt in (("tt_vgg", main + supp), ("tt_countcap", main + supp)):
            key = "mean_" + lab
            if f"{abs(cc[key]):.2f}" not in txt:
                fail(f"text missing count-cap {key} = {cc[key]}")
        if f"{abs(cc['median_vgg_minus_tcc']):.2f}" not in supp or str(cc["vgg_better_than_tcc"]) not in main:
            fail("count-cap median difference or win count missing")
    cl = nums.get("closedloop")
    if cl:
        sm = cl["summary"]
        for lab, v in (("mean_dpsnr", sm["mean_dpsnr"]), ("median_date_cm", sm["median_date_cm"])):
            if f"{abs(v):.2f}" not in supp:
                fail(f"supp.tex missing closed-loop {lab} = {v}")

    me = nums.get("method")
    if me:
        tv, ft = me["tt_vgg"]["test"], me["tt_vgg"]["fresh"]
        need = {"tt_vgg test median": f"{tv['median']:.2f}", "tt_vgg test mean": f"{tv['mean']:.2f}", "tt_vgg test worst": f"{tv['min']:.2f}",
                "tt_vgg fresh mean": f"{ft['mean']:.2f}", "tt mean": f"{me['tt']['test']['mean']:.2f}",
                "cap worst": f"{abs(me['spawn_cap_always20']['test']['min']):.2f}", "prop mean": f"{me['error_budget_prop']['test']['mean']:.2f}",
                "prop dt": f"{me['error_budget_prop']['test']['mean_time_delta_ms']:.0f}", "tt+cap worst": f"{abs(me['tt_cap']['test']['min']):.2f}",
                "tt_vgg dt": f"{tv['mean_time_delta_ms']:.0f}", "vgg-over-tt test mean": f"{nums['method_vgg_over_tt']['test']['mean']:.2f}",
                "vgg-over-tt fresh mean": f"{nums['method_vgg_over_tt']['fresh']['mean']:.2f}"}
        for k, v in need.items():
            if v not in main_only:
                fail(f"main.tex missing controller number {k} = {v}")
        if me["tt_vgg"]["test"]["pos"] != me["tt_vgg"]["test"]["n"] or me["tt_vgg"]["fresh"]["pos"] != me["tt_vgg"]["fresh"]["n"]:
            fail("TT+VGG no longer positive on every scene/window; revise the 'all scenes' claims")
        ps = nums["method_scene"]
        for sc, k in (("tum_fr2_xyz", "tt"), ("tum_fr2_xyz", "tt_vgg"), ("tum_fr1_xyz", "tt_cap"), ("tum_fr1_xyz", "tt_vgg")):
            if f"{ps[sc][k]:.2f}" not in main_only:
                fail(f"main.tex missing per-scene {sc} {k} = {ps[sc][k]}")
        for ms in (ps["tum_fr2_xyz"]["tt_vgg_ms"],):
            if str(ms) not in main_only:
                fail(f"main.tex missing frame time {ms}")
        dom = nums["method_dominance"]
        if f"{dom['n']} of {dom['of']}" not in main_only:
            fail("main.tex missing dominance count")
        pr = nums["pareto"]
        for v in (pr["tum_fr2_xyz"]["100"], pr["tum_fr2_xyz"]["baseline"]["1"], pr["tum_fr2_xyz"]["baseline"]["4"], pr["tum_fr1_xyz"]["120"]):
            if f"{v[0]:.2f}" not in main_only:
                fail(f"main.tex missing Pareto value {v}")
        hv = nums.get("holdout_vgg", {})
        for sc, v in hv.items():
            if f"{v['holdout_delta']:.2f}" not in main_only:
                fail(f"main.tex missing held-out controller value {sc} {v['holdout_delta']}")
            if f"{round(100 * v['retained'])}" not in main_only:
                fail(f"main.tex missing retained percentage {sc} {v['retained']}")

    # (3) README + stale values
    rd = README.read_text()
    for key in ("oursPSNR", "gainBest", "headNoop"):
        if macros[key] not in rd:
            fail(f"README.md missing {key} = {macros[key]}")
    supp = SUPP_TEX.read_text()
    if "before the evaluator correction" not in supp:
        fail("supp.tex lacks the earlier-revision notice for Tabs. S4-S6")
    texts = {"main.tex": main_only, "README.md": rd}
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
