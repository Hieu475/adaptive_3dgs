#!/usr/bin/env python3
"""Generate LaTeX tables for the dilution-extension experiments from frozen JSON.

Writes docs/paper/iccv_official/tables/{tab_replica,tab_dose}.tex and tables/numbers.json
(the numbers cited in prose, so check_numbers.py can verify them).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_final_confirmation import bootstrap_ci_95  # noqa: E402

OUT = ROOT / "docs/paper/iccv_official/tables"
EXT = ROOT / "results/dilution_extension/analysis.json"
DOSE_TUM = ROOT / "results/dilution_dose"
DOSE_REP = {"replica_office0": ROOT / "results/dilution_dose_replica_cap/office0_cap60k",
            "replica_office1": ROOT / "results/dilution_dose_replica_cap/office1_cap40k"}
LABEL = {"tum_fr2_xyz": r"\texttt{fr2\_xyz}", "tum_fr1_xyz": r"\texttt{fr1\_xyz}",
         "replica_office0": r"\texttt{office0}", "replica_office1": r"\texttt{office1}"}
CELLS = [f"{r}_x{m}" for r in ("free", "cap") for m in (1, 2, 4)]


def tab_replica(numbers):
    ext = json.loads(EXT.read_text())
    rows = [("Error, no throttle", "error_off"), ("Error, coverage", "error_coverage"),
            ("Error, coverage + backlog", "error_coverage_backlog"),
            ("Temporal$\\times$influence, no throttle", "temporal_off"),
            ("Temporal$\\times$influence, coverage", "temporal_coverage"),
            ("Temporal$\\times$influence, coverage + backlog", "temporal_coverage_backlog"),
            ("Error, optimization budget $\\times4$", "error_budget4"),
            ("Error, budget $\\propto$ map size", "error_budget_prop")]
    sc = ["replica_office0", "replica_office1"]
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{3}{c}{Replica \texttt{office0}} & \multicolumn{3}{c}{Replica \texttt{office1}} \\",
             r"\textbf{Condition} & \textbf{PSNR} & \textbf{Depth L1} & \textbf{Map (k)} & \textbf{PSNR} & \textbf{Depth L1} & \textbf{Map (k)} \\",
             r"\midrule"]
    for label, key in rows:
        cells = []
        for s in sc:
            c = ext[s]["conditions"][key]
            cells += [f"{c['mean_psnr']:.2f}", f"{c['mean_depth_l1']:.4f}".lstrip("0"), f"{c['N_final'] / 1000:.1f}"]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
        if key in ("error_coverage_backlog", "temporal_coverage_backlog"):
            lines.append(r"\midrule" if key == "temporal_coverage_backlog" else "")
    lines = [ln for ln in lines if ln != ""]
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Replica factorial and map-proportional budget, 99 (\texttt{office0}) and 119 (\texttt{office1}) frames $\times$ 8 seeds per scene.} Same cells as Tab.~\ref{tab:factorial}; all runs use the current code revision and a $15\,\mathrm{ms}$ modeled budget unless stated. The proportional cell sets the optimization budget to $15\,\mathrm{ms}\times N_t/N_{\mathrm{ref}}$ (mean scale $3.7$--$3.9\times$) with the spawn allowance fixed at $15\,\mathrm{ms}$. Coverage and backlog throttling lower PSNR on Replica, whereas more optimization budget raises it.}",
              r"\label{tab:replica}", r"\end{table*}"]
    for s in ("tum_fr2_xyz", "tum_fr1_xyz") + tuple(sc):
        cs = ext[s]["contrasts"]
        numbers[s] = {k: round(v["mean_db"], 2) for k, v in cs.items()}
        numbers[s]["error_off_psnr"] = round(ext[s]["conditions"]["error_off"]["mean_psnr"], 2)
    (OUT / "tab_replica.tex").write_text("\n".join(lines) + "\n")


def load_dose(d):
    runs = {}
    for f in d.glob("*_seed*.json"):
        r = json.loads(f.read_text())
        runs[(r["condition"], r["seed"])] = r["summary"]
    seeds = [s for s in range(42, 50) if all((c, s) in runs for c in CELLS)]
    return runs, seeds


def tab_dose(numbers):
    sources = {"tum_fr2_xyz": DOSE_TUM / "tum_fr2_xyz", "tum_fr1_xyz": DOSE_TUM / "tum_fr1_xyz"}
    caps = {"tum_fr2_xyz": 23, "tum_fr1_xyz": 23}
    for s, d in DOSE_REP.items():
        if (d / s).exists():
            sources[s] = d / s
            caps[s] = 60 if "office0" in s else 40
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lccccccccc}", r"\toprule",
             r"& \multicolumn{3}{c}{Free map growth} & \multicolumn{3}{c}{Hard map cap} & & \\",
             r"\textbf{Scene (cap)} & $\times1$ & $\times2$ & $\times4$ & $\times1$ & $\times2$ & $\times4$ & "
             r"$\rho_{\mathrm{all}}$ & $\rho_{\mathrm{free}}$ \\", r"\midrule"]
    for s, d in sources.items():
        runs, seeds = load_dose(d)
        if len(seeds) < 3:
            continue
        cells = []
        for c in CELLS:
            ps = np.mean([runs[(c, sd)]["mean_psnr"] for sd in seeds])
            cv = np.mean([runs[(c, sd)]["mature_update_coverage"] for sd in seeds])
            cells.append(f"{ps:.2f} ({cv:.2f})")
        r_all = np.nanmean([spearmanr([runs[(c, sd)]["mature_update_coverage"] for c in CELLS],
                                      [runs[(c, sd)]["mean_psnr"] for c in CELLS])[0] for sd in seeds])
        r_free = np.nanmean([spearmanr([runs[(c, sd)]["mature_update_coverage"] for c in CELLS[:3]],
                                       [runs[(c, sd)]["mean_psnr"] for c in CELLS[:3]])[0] for sd in seeds])
        lines.append(f"{LABEL[s]} ({caps[s]}k) & " + " & ".join(cells) + f" & {r_all:+.2f} & {r_free:+.2f} " + r"\\")
        numbers.setdefault("dose", {})[s] = {"cells": cells, "rho_all": round(float(r_all), 2),
                                             "rho_free": round(float(r_free), 2), "n_seeds": len(seeds)}
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Dose-response in optimization budget, 8 seeds.} Raw-error selection, no throttling; entries are mean PSNR (dB) with the mature-update coverage in parentheses, for a budget of $15\times\{1,2,4\}\,\mathrm{ms}$ with the spawn allowance fixed at $15\,\mathrm{ms}$, with free growth or a hard Gaussian cap. $\rho$ is the per-seed Spearman correlation between PSNR and mature-update coverage over the cells (all six, or the three free-growth cells), averaged over seeds. Once coverage reaches $1.00$ extra budget is unused and PSNR saturates. On Replica the caps remove needed capacity (PSNR $6$--$11$\,dB lower at equal budget), so the cross-regime correlation does not hold there.}",
              r"\label{tab:dose}", r"\end{table*}"]
    (OUT / "tab_dose.tex").write_text("\n".join(lines) + "\n")


def tab_scenes(numbers):
    path = ROOT / "results/dilution_scenes/analysis.json"
    if not path.exists():
        return
    a = json.loads(path.read_text())
    order = ["tum_fr2_xyz", "tum_fr1_xyz", "tum_fr1_desk", "tum_fr1_room", "tum_fr1_rpy", "tum_fr3_sitting_static",
             "tum_fr3_structure_texture_near", "replica_office0", "replica_office1", "replica_office2",
             "replica_office3", "replica_office4", "replica_room0", "replica_room1", "replica_room2"]
    nice = lambda s: s.replace("tum_", "").replace("replica_", "").replace("_", "\\_")
    rows = []
    for sc in order:
        v = a["scenes"][sc]
        rows.append((f"\\texttt{{{nice(sc)}}}" + (" (T)" if sc.startswith("tum") else " (R)"),
                     f"{v['prop']['mean_db']:+.2f}".replace("-", "$-$"), f"{v['cov']['mean_db']:+.2f}".replace("-", "$-$"),
                     f"{v['off_map'] / 1000:.0f}"))
    half = (len(rows) + 1) // 2
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lccclccc}", r"\toprule",
         r"\textbf{Scene} & \textbf{Prop.} & \textbf{Cov.} & \textbf{Map (k)} & \textbf{Scene} & \textbf{Prop.} & \textbf{Cov.} & \textbf{Map (k)} \\",
         r"\midrule"]
    for i in range(half):
        left = " & ".join(rows[i])
        right = " & ".join(rows[i + half]) if i + half < len(rows) else " & & & "
        L.append(left + " & " + right + r" \\")
    sm, cm = a["summary"]["prop"], a["summary"]["cov"]
    rho = a["summary"]["map_size_vs_prop_gain"]
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Scene-level change in mean PSNR (dB) relative to unthrottled raw-error selection, 15 scenes.} Prop.: optimization budget proportional to map size; Cov.: coverage throttling; Map: final size of the unthrottled map. T: TUM, R: Replica. \texttt{fr2\_xyz} and \texttt{fr1\_xyz} use 150 frames, \texttt{office0} and \texttt{office1} use 99 and 119 frames, all with eight seeds; the others use 99 frames and three seeds. Proportional budget is positive on %d of %d scenes (median $%+.2f$, exact sign test $p<10^{-4}$); coverage throttling on %d of %d (median $%+.2f$, $p=%.2f$). Spearman correlation between map size and proportional-budget gain: $%+.2f$ ($p=%.3f$, exploratory).}"
          % (sm["n_positive"], sm["n_scenes"], sm["median_db"], cm["n_positive"], cm["n_scenes"],
             cm["median_db"], cm["sign_test_p"], rho["spearman"], rho["p"]),
          r"\label{tab:scenelevel}", r"\end{table*}"]
    (OUT / "tab_scenelevel.tex").write_text("\n".join(L) + "\n")
    numbers["scene_level"] = {"prop_pos": sm["n_positive"], "n": sm["n_scenes"], "prop_median": round(sm["median_db"], 2),
                              "prop_min": round(sm["min_db"], 2), "prop_max": round(sm["max_db"], 2),
                              "cov_pos": cm["n_positive"], "cov_median": round(cm["median_db"], 2),
                              "cov_min": round(cm["min_db"], 2), "cov_max": round(cm["max_db"], 2),
                              "cov_p": round(cm["sign_test_p"], 2), "rho": round(rho["spearman"], 2),
                              "rho_p": round(rho["p"], 3)}


def tab_long(numbers):
    path = ROOT / "results/dilution_long/analysis.json"
    if not path.exists():
        return
    a = json.loads(path.read_text())
    order = ["tum_fr2_xyz", "tum_fr1_xyz", "tum_fr1_desk"]
    L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llccc}", r"\toprule",
         r"\textbf{Scene} & \textbf{Frames} & \textbf{Base} & \textbf{Cov.} & \textbf{Prop.} \\", r"\midrule"]
    fmt = lambda x: f"{x:+.2f}".replace("-", "$-$")
    for sc in order:
        for i, (w, v) in enumerate(a[sc]["windows"].items()):
            name = f"\\texttt{{{sc.replace('tum_', '').replace('_', chr(92) + '_')}}}" if i == 0 else ""
            L.append(f"{name} & {w.replace('-', '--')} & {v['base']:.2f} & {fmt(v['cov'])} & {fmt(v['prop'])} \\\\")
        if sc != order[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Long horizon (500 frames, two seeds).} Mean PSNR of the unthrottled raw-error baseline (Base) and the change from coverage throttling (Cov.) and the map-proportional budget (Prop., mean scale $5.4$--$6.7\times$) in three frame windows. The baseline degrades with horizon and the gains grow; coverage throttling is not monotone. Replica sequences are too short for this test.}",
          r"\label{tab:long}", r"\end{table}"]
    (OUT / "tab_long.tex").write_text("\n".join(L) + "\n")
    numbers["long"] = {sc: {w: {k: round(x, 2) for k, x in v.items() if k in ("base", "cov", "prop")}
                            for w, v in a[sc]["windows"].items()} for sc in order}


def tab_ablation(numbers):
    path = ROOT / "results/throttle_ablation/analysis.json"
    if not path.exists():
        return
    a = json.loads(path.read_text())
    scenes = ["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0", "replica_office1"]
    rows = [("Coverage throttling (full)", "error_coverage"), ("\\quad spawn cap only", "cov_spawn_only"),
            ("\\quad threshold only", "cov_thresh_only"), ("Spawn cap $\\times0.2$, ungated", "spawn_cap_always20"),
            ("Spawn cap $\\times0.5$, ungated", "spawn_cap_always50"), None,
            ("Budget $\\propto$ map size", "error_budget_prop"), ("Throttling + budget $\\propto$ map", "cov_plus_prop")]
    fmt = lambda x: f"{x:+.2f}".replace("-", "$-$")
    L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"\textbf{Condition} & \texttt{fr2\_xyz} & \texttt{fr1\_xyz} & \texttt{office0} & \texttt{office1} \\", r"\midrule"]
    nums = {}
    for r in rows:
        if r is None:
            L.append(r"\midrule")
            continue
        vals = [a[sc]["contrasts"][r[1]]["mean_db"] for sc in scenes]
        nums[r[1]] = [round(v, 2) for v in vals]
        L.append(r[0] + " & " + " & ".join(fmt(v) for v in vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Decomposition of coverage throttling, change in mean PSNR (dB) relative to unthrottled raw-error selection, four seeds.} Spawn-cap-only fixes the densification-threshold multiplier at 1; threshold-only disables the spawn cap; the ungated caps apply whenever the view is settled. Budget $\propto$ map size uses a mean $3.7$--$6\times$ larger optimization budget than the other rows.}",
          r"\label{tab:ablation}", r"\end{table}"]
    (OUT / "tab_ablation.tex").write_text("\n".join(L) + "\n")
    numbers["ablation"] = nums


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {}
    tab_replica(numbers)
    tab_dose(numbers)
    tab_scenes(numbers)
    tab_long(numbers)
    tab_ablation(numbers)
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1)[:1500])


if __name__ == "__main__":
    main()
