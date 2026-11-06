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
              r"\caption{\textbf{Replica factorial and map-proportional budget, 150 frames $\times$ 8 seeds per scene.} Same cells as Tab.~\ref{tab:factorial}; all runs use the current code revision and a $15\,\mathrm{ms}$ modeled budget unless stated. The proportional cell sets the optimization budget to $15\,\mathrm{ms}\times N_t/N_{\mathrm{ref}}$ (mean scale $3.7$--$3.9\times$) with the spawn allowance fixed at $15\,\mathrm{ms}$. Coverage and backlog throttling lower PSNR on Replica, whereas more optimization budget raises it.}",
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {}
    tab_replica(numbers)
    tab_dose(numbers)
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1)[:1500])


if __name__ == "__main__":
    main()
