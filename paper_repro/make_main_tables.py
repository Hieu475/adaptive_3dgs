#!/usr/bin/env python3
"""Generate the main-benchmark and factorial tables, their prose, and number macros.

Source of truth: results/rerun_paper1/ (commit tagged paper-rerun-1, see SOURCE.txt) plus
results/dilution_extension/analysis.json for the proportional-budget range. Writes to
docs/paper/iccv_official/tables/: tab_main.tex, tab_factorial.tex, par_core.tex, macros.tex.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RERUN = ROOT / "results/rerun_paper1"
MAIN = json.loads((RERUN / "main/final_confirmation_results.json").read_text())
FACT = json.loads((RERUN / "dilution_factorial/analysis.json").read_text())
EXT = json.loads((ROOT / "results/dilution_extension/analysis.json").read_text())
OUT = ROOT / "docs/paper/iccv_official/tables"
OUT.mkdir(parents=True, exist_ok=True)

S = MAIN["policy_stats"]
ours = S["ours"]["mean_psnr"]["mean"]
noop = S["no_op"]["mean_psnr"]["mean"]
full = S["full"]["mean_psnr"]["mean"]
budgeted = {p: S[p]["mean_psnr"]["mean"] for p in
            ("error_only", "error_influence", "error_influence_temporal", "rtg_slam_reimpl")}
best = max(budgeted.values())
cmp_ = {c["baseline"]: c for c in MAIN["paired_comparisons"]}
SC = {"a": "tum_fr2_xyz", "b": "tum_fr1_xyz"}


def f2(x):
    return f"{x:.2f}"


def sgn(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", "$-$") if False else f"{x:+.{nd}f}"


def con(sc, name):
    return FACT[sc]["contrasts"][name]


def cond(sc, name):
    return FACT[sc]["conditions"][name]


# ------------------------------------------------------------------ main table
def tab_main():
    rows = [("no\\_op", "no_op"), ("error\\_only", "error_only"), ("error\\_influence", "error_influence"),
            ("error\\_influence\\_temporal", "error_influence_temporal"), ("rtg\\_slam\\_reimpl", "rtg_slam_reimpl"),
            ("OURS", "ours"), ("FULL", "full")]
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"\textbf{Policy} & \textbf{Mean PSNR} & \textbf{Final PSNR} & \textbf{SSIM} & \textbf{FPS} & \textbf{Map} & \textbf{$\Delta$ vs OURS} \\",
         r"\midrule"]
    for label, p in rows:
        s = S[p]
        vals = [f2(s["mean_psnr"]["mean"]), f2(s["final_psnr"]["mean"]), f"{s['mean_ssim']['mean']:.4f}",
                f"{s['fps']['mean']:.1f}", f"{s['N_final']['mean']:,.0f}".replace(",", ",")]
        if p == "ours":
            row = r"\textbf{OURS} & " + " & ".join(r"\textbf{%s}" % v for v in vals) + r" & --- \\"
        else:
            c = cmp_[p]
            delta = ours - s["mean_psnr"]["mean"]
            tick = r" $\checkmark$" if c["p_holm_mean_psnr"] < 0.05 else ""
            row = (r"\texttt{%s} & " % label if p != "full" else r"\textit{FULL} & ") + " & ".join(vals) + f" & {delta:+.2f}{tick} \\\\"
            if p == "full":
                row = r"\textit{FULL} & " + " & ".join(r"\textit{%s}" % v for v in vals) + f" & {delta:+.2f}{tick} \\\\"
        L.append(row)
    holm = max(c["p_holm_mean_psnr"] for c in MAIN["paired_comparisons"])
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Main benchmark, TUM \texttt{fr2\_xyz}, 150 frames $\times$ 8 seeds.} All six controls differ significantly from OURS after Holm correction ($p_{\mathrm{Holm}} = %.4f$). FULL is $%.2f\,\mathrm{dB}$ higher. Headroom recovery is $%.1f\%%$ relative to no-op and $%.1f\%%$ relative to the strongest budgeted baseline, at a $%.1f\times$ smaller map than FULL. Source: commit tag \texttt{paper-rerun-1}.}"
          % (holm, full - ours, 100 * (ours - noop) / (full - noop), 100 * (ours - best) / (full - best),
             S["full"]["N_final"]["mean"] / S["ours"]["N_final"]["mean"]),
          r"\label{tab:main}", r"\end{table*}"]
    (OUT / "tab_main.tex").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------- factorial table
def tab_factorial():
    rows = [("Error, no throttle", "error_off"), ("Error, coverage", "error_coverage"),
            ("Error, coverage + backlog", "error_coverage_backlog"),
            ("Temporal$\\times$influence, no throttle", "temporal_off"),
            ("Temporal$\\times$influence, coverage", "temporal_coverage"),
            ("Temporal$\\times$influence, coverage + backlog", "temporal_coverage_backlog"), None,
            ("Error, optimization budget $\\times4$", "error_budget4"),
            ("Error, hard map cap 23k", "error_cap23k"), ("Error, hard map cap 39k", "error_cap39k")]
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"& \multicolumn{3}{c}{TUM \texttt{fr2\_xyz}} & \multicolumn{3}{c}{TUM \texttt{fr1\_xyz}} \\",
         r"\textbf{Selector / map control} & \textbf{PSNR} & \textbf{Depth L1} & \textbf{Map (k)} & \textbf{PSNR} & \textbf{Depth L1} & \textbf{Map (k)} \\",
         r"\midrule"]
    for r in rows:
        if r is None:
            L.append(r"\midrule")
            continue
        cells = []
        for sc in SC.values():
            c = cond(sc, r[1])
            cells += [f2(c["mean_psnr"]), f"{c['mean_depth_l1']:.4f}".lstrip("0"), f"{c['N_final'] / 1000:.1f}"]
        L.append(r[0] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Corrected-pose factorial and dilution controls, 150 frames $\times$ 8 seeds per scene.} All factorial cells use the same modeled $15\,\mathrm{ms}$ optimization budget. The $\times4$ control uses $60\,\mathrm{ms}$ for optimization while keeping the original $15\,\mathrm{ms}$ spawn allowance. Depth L1 is in meters. Backlog is enabled independently of warmup only in the last throttle level. Per-run traces and paired intervals are in \texttt{results/rerun\_paper1/dilution\_factorial/}.}",
          r"\label{tab:factorial}", r"\end{table*}"]
    (OUT / "tab_factorial.tex").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------- prose
def ci(c):
    return f"[{c['ci95_db'][0]:.2f},{c['ci95_db'][1]:.2f}]"


def par_core():
    a, b = SC["a"], SC["b"]
    ca, cb = con(a, "coverage on error"), con(b, "coverage on error")
    ta, tb = con(a, "coverage on temporal"), con(b, "coverage on temporal")
    sa, sb = con(a, "selector with coverage"), con(b, "selector with coverage")
    ia, ib = con(a, "selector x coverage interaction"), con(b, "selector x coverage interaction")
    bea, beb = con(a, "backlog on error"), con(b, "backlog on error")
    bta, btb = con(a, "backlog on temporal"), con(b, "backlog on temporal")
    x4a, x4b = con(a, "error budget x4"), con(b, "error budget x4")
    c23a, c23b = con(a, "error cap 23k"), con(b, "error cap 23k")
    c39a, c39b = con(a, "error cap 39k"), con(b, "error cap 39k")
    ratio = lambda sc, k: cond(sc, k)["mean_depth_l1"] / cond(sc, "error_off")["mean_depth_l1"]
    s = []
    s.append(r"\textbf{Main result (Tab.~\ref{tab:main}; Supp.\ Figs.~S4--S5).}")
    s.append(f"OURS reaches ${ours:.2f} \\pm {S['ours']['mean_psnr']['std']:.2f}\\,\\mathrm{{dB}}$: ${ours - best:+.2f}$ over the strongest budgeted baseline, "
             f"${cmp_['rtg_slam_reimpl']['delta_mean_psnr']:+.2f}$ over the RTG-SLAM-policy re-implementation, and ${cmp_['no_op']['delta_mean_psnr']:+.2f}$ over pass-through. "
             "These results evaluate the combined policy on this trajectory.\n")
    s.append(r"\textbf{Selector $\times$ throttle factorial (Tab.~\ref{tab:factorial}).}")
    s.append(f"With selector fixed to raw error, coverage adds ${ca['mean_db']:+.2f}$ ${ci(ca)}$ dB on \\texttt{{fr2\\_xyz}} and ${cb['mean_db']:+.2f}$ ${ci(cb)}$ dB on \\texttt{{fr1\\_xyz}}. "
             f"With temporal$\\times$influence fixed, its gains are ${ta['mean_db']:+.2f}$ ${ci(ta)}$ and ${tb['mean_db']:+.2f}$ ${ci(tb)}$ dB. "
             "Brackets are paired seed-bootstrap 95\\% CIs; all four coverage contrasts have two-sided paired Wilcoxon $p=.0078$. "
             f"Temporal selection itself adds ${sa['mean_db']:+.2f}$ dB with coverage on \\texttt{{fr2\\_xyz}} and ${sb['mean_db']:+.2f}$ dB on \\texttt{{fr1\\_xyz}}; "
             f"the interaction is negative (${ia['mean_db']:+.2f}$ and ${ib['mean_db']:+.2f}$ dB), so selector and coverage effects are not additive. "
             f"The independent backlog option adds ${bea['mean_db']:+.2f}$/${bta['mean_db']:+.2f}$ dB on \\texttt{{fr2\\_xyz}} and ${beb['mean_db']:+.2f}$/${btb['mean_db']:+.2f}$ dB on \\texttt{{fr1\\_xyz}} for error/temporal selection.\n")
    s.append(r"\textbf{Dilution controls and direct allocation.}")
    off_a, cov_a = cond(a, "error_off"), cond(a, "error_coverage")
    off_b, cov_b = cond(b, "error_off"), cond(b, "error_coverage")
    s.append(f"Holding the spawn allowance at its $15\\,\\mathrm{{ms}}$ value, giving raw-error selection $60\\,\\mathrm{{ms}}$ for optimization raises PSNR by ${x4a['mean_db']:+.2f}$ ${ci(x4a)}$ and ${x4b['mean_db']:+.2f}$ ${ci(x4b)}$ dB on \\texttt{{fr2\\_xyz}}/\\texttt{{fr1\\_xyz}}, "
             f"despite maps of ${cond(a, 'error_budget4')['N_final'] / 1000:.0f}\\mathrm{{k}}$/${cond(b, 'error_budget4')['N_final'] / 1000:.0f}\\mathrm{{k}}$. "
             f"Capping the error-only map at 23k adds ${c23a['mean_db']:+.2f}$ ${ci(c23a)}$ and ${c23b['mean_db']:+.2f}$ ${ci(c23b)}$ dB, but multiplies depth L1 by ${ratio(b, 'error_cap23k'):.1f}$ on \\texttt{{fr1\\_xyz}}. "
             f"A 39k cap, close to that scene's coverage map size, adds ${c39b['mean_db']:+.2f}$ ${ci(c39b)}$ dB there (and ${c39a['mean_db']:+.2f}$ ${ci(c39a)}$ dB on \\texttt{{fr2\\_xyz}}). "
             "Thus a static size limit does not reproduce coverage throttling's effect on \\texttt{fr1\\_xyz}. "
             f"For raw-error selection on \\texttt{{fr2\\_xyz}}, coverage reduces the share of modeled update budget spent on Gaussians younger than ten frames from ${100 * off_a['young_budget_share']:.1f}\\%$ to ${100 * cov_a['young_budget_share']:.1f}\\%$, "
             f"while the per-frame selection rate among mature Gaussians rises from ${100 * off_a['old_selection_rate']:.1f}\\%$ to ${100 * cov_a['old_selection_rate']:.1f}\\%$. "
             f"The corresponding mature selection rate on \\texttt{{fr1\\_xyz}} rises from ${100 * off_b['old_selection_rate']:.1f}\\%$ to ${100 * cov_b['old_selection_rate']:.1f}\\%$. "
             "Allocation shifts and the budget intervention support compute dilution under the tested protocol, but coverage throttling also changes the optimization trajectory (its spawn cap carries most of the effect; see the decomposition below). "
             "Map size alone is not identified as the unique cause.\n")
    (OUT / "par_core.tex").write_text("\n".join(s))


# ------------------------------------------------------------------- macros
def macros():
    props = [EXT[sc]["contrasts"]["error budget proportional to map"]["mean_db"] for sc in EXT
             if "error budget proportional to map" in EXT[sc]["contrasts"]]
    a, b = SC["a"], SC["b"]
    off_a, cov_a = cond(a, "error_off"), cond(a, "error_coverage")
    grow = [S[p]["N_final"]["mean"] for p in ("error_only", "error_influence", "error_influence_temporal", "rtg_slam_reimpl")]
    M = {"oursPSNR": f2(ours), "oursStd": f2(S["ours"]["mean_psnr"]["std"]),
         "gainBest": f2(ours - best), "gainRTG": f2(cmp_["rtg_slam_reimpl"]["delta_mean_psnr"]),
         "gainErr": f2(cmp_["error_only"]["delta_mean_psnr"]), "gainNoop": f2(cmp_["no_op"]["delta_mean_psnr"]),
         "gapFull": f2(full - ours), "headNoop": f"{100 * (ours - noop) / (full - noop):.1f}",
         "headBest": f"{100 * (ours - best) / (full - best):.1f}",
         "mapRatio": f"{S['full']['N_final']['mean'] / S['ours']['N_final']['mean']:.1f}",
         "covErrA": f2(con(a, "coverage on error")["mean_db"]), "covErrB": f2(con(b, "coverage on error")["mean_db"]),
         "budgA": f2(con(a, "error budget x4")["mean_db"]), "budgB": f2(con(b, "error budget x4")["mean_db"]),
         "matOffA": f2(off_a["old_selection_rate"]), "matOnA": f2(cov_a["old_selection_rate"]),
         "propMin": f2(min(props)), "propMax": f2(max(props)),
         "growMin": f"{min(grow) / 1000:.0f}", "growMax": f"{max(grow) / 1000:.0f}",
         "errPSNR": f2(budgeted["error_only"]), "oursMap": f"{S['ours']['N_final']['mean'] / 1000:.0f}", "errMap": f"{S['error_only']['N_final']['mean'] / 1000:.0f}"}
    (OUT / "macros.tex").write_text("".join("\\newcommand{\\%s}{%s}\n" % (k, v) for k, v in M.items()))
    (OUT / "macros.json").write_text(json.dumps(M, indent=1))


if __name__ == "__main__":
    tab_main()
    tab_factorial()
    par_core()
    macros()
    print((OUT / "macros.json").read_text())
