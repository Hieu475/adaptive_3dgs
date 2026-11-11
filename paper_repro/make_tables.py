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
from experiments.run.run_final_confirmation import bootstrap_ci_95  # noqa: E402

OUT = ROOT / "docs/paper/iccv_official/tables"
EXT = ROOT / "results/dilution/dilution_extension/analysis.json"
DOSE_TUM = ROOT / "results/dilution/dilution_dose"
DOSE_REP = {"replica_office0": ROOT / "results/dilution/dilution_dose_replica_cap/office0_cap60k",
            "replica_office1": ROOT / "results/dilution/dilution_dose_replica_cap/office1_cap40k"}
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
    path = ROOT / "results/dilution/dilution_scenes/analysis.json"
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
    path = ROOT / "results/dilution/dilution_long/analysis.json"
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
    path = ROOT / "results/dilution/throttle_ablation/analysis.json"
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


def tab_holdout(numbers):
    path = ROOT / "results/holdout/holdout_extension/analysis.json"
    if not path.exists():
        return
    a = json.loads(path.read_text())
    scenes = ["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0", "replica_office1"]
    rows = [("Coverage throttling", "error_coverage"), ("Spawn cap $\\times0.2$", "spawn_cap_always20"),
            ("Budget $\\propto$ map", "error_budget_prop")]
    f = lambda x: f"{x:+.2f}".replace("-", "$-$")
    L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{1pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"\textbf{Mapped / held-out} & \texttt{fr2\_xyz} & \texttt{fr1\_xyz} & \texttt{office0} & \texttt{office1} \\", r"\midrule"]
    nums = {}
    for label, c in rows:
        cells = [f"{f(a[sc][c]['train_delta'])}/{f(a[sc][c]['holdout_delta'])}" for sc in scenes]
        nums[c] = [[round(a[sc][c]["train_delta"], 2), round(a[sc][c]["holdout_delta"], 2)] for sc in scenes]
        L.append(label + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    gaps = [f"{a[sc]['base_gap']:.2f}" for sc in scenes]
    pgaps = [f"{a[sc]['error_budget_prop']['gap']:.2f}" for sc in scenes]
    L.append("Gap, unthrottled & " + " & ".join(gaps) + r" \\")
    L.append("Gap, budget $\\propto$ map & " + " & ".join(pgaps) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Mapped versus held-out views, change in mean PSNR (dB) relative to unthrottled raw-error selection, four seeds.} Every eighth frame is rendered but never mapped (a same-trajectory proxy, not an independent trajectory). The last rows give the train-minus-held-out PSNR gap.}",
          r"\label{tab:holdout}", r"\end{table}"]
    (OUT / "tab_holdout.tex").write_text("\n".join(L) + "\n")
    numbers["holdout"] = {"deltas": nums, "gap_base": [round(a[sc]["base_gap"], 2) for sc in scenes],
                          "gap_prop": [round(a[sc]["error_budget_prop"]["gap"], 2) for sc in scenes]}


def tab_compute(numbers):
    path = ROOT / "results/dilution/compute_matched/analysis.json"
    if not path.exists():
        return
    a = json.loads(path.read_text())
    scenes = ["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0", "replica_office1"]
    cells = ["free_x1", "free_x2", "free_x4", "spawn_x1", "spawn_x2", "spawn_x4"]
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"& \multicolumn{3}{c}{Free map growth} & \multicolumn{3}{c}{Spawn cap $\times0.2$} \\",
         r"\textbf{Scene} & $\times1$ & $\times2$ & $\times4$ & $\times1$ & $\times2$ & $\times4$ \\", r"\midrule"]
    nums = {}
    for sc in scenes:
        row = []
        nums[sc] = {}
        for c in cells:
            v = a[sc]["cells"][c]
            row.append(f"{v['mean_psnr']:.2f} ({v['mean_frame_ms']:.0f})")
            nums[sc][c] = [round(v["mean_psnr"], 2), round(v["mean_frame_ms"])]
        name = sc.replace("tum_", "").replace("replica_", "").replace("_", "\\_")
        L.append(f"\\texttt{{{name}}} & " + " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Quality at measured cost.} Mean PSNR (dB) with the mean end-to-end frame time in ms in parentheses, for a modeled optimization budget of $15\times\{1,2,4\}\,\mathrm{ms}$ (spawn allowance fixed) with free growth or a $\times0.2$ spawn cap; raw-error selection, four seeds. Timings come from one laptop GPU and indicate relative costs only.}",
          r"\label{tab:compute}", r"\end{table*}"]
    (OUT / "tab_compute.tex").write_text("\n".join(L) + "\n")
    numbers["compute"] = nums
    numbers["compute_matched"] = {sc: {k: (None if v["gain_vs_free_frame"] is None else round(v["gain_vs_free_frame"], 2))
                                       for k, v in a[sc]["matched"].items()} for sc in scenes}


def tab_closedloop(numbers):
    path = ROOT / "results/tracking/closed_loop/analysis.json"
    if not path.exists():
        return
    a = json.loads(path.read_text())
    f = lambda x: f"{x:+.2f}".replace("-", "$-$")
    scenes = ["tum_fr2_xyz", "tum_fr1_xyz", "tum_fr1_rpy", "tum_fr3_sitting_static", "tum_fr1_room"]
    L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"\textbf{Scene} & \textbf{Base ATE} & \textbf{TT+VGG $\Delta$PSNR} & \textbf{TT+VGG $\Delta$ATE} & \textbf{Prop. $\Delta$PSNR} \\",
         r"\midrule"]
    nums = {}
    for sc in scenes:
        k, kp = a[f"{sc}/tt_vgg"], a[f"{sc}/error_budget_prop"]
        div = k["base_ate_cm"] >= 50
        name = "\\texttt{" + sc.replace("tum_", "").replace("_", "\\_") + "}" + ("$^\\dagger$" if div else "")
        L.append(f"{name} & {k['base_ate_cm']:.1f} cm & {f(k['dpsnr'])} & {f(k['date_cm'])} cm & {f(kp['dpsnr'])} \\\\")
        nums[sc] = {"base_ate_cm": round(k["base_ate_cm"], 2), "tt_vgg_dpsnr": round(k["dpsnr"], 2), "tt_vgg_date_cm": round(k["date_cm"], 2)}
    sm = a["_summary"]["tt_vgg"]
    nums["summary"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in sm.items()}
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Closed-loop mapping with map-based tracking, three seeds.} Poses come from aligning each frame to the current map (Supp.\ text); the same controllers then map with the estimated poses. $\Delta$PSNR (dB) is measured at the tracked pose, $\Delta$ATE is TT+VGG minus the unthrottled baseline. $^\dagger$Tracking diverged for every condition and the scene is excluded from the summary: over the four converged scenes TT+VGG changes PSNR by %s dB on average and ATE by a median of %s cm (largest $|\Delta|$ %.2f cm).}" % (f(sm["mean_dpsnr"]), f(sm["median_date_cm"]), sm["max_abs_date_cm"]),
          r"\label{tab:closedloop}", r"\end{table}"]
    (OUT / "tab_closedloop.tex").write_text("\n".join(L) + "\n")
    numbers["closedloop"] = nums


def tab_countcap(numbers):
    path = ROOT / "results/dilution/count_cap/analysis.json"
    if not path.exists():
        return
    a = json.loads(path.read_text())
    sm = a["_summary"]
    f = lambda x: f"{x:+.2f}".replace("-", "$-$")
    L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"\textbf{Scene} & \textbf{TT+VGG} & \textbf{TT + count cap} & \textbf{Count cap} \\", r"\midrule"]
    for sc, v in a.items():
        if sc.startswith("_"):
            continue
        L.append("\\texttt{" + sc.replace("_", "\\_") + "} & " + f"{f(v['tt_vgg'])} & {f(v['tt_countcap'])} & {f(v['countcap'])} \\\\")
    L += [r"\midrule", f"mean & {f(sm['mean_tt_vgg'])} & {f(sm['mean_tt_countcap'])} & {f(sm['mean_countcap'])} \\\\",
          r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Value-gated growth versus a hard map-size budget of the same size.} PSNR change (dB) against the unthrottled baseline, two seeds per scene. The cap on the number of Gaussians equals TT+VGG's final map size on that scene and seed, so the cap is informed by the answer; ``TT + count cap'' also uses the time-targeted budget.}",
          r"\label{tab:countcap}", r"\end{table}"]
    (OUT / "tab_countcap.tex").write_text("\n".join(L) + "\n")
    numbers["countcap"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in sm.items()}


def tab_tracked(numbers):
    path = ROOT / "results/tracking/tracked_poses/analysis.json"
    if not path.exists():
        return
    a = json.loads(path.read_text())
    scenes = ["tum_fr2_xyz", "tum_fr1_xyz"]
    f = lambda x: f"{x:+.2f}".replace("-", "$-$")
    L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{llcc}", r"\toprule",
         r"\textbf{Scene} & \textbf{Condition} & \textbf{Tracked pose} & \textbf{GT pose} \\", r"\midrule"]
    nums = {}
    for sc in scenes:
        b = a[sc]["base"]
        nums[sc] = {"base_tracked": round(b["psnr_tracked"], 2), "base_gt": round(b["psnr_at_gt_pose"], 2), "ate_cm": round(b["ate_cm"], 2)}
        name = "\\texttt{" + sc.replace("tum_", "").replace("_", "\\_") + "} (ATE %.2f cm)" % b["ate_cm"]
        L.append(f"{name} & unthrottled & {b['psnr_tracked']:.2f} & {b['psnr_at_gt_pose']:.2f} \\\\")
        for lab, c in (("coverage", "error_coverage"), ("spawn cap $\\times0.2$", "spawn_cap_always20"), ("budget $\\propto$ map", "error_budget_prop")):
            L.append(f" & {lab} & {f(a[sc][c]['tracked_delta'])} & {f(a[sc][c]['gt_delta'])} \\\\")
            nums[sc][c] = [round(a[sc][c]["tracked_delta"], 2), round(a[sc][c]["gt_delta"], 2)]
        if sc != scenes[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Mapping with estimated poses, three seeds.} PSNR (dB) of the unthrottled baseline and the change for each condition, evaluated at the tracker's pose and when the same map renders from the ground-truth pose. The tracker is a frame-to-frame RGB-D Gauss-Newton odometer that does not use the map.}",
          r"\label{tab:tracked}", r"\end{table}"]
    (OUT / "tab_tracked.tex").write_text("\n".join(L) + "\n")
    numbers["tracked"] = nums


def tab_method(numbers):
    t = ROOT / "results/controllers/vgg_test/analysis_tt.json"
    f = ROOT / "results/controllers/vgg_fresh/analysis_tt.json"
    if not (t.exists() and f.exists()):
        return
    a, b = json.loads(t.read_text()), json.loads(f.read_text())
    rows = [("Fixed spawn cap $\\times0.2$", "spawn_cap_always20"), ("Budget $\\propto$ map (+%d ms)", "error_budget_prop"),
            ("VGG alone", "vgg"), ("TT alone", "tt"), ("TT + fixed cap", "tt_cap"), ("\\textbf{TT + VGG (ours)}", "tt_vgg")]
    fm = lambda x: f"{x:+.2f}".replace("-", "$-$")
    L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{lccccccc}", r"\toprule",
         r"& \multicolumn{4}{c}{11 held-out scenes} & \multicolumn{3}{c}{5 fresh windows} \\",
         r"\textbf{Method} & $>0$ & worst & mean & $\Delta$ms & $>0$ & mean & $\Delta$ms \\", r"\midrule"]
    nums = {}
    for label, key in rows:
        sa = a["summary"].get(key)
        sb = b["summary"].get(key)
        if label.startswith("Budget"):
            label = label % round(sa["mean_time_delta_ms"])
        ca = f"{sa['n_positive']}/{sa['n_scenes']} & {fm(sa['min'])} & {fm(sa['mean'])} & {sa['mean_time_delta_ms']:+.0f}" if sa else " & & & "
        cb = f"{sb['n_positive']}/{sb['n_scenes']} & {fm(sb['mean'])} & {sb['mean_time_delta_ms']:+.0f}" if sb else " & & "
        L.append(f"{label} & {ca} & {cb} \\\\")
        nums[key] = {"test": None if not sa else {k: round(sa[k], 2) for k in ("min", "median", "mean", "max", "mean_time_delta_ms")} | {"pos": sa["n_positive"], "n": sa["n_scenes"]},
                     "fresh": None if not sb else {k: round(sb[k], 2) for k in ("min", "median", "mean", "max", "mean_time_delta_ms")} | {"pos": sb["n_positive"], "n": sb["n_scenes"]}}
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Controllers versus the unthrottled baseline (modeled $15\,\mathrm{ms}$ budget), change in mean PSNR (dB).} Left: 11 scenes not used to set any constant (hyperparameters were frozen on four development scenes); right: five fresh 150-frame windows of long TUM sequences. $>0$: scenes improved; worst/mean over scenes; $\Delta$ms: mean change in end-to-end frame time. TT: time-targeted budget ($T=100$ ms); VGG: value-gated growth. Seed-paired, three or four seeds per scene.}",
          r"\label{tab:method}", r"\end{table}"]
    (OUT / "tab_method.tex").write_text("\n".join(L) + "\n")
    numbers["method"] = nums
    numbers["method_vgg_over_tt"] = {"test": {k: round(v, 2) if isinstance(v, float) else v for k, v in a["summary"]["vgg_over_tt"].items()},
                                     "fresh": {k: round(v, 2) if isinstance(v, float) else v for k, v in b["summary"]["vgg_over_tt"].items()}}
    numbers["method_dominance"] = a["summary"]["tt_vgg_better_and_within_15pct_time"]
    sc = a["scenes"]
    numbers["method_scene"] = {k: {"tt": round(v["methods"]["tt"]["gain"], 2), "tt_vgg": round(v["methods"]["tt_vgg"]["gain"], 2),
                                   "tt_cap": round(v["methods"]["tt_cap"]["gain"], 2), "cap": round(v["methods"]["spawn_cap_always20"]["gain"], 2),
                                   "tt_vgg_ms": round(v["methods"]["tt_vgg"]["ms"]), "base_ms": round(v["base_ms"])} for k, v in sc.items()}
    # per-scene supplementary table
    def scene_rows(an, keys):
        rows = []
        for name, v in an["scenes"].items():
            nm = "\\texttt{" + name.replace("tum_", "").replace("replica_", "").replace("_", "\\_") + "}"
            rows.append(f"{nm} & {v['n']} & {v['base_psnr']:.2f} ({v['base_ms']:.0f}) & " + " & ".join(
                f"{fm(v['methods'][k]['gain'])} ({v['methods'][k]['ms']:.0f})" for k in keys) + r" \\")
        return rows
    keys = ["spawn_cap_always20", "tt", "tt_cap", "tt_vgg"]
    for fname, an, cap in (("tab_method_scenes", a, "11 held-out scenes"), ("tab_method_fresh", b, "five fresh frame windows")):
        L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{1.5pt}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"\textbf{Scene} & $n$ & \textbf{Base} & \textbf{Cap} & \textbf{TT} & \textbf{TT+cap} & \textbf{TT+VGG} \\".replace("& \\textbf{TT+cap}", "& \\textbf{TT+cap}"),
             r"\midrule"] if False else [r"\begin{table}[tbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{1.5pt}",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"\textbf{Scene} & $n$ & \textbf{Base} & \textbf{Cap} & \textbf{TT} & \textbf{TT+cap} & \textbf{TT+VGG} \\", r"\midrule"]
        L = [l.replace("{llcccc}", "{lccccccc}".replace("ccccccc","cccccc")) for l in L]
        L = [l.replace("\\textbf{Scene} & $n$ & \\textbf{Base} & \\textbf{Cap} & \\textbf{TT} & \\textbf{TT+cap} & \\textbf{TT+VGG}", "\\textbf{Scene} & $n$ & \\textbf{Base} & \\textbf{Cap} & \\textbf{TT} & \\textbf{TT+cap} & \\textbf{TT+VGG}") for l in L]
        L += scene_rows(an, keys)
        L += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Per-scene results, %s.} Base: unthrottled PSNR (ms); other columns: change in PSNR (dB) with the mean frame time in ms in parentheses.}" % cap,
              r"\label{tab:%s}" % fname.replace("tab_", ""), r"\end{table}"]
        (OUT / (fname + ".tex")).write_text("\n".join(L) + "\n")
    # Pareto table
    pj = ROOT / "results/controllers/tt_pareto/analysis.json"
    if pj.exists():
        pa = json.loads(pj.read_text())
        L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{lccccc}", r"\toprule",
             r"\textbf{Scene} & \textbf{Baseline $\times1/\times2/\times4$} & $T=70$ & $T=85$ & $T=100$ & $T=120$ & $T=140$ \\".replace("{lccccc}", "{lcccccc}"), r"\midrule"]
        L[3] = r"\begin{tabular}{lcccccc}"
        pn = {}
        for sc, v in pa.items():
            bl = " / ".join(f"{v['baseline'][m]['psnr']:.1f}@{v['baseline'][m]['ms']:.0f}" for m in ("1", "2", "4"))
            cells = [f"{v['points'][T]['psnr']:.2f}@{v['points'][T]['ms']:.0f}" for T in ("70", "85", "100", "120", "140")]
            nm = "\\texttt{" + sc.replace("tum_", "").replace("replica_", "").replace("_", "\\_") + "}"
            L.append(f"{nm} & {bl} & " + " & ".join(cells) + r" \\")
            pn[sc] = {T: [round(v["points"][T]["psnr"], 2), round(v["points"][T]["ms"])] for T in v["points"]}
            pn[sc]["baseline"] = {m: [round(x["psnr"], 2), round(x["ms"])] for m, x in v["baseline"].items()}
        L += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{TT+VGG at five time targets versus fixed budgets.} Mean PSNR (dB) @ mean end-to-end frame time (ms), four seeds. The baseline column gives free growth at modeled budgets $\times1$, $\times2$, $\times4$.}",
              r"\label{tab:pareto}", r"\end{table}"]
        (OUT / "tab_pareto.tex").write_text("\n".join(L) + "\n")
        numbers["pareto"] = pn
    hv = ROOT / "results/holdout/holdout_vgg/analysis.json"
    if hv.exists():
        h = json.loads(hv.read_text())
        numbers["holdout_vgg"] = {sc: {k: round(v[k], 2) if v[k] is not None else None for k in ("train_delta", "holdout_delta", "retained")}
                                  for sc, d in h.items() for v in [d["tt_vgg"]]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {}
    tab_replica(numbers)
    tab_dose(numbers)
    tab_scenes(numbers)
    tab_long(numbers)
    tab_ablation(numbers)
    tab_holdout(numbers)
    tab_compute(numbers)
    tab_tracked(numbers)
    tab_closedloop(numbers)
    tab_countcap(numbers)
    tab_method(numbers)
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1)[:1500])


if __name__ == "__main__":
    main()
