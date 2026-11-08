#!/usr/bin/env python3
"""Tables, figure and number macros for the steps-regime paper, from the frozen bundles.

Inputs:  results/engine/analysis.json (+ per-run JSON), results/engine_ablation, results/holdout_engine2,
         results/core_speed/{before,after}.json
Outputs: docs/paper/iccv_official/tables/{tab_engine_*.tex, engine_macros.tex, engine_numbers.json}
         docs/paper/iccv_official/figs/fig_engine_pareto.pdf
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "docs/paper/iccv_official/tables"
FIG = ROOT / "docs/paper/iccv_official/figs"
SCEN4 = ["tum_fr2_xyz", "tum_fr1_xyz", "replica_office0", "replica_office1"]
NAME = {"tum_fr2_xyz": r"\texttt{fr2\_xyz}", "tum_fr1_xyz": r"\texttt{fr1\_xyz}",
        "replica_office0": r"\texttt{office0}", "replica_office1": r"\texttt{office1}"}
W = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
MACROS = {}
NUM = {}


def cname(s):
    """CamelCase letters-only macro name from an identifier (digits spelled out)."""
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return "".join(W[c] if c.isdigit() else c for c in s.title().replace(" ", ""))


def setm(name, val, fmt="{:.2f}"):
    MACROS[name] = (fmt.format(val) if not isinstance(val, str) else val).replace("-", "$-$")
    NUM[name] = val


def sgn(x, nd=2):
    if abs(x) < 0.5 * 10 ** (-nd):
        return f"{0:.{nd}f}"
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A = json.loads((R / "engine/analysis.json").read_text())
    S, U = A["summary"], A["units"]
    for split in ("test", "fresh"):
        for cond, v in ((k.split("/", 1)[1], v) for k, v in S.items() if k.startswith(split + "/")):
            base = "eng" + split.title() + cname(cond)
            for key, nm in (("ms", "Ms"), ("gain_mean", "GainMean"), ("gain_median", "GainMed"), ("gain_min", "GainMin"),
                            ("eq_mean", "EqMean"), ("eq_median", "EqMed"), ("eq_min", "EqMin")):
                setm(base + nm, v[key], "{:.0f}" if key == "ms" else "{:.2f}")
            setm(base + "EqPos", v["eq_pos"], "{:d}")
            setm(base + "N", v["n"], "{:d}")
    # linear cost model ms = a + b K per unit
    a_l, b_l = [], []
    for u in U.values():
        ks = np.array([3, 6, 10, 15.0]); ms = np.array([u[f"tts_fixK{int(k)}"]["ms"] for k in ks])
        b, a = np.polyfit(ks, ms, 1); a_l.append(a); b_l.append(b)
    setm("engCostIntercept", float(np.median(a_l)), "{:.0f}"); setm("engCostSlope", float(np.median(b_l)), "{:.1f}")
    setm("engCostSlopeMin", float(np.min(b_l)), "{:.1f}"); setm("engCostSlopeMax", float(np.max(b_l)), "{:.1f}")
    setm("engFixedShareAtFive", float(np.median([a / (a + 5 * b) for a, b in zip(a_l, b_l)]) * 100), "{:.0f}")
    # time targeting accuracy
    for T in (45, 60, 75, 90):
        c = f"tts_vgg_T{T}"
        dev = [abs(u[c]["ms"] - T) / T * 100 for u in U.values()]
        setm(f"engHit{cname(str(T))}MeanPct", float(np.mean(dev)), "{:.1f}"); setm(f"engHit{cname(str(T))}MaxPct", float(np.max(dev)), "{:.1f}")
    allpct = [abs(u[f"tts_vgg_T{T}"]["ms"] - T) / T * 100 for u in U.values() for T in (45, 60, 75, 90)]
    setm("engHitMaxPct", float(max(allpct)), "{:.1f}"); setm("engHitMeanPct", float(np.mean(allpct)), "{:.1f}")
    k10 = [u["tts_fixK10"]["ms"] for u in U.values()]
    setm("engFixKTenMsMin", float(min(k10)), "{:.0f}"); setm("engFixKTenMsMax", float(max(k10)), "{:.0f}")
    t75 = [u["tts_vgg_T75"]["ms"] for u in U.values()]
    setm("engTSeventyFiveMsMin", float(min(t75)), "{:.0f}"); setm("engTSeventyFiveMsMax", float(max(t75)), "{:.0f}")
    setm("engUnits", len(U), "{:d}")
    # error_off vs fixK6 cost of subset selection
    setm("engSubsetTimeSaved", float(np.mean([u["tts_fixK6"]["ms"] - u["error_off"]["ms"] for k, u in U.items() if k.startswith("test")])), "{:.0f}")

    # ---- main table
    rows = [("Original budgeted system", "error_off"), ("+ steps-regime optimizer", "error_off_opt"),
            ("Prior TT+VGG controllers", "tt_vgg"), ("Prior TT+VGG + optimizer", "tt_vgg_opt"),
            ("Uniform $K{=}3$ steps", "tts_fixK3"), ("Uniform $K{=}6$ steps", "tts_fixK6"), ("Uniform $K{=}10$ steps", "tts_fixK10"),
            ("Uniform $K{=}15$ steps", "tts_fixK15"),
            ("Time-targeted steps, $T{=}60$", "tts_T60"), ("\\quad + VGG, $T{=}45$", "tts_vgg_T45"), ("\\quad + VGG, $T{=}60$", "tts_vgg_T60"),
            ("\\quad + VGG, $T{=}75$", "tts_vgg_T75"), ("\\quad + VGG, $T{=}90$", "tts_vgg_T90")]
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"& \multicolumn{4}{c}{11 test scenes} & \multicolumn{2}{c}{5 fresh windows} & \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
         r"\textbf{Method} & ms & $\Delta$PSNR mean & median & worst & ms & $\Delta$PSNR mean & vs.\ uniform $K$ at equal time (test) \\", r"\midrule"]
    for lab, c in rows:
        t, f = S[f"test/{c}"], S.get(f"fresh/{c}")
        eq = "---" if c.startswith("tts_fixK") or c == "error_off" else sgn(t["eq_mean"]) if c.startswith("tts") or c.endswith("_opt") else sgn(t["eq_mean"])
        if c == "error_off":
            eq = sgn(t["eq_mean"])
        L.append(f"{lab} & {t['ms']:.0f} & {sgn(t['gain_mean'])} & {sgn(t['gain_median'])} & {sgn(t['gain_min'])} & "
                 f"{f['ms']:.0f} & {sgn(f['gain_mean'])} & {eq} \\\\")
        if c in ("error_off_opt", "tt_vgg_opt", "tts_fixK15"):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Steps regime versus the original budgeted system.} PSNR change in dB on the mapped frames relative to the original system (unthrottled raw-error selection, 5 steps, modeled 15\,ms budget) on the same code; mean measured end-to-end frame time in ms. The last column is the mean PSNR difference to the uniform-$K$ curve (linear interpolation in measured time) at the method's own mean time; the original system lies far below that curve. Two seeds per scene; ``test'' are the 11 scenes of the earlier protocol, ``fresh'' are five later windows of long TUM sequences. All optimizer constants were set on four development scenes.}",
          r"\label{tab:engine}", r"\end{table*}"]
    (OUT / "tab_engine_main.tex").write_text("\n".join(L) + "\n")

    # ---- ablation of the optimizer preset (K=10, 4 main scenes)
    ab = {}
    for f in (R / "engine_ablation").glob("*/*_seed*.json"):
        j = json.loads(f.read_text()); ab.setdefault(j["condition"], {}).setdefault(j["scene"], []).append((j["summary"]["mean_psnr"], j["summary"]["mean_frame_ms"]))
    full = {s: np.mean([x[0] for x in ab["tts_full"][s]]) for s in SCEN4}
    SHORT = {"tum_fr2_xyz": "fr2", "tum_fr1_xyz": "fr1", "replica_office0": "off0", "replica_office1": "off1"}
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"\textbf{Optimizer} & " + " & ".join(SHORT[s] for s in SCEN4) + r" & mean & $\Delta$ \\", r"\midrule"]
    for lab, c in (("full", "tts_full"), ("default betas", "tts_nobetas"),
                   ("default lr", "tts_nolr"), ("no lr decay", "tts_nodecay"), ("none of three", "tts_noopt")):
        v = [np.mean([x[0] for x in ab[c][s]]) for s in SCEN4]
        L.append(f"{lab} & " + " & ".join(f"{x:.2f}" for x in v) + f" & {np.mean(v):.2f} & {sgn(np.mean(v) - np.mean(list(full.values())))} \\\\")
        setm("engAbl" + cname(c.replace("tts_", "")) + "Mean", float(np.mean(v)), "{:.2f}")
        setm("engAbl" + cname(c.replace("tts_", "")) + "Delta", float(np.mean(v) - np.mean(list(full.values()))), "{:.2f}")
        setm("engAbl" + cname(c.replace("tts_", "")) + "DeltaAbs", abs(float(np.mean(v) - np.mean(list(full.values())))), "{:.2f}")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Optimizer ablation} at uniform $K{=}10$ steps (about 68\,ms), mean PSNR (dB) on mapped frames, two seeds, four test scenes (fr2, fr1: TUM xyz sequences; off0, off1: Replica).}",
          r"\label{tab:engineabl}", r"\end{table}"]
    (OUT / "tab_engine_ablation.tex").write_text("\n".join(L) + "\n")

    # ---- held-out
    h = {}
    for f in (R / "holdout_engine2").glob("*/*_seed*.json"):
        j = json.loads(f.read_text()); h.setdefault(j["condition"], {}).setdefault(j["scene"], []).append((j["train_psnr"], j["holdout_psnr"]))
    mean = lambda c, s, i: float(np.mean([x[i] for x in h[c][s]]))
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrr}", r"\toprule",
         r"& & " + " & ".join({"tum_fr2_xyz": "fr2", "tum_fr1_xyz": "fr1", "replica_office0": "off0", "replica_office1": "off1"}[s] for s in SCEN4) + r" \\", r"\midrule",
         "original & mapped & " + " & ".join(f"{mean('error_off', s, 0):.1f}" for s in SCEN4) + r" \\",
         " & held-out & " + " & ".join(f"{mean('error_off', s, 1):.1f}" for s in SCEN4) + r" \\", r"\midrule"]
    for lab, c in (("$K{=}3$", "fixK3"), ("$K{=}10$", "fixK10"), ("TTS, $T{=}75$", "tts_vgg_T75")):
        L.append(f"{lab} & $\\Delta$ mapped & " + " & ".join(sgn(mean(c, s, 0) - mean('error_off', s, 0), 1) for s in SCEN4) + r" \\")
        L.append(" & $\\Delta$ held-out & " + " & ".join(sgn(mean(c, s, 1) - mean('error_off', s, 1), 1) for s in SCEN4) + r" \\")
        for s in SCEN4:
            setm(f"engHold{cname(c)}{cname(s)}", mean(c, s, 1) - mean('error_off', s, 1), "{:.1f}")
            setm(f"engMapped{cname(c)}{cname(s)}", mean(c, s, 0) - mean('error_off', s, 0), "{:.1f}")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Held-out views.} Every eighth frame is rendered but never mapped. PSNR (dB) of the original system and the change for three configurations; two seeds. Gains on held-out frames are positive on every scene but a fraction of the gain on mapped frames.}",
          r"\label{tab:engineheld}", r"\end{table}"]
    (OUT / "tab_engine_holdout.tex").write_text("\n".join(L) + "\n")
    for c in ("fixK3", "fixK10"):
        ret = [100 * (mean(c, s, 1) - mean('error_off', s, 1)) / (mean(c, s, 0) - mean('error_off', s, 0)) for s in SCEN4]
        setm(f"engRet{cname(c)}Min", min(ret), "{:.0f}"); setm(f"engRet{cname(c)}Max", max(ret), "{:.0f}")
        hv = [mean(c, s, 1) - mean('error_off', s, 1) for s in SCEN4]
        setm(f"engHold{cname(c)}Min", min(hv), "{:.1f}"); setm(f"engHold{cname(c)}Max", max(hv), "{:.1f}")

    # ---- core speed
    bf, af = json.loads((R / "core_speed/before.json").read_text()), json.loads((R / "core_speed/after.json").read_text())
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"\textbf{Scene} & ms before & ms after & speed-up & PSNR before & PSNR after \\", r"\midrule"]
    sp = []
    for sc in ("tum_fr2_xyz", "tum_fr1_xyz", "replica_office0"):
        kb = [k for k in bf if k.startswith(sc)]
        mb, ma = np.mean([bf[k]["ms"] for k in kb]), np.mean([af[k]["ms"] for k in kb])
        pb, pa = np.mean([bf[k]["psnr"] for k in kb]), np.mean([af[k]["psnr"] for k in kb])
        sp.append(mb / ma)
        L.append(f"{NAME[sc]} & {mb:.0f} & {ma:.0f} & {mb / ma:.1f}$\\times$ & {pb:.2f} & {pa:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Core speed-ups at unchanged output.} Original budgeted configuration on the code revision before and after the changes of Sec.~\ref{sec:core}; end-to-end frame time in ms (including per-frame evaluation), mean PSNR (dB), two seeds, 100 frames.}",
          r"\label{tab:enginespeed}", r"\end{table}"]
    (OUT / "tab_engine_speed.tex").write_text("\n".join(L) + "\n")
    setm("engSpeedupMin", min(sp), "{:.1f}"); setm("engSpeedupMax", max(sp), "{:.1f}")

    # ---- per-unit table (supplement)
    cols = [("error_off", "orig."), ("tts_fixK3", "$K{=}3$"), ("tts_fixK10", "$K{=}10$"), ("tts_vgg_T45", "T45"), ("tts_vgg_T75", "T75")]
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l" + "rr" * len(cols) + "}", r"\toprule",
         "Scene / window & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{l}}}" for _, l in cols) + r" \\",
         r"& " + " & ".join("dB & ms" for _ in cols) + r" \\", r"\midrule"]
    for k, u in U.items():
        nm = k.replace("_", "\\_")
        L.append(f"\\texttt{{{nm}}} & " + " & ".join(f"{u[c]['psnr']:.2f} & {u[c]['ms']:.0f}" for c, _ in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Per-scene results of the steps regime.} Mean PSNR (dB) and measured frame time (ms) per test scene and fresh window (``fresh/'' rows), two seeds. T45 and T75 are time-targeted steps with value-gated growth at 45 and 75\,ms targets.}",
          r"\label{tab:engineunits}", r"\end{table*}"]
    (OUT / "tab_engine_units.tex").write_text("\n".join(L) + "\n")


    # ---- exploratory negative results on the four development scenes (single seed; code not retained)
    DEV = ["tum_fr1_desk", "tum_fr3_structure_texture_near", "replica_room1", "replica_office2"]
    def dev(name):
        v = []
        for s_ in DEV:
            j = json.loads((R / f"engine_dev/{s_}/{name}_seed42.json").read_text())["summary"]
            v.append((j["mean_psnr"], j["mean_frame_ms"]))
        return float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))
    curve = sorted(dev(n) [::-1] for n in ("tts_e_fix3", "tts_e_fix5", "tts_h_base", "tts_e_fix11", "tts_e_fix15"))
    xs, ys = np.array([c[0] for c in curve]), np.array([c[1] for c in curve])
    for tag, name in (("0.03", "tts_e_es0.03"), ("0.015", "tts_e_es0.015"), ("0.007", "tts_e_es0.007")):
        p_, t_ = dev(name)
        setm("engNegEarly" + cname(tag), p_ - float(np.interp(t_, xs, ys)), "{:.2f}")
    ref_p, ref_t = dev("tts_f_1.4_0.6")
    slope = float((np.interp(76, xs, ys) - np.interp(44, xs, ys)) / 32.0)
    setm("engNegSlope", slope, "{:.3f}")
    for tag, name in (("Pos", "tts_s_b0.5"), ("PosOne", "tts_s_b1.0"), ("Neg", "tts_s_bm0.5"), ("NegOne", "tts_s_bm1")):
        p_, t_ = dev(name)
        setm("engNegAlloc" + tag, p_ - (ref_p + slope * (t_ - ref_t)), "{:.2f}")
    hd = {}
    for f in (R / "holdout_dev").glob("*/*_seed42.json"):
        j = json.loads(f.read_text()); hd.setdefault(j["condition"], {})[j["scene"]] = (j["train_psnr"], j["holdout_psnr"])
    for c in ("fixK10_replay3", "fixK10_r2s8", "fixK10_r3s16"):
        setm("engNegReplay" + cname(c) + "Hold", float(np.mean([hd[c][s_][1] - hd["fixK10"][s_][1] for s_ in DEV])), "{:.2f}")
        setm("engNegReplay" + cname(c) + "Mapped", float(np.mean([hd[c][s_][0] - hd["fixK10"][s_][0] for s_ in DEV])), "{:.2f}")

    # ---- development-scene sweeps behind the optimizer constants (K=8, four dev scenes, one seed)
    rows = [("Adam betas $(\\beta_1,\\beta_2)$", [("(0.9, 0.999) default", "tts_b_base"), ("(0.95, 0.999)", "tts_b_0.95_0.999"), ("(0.9, 0.99)", "tts_b_0.9_0.99"),
                                              ("(0.8, 0.99)", "tts_b_0.8_0.99"), ("(0.5, 0.99)", "tts_b_0.5,0.99"), ("(0.0, 0.99)", "tts_b_0.0,0.99"),
                                              ("(0.5, 0.9)", "tts_b_0.5,0.9"), ("(0.5, 0.8) chosen", "tts_b_0.5,0.8"), ("(0.5, 0.7)", "tts_b_0.5,0.7"),
                                              ("(0.3, 0.8)", "tts_b_0.3,0.8"), ("(0.5, 0.5)", "tts_b_0.5,0.5")]),
            ("learning-rate multiplier (betas 0.5, 0.8)", [("0.3", "tts_l2_0.3"), ("0.5", "tts_l2_0.5"), ("0.7", "tts_l2_0.7"),
                                                           ("1.0 chosen", "tts_b_0.5,0.8"), ("1.4", "tts_l2_1.4"), ("2.0", "tts_l2_2.0")])]
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llrr}", r"\toprule",
         r"\textbf{Setting} & \textbf{value} & PSNR & ms \\", r"\midrule"]
    for grp, items in rows:
        for i, (lab, name) in enumerate(items):
            p_, t_ = dev(name)
            L.append((grp if i == 0 else "") + f" & {lab} & {p_:.2f} & {t_:.0f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\caption{\textbf{Development-scene sweeps} behind the optimizer constants: mean PSNR (dB) and frame time over the four development scenes at uniform $K{=}8$ steps, one seed. The learning-rate sweep multiplies all rates of the preset.}",
          r"\label{tab:enginedev}", r"\end{table}"]
    (OUT / "tab_engine_devsweeps.tex").write_text("\n".join(L) + "\n")

    # ---- closed-loop map-based tracking in the steps regime (TUM, three seeds)
    cl = {}
    for f in (R / "closed_loop_steps").glob("*/*_seed*.json"):
        j = json.loads(f.read_text()); cl.setdefault(j["scene"], {}).setdefault(j["condition"], []).append((j["psnr_tracked"], 100 * j["ate_rmse_m"]))
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"\textbf{Scene} & orig.\ dB & ours dB & orig.\ ATE & ours ATE \\", r"\midrule"]
    d_p, d_a = [], []
    for sc in ("tum_fr2_xyz", "tum_fr1_xyz", "tum_fr1_rpy", "tum_fr3_sitting_static", "tum_fr1_room"):
        o, t = cl[sc]["error_off"], cl[sc]["tts"]
        op, oa, tp, ta = (float(np.mean([x[i] for x in v])) for v in (o, t) for i in (0, 1))
        dag = "$^\\dagger$" if sc == "tum_fr1_room" else ""
        L.append("\\texttt{" + sc.replace("tum_", "").replace("_", "\\_") + "}" + dag + f" & {op:.1f} & {tp:.1f} & {oa:.1f} cm & {ta:.1f} cm \\\\")
        if sc != "tum_fr1_room":
            d_p.append(tp - op); d_a.append(ta - oa)
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Closed-loop map-based tracking in the steps regime}, three seeds, TUM, ``ours'' = time-targeted steps at 60\,ms. PSNR is measured at the tracked pose. $^\dagger$Tracking fails for both (excluded from the summary).}",
          r"\label{tab:engineclosed}", r"\end{table}"]
    (OUT / "tab_engine_closedloop.tex").write_text("\n".join(L) + "\n")
    setm("engClosedDPsnrMin", min(d_p), "{:.1f}"); setm("engClosedDPsnrMax", max(d_p), "{:.1f}")
    setm("engClosedDAteMin", min(d_a), "{:.1f}"); setm("engClosedDAteMax", max(d_a), "{:.1f}")
    setm("engClosedAteWorse", int(sum(x > 0 for x in d_a)), "{:d}")

    # ---- retention: final-map PSNR on frames that were never mapped (and early mapped frames)
    def load_ret(split):
        u = {}
        for f in (R / "retention" / split).glob("**/*_seed*.json"):
            j = json.loads(f.read_text()); u.setdefault(j["scene"] + ("@" + str(f.parent.parent.name) if split == "fresh" else ""), {}).setdefault(j["cond"], []).append(j)
        return u
    RC = ["orig", "k3", "k3_rp", "k3_jr", "k10", "k10_rp"]
    RL = {"orig": "original system", "k3": "$K{=}3$ steps", "k3_rp": "$K{=}3$ + replay", "k3_jr": "$K{=}3$ + joint replay", "k10": "$K{=}10$ steps", "k10_rp": "$K{=}10$ + replay"}
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"& \multicolumn{3}{c}{11 test scenes} & \multicolumn{3}{c}{5 fresh windows} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"\textbf{Method} & next & final & early & next & final & early \\", r"\midrule"]
    RET = {}
    for split in ("test", "fresh"):
        U = {k: v for k, v in load_ret(split).items() if all(c in v for c in RC)}
        RET[split] = U
    for c in RC:
        cells = []
        for split in ("test", "fresh"):
            U = RET[split]
            for m in ("heldout_causal", "heldout_final", "mapped_early_final"):
                v = float(np.mean([np.mean([x[m] for x in u[c]]) for u in U.values()]))
                cells.append(f"{v:.1f}")
                setm(f"engKeep{split.title()}{cname(c)}{cname(m)}", v, "{:.2f}")
        L.append(RL[c] + " & " + " & ".join(cells) + r" \\")
        if c in ("orig", "k3_rp"):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Keeping earlier views.} Mean PSNR (dB) with ground-truth poses; frames with index divisible by 5 are never mapped. next: those frames when they arrive; final: the final map on those frames; early: the final map on mapped frames of the first third of the sequence. Two seeds. Replay: every third step uses a stored past frame. Joint replay: steps 2 and 3 optimize the current frame together with two stored past frames.}",
          r"\label{tab:keep}", r"\end{table}"]
    (OUT / "tab_engine_retention.tex").write_text("\n".join(L) + "\n")
    for split in ("test", "fresh"):
        U = RET[split]
        for a_, b_ in (("k3_rp", "k3"), ("k3_rp", "orig"), ("k10_rp", "k10"), ("k10_rp", "orig"), ("k3", "orig"), ("k10", "orig"), ("k3_jr", "orig"), ("k3_jr", "k3_rp"), ("k3_jr", "k3")):
            d = np.array([np.mean([x["heldout_final"] for x in u[a_]]) - np.mean([x["heldout_final"] for x in u[b_]]) for u in U.values()])
            base = f"engKeepD{split.title()}{cname(a_)}Minus{cname(b_)}"
            setm(base + "Mean", float(d.mean()), "{:.2f}"); setm(base + "Worst", float(d.min()), "{:.2f}"); setm(base + "Pos", int((d > 0).sum()), "{:d}")
        setm(f"engKeep{split.title()}N", len(U), "{:d}")
    # batched multi-view joint steps (one rasterizer call for the current and the replayed cameras)
    for split in ("test", "fresh"):
        U = RET[split]
        for a_, b_ in (("k3_jrb", "orig"), ("k3_jr3b", "orig"), ("k3_jr3b", "k3_jr"), ("k3_jrb", "k3_jr")):
            if not all(a_ in u and b_ in u for u in U.values()):
                continue
            d = np.array([np.mean([x["heldout_final"] for x in u[a_]]) - np.mean([x["heldout_final"] for x in u[b_]]) for u in U.values()])
            base = f"engKeepD{split.title()}{cname(a_)}Minus{cname(b_)}"
            setm(base + "Mean", float(d.mean()), "{:.2f}"); setm(base + "Worst", float(d.min()), "{:.2f}"); setm(base + "Pos", int((d > 0).sum()), "{:d}")
        for c_ in ("k3_jr", "k3_jrb", "k3_jr3b"):
            ms = [x["mean_frame_ms"] for u in U.values() for x in u[c_] if "mean_frame_ms" in x]
            if ms:
                setm(f"engMs{split.title()}{cname(c_)}", float(np.mean(ms)), "{:.0f}")
    # full resolution (640x480) on the same test scenes
    hr = {}
    for f in (R / "retention_640" / "test").glob("*/*_seed*.json"):
        j = json.loads(f.read_text()); hr.setdefault(j["cond"], {}).setdefault(j["scene"], []).append(j)
    if hr and all(c in hr for c in ("orig", "k3", "k3_jr", "k3_jr3b", "k2_jr2b")):
        hb = {sc: np.mean([x["heldout_final"] for x in v]) for sc, v in hr["orig"].items()}
        for c in ("orig", "k3", "k3_jr", "k3_jr3b", "k2_jr2b"):
            fin = {sc: np.mean([x["heldout_final"] for x in v]) for sc, v in hr[c].items()}
            dd = np.array([fin[sc] - hb[sc] for sc in fin])
            setm(f"engHiRes{cname(c)}Final", float(np.mean(list(fin.values()))), "{:.1f}")
            setm(f"engHiRes{cname(c)}Gain", float(dd.mean()), "{:.2f}"); setm(f"engHiRes{cname(c)}Worst", float(dd.min()), "{:.2f}")
            setm(f"engHiRes{cname(c)}Pos", int((dd > 0).sum()), "{:d}")
            setm(f"engHiRes{cname(c)}Ms", float(np.mean([x["mean_frame_ms"] for v in hr[c].values() for x in v])), "{:.0f}")
        setm("engHiResN", len(hb), "{:d}")
    tm = {}
    for f in (R / "retention_time").glob("*/*.json"):
        j = json.loads(f.read_text()); tm.setdefault(j["cond"], []).append(j["mean_frame_ms"])
    for c_ in ("orig", "k3", "k3_rp", "k3_jr", "k10", "k10_rp"):
        setm("engMs" + cname(c_), float(np.mean(tm[c_])), "{:.0f}")
    setm("engReplayTimeOverheadKThree", float(100 * (np.mean(tm["k3_rp"]) / np.mean(tm["k3"]) - 1)), "{:.0f}")
    setm("engReplayTimeOverheadKTen", float(100 * (np.mean(tm["k10_rp"]) / np.mean(tm["k10"]) - 1)), "{:.0f}")

    # ---- replay design on the development scenes (final-map retention, GT poses, one seed)
    DEVR = ["tum_fr1_desk", "tum_fr3_structure_texture_near", "replica_room1", "replica_office2"]
    dr = {}
    for f in (R / "retention_dev").glob("*/*_seed42.json"):
        j = json.loads(f.read_text()); dr.setdefault(j["cond"], {})[j["scene"]] = j
    for c in ("orig", "k3", "k3_rp3", "k3_r2", "k3_r4", "k3_s16", "k3_s64", "k3_t1", "k3_t3", "k3_noprune", "k3_rp_noprune", "k3_pe",
              "k10", "k10_rp3", "k10_e2", "k10_e5", "k10_pe"):
        if c in dr and all(s_ in dr[c] for s_ in DEVR):
            setm("engDevRet" + cname(c) + "Final", float(np.mean([dr[c][s_]["heldout_final"] for s_ in DEVR])), "{:.2f}")
            setm("engDevRet" + cname(c) + "Next", float(np.mean([dr[c][s_]["heldout_causal"] for s_ in DEVR])), "{:.2f}")

    # ---- joint-replay design curve on the development scenes (equal measured time)
    d2 = {}
    for f in (R / "retention_dev2").glob("*/*_seed42.json"):
        j = json.loads(f.read_text()); d2.setdefault(j["cond"], {})[j["scene"]] = j
    for alias, c in (("SeqThree", "k3_rp"), ("SeqFive", "k5_rp"), ("SeqSix", "k6_rp"), ("SeqEight", "k8_rp"), ("JointTwoThree", "k2_jt_e1j3"),
                     ("JointThreeTwo", "k3_jt_e1j2"), ("JointThreeFive", "k3_jt_e1j5")):
        if c in d2 and all(s_ in d2[c] for s_ in DEVR):
            setm(f"engJr{alias}Final", float(np.mean([d2[c][s_]["heldout_final"] for s_ in DEVR])), "{:.1f}")
            setm(f"engJr{alias}Ms", float(np.mean([d2[c][s_]["mean_frame_ms"] for s_ in DEVR])), "{:.0f}")

    # ---- MonoGS tracker + our mapper (hybrid) versus original MonoGS, MonoGS' own protocol
    hy = {}
    for f in (R / "monogs_hybrid").glob("*_seed*.json"):
        j = json.loads(f.read_text()); hy.setdefault(j["scene"], {}).setdefault(j["cond"], {})[j["seed"]] = j
    runs = json.loads((R / "monogs_hybrid/monogs_baseline_runs.json").read_text())
    mb = {sc: {k: float(np.mean([r[k] for r in rs])) for k in ("psnr", "ate5", "ms")} | {"n": len(rs)} for sc, rs in runs.items()}
    SC = ["fr1_desk", "fr2_xyz", "fr1_xyz", "fr1_rpy", "fr2_desk", "fr3_sitting_static"]
    def hyb(sc, c):  # mean over seeds (joint replay has three seeds, the others one)
        rs = list(hy[sc][c].values())
        return {"heldout_final_psnr": float(np.mean([r["heldout_final_psnr"] for r in rs])), "ate_every5_m": float(np.mean([r["ate_every5_m"] for r in rs])),
                "track_ms": float(np.mean([r["track_ms"] for r in rs])), "map_ms": float(np.mean([r["map_ms"] for r in rs])), "n": len(rs)}
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{lrrrrrrrrrrrr}", r"\toprule",
         r"& \multicolumn{2}{c}{MonoGS} & \multicolumn{2}{c}{+ original mapper} & \multicolumn{2}{c}{+ ours $K{=}3$} & \multicolumn{2}{c}{+ ours $K{=}3$, replay} & \multicolumn{2}{c}{+ ours $K{=}3$, joint replay} & \multicolumn{2}{c}{+ joint replay, 3 batched views} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
         r"\textbf{Sequence} & dB & cm & dB & cm & dB & cm & dB & cm & dB & cm & dB & cm \\", r"\midrule"]
    cols = ("orig", "k3", "k3_rp", "k3_jr", "k3_jr3b")
    acc = {c: [] for c in ("mono",) + cols}
    for sc in SC:
        m = mb[sc]; row = [f"{m['psnr']:.1f} & {m['ate5']:.2f}"]; acc["mono"].append((m["psnr"], m["ate5"], m["ms"]))
        for c in cols:
            j = hyb(sc, c); row.append(f"{j['heldout_final_psnr']:.1f} & {100 * j['ate_every5_m']:.2f}")
            acc[c].append((j["heldout_final_psnr"], 100 * j["ate_every5_m"], j["track_ms"] + j["map_ms"]))
        L.append("\\texttt{" + sc.replace("_", "\\_") + "} & " + " & ".join(row) + r" \\")
    L.append(r"\midrule")
    L.append("mean & " + " & ".join(f"{np.mean([x[0] for x in acc[k]]):.1f} & {np.mean([x[1] for x in acc[k]]):.2f}" for k in ("mono",) + cols) + r" \\")
    L.append("ms/frame & \\multicolumn{2}{c}{" + f"{np.mean([x[2] for x in acc['mono']]):.0f}" + "} & " + " & ".join("\\multicolumn{2}{c}{" + f"{np.mean([x[2] for x in acc[k]]):.0f}" + "}" for k in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{MonoGS's tracker with our mapper, under MonoGS's own evaluation.} First 150 frames of six TUM sequences at $640{\times}480$. MonoGS: mean of three runs (one run for \texttt{fr3\_sitting\_static}, where the repeats crashed); joint replay: mean of three seeds; other columns: one run. dB: PSNR of the final map on every fifth frame (MonoGS: those that are not keyframes; ours: all of them, none was mapped), rendered at the estimated pose. cm: Umeyama-aligned ATE RMSE over every fifth tracked frame. Per-frame time is MonoGS's total, or tracking plus mapping.}",
          r"\label{tab:hybrid}", r"\end{table*}"]
    (OUT / "tab_engine_hybrid.tex").write_text("\n".join(L) + "\n")
    mean = lambda k, i: float(np.mean([x[i] for x in acc[k]]))
    setm("engHybMonoPsnr", mean("mono", 0)); setm("engHybMonoAte", mean("mono", 1)); setm("engHybMonoMs", mean("mono", 2), "{:.0f}")
    for c in cols:
        setm(f"engHyb{cname(c)}Psnr", mean(c, 0)); setm(f"engHyb{cname(c)}Ate", mean(c, 1)); setm(f"engHyb{cname(c)}Ms", mean(c, 2), "{:.0f}")
    setm("engHybSpeedup", mean("mono", 2) / mean("k3_jr", 2), "{:.1f}")
    setm("engHybGapKThreeJr", mean("k3_jr", 0) - mean("mono", 0), "{:.1f}")
    setm("engHybGapAbs", abs(mean("k3_jr", 0) - mean("mono", 0)), "{:.1f}")
    d = [acc["k3_jr"][i][0] - acc["mono"][i][0] for i in range(len(SC))]
    setm("engHybBetterScenes", int(sum(x > 0.5 for x in d)), "{:d}"); setm("engHybWorstScenes", int(sum(x < -0.5 for x in d)), "{:d}")
    setm("engHybEqualScenes", int(sum(abs(x) <= 0.5 for x in d)), "{:d}"); setm("engHybScenes", len(SC), "{:d}")
    setm("engHybWorstGap", min(d), "{:.1f}"); setm("engHybBestGap", max(d), "{:.1f}"); setm("engHybMedianGap", float(np.median(d)), "{:.1f}")
    setm("engHybAteDiff", mean("k3_jr", 1) - mean("mono", 1), "{:.2f}")
    setm("engHybMonoRpyPsnr", mb["fr1_rpy"]["psnr"], "{:.1f}")
    setm("engHybMonoRangeMax", max(max(r["psnr"] for r in rs) - min(r["psnr"] for r in rs) for rs in runs.values()), "{:.1f}")
    setm("engHybOursRangeMax", max(max(r["heldout_final_psnr"] for r in hy[sc]["k3_jr"].values()) - min(r["heldout_final_psnr"] for r in hy[sc]["k3_jr"].values()) for sc in SC), "{:.1f}")
    setm("engHybClearScenes", int(sum(abs(acc["k3_jr"][i][0] - acc["mono"][i][0]) > max(max(r["psnr"] for r in runs[SC[i]]) - min(r["psnr"] for r in runs[SC[i]]), max(r["heldout_final_psnr"] for r in hy[SC[i]]["k3_jr"].values()) - min(r["heldout_final_psnr"] for r in hy[SC[i]]["k3_jr"].values())) for i in range(len(SC)))), "{:d}")
    d3 = [acc["k3_jr3b"][i][0] - acc["mono"][i][0] for i in range(len(SC))]
    setm("engHybGapThreeB", mean("k3_jr3b", 0) - mean("mono", 0), "{:.1f}"); setm("engHybMedianGapThreeB", float(np.median(d3)), "{:.1f}")
    setm("engHybSpeedupThreeB", mean("mono", 2) / mean("k3_jr3b", 2), "{:.1f}"); setm("engHybAteDiffThreeB", mean("k3_jr3b", 1) - mean("mono", 1), "{:.2f}")
    setm("engHybBetterThreeB", int(sum(x > 0.5 for x in d3)), "{:d}"); setm("engHybEqualThreeB", int(sum(abs(x) <= 0.5 for x in d3)), "{:d}")
    setm("engHybWorseThreeB", int(sum(x < -0.5 for x in d3)), "{:d}"); setm("engHybWorstGapThreeB", float(min(d3)), "{:.1f}")
    setm("engHybReplayGain", mean("k3_jr", 0) - mean("k3", 0), "{:.1f}")

    # ---- Pareto figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tu = {k: u for k, u in U.items() if k.startswith("test/")}
    mp = lambda c: (np.mean([u[c]["ms"] for u in tu.values()]), np.mean([u[c]["psnr"] for u in tu.values()]))
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ks = [mp(f"tts_fixK{k}") for k in (3, 6, 10, 15)]
    ax.plot([x[0] for x in ks], [x[1] for x in ks], "-o", color="#4c78a8", label="uniform $K$ steps", ms=3.5, lw=1.2)
    ts = [mp(f"tts_vgg_T{t}") for t in (45, 60, 75, 90)]
    ax.plot([x[0] for x in ts], [x[1] for x in ts], "s", color="#f58518", label="time-targeted (+VGG)", ms=4, mfc="none")
    for c, lab, mk, col in (("error_off", "original", "X", "#e45756"), ("error_off_opt", "original + optimizer", "P", "#b279a2"),
                            ("tt_vgg", "prior TT+VGG", "v", "#9d755d"), ("tt_vgg_opt", "prior TT+VGG + opt.", "^", "#54a24b")):
        x = mp(c); ax.plot([x[0]], [x[1]], mk, color=col, label=lab, ms=5)
    ax.set_xlabel("measured frame time (ms)"); ax.set_ylabel("mean PSNR (dB)")
    ax.legend(fontsize=5.5, frameon=False, loc="lower right"); ax.grid(alpha=0.25, lw=0.4)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig_engine_pareto.pdf"); plt.close(fig)

    (OUT / "engine_macros.tex").write_text("% generated by paper_repro/make_engine.py; do not edit\n" +
                                           "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())) + "\n")
    (OUT / "engine_numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    print(len(MACROS), "macros")


if __name__ == "__main__":
    main()
