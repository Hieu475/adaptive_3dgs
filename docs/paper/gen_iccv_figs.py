#!/usr/bin/env python3
"""Generate ICCV skeleton figures from frozen measured numbers (no reruns).

Fig1: v3 sensor-noise separation across 5 scenes (raw hole-rate based).
Fig2: main-benchmark deltas vs OURS with 95% CI (fr2_xyz 150f x8).
Fig3: kappa ablation on replica_office0 (30f x3) + selection gap.
Fig4: map-size vs PSNR Pareto (Phase-14 6 policies).

All values anchored to results/ artifacts; see JSON sidecars.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 200})

# ---- Fig1: v3 separation (measured 2026-10-01, 10 frames/scene means) ----
scenes = ["replica_office0", "replica_room0", "tum_fr1_xyz", "tum_fr1_desk", "tum_fr2_xyz"]
v3 = [0.017, 0.022, 0.706, 0.493, 0.714]
cols = ["tab:green" if s.startswith("replica") else "tab:blue" for s in scenes]
fig, ax = plt.subplots(figsize=(3.4, 1.9))
ax.bar(range(len(scenes)), v3, color=cols)
ax.axhspan(0.10, 0.30, color="gray", alpha=0.15, label="blend band")
ax.axhline(0.10, color="k", ls="--", lw=0.7)
ax.set_xticks(range(len(scenes)))
ax.set_xticklabels([s.replace("replica_", "R:").replace("tum_", "T:") for s in scenes], rotation=18, ha="right")
ax.set_ylabel("v3 noise score")
ax.legend(["blend band", "threshold"], fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig1_noise_separation.pdf")
json.dump({"scenes": scenes, "v3": v3}, open(OUT / "fig1.json", "w"), indent=1)

# ---- Fig2: paired differences vs OURS (Phase-14 main, 8 seeds) ----
# Plot the frozen paired bootstrap intervals as a forest plot; positive means OURS wins.
stats_path = Path(__file__).resolve().parents[2] / "results" / "phase14_projected_area_fix_clean_7policy_8seeds" / "final_confirmation_results.json"
stats = json.load(open(stats_path))
comparison_names = {
    "no_op": "No-op",
    "rtg_slam_reimpl": "RTG-SLAM",
    "error_only": "Error-only",
    "error_influence": "Error + influence",
    "error_influence_temporal": "Temporal + influence",
    "full": "FULL ceiling",
}
comparisons = {row["baseline"]: row for row in stats["paired_comparisons"]}
order = ["no_op", "rtg_slam_reimpl", "error_only", "error_influence", "error_influence_temporal", "full"]
fig, ax = plt.subplots(figsize=(3.4, 2.55))
y = np.arange(len(order))
for yi, name in zip(y, order):
    row = comparisons[name]
    delta = row["delta_mean_psnr"]
    lo, hi = row["ci_mean_psnr"]
    color = "0.35" if name == "full" else "#16834a"
    marker = "D" if name == "full" else "o"
    ax.errorbar(delta, yi, xerr=[[delta - lo], [hi - delta]], fmt=marker,
                color=color, ecolor=color, markeredgecolor="white", markeredgewidth=0.5,
                markersize=4.5, elinewidth=1.2, capsize=2.5, zorder=3)
    ax.text(10.15, yi, f"{delta:+.2f}", va="center", ha="right", fontsize=7)
ax.axvline(0, color="0.2", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels([comparison_names[n] for n in order], fontsize=7)
ax.invert_yaxis()
ax.set_xlim(-0.55, 10.35)
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_xlabel("OURS $-$ baseline mean PSNR (dB)", fontsize=7.5)
ax.set_ylabel("")
ax.grid(axis="x", color="0.88", lw=0.5)
ax.set_axisbelow(True)
ax.text(10.15, 1.02, "$\\Delta$", transform=ax.get_xaxis_transform(),
        ha="right", va="bottom", fontsize=7)
fig.tight_layout(pad=0.6)
fig.savefig(OUT / "fig2_main_deltas.pdf")

# ---- Fig3: separate coverage throttling from selector choice (office0 30f x3) ----
kappa_path = Path(__file__).resolve().parents[2] / "results" / "phase14_multiscene" / "replica_kappa_ablation.json"
kappa_data = json.load(open(kappa_path))
conds = ["$\\kappa=.80$", "$\\kappa=.90$", "$\\kappa=.95$", "Throttle\noff", "Raw-error\nselection"]
keys = ["k0.80", "k0.90", "k0.95", "off", "error_only_ref"]
vals = [kappa_data[k]["mean_psnr"] for k in keys]
fig, ax = plt.subplots(figsize=(3.4, 2.35))
x = np.array([0, 1, 2, 3, 4.45])
ax.plot(x[:4], vals[:4], color="#8b9aa7", lw=0.9, zorder=1)
ax.scatter(x[:4], vals[:4], s=27, color="#2878a5", edgecolors="white", linewidths=0.6, zorder=3,
           label="coverage-throttle sweep")
ax.scatter(x[4], vals[4], s=34, marker="D", color="#16834a", edgecolors="white", linewidths=0.6,
           zorder=3, label="raw-error selector")
for xi, value in zip(x, vals):
    ax.text(xi, value + 0.19, f"{value:.2f}", ha="center", va="bottom", fontsize=6.7)
ax.axvline(3.72, color="0.55", ls=(0, (2, 2)), lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(conds, fontsize=6.7)
ax.set_ylabel("mean PSNR (dB)", fontsize=7.5)
ax.set_ylim(22.5, 30.5)
ax.set_xlim(-0.55, 5.05)
ax.set_yticks([23, 25, 27, 29])
ax.grid(axis="y", color="0.88", lw=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=6.2, loc="upper left", frameon=True)
fig.tight_layout(pad=0.6)
fig.savefig(OUT / "fig3_kappa_ablation.pdf")
json.dump({"source": str(kappa_path.relative_to(kappa_path.parents[2])), "conditions": keys, "mean_psnr": vals,
           "off_minus_k090_db": vals[3] - vals[1],
           "error_only_minus_off_db": vals[4] - vals[3]},
          open(OUT / "fig3.json", "w"), indent=1)

# ---- Fig4: quality vs map size (Phase-14, 8 seeds) ----
# Only OURS and FULL are nondominated; three growing heuristics exceed 65k splats.
phase_path = Path(__file__).resolve().parents[2] / "results" / "phase14_projected_area_fix_clean_7policy_8seeds" / "final_confirmation_results.json"
phase = json.load(open(phase_path))
policy_stats = phase["policy_stats"]
names = ["no_op", "ours", "rtg_slam_reimpl", "error_only", "full", "error_influence"]
short_names = {"no_op": "No-op", "ours": "OURS", "rtg_slam_reimpl": "RTG-SLAM",
               "error_only": "Error-only", "full": "FULL", "error_influence": "Error + influence"}
points = {name: (policy_stats[name]["N_final"]["mean"] / 1000,
                 policy_stats[name]["mean_psnr"]["mean"]) for name in names}
fig, ax = plt.subplots(figsize=(3.4, 2.35))
ax.axvspan(65, 125, color="#dfe7ed", alpha=0.42, zorder=0)
ax.axvline(65, color="0.45", ls=(0, (3, 2)), lw=0.8, zorder=1)
ax.text(66.5, 17.35, ">65k", fontsize=6.5, color="0.35", va="bottom")
label_offsets = {
    "no_op": (4, -3), "ours": (9, 9), "rtg_slam_reimpl": (-37, -13),
    "error_only": (6, -12), "full": (5, 4), "error_influence": (-56, -13),
}
for name in names:
    mx, quality = points[name]
    if name == "ours":
        ax.scatter(mx, quality, s=90, c="#c63d3d", marker="*", edgecolors="0.2", lw=0.5, zorder=4)
    elif name == "full":
        ax.scatter(mx, quality, s=36, c="0.35", marker="D", edgecolors="white", lw=0.5, zorder=4)
    elif name == "no_op":
        ax.scatter(mx, quality, s=28, facecolors="white", edgecolors="0.35", marker="o", lw=1, zorder=3)
    else:
        ax.scatter(mx, quality, s=30, c="#2878a5", marker="o", edgecolors="white", lw=0.5, zorder=3)
    dx, dy = label_offsets[name]
    ax.annotate(f"{short_names[name]}\n{mx:.0f}k, {quality:.2f}", (mx, quality),
                xytext=(dx, dy), textcoords="offset points", fontsize=5.8,
                ha="left" if dx >= 0 else "right", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.45, color="0.45"))
# Exact nondominated segment under the objectives: minimize map size, maximize PSNR.
ax.plot([points["ours"][0], points["full"][0]],
        [points["ours"][1], points["full"][1]],
        color="#c63d3d", ls=(0, (3, 2)), lw=0.9, zorder=2)
ax.set_xlim(0, 125)
ax.set_ylim(17.2, 29.6)
ax.set_xticks([0, 25, 50, 75, 100, 125])
ax.set_yticks([18, 20, 22, 24, 26, 28])
ax.set_xlabel("final map size (k primitives)", fontsize=7.5)
ax.set_ylabel("mean PSNR (dB)", fontsize=7.5)
ax.grid(color="0.9", lw=0.45)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.6)
fig.savefig(OUT / "fig4_pareto.pdf")
json.dump({"source": str(phase_path.relative_to(phase_path.parents[2])), "points": points,
           "pareto_frontier": ["ours", "full"], "heuristic_map_threshold_k": 65},
          open(OUT / "fig4.json", "w"), indent=1)

# ---- Fig6: budget curve (measured 30-frame runs, 3 seeds) ----
budget_path = Path(__file__).resolve().parents[2] / "results" / "phase14_budget_curve" / "budget_curve.json"
budget = json.load(open(budget_path))
budgets = budget["metadata"]["budgets"]
rows = budget["rows"]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for policy, marker, ls in [("ours", "o", "-"), ("error_only", "s", "--")]:
    selected = [r for r in rows if r["policy"] == policy]
    ax.errorbar([r["budget_ms"] for r in selected], [r["mean_psnr"] for r in selected],
                yerr=[r["std_psnr"] for r in selected], marker=marker, ls=ls,
                capsize=2, lw=1.2, ms=4, label=policy)
ax.set_ylabel("mean PSNR (dB)")
ax.set_xlabel("budget (ms)")
ax.set_xticks(budgets)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig6_budget.pdf")

# ---- Fig7: quality and map growth traces (seed 42, 149 frames) ----
trace_path = OUT / "fig7_traces_seed42.json"
traces = json.load(open(trace_path))
fig, (ax_n, ax_q) = plt.subplots(2, 1, figsize=(3.4, 2.2), sharex=True)
trace_styles = {"ours": ("tab:red", "-", "ours (23K)"),
                "error_only": ("tab:blue", "--", "error_only (69K)")}
for policy, (color, ls, label) in trace_styles.items():
    logs = traces[policy]["logs"]
    frames = [r["frame"] for r in logs]
    ax_n.plot(frames, [r["n_gaussians"] / 1000 for r in logs], color=color, ls=ls, lw=1.1, label=label)
    ax_q.plot(frames, [r["psnr"] for r in logs], color=color, ls=ls, lw=1.1,
              label=f"{policy} final {logs[-1]['psnr']:.1f}")
ax_n.set_ylabel("map size (K)")
ax_q.set_ylabel("PSNR (dB)")
ax_q.set_xlabel("frame")
ax_n.legend(fontsize=6.5, loc="upper left")
ax_q.legend(fontsize=6.5, loc="lower left")
ax_q.set_ylim(15, 32)
ax_q.set_xlim(-5, 155)
fig.tight_layout(pad=0.6)
fig.savefig(OUT / "fig7_dilution.pdf")

print("wrote", sorted(p.name for p in OUT.glob("fig*")))
