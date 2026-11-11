#!/usr/bin/env python3
"""Regenerate the compact utility-learning figures included in bao_cao."""
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "legacy" / "learned_utility"
OUT = ROOT / "bao_cao" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
})


def prediction_scatter():
    rows = json.loads((RESULTS / "phase5_interface" / "predictions_test.json").read_text())
    rows = [r for r in rows if r.get("seed") == 42]
    x_raw = np.array([r["oracle_utility"] for r in rows])
    y_raw = np.array([r["predicted_utility"] for r in rows])
    # The saved prediction and target heads do not share a verified physical
    # scale. Plot ranks only; this figure makes no calibration claim.
    x = rankdata(x_raw, method="average") / len(x_raw)
    y = rankdata(y_raw, method="average") / len(y_raw)
    positive = np.array([r["oracle_utility"] >= 0 for r in rows])
    rho = 0.037

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax.scatter(x[positive], y[positive], s=12, alpha=.7, label="Oracle utility ≥ 0")
    ax.scatter(x[~positive], y[~positive], s=18, marker="x", alpha=.8, label="Oracle utility < 0")
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(grid, slope * grid + intercept, color="#222222", lw=1.1, label="Linear fit")
    ax.axhline(0, color="0.5", lw=.7, ls=":")
    ax.set_xlabel("Oracle utility rank (percentile)")
    ax.set_ylabel("Predicted utility rank (percentile)")
    ax.set_title("Ranking agreement on held-out candidates (checkpoint 42)")
    ax.text(.02, .97, f"N = {len(rows)}\nSpearman $\\rho$ = {rho:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.8", "alpha": .9})
    ax.grid(alpha=.2)
    ax.legend(loc="upper right", frameon=True, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "utility_prediction_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def quality_latency_tradeoff():
    path = ROOT / "results" / "frozen" / "phase14_projected_area_fix_clean_7policy_8seeds" / "final_confirmation_results.json"
    runs = json.loads(path.read_text())["raw_runs"]
    policies = {}
    for r in runs:
        policies.setdefault(r["policy"], []).append(r)
    labels = {
        "no_op": "No-op", "error_only": "Error", "error_influence": "Error × influence",
        "error_influence_temporal": "Temporal–influence", "rtg_slam_reimpl": "RTG-SLAM (reimpl.)",
        "ours": "OURS", "full": "FULL",
    }
    colors = {"ours": "#2E8B57", "full": "#555555", "no_op": "#999999"}
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for name, group in policies.items():
        latency = np.array([r["mean_frame_time_ms"] for r in group], dtype=float)
        psnr = np.array([r["mean_psnr"] for r in group], dtype=float)
        nfinal = np.array([r["N_final"] for r in group], dtype=float)
        x, y = latency.mean(), psnr.mean()
        # Marker area encodes mean final map size; error bars show seed SD.
        size = 35 + 150 * (nfinal.mean() - min(v["N_final"] for q in policies.values() for v in q)) / (max(v["N_final"] for q in policies.values() for v in q) - min(v["N_final"] for q in policies.values() for v in q))
        ax.errorbar(x, y, xerr=latency.std(ddof=1), yerr=psnr.std(ddof=1),
                    fmt="none", ecolor=colors.get(name, "#4C78A8"), alpha=.48, lw=.9, capsize=2)
        ax.scatter(x, y, s=size, color=colors.get(name, "#4C78A8"), edgecolor="white", linewidth=.7,
                   label=labels.get(name, name), zorder=3)
    offsets = {"ours": (7, 8), "full": (7, -14), "no_op": (7, -10), "error_influence": (7, 6), "error_influence_temporal": (7, -13), "rtg_slam_reimpl": (7, 5), "error_only": (7, 6)}
    for name, group in policies.items():
        x = np.mean([r["mean_frame_time_ms"] for r in group])
        y = np.mean([r["mean_psnr"] for r in group])
        ax.annotate(labels.get(name, name), (x, y), xytext=offsets.get(name, (5, 5)), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("Measured mean frame latency (ms)")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title("Quality–latency trade-off (8 seeds per policy)")
    ax.grid(alpha=.2)
    ax.text(.99, .02, "Marker area ∝ mean final Gaussian count; bars show ±1 SD across seeds.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4)
    fig.tight_layout()
    fig.savefig(OUT / "quality_latency_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def quota_trajectory_diagnostic():
    path = ROOT / "results" / "legacy" / "phase14_generalization_postfix" / "room1_trajectory_age_quota_seeds270_272" / "frame_diagnostics.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    frames = np.array(sorted({int(r["frame"]) for r in rows}))
    seed_ids = sorted({int(r["seed"]) for r in rows})

    def per_frame(field):
        values = np.array([[float(next(r[field] for r in rows if int(r["seed"]) == seed and int(r["frame"]) == frame))
                            for frame in frames] for seed in seed_ids])
        return values.mean(axis=0), values.std(axis=0, ddof=1)

    share25, share25_sd = per_frame("mature25_realized_update_share")
    share75, share75_sd = per_frame("mature75_realized_update_share")
    delta_psnr, delta_psnr_sd = per_frame("delta_psnr_75_minus_25")
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    left.plot(frames, share25, color="#4C78A8", lw=1.4, label="Mature quota 25%")
    left.fill_between(frames, share25-share25_sd, share25+share25_sd, color="#4C78A8", alpha=.16, linewidth=0)
    left.plot(frames, share75, color="#E17C05", lw=1.4, label="Mature quota 75%")
    left.fill_between(frames, share75-share75_sd, share75+share75_sd, color="#E17C05", alpha=.16, linewidth=0)
    left.set_title("Realized mature update share")
    left.set_xlabel("Frame")
    left.set_ylabel("Share of optimizer updates")
    left.set_ylim(-.04, 1.04)
    left.legend(frameon=False, fontsize=6.5, loc="upper left")
    right.plot(frames, delta_psnr, color="#4C78A8", lw=1.2)
    right.fill_between(frames, delta_psnr-delta_psnr_sd, delta_psnr+delta_psnr_sd,
                       color="#4C78A8", alpha=.18, linewidth=0)
    right.axhline(0, color="0.4", lw=.8, ls="--")
    right.set_title("PSNR difference (75% − 25%)")
    right.set_xlabel("Frame")
    right.set_ylabel("Δ PSNR (dB)")
    for ax in (left, right):
        ax.grid(alpha=.18)
    fig.suptitle("Frame-wise age-quota intervention (Replica room1, 3 seeds)", y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "quota_trajectory_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)


def feature_ablation():
    data = json.loads((RESULTS / "learned_utility_summary.json").read_text())["v0_v7_ablation"]
    labels = [f"V{i}" for i in range(len(data))]
    x = np.arange(len(data))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.6))
    left.errorbar(x, [d["spearman_rho_mean"] for d in data],
                  yerr=[d["spearman_rho_std"] for d in data],
                  marker="o", capsize=3, lw=1.2, color="#3572A5")
    left.axhline(0, color="0.4", lw=.7)
    left.set_title("Rank correlation")
    left.set_ylabel("Spearman $\\rho$")
    right.errorbar(x, [d["ndcg_20pct_mean"] for d in data],
                   yerr=[d["ndcg_20pct_std"] for d in data],
                   marker="o", capsize=3, lw=1.2, color="#D97721")
    right.set_title("Top 20% selection")
    right.set_ylabel("NDCG@20%")
    for ax in (left, right):
        ax.set_xticks(x, labels)
        ax.set_xlabel("Feature set")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Utility feature ablation (five historical checkpoints)", y=.99)
    mapping = ("V0 RGB   V1 +depth   V2 +gradient   V3 +visibility   "
               "V4 +influence\nV5 +temporal   V6 +uncertainty   V7 +cost/lifecycle")
    fig.text(.5, .015, mapping, ha="center", va="bottom", fontsize=6.8)
    fig.tight_layout(rect=(0, .12, 1, .94))
    fig.savefig(OUT / "utility_feature_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def offline_and_closed_loop():
    table = json.loads((RESULTS / "benchmark_table.json").read_text())
    lookup = {r["method"]: r for r in table}
    heuristic = lookup["B3: Error × Influence"]
    mlp = lookup["B7: Two-Head MLP (Ours)"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.7))
    x = np.arange(2)
    width = .34
    left.bar(x - width / 2, [heuristic["ndcg_20pct_mean"], mlp["ndcg_20pct_mean"]],
             width, label="NDCG@20%", color="#4C78A8")
    left.bar(x + width / 2, [heuristic["ose_20pct_mean"], mlp["ose_20pct_mean"]],
             width, label="OSE@20%", color="#F28E2B")
    left.set_xticks(x, ["Heuristic", "Two-head MLP"])
    left.set_ylabel("Offline ranking score")
    left.set_title("Zero-shot offline test")
    left.set_ylim(0, .68)
    left.legend(frameon=False, ncol=2, loc="upper right")
    left.grid(axis="y", alpha=.2)

    values = [17.34, 14.83]
    right.bar([0, 1], values, color=["#4C78A8", "#E45756"], width=.58)
    right.set_xticks([0, 1], ["Heuristic\n+ coverage", "MLP\n+ coverage"])
    right.set_ylabel("Mean PSNR (dB)")
    right.set_title("Closed-loop pilot")
    right.set_ylim(0, 20)
    right.annotate("−2.51 dB", xy=(1, 14.83), xytext=(.5, 18.4), ha="center",
                   arrowprops={"arrowstyle": "->", "lw": .9})
    right.grid(axis="y", alpha=.2)
    fig.suptitle("Offline ranking and online mapping use separate protocols", y=.99)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / "utility_offline_vs_closed_loop.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    prediction_scatter()
    feature_ablation()
    offline_and_closed_loop()
    quality_latency_tradeoff()
    quota_trajectory_diagnostic()
    print(f"Wrote utility figures to {OUT}")
