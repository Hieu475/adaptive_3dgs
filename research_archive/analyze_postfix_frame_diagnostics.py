#!/usr/bin/env python3
"""Analyze paired frame diagnostics, clustering uncertainty by trajectory seed."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/legacy/phase14_generalization_postfix/frame_diagnostics"
OUT = ROOT / "results/legacy/phase14_generalization_postfix"
SCENES = ("replica_office1", "tum_fr1_rpy")
SEEDS = tuple(range(42, 50))
FEATURES = (
    "camera_translation_m_per_frame",
    "camera_angular_deg_per_frame",
    "new_surface_ratio_voxel_5cm",
    "depth_valid_ratio_sampled",
    "coverage",
    "mature_update_coverage",
    "growth_pressure",
    "depth_l1_pre",
)


def load_run(scene: str, policy: str, seed: int) -> dict:
    p = BASE / scene / policy / f"seed{seed}.json"
    return json.loads(p.read_text())


def ci_by_seed(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261017)
    means = np.mean(rng.choice(arr, size=(10000, len(arr)), replace=True), axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def main() -> None:
    all_summaries: dict[str, dict] = {}
    binned_rows: list[dict] = []
    lines = [
        "# Frame-level diagnosis of post-fix regressions",
        "",
        "This report pairs each OURS and `error_only` trajectory by scene, seed, and frame. All quality deltas are OURS minus `error_only`; negative values favor `error_only`. Uncertainty intervals resample the eight trajectory seeds. Frames are repeated observations and are not treated as independent replicates.",
        "",
        "Scene-state features are camera translation/rotation per frame from ground-truth poses; sampled valid-depth ratio; and 5 cm world-voxel surface novelty. Novelty is the fraction of unique current valid-depth voxels not observed in previous frames. Policy metrics come from the runner: coverage, mature-update coverage (age ≥10 frames), growth pressure, pre-update depth L1, and quality after each mapping update.",
        "",
        "## Overall paired trajectory results",
        "",
        "| Scene | Mean frame PSNR delta dB (95% seed CI) | Final frame delta dB (95% seed CI) | Mean depth-L1 delta (OURS − error_only) |",
        "|---|---:|---:|---:|",
    ]
    overall_rows: list[str] = []
    for scene in SCENES:
        per_seed: dict[int, dict[int, tuple[dict, dict]]] = {}
        all_rows: list[dict] = []
        summary = {}
        for seed in SEEDS:
            ours = load_run(scene, "ours", seed)["frames"]
            err = load_run(scene, "error_only", seed)["frames"]
            err_by_frame = {int(r["frame"]): r for r in err}
            pairs = []
            for o in ours:
                f = int(o["frame"])
                e = err_by_frame[f]
                rec = {"seed": seed, "frame": f,
                       "delta_psnr_db": float(o["psnr"] - e["psnr"]),
                       "delta_depth_l1": float(o["depth_l1"] - e["depth_l1"]),
                       "ours_psnr": float(o["psnr"]), "error_only_psnr": float(e["psnr"]),
                       "ours_final_psnr": float(o["psnr_post"]),
                       "error_only_final_psnr": float(e["psnr_post"]),
                       "ours_mature_update_coverage": float(o["mature_update_coverage"]),
                       "ours_mature_eligible": int(o["mature_eligible"]),
                       "ours_mature_updated": int(o["mature_updated"]),
                       "ours_growth_pressure": float(o["growth_pressure"]),
                       "ours_coverage": float(o["coverage"]),
                       "ours_n_backlog": int(o["n_backlog"]),
                       "ours_n_gaussians": int(o["n_gaussians"]),
                       "ours_n_spawned": int(o["n_spawned"]),
                       "ours_n_optimized": int(o["n_optimized"]),
                       "ours_compute_return_db_per_opt_ms": float(o["compute_return_db_per_opt_ms"]),
                       "ours_depth_l1_pre": float(o["depth_l1_pre"]),
                       "ours_opt_time_ms": float(o["opt_time_ms"]),
                       "error_only_opt_time_ms": float(e["opt_time_ms"])}
                rec.update({k: float(o[k]) for k in (
                    "camera_translation_m_per_frame", "camera_angular_deg_per_frame",
                    "new_surface_ratio_voxel_5cm", "depth_valid_ratio_sampled")})
                pairs.append((o, e))
                all_rows.append(rec)
            per_seed[seed] = {int(o["frame"]): (o, e) for o, e in pairs}

        mean_deltas = [float(np.mean([r["delta_psnr_db"] for r in all_rows if r["seed"] == seed])) for seed in SEEDS]
        final_deltas = [float(per_seed[s][max(per_seed[s])][0]["psnr"] - per_seed[s][max(per_seed[s])][1]["psnr"]) for s in SEEDS]
        depth_deltas = [float(np.mean([r["delta_depth_l1"] for r in all_rows if r["seed"] == seed])) for seed in SEEDS]
        mean_ci = ci_by_seed(mean_deltas)
        final_ci = ci_by_seed(final_deltas)
        overall_rows.append(f"| `{scene}` | {np.mean(mean_deltas):+.3f} [{mean_ci[0]:+.3f}, {mean_ci[1]:+.3f}] | {np.mean(final_deltas):+.3f} [{final_ci[0]:+.3f}, {final_ci[1]:+.3f}] | {np.mean(depth_deltas):+.5f} |")

        # Trajectory summaries for resource and map-growth behavior.
        per_policy = {}
        for policy in ("ours", "error_only"):
            run_files = [load_run(scene, policy, seed) for seed in SEEDS]
            frames_all = [f for d in run_files for f in d["frames"]]
            eligible = sum(int(f.get("mature_eligible", 0)) for f in frames_all)
            updated = sum(int(f.get("mature_updated", 0)) for f in frames_all)
            per_policy[policy] = {
                "mean_psnr_db": float(np.mean([d["metadata"]["summary"]["mean_psnr"] for d in run_files])),
                "mean_final_gaussians": float(np.mean([d["metadata"]["summary"]["N_final"] for d in run_files])),
                "mean_coverage": float(np.mean([f["coverage"] for f in frames_all])),
                "mature_update_coverage_weighted": updated / eligible if eligible else 0.0,
                "total_mature_eligible": eligible,
                "total_mature_updated": updated,
                "mean_growth_pressure": float(np.mean([f["growth_pressure"] for f in frames_all])),
                "mean_depth_valid_ratio": float(np.mean([f["depth_valid_ratio_sampled"] for f in frames_all])),
                "mean_frame_time_ms": float(np.mean([f["frame_time_ms"] for f in frames_all])),
                "mean_opt_time_ms": float(np.mean([f["opt_time_ms"] for f in frames_all])),
                "negative_compute_return_fraction": float(np.mean([f["compute_return_db_per_opt_ms"] < 0 for f in frames_all])),
                "mean_n_backlog": float(np.mean([f["n_backlog"] for f in frames_all])),
            }

        lines += [
            "",
            f"### {scene}: map and update allocation summary",
            "",
            "| Policy | Mean PSNR dB | Final map Gaussians | Coverage | Mature update coverage (weighted) | Mean growth pressure | Mean frame time ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for policy in ("ours", "error_only"):
            p = per_policy[policy]
            lines.append(
                f"| `{policy}` | {p['mean_psnr_db']:.2f} | {p['mean_final_gaussians']:.0f} | "
                f"{p['mean_coverage']:.3f} | {p['mature_update_coverage_weighted']:.3f} | "
                f"{p['mean_growth_pressure']:.4f} | {p['mean_frame_time_ms']:.1f} |"
            )

        # Paired deltas by quartiles of each feature. Bin deltas per seed first,
        # then bootstrap those eight seed-level summaries.
        lines += ["", f"## {scene}: conditions associated with the frame-level quality gap", "",
                  "For each feature, pooled observations define four quantile groups. Each group first averages frames within each seed; reported mean and CI then use the eight seed-level means. These are descriptive associations, not causal effects.", "",
                  "| Feature and bin (low → high) | Feature range | Paired mean PSNR delta dB (95% seed CI) | seeds represented |",
                  "|---|---:|---:|---:|"]
        for feature in FEATURES:
            frame_values = {}
            for frame in sorted({r["frame"] for r in all_rows}):
                frame_rows = [r for r in all_rows if r["frame"] == frame]
                frame_values[frame] = float(np.mean([
                    r[feature] if feature in r else r[f"ours_{feature}"] for r in frame_rows
                ]))
            ordered_frames = sorted(frame_values, key=lambda f: (frame_values[f], f))
            frame_bin = {frame: min((rank * 4) // len(ordered_frames), 3)
                         for rank, frame in enumerate(ordered_frames)}
            for b in range(4):
                selected_frames = {f for f, bid in frame_bin.items() if bid == b}
                group = [r for r in all_rows if r["frame"] in selected_frames]
                if not group:
                    continue
                seed_means = [float(np.mean([r["delta_psnr_db"] for r in group if r["seed"] == seed]))
                              for seed in SEEDS if any(r["seed"] == seed for r in group)]
                lo, hi = ci_by_seed(seed_means)
                feature_values = [frame_values[f] for f in selected_frames]
                label = f"{feature} Q{b+1}"
                lines.append(f"| {label} | {min(feature_values):.4g}–{max(feature_values):.4g} | {np.mean(seed_means):+.3f} [{lo:+.3f}, {hi:+.3f}] | {len(seed_means)} |")
                binned_rows.append({"scene": scene, "feature": feature, "quantile_bin": b+1,
                                    "feature_min": min(feature_values), "feature_max": max(feature_values),
                                    "mean_delta_psnr_db": float(np.mean(seed_means)),
                                    "ci95_low_seed_cluster": lo, "ci95_high_seed_cluster": hi,
                                    "n_seed_summaries": len(seed_means)})

        # Within-seed rank correlations, summarized by median and range.
        lines += ["", "### Within-trajectory rank association (descriptive)", "",
                  "| Feature | Median Spearman ρ across seeds | range |", "|---|---:|---:|"]
        for feature in FEATURES:
            rhos = []
            for seed in SEEDS:
                rows = [r for r in all_rows if r["seed"] == seed]
                x = np.asarray([r[feature] if feature in r else r[f"ours_{feature}"] for r in rows], dtype=float)
                y = np.asarray([r["delta_psnr_db"] for r in rows], dtype=float)
                rho = safe_corr(x, y)
                if np.isfinite(rho):
                    rhos.append(rho)
            lines.append(f"| {feature} | {np.median(rhos):+.3f} | [{min(rhos):+.3f}, {max(rhos):+.3f}] |" if rhos else f"| {feature} | n/a | n/a |")

        # Chronological segments show whether the gap accumulates or reverses.
        frames = sorted({r["frame"] for r in all_rows})
        chunks = np.array_split(np.asarray(frames), 3)
        lines += ["", "### Temporal progression", "", "| Frame segment | OURS − error_only mean PSNR delta dB (seed-clustered 95% CI) | OURS mean map size | OURS mature update coverage | OURS mean coverage |", "|---|---:|---:|---:|---:|"]
        for chunk in chunks:
            ids = set(int(v) for v in chunk)
            sd, smap, smature, scov = [], [], [], []
            for seed in SEEDS:
                rs = [r for r in all_rows if r["seed"] == seed and r["frame"] in ids]
                sd.append(float(np.mean([r["delta_psnr_db"] for r in rs])))
                smap.append(float(np.mean([r["ours_n_gaussians"] for r in rs])))
                eligible = sum(r["ours_mature_eligible"] for r in rs)
                upd = sum(r["ours_mature_updated"] for r in rs)
                smature.append(upd / eligible if eligible else 0.0)
                scov.append(float(np.mean([r["ours_coverage"] for r in rs])))
            lo, hi = ci_by_seed(sd)
            lines.append(f"| {min(ids)}–{max(ids)} | {np.mean(sd):+.3f} [{lo:+.3f}, {hi:+.3f}] | {np.mean(smap):.0f} | {np.mean(smature):.3f} | {np.mean(scov):.3f} |")

        summary[scene] = {"ours": per_policy["ours"], "error_only": per_policy["error_only"],
                          "paired_mean_delta_by_seed_db": mean_deltas,
                          "paired_final_delta_by_seed_db": final_deltas,
                          "paired_mean_delta_db": float(np.mean(mean_deltas)),
                          "paired_mean_delta_ci95_seed_cluster": list(mean_ci),
                          "paired_final_delta_db": float(np.mean(final_deltas)),
                          "paired_final_delta_ci95_seed_cluster": list(final_ci)}
        all_summaries[scene] = summary[scene]
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "Depth-valid ratio is the fraction of sampled pixels with usable depth in this loader; it reflects scene/sensor coverage and should not be interpreted by itself as sensor noise. Pipeline `coverage` is the per-view fraction of valid-depth pixels whose rendered transmission is below 0.5, not global map coverage. `n_backlog` counts warmup items; this policy disables warmup, so a near-zero value is expected and does not mean the optimization queue is empty. `compute_return_db_per_opt_ms` uses the pipeline's unsynchronized host timer in this benchmark and remains diagnostic only; do not use it as measured hardware utility.",
        "",
        "The frame-diagnostics run is an independent rerun of the same frozen policy and seed protocol, not a byte-identical replay. Its scene-level gaps are similar to the earlier confirmation run, but seed-to-seed and endpoint values differ slightly under gsplat CUDA execution. Use this report's own paired run files for all frame-level results; do not splice per-seed rows from the two runs.",
        "",
        "The strongest observed pattern is trajectory-specific: on Replica office1, the paired gap widens from −1.58 dB in frames 1–40 to −4.06 dB in frames 81–119 while OURS mature-update coverage falls from 0.353 to 0.182. On TUM fr1_rpy, the gap instead narrows from −1.96 dB in frames 1–50 to −0.28 dB in frames 101–150 while mature-update coverage also falls (0.582 to 0.144). Thus low mature-update coverage accompanies the Replica degradation but does not consistently predict the cross-scene quality gap. Motion and novelty correlations also change sign or remain weak across the two scenes. The diagnostics do not justify a universal motion/novelty router or a controller based on mature-update coverage alone.",
        "",
        "Novelty bins describe the shared GT trajectory, while coverage, growth, mature-update coverage, residuals, and map size describe the OURS arm. Associations between them and the quality gap are observational: policy actions alter map state, and map state feeds back into subsequent metrics. Frame-level bins cannot establish the causal effect of a scene condition. The two scenes were selected after known regressions, so these results do not establish held-out generalization.",
        "",
    ]
    separator_index = lines.index("|---|---:|---:|---:|")
    lines[separator_index + 1:separator_index + 1] = overall_rows
    (OUT / "frame_level_diagnosis.md").write_text("\n".join(lines))
    (OUT / "frame_level_diagnosis.json").write_text(json.dumps({"status": "exploratory", "scenes": all_summaries}, indent=2) + "\n")
    with (OUT / "frame_level_bins.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(binned_rows[0]))
        writer.writeheader()
        writer.writerows(binned_rows)
    print(OUT / "frame_level_diagnosis.md")


if __name__ == "__main__":
    main()
