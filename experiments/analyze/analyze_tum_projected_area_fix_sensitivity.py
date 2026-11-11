#!/usr/bin/env python3
"""Compare two-seed corrected-area rerun to the retained Phase-14 TUM runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OLD = RESULTS / "baselines"/"rtg_slam_150f_8seeds" / "final_confirmation_results.json"
NEW = RESULTS / "frozen"/"phase14_projected_area_fix_sensitivity_tum_fr2" / "final_confirmation_results.json"
SEEDS = (42, 43)
POLICIES = ("ours", "error_only")
METRICS = ("mean_psnr", "final_psnr", "mean_depth_l1", "N_final", "fps", "mean_frame_time_ms")


def load(path):
    data = json.loads(path.read_text())
    metadata = data["metadata"]
    if metadata["scene"] != "tum_fr2_xyz" or metadata["n_frames"] != 150 or metadata["budget_ms"] != 15.0:
        raise ValueError(f"Unexpected protocol metadata in {path}")
    return data


def main():
    old, new = load(OLD), load(NEW)
    old_runs = {(r["policy"], r["seed"]): r for r in old["raw_runs"]}
    new_runs = {(r["policy"], r["seed"]): r for r in new["raw_runs"]}
    lines = [
        "# TUM fr2_xyz projected-area fix sensitivity", "",
        "Paired two-seed comparison (42, 43), 150 frames, 15 ms modeled budget, corrected TUM W2C loader. The earlier values come from the retained 8-seed Phase-14 artifact; the rerun contains only OURS and error-only.", "",
        "| Policy | Seed | Metric | Earlier | After fix | Change |", "|:--|--:|:--|--:|--:|--:|",
    ]
    for policy in POLICIES:
        for seed in SEEDS:
            before = old_runs[(policy, seed)]
            after = new_runs[(policy, seed)]
            for metric in METRICS:
                delta = after[metric] - before[metric]
                if metric in ("mean_psnr", "final_psnr"):
                    values = f"{before[metric]:.3f} | {after[metric]:.3f} | {delta:+.3f} dB"
                elif metric == "mean_depth_l1":
                    values = f"{before[metric]:.5f} | {after[metric]:.5f} | {delta:+.5f} m"
                elif metric == "N_final":
                    values = f"{before[metric]:,.0f} | {after[metric]:,.0f} | {delta:+,.0f}"
                else:
                    values = f"{before[metric]:.2f} | {after[metric]:.2f} | {delta:+.2f}"
                lines.append(f"| {policy} | {seed} | {metric} | {values} |")
    lines += [
        "", "## Rerun summary", "",
        "The corrected-area rerun completed all four trajectories (OURS/error-only × seeds 42/43) with finite aggregate metrics. OURS averaged 27.630 dB mean PSNR and 28.792 dB final PSNR; error-only averaged 23.967 dB and 20.066 dB. At n=2, the paired Wilcoxon p-value is 0.5, so this run has no confirmatory statistical power. It is a sensitivity check only.", "",
        "The run measured about 88–90 ms mean end-to-end frame time for OURS despite the 15 ms modeled optimization budget. It does not support a real-time claim.", "",
        "## Provenance and comparison limits", "",
        "The earlier artifact is retained as an 8-seed run, while the rerun intentionally uses only seeds 42 and 43. Both use the same declared scene, frame count, policy definitions, and modeled budget, but the earlier manifest records a dirty source state and does not hash the complete working-tree diff. The comparison is descriptive and cannot certify the exact source change as the sole cause. A clean, committed 8-seed rerun is still required for paper claims.", "",
    ]
    out = RESULTS / "frozen"/"tum_fr2_projected_area_fix_sensitivity.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
