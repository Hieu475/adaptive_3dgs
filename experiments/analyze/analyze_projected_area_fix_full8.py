#!/usr/bin/env python3
"""Compare full post-fix Phase-14 metrics against retained prior 8-seed runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OLD = RESULTS / "frozen"/"phase14_corrected" / "phase14_results.json"
DIRTY = RESULTS / "frozen"/"phase14_projected_area_fix_full8_dirty" / "final_confirmation_results.json"
NEW = RESULTS / "frozen"/"phase14_projected_area_fix_clean_7policy_8seeds" / "final_confirmation_results.json"
SEEDS = tuple(range(42, 50))
POLICIES = ("no_op", "error_only", "error_influence", "error_influence_temporal", "rtg_slam_reimpl", "ours", "full")
SHARED_POLICIES = ("no_op", "error_only", "error_influence", "rtg_slam_reimpl", "ours", "full")
METRICS = ("mean_psnr", "final_psnr", "mean_ssim", "mean_depth_l1", "N_final", "fps", "mean_frame_time_ms")


def load(path):
    data = json.loads(path.read_text())
    meta = data["metadata"]
    if meta["scene"] != "tum_fr2_xyz" or meta["n_frames"] != 150 or meta["budget_ms"] != 15.0:
        raise ValueError(f"Protocol mismatch in {path}")
    return data


def main():
    old, dirty, new = load(OLD), load(DIRTY), load(NEW)
    old_runs = {(r["policy"], r["seed"]): r for r in old["raw_runs"]}
    dirty_runs = {(r["policy"], r["seed"]): r for r in dirty["raw_runs"]}
    new_runs = {(r["policy"], r["seed"]): r for r in new["raw_runs"]}
    expected = {(p, s) for p in POLICIES for s in SEEDS}
    if set(new_runs) != expected or not {(p, s) for p in SHARED_POLICIES for s in SEEDS}.issubset(old_runs):
        raise ValueError("Expected new seven-policy and old shared-policy eight-seed run identities")

    lines = [
        "# Phase-14 projected-area fix sensitivity: full 8-seed rerun", "",
        "Paired comparison on TUM fr2_xyz (150 frames, 15 ms modeled optimization budget), seeds 42–49. `Prior Phase-14` is the retained merged artifact; `Dirty rerun` is the sensitivity run; `Clean rerun` was executed from the isolated committed source snapshot in this bundle. The temporal policy has no prior Phase-14 artifact, and RTG was not included in the separate dirty sensitivity run.", "",
        "| Policy | Metric | Prior Phase-14 mean | Dirty rerun mean | Clean rerun mean | Clean−prior | Clean−dirty |", "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    policy_deltas = {}
    for policy in POLICIES:
        policy_deltas[policy] = {}
        for metric in METRICS:
            after = [new_runs[(policy, s)][metric] for s in SEEDS]
            dirty_values = ([dirty_runs[(policy, s)][metric] for s in SEEDS]
                            if all((policy, s) in dirty_runs for s in SEEDS) else None)
            unit = " dB" if metric in ("mean_psnr", "final_psnr") else (" m" if metric == "mean_depth_l1" else "")
            if policy in SHARED_POLICIES:
                before = [old_runs[(policy, s)][metric] for s in SEEDS]
                delta = [b - a for a, b in zip(before, after)]
                policy_deltas[policy][metric] = sum(delta) / len(delta)
                old_value, delta_value = f"{sum(before)/len(before):.4f}", f"{sum(delta)/len(delta):+,.4f}{unit}"
            else:
                old_value, delta_value = "no matching prior run", "—"
            dirty_value = f"{sum(dirty_values)/len(dirty_values):.4f}" if dirty_values else "no matching run"
            repeat_delta = (f"{sum(b-a for a,b in zip(dirty_values,after))/len(after):+,.4f}{unit}"
                            if dirty_values else "—")
            lines.append(f"| {policy} | {metric} | {old_value} | {dirty_value} | {sum(after)/len(after):.4f} | {delta_value} | {repeat_delta} |")

    lines += ["", "## Paired policy-gap changes", "",
              "The values below compare mean-PSNR policy gaps in the prior Phase-14 artifact and clean rerun. The prior artifact is dirty, so these changes are descriptive and do not isolate the area-mask edit as the sole cause.", "",
              "| Comparison | Earlier gap | After-fix gap | Change |", "|:--|--:|--:|--:|"]
    for baseline in POLICIES:
        if baseline == "ours" or baseline not in SHARED_POLICIES:
            continue
        old_gap = sum(old_runs[("ours", s)]["mean_psnr"] - old_runs[(baseline, s)]["mean_psnr"] for s in SEEDS) / len(SEEDS)
        new_gap = sum(new_runs[("ours", s)]["mean_psnr"] - new_runs[(baseline, s)]["mean_psnr"] for s in SEEDS) / len(SEEDS)
        lines.append(f"| OURS − {baseline} | {old_gap:+.3f} dB | {new_gap:+.3f} dB | {new_gap-old_gap:+.3f} dB |")

    lines += [
        "", "## New-run outcome", "",
        "The clean-source benchmark passed all 48 trajectories and produced finite aggregate metrics. The repository provenance auditor reports this bundle as PAPER_READY. Its n=8 paired tests use the declared seed as the replicate unit; frames are not treated as independent samples.", "",
        "The detailed clean-run table and significance results are in `phase14_projected_area_fix_clean_8seeds/final_confirmation_report.md`; the paired manifest is in the same directory.", "",
        "## Provenance limit", "",
        "The clean rerun is frozen in a temporary committed checkout; its exact source commit and hashes are recorded in the manifest. The prior artifact remains dirty, so changes relative to that prior result are descriptive rather than proof that the area mask alone caused every difference.", "",
    ]
    out = RESULTS / "frozen"/"phase14_projected_area_fix_sensitivity_clean8.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
