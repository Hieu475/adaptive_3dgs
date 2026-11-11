#!/usr/bin/env python3
"""Audit and summarize a long-horizon benchmark run from JSON and CSV logs."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "results" / "long_horizon" / "long_horizon_500" / "run1")
    ap.add_argument("--max_gaussians", type=int, default=150000)
    args = ap.parse_args()
    results = json.loads((args.input / "long_horizon_results.json").read_text())
    expected_n = int(results["metadata"]["n_frames"])
    scene = results["metadata"].get("scene", "unknown scene")
    policies = {}
    for summary in results["policies"]:
        name = summary["policy"]
        with (args.input / f"trajectory_log_{name}.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        bad_rows = [int(r["frame"]) for r in rows if not all(
            math.isfinite(float(r[k])) for k in
            ("psnr", "ssim", "depth_l1", "depth_l1_pre", "frame_time_ms"))]
        count_ok = len(rows) == max(0, expected_n - 1)
        cap_frames = [int(r["frame"]) for r in rows
                      if int(r["n_gaussians"]) >= args.max_gaussians]
        checkpoints = []
        by_frame = {int(r["frame"]): r for r in rows}
        checkpoint_step = max(50, expected_n // 10)
        checkpoint_frames = [1, *range(checkpoint_step, expected_n, checkpoint_step), expected_n - 1]
        for frame in sorted(set(checkpoint_frames)):
            if frame not in by_frame:
                continue
            row = by_frame[frame]
            checkpoints.append({
                "frame": frame,
                "n_gaussians": int(row["n_gaussians"]),
                "n_active_gaussians": int(row["n_active_gaussians"]),
                "psnr": float(row["psnr"]),
                "depth_l1_post": float(row["depth_l1"]),
                "depth_l1_pre": float(row["depth_l1_pre"]),
                "frame_time_ms": float(row["frame_time_ms"]),
            })
        policies[name] = {
            "summary": summary,
            "logged_processed_frames": len(rows),
            "frame_count_gate_pass": count_ok,
            "nonfinite_frames": bad_rows,
            "quality_finite_gate_pass": not bad_rows,
            "catastrophic_failures": int(summary["catastrophic_failures"]),
            "first_frame_at_or_above_gaussian_limit": min(cap_frames) if cap_frames else None,
            "frames_at_or_above_gaussian_limit": len(cap_frames),
            "max_observed_gaussians": max(int(r["n_gaussians"]) for r in rows) if rows else None,
            "checkpoints": checkpoints,
        }
    all_gates_pass = all(
        p["frame_count_gate_pass"] and p["quality_finite_gate_pass"]
        and p["catastrophic_failures"] == 0 for p in policies.values())
    analysis = {
        "protocol_version": results["metadata"]["protocol_version"],
        "n_sequence_frames_including_initialization": expected_n,
        "n_processed_frames_per_policy_expected": max(0, expected_n - 1),
        "all_completion_and_finite_metric_gates_pass": all_gates_pass,
        "gaussian_limit_used_for_diagnostic": args.max_gaussians,
        "policies": policies,
        "paired_differences_ours_minus_error_influence": {},
    }
    by_name = {p["policy"]: p for p in results["policies"]}
    if "ours" in by_name and "error_influence" in by_name:
        for key in ("mean_psnr", "mean_ssim", "mean_lpips", "mean_depth_l1",
                    "fps", "p50_frame_time_ms", "p95_frame_time_ms", "p99_frame_time_ms"):
            analysis["paired_differences_ours_minus_error_influence"][key] = (
                float(by_name["ours"][key]) - float(by_name["error_influence"][key]))

    lines = [
        f"# {expected_n}-frame diagnostics: {scene}",
        "",
        f"Single-seed mapping-only prefix on {scene} with ground-truth poses. This is not a full-sequence or cross-scene claim.",
        "",
        f"Completion/finite metric gates: **{'PASS' if all_gates_pass else 'FAIL'}**. Each policy should have {expected_n - 1} processed-frame records plus frame 0 initialization.",
        "",
        "| Policy | Processed rows | NaN/Inf frames | First frame at ≥ Gaussian limit | Frames at limit | Mean PSNR | Mean depth L1 | Mean FPS | p95 frame ms | Final N / active N |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name, p in policies.items():
        s = p["summary"]
        lines.append(
            f"| {name} | {p['logged_processed_frames']} | {len(p['nonfinite_frames'])} | "
            f"{p['first_frame_at_or_above_gaussian_limit'] or '—'} | {p['frames_at_or_above_gaussian_limit']} | "
            f"{s['mean_psnr']:.3f} | {s['mean_depth_l1']:.5f} | {s['fps']:.2f} | "
            f"{s['p95_frame_time_ms']:.1f} | {s['N_final']} / {s['N_final_active']} |"
        )
    lines += ["", "## Map growth and quality checkpoints", ""]
    for name, p in policies.items():
        lines += [f"### {name}", "",
                  "| Frame | Gaussians | Active | PSNR | Post-map depth L1 | Pre-update depth L1 | Synchronized frame ms |",
                  "|--:|--:|--:|--:|--:|--:|--:|"]
        for row in p["checkpoints"]:
            lines.append(
                f"| {row['frame']} | {row['n_gaussians']} | {row['n_active_gaussians']} | "
                f"{row['psnr']:.3f} | {row['depth_l1_post']:.5f} | {row['depth_l1_pre']:.5f} | {row['frame_time_ms']:.1f} |"
            )
        lines.append("")
    all_at_cap = all(p["frames_at_or_above_gaussian_limit"] > 0 for p in policies.values())
    any_at_cap = any(p["frames_at_or_above_gaussian_limit"] > 0 for p in policies.values())
    if all_at_cap:
        cap_note = (f"Every policy reached the {args.max_gaussians:,}-Gaussian limit. "
                    "These results describe growth under the configured cap, not unbounded growth.")
    elif any_at_cap:
        cap_note = (f"At least one policy reached the {args.max_gaussians:,}-Gaussian limit; "
                    "interpret map growth in light of this configured cap.")
    else:
        cap_note = (f"No policy reached the {args.max_gaussians:,}-Gaussian diagnostic limit "
                    "during this prefix.")
    lines += [
        "## Interpretation",
        "",
        cap_note,
        "",
        "Synchronized p95 latency is far above 15 ms; 15 ms is the modeled optimization-kernel budget, not an end-to-end real-time guarantee.",
    ]
    (args.input / "long_horizon_diagnostics.json").write_text(json.dumps(analysis, indent=2))
    (args.input / "long_horizon_diagnostics.md").write_text("\n".join(lines) + "\n")
    print(args.input / "long_horizon_diagnostics.md")


if __name__ == "__main__":
    main()
