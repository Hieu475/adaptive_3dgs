#!/usr/bin/env python3
"""Compare registered all-scene E1/E2 effects across independent seeds."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-a", type=int, default=44)
    ap.add_argument("--seed-b", type=int, default=45)
    ap.add_argument("--input-root", type=Path, default=ROOT / "results" / "replica")
    ap.add_argument("--output", type=Path, default=ROOT / "results/replica/replica_E1E2_seed_pair_comparison")
    args = ap.parse_args()
    root = args.input_root if args.input_root.is_absolute() else ROOT / args.input_root
    out = args.output if args.output.is_absolute() else ROOT / args.output
    rows = []
    for scene in ("office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"):
        records = {}
        for seed in (args.seed_a, args.seed_b):
            base = root / f"replica_E1E2_seed{seed}_capacity_fixed" / "analysis.json"
            analysis = json.loads(base.read_text())
            records[seed] = next(r for r in analysis["scenes"] if r["scene"] == scene)
        row = {"scene": scene, "seeds": {}}
        for seed, r in records.items():
            row["seeds"][str(seed)] = {
                "E1_final_psnr_delta": r["E1_delta_coverage_minus_no_throttle"]["psnr"],
                "E1_final_depth_delta": r["E1_delta_coverage_minus_no_throttle"]["depth_l1"],
                "E1_natural_mean_psnr_delta": r["E1_natural_trajectory_delta"]["mean_psnr"],
                "E2_mean_psnr_delta": r["E2_late_minus_early"]["mean_psnr"],
                "E2_final_psnr_delta": r["E2_late_minus_early"]["final_psnr"],
                "E2_mean_depth_delta": r["E2_late_minus_early"]["mean_depth_l1"],
                "E2_final_depth_delta": r["E2_late_minus_early"]["final_depth_l1"],
                "all_gates_pass": r["E1_valid"] and r["E2_valid"] and r["all_frame_metrics_finite"],
            }
        for metric in ("E1_final_psnr_delta", "E1_final_depth_delta", "E1_natural_mean_psnr_delta",
                       "E2_mean_psnr_delta", "E2_final_psnr_delta", "E2_mean_depth_delta", "E2_final_depth_delta"):
            a, b = (row["seeds"][str(seed)][metric] for seed in (args.seed_a, args.seed_b))
            row[metric + "_mean"] = (a + b) / 2
            row[metric + "_same_direction"] = (a > 0 and b > 0) or (a < 0 and b < 0) or (a == b == 0)
        rows.append(row)
    metrics = ("E1_final_psnr_delta", "E1_final_depth_delta", "E1_natural_mean_psnr_delta",
               "E2_mean_psnr_delta", "E2_final_psnr_delta", "E2_mean_depth_delta", "E2_final_depth_delta")
    summary = {m: {
        "same_direction_scenes": sum(r[m + "_same_direction"] for r in rows),
        "mean_effect_seed_a": sum(r["seeds"][str(args.seed_a)][m] for r in rows) / len(rows),
        "mean_effect_seed_b": sum(r["seeds"][str(args.seed_b)][m] for r in rows) / len(rows),
        "mean_effect_both_seeds": sum(r[m + "_mean"] for r in rows) / len(rows),
    } for m in metrics}
    result = {"protocol_version": "matched-dilution-controls-replica-all-scenes-v2",
              "seeds": [args.seed_a, args.seed_b], "scene_count": len(rows),
              "all_gates_pass": all(r["seeds"][str(s)]["all_gates_pass"] for r in rows for s in (args.seed_a, args.seed_b)),
              "scenes": rows, "summary": summary,
              "interpretation": "Two-seed exploratory repeatability check across eight short scene prefixes. Direction agreement is descriptive only; scenes are not independent random samples and two seeds do not support strong uncertainty claims."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# Replica E1/E2 paired-seed comparison ({args.seed_a}, {args.seed_b})", "",
             f"All exact-count and finite-metric gates pass: **{'PASS' if result['all_gates_pass'] else 'FAIL'}**.", "",
             "Effects are arm B minus arm A: coverage throttle minus no throttle for E1, late spawn minus early spawn for E2. Depth L1: negative favors the listed first policy/arm.", "",
             "| Scene | E1 matched PSNR Δ (a/b) | E1 matched depth Δ (a/b) | E1 natural mean PSNR Δ (a/b) | E2 mean PSNR Δ (a/b) | E2 mean depth Δ (a/b) |", "|:--|--:|--:|--:|--:|--:|"]
    display = (("E1_final_psnr_delta", "dB"), ("E1_final_depth_delta", "m"), ("E1_natural_mean_psnr_delta", "dB"), ("E2_mean_psnr_delta", "dB"), ("E2_mean_depth_delta", "m"))
    for r in rows:
        vals = []
        for metric, unit in display:
            a, b = (r["seeds"][str(s)][metric] for s in (args.seed_a, args.seed_b))
            vals.append(f"{a:+.3f}/{b:+.3f} {unit}")
        lines.append(f"| {r['scene']} | " + " | ".join(vals) + " |")
    lines += ["", "## Direction agreement", ""]
    for metric, label in (("E1_final_psnr_delta", "E1 matched-map PSNR"), ("E1_final_depth_delta", "E1 matched-map depth L1"),
                          ("E1_natural_mean_psnr_delta", "E1 natural-trajectory mean PSNR"),
                          ("E2_mean_psnr_delta", "E2 mean PSNR"), ("E2_mean_depth_delta", "E2 mean depth L1"),
                          ("E2_final_depth_delta", "E2 final depth L1")):
        s = summary[metric]
        lines.append(f"- {label}: same effect direction on {s['same_direction_scenes']}/8 scenes; mean effect seed {args.seed_a} {s['mean_effect_seed_a']:+.4f}, seed {args.seed_b} {s['mean_effect_seed_b']:+.4f}.")
    lines += ["", result["interpretation"], ""]
    (out / "comparison.md").write_text("\n".join(lines))
    print(out / "comparison.md")


if __name__ == "__main__":
    main()
