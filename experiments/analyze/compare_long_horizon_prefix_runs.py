#!/usr/bin/env python3
"""Compare a short prefix run with the identical prefix inside a longer run."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIELDS = ("psnr", "ssim", "lpips", "depth_l1", "n_gaussians", "frame_time_ms")


def load_rows(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in FIELDS:
            row[key] = float(row[key])
    return rows


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", type=Path, default=ROOT / "results/long_horizon/long_horizon_1000/run1")
    ap.add_argument("--long", type=Path, default=ROOT / "results/long_horizon/long_horizon_3000/run1")
    ap.add_argument("--output", type=Path, default=ROOT / "results/long_horizon/long_horizon_prefix_comparison")
    ap.add_argument("--map-cap", type=int, default=150000)
    args = ap.parse_args()
    short = args.short if args.short.is_absolute() else ROOT / args.short
    long = args.long if args.long.is_absolute() else ROOT / args.long
    output = args.output if args.output.is_absolute() else ROOT / args.output
    short_meta = json.loads((short / "long_horizon_results.json").read_text())["metadata"]
    long_meta = json.loads((long / "long_horizon_results.json").read_text())["metadata"]
    if short_meta["seed"] != long_meta["seed"] or short_meta["scene"] != long_meta["scene"]:
        raise ValueError("Short and long runs must share scene and seed")
    result = {"short_frames": short_meta["n_frames"], "long_frames": long_meta["n_frames"],
              "scene": long_meta["scene"], "seed": long_meta["seed"], "map_cap": args.map_cap,
              "config_sha256_matches": short_meta["config_sha256"] == long_meta["config_sha256"],
              "hardware_identity_matches": short_meta["hardware"].get("device_name") == long_meta["hardware"].get("device_name")
                  and short_meta["hardware"].get("pytorch") == long_meta["hardware"].get("pytorch")
                  and short_meta["hardware"].get("cuda_version") == long_meta["hardware"].get("cuda_version"),
              "policies": {}}
    for policy in short_meta["policies"]:
        a = load_rows(short / f"trajectory_log_{policy}.csv")
        b = load_rows(long / f"trajectory_log_{policy}.csv")
        n = min(len(a), len(b))
        overlap = {}
        for key in FIELDS:
            deltas = [abs(a[i][key] - b[i][key]) for i in range(n)]
            overlap[key] = {"mean_absolute_difference": mean([{"x": x} for x in deltas], "x"),
                            "max_absolute_difference": max(deltas)}
        psnr_diff = [abs(a[i]["psnr"] - b[i]["psnr"]) for i in range(n)]
        count_diff = [int(a[i]["n_gaussians"]) != int(b[i]["n_gaussians"]) for i in range(n)]
        overlap_diagnostics = {
            "first_gaussian_count_mismatch_frame": int(a[next(i for i, v in enumerate(count_diff) if v)]["frame"]) if any(count_diff) else None,
            "first_psnr_difference_over_0_01_db_frame": int(a[next(i for i, v in enumerate(psnr_diff) if v > 0.01)]["frame"]) if any(v > 0.01 for v in psnr_diff) else None,
            "frames_psnr_difference_over_0_1_db": sum(v > 0.1 for v in psnr_diff),
        }
        early = [row for row in b if 1 <= int(row["frame"]) < short_meta["n_frames"]]
        tail = [row for row in b if int(row["frame"]) >= short_meta["n_frames"]]
        phases = {}
        for name, rows in (("overlapping_prefix", early), ("extended_tail", tail)):
            phases[name] = {"n_frames": len(rows), "mean_psnr": mean(rows, "psnr") if rows else None,
                            "mean_ssim": mean(rows, "ssim") if rows else None,
                            "mean_lpips": mean(rows, "lpips") if rows else None,
                            "mean_depth_l1": mean(rows, "depth_l1") if rows else None,
                            "mean_gaussians": mean(rows, "n_gaussians") if rows else None,
                            "fraction_at_map_cap": sum(r["n_gaussians"] >= args.map_cap for r in rows) / len(rows) if rows else None,
                            "p50_frame_ms": sorted(r["frame_time_ms"] for r in rows)[len(rows)//2] if rows else None,
                            "p95_frame_ms": sorted(r["frame_time_ms"] for r in rows)[math.ceil(.95*len(rows))-1] if rows else None}
        result["policies"][policy] = {"overlap_absolute_differences": overlap,
                                       "overlap_first_divergence": overlap_diagnostics,
                                       "phases": phases}
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# Long-horizon prefix comparison: {result['scene']}", "",
             f"Same seed {result['seed']}; compare {result['short_frames']}-frame run with its prefix in the {result['long_frames']}-frame run.",
             f"Config SHA matches: **{result['config_sha256_matches']}**; hardware/software identity matches: **{result['hardware_identity_matches']}**.",
             "Overlap differences are absolute per-frame metric differences. Phase summaries are descriptive, not independent-frame inference.", ""]
    for policy, data in result["policies"].items():
        first = data["overlap_first_divergence"]
        lines += [f"## {policy}", "",
                  f"First map-count mismatch: frame {first['first_gaussian_count_mismatch_frame']}; first PSNR difference >0.01 dB: frame {first['first_psnr_difference_over_0_01_db_frame']}; frames with PSNR difference >0.1 dB: {first['frames_psnr_difference_over_0_1_db']}.", "",
                  "| Metric | Mean absolute difference on overlap | Max absolute difference |", "|:--|--:|--:|"]
        for key, val in data["overlap_absolute_differences"].items():
            lines.append(f"| {key} | {val['mean_absolute_difference']:.6g} | {val['max_absolute_difference']:.6g} |")
        lines += ["", "| Phase | Frames | Mean PSNR | Mean SSIM | Mean LPIPS | Mean depth L1 | Mean Gaussians | At cap | p50 ms | p95 ms |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for name, p in data["phases"].items():
            vals = ["—" if p[k] is None else f"{p[k]:.3f}" for k in ("mean_psnr", "mean_ssim", "mean_lpips", "mean_depth_l1")]
            nums = ["—" if p[k] is None else f"{p[k]:.0f}" for k in ("mean_gaussians",)]
            tails = ["—" if p[k] is None else f"{p[k]:.1f}" for k in ("fraction_at_map_cap", "p50_frame_ms", "p95_frame_ms")]
            if p["fraction_at_map_cap"] is not None:
                tails[0] = f"{100*p['fraction_at_map_cap']:.1f}%"
            lines.append(f"| {name} | {p['n_frames']} | " + " | ".join(vals + nums + tails) + " |")
        lines.append("")
    lines += ["The map cap constrains observed growth; results do not establish uncapped population stability or tracking performance.", ""]
    (output / "comparison.md").write_text("\n".join(lines))
    print(output / "comparison.md")


if __name__ == "__main__":
    main()
