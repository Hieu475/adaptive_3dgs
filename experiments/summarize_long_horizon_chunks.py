#!/usr/bin/env python3
"""Summarize quality, map size, and synchronized latency in fixed frame chunks."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--max-gaussians", type=int, default=150000)
    args = ap.parse_args()
    base = args.input if args.input.is_absolute() else ROOT / args.input
    results = json.loads((base / "long_horizon_results.json").read_text())
    result = {"protocol_version": results["metadata"]["protocol_version"],
              "scene": results["metadata"]["scene"], "seed": results["metadata"]["seed"],
              "chunk_size": args.chunk_size, "policies": {}}
    for summary in results["policies"]:
        name = summary["policy"]
        with (base / f"trajectory_log_{name}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        out_chunks = []
        if not rows or any(not all(math.isfinite(float(r[k])) for k in ("psnr", "depth_l1", "frame_time_ms")) for r in rows):
            raise ValueError(f"Missing rows or nonfinite values for policy {name}")
        for start in range(1, int(rows[-1]["frame"]) + 1, args.chunk_size):
            chunk = [r for r in rows if start <= int(r["frame"]) < start + args.chunk_size]
            if not chunk:
                continue
            nums = lambda key: [float(r[key]) for r in chunk]
            out_chunks.append({
                "frame_start": int(chunk[0]["frame"]), "frame_end": int(chunk[-1]["frame"]),
                "n_frames": len(chunk), "mean_psnr": sum(nums("psnr")) / len(chunk),
                "mean_ssim": sum(nums("ssim")) / len(chunk), "mean_depth_l1": sum(nums("depth_l1")) / len(chunk),
                "mean_gaussians": sum(nums("n_gaussians")) / len(chunk),
                "fraction_at_gaussian_cap": sum(int(float(r["n_gaussians"]) >= args.max_gaussians) for r in chunk) / len(chunk),
                "p50_frame_time_ms": sorted(nums("frame_time_ms"))[len(chunk) // 2],
                "p95_frame_time_ms": sorted(nums("frame_time_ms"))[max(0, math.ceil(.95 * len(chunk)) - 1)],
            })
        result["policies"][name] = {"summary": summary, "chunks": out_chunks}
    (base / "long_horizon_chunks.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# {result['scene']} {results['metadata']['n_frames']}-frame chunk diagnostics", "",
             f"Seed {result['seed']}; chunks of {args.chunk_size} processed frames. Map-cap fraction uses {args.max_gaussians:,} Gaussian threshold.", ""]
    for name, data in result["policies"].items():
        lines += [f"## {name}", "", "| Frames | Mean PSNR | Mean SSIM | Mean depth L1 | Mean Gaussians | At cap | p50 ms | p95 ms |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
        for c in data["chunks"]:
            lines.append(f"| {c['frame_start']}–{c['frame_end']} | {c['mean_psnr']:.3f} | {c['mean_ssim']:.4f} | {c['mean_depth_l1']:.5f} | {c['mean_gaussians']:.0f} | {100*c['fraction_at_gaussian_cap']:.1f}% | {c['p50_frame_time_ms']:.1f} | {c['p95_frame_time_ms']:.1f} |")
        lines.append("")
    lines += ["Chunk means are descriptive; frames within a trajectory are temporally dependent and are not independent samples.", ""]
    (base / "long_horizon_chunks.md").write_text("\n".join(lines))
    print(base / "long_horizon_chunks.md")


if __name__ == "__main__":
    main()
