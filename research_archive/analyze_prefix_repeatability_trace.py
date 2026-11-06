#!/usr/bin/env python3
"""Find the first initialization/frame/stage hash mismatch between prefix runs."""
import argparse
import json
from pathlib import Path

import numpy as np


def load_run(path):
    path = Path(path)
    return (json.loads((path / "manifest.json").read_text()),
            json.loads((path / "prefix_trace.json").read_text()))


def compare_gradient_dumps(path_a, path_b):
    a_path, b_path = Path(path_a) / "init_step_0_gradients.npz", Path(path_b) / "init_step_0_gradients.npz"
    if not a_path.exists() or not b_path.exists():
        return None
    result = {}
    with np.load(a_path) as a, np.load(b_path) as b:
        for name in sorted(set(a.files) & set(b.files)):
            av, bv = a[name].astype(np.float64), b[name].astype(np.float64)
            delta = np.abs(av - bv)
            denom = np.linalg.norm(av.ravel())
            result[name] = {
                "shape": list(av.shape),
                "exact_equal": bool(np.array_equal(av, bv)),
                "different_element_fraction": float(np.mean(av != bv)),
                "max_abs_difference": float(delta.max(initial=0.0)),
                "mean_abs_difference": float(delta.mean()),
                "p99_abs_difference": float(np.quantile(delta, 0.99)),
                "relative_l2_difference": float(np.linalg.norm(delta.ravel()) / max(denom, 1e-30)),
            }
    return result


def first_field_difference(frames_a, frames_b, field):
    for a, b in zip(frames_a, frames_b):
        if a.get(field) != b.get(field):
            return {"frame": a.get("input_frame_index"),
                    "run_a": a.get(field), "run_b": b.get(field)}
    if len(frames_a) != len(frames_b):
        return {"reason": "frame_count_mismatch", "run_a": len(frames_a), "run_b": len(frames_b)}
    return None


def compare_stage_lists(left, right, location):
    if len(left) != len(right):
        return {"location": location, "reason": "stage_count_mismatch",
                "left_count": len(left), "right_count": len(right)}
    for i, (a, b) in enumerate(zip(left, right)):
        if a.get("stage") != b.get("stage"):
            return {"location": f"{location}[{i}]", "reason": "stage_name_mismatch",
                    "left": a.get("stage"), "right": b.get("stage")}
        ah, bh = a.get("hashes", {}), b.get("hashes", {})
        differing = sorted(k for k in set(ah) | set(bh) if ah.get(k) != bh.get(k))
        if differing:
            left_summary = a.get("summaries", {})
            right_summary = b.get("summaries", {})
            return {"location": f"{location}[{i}]", "stage": a.get("stage"),
                    "reason": "tensor_hash_mismatch", "tensors": differing,
                    "summary_comparison": {name: {"run_a": left_summary.get(name),
                                                   "run_b": right_summary.get(name)}
                                           for name in differing
                                           if name in left_summary or name in right_summary}}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_a", type=Path, required=True)
    ap.add_argument("--run_b", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    ma, a = load_run(args.run_a)
    mb, b = load_run(args.run_b)
    criteria = ["scene", "seed", "snapshot_frame", "prefix_policy", "pose_convention", "renderer_backend"]
    config_a, config_b = a.get("config", {}), b.get("config", {})
    mismatched_config = {key: [config_a.get(key), config_b.get(key)]
                         for key in sorted(set(config_a) | set(config_b))
                         if config_a.get(key) != config_b.get(key)}
    provenance = {key: [ma.get(key), mb.get(key)] for key in
                  ("git_sha", "git_branch", "config_sha256", "dataset_name", "torch_version", "cuda_version", "gpu")}
    first = compare_stage_lists(a.get("init_repro_trace", []),
                                b.get("init_repro_trace", []), "initialization")
    first_frame = None
    fa, fb = a["frames"], b["frames"]
    for i, (ra, rb) in enumerate(zip(fa, fb)):
        frame = int(ra["input_frame_index"])
        if ra["input_frame_index"] != rb["input_frame_index"]:
            first_frame = {"location": f"frame_index[{i}]", "reason": "frame_index_mismatch",
                           "left": ra["input_frame_index"], "right": rb["input_frame_index"]}
            break
        first_frame = compare_stage_lists(ra.get("repro_trace", []),
                                          rb.get("repro_trace", []), f"frame_{frame}")
        if first_frame:
            first_frame["input_frame_index"] = frame
            first_frame["n_gaussians"] = [ra.get("n_gaussians"), rb.get("n_gaussians")]
            first_frame["spawned"] = [ra.get("n_spawned"), rb.get("n_spawned")]
            first_frame["updates"] = [ra.get("n_micro_updates"), rb.get("n_micro_updates")]
            first_frame["rng_digest_after_frame_equal"] = ra.get("rng_digest_after_frame") == rb.get("rng_digest_after_frame")
            first_frame["map_digest_after_frame_equal"] = ra.get("map_digest_after_frame") == rb.get("map_digest_after_frame")
            break
    if first_frame is None and len(fa) != len(fb):
        first_frame = {"reason": "frame_count_mismatch", "left_count": len(fa), "right_count": len(fb)}
    map_hash_match = a.get("final_map_digest") == b.get("final_map_digest")
    result = {
        "study": "same_seed_prefix_repeatability_trace",
        "run_a": str(args.run_a), "run_b": str(args.run_b),
        "comparison_criteria": criteria,
        "run_level_config_mismatches": mismatched_config,
        "provenance": provenance,
        "init_stage_first_mismatch": first,
        "frame_stage_first_mismatch": first_frame,
        "first_frame_field_differences": {
            field: first_field_difference(a.get("frames", []), b.get("frames", []), field)
            for field in ("n_gaussians", "n_spawned", "n_micro_updates",
                          "spawn_control_count", "rng_digest_after_frame",
                          "map_digest_after_frame")},
        "final_map_digest_match": map_hash_match,
        "final_map_digest": [a.get("final_map_digest"), b.get("final_map_digest")],
        "init_gradient_elementwise_comparison": compare_gradient_dumps(args.run_a, args.run_b),
        "final_gaussian_count": [
            (a.get("frames") or [{}])[-1].get("n_gaussians"),
            (b.get("frames") or [{}])[-1].get("n_gaussians"),
        ],
        "interpretation": "exact repeated-prefix diagnostic; a hash mismatch localizes but does not explain the underlying CUDA or algorithmic source",
    }
    out = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Same-seed prefix repeatability comparison", "",
             f"Run A: `{args.run_a}`  ", f"Run B: `{args.run_b}`", "",
             f"- Config mismatches: `{json.dumps(mismatched_config, sort_keys=True)}`",
             f"- Final map digest match: **{map_hash_match}**",
             f"- Final Gaussian counts: {result['final_gaussian_count']}", "",
             "## First mismatch", ""]
    if first:
        lines.append(f"Initialization: `{json.dumps(first, sort_keys=True)}`")
    else:
        lines.append("Initialization hashes match.")
    if first_frame:
        lines.append(f"Frame trace: `{json.dumps(first_frame, sort_keys=True)}`")
    else:
        lines.append("Per-frame stage hashes match.")
    lines += ["", "## Initial gradient numeric deltas", ""]
    if result["init_gradient_elementwise_comparison"]:
        for name, stats in result["init_gradient_elementwise_comparison"].items():
            lines.append(f"- `{name}`: differing={stats['different_element_fraction']:.6%}; max |Δ|={stats['max_abs_difference']:.9g}; mean |Δ|={stats['mean_abs_difference']:.9g}; relative L2={stats['relative_l2_difference']:.9g}")
    else:
        lines.append("Exact gradient dumps unavailable.")
    lines += ["", "## First frame-level state/count differences", ""]
    for field, difference in result["first_frame_field_differences"].items():
        lines.append(f"- `{field}`: `{json.dumps(difference, sort_keys=True)}`")
    lines += ["", "A hash mismatch localizes the earliest recorded difference. Numeric gradient deltas help quantify it but do not by themselves establish whether CUDA reduction ordering, sampler randomness, or another source caused it.", ""]
    (out / "comparison.md").write_text("\n".join(lines))
    print(out / "comparison.md")


if __name__ == "__main__":
    main()
