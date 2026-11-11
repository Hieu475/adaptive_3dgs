#!/usr/bin/env python3
"""Add post-run hashes and gates to a long-horizon prefix artifact directory."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--preregistration", required=True, type=Path)
    ap.add_argument("--command", required=True)
    args = ap.parse_args()
    base = args.input if args.input.is_absolute() else ROOT / args.input
    prereg = args.preregistration if args.preregistration.is_absolute() else ROOT / args.preregistration
    results_path = base / "long_horizon_results.json"
    diagnostics_path = base / "long_horizon_diagnostics.json"
    results, diagnostics = json.loads(results_path.read_text()), json.loads(diagnostics_path.read_text())
    source_paths = ("experiments/run/run_full_sequence_benchmark.py", "experiments/analyze/analyze_long_horizon.py",
                    "experiments/analyze/summarize_long_horizon_chunks.py",
                    "experiments/analyze/compare_long_horizon_prefix_runs.py",
                    "experiments/make/write_long_horizon_manifest.py", "experiments/run/run_frozen_benchmark.py",
                    "src/core/pipeline.py", "src/utility/closed_loop_runtime.py", "datasets/tum_dataset.py")
    output_paths = [p.name for p in base.iterdir() if p.is_file() and p.name != "run_manifest.json"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    manifest = {
        "status": "completed", "command": args.command,
        "protocol_version": results["metadata"]["protocol_version"],
        "scene": results["metadata"]["scene"], "sequence_frames": results["metadata"]["n_frames"],
        "seed": results["metadata"]["seed"], "policies": results["metadata"]["policies"],
        "pose_protocol": results["metadata"]["framing"], "synchronized_frame_timing": results["metadata"]["synchronized_frame_timing"],
        "modeled_optimizer_budget_ms": 15.0, "git_commit": commit + ("-dirty" if dirty else ""),
        "config_sha256": results["metadata"]["config_sha256"],
        "preregistration_sha256": sha(prereg),
        "source_sha256": {p: sha(ROOT / p) for p in source_paths},
        "output_sha256": {p: sha(base / p) for p in sorted(output_paths)},
        "all_completion_and_finite_metric_gates_pass": diagnostics["all_completion_and_finite_metric_gates_pass"],
        "policy_gates": {name: {
            "logged_processed_frames": p["logged_processed_frames"],
            "expected_processed_frames": diagnostics["n_processed_frames_per_policy_expected"],
            "nonfinite_frames": p["nonfinite_frames"],
            "catastrophic_failures": p["catastrophic_failures"],
            "n_final": p["summary"]["N_final"], "n_final_active": p["summary"]["N_final_active"],
            "p50_frame_time_ms": p["summary"]["p50_frame_time_ms"],
            "p95_frame_time_ms": p["summary"]["p95_frame_time_ms"],
            "p99_frame_time_ms": p["summary"]["p99_frame_time_ms"],
        } for name, p in diagnostics["policies"].items()},
    }
    (base / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(base / "run_manifest.json")


if __name__ == "__main__":
    main()
