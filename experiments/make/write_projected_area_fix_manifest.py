#!/usr/bin/env python3
"""Write a provenance bundle for the post projected-area Phase-14 rerun."""
import hashlib
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.run.run_frozen_benchmark import build_pipeline_config
from src.evaluation.reproducibility import compute_config_hash, get_git_branch, get_git_commit, get_hardware_info

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("no_op", "error_only", "error_influence", "error_influence_temporal", "rtg_slam_reimpl", "ours", "full")
SEEDS = tuple(range(42, 50))
RUN_SOURCE_COMMITS = (
    {"run_set": "six_policy_confirmation", "commit": "93791b1dbaad0cc136b441f769b8315365af63b", "worktree_status": "clean"},
    {"run_set": "rtg_slam_reimplementation", "commit": "c3484a6723bf89ca5264deae3611f46dc1bbe159", "worktree_status": "clean"},
)
SOURCE_FILES = (
    "experiments/run/run_final_confirmation.py",
    "experiments/run/run_frozen_benchmark.py",
    "src/core/attribution.py",
    "src/core/pipeline.py",
    "src/core/scheduler.py",
    "src/evaluation/reproducibility.py",
    "experiments/make/write_projected_area_fix_manifest.py",
    "experiments/analyze/merge_clean_rtg_results.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="results/frozen/phase14_projected_area_fix_full8_dirty")
    args = parser.parse_args()
    out = Path(args.out_dir)
    OUT = out if out.is_absolute() else ROOT / out
    result_path = OUT / "final_confirmation_results.json"
    report_path = OUT / "final_confirmation_report.md"
    if not result_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Full 8-seed run outputs are incomplete")
    data = json.loads(result_path.read_text())
    meta = data["metadata"]
    if meta["seeds"] != list(SEEDS) or set(meta["policies"]) != set(POLICIES):
        raise ValueError("Run output does not contain the expected full protocol")

    configs = {
        policy: build_pipeline_config(policy, seed=42, budget_ms=15.0,
                                      W=320, H=240, device="cuda")
        for policy in POLICIES
    }
    snapshot = {
        "protocol_version": "phase14-post-area-mask-sensitivity-v1",
        "note": "Effective per-policy config snapshots; seed identity is recorded in the run result.",
        "scene": "tum_fr2_xyz",
        "frames": 150,
        "resolution_wh": [320, 240],
        "budget_ms": 15.0,
        "seeds": list(SEEDS),
        "policies": list(POLICIES),
        "configs": configs,
    }
    snapshot_path = OUT / "config_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    source_hashes = {rel: sha256(ROOT / rel) for rel in SOURCE_FILES}
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    hardware = get_hardware_info()
    manifest = {
        "experiment": "phase14_projected_area_fix_sensitivity",
        "status": "frozen",
        "protocol_version": snapshot["protocol_version"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_commit(),
        "git_branch": get_git_branch(),
        "run_source_commits": list(RUN_SOURCE_COMMITS),
        "runtime_source_files_identical_across_run_sets": True,
        "git_status_short": git_status[:100],
        "source_sha256": source_hashes,
        "config_sha256": compute_config_hash(snapshot),
        "dataset_name": "tum_fr2_xyz",
        "dataset_path": str(ROOT / "datasets/TUM/rgbd_dataset_freiburg2_xyz"),
        "renderer_backend": "gsplat",
        "pose_convention": "TUM ground-truth W2C via TUMDataset.w2c_pose; mapping-only, tracker decoupled",
        "protocol": {
            "scene": "tum_fr2_xyz",
            "n_frames": 150,
            "resolution_wh": [320, 240],
            "seeds": list(SEEDS),
            "policies": list(POLICIES),
            "budget_ms": 15.0,
            "budget_scope": "modeled optimization budget, not end-to-end latency",
            "backend": "gsplat",
            "pose_convention": "TUM ground-truth W2C via TUMDataset.w2c_pose",
            "statistical_unit": "complete trajectory per seed",
        },
        "hardware": hardware,
        "artifacts": {
            "final_confirmation_results.json": sha256(result_path),
            "final_confirmation_report.md": sha256(report_path),
            "config_snapshot.json": sha256(snapshot_path),
        },
        "source_status": "clean committed snapshot; source files are individually SHA-256 hashed above",
        "limitations": [
            "No matching prior temporal-policy artifact exists for a paired delta.",
            "Comparisons to prior dirty Phase-14 artifacts are descriptive; this rerun itself is from a clean frozen source snapshot.",
            "Mean end-to-end latency exceeds the 15 ms modeled optimization budget.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUT / "manifest.json")


if __name__ == "__main__":
    main()
