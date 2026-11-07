#!/usr/bin/env python3
"""Write post-run hashes and validation gates for an all-scene Replica E1/E2 suite."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES = ("office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2")
SOURCES = ("experiments/run_matched_dilution_controls.py", "experiments/run_dilution_factorial.py",
           "experiments/run_frozen_benchmark.py", "experiments/analyze_replica_e1_e2_suite.py",
           "experiments/compare_replica_e1_e2_seeds.py", "experiments/write_replica_suite_manifest.py",
           "research/pipeline.py", "research/closed_loop_runtime.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    base = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    prereg = base / "preregistration.json"
    analysis_path = base / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    scene_gates, output_files = {}, ["analysis.json", "analysis.md", "manifest.json", "preregistration.json"]
    for scene in SCENES:
        folder = base / f"replica_{scene}"
        e1p, e2p = folder / f"E1_seed{args.seed}.json", folder / f"E2_seed{args.seed}.json"
        e1, e2 = json.loads(e1p.read_text()), json.loads(e2p.read_text())
        row = next(r for r in analysis["scenes"] if r["scene"] == scene)
        scene_gates[f"replica_{scene}"] = {
            "E1_valid": bool(e1["valid"]), "E1_target_final_gaussians": int(e1["target_final_gaussians"]),
            "E1_realized_final_counts": e1["validation"]["final_counts"],
            "E2_valid": bool(e2["valid"]), "E2_target_births": int(e2["target_total_spawned"]),
            "E2_shared_gaussian_capacity": int(e2["shared_gaussian_capacity"]),
            "E2_realized_births": e2["validation"]["realized_total_spawned"],
            "E2_online_frames": int(e2["validation"]["realized_online_frames"]),
            "all_frame_metrics_finite": bool(row["all_frame_metrics_finite"]),
        }
        output_files += [f"replica_{scene}/{e1p.name}", f"replica_{scene}/{e2p.name}"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    manifest = {
        "status": "completed", "protocol_version": analysis["protocol_version"],
        "command": f"python experiments/run_matched_dilution_controls.py --scenes {' '.join('replica_' + s for s in SCENES)} --seeds {args.seed} --n_frames 120 --controls e1 e2 --device cuda --out_dir {base.relative_to(ROOT)}",
        "seed": args.seed, "git_commit": commit + ("-dirty" if dirty else ""),
        "preregistration_sha256": sha(prereg), "source_sha256": {p: sha(ROOT / p) for p in SOURCES},
        "output_sha256": {p: sha(base / p) for p in output_files if (base / p).is_file()},
        "scene_validation": scene_gates,
        "all_count_and_finiteness_gates_pass": bool(analysis["all_count_and_finiteness_gates_pass"]),
    }
    (base / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(base / "run_manifest.json")


if __name__ == "__main__":
    main()
