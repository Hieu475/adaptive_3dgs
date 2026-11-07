#!/usr/bin/env python3
"""Run E1/E2 with common per-scene targets frozen before validation."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_dilution_factorial import config_for
from experiments.run_matched_dilution_controls import run_arm, save_pair
from experiments.run_frozen_benchmark import load_benchmark_sequence
from src.evaluation.reproducibility import create_provenance_manifest


TARGETS_PATH = ROOT / "results" / "matched_dilution_controls" / "fixed_targets.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "tum_fr1_xyz"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 50)))
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, default=TARGETS_PATH.parent)
    ap.add_argument("--controls", nargs="+", choices=("e1", "e2"), default=["e1", "e2"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    targets = json.loads(TARGETS_PATH.read_text())
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)

    run_manifest = create_provenance_manifest(
        config={"scenes": args.scenes, "seeds": args.seeds, "frames": args.n_frames,
                "budget_ms": args.budget_ms, "fixed_targets": targets,
                "controls": args.controls},
        dataset_name=",".join(args.scenes),
        renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
        extra_metadata={"experiment": "fixed_target_e1_e2_confirmation",
                        "status": "exploratory_confirmation",
                        "pose_convention": "W2C via inverse TUM C2W groundtruth"})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_manifest.json"
    if not manifest_path.exists() or args.force:
        manifest_path.write_text(json.dumps(run_manifest, indent=2))

    for scene in args.scenes:
        if scene not in targets["scenes"]:
            raise ValueError(f"No frozen targets for {scene}")
        frames, intrinsics = load_benchmark_sequence(
            scene_name=scene, n_frames=args.n_frames + 1, H=240, W=320, device=args.device)
        for seed in args.seeds:
            if "e1" in args.controls:
                out = args.out_dir / scene / f"E1_seed{seed}.json"
                target = int(targets["scenes"][scene]["target_final_gaussians"])
                if not out.exists() or args.force:
                    arms = []
                    for name, policy in (("no_throttle_matched_final", "error_off"),
                                         ("coverage_throttle_matched_final", "error_coverage")):
                        _, cfg = config_for(policy, seed, args.budget_ms, args.device)
                        arms.append(run_arm(scene, seed, name, cfg, frames, intrinsics, args,
                                            final_gaussian_target=target))
                    counts = [int(a["summary"]["matched_final_map"]["n_gaussians"]) for a in arms]
                    valid = counts == [target, target] and all(
                        a["summary"]["matched_final_map"]["valid"] for a in arms)
                    save_pair(out, {"experiment": "E1_fixed_target_matched_final_map_size",
                        "scene": scene, "seed": seed, "target_final_gaussians": target,
                        "target_source": "pre-frozen fixed_targets.json; minimum of prior clean pilot targets",
                        "valid": valid, "validation": {"final_counts": counts,
                            "exact_target_reached_by_both": valid}, "arms": arms})

            if "e2" in args.controls:
                out = args.out_dir / scene / f"E2_seed{seed}.json"
                target = int(targets["scenes"][scene]["target_total_spawned"])
                if not out.exists() or args.force:
                    midpoint = max(1, args.n_frames // 2)
                    arms = []
                    for name, start, end in (("early_spawn", 1, midpoint),
                                             ("late_spawn", midpoint + 1, args.n_frames)):
                        _, cfg = config_for("error_off", seed, args.budget_ms, args.device)
                        cfg["densification"]["spawn_control"] = {
                            "target_total": target, "start_frame": start, "end_frame": end,
                            "candidate_mode": "uniform_depth"}
                        arms.append(run_arm(scene, seed, name, cfg, frames, intrinsics, args))
                    realized = [int(sum(f["n_spawned"] for f in a["frames"])) for a in arms]
                    valid = realized == [target, target]
                    save_pair(out, {"experiment": "E2_fixed_target_matched_spawn_count_timing",
                        "scene": scene, "seed": seed, "target_total_spawned": target,
                        "target_source": "pre-frozen fixed_targets.json; minimum of prior clean pilot targets",
                        "valid": valid, "validation": {"realized_total_spawned": realized,
                            "exact_target_reached_by_both": valid,
                            "fallback_spawned": [int(sum(f.get("n_spawned_fallback", 0)
                                for f in a["frames"])) for a in arms],
                            "early_window": [1, midpoint], "late_window": [midpoint + 1, args.n_frames],
                            "candidate_mode": "uniform_depth"}, "arms": arms})


if __name__ == "__main__":
    main()
