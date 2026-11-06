#!/usr/bin/env python3
"""Run a tiny reference-renderer online-mapping trajectory twice and compare stages."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_full_sequence_benchmark import preload_tum_sequence_cpu
from experiments.run_phase13_frozen_benchmark import build_pipeline_config
from research.pipeline import OnlineReconstructionPipeline
from research.reproducibility import get_git_commit, get_hardware_info, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_path", default="datasets/TUM/rgbd_dataset_freiburg2_xyz")
    parser.add_argument("--camera", default="freiburg2")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/gsplat_backward_repeat/reference_pipeline_control.json")
    args = parser.parse_args()
    if args.frames < 2:
        raise ValueError("Use at least two frames to exercise online mapping.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)

    rgb_cpu, depth_cpu, poses_cpu, intrinsics, *_ = preload_tum_sequence_cpu(
        str(REPO_ROOT / args.scene_path),
        camera=args.camera,
        W=args.width,
        H=args.height,
        max_frames=args.frames,
    )
    config = build_pipeline_config(
        policy="full", seed=args.seed, budget_ms=100.0,
        W=args.width, H=args.height, device=args.device,
    )
    config = copy.deepcopy(config)
    config["system"]["repro_trace"] = True
    config["gaussian"]["init_stride"] = 2
    config["gaussian"]["init_refine_steps"] = 1
    config["gaussian"]["max_gaussians"] = 5000
    config["densification"]["max_new_per_frame"] = 8
    config["scheduler"]["policy"] = "full"
    config["rendering"]["backend"] = "reference"
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    config_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    runs = []
    for run_idx in range(2):
        set_seed(args.seed)
        pipeline = OnlineReconstructionPipeline(config=copy.deepcopy(config), device=args.device)
        pipeline.initialize(
            rgb=(rgb_cpu[0].float() / 255.0).to(device),
            depth=depth_cpu[0].float().to(device),
            intrinsics=intrinsics.to(device),
            pose=poses_cpu[0].to(device),
        )
        frame_records = []
        for frame_idx in range(1, args.frames):
            metrics = pipeline.process_frame(
                rgb=(rgb_cpu[frame_idx].float() / 255.0).to(device),
                depth=depth_cpu[frame_idx].float().to(device),
                gt_pose=poses_cpu[frame_idx].to(device),
            )
            frame_records.append({
                "frame": frame_idx,
                "psnr": float(metrics["psnr"]),
                "ssim": float(metrics["ssim"]),
                "depth_l1": float(metrics["depth_l1"]),
                "n_gaussians": int(metrics["n_gaussians"]),
                "trace": metrics.get("repro_trace", []),
            })
        runs.append({
            "initialization_trace": pipeline._init_repro_trace,
            "frames": frame_records,
            "final_gaussians": pipeline.gaussian_model.num_gaussians,
        })
        print(f"Run {run_idx}: final Gaussian count {pipeline.gaussian_model.num_gaussians}")
        del pipeline

    first_divergence = None
    first = runs[0]
    second = runs[1]
    for stage_idx, (a, b) in enumerate(zip(first["initialization_trace"], second["initialization_trace"])):
        if a["hashes"] != b["hashes"]:
            first_divergence = {"scope": "initialization", "stage": a["stage"], "stage_index": stage_idx}
            break
    if first_divergence is None:
        for frame_a, frame_b in zip(first["frames"], second["frames"]):
            traces_a = frame_a["trace"]
            traces_b = frame_b["trace"]
            for stage_idx, (a, b) in enumerate(zip(traces_a, traces_b)):
                if a["hashes"] != b["hashes"]:
                    first_divergence = {
                        "scope": "mapping_frame",
                        "frame": frame_a["frame"],
                        "stage": a["stage"],
                        "stage_index": stage_idx,
                    }
                    break
            if first_divergence:
                break

    frame_metrics_equal = all(
        {k: frame_a[k] for k in ("psnr", "ssim", "depth_l1", "n_gaussians")}
        == {k: frame_b[k] for k in ("psnr", "ssim", "depth_l1", "n_gaussians")}
        for frame_a, frame_b in zip(first["frames"], second["frames"])
    )
    payload = {
        "protocol": "two same-seed full pipeline runs using the PyTorch reference renderer",
        "scene": "tum_fr2_xyz",
        "frames": args.frames,
        "resolution": [args.width, args.height],
        "seed": args.seed,
        "policy": "full",
        "renderer_backend": "reference",
        "device": str(device),
        "config_sha256": config_hash,
        "git_commit": get_git_commit(),
        "hardware": get_hardware_info(),
        "first_stage_divergence": first_divergence,
        "frame_metrics_bitwise_equal": frame_metrics_equal,
        "final_gaussian_counts": [run["final_gaussians"] for run in runs],
        "runs": runs,
        "caveat": "Tiny low-resolution repeatability diagnostic, not a quality or latency benchmark.",
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"First stage divergence: {first_divergence}")
    print(f"Frame metrics bitwise equal: {frame_metrics_equal}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
