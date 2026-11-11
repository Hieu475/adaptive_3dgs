#!/usr/bin/env python3
"""Measure first-step gsplat backward gradient variation across same-seed inits."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run.run_full_sequence_benchmark import (
    build_experiment_configs,
    preload_tum_sequence_cpu,
)
from src.core.pipeline import OnlineReconstructionPipeline
from src.evaluation.reproducibility import get_git_commit, get_hardware_info, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_path", default="datasets/TUM/rgbd_dataset_freiburg2_xyz")
    parser.add_argument("--camera", default="freiburg2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out_dir", default="results/diagnostics/gsplat_backward_repeat")
    parser.add_argument("--strict_determinism", action="store_true")
    args = parser.parse_args()

    if args.repeats < 2:
        raise ValueError("At least two independent initializations are required.")
    if args.strict_determinism:
        import os
        if args.device.startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python for strict CUDA mode.")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    W, H = 320, 240
    rgb_cpu, depth_cpu, poses_cpu, intrinsics, *_ = preload_tum_sequence_cpu(
        str(REPO_ROOT / args.scene_path), camera=args.camera, W=W, H=H, max_frames=1
    )
    dev = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rgb = (rgb_cpu[0].float() / 255.0).to(dev)
    depth = depth_cpu[0].float().to(dev)
    pose = poses_cpu[0].to(dev)
    intrinsics = intrinsics.to(dev)

    config = build_experiment_configs(
        seed=args.seed, W=W, H=H, device=args.device, sh_degree=1
    )["ours"]
    config = copy.deepcopy(config)
    config.setdefault("system", {})["repro_trace"] = True
    config_blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    config_hash = hashlib.sha256(config_blob.encode("utf-8")).hexdigest()

    captures = []
    original_trace = OnlineReconstructionPipeline._trace_init_stage

    def capture_trace(self, stage, tensors):
        if stage in (
            "init_step_0_render_color_gradient",
            "init_step_0_render_depth_gradient",
            "init_step_0_gradients",
            "init_step_0_updated_map",
        ):
            captures[-1][stage] = {
                key: value.detach().contiguous().cpu().clone()
                for key, value in tensors.items() if value is not None
            }
        return original_trace(self, stage, tensors)

    OnlineReconstructionPipeline._trace_init_stage = capture_trace
    try:
        for repeat_idx in range(args.repeats):
            captures.append({})
            set_seed(args.seed)
            pipeline = OnlineReconstructionPipeline(config=copy.deepcopy(config), device=args.device)
            pipeline.initialize(rgb=rgb, depth=depth, intrinsics=intrinsics, pose=pose)
            del pipeline
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        OnlineReconstructionPipeline._trace_init_stage = original_trace

    stage_names = (
        "init_step_0_render_color_gradient",
        "init_step_0_render_depth_gradient",
        "init_step_0_gradients",
        "init_step_0_updated_map",
    )
    comparisons = {}
    for stage in stage_names:
        tensor_names = sorted(set.intersection(*[
            set(capture.get(stage, {})) for capture in captures
        ]))
        comparisons[stage] = {}
        for tensor_name in tensor_names:
            reference = captures[0][stage][tensor_name].float()
            per_repeat = []
            for repeat_idx, capture in enumerate(captures[1:], start=1):
                current = capture[stage][tensor_name].float()
                diff = (current - reference).abs()
                per_repeat.append({
                    "repeat": repeat_idx,
                    "bitwise_equal": bool(torch.equal(current, reference)),
                    "different_elements": int((current != reference).sum().item()),
                    "num_elements": int(current.numel()),
                    "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
                    "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
                    "relative_l2_diff": float(
                        torch.linalg.vector_norm(diff) /
                        (torch.linalg.vector_norm(reference) + 1e-30)
                    ),
                    "diff_gt_1e-7": int((diff > 1e-7).sum().item()),
                    "diff_gt_1e-5": int((diff > 1e-5).sum().item()),
                })
            comparisons[stage][tensor_name] = per_repeat

    payload = {
        "protocol": "same-seed independent frame-0 initializations; compare first backward only",
        "scene": "tum_fr2_xyz",
        "seed": args.seed,
        "repeats": args.repeats,
        "device": str(dev),
        "strict_determinism": args.strict_determinism,
        "config_sha256": config_hash,
        "git_commit": get_git_commit(),
        "hardware": get_hardware_info(),
        "comparisons_vs_repeat_0": comparisons,
        "caveat": "This diagnostic uses gsplat backward and is not a quality or latency benchmark.",
    }
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gradient_repeatability.json", "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# gsplat first-backward repeatability diagnostic", "",
        f"Same seed: {args.seed}; repeats: {args.repeats}; strict mode: {args.strict_determinism}.",
        "Values compare each repeat to repeat 0.", "",
        "| Stage | Tensor | Repeat | Bitwise equal | Changed values | Max abs diff | Mean abs diff | Relative L2 |",
        "|:---|:---|---:|:---:|---:|---:|---:|---:|",
    ]
    for stage, tensors in comparisons.items():
        for tensor_name, repeat_values in tensors.items():
            for item in repeat_values:
                lines.append(
                    f"| {stage} | {tensor_name} | {item['repeat']} | {item['bitwise_equal']} | "
                    f"{item['different_elements']}/{item['num_elements']} | {item['max_abs_diff']:.9g} | "
                    f"{item['mean_abs_diff']:.9g} | {item['relative_l2_diff']:.9g} |"
                )
    lines.extend(["", "This diagnostic is not a quality or latency benchmark.", ""])
    with open(out_dir / "gradient_repeatability.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Saved: {out_dir / 'gradient_repeatability.json'}")


if __name__ == "__main__":
    main()
