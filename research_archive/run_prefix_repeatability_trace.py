#!/usr/bin/env python3
"""Capture frame/stage hashes for one exact-seed common mapping prefix."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_dilution_factorial import config_for
from experiments.run_phase13_frozen_benchmark import load_phase10_sequence
from research.pipeline import OnlineReconstructionPipeline
from research.reproducibility import create_provenance_manifest, set_seed


def digest_map(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    for name, value in sorted(vars(model.state_store).items()):
        if torch.is_tensor(value):
            h.update(name.encode())
            h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def rng_digest():
    h = hashlib.sha256()
    h.update(repr(random.getstate()).encode())
    np_state = np.random.get_state()
    h.update(np_state[0].encode())
    h.update(np_state[1].tobytes())
    h.update(repr(np_state[2:]).encode())
    h.update(torch.random.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="tum_fr1_rpy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snapshot_frame", type=int, default=90)
    ap.add_argument("--n_frames", type=int, default=130,
                    help="Match the factorial runner's n_frames argument; loader receives n_frames+1")
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--deterministic_algorithms", action="store_true",
                    help="Enable torch deterministic algorithms for this diagnostic run")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    frames, intrinsics = load_phase10_sequence(
        scene_name=args.scene, n_frames=args.n_frames + 1,
        H=240, W=320, device=args.device)
    if len(frames) <= args.snapshot_frame:
        raise ValueError(f"snapshot {args.snapshot_frame} outside loaded sequence ({len(frames)} frames)")
    set_seed(args.seed)
    _, cfg = config_for("error_off", args.seed, args.budget_ms, args.device)
    cfg["system"]["synchronized_stage_timing"] = True
    cfg["system"]["repro_trace"] = True
    cfg["system"]["repro_trace_summaries"] = True
    cfg["system"]["repro_trace_dump_init_gradients"] = True
    pipeline = OnlineReconstructionPipeline(config=cfg, device=args.device)
    pipeline.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
        intrinsics=intrinsics,
        pose=frames[0].get("pose", torch.eye(4, device=args.device)))
    gradient_dump_path = out / "init_step_0_gradients.npz"
    if pipeline._init_gradient_snapshots:
        np.savez_compressed(gradient_dump_path, **{
            name: value.numpy() for name, value in pipeline._init_gradient_snapshots.items()})
    init_trace = pipeline._init_repro_trace
    frame_rows = []
    for idx in range(1, args.snapshot_frame):
        frame = frames[idx]
        metrics = pipeline.process_frame(
            rgb=frame["rgb"], depth=frame["depth"], gt_pose=frame.get("pose"))
        row = {
            "input_frame_index": idx,
            "pipeline_frame": int(metrics["frame"]),
            "n_gaussians": int(pipeline.gaussian_model.num_gaussians),
            "n_spawned": int(metrics.get("n_spawned", 0)),
            "n_micro_updates": int(metrics.get("n_micro_updates", 0)),
            "spawn_control_count": int(pipeline._spawn_control_count),
            "map_digest_after_frame": digest_map(pipeline.gaussian_model),
            "rng_digest_after_frame": rng_digest(),
            "repro_trace": metrics.get("repro_trace", []),
        }
        frame_rows.append(row)
        if idx == 1 or idx % 15 == 0 or idx == args.snapshot_frame - 1:
            print(f"FRAME {idx}/{args.snapshot_frame - 1}: N={row['n_gaussians']} "
                  f"spawned={row['n_spawned']} updates={row['n_micro_updates']}", flush=True)
    record = {
        "schema": "prefix-repeatability-trace-v1",
        "scene": args.scene, "seed": args.seed,
        "snapshot_frame": args.snapshot_frame,
        "prefix_policy": "error_only",
        "pose_convention": "ground-truth W2C via Phase 10 loader",
        "device": args.device,
        "torch_deterministic_algorithms": args.deterministic_algorithms,
        "renderer_backend": cfg.get("rendering", {}).get("backend", "gsplat"),
        "config": {"budget_ms": args.budget_ms,
                   "synchronized_stage_timing": True,
                   "repro_trace": True,
                   "repro_trace_summaries": True,
                   "repro_trace_dump_init_gradients": True,
                   "runner_n_frames": args.n_frames,
                   "n_frames_loaded": len(frames)},
        "init_repro_trace": init_trace,
        "final_map_digest": digest_map(pipeline.gaussian_model),
        "init_gradient_dump": gradient_dump_path.name if gradient_dump_path.exists() else None,
        "frames": frame_rows,
    }
    manifest = create_provenance_manifest(
        config={"scene": args.scene, "seed": args.seed,
                "snapshot_frame": args.snapshot_frame,
                "n_frames": args.n_frames,
                "budget_ms": args.budget_ms,
                "prefix_policy": "error_only",
                "repro_trace": True,
                "repro_trace_summaries": True,
                "repro_trace_dump_init_gradients": True,
                "hash_map_each_frame": True},
        dataset_name=args.scene,
        renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
        extra_metadata={"experiment": "same_seed_prefix_repeatability_trace",
                        "status": "exploratory_determinism_audit",
                        "pose_convention": "ground-truth W2C via Phase 10 loader"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "prefix_trace.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"FINAL {record['final_map_digest']} N={pipeline.gaussian_model.num_gaussians}", flush=True)
    pipeline.cleanup()


if __name__ == "__main__":
    main()
