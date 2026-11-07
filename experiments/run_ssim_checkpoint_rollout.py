#!/usr/bin/env python3
"""Paired direct/separable SSIM rollouts from one full replay checkpoint.

At each frame, every branch receives the same saved global RNG state. The
first direct branch defines the common RNG stream for the next frame. This
common-random-number protocol helps distinguish implementation effects from
random candidate sampling while each branch evolves its own map and optimizer.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_joint_spawn_refine_replay import _memoize_detached_nonleaf_tensors
from experiments.run_frozen_benchmark import build_pipeline_config, load_benchmark_sequence
from research.pipeline import OnlineReconstructionPipeline
from research.pipeline_checkpoint import load_pipeline_checkpoint, gaussian_map_digest
from research.reproducibility import get_git_commit


def capture_rng(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_digest(state):
    h = hashlib.sha256()
    h.update(state["torch"].cpu().numpy().tobytes())
    if state["cuda"] is not None:
        for item in state["cuda"]:
            h.update(item.cpu().numpy().tobytes())
    h.update(repr(state["python"]).encode())
    h.update(state["numpy"][1].tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=("replica_room1", "replica_office1"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--snapshot_frame", type=int, required=True)
    ap.add_argument("--rollout_frames", type=int, default=30)
    ap.add_argument("--checkpoint_dir", type=Path, required=True)
    ap.add_argument("--micro_updates_per_frame", type=int, default=5000)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--repeats_per_implementation", type=int, default=2)
    ap.add_argument("--freeze_scheduler_profiling", action="store_true",
                    help="Disable measured-latency feedback so scheduler costs remain fixed during rollout")
    ap.add_argument("--freeze_spawn_and_update_decisions", action="store_true",
                    help="Build a direct canonical per-frame persistent-ID/k schedule with zero spawn/prune, then replay it for both implementations")
    ap.add_argument("--expected_snapshot_sha256", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("This paired rollout requires the production CUDA/gsplat path")
    if args.rollout_frames <= 0 or args.repeats_per_implementation < 2:
        raise ValueError("rollout_frames must be positive and repeats_per_implementation >= 2")

    args.out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    frames, intrinsics = load_benchmark_sequence(
        scene_name=args.scene,
        n_frames=args.snapshot_frame + args.rollout_frames + 1,
        H=240, W=320, device=args.device)
    first = args.snapshot_frame + 1
    last = args.snapshot_frame + args.rollout_frames
    if last >= len(frames):
        raise ValueError(f"requested rollout through frame {last}, but only {len(frames)} frames loaded")

    config = build_pipeline_config("ours", args.seed, args.budget_ms,
                                   320, 240, args.device)
    config["system"]["synchronized_stage_timing"] = True
    config["training"]["optimization_control"] = {
        "micro_updates_per_frame": args.micro_updates_per_frame,
    }
    base = OnlineReconstructionPipeline(config=config, device=args.device)
    checkpoint = (args.checkpoint_dir / "snapshots" / args.scene /
                  f"seed{args.seed}_frame{args.snapshot_frame}.checkpoint.pt")
    metadata = load_pipeline_checkpoint(
        checkpoint, base, scene=args.scene, seed=args.seed,
        snapshot_frame=args.snapshot_frame, restore_rng=True)
    base_digest = gaussian_map_digest(base.gaussian_model)
    base_ids = base.gaussian_model.state_store.persistent_ids[
        :base.gaussian_model.num_gaussians].detach().contiguous().cpu()
    base_ids_sha256 = hashlib.sha256(base_ids.numpy().tobytes()).hexdigest()
    if base_digest != metadata["map_digest"]:
        raise RuntimeError("Loaded checkpoint map digest failed validation")
    if args.expected_snapshot_sha256 and base_digest != args.expected_snapshot_sha256:
        raise RuntimeError(f"Snapshot hash mismatch: {base_digest} != {args.expected_snapshot_sha256}")

    initial_rng = capture_rng(args.device)
    decision_schedule = None
    decision_schedule_sha256 = None
    freeze_profile = (args.freeze_scheduler_profiling
                      or args.freeze_spawn_and_update_decisions)
    if args.freeze_spawn_and_update_decisions:
        schedule_memo = {}
        _memoize_detached_nonleaf_tensors(base, schedule_memo)
        canonical = copy.deepcopy(base, schedule_memo)
        canonical.config["densification"]["max_new_per_frame"] = 0
        canonical.config["densification"]["disable_pruning"] = True
        canonical.config.setdefault("system", {})["disable_periodic_compaction"] = True
        canonical.config["losses"]["ssim_implementation"] = "direct"
        canonical.scheduler.adjust_budget_from_profiling = lambda *a, **kw: None
        decision_schedule = {}
        schedule_rng = initial_rng
        for frame_idx in range(first, last + 1):
            restore_rng(schedule_rng)
            frame = frames[frame_idx]
            metrics = canonical.process_frame(
                rgb=frame["rgb"], depth=frame["depth"], gt_pose=frame.get("pose"))
            canonical_n = int(canonical.gaussian_model.num_gaussians)
            canonical_ids = canonical.gaussian_model.state_store.persistent_ids[
                :canonical_n].detach().contiguous().cpu()
            canonical_ids_sha256 = hashlib.sha256(canonical_ids.numpy().tobytes()).hexdigest()
            if (int(metrics.get("n_spawned", 0)) != 0
                    or canonical_n != int(base.gaussian_model.num_gaussians)
                    or canonical_ids_sha256 != base_ids_sha256):
                raise RuntimeError(
                    f"Canonical map roster changed at frame {frame_idx}: "
                    f"count={canonical_n}, births={metrics.get('n_spawned')}"
                )
            allocations = canonical._last_k_alloc.detach().cpu()
            persistent_ids = canonical.gaussian_model.state_store.persistent_ids[
                :canonical_n].detach().cpu()
            if allocations.numel() != canonical_n or persistent_ids.numel() != canonical_n:
                raise RuntimeError(
                    f"Allocation/state identity length mismatch at frame {frame_idx}: "
                    f"allocations={allocations.numel()}, ids={persistent_ids.numel()}, map={canonical_n}"
                )
            entries = [
                {"persistent_id": int(pid), "k": int(k)}
                for pid, k in zip(persistent_ids.tolist(), allocations.tolist()) if int(k) > 0
            ]
            scheduled_updates = sum(entry["k"] for entry in entries)
            if scheduled_updates != int(metrics.get("n_micro_updates", 0)):
                raise RuntimeError(
                    f"Canonical schedule/update mismatch at frame {frame_idx}: "
                    f"{scheduled_updates} != {metrics.get('n_micro_updates')}"
                )
            decision_schedule[str(frame_idx)] = {
                "entries": entries,
                "n_selected_ids": len(entries),
                "n_micro_updates": scheduled_updates,
                "map_count": int(canonical.gaussian_model.num_gaussians),
            }
            schedule_rng = capture_rng(args.device)
            if int(metrics.get("n_spawned", 0)) != 0:
                raise RuntimeError(f"Canonical schedule unexpectedly spawned at frame {frame_idx}")
        canonical.cleanup()
        schedule_blob = json.dumps(decision_schedule, sort_keys=True, separators=(",", ":"))
        decision_schedule_sha256 = hashlib.sha256(schedule_blob.encode()).hexdigest()

    branches = []
    for implementation in ("direct", "separable"):
        for repeat in range(args.repeats_per_implementation):
            memo = {}
            _memoize_detached_nonleaf_tensors(base, memo)
            branch = copy.deepcopy(base, memo)
            branch.config.setdefault("losses", {})["ssim_implementation"] = implementation
            if freeze_profile:
                branch.scheduler.adjust_budget_from_profiling = lambda *a, **kw: None
            if args.freeze_spawn_and_update_decisions:
                branch.config["densification"]["max_new_per_frame"] = 0
                branch.config["densification"]["disable_pruning"] = True
                branch.config.setdefault("system", {})["disable_periodic_compaction"] = True
            branches.append({
                "implementation": implementation,
                "repeat": repeat,
                "name": f"{implementation}_repeat{repeat}",
                "pipeline": branch,
                "rows": [],
            })

    common_rng = initial_rng
    rng_frame_digests = []
    for frame_idx in range(first, last + 1):
        frame_rng = common_rng
        rng_frame_digests.append({"frame": frame_idx, "sha256": rng_digest(frame_rng)})
        next_rng = None
        frame = frames[frame_idx]
        for branch_idx, item in enumerate(branches):
            restore_rng(frame_rng)
            pipeline = item["pipeline"]
            if decision_schedule is not None:
                pipeline.config["training"]["optimization_control"][
                    "fixed_persistent_id_k_alloc"] = decision_schedule[str(frame_idx)]["entries"]
                pipeline.config["training"]["optimization_control"][
                    "fixed_persistent_id_k_alloc_strict"] = True
            metrics = pipeline.process_frame(
                rgb=frame["rgb"], depth=frame["depth"], gt_pose=frame.get("pose"))
            n_gaussians = int(pipeline.gaussian_model.num_gaussians)
            persistent_ids = pipeline.gaussian_model.state_store.persistent_ids[
                :n_gaussians].detach().contiguous().cpu()
            persistent_ids_sha256 = hashlib.sha256(persistent_ids.numpy().tobytes()).hexdigest()
            if decision_schedule is not None:
                expected_decision = decision_schedule[str(frame_idx)]
                actual_updates = int(metrics.get("n_micro_updates", 0))
                if actual_updates != int(expected_decision["n_micro_updates"]):
                    raise RuntimeError(
                        f"Fixed update-count mismatch for {item['name']} frame {frame_idx}: "
                        f"{actual_updates} != {expected_decision['n_micro_updates']}")
                if int(metrics.get("n_spawned", 0)) != 0:
                    raise RuntimeError(f"Unexpected Gaussian spawn in fixed-decision replay at frame {frame_idx}")
                if n_gaussians != int(expected_decision["map_count"]):
                    raise RuntimeError(f"Map size changed in fixed-decision replay at frame {frame_idx}")
                if persistent_ids_sha256 != base_ids_sha256:
                    raise RuntimeError(f"Persistent-ID roster changed in fixed-decision replay at frame {frame_idx}")
            if branch_idx == 0:
                next_rng = capture_rng(args.device)
            item["rows"].append({
                "frame": int(frame_idx),
                "psnr": float(metrics.get("psnr_post", metrics["psnr"])),
                "ssim": float(metrics["ssim"]),
                "depth_l1": float(metrics["depth_l1"]),
                "depth_l1_pre": float(metrics.get("depth_l1_pre", metrics["depth_l1"])),
                "n_spawned": int(metrics.get("n_spawned", 0)),
                "n_gaussians": n_gaussians,
                "persistent_ids_sha256": persistent_ids_sha256,
                "n_micro_updates": int(metrics.get("n_micro_updates", 0)),
                "optimizer_iteration_count": int(metrics.get("optimizer_iteration_count", 0)),
                "mature_update_coverage": float(metrics.get("mature_update_coverage", 0.0)),
                "growth_pressure": float(metrics.get("growth_pressure", 0.0)),
                "opt_stage_cuda_event_ms": float(metrics.get("opt_stage_cuda_event_ms", 0.0)),
                "frame_time_ms": float(metrics.get("frame_time_ms", 0.0)),
                "map_digest": gaussian_map_digest(pipeline.gaussian_model),
            })
        if next_rng is None:
            raise RuntimeError("No canonical direct branch produced the next RNG state")
        common_rng = next_rng
        if frame_idx % 10 == 0 or frame_idx == last:
            print(f"ROLL OUT frame={frame_idx}/{last} completed for {len(branches)} branches", flush=True)

    result = {
        "schema": "paired-ssim-checkpoint-rollout-v1",
        "scene": args.scene, "seed": int(args.seed),
        "snapshot_frame": int(args.snapshot_frame),
        "rollout_start_frame": int(first), "rollout_end_frame": int(last),
        "rollout_frames": int(args.rollout_frames),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "snapshot_map_sha256": base_digest,
        "initial_persistent_ids_sha256": base_ids_sha256,
        "renderer_backend": "gsplat", "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "git_commit": get_git_commit(),
        "rng_protocol": "Restore the same per-frame RNG state before each branch; first direct repeat defines the next frame RNG state.",
        "scheduler_profiling_feedback_frozen": bool(freeze_profile),
        "spawn_and_update_decisions_frozen": bool(args.freeze_spawn_and_update_decisions),
        "decision_schedule_sha256": decision_schedule_sha256,
        "fixed_decision_gates_passed": (None if decision_schedule is None else True),
        "decision_schedule": decision_schedule,
        "per_frame_rng_digests": rng_frame_digests,
        "branches": [{
            "implementation": item["implementation"],
            "repeat": item["repeat"],
            "name": item["name"],
            "final_map_sha256": item["rows"][-1]["map_digest"],
            "frames": item["rows"],
        } for item in branches],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "paired_rollout.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"out": str(out_path), "snapshot_map_sha256": base_digest,
                      "frames": args.rollout_frames, "branches": len(branches),
                      "decision_schedule_sha256": decision_schedule_sha256}, indent=2), flush=True)
    for item in branches:
        item["pipeline"].cleanup()
    base.cleanup()


if __name__ == "__main__":
    main()
