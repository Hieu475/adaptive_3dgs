#!/usr/bin/env python3
"""Same-snapshot spawn-dose x mature-update-quota replay.

Each snapshot is produced once by an error-only prefix. Deep-copied pipelines
process the same next RGB-D frame with nested spawn doses crossed with a
low/high mature-update quota. RNG state and candidate ordering are shared.
"""
import argparse
import atexit
import copy
import hashlib
import json
import random
import sys
import contextlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_dilution_factorial import config_for
from experiments.run_phase13_frozen_benchmark import build_pipeline_config, load_phase10_sequence
from research import pipeline as pipeline_module
from research.pipeline import OnlineReconstructionPipeline
from research.reproducibility import create_provenance_manifest, set_seed
from research.map_snapshots import validate_map_snapshot
from research.pipeline_checkpoint import (
    load_pipeline_checkpoint, save_pipeline_checkpoint,
    gaussian_map_digest,
)
from research.gpu_telemetry import GPUTelemetrySampler


def digest_map(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    store = model.state_store
    for name, value in sorted(vars(store).items()):
        if torch.is_tensor(value):
            h.update(name.encode())
            h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def save_map_snapshot(path, pipeline, scene, seed, snapshot_frame, map_hash):
    """Persist Gaussian parameters and identity/age state for replay provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state_store = {}
    for name, value in vars(pipeline.gaussian_model.state_store).items():
        if torch.is_tensor(value):
            state_store[name] = value.detach().contiguous().cpu()
        else:
            state_store[name] = copy.deepcopy(value)
    payload = {
        "schema": "gaussian-map-snapshot-v1",
        "scene": scene,
        "seed": int(seed),
        "snapshot_frame": int(snapshot_frame),
        "map_digest": map_hash,
        "gaussian_model_state_dict": {
            name: value.detach().contiguous().cpu()
            for name, value in pipeline.gaussian_model.state_dict().items()
        },
        "gaussian_state_store": state_store,
        "prefix_state": {
            "frame_count": int(pipeline.frame_count),
            "spawn_control_count": int(pipeline._spawn_control_count),
            "current_pose": (pipeline.current_pose.detach().contiguous().cpu()
                             if torch.is_tensor(pipeline.current_pose) else None),
        },
    }
    torch.save(payload, path)


def _memoize_detached_nonleaf_tensors(value, memo, seen=None):
    """Make pipeline deepcopy work with cached autograd tensors in map state."""
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return
    seen.add(obj_id)
    if torch.is_tensor(value):
        if not value.is_leaf:
            memo[obj_id] = value.detach().clone()
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _memoize_detached_nonleaf_tensors(key, memo, seen)
            _memoize_detached_nonleaf_tensors(item, memo, seen)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _memoize_detached_nonleaf_tensors(item, memo, seen)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        _memoize_detached_nonleaf_tensors(vars(value), memo, seen)


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


def run_branch(base, rng_state, scene, seed, snapshot_frame, action,
              frame, intrinsics, args, common_pool, eval_views=None):
    telemetry_enabled = bool(action.get("gpu_telemetry_enabled", args.gpu_telemetry))
    if args._gpu_telemetry is not None:
        if telemetry_enabled:
            args._gpu_telemetry.resume()
        else:
            args._gpu_telemetry.pause()
    restore_rng(rng_state)
    # Cached state-store tensors can retain an autograd history even though the
    # next frame is a fresh action. Clone those values detached; leaf Parameters
    # and buffers still follow normal deepcopy semantics.
    deepcopy_memo = {}
    _memoize_detached_nonleaf_tensors(base, deepcopy_memo)
    branch = copy.deepcopy(base, deepcopy_memo)
    start_hash = digest_map(branch.gaussian_model)
    if start_hash != digest_map(base.gaussian_model):
        raise RuntimeError("Deep-copied branch does not match the common map snapshot")

    spawn_count = int(action["spawn_count"])
    spawn_count_before = int(branch._spawn_control_count)
    branch.config["densification"]["spawn_control"] = {
        # The pipeline's counter includes births from the common prefix.
        # Express the intervention as an additional allowance after that base.
        "target_total": spawn_count_before + spawn_count,
        "start_frame": snapshot_frame,
        "end_frame": snapshot_frame,
        "candidate_mode": "uniform_depth",
    }
    optimization_control = {
        "micro_updates_per_frame": int(action.get("micro_updates", args.micro_updates)),
        "selection": "descending importance within age group",
        "mature_age_frames": 10,
        "mature_update_fraction": action["mature_fraction"],
        # Read-only summaries from the shared pre-spawn attribution pass.
        "capture_age_diagnostics": True,
    }
    if action["mature_fraction"] == "coverage_saturating":
        optimization_control["mature_update_fraction_cap"] = float(
            action.get("mature_fraction_cap", 0.75)
        )
    branch.config["training"]["optimization_control"] = optimization_control
    if args.loss_ssim_weight_override is not None:
        branch.config.setdefault("losses", {})["weight_ssim"] = float(
            args.loss_ssim_weight_override
        )
    ssim_implementation = action.get("ssim_implementation", args.ssim_implementation)
    if ssim_implementation in ("direct", "separable"):
        branch.config.setdefault("losses", {})["ssim_implementation"] = ssim_implementation

    captured_first_backward = {}
    capture_gradient_this_branch = (
        args.capture_first_backward_gradients
        and action.get("capture_first_backward_gradient", True))
    if capture_gradient_this_branch:
        original_optimizer_step = branch.optimizer.step

        def capture_then_step(mask=None, active_idx=None):
            if not captured_first_backward:
                if active_idx is None:
                    indices = (torch.where(mask)[0] if mask is not None else
                               torch.arange(branch.gaussian_model.num_gaussians,
                                            device=args.device))
                else:
                    indices = active_idx
                stored_indices = indices.detach().contiguous().cpu()
                parameter_gradients = {}
                for parameter_name in ("_xyz", "_scaling", "_rotation", "_opacity",
                                       "_features_dc", "_features_rest"):
                    parameter = getattr(branch.gaussian_model, parameter_name)
                    grad = parameter.grad
                    if grad is None:
                        parameter_gradients[parameter_name] = None
                        continue
                    valid_indices = indices[indices < parameter.shape[0]]
                    parameter_gradients[parameter_name] = grad.index_select(
                        0, valid_indices).detach().contiguous().cpu()
                captured_first_backward.update({
                    "indices": stored_indices,
                    "parameter_gradients": parameter_gradients,
                })
            return original_optimizer_step(mask=mask, active_idx=active_idx)

        branch.optimizer.step = capture_then_step

    sampled = []
    frame_count_before = int(branch.frame_count)
    original_sampler = pipeline_module.sample_candidates
    def recording_sampler(*call_args, **call_kwargs):
        requested = int(call_kwargs.get("num_samples", call_args[1] if len(call_args) > 1 else 0))
        candidates = common_pool[:requested].to(frame["depth"].device)
        if len(candidates) != requested:
            raise RuntimeError(f"common candidate pool too small: requested {requested}, got {len(candidates)}")
        sampled.append(candidates.detach().cpu().tolist())
        return candidates
    pipeline_module.sample_candidates = recording_sampler
    profiler = None
    profile_this_branch = (
        args.profile_dir is not None
        and not args._profiler_captured
        and scene == args.profile_scene
        and seed == args.profile_seed
        and snapshot_frame == args.profile_frame
        and action["name"] == args.profile_action
    )
    try:
        if profile_this_branch:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if str(args.device).startswith("cuda"):
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            profiler.__enter__()
        metrics = branch.process_frame(
            rgb=frame["rgb"], depth=frame["depth"], gt_pose=frame.get("pose"))
        if profiler is not None:
            profiler.__exit__(None, None, None)
            args._profiler_captured = True
            profile_path = args.profile_dir / scene / (
                f"seed{seed}_frame{snapshot_frame}_{action['name']}.trace.json")
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(profile_path))
            summary_path = profile_path.with_suffix(".summary.txt")
            sort_key = ("cuda_time_total" if str(args.device).startswith("cuda")
                        else "self_cpu_time_total")
            summary_path.write_text(
                f"Profiled branch: {scene} seed={seed} frame={snapshot_frame} "
                f"action={action['name']}\n"
                f"snapshot_sha256={start_hash}\n"
                f"gaussians_after_action={branch.gaussian_model.num_gaussians}\n"
                f"torch={torch.__version__} device={args.device}\n"
                "Trace includes process_frame only; profiler timing is diagnostic, "
                "not benchmark timing.\n\n"
                + profiler.key_averages(group_by_input_shape=True).table(
                    sort_by=sort_key, row_limit=100)
            )
            print(f"PROFILE_TRACE {profile_path}", flush=True)
        # process_frame's depth_l1 uses depth_err computed before optimizer
        # updates. Re-render the final map at the same GT view for a true
        # post-action geometric diagnostic; eval_frame is read-only.
        pose = frame.get("pose", branch.current_pose)
        final_map_eval = branch.eval_frame(
            rgb=frame["rgb"], depth=frame["depth"], pose=pose)
        fixed_view_evals = []
        for eval_frame_idx, eval_frame in (eval_views or []):
            eval_pose = eval_frame.get("pose", branch.current_pose)
            eval_result = branch.eval_frame(
                rgb=eval_frame["rgb"], depth=eval_frame["depth"], pose=eval_pose)
            fixed_view_evals.append({
                "frame": int(eval_frame_idx),
                "psnr": float(eval_result["psnr"]),
                "ssim": float(eval_result["ssim"]),
                "depth_l1": float(eval_result["depth_l1"]),
                "n_gaussians": int(eval_result["n_gaussians"]),
            })
    finally:
        pipeline_module.sample_candidates = original_sampler

    gradient_record = None
    if capture_gradient_this_branch:
        if not captured_first_backward:
            raise RuntimeError("Gradient capture requested, but this branch made no optimizer update")
        gradient_path = (args.out_dir / "first_backward_gradients" / scene /
                         f"seed{seed}_frame{snapshot_frame}_{action['name']}.pt")
        gradient_path.parent.mkdir(parents=True, exist_ok=True)
        gradient_payload = {
            "schema": "first-active-subset-gradient-v1",
            "scene": scene, "seed": int(seed),
            "snapshot_frame": int(snapshot_frame),
            "action": action,
            "snapshot_hash": start_hash,
            "indices": captured_first_backward["indices"],
            "parameter_gradients": captured_first_backward["parameter_gradients"],
        }
        torch.save(gradient_payload, gradient_path)
        gradient_bytes = gradient_path.read_bytes()
        gradient_record = {
            "path": str(gradient_path.relative_to(args.out_dir)),
            "sha256": hashlib.sha256(gradient_bytes).hexdigest(),
            "active_index_sha256": hashlib.sha256(
                captured_first_backward["indices"].numpy().tobytes()).hexdigest(),
            "active_gaussians": int(captured_first_backward["indices"].numel()),
            "parameter_shapes": {
                name: (None if tensor is None else list(tensor.shape))
                for name, tensor in captured_first_backward["parameter_gradients"].items()
            },
            "ssim_implementation": ssim_implementation or "direct",
        }

    realized_updates = int(metrics.get("n_micro_updates", -1))
    realized_spawn = int(metrics.get("n_spawned", 0))
    target_updates = int(action.get("micro_updates", args.micro_updates))
    if action["mature_fraction"] == "coverage_saturating":
        quota_cap = min(1.0, max(0.0, float(action.get("mature_fraction_cap", 0.75))))
        expected_mature_updates = min(
            int(round(target_updates * quota_cap)),
            int(metrics.get("mature_eligible", 0)),
        )
        requested_mature_fraction = expected_mature_updates / max(target_updates, 1)
    else:
        expected_mature_updates = int(round(target_updates * action["mature_fraction"]))
        requested_mature_fraction = float(action["mature_fraction"])
    expected_young_updates = target_updates - expected_mature_updates
    branch_record = {
        "action": action,
        "snapshot_hash": start_hash,
        "frame": int(metrics["frame"]),
        "frame_count_before": frame_count_before,
        "spawn_quota_this_frame": metrics.get("spawn_quota_this_frame"),
        "pre_psnr": float(metrics.get("psnr_pre", metrics["psnr"])),
        "post_psnr": float(metrics.get("psnr_post", metrics["psnr"])),
        "final_map_eval_psnr": float(final_map_eval["psnr"]),
        "post_ssim": float(metrics["ssim"]),
        "pipeline_depth_l1": float(metrics["depth_l1"]),
        "pipeline_depth_l1_pre": float(metrics.get("depth_l1_pre", metrics["depth_l1"])),
        "final_map_eval_depth_l1": float(final_map_eval["depth_l1"]),
        "fixed_view_evals": fixed_view_evals,
        "n_spawned": realized_spawn,
        "n_gaussians_after_action": int(branch.gaussian_model.num_gaussians),
        "n_active_gaussians_after_action": int(metrics.get("n_active_gaussians", 0)),
        "n_optimized": int(metrics.get("n_optimized", 0)),
        "n_cache_rebuilds": int(metrics.get("n_cache_rebuilds", 0)),
        "n_cache_frozen_gaussians": int(metrics.get("n_cache_frozen_gaussians", 0)),
        "selected_young": int(metrics.get("selected_young", 0)),
        "selected_mature": int(metrics.get("selected_old", 0)),
        "mean_k": float(metrics.get("mean_k", 0.0)),
        "n_visible": int(metrics.get("n_visible", 0)),
        "peak_vram_mb": float(metrics.get("peak_vram_mb", 0.0)),
        "target_spawned": spawn_count,
        "spawn_count_before_action": spawn_count_before,
        "n_micro_updates": realized_updates,
        "target_micro_updates": target_updates,
        "mature_eligible": int(metrics.get("mature_eligible", 0)),
        "mature_updated": int(metrics.get("mature_updated", 0)),
        "mature_update_coverage": float(metrics.get("mature_update_coverage", 0.0)),
        "quota_policy": ("coverage_saturating" if action["mature_fraction"] == "coverage_saturating"
                         else "fixed_fraction"),
        "loss_ssim_weight": float(branch.config.get("losses", {}).get("weight_ssim", 0.2)),
        "ssim_implementation": branch.config.get("losses", {}).get("ssim_implementation", "direct"),
        "first_backward_gradient": gradient_record,
        "requested_mature_fraction": requested_mature_fraction,
        "realized_mature_fraction": (int(metrics.get("age_quota_mature_micro_updates") or 0)
                                      / max(realized_updates, 1)),
        "realized_mature_micro_updates": int(metrics.get("age_quota_mature_micro_updates") or 0),
        "realized_young_micro_updates": int(metrics.get("age_quota_young_micro_updates") or 0),
        "growth_pressure": float(metrics.get("growth_pressure", 0.0)),
        "compute_return_db_per_opt_ms": float(metrics.get("compute_return_db_per_opt_ms", 0.0)),
        "opt_stage_sync_ms": float(metrics.get("opt_stage_sync_ms", 0.0)),
        "opt_stage_cuda_event_ms": float(metrics.get("opt_stage_cuda_event_ms", 0.0)),
        "cache_time_host_ms": float(metrics.get("cache_time_ms", 0.0)),
        "opt_time_host_ms": float(metrics.get("opt_time_ms", 0.0)),
        "cache_cuda_event_ms": float(metrics.get("cache_cuda_event_ms", 0.0)),
        "noncache_cuda_event_ms": max(0.0, float(metrics.get("opt_stage_cuda_event_ms", 0.0))
                                        - float(metrics.get("cache_cuda_event_ms", 0.0))),
        "optimizer_iteration_count": int(metrics.get("optimizer_iteration_count", 0)),
        "optimizer_forward_cuda_event_ms": float(metrics.get("optimizer_forward_cuda_event_ms", 0.0)),
        "optimizer_backward_cuda_event_ms": float(metrics.get("optimizer_backward_cuda_event_ms", 0.0)),
        "optimizer_backward_only_cuda_event_ms": float(metrics.get("optimizer_backward_only_cuda_event_ms", 0.0)),
        "optimizer_step_cuda_event_ms": float(metrics.get("optimizer_step_cuda_event_ms", 0.0)),
        "opt_stage_start_unix_ns": metrics.get("opt_stage_start_unix_ns"),
        "opt_stage_end_unix_ns": metrics.get("opt_stage_end_unix_ns"),
        "gpu_telemetry_enabled": telemetry_enabled,
        "gpu_telemetry": (args._gpu_telemetry.window(
            metrics.get("opt_stage_start_unix_ns"), metrics.get("opt_stage_end_unix_ns"))
            if args._gpu_telemetry is not None and telemetry_enabled else None),
        "frame_time_ms": float(metrics.get("frame_time_ms", 0.0)),
        "age_stratified_diagnostics": metrics.get("age_stratified_diagnostics"),
        "spawn_candidates_uv": sampled[-1] if spawn_count and sampled else [],
        "valid": (realized_spawn == spawn_count and realized_updates == target_updates
                  and int(metrics.get("age_quota_mature_micro_updates") or 0) == expected_mature_updates
                  and int(metrics.get("age_quota_young_micro_updates") or 0) == expected_young_updates),
    }
    branch.cleanup()
    del branch
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return branch_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum_fr2_xyz", "tum_fr1_xyz"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--snapshot_frames", nargs="+", type=int, default=[30, 60])
    ap.add_argument("--eval_frames", nargs="*", type=int, default=[],
                    help="Optional fixed future frames scored read-only after each replay action")
    ap.add_argument("--prefix_policy", choices=("error_only", "ours"), default="error_only",
                    help="Mapping policy used to create the common prefix snapshot")
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--micro_updates", type=int, default=5000)
    ap.add_argument("--loss_ssim_weight_override", type=float, default=None,
                    help="Diagnostic intervention: override SSIM loss weight on each replay branch")
    ap.add_argument("--ssim_implementation", choices=("direct", "separable", "both"), default=None,
                    help="SSIM implementation for replay branches; 'both' pairs direct and separable arms")
    ap.add_argument("--capture_first_backward_gradients", action="store_true",
                    help="Save gradients for the first active-subset optimizer step of each branch")
    ap.add_argument("--spawn_counts", nargs="+", type=int, default=[0, 128, 256])
    ap.add_argument("--mature_fractions", nargs="+", type=float, default=[0.25, 0.75])
    ap.add_argument("--coverage_saturating_arm", action="store_true",
                    help="Add a policy that gives at most one update to each visible mature Gaussian, capped by a fixed mature-budget fraction")
    ap.add_argument("--coverage_saturating_cap", type=float, default=0.75)
    ap.add_argument("--mature_fraction_update_counts", nargs="*", default=[],
                    help="Optional FRACTION:UPDATES pairs, e.g. 0.25:5000 0.75:4800; useful for pilot matched-time calibration")
    ap.add_argument("--action_update_counts", nargs="*", default=[],
                    help="Optional SPAWN:FRACTION:UPDATES overrides, e.g. 0:0.75:4189 128:0.75:4374")
    ap.add_argument("--warmup_updates", type=int, default=0,
                    help="Run and discard one representative same-snapshot action before timing measured arms")
    ap.add_argument("--action_order_seed", type=int, default=0,
                    help="Seed for reproducibly shuffling action execution order per snapshot")
    ap.add_argument("--repeat_actions", type=int, default=1,
                    help="Repeat every same-snapshot action this many times, with unique record names and randomized order")
    ap.add_argument("--torch_num_threads", type=int, default=None,
                    help="Optional PyTorch intra-op CPU thread cap for timing diagnostics")
    ap.add_argument("--torch_num_interop_threads", type=int, default=None,
                    help="Optional PyTorch inter-op CPU thread cap for timing diagnostics")
    ap.add_argument("--gpu_telemetry", action="store_true",
                    help="Sample per-device NVML utilization, clocks, temperature, memory, and power during replay")
    ap.add_argument("--gpu_telemetry_interval_ms", type=float, default=20.0,
                    help="Background NVML sampling interval when --gpu_telemetry is enabled")
    ap.add_argument("--gpu_telemetry_interleave", action="store_true",
                    help="Duplicate each quota repeat with telemetry on/off, then shuffle all branches")
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "joint_spawn_refine_replay")
    ap.add_argument("--save_map_snapshots", action="store_true",
                    help="Persist CPU Gaussian parameters and identity/age state for every common replay snapshot")
    ap.add_argument("--save_replay_checkpoints", action="store_true",
                    help="Also save complete pipeline/optimizer/RNG checkpoints for independent replay")
    ap.add_argument("--replay_checkpoint_dir", type=Path, default=None,
                    help="Load checkpoints from this experiment root instead of rebuilding the common prefix")
    ap.add_argument("--expected_snapshot_sha256", default=None,
                    help="Require the generated/loaded common snapshot to match this map-state digest")
    ap.add_argument("--profile_dir", type=Path, default=None,
                    help="Capture one PyTorch operator trace for the exact selected replay branch")
    ap.add_argument("--profile_scene", default=None)
    ap.add_argument("--profile_seed", type=int, default=None)
    ap.add_argument("--profile_frame", type=int, default=None)
    ap.add_argument("--profile_action", default=None,
                    help="Exact arm name, e.g. spawn0_mature25; profile target must match all selectors")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    profile_selectors = (args.profile_scene, args.profile_seed,
                         args.profile_frame, args.profile_action)
    if args.profile_dir is not None and any(x is None for x in profile_selectors):
        ap.error("--profile_dir requires --profile_scene, --profile_seed, --profile_frame, and --profile_action")
    if args.profile_dir is None and any(x is not None for x in profile_selectors):
        ap.error("profile selectors require --profile_dir")
    if args.torch_num_threads is not None:
        if args.torch_num_threads < 1:
            raise ValueError("torch_num_threads must be positive")
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads is not None:
        if args.torch_num_interop_threads < 1:
            raise ValueError("torch_num_interop_threads must be positive")
        torch.set_num_interop_threads(args.torch_num_interop_threads)
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir if args.profile_dir.is_absolute() else ROOT / args.profile_dir
    args._profiler_captured = False
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    if any(x < 0 for x in args.spawn_counts) or sorted(set(args.spawn_counts)) != args.spawn_counts:
        raise ValueError("spawn_counts must be unique, nonnegative, and increasing")
    if args.loss_ssim_weight_override is not None and args.loss_ssim_weight_override < 0:
        raise ValueError("loss_ssim_weight_override must be nonnegative")
    if max(args.spawn_counts) < 1:
        raise ValueError("spawn_counts must include a positive dose")
    if any(not 0.0 <= x <= 1.0 for x in args.mature_fractions):
        raise ValueError("mature_fractions must lie in [0, 1]")
    if not 0.0 <= args.coverage_saturating_cap <= 1.0:
        raise ValueError("coverage_saturating_cap must lie in [0, 1]")
    update_counts = {float(item.split(":", 1)[0]): int(item.split(":", 1)[1])
                     for item in args.mature_fraction_update_counts}
    if any(n <= 0 for n in update_counts.values()):
        raise ValueError("arm update counts must be positive")
    if not set(update_counts).issubset(set(args.mature_fractions)):
        raise ValueError("arm update counts reference a mature fraction not in --mature_fractions")
    action_update_counts = {}
    for item in args.action_update_counts:
        spawn_s, fraction_s, updates_s = item.split(":", 2)
        key = (int(spawn_s), float(fraction_s))
        if key in action_update_counts:
            raise ValueError(f"duplicate action update override: {item}")
        action_update_counts[key] = int(updates_s)
    if any(n <= 0 for n in action_update_counts.values()):
        raise ValueError("action update overrides must be positive")
    if any(spawn not in args.spawn_counts or fraction not in args.mature_fractions
           for spawn, fraction in action_update_counts):
        raise ValueError("action update override references a spawn dose or mature fraction not in this run")
    if args.repeat_actions < 1:
        raise ValueError("repeat_actions must be at least one")
    if args.capture_first_backward_gradients and args.ssim_implementation != "both":
        raise ValueError("gradient capture requires --ssim_implementation both for paired comparisons")
    if args.gpu_telemetry and not args.device.startswith("cuda"):
        raise ValueError("--gpu_telemetry requires a CUDA device")
    if args.gpu_telemetry_interleave and not args.gpu_telemetry:
        raise ValueError("--gpu_telemetry_interleave requires --gpu_telemetry")
    if args.gpu_telemetry_interval_ms <= 0:
        raise ValueError("gpu_telemetry_interval_ms must be positive")
    base_arms = [{"spawn_count": dose, "mature_fraction": frac,
                  "micro_updates": action_update_counts.get(
                      (dose, frac), update_counts.get(frac, args.micro_updates)),
                  "name": f"spawn{dose}_mature{int(round(frac * 100))}"}
                 for frac in args.mature_fractions for dose in args.spawn_counts]
    if args.coverage_saturating_arm:
        for dose in args.spawn_counts:
            base_arms.append({
                "spawn_count": dose,
                "mature_fraction": "coverage_saturating",
                "mature_fraction_cap": args.coverage_saturating_cap,
                "micro_updates": args.micro_updates,
                "name": f"spawn{dose}_coverage_saturating",
            })
    arms = []
    implementation_pairs = (["direct", "separable"]
                             if args.ssim_implementation == "both"
                             else [args.ssim_implementation])
    for repeat_idx in range(args.repeat_actions):
        for action in base_arms:
            modes = ([(False, "telemetry_off"), (True, "telemetry_on")]
                     if args.gpu_telemetry_interleave else [(args.gpu_telemetry, None)])
            for enabled, suffix in modes:
                for implementation in implementation_pairs:
                    item = dict(action)
                    item["repeat_index"] = repeat_idx
                    item["gpu_telemetry_enabled"] = enabled
                    if args.repeat_actions > 1:
                        item["name"] = f"{action['name']}_repeat{repeat_idx}"
                    if suffix is not None:
                        item["name"] = f"{item['name']}_{suffix}"
                    item["paired_action_name"] = item["name"]
                    if args.ssim_implementation == "both":
                        item["ssim_implementation"] = implementation
                        item["name"] = f"{item['name']}_{implementation}"
                    arms.append(item)
    manifest = create_provenance_manifest(
        config={"scenes": args.scenes, "seeds": args.seeds,
                "snapshot_frames": args.snapshot_frames, "n_frames": args.n_frames,
                "fixed_eval_frames": args.eval_frames,
                "micro_updates_per_action": args.micro_updates,
                "loss_ssim_weight_override": args.loss_ssim_weight_override,
                "ssim_implementation": args.ssim_implementation,
                "capture_first_backward_gradients": args.capture_first_backward_gradients,
                "mature_fraction_update_counts": update_counts,
                "action_update_counts": {f"{spawn}:{fraction}": count
                                          for (spawn, fraction), count in action_update_counts.items()},
                "warmup_updates": args.warmup_updates,
                "action_order_seed": args.action_order_seed,
                "repeat_actions": args.repeat_actions,
                "coverage_saturating_arm": args.coverage_saturating_arm,
                "coverage_saturating_cap": args.coverage_saturating_cap,
                "torch_num_threads": args.torch_num_threads,
                "torch_num_interop_threads": args.torch_num_interop_threads,
                "gpu_telemetry": args.gpu_telemetry,
                "gpu_telemetry_interval_ms": (args.gpu_telemetry_interval_ms
                                               if args.gpu_telemetry else None),
                "gpu_telemetry_interleave": args.gpu_telemetry_interleave,
                "mature_fraction_update_counts": update_counts,
                "prefix_micro_updates_per_frame": (args.micro_updates if args.prefix_policy == "ours" else None),
                "prefix_policy": args.prefix_policy,
                "save_map_snapshots": args.save_map_snapshots,
                "save_replay_checkpoints": args.save_replay_checkpoints,
                "replay_checkpoint_dir": (str(args.replay_checkpoint_dir) if args.replay_checkpoint_dir else None),
                "spawn_doses": args.spawn_counts,
                "arms": arms, "prefix": "error_only, identical deterministic replay"},
        dataset_name=",".join(args.scenes),
        renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
        extra_metadata={"experiment": "joint_spawn_refine_same_snapshot_replay",
                        "status": "expanded_exploratory_confirmation",
                        "pose_convention": "ground-truth W2C via Phase 10 loader",
                        "prefix_policy": args.prefix_policy,
                        "diagnostic_instrumentation": "pre-spawn visibility and residual means stratified by age; opt-in replay only",
                        "gpu_telemetry": args.gpu_telemetry,
                        "gpu_telemetry_interval_ms": (args.gpu_telemetry_interval_ms
                                                       if args.gpu_telemetry else None),
                        "gpu_telemetry_interleave": args.gpu_telemetry_interleave})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    args._gpu_telemetry = None
    if args.gpu_telemetry:
        device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        args._gpu_telemetry = GPUTelemetrySampler(
            device_index=device_index, interval_ms=args.gpu_telemetry_interval_ms).start()
        atexit.register(args._gpu_telemetry.stop)

    for scene in args.scenes:
        frames, intrinsics = load_phase10_sequence(scene_name=scene,
            n_frames=args.n_frames + 1, H=240, W=320, device=args.device)
        if any(frame_idx < 0 or frame_idx >= len(frames) for frame_idx in args.eval_frames):
            raise ValueError(f"--eval_frames must be in [0, {len(frames) - 1}]")
        for seed in args.seeds:
            for snapshot_frame in args.snapshot_frames:
                if any(eval_idx <= snapshot_frame for eval_idx in args.eval_frames):
                    raise ValueError(
                        f"Every --eval_frames entry must be after snapshot frame {snapshot_frame}")
                target = args.out_dir / scene / f"seed{seed}_frame{snapshot_frame}.json"
                if target.exists() and not args.force:
                    print(f"skip {target}", flush=True)
                    continue
                if snapshot_frame <= 10 or snapshot_frame >= len(frames):
                    raise ValueError(f"snapshot frame {snapshot_frame} is outside the valid sequence")
                set_seed(seed)
                if args.prefix_policy == "ours":
                    cfg = build_pipeline_config("ours", seed, args.budget_ms, 320, 240,
                                                args.device)
                    # Match the global_importance trajectory protocol used by
                    # the continuous age-quota study. Without this control the
                    # prefix falls back to the modeled-budget scheduler and can
                    # reach a different map-size/age state before replay.
                    cfg["training"]["optimization_control"] = {
                        "micro_updates_per_frame": args.micro_updates,
                    }
                else:
                    _, cfg = config_for("error_off", seed, args.budget_ms, args.device)
                cfg["system"]["synchronized_stage_timing"] = True
                base = OnlineReconstructionPipeline(config=cfg, device=args.device)
                replay_checkpoint = None
                replay_checkpoint_source_sha256 = None
                if args.replay_checkpoint_dir is not None:
                    checkpoint_root = (args.replay_checkpoint_dir if args.replay_checkpoint_dir.is_absolute()
                                       else ROOT / args.replay_checkpoint_dir)
                    replay_checkpoint = (checkpoint_root / "snapshots" / scene /
                                         f"seed{seed}_frame{snapshot_frame}.checkpoint.pt")
                    if not replay_checkpoint.is_file():
                        raise FileNotFoundError(f"Replay checkpoint not found: {replay_checkpoint}")
                    load_pipeline_checkpoint(
                        replay_checkpoint, base, scene=scene, seed=seed,
                        snapshot_frame=snapshot_frame, restore_rng=True)
                    replay_checkpoint_source_sha256 = hashlib.sha256(
                        replay_checkpoint.read_bytes()).hexdigest()
                else:
                    base.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                        intrinsics=intrinsics, pose=frames[0].get("pose", torch.eye(4, device=args.device)))
                    for t in range(1, snapshot_frame):
                        base.process_frame(rgb=frames[t]["rgb"], depth=frames[t]["depth"],
                                           gt_pose=frames[t].get("pose"))
                base_hash = digest_map(base.gaussian_model)
                if base.frame_count != snapshot_frame or base_hash != gaussian_map_digest(base.gaussian_model):
                    raise RuntimeError("Replay checkpoint frame/map validation failed")
                if (args.expected_snapshot_sha256 is not None and
                        base_hash != args.expected_snapshot_sha256):
                    raise RuntimeError(
                        "Common snapshot does not match --expected_snapshot_sha256: "
                        f"{base_hash} != {args.expected_snapshot_sha256}"
                    )
                snapshot_path = None
                snapshot_sha256 = None
                if args.save_map_snapshots:
                    snapshot_path = (args.out_dir / "snapshots" / scene /
                                     f"seed{seed}_frame{snapshot_frame}.pt")
                    if not snapshot_path.exists() or args.force:
                        save_map_snapshot(snapshot_path, base, scene, seed,
                                          snapshot_frame, base_hash)
                    else:
                        validate_map_snapshot(
                            snapshot_path, scene=scene, seed=seed,
                            snapshot_frame=snapshot_frame, map_digest=base_hash)
                    snapshot_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
                replay_checkpoint_path = None
                replay_checkpoint_sha256 = None
                if args.save_replay_checkpoints:
                    replay_checkpoint_path = (args.out_dir / "snapshots" / scene /
                                              f"seed{seed}_frame{snapshot_frame}.checkpoint.pt")
                    if replay_checkpoint_path.exists() and not args.force:
                        saved = torch.load(replay_checkpoint_path, map_location="cpu", weights_only=True)
                        meta = saved.get("metadata", {})
                        if (meta.get("map_digest") != base_hash or meta.get("scene") != scene or
                                meta.get("seed") != seed or meta.get("snapshot_frame") != snapshot_frame):
                            raise RuntimeError(f"Existing replay checkpoint does not match this prefix: {replay_checkpoint_path}")
                    else:
                        save_pipeline_checkpoint(
                            replay_checkpoint_path, base, scene=scene, seed=seed,
                            snapshot_frame=snapshot_frame)
                    replay_checkpoint_sha256 = hashlib.sha256(replay_checkpoint_path.read_bytes()).hexdigest()
                pre_action_eval = base.eval_frame(
                    rgb=frames[snapshot_frame]["rgb"],
                    depth=frames[snapshot_frame]["depth"],
                    pose=frames[snapshot_frame].get("pose", base.current_pose))
                base_rng = capture_rng(args.device)
                # Draw one ordered pool from the common pre-action depth mask.
                # All spawn doses use prefixes of this pool, ensuring nesting.
                restore_rng(base_rng)
                common_pool = pipeline_module.sample_candidates(
                    error_mask=frames[snapshot_frame]["depth"] > 0,
                    num_samples=max(args.spawn_counts), strategy="uniform")
                restore_rng(base_rng)
                candidate_pool_hash = hashlib.sha256(
                    common_pool.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
                warmup_record = None
                if args.warmup_updates > 0:
                    warmup_action = {"spawn_count": 0, "mature_fraction": 0.5,
                                     "micro_updates": args.warmup_updates,
                                     "name": "discarded_timing_warmup",
                                     "capture_first_backward_gradient": False}
                    warmup = run_branch(base, base_rng, scene, seed, snapshot_frame,
                                        warmup_action, frames[snapshot_frame],
                                        intrinsics, args, common_pool)
                    # The warm-up is discarded and exists only to stabilize
                    # device/runtime state. Its age-group split can be limited
                    # by the eligible populations at an early snapshot; only
                    # require the common snapshot and exact total update count.
                    if (warmup["snapshot_hash"] != base_hash or
                            warmup["n_micro_updates"] != args.warmup_updates):
                        raise RuntimeError("Warm-up branch failed snapshot/total-update validation")
                    warmup_record = {"snapshot_hash": warmup["snapshot_hash"],
                                     "n_micro_updates": warmup["n_micro_updates"],
                                     "opt_stage_sync_ms": warmup["opt_stage_sync_ms"],
                                     "opt_stage_cuda_event_ms": warmup["opt_stage_cuda_event_ms"],
                                     "cache_time_host_ms": warmup["cache_time_host_ms"],
                                     "cache_cuda_event_ms": warmup["cache_cuda_event_ms"],
                                     "n_cache_rebuilds": warmup["n_cache_rebuilds"],
                                     "n_optimized": warmup["n_optimized"],
                                     "optimizer_iteration_count": warmup["optimizer_iteration_count"],
                                     "optimizer_forward_cuda_event_ms": warmup["optimizer_forward_cuda_event_ms"],
                                     "optimizer_backward_cuda_event_ms": warmup["optimizer_backward_cuda_event_ms"],
                                     "optimizer_backward_only_cuda_event_ms": warmup["optimizer_backward_only_cuda_event_ms"],
                                     "optimizer_step_cuda_event_ms": warmup["optimizer_step_cuda_event_ms"],
                                     "gpu_telemetry": warmup["gpu_telemetry"],
                                     "requested_mature_fraction": warmup_action["mature_fraction"],
                                     "realized_mature_micro_updates": warmup["realized_mature_micro_updates"],
                                     "realized_young_micro_updates": warmup["realized_young_micro_updates"],
                                     "exact_age_split_valid": warmup["valid"],
                                     "discarded": True}
                ordered_arms = list(arms)
                random.Random(args.action_order_seed + seed + snapshot_frame).shuffle(ordered_arms)
                action_records = []
                for arm in ordered_arms:
                    print(f"REPLAY {scene} seed={seed} frame={snapshot_frame} {arm['name']}", flush=True)
                    rec = run_branch(base, base_rng, scene, seed, snapshot_frame, arm,
                                     frames[snapshot_frame], intrinsics, args, common_pool,
                                     [(idx, frames[idx]) for idx in args.eval_frames])
                    action_records.append(rec)
                hashes = {r["snapshot_hash"] for r in action_records}
                pre_psnrs = [r["pre_psnr"] for r in action_records]
                spawn_sets = {int(r["action"]["spawn_count"]): r["spawn_candidates_uv"]
                              for r in action_records if r["action"]["spawn_count"]}
                same_spawn_pool = all(
                    spawn_sets[dose] == common_pool[:dose].detach().cpu().tolist()
                    for dose in spawn_sets)
                nested_pool = all(
                    spawn_sets[a] == spawn_sets[b][:a]
                    for a in spawn_sets for b in spawn_sets if a < b)
                valid = (len(hashes) == 1 and next(iter(hashes)) == base_hash
                         and all(r["valid"] for r in action_records)
                         and max(pre_psnrs) - min(pre_psnrs) <= 1e-6
                         and same_spawn_pool and nested_pool)
                same_count_by_spawn_dose = all(
                    len({r["n_gaussians_after_action"] for r in action_records
                         if int(r["action"]["spawn_count"]) == dose}) == 1
                    for dose in args.spawn_counts)
                valid = valid and same_count_by_spawn_dose
                gradient_pairs_valid = None
                if args.capture_first_backward_gradients:
                    pairs = {}
                    for branch_record in action_records:
                        pair_name = branch_record["action"]["paired_action_name"]
                        pairs.setdefault(pair_name, []).append(branch_record)
                    gradient_pairs_valid = bool(pairs) and all(
                        len(pair) == 2
                        and {r["ssim_implementation"] for r in pair} == {"direct", "separable"}
                        and all(r["first_backward_gradient"] is not None for r in pair)
                        and pair[0]["first_backward_gradient"]["active_index_sha256"]
                            == pair[1]["first_backward_gradient"]["active_index_sha256"]
                        and pair[0]["first_backward_gradient"]["parameter_shapes"]
                            == pair[1]["first_backward_gradient"]["parameter_shapes"]
                        for pair in pairs.values())
                    valid = valid and gradient_pairs_valid
                record = {"experiment": "joint_spawn_refine_same_snapshot_replay",
                    "scene": scene, "seed": seed, "snapshot_frame": snapshot_frame,
                    "fixed_eval_frames": args.eval_frames,
                    "prefix_policy": args.prefix_policy, "base_snapshot_hash": base_hash,
                    "map_snapshot_path": (str(snapshot_path.relative_to(args.out_dir))
                                          if snapshot_path is not None else None),
                    "map_snapshot_sha256": snapshot_sha256,
                    "replay_checkpoint_path": (str(replay_checkpoint_path.relative_to(args.out_dir))
                                                if replay_checkpoint_path is not None else None),
                    "replay_checkpoint_sha256": replay_checkpoint_sha256,
                    "replay_checkpoint_source": (str(replay_checkpoint) if replay_checkpoint is not None else None),
                    "replay_checkpoint_source_sha256": replay_checkpoint_source_sha256,
                    "pre_action_snapshot_eval": {
                        "psnr": float(pre_action_eval["psnr"]),
                        "ssim": float(pre_action_eval["ssim"]),
                        "depth_l1": float(pre_action_eval["depth_l1"]),
                    },
                    "pre_action_reference_psnr_range": [min(pre_psnrs), max(pre_psnrs)],
                    "common_candidate_pool_sha256": candidate_pool_hash,
                    "common_candidate_pool_count": int(len(common_pool)),
                    "warmup_record": warmup_record,
                    "action_order_seed": args.action_order_seed,
                    "action_execution_order": [a["name"] for a in ordered_arms],
                    "spawn_candidates_are_prefixes_of_common_pool": same_spawn_pool,
                    "spawn_doses_are_nested": nested_pool,
                    "map_count_matched_within_spawn_dose": same_count_by_spawn_dose,
                    "gradient_pairs_valid": gradient_pairs_valid,
                    "all_count_and_snapshot_gates_passed": valid,
                    "target_spawn_doses": args.spawn_counts,
                    "target_micro_updates_per_arm": {r["action"]["name"]: r["target_micro_updates"]
                                                     for r in action_records},
                    "arms": action_records}
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp")
                tmp.write_text(json.dumps(record, indent=2))
                tmp.replace(target)
                print(f"DONE {scene} seed={seed} frame={snapshot_frame}: valid={valid}", flush=True)
                base.cleanup()
                del base
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
