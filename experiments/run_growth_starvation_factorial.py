#!/usr/bin/env python3
"""Same-snapshot growth x stale-mature-update rescue factorial.

Each snapshot is produced once by a frozen common prefix. Deep-copied pipelines
process the same next RGB-D frame with nested spawn doses crossed with either
global-importance selection or a fixed stale-mature rescue quota. The rescue is
an experimental intervention, not a proposed online policy.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_dilution_factorial import config_for
from experiments.run_frozen_benchmark import build_pipeline_config, load_benchmark_sequence
from src.core import pipeline as pipeline_module
from src.core.pipeline import OnlineReconstructionPipeline
from src.evaluation.reproducibility import create_provenance_manifest, set_seed
from src.core.map_snapshots import validate_map_snapshot
from src.core.pipeline_checkpoint import (
    load_pipeline_checkpoint, save_pipeline_checkpoint,
    gaussian_map_digest,
)
from src.core.attribution import render_with_attribution, compute_gaussian_statistics


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


def sample_identity_trace_cohort(pipeline, frame_idx, seed, per_stratum=8):
    """Choose a small, deterministic pre-intervention cohort by age/staleness."""
    store = pipeline.gaussian_model.state_store
    ids = store.persistent_ids.detach().cpu().numpy()
    ages = np.maximum(0, int(frame_idx) - store.creation_frames.detach().cpu().numpy())
    stale = np.maximum(0, int(frame_idx) - store.last_update_frames.detach().cpu().numpy())
    active = pipeline.gaussian_model._state.detach().cpu().numpy() != 3
    strata = {"young_lt_10": active & (ages < 10)}
    for name, low, high in (("mature_stale_0_2", 0, 2), ("mature_stale_3_5", 3, 5),
                            ("mature_stale_6_9", 6, 9), ("mature_stale_10_19", 10, 19),
                            ("mature_stale_20_plus", 20, None)):
        mask = active & (ages >= 10) & (stale >= low)
        if high is not None:
            mask &= stale <= high
        strata[name] = mask
    rng = np.random.default_rng(int(seed) * 1000003 + int(frame_idx))
    selected, counts = [], {}
    for name, mask in strata.items():
        candidates = np.flatnonzero(mask)
        take = min(int(per_stratum), len(candidates))
        chosen = np.sort(rng.choice(candidates, size=take, replace=False)) if take else np.array([], dtype=int)
        values = ids[chosen].astype(np.int64).tolist()
        selected.extend(values)
        counts[name] = {"population": int(len(candidates)), "sampled": len(values),
                        "persistent_ids": values}
    return selected, counts


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


def adapt_micro_update_budget(current_updates, measured_stage_ms, target_stage_ms,
                              min_updates, max_updates):
    """Proportional feedback for a measured optimization-stage time target."""
    if measured_stage_ms <= 0 or target_stage_ms <= 0:
        return int(current_updates)
    ratio = target_stage_ms / measured_stage_ms
    # Bound each frame's adjustment to avoid unstable jumps from timing noise.
    ratio = min(1.35, max(0.70, ratio))
    return int(min(max_updates, max(min_updates, round(current_updates * ratio))))


def contribution_reference_eval(pipeline, view, persistent_ids, top_k=8):
    """Top-k contribution-weighted RGB-D residuals on a fixed reference view."""
    model = pipeline.gaussian_model
    pose = view.get("pose", pipeline.current_pose)
    if pose is None:
        pose = pipeline.current_pose
    pose = pose.to(pipeline.device)
    rgb, depth = view["rgb"], view["depth"]
    h, w = depth.shape
    cov3d = model.build_covariance()
    if model.sh_degree > 0:
        from src.core.rasterizer import _view_directions
        dirs = _view_directions(model.positions, pose)
    else:
        dirs = None
    with torch.no_grad():
        rendered = render_with_attribution(
            means3D=model.positions, cov3D=cov3d, colors=model.get_colors(dirs),
            opacities=model.opacities.squeeze(-1), extrinsics=pose,
            intrinsics=pipeline.intrinsics, image_width=w, image_height=h,
            tile_size=pipeline.config["rendering"]["tile_size"], top_k=top_k)
        stats = compute_gaussian_statistics(
            rendered_color=rendered["color"], rendered_depth=rendered["depth"],
            gt_color=rgb, gt_depth=depth,
            contrib_weights=rendered["contrib_weights"],
            contrib_indices=rendered["contrib_indices"],
            n_gaussians=model.num_gaussians)
        ids_now = model.state_store.persistent_ids[:model.num_gaussians]
        order = torch.argsort(ids_now)
        sorted_ids = ids_now[order]
        query = torch.as_tensor(persistent_ids, dtype=torch.long, device=pipeline.device)
        pos = torch.searchsorted(sorted_ids, query)
        in_range = pos < sorted_ids.numel()
        safe = pos.clamp(max=max(sorted_ids.numel() - 1, 0))
        present = (in_range & (sorted_ids[safe] == query)
                   if sorted_ids.numel() else torch.zeros_like(in_range))
        idx = order[safe] if sorted_ids.numel() else torch.zeros_like(safe)
        def vals(key, fill=0.0):
            out = torch.full(query.shape, fill, dtype=torch.float32, device=pipeline.device)
            if present.any():
                out[present] = stats[key][idx[present]].float()
            return out.detach().cpu().tolist()
        visible = torch.zeros_like(present)
        if present.any():
            visible[present] = stats["visibility_mask"][idx[present]]
        rgb_contrib_err = stats["color_error"] * stats["contribution_mass"] / float(h * w)
        depth_contrib_err = stats["depth_error"] * stats["contribution_mass"] / float(h * w)
        def weighted_vals(source, fill=0.0):
            out = torch.full(query.shape, fill, dtype=torch.float32, device=pipeline.device)
            if present.any():
                out[present] = source[idx[present]].float()
            return out.detach().cpu().tolist()
        return {"persistent_ids": query.detach().cpu().tolist(),
                "present": present.detach().cpu().tolist(),
                "visible": visible.detach().cpu().tolist(),
                "rgb_contribution_weighted_error": vals("color_error"),
                "depth_contribution_weighted_error": vals("depth_error"),
                "contribution_mass_topk": vals("contribution_mass"),
                "rgb_attributed_abs_error_per_pixel": weighted_vals(rgb_contrib_err),
                "depth_attributed_abs_error_per_pixel": weighted_vals(depth_contrib_err),
                "metric_definition": f"pixel error weighted by top-{top_k} alpha-compositing contributions, reference attribution renderer"}


def run_branch(base, rng_state, scene, seed, snapshot_frame, action,
              frame, frames, intrinsics, args, common_pool, fixed_eval_frames,
              pre_action_fixed_eval, trace_persistent_ids=None):
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
        "micro_updates_per_frame": args.micro_updates,
        "selection": "descending importance; stale-rescue quota is opt-in",
        "mature_age_frames": 10,
        # Read-only summaries from the shared pre-spawn attribution pass.
        "capture_age_diagnostics": True,
    }
    if trace_persistent_ids:
        optimization_control["trace_persistent_ids"] = trace_persistent_ids
    if action["allocation"] == "stale_rescue":
        optimization_control.update({
            "stale_update_fraction": float(action.get("stale_fraction", args.stale_fraction)),
            "stale_after_frames": args.stale_after_frames,
        })
    elif action["allocation"] == "stratified_randomized":
        optimization_control.update({
            "randomized_candidate_persistent_ids": trace_persistent_ids or [],
            "randomization_seed": int(seed * 100003 + snapshot_frame + getattr(args, "randomization_replicate", 0) * 10000019),
            "mature_age_frames": 10,
        })
    branch.config["training"]["optimization_control"] = optimization_control
    current_micro_updates = int(args.micro_updates)
    optimization_control["micro_updates_per_frame"] = current_micro_updates

    sampled = []
    frame_count_before = int(branch.frame_count)
    original_sampler = pipeline_module.sample_candidates
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    action_start = time.perf_counter()
    def recording_sampler(*call_args, **call_kwargs):
        requested = int(call_kwargs.get("num_samples", call_args[1] if len(call_args) > 1 else 0))
        candidates = common_pool[:requested].to(frame["depth"].device)
        if len(candidates) != requested:
            raise RuntimeError(f"common candidate pool too small: requested {requested}, got {len(candidates)}")
        sampled.append(candidates.detach().cpu().tolist())
        return candidates
    pipeline_module.sample_candidates = recording_sampler
    try:
        metrics = branch.process_frame(
            rgb=frame["rgb"], depth=frame["depth"], gt_pose=frame.get("pose"))
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        metrics["runner_frame_wall_sync_ms"] = (time.perf_counter() - action_start) * 1000.0
        # process_frame's depth_l1 uses depth_err computed before optimizer
        # updates. Re-render the final map at the same GT view for a true
        # post-action geometric diagnostic; eval_frame is read-only.
        pose = frame.get("pose", branch.current_pose)
        final_map_eval = branch.eval_frame(
            rgb=frame["rgb"], depth=frame["depth"], pose=pose)
        horizon_metrics = []
        for future_idx in range(snapshot_frame + 1,
                                snapshot_frame + args.outcome_horizon_frames + 1):
            if args.target_opt_stage_ms is not None:
                current_micro_updates = adapt_micro_update_budget(
                    current_micro_updates,
                    float(metrics.get("opt_stage_sync_ms", 0.0)) if not horizon_metrics else
                    float(horizon_metrics[-1].get("opt_stage_sync_ms", 0.0)),
                    args.target_opt_stage_ms,
                    args.min_micro_updates, args.max_micro_updates)
                optimization_control["micro_updates_per_frame"] = current_micro_updates
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            future = frames[future_idx]
            future_metrics = branch.process_frame(
                rgb=future["rgb"], depth=future["depth"],
                gt_pose=future.get("pose"))
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            future_metrics["runner_frame_wall_sync_ms"] = (
                time.perf_counter() - start) * 1000.0
            horizon_metrics.append(future_metrics)
        fixed_eval = []
        for eval_idx in fixed_eval_frames:
            view = frames[eval_idx]
            result = branch.eval_frame(
                rgb=view["rgb"], depth=view["depth"],
                pose=view.get("pose", branch.current_pose),
                trace_persistent_ids=(trace_persistent_ids
                                      if trace_persistent_ids else None))
            item = {"frame": int(eval_idx),
                "psnr": float(result["psnr"]), "ssim": float(result["ssim"]),
                "depth_l1": float(result["depth_l1"]),
                "identity_reference_residual": result.get("identity_reference_residual")}
            if action["allocation"] == "stratified_randomized":
                item["identity_contribution_residual"] = contribution_reference_eval(
                    branch, view, trace_persistent_ids)
            fixed_eval.append(item)
    finally:
        pipeline_module.sample_candidates = original_sampler

    realized_updates = int(metrics.get("n_micro_updates", -1))
    realized_spawn = int(metrics.get("n_spawned", 0))
    action_stale_fraction = action.get("stale_fraction")
    requested_stale_fraction = float(
        args.stale_fraction if action_stale_fraction is None else action_stale_fraction)
    expected_stale_updates = int(round(args.micro_updates * requested_stale_fraction))
    stale_quota_realized = metrics.get("stale_quota_micro_updates")
    realized_stale_quota = int(stale_quota_realized or 0)
    if action["allocation"] == "stratified_randomized":
        assignment = metrics.get("randomized_assignment") or {}
        allocation_gate = (int(assignment.get("treated_ids", 0)) > 0
                           and int(assignment.get("control_ids", 0)) > 0
                           and int(assignment.get("treated_ids", 0))
                               + int(assignment.get("control_ids", 0))
                               == int(assignment.get("randomizable_ids", -1)))
    else:
        allocation_gate = (action["allocation"] == "global" or
            ((0 <= realized_stale_quota <= expected_stale_updates)
             if args.allow_partial_stale_quota else
             (realized_stale_quota == expected_stale_updates
              and int(metrics.get("stale_mature_micro_updates", 0)) == expected_stale_updates)))
    all_budget_frames = [metrics] + horizon_metrics
    stage_times = [float(m.get("opt_stage_sync_ms", 0.0)) for m in all_budget_frames]
    wall_times = [float(m.get("runner_frame_wall_sync_ms", 0.0)) for m in all_budget_frames]
    horizon_update_counts = [int(m.get("n_micro_updates", 0)) for m in horizon_metrics]
    budget_gate = True
    if args.target_opt_stage_ms is not None:
        settled_stage = np.asarray(stage_times[min(3, len(stage_times)):], dtype=float)
        budget_gate = bool(len(settled_stage) > 0 and np.all(np.isfinite(settled_stage))
            and np.median(settled_stage) >= args.target_opt_stage_ms * 0.70
            and np.median(settled_stage) <= args.target_opt_stage_ms * 1.30)
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
        "outcome_horizon_frames_requested": int(args.outcome_horizon_frames),
        "outcome_horizon_frames_completed": len(horizon_metrics),
        "budget_mode": "measured_opt_stage_time" if args.target_opt_stage_ms is not None else "fixed_micro_updates",
        "target_opt_stage_ms": args.target_opt_stage_ms,
        "initial_micro_updates": int(args.micro_updates),
        "horizon_micro_update_counts": horizon_update_counts,
        "mean_micro_updates_per_horizon_frame": float(np.mean(horizon_update_counts)) if horizon_update_counts else 0.0,
        "mean_opt_stage_sync_ms_all_frames": float(np.mean(stage_times)) if stage_times else 0.0,
        "p95_opt_stage_sync_ms_all_frames": float(np.percentile(stage_times, 95)) if stage_times else 0.0,
        "mean_frame_wall_sync_ms_all_frames": float(np.mean(wall_times)) if wall_times else 0.0,
        "p95_frame_wall_sync_ms_all_frames": float(np.percentile(wall_times, 95)) if wall_times else 0.0,
        "measured_time_budget_gate": budget_gate,
        "horizon_frame_metrics": [{
            "frame": int(m["frame"]),
            "n_micro_updates": int(m.get("n_micro_updates", 0)),
            "mature_eligible": int(m.get("mature_eligible", 0)),
            "stale_mature_eligible": int(m.get("stale_mature_eligible", 0)),
            "stale_mature_updated": int(m.get("stale_mature_updated", 0)),
            "stale_mature_micro_updates": int(m.get("stale_mature_micro_updates", 0)),
            "mature_update_coverage": float(m.get("mature_update_coverage", 0.0)),
            "stale_mature_update_coverage": float(m.get("stale_mature_update_coverage", 0.0)),
            "n_spawned": int(m.get("n_spawned", 0)),
            "opt_stage_sync_ms": float(m.get("opt_stage_sync_ms", 0.0)),
            "frame_wall_sync_ms": float(m.get("runner_frame_wall_sync_ms", 0.0)),
            "age_stratified_diagnostics": m.get("age_stratified_diagnostics"),
            "identity_trace": m.get("identity_trace"),
        } for m in horizon_metrics],
        "fixed_eval_frames": fixed_eval_frames,
        "fixed_eval_after_horizon": fixed_eval,
        "fixed_eval_before_action": pre_action_fixed_eval,
        "fixed_eval_psnr_gain_db": (float(np.mean([r["psnr"] for r in fixed_eval]) -
            np.mean([r["psnr"] for r in pre_action_fixed_eval])) if fixed_eval else None),
        "fixed_eval_depth_l1_change": (float(np.mean([r["depth_l1"] for r in fixed_eval]) -
            np.mean([r["depth_l1"] for r in pre_action_fixed_eval])) if fixed_eval else None),
        "n_spawned": realized_spawn,
        "n_gaussians_after_action": int(branch.gaussian_model.num_gaussians),
        "target_spawned": spawn_count,
        "spawn_count_before_action": spawn_count_before,
        "n_micro_updates": realized_updates,
        "target_micro_updates": args.micro_updates,
        "mature_eligible": int(metrics.get("mature_eligible", 0)),
        "mature_updated": int(metrics.get("mature_updated", 0)),
        "mature_update_coverage": float(metrics.get("mature_update_coverage", 0.0)),
        "stale_after_frames": int(metrics.get("stale_after_frames", args.stale_after_frames)),
        "stale_mature_eligible": int(metrics.get("stale_mature_eligible", 0)),
        "stale_mature_updated": int(metrics.get("stale_mature_updated", 0)),
        "stale_mature_update_coverage": float(metrics.get("stale_mature_update_coverage", 0.0)),
        "stale_mature_micro_updates": int(metrics.get("stale_mature_micro_updates", 0)),
        "requested_stale_update_fraction": (requested_stale_fraction
            if action["allocation"] == "stale_rescue" else None),
        "realized_stale_quota_micro_updates": int(stale_quota_realized or 0),
        "realized_other_quota_micro_updates": int(metrics.get("other_quota_micro_updates") or 0),
        "growth_pressure": float(metrics.get("growth_pressure", 0.0)),
        "compute_return_db_per_opt_ms": float(metrics.get("compute_return_db_per_opt_ms", 0.0)),
        "opt_stage_sync_ms": float(metrics.get("opt_stage_sync_ms", 0.0)),
        "frame_time_ms": float(metrics.get("frame_time_ms", 0.0)),
        "age_stratified_diagnostics": metrics.get("age_stratified_diagnostics"),
        "identity_trace": metrics.get("identity_trace"),
        "randomized_assignment": metrics.get("randomized_assignment"),
        "spawn_candidates_uv": sampled[-1] if spawn_count and sampled else [],
        "valid": (realized_spawn == spawn_count and
                  (realized_updates == args.micro_updates if action["allocation"] != "stratified_randomized"
                   else (metrics.get("randomized_assignment") is not None and
                         realized_updates == int(metrics["randomized_assignment"]["treated_ids"])))
                  and len(horizon_metrics) == args.outcome_horizon_frames
                  and (all(int(m.get("n_micro_updates", -1)) == args.micro_updates
                           for m in horizon_metrics)
                       if args.target_opt_stage_ms is None else
                       all(int(m.get("n_micro_updates", 0)) > 0 for m in horizon_metrics)
                       and budget_gate)
                  and allocation_gate),
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
    ap.add_argument("--prefix_policy", choices=("error_only", "ours"), default="error_only",
                    help="Mapping policy used to create the common prefix snapshot")
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--micro_updates", type=int, default=5000)
    ap.add_argument("--outcome_horizon_frames", type=int, default=0,
                    help="Process this many post-intervention frames with no further spawning, then score fixed prefix views")
    ap.add_argument("--target_opt_stage_ms", type=float, default=None,
                    help="Optional measured CUDA optimization-stage target; proportional feedback changes updates each frame")
    ap.add_argument("--min_micro_updates", type=int, default=500)
    ap.add_argument("--max_micro_updates", type=int, default=8000)
    ap.add_argument("--spawn_counts", nargs="+", type=int, default=[0, 128, 256])
    ap.add_argument("--allocation_modes", nargs="+", choices=("global", "stale_rescue", "stratified_randomized"),
                    default=["global", "stale_rescue"])
    ap.add_argument("--stale_fraction", type=float, default=0.50,
                    help="Fraction of fixed micro-update budget reserved for stale mature Gaussians")
    ap.add_argument("--stale_fractions", nargs="+", type=float, default=None,
                    help="Optional stale-rescue dose curve; adds one stale-rescue arm per fraction alongside global arms")
    ap.add_argument("--allow_partial_stale_quota", action="store_true",
                    help="Keep arms valid when the requested stale quota exceeds eligible candidate capacity; report realized quota")
    ap.add_argument("--stale_after_frames", type=int, default=10)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "joint_spawn_refine_replay")
    ap.add_argument("--save_map_snapshots", action="store_true",
                    help="Persist CPU Gaussian parameters and identity/age state for every common replay snapshot")
    ap.add_argument("--save_replay_checkpoints", action="store_true",
                    help="Also save complete pipeline/optimizer/RNG checkpoints for independent replay")
    ap.add_argument("--replay_checkpoint_dir", type=Path, default=None,
                    help="Load checkpoints from this experiment root instead of rebuilding the common prefix")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--capture_identity_trace", action="store_true",
                    help="Trace a deterministic pre-intervention persistent-ID cohort through replay frames")
    ap.add_argument("--trace_ids_per_stratum", type=int, default=8)
    ap.add_argument("--identity_revisit_offsets", nargs="+", type=int, default=None,
                    help="Evaluate randomized-trial identity outcomes at these future fixed-view offsets from the action frame")
    ap.add_argument("--randomization_replicate", type=int, default=0,
                    help="Random assignment replicate index for reruns from the same saved checkpoint")
    args = ap.parse_args()
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    if any(x < 0 for x in args.spawn_counts) or sorted(set(args.spawn_counts)) != args.spawn_counts:
        raise ValueError("spawn_counts must be unique, nonnegative, and increasing")
    if max(args.spawn_counts) < 1 and args.allocation_modes != ["stratified_randomized"]:
        raise ValueError("spawn_counts must include a positive dose")
    if "stratified_randomized" in args.allocation_modes and args.spawn_counts != [0]:
        raise ValueError("stratified_randomized must use spawn_counts=[0] to keep the randomized update cohort fixed")
    if "stratified_randomized" in args.allocation_modes and not args.capture_identity_trace:
        raise ValueError("stratified_randomized requires --capture_identity_trace to define the randomized cohort")
    if "stratified_randomized" in args.allocation_modes and not args.identity_revisit_offsets:
        raise ValueError("stratified_randomized requires future --identity_revisit_offsets for fixed-view outcomes")
    if args.identity_revisit_offsets is not None and any(x <= 0 for x in args.identity_revisit_offsets):
        raise ValueError("identity revisit offsets must be positive")
    if not 0.0 <= args.stale_fraction <= 1.0:
        raise ValueError("stale_fraction must lie in [0, 1]")
    if args.stale_fractions is not None and (
            any(not 0.0 <= x <= 1.0 for x in args.stale_fractions)
            or len(set(args.stale_fractions)) != len(args.stale_fractions)):
        raise ValueError("stale_fractions must be unique values in [0, 1]")
    if args.stale_after_frames <= 0:
        raise ValueError("stale_after_frames must be positive")
    if args.stale_fractions is None:
        arms = [{"spawn_count": dose, "allocation": mode,
                 "mature_fraction": None,
                 "stale_fraction": args.stale_fraction if mode == "stale_rescue" else None,
                 "name": f"spawn{dose}_{mode}"}
                for mode in args.allocation_modes for dose in args.spawn_counts]
    else:
        arms = ([{"spawn_count": dose, "allocation": "global",
                  "mature_fraction": None, "stale_fraction": None,
                  "name": f"spawn{dose}_global"} for dose in args.spawn_counts] +
                [{"spawn_count": dose, "allocation": "stale_rescue",
                  "mature_fraction": None, "stale_fraction": fraction,
                  "name": f"spawn{dose}_stale{int(round(fraction * 100))}"}
                 for fraction in args.stale_fractions for dose in args.spawn_counts])
    manifest = create_provenance_manifest(
        config={"scenes": args.scenes, "seeds": args.seeds,
                "snapshot_frames": args.snapshot_frames, "n_frames": args.n_frames,
                "micro_updates_per_action": args.micro_updates,
                "outcome_horizon_frames": args.outcome_horizon_frames,
                "target_opt_stage_ms": args.target_opt_stage_ms,
                "min_micro_updates": args.min_micro_updates,
                "max_micro_updates": args.max_micro_updates,
                "prefix_micro_updates_per_frame": (args.micro_updates if args.prefix_policy == "ours" else None),
                "prefix_policy": args.prefix_policy,
                "save_map_snapshots": args.save_map_snapshots,
                "save_replay_checkpoints": args.save_replay_checkpoints,
                "replay_checkpoint_dir": (str(args.replay_checkpoint_dir) if args.replay_checkpoint_dir else None),
                "spawn_doses": args.spawn_counts,
                "allocation_modes": args.allocation_modes,
                "stale_update_fraction": args.stale_fraction,
                "stale_update_fractions_dose_curve": args.stale_fractions,
                "allow_partial_stale_quota": args.allow_partial_stale_quota,
                "capture_identity_trace": args.capture_identity_trace,
                "trace_ids_per_stratum": args.trace_ids_per_stratum,
                "randomization_replicate": args.randomization_replicate,
                "stale_after_frames": args.stale_after_frames,
                "arms": arms, "prefix": f"{args.prefix_policy} common prefix; exact update count when configured"},
        dataset_name=",".join(args.scenes),
        renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
        extra_metadata={"experiment": "growth_starvation_same_snapshot_factorial",
                        "status": "exploratory_pilot",
                        "pose_convention": "ground-truth W2C via Phase 10 loader",
                        "prefix_policy": args.prefix_policy,
        "diagnostic_instrumentation": "pre-action stale-mature visible eligibility and residual means stratified by age; optional persistent-ID cohort trace; opt-in replay only"})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    for scene in args.scenes:
        frames, intrinsics = load_benchmark_sequence(scene_name=scene,
            n_frames=args.n_frames + 1, H=240, W=320, device=args.device)
        for seed in args.seeds:
            for snapshot_frame in args.snapshot_frames:
                target = args.out_dir / scene / f"seed{seed}_frame{snapshot_frame}.json"
                if target.exists() and not args.force:
                    print(f"skip {target}", flush=True)
                    continue
                if (snapshot_frame <= 10 or
                        snapshot_frame + args.outcome_horizon_frames >= len(frames)):
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
                else:
                    base.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                        intrinsics=intrinsics, pose=frames[0].get("pose", torch.eye(4, device=args.device)))
                    for t in range(1, snapshot_frame):
                        base.process_frame(rgb=frames[t]["rgb"], depth=frames[t]["depth"],
                                           gt_pose=frames[t].get("pose"))
                base_hash = digest_map(base.gaussian_model)
                if base.frame_count != snapshot_frame or base_hash != gaussian_map_digest(base.gaussian_model):
                    raise RuntimeError("Replay checkpoint frame/map validation failed")
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
                trace_ids, trace_strata = (sample_identity_trace_cohort(
                    base, snapshot_frame, seed, args.trace_ids_per_stratum)
                    if args.capture_identity_trace else ([], {}))
                revisit_support = []
                if args.identity_revisit_offsets:
                    candidate_offsets = sorted(set(args.identity_revisit_offsets))
                    if snapshot_frame + max(candidate_offsets) >= len(frames):
                        raise ValueError("Requested identity revisit offset exceeds loaded sequence")
                    source_eval = base.eval_frame(
                        rgb=frames[snapshot_frame]["rgb"], depth=frames[snapshot_frame]["depth"],
                        pose=frames[snapshot_frame].get("pose", base.current_pose),
                        trace_persistent_ids=trace_ids)
                    source_vis = source_eval["identity_reference_residual"]["visible"]
                    hidden_since_source = [not bool(v) for v in source_vis]
                    fixed_eval_frames = []
                    for idx in range(snapshot_frame + 1,
                                     snapshot_frame + max(candidate_offsets) + 1):
                        view = frames[idx]
                        vis_eval = base.eval_frame(rgb=view["rgb"], depth=view["depth"],
                            pose=view.get("pose", base.current_pose),
                            trace_persistent_ids=trace_ids)
                        visible_now = vis_eval["identity_reference_residual"]["visible"]
                        revisited = sum(bool(source_vis[i]) and hidden_since_source[i]
                                        and bool(visible_now[i]) for i in range(len(trace_ids)))
                        for i, is_visible in enumerate(visible_now):
                            if not is_visible:
                                hidden_since_source[i] = True
                        offset = idx - snapshot_frame
                        if offset in candidate_offsets and revisited > 0:
                            fixed_eval_frames.append(idx)
                            revisit_support.append({"frame": idx,
                                "source_visible_ids": int(sum(bool(v) for v in source_vis)),
                                "revisited_visible_ids": int(revisited)})
                    if not fixed_eval_frames:
                        raise ValueError("No requested future frame contains a cohort ID revisit; choose a later fixed-view offset")
                else:
                    fixed_eval_frames = sorted(set(idx for idx in (
                        snapshot_frame - 20, snapshot_frame - 10, snapshot_frame - 1) if idx > 0))
                pre_action_fixed_eval = []
                for eval_idx in fixed_eval_frames:
                    view = frames[eval_idx]
                    result = base.eval_frame(
                        rgb=view["rgb"], depth=view["depth"],
                        pose=view.get("pose", base.current_pose),
                        trace_persistent_ids=(trace_ids if trace_ids else None))
                    item = {"frame": int(eval_idx),
                        "psnr": float(result["psnr"]), "ssim": float(result["ssim"]),
                        "depth_l1": float(result["depth_l1"]),
                        "identity_reference_residual": result.get("identity_reference_residual")}
                    if "stratified_randomized" in args.allocation_modes and trace_ids:
                        item["identity_contribution_residual"] = contribution_reference_eval(
                            base, view, trace_ids)
                    pre_action_fixed_eval.append(item)
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
                action_records = []
                for arm in arms:
                    print(f"REPLAY {scene} seed={seed} frame={snapshot_frame} {arm['name']}", flush=True)
                    rec = run_branch(base, base_rng, scene, seed, snapshot_frame, arm,
                                     frames[snapshot_frame], frames, intrinsics, args,
                                     common_pool, fixed_eval_frames,
                                     pre_action_fixed_eval, trace_ids)
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
                record = {"experiment": "growth_starvation_same_snapshot_factorial",
                    "scene": scene, "seed": seed, "snapshot_frame": snapshot_frame,
                    "prefix_policy": args.prefix_policy, "base_snapshot_hash": base_hash,
                    "map_snapshot_path": (str(snapshot_path.relative_to(args.out_dir))
                                          if snapshot_path is not None else None),
                    "map_snapshot_sha256": snapshot_sha256,
                    "replay_checkpoint_path": (str(replay_checkpoint_path.relative_to(args.out_dir))
                                                if replay_checkpoint_path is not None else None),
                    "replay_checkpoint_sha256": replay_checkpoint_sha256,
                    "replay_checkpoint_source": (str(replay_checkpoint) if replay_checkpoint is not None else None),
                    "pre_action_snapshot_eval": {
                        "psnr": float(pre_action_eval["psnr"]),
                        "ssim": float(pre_action_eval["ssim"]),
                        "depth_l1": float(pre_action_eval["depth_l1"]),
                    },
                    "pre_action_reference_psnr_range": [min(pre_psnrs), max(pre_psnrs)],
                    "common_candidate_pool_sha256": candidate_pool_hash,
                    "common_candidate_pool_count": int(len(common_pool)),
                    "spawn_candidates_are_prefixes_of_common_pool": same_spawn_pool,
                    "spawn_doses_are_nested": nested_pool,
                    "map_count_matched_within_spawn_dose": same_count_by_spawn_dose,
                    "all_count_and_snapshot_gates_passed": valid,
                    "target_spawn_doses": args.spawn_counts,
                    "allocation_modes": args.allocation_modes,
                    "stale_update_fraction": args.stale_fraction,
                    "stale_update_fractions_dose_curve": args.stale_fractions,
                    "stale_after_frames": args.stale_after_frames,
                    "target_micro_updates_per_arm": args.micro_updates,
                    "outcome_horizon_frames": args.outcome_horizon_frames,
                    "budget_mode": "measured_opt_stage_time" if args.target_opt_stage_ms is not None else "fixed_micro_updates",
                    "target_opt_stage_ms": args.target_opt_stage_ms,
                    "fixed_eval_frames": fixed_eval_frames,
                    "identity_trace_cohort_enabled": bool(args.capture_identity_trace),
                    "identity_trace_cohort_strata": trace_strata,
                    "identity_revisit_offsets": args.identity_revisit_offsets,
                    "identity_revisit_support": revisit_support,
                    "randomization_replicate": args.randomization_replicate,
                    "fixed_eval_protocol": ("fixed future sequence views rendered before and immediately after the randomized action"
                        if args.identity_revisit_offsets else
                        "fixed pre-intervention prefix views rendered before and after the intervention horizon"),
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
