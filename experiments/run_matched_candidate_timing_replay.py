#!/usr/bin/env python3
"""Replay an identical Gaussian batch at early or late trajectory time.

This is a controlled timing intervention: a candidate batch is materialized
once from one observed RGB-D frame and injected unchanged into two branches
copied from the same prefix snapshot. The branches process the same mapping
frames with an exact per-frame Gaussian micro-update quota. Fixed evaluation
frames are excluded from mapping and scored only after the final frame.
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
from experiments.run_phase13_frozen_benchmark import load_phase10_sequence
from research.densification import create_gaussians_from_candidates, sample_candidates
from research.pipeline import OnlineReconstructionPipeline
from research.pipeline_checkpoint import load_pipeline_checkpoint, save_pipeline_checkpoint
from research.reproducibility import create_provenance_manifest, set_seed


def _memoize_detached_nonleaf_tensors(value, memo, seen=None):
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


def _capture_rng(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _tensor_bundle_sha256(bundle):
    digest = hashlib.sha256()
    for name in sorted(bundle):
        tensor = bundle[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _pipeline_state_sha256(pipeline):
    digest = hashlib.sha256()
    for name, tensor in sorted(pipeline.gaussian_model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    for name, value in sorted(vars(pipeline.gaussian_model.state_store).items()):
        if torch.is_tensor(value):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _new_pipeline(scene, seed, args, frames, intrinsics):
    _, cfg = config_for("error_off", seed, args.budget_ms, args.device)
    cfg["training"]["optimization_control"] = {
        "micro_updates_per_frame": args.micro_updates,
        "selection": "descending_importance_stable",
        "capture_age_diagnostics": True,
        "mature_age_frames": args.mature_age_frames,
    }
    cfg["system"]["synchronized_stage_timing"] = True
    # Disable endogenous births. The replayed batch is the only treatment.
    cfg["densification"]["spawn_control"] = {
        "target_total": 0,
        "start_frame": 1,
        "end_frame": args.n_frames,
        "candidate_mode": "uniform_depth",
    }
    cfg["gaussian"]["max_gaussians"] = max(
        int(cfg["gaussian"].get("max_gaussians", 0)), args.max_gaussians
    )
    pipeline = OnlineReconstructionPipeline(config=cfg, device=args.device)
    first_pose = frames[0].get("pose", torch.eye(4, device=torch.device(args.device)))
    pipeline.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                        intrinsics=intrinsics, pose=first_pose)
    return pipeline


def _process(pipeline, frame, args):
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    start = time.perf_counter()
    metrics = pipeline.process_frame(rgb=frame["rgb"], depth=frame["depth"],
                                     gt_pose=frame.get("pose"))
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    wall_ms = (time.perf_counter() - start) * 1000.0
    return metrics, wall_ms


def _append_batch(pipeline, batch):
    n = int(batch["xyz"].shape[0])
    if n <= 0:
        raise RuntimeError("Candidate batch is empty")
    pipeline.gaussian_model.add_gaussians(batch, frame_idx=pipeline.frame_count)
    pipeline.importance_estimator.expand_buffers(n, pipeline.device)
    if pipeline.optimizer is not None:
        pipeline._reregister_optimizer_params()
        pipeline.optimizer.extend_state(n, device=pipeline.device)
    else:
        pipeline._setup_optimizer()
    return n


def _run_branch(base, rng_state, candidate_batch, frames, intrinsics, scene, seed,
                arm, args, eval_frames, expected_prefix_hash, candidate_hash):
    _restore_rng(rng_state)
    memo = {}
    _memoize_detached_nonleaf_tensors(base, memo)
    pipeline = copy.deepcopy(base, memo)
    prefix_hash = _pipeline_state_sha256(pipeline)
    if prefix_hash != expected_prefix_hash:
        raise RuntimeError("Replay branch does not match the common prefix map snapshot")
    optimization_control = {
        "micro_updates_per_frame": args.micro_updates,
        "selection": "descending_importance_stable",
        "capture_age_diagnostics": True,
        "mature_age_frames": args.mature_age_frames,
    }
    if arm["allocation"] == "stale_rescue":
        optimization_control.update({
            "stale_update_fraction": args.stale_fraction,
            "stale_after_frames": args.stale_after_frames,
        })
    elif arm["allocation"] != "global":
        raise ValueError(f"Unknown update-allocation mode: {arm['allocation']}")
    pipeline.config.setdefault("training", {})["optimization_control"] = optimization_control
    start_count = int(pipeline.gaussian_model.num_gaussians)
    rows, wall_times = [], []
    injected = False
    for raw_frame_idx in range(1, args.n_frames + 1):
        if raw_frame_idx in eval_frames:
            continue
        if raw_frame_idx == arm["insert_frame"]:
            added = _append_batch(pipeline, candidate_batch)
            injected = True
        else:
            added = 0
        metrics, wall_ms = _process(pipeline, frames[raw_frame_idx], args)
        wall_times.append(wall_ms)
        rows.append({
            "dataset_frame": raw_frame_idx,
            "pipeline_frame": int(metrics["frame"]),
            "psnr": float(metrics["psnr"]),
            "ssim": float(metrics["ssim"]),
            "depth_l1": float(metrics["depth_l1"]),
            "n_gaussians": int(metrics["n_gaussians"]),
            "manual_batch_added": added,
            "n_spawned_automatic": int(metrics.get("n_spawned", 0)),
            "n_micro_updates": int(metrics.get("n_micro_updates", 0)),
            "mature_eligible": int(metrics.get("mature_eligible", 0)),
            "mature_updated": int(metrics.get("mature_updated", 0)),
            "mature_update_coverage": float(metrics.get("mature_update_coverage", 0.0)),
            "stale_mature_eligible": int(metrics.get("stale_mature_eligible", 0)),
            "stale_mature_updated": int(metrics.get("stale_mature_updated", 0)),
            "stale_mature_update_coverage": float(metrics.get("stale_mature_update_coverage", 0.0)),
            "stale_mature_micro_updates": int(metrics.get("stale_mature_micro_updates", 0)),
            "stale_quota_micro_updates": metrics.get("stale_quota_micro_updates"),
            "opt_stage_sync_ms": float(metrics.get("opt_stage_sync_ms", 0.0)),
            "frame_wall_sync_ms": wall_ms,
        })
    if not injected:
        raise RuntimeError(f"Batch was not inserted for arm {arm['name']}")
    final_views = []
    for idx in eval_frames:
        out = pipeline.eval_frame(rgb=frames[idx]["rgb"], depth=frames[idx]["depth"],
                                  pose=frames[idx].get("pose", pipeline.current_pose))
        final_views.append({"frame": idx, "psnr": float(out["psnr"]),
                            "ssim": float(out["ssim"]),
                            "depth_l1": float(out["depth_l1"])})
    rows_by_key = {key: [r[key] for r in rows] for key in
                   ("psnr", "ssim", "depth_l1", "n_micro_updates",
                    "mature_update_coverage", "stale_mature_update_coverage",
                    "stale_mature_micro_updates", "opt_stage_sync_ms")}
    record = {
        "arm": arm,
        "prefix_state_sha256": prefix_hash,
        "candidate_batch_sha256": candidate_hash,
        "start_gaussians": start_count,
        "final_gaussians": int(pipeline.gaussian_model.num_gaussians),
        "automatic_births": int(sum(r["n_spawned_automatic"] for r in rows)),
        "manual_batch_births": int(sum(r["manual_batch_added"] for r in rows)),
        "exact_micro_update_gate": all(v == args.micro_updates for v in rows_by_key["n_micro_updates"]),
        "trajectory_mean_psnr": float(np.mean(rows_by_key["psnr"])) if rows else None,
        "trajectory_mean_depth_l1": float(np.mean(rows_by_key["depth_l1"])) if rows else None,
        "mean_mature_update_coverage": float(np.mean(rows_by_key["mature_update_coverage"])) if rows else None,
        "mean_stale_mature_update_coverage": float(np.mean(rows_by_key["stale_mature_update_coverage"])) if rows else None,
        "total_stale_mature_micro_updates": int(sum(rows_by_key["stale_mature_micro_updates"])),
        "realized_stale_update_fraction": (
            float(sum(rows_by_key["stale_mature_micro_updates"]))
            / max(sum(rows_by_key["n_micro_updates"]), 1)
        ),
        "total_micro_updates": int(sum(rows_by_key["n_micro_updates"])),
        "mean_opt_stage_sync_ms": float(np.mean(rows_by_key["opt_stage_sync_ms"])) if rows else None,
        "mean_frame_wall_sync_ms": float(np.mean(wall_times)) if wall_times else None,
        "p95_frame_wall_sync_ms": float(np.percentile(wall_times, 95)) if wall_times else None,
        "fixed_heldout_mean_psnr": float(np.mean([r["psnr"] for r in final_views])) if final_views else None,
        "fixed_heldout_mean_ssim": float(np.mean([r["ssim"] for r in final_views])) if final_views else None,
        "fixed_heldout_mean_depth_l1": float(np.mean([r["depth_l1"] for r in final_views])) if final_views else None,
        "fixed_heldout_views": final_views,
        "final_map_heldout_frames": [int(row["frame"]) for row in final_views],
        "frames": rows,
    }
    pipeline.cleanup()
    del pipeline
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["replica_room2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[57])
    ap.add_argument("--n_frames", type=int, default=119)
    ap.add_argument("--candidate_frame", type=int, default=30,
                    help="Observed RGB-D frame used once to create the frozen batch")
    ap.add_argument("--early_insert_frame", type=int, default=30)
    ap.add_argument("--late_insert_frame", type=int, default=90)
    ap.add_argument("--candidate_count", type=int, default=5000)
    ap.add_argument("--micro_updates", type=int, default=5000)
    ap.add_argument("--mature_age_frames", type=int, default=10)
    ap.add_argument("--allocation_modes", nargs="+", choices=("global", "stale_rescue"),
                    default=["global", "stale_rescue"])
    ap.add_argument("--stale_fraction", type=float, default=0.50)
    ap.add_argument("--stale_after_frames", type=int, default=10)
    ap.add_argument("--heldout_frames", nargs="+", type=int, default=[20, 40, 60, 80, 100, 119])
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--max_gaussians", type=int, default=160000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path,
                    default=ROOT / "results" / "matched_candidate_timing_replay")
    ap.add_argument("--artifact_dir", type=Path, default=None,
                    help="Directory for frozen prefix checkpoints and candidate batches")
    ap.add_argument("--reuse_frozen_artifacts", action="store_true",
                    help="Load existing prefix checkpoint and candidate batch instead of rebuilding them")
    ap.add_argument("--run_id", default="",
                    help="Optional output suffix for repeated replays from one frozen prefix")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (1 <= args.candidate_frame <= args.n_frames):
        ap.error("candidate_frame must lie in [1, n_frames]")
    if args.early_insert_frame != args.candidate_frame:
        ap.error("The early arm must insert the frozen batch at candidate_frame")
    if not (args.early_insert_frame < args.late_insert_frame <= args.n_frames):
        ap.error("Require early_insert_frame < late_insert_frame <= n_frames")
    if args.candidate_count <= 0 or args.micro_updates <= 0:
        ap.error("candidate_count and micro_updates must be positive")
    if not 0.0 <= args.stale_fraction <= 1.0 or args.stale_after_frames < 0:
        ap.error("stale_fraction must be in [0, 1] and stale_after_frames nonnegative")
    if any(i < 1 or i > args.n_frames for i in args.heldout_frames):
        ap.error("heldout frame indices must lie in [1, n_frames]")
    if args.candidate_frame in args.heldout_frames or args.late_insert_frame in args.heldout_frames:
        ap.error("candidate and insertion frames cannot be held out")
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if args.artifact_dir is None:
        args.artifact_dir = args.out_dir / "frozen_artifacts"
    elif not args.artifact_dir.is_absolute():
        args.artifact_dir = ROOT / args.artifact_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)

    protocol_config = {
        "design": "same-prefix, frozen-candidate-parameter replay; insertion timing crossed with global vs stale-mature-rescue update allocation",
        "candidate_frame": args.candidate_frame,
        "early_insert_frame": args.early_insert_frame,
        "late_insert_frame": args.late_insert_frame,
        "candidate_count": args.candidate_count,
        "micro_updates_per_mapped_frame": args.micro_updates,
        "allocation_modes": args.allocation_modes,
        "stale_rescue": {"fraction": args.stale_fraction,
                         "mature_age_frames": args.mature_age_frames,
                         "stale_after_frames": args.stale_after_frames},
        "heldout_frames_excluded_from_mapping": args.heldout_frames,
        "n_frames": args.n_frames,
        "budget_ms_modeled": args.budget_ms,
        "automatic_spawning": "disabled in both arms",
        "primary_endpoints": ["final heldout mean PSNR", "final heldout mean Depth L1",
                              "timing x rescue difference-in-differences"],
        "secondary_endpoints": ["trajectory mean PSNR", "mature/stale update coverage", "measured latency"],
        "status": "exploratory pilot; candidate locations and Gaussian parameters are frozen from the early observation",
    }
    manifest = create_provenance_manifest(
        config={**protocol_config, "scenes": args.scenes, "seeds": args.seeds,
                "device": args.device},
        dataset_name=",".join(args.scenes),
        renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
        extra_metadata={"experiment": "matched_candidate_timing_replay",
                        "pose_convention": "Phase 10 sequence loader ground-truth W2C poses",
                        "seed_list": args.seeds},
    )
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    source_paths = [Path(__file__), ROOT / "research" / "pipeline.py",
                    ROOT / "research" / "densification.py",
                    ROOT / "experiments" / "run_dilution_factorial.py",
                    ROOT / "experiments" / "run_phase13_frozen_benchmark.py"]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    (args.out_dir / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")

    for scene in args.scenes:
        frames, intrinsics = load_phase10_sequence(scene_name=scene,
            n_frames=args.n_frames + 1, H=240, W=320, device=args.device)
        if len(frames) <= args.n_frames:
            raise RuntimeError(f"{scene} has only {len(frames)} loaded frames")
        for seed in args.seeds:
            output_suffix = f"_{args.run_id}" if args.run_id else ""
            out_path = args.out_dir / scene / f"seed{seed}{output_suffix}.json"
            if out_path.exists() and not args.force:
                print(f"SKIP existing {out_path}", flush=True)
                continue
            set_seed(seed)
            base = _new_pipeline(scene, seed, args, frames, intrinsics)
            eval_set = set(args.heldout_frames)
            frozen_dir = args.artifact_dir / scene
            frozen_dir.mkdir(parents=True, exist_ok=True)
            prefix_path = frozen_dir / f"seed{seed}_prefix_frame{args.candidate_frame - 1}.checkpoint.pt"
            candidate_path = frozen_dir / f"seed{seed}_candidate_frame{args.candidate_frame}.pt"
            if args.reuse_frozen_artifacts:
                if not prefix_path.is_file() or not candidate_path.is_file():
                    raise FileNotFoundError(
                        f"Frozen replay artifacts missing: {prefix_path} or {candidate_path}")
                prefix_meta = load_pipeline_checkpoint(
                    prefix_path, base, scene=scene, seed=seed,
                    snapshot_frame=args.candidate_frame - 1, restore_rng=True)
                candidate_payload = torch.load(candidate_path, map_location="cpu", weights_only=True)
                if (candidate_payload.get("scene") != scene
                        or int(candidate_payload.get("seed", -1)) != seed
                        or int(candidate_payload.get("candidate_frame", -1)) != args.candidate_frame
                        or int(candidate_payload.get("candidate_count", -1)) != args.candidate_count):
                    raise ValueError(f"Frozen candidate metadata mismatch: {candidate_path}")
                uv = candidate_payload["uv"].to(args.device)
                candidate_batch = {key: value.to(args.device)
                                   for key, value in candidate_payload["candidate_batch"].items()}
                candidate_hash = _tensor_bundle_sha256(candidate_batch)
                if candidate_hash != candidate_payload["candidate_batch_sha256"]:
                    raise RuntimeError("Frozen candidate parameter hash mismatch")
                if _pipeline_state_sha256(base) != prefix_meta["map_digest"]:
                    raise RuntimeError("Loaded prefix map digest mismatch")
                if int(base.frame_count) != args.candidate_frame - 1:
                    raise RuntimeError("Loaded prefix is at the wrong frame boundary")
                prefix_hash = _pipeline_state_sha256(base)
                base_rng = _capture_rng(args.device)
            else:
                if prefix_path.exists() or candidate_path.exists():
                    raise FileExistsError(
                        f"Frozen artifacts already exist under {frozen_dir}; use --reuse_frozen_artifacts or a new --artifact_dir")
                for frame_idx in range(1, args.candidate_frame):
                    if frame_idx in eval_set:
                        continue
                    _process(base, frames[frame_idx], args)
                # Candidate batch is generated once from the early intervention frame.
                candidate_frame = frames[args.candidate_frame]
                depth_valid = candidate_frame["depth"] > 0
                set_seed(seed * 1009 + args.candidate_frame)
                uv = sample_candidates(depth_valid, args.candidate_count, strategy="uniform")
                if len(uv) != args.candidate_count:
                    raise RuntimeError(f"Candidate pool shortfall: {len(uv)} < {args.candidate_count}")
                pose = candidate_frame.get("pose", base.current_pose)
                candidate_batch = create_gaussians_from_candidates(
                    uv, candidate_frame["rgb"], candidate_frame["depth"], intrinsics, pose,
                    initial_scale=base.config["gaussian"].get("initial_scale", 0.01),
                    initial_opacity=base.config["gaussian"].get("initial_opacity", 0.5),
                    scale_mode=base.config["gaussian"].get("scale_mode", "depth_adaptive"),
                    scale_pixel_multiplier=base.config["gaussian"].get("scale_pixel_multiplier", 1.0),
                )
                candidate_hash = _tensor_bundle_sha256(candidate_batch)
                prefix_hash = _pipeline_state_sha256(base)
                candidate_payload = {
                    "schema": "frozen-candidate-batch-v1", "scene": scene,
                    "seed": int(seed), "candidate_frame": int(args.candidate_frame),
                    "candidate_count": int(args.candidate_count),
                    "candidate_batch_sha256": candidate_hash,
                    "candidate_uv_sha256": hashlib.sha256(
                        uv.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
                    "uv": uv.detach().contiguous().cpu(),
                    "candidate_batch": {key: value.detach().contiguous().cpu()
                                        for key, value in candidate_batch.items()},
                }
                torch.save(candidate_payload, candidate_path)
                save_pipeline_checkpoint(
                    prefix_path, base, scene=scene, seed=seed,
                    snapshot_frame=args.candidate_frame - 1)
                base_rng = _capture_rng(args.device)
            prefix_checkpoint_sha256 = hashlib.sha256(prefix_path.read_bytes()).hexdigest()
            candidate_file_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
            arms = []
            timing_arms = (("early", args.early_insert_frame),
                           ("late", args.late_insert_frame))
            for allocation in args.allocation_modes:
                for timing_name, insert_frame in timing_arms:
                    arm = {"name": f"{timing_name}_{allocation}",
                           "timing": timing_name, "allocation": allocation,
                           "insert_frame": insert_frame}
                    print(f"RUN {scene} seed={seed} arm={arm['name']} insert={arm['insert_frame']} "
                          f"candidates={args.candidate_count}", flush=True)
                    arms.append(_run_branch(base, base_rng, candidate_batch, frames, intrinsics,
                                            scene, seed, arm, args, sorted(eval_set), prefix_hash,
                                            candidate_hash))
            by_condition = {f"{arm['arm']['timing']}_{arm['arm']['allocation']}": arm
                            for arm in arms}
            metric_keys = ("trajectory_mean_psnr", "trajectory_mean_depth_l1",
                           "mean_mature_update_coverage", "mean_stale_mature_update_coverage",
                           "mean_opt_stage_sync_ms", "fixed_heldout_mean_psnr",
                           "fixed_heldout_mean_ssim", "fixed_heldout_mean_depth_l1")
            timing_contrasts = {}
            for allocation in args.allocation_modes:
                early_arm = by_condition[f"early_{allocation}"]
                late_arm = by_condition[f"late_{allocation}"]
                timing_contrasts[allocation] = {key: late_arm[key] - early_arm[key]
                                                for key in metric_keys}
            rescue_contrasts = {}
            if "global" in args.allocation_modes and "stale_rescue" in args.allocation_modes:
                for timing_name in ("early", "late"):
                    global_arm = by_condition[f"{timing_name}_global"]
                    rescue_arm = by_condition[f"{timing_name}_stale_rescue"]
                    rescue_contrasts[timing_name] = {
                        key: rescue_arm[key] - global_arm[key] for key in metric_keys
                    }
                difference_in_differences = {
                    key: timing_contrasts["stale_rescue"][key] - timing_contrasts["global"][key]
                    for key in metric_keys
                }
            else:
                difference_in_differences = None
            record = {
                "schema": "matched-candidate-timing-rescue-factorial-v1",
                "experiment": "frozen candidate insertion timing x stale-mature update rescue factorial",
                "scene": scene, "seed": seed,
                "protocol": protocol_config,
                "candidate_batch_sha256": candidate_hash,
                "candidate_uv_sha256": hashlib.sha256(uv.detach().cpu().numpy().tobytes()).hexdigest(),
                "prefix_state_sha256": prefix_hash,
                "frozen_prefix_checkpoint": str(prefix_path.relative_to(ROOT)),
                "frozen_prefix_checkpoint_sha256": prefix_checkpoint_sha256,
                "frozen_candidate_file": str(candidate_path.relative_to(ROOT)),
                "frozen_candidate_file_sha256": candidate_file_sha256,
                "candidate_uv": uv.detach().cpu().tolist(),
                "prefix_frame_count": int(base.frame_count),
                "prefix_gaussians": int(base.gaussian_model.num_gaussians),
                "arms": arms,
                "arms_by_condition": {name: {
                    key: value[key] for key in metric_keys + ("final_gaussians",
                        "mean_stale_mature_update_coverage", "realized_stale_update_fraction",
                        "total_stale_mature_micro_updates")
                } for name, value in by_condition.items()},
                "late_minus_early_by_allocation": timing_contrasts,
                "stale_rescue_minus_global_by_timing": rescue_contrasts,
                "timing_by_rescue_difference_in_differences": difference_in_differences,
                "valid": all(a["exact_micro_update_gate"] and a["automatic_births"] == 0
                             and a["manual_batch_births"] == args.candidate_count for a in arms)
                    and all(a["start_gaussians"] == int(base.gaussian_model.num_gaussians)
                            for a in arms)
                    and all(a["prefix_state_sha256"] == prefix_hash for a in arms)
                    and all(a["candidate_batch_sha256"] == candidate_hash for a in arms)
                    and len({tuple(a["final_map_heldout_frames"]) for a in arms}) == 1
                    and len({a["total_micro_updates"] for a in arms}) == 1,
                "replayed_from_frozen_artifacts": bool(args.reuse_frozen_artifacts),
                "run_id": args.run_id or "initial",
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2) + "\n")
            tmp.replace(out_path)
            print(f"DONE {out_path} valid={record['valid']} "
                  f"DiD_PSNR={record['timing_by_rescue_difference_in_differences']['fixed_heldout_mean_psnr']:+.4f} "
                  f"DiD_depth={record['timing_by_rescue_difference_in_differences']['fixed_heldout_mean_depth_l1']:+.5f}",
                  flush=True)
            base.cleanup()
            del base
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
