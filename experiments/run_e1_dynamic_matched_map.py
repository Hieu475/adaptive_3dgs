#!/usr/bin/env python3
"""Paired online E1 pilot with Gaussian count matched after every frame.

The two policies start from identical initialization and process the same
RGB-D frame with aligned RNG state. After both updates, the larger map is
pruned by the configured selector to the smaller active count. Both maps then
continue from the matched count into the next frame. This controls count
trajectories, but not Gaussian identity or pruning history.
"""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiments.run_dilution_factorial import config_for
from experiments.run_frozen_benchmark import load_benchmark_sequence
from research.gaussian_repr import GaussianState
from research.pipeline import OnlineReconstructionPipeline
from research.reproducibility import create_provenance_manifest, set_seed

ROOT = Path(__file__).resolve().parents[1]


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


@torch.no_grad()
def compact(pipeline):
    model = pipeline.gaussian_model
    keep_mask = model._state != GaussianState.PRUNED
    if not bool(keep_mask.all()):
        keep = model.compact()
        pipeline.importance_estimator.prune_buffers(keep)
        pipeline.optimizer.prune_state(keep)
        pipeline._reregister_optimizer_params()
    return int(model.num_gaussians)


@torch.no_grad()
def prune_to_count(pipeline, target, strategy="lowest_opacity", random_seed=0):
    current = compact(pipeline)
    if current < target:
        raise RuntimeError(f"cannot increase map from {current} to {target}")
    remove_n = current - int(target)
    if remove_n:
        opacity = pipeline.gaussian_model.opacities.squeeze(-1)
        if strategy == "lowest_opacity":
            remove = torch.topk(opacity, k=remove_n, largest=False).indices
        elif strategy == "uniform_random":
            generator = torch.Generator(device=opacity.device)
            generator.manual_seed(int(random_seed))
            remove = torch.rand(opacity.shape, generator=generator,
                                device=opacity.device).topk(remove_n, largest=False).indices
        else:
            raise ValueError(f"Unknown count-match pruning strategy: {strategy}")
        mask = torch.zeros(current, dtype=torch.bool, device=opacity.device)
        mask[remove] = True
        pipeline.gaussian_model.prune_gaussians(mask)
        keep = pipeline.gaussian_model.compact()
        pipeline.importance_estimator.prune_buffers(keep)
        pipeline.optimizer.prune_state(keep)
        pipeline._reregister_optimizer_params()
    return int(pipeline.gaussian_model.num_gaussians)


def make_pipeline(policy, seed, budget_ms, device, frames, intrinsics):
    set_seed(seed)
    _, cfg = config_for(policy, seed, budget_ms, device)
    pipeline = OnlineReconstructionPipeline(config=copy.deepcopy(cfg), device=device)
    pipeline.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                        intrinsics=intrinsics,
                        pose=frames[0].get("pose", torch.eye(4, device=device)))
    return pipeline, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="replica_office1")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--n_frames", type=int, default=119,
                        help="Online frames after the shared initialization frame")
    parser.add_argument("--budget_ms", type=float, default=15.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_schedule", type=Path, default=None,
                        help="Frozen exogenous per-frame target JSON; omit for endogenous lower-of-pair schedule")
    parser.add_argument("--prune_strategy", choices=("lowest_opacity", "uniform_random"),
                        default="lowest_opacity",
                        help="How to choose Gaussians when an arm must be pruned to match counts")
    parser.add_argument("--out_dir", type=Path,
                        default=ROOT / "results/replica_e1_dynamic_matched_map_office1_seed44")
    args = parser.parse_args()
    output = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    schedule_record = None
    schedule_targets = None
    schedule_sha256 = None
    schedule_scene_key = args.scene.removeprefix("replica_")
    if args.target_schedule is not None:
        schedule_path = args.target_schedule if args.target_schedule.is_absolute() else ROOT / args.target_schedule
        schedule_record = json.loads(schedule_path.read_text())
        if schedule_scene_key not in schedule_record.get("scenes", {}):
            raise ValueError(f"No frozen count schedule for {args.scene}: {schedule_path}")
        schedule_targets = schedule_record["scenes"][schedule_scene_key]["targets"]
        schedule_sha256 = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    if str(args.device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)

    frames, intrinsics = load_benchmark_sequence(scene_name=args.scene,
        n_frames=args.n_frames + 1, H=240, W=320, device=args.device)
    if len(frames) < 2:
        raise ValueError("Need at least an initialization frame and one online frame")
    actual_online = min(args.n_frames, len(frames) - 1)
    a, cfg_a = make_pipeline("error_off", args.seed, args.budget_ms,
                             args.device, frames, intrinsics)
    b, cfg_b = make_pipeline("error_coverage", args.seed, args.budget_ms,
                             args.device, frames, intrinsics)
    if a.gaussian_model.num_gaussians != b.gaussian_model.num_gaussians:
        raise RuntimeError("The policies did not start from the same map size")
    if schedule_targets is not None:
        if len(schedule_targets) < actual_online:
            raise ValueError(f"Frozen schedule has {len(schedule_targets)} targets, needs {actual_online}")
        expected_initial = int(schedule_record["scenes"][schedule_scene_key]["initial_gaussians"])
        if expected_initial != int(a.gaussian_model.num_gaussians):
            raise ValueError(f"Frozen schedule initialization count {expected_initial} != actual {a.gaussian_model.num_gaussians}")

    rows = []
    schedule_gate_failure = None
    start = time.perf_counter()
    for t in range(1, actual_online + 1):
        frame = frames[t]
        shared_rng = capture_rng(args.device)
        metrics = {}
        post_rng = None
        for name, pipeline in (("no_throttle", a), ("coverage_throttle", b)):
            restore_rng(shared_rng)
            metrics[name] = pipeline.process_frame(
                rgb=frame["rgb"], depth=frame["depth"], gt_pose=frame.get("pose"))
            if name == "no_throttle":
                post_rng = capture_rng(args.device)
        # Advance the shared random stream once per frame.
        restore_rng(post_rng)

        pre_count = {"no_throttle": compact(a), "coverage_throttle": compact(b)}
        pre_match_evals = {
            "no_throttle": a.eval_frame(frame["rgb"], frame["depth"], frame.get("pose", a.current_pose)),
            "coverage_throttle": b.eval_frame(frame["rgb"], frame["depth"], frame.get("pose", b.current_pose)),
        }
        target_count = (int(schedule_targets[t - 1]) if schedule_targets is not None
                        else min(pre_count.values()))
        if min(pre_count.values()) < target_count:
            schedule_gate_failure = {"frame": t, "requested_target": target_count,
                                     "realized_counts": pre_count}
            rows.append({"online_frame": t, "count_match_gate_passed": False,
                         "target_gaussian_count": target_count,
                         "pre_match_counts": pre_count,
                         "policies": {name: {
                             "pre_match_psnr": float(pre_match_evals[name]["psnr"]),
                             "pre_match_ssim": float(pre_match_evals[name]["ssim"]),
                             "pre_match_depth_l1": float(pre_match_evals[name]["depth_l1"]),
                         } for name in ("no_throttle", "coverage_throttle")}})
            print(f"COUNT SCHEDULE GATE FAILED at frame {t}: target={target_count}, counts={pre_count}", flush=True)
            break
        prune_count = {
            "no_throttle": pre_count["no_throttle"] - target_count,
            "coverage_throttle": pre_count["coverage_throttle"] - target_count,
        }
        realized = {
            "no_throttle": prune_to_count(a, target_count, args.prune_strategy,
                                           args.seed * 1000003 + t * 2),
            "coverage_throttle": prune_to_count(b, target_count, args.prune_strategy,
                                                  args.seed * 1000003 + t * 2 + 1),
        }
        if realized["no_throttle"] != realized["coverage_throttle"]:
            raise RuntimeError(f"matched-count gate failed at frame {t}: {realized}")
        evals = {
            "no_throttle": a.eval_frame(frame["rgb"], frame["depth"], frame.get("pose", a.current_pose)),
            "coverage_throttle": b.eval_frame(frame["rgb"], frame["depth"], frame.get("pose", b.current_pose)),
        }
        rows.append({
            "online_frame": t,
            "count_match_gate_passed": True,
            "matched_gaussian_count": target_count,
            "pre_match_counts": pre_count,
            "pruned_for_count_match": prune_count,
            "policies": {
                name: {
                    "pre_match_psnr": float(pre_match_evals[name]["psnr"]),
                    "pre_match_ssim": float(pre_match_evals[name]["ssim"]),
                    "pre_match_depth_l1": float(pre_match_evals[name]["depth_l1"]),
                    "psnr": float(evals[name]["psnr"]),
                    "ssim": float(evals[name]["ssim"]),
                    "depth_l1": float(evals[name]["depth_l1"]),
                    "n_spawned": int(metrics[name].get("n_spawned", 0)),
                    "mature_update_coverage": float(metrics[name].get("mature_update_coverage", 0.0)),
                    "growth_pressure": float(metrics[name].get("growth_pressure", 0.0)),
                    "frame_time_ms": float(metrics[name].get("frame_time_ms", 0.0)),
                } for name in ("no_throttle", "coverage_throttle")
            },
        })
        if t % 20 == 0 or t == actual_online:
            print(f"[{t}/{actual_online}] matched N={target_count:,} | "
                  f"PSNR no-throttle/coverage="
                  f"{rows[-1]['policies']['no_throttle']['psnr']:.2f}/"
                  f"{rows[-1]['policies']['coverage_throttle']['psnr']:.2f}", flush=True)

    elapsed = time.perf_counter() - start
    a.cleanup()
    b.cleanup()
    del a, b
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    matched_rows = [r for r in rows if r.get("count_match_gate_passed")]
    config_payload = {"no_throttle": cfg_a, "coverage_throttle": cfg_b,
                      "seed": args.seed, "scene": args.scene,
                      "online_frames": actual_online, "budget_ms": args.budget_ms,
                      "prune_strategy": args.prune_strategy,
                      "schedule_id": schedule_record.get("schedule_id") if schedule_record else "endogenous_lower_of_pair",
                      "schedule_sha256": schedule_sha256}
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True,
        separators=(",", ":"), default=str).encode()).hexdigest()
    deltas = {
        field: [r["policies"]["coverage_throttle"][field] -
                r["policies"]["no_throttle"][field] for r in matched_rows]
        for field in ("psnr", "ssim", "depth_l1", "mature_update_coverage", "growth_pressure")
    }
    summary = {f"mean_delta_coverage_minus_no_throttle_{field}": float(np.mean(values)) if values else None
               for field, values in deltas.items()}
    for field in ("psnr", "ssim", "depth_l1"):
        summary[f"mean_pre_match_delta_coverage_minus_no_throttle_{field}"] = float(np.mean([
            r["policies"]["coverage_throttle"][f"pre_match_{field}"] -
            r["policies"]["no_throttle"][f"pre_match_{field}"] for r in matched_rows])) if matched_rows else None
        for policy in ("no_throttle", "coverage_throttle"):
            summary[f"mean_matching_prune_delta_{policy}_{field}"] = float(np.mean([
                r["policies"][policy][field] - r["policies"][policy][f"pre_match_{field}"]
                for r in matched_rows])) if matched_rows else None
    summary.update({
        "processed_frames": len(matched_rows),
        "requested_online_frames": actual_online,
        "matched_count_gate_passed_every_frame": schedule_gate_failure is None and len(matched_rows) == actual_online,
        "mean_matched_gaussians": float(np.mean([r["matched_gaussian_count"] for r in matched_rows])) if matched_rows else None,
        "total_pruned_no_throttle_for_matching": int(sum(r["pruned_for_count_match"]["no_throttle"] for r in matched_rows)),
        "total_pruned_coverage_for_matching": int(sum(r["pruned_for_count_match"]["coverage_throttle"] for r in matched_rows)),
        "paired_frames_coverage_psnr_wins": int(sum(x > 0 for x in deltas["psnr"])),
        "paired_frames_coverage_psnr_losses": int(sum(x < 0 for x in deltas["psnr"])),
        "wall_time_sec_excluding_preload": elapsed,
    })
    provenance = create_provenance_manifest(
        config=config_payload, dataset_name=args.scene, renderer_backend="gsplat" if str(args.device).startswith("cuda") else "reference",
        extra_metadata={"experiment": "E1_dynamic_matched_map_size", "status": "exploratory fixed-schedule pilot" if schedule_targets is not None else "single-scene endogenous-schedule pilot",
                        "pose_convention": "Replica ground-truth W2C extrinsics from Phase 10 loader",
                        "count_match": f"after every paired online frame, both maps are pruned by {args.prune_strategy} to the lower realized count"})
    result = {
        "experiment": "E1_dynamic_matched_map_size",
        "status": "count_schedule_gate_failed" if schedule_gate_failure else "exploratory_single_scene_single_seed",
        "scene": args.scene, "seed": args.seed, "online_frames": actual_online,
        "device": args.device, "protocol_version": "dynamic-e1-frame-matched-count-v3",
        "config_sha256": config_hash, "elapsed_sec_excluding_preload": elapsed,
        "target_schedule_id": schedule_record.get("schedule_id") if schedule_record else "endogenous_lower_of_pair",
        "target_schedule_sha256": schedule_sha256,
        "count_schedule_gate_failure": schedule_gate_failure,
        "definition": f"Both policies process the same frame from identical prior map counts; after both updates, {args.prune_strategy} pruning reduces each map to the frozen per-frame target, and both continue from that matched count." if schedule_targets is not None else f"Both policies process the same frame from identical prior map counts; after both updates, {args.prune_strategy} pruning reduces each map to the smaller realized count, and both continue from that matched count.",
        "prune_strategy": args.prune_strategy,
        "treatment": "coverage throttle versus no throttle",
        "summary": summary, "frames": rows, "provenance": provenance,
        "limitations": [
            "One Replica scene and one seed; exploratory only.",
            "Matching map cardinality does not match Gaussian identities or spatial coverage.",
            f"{args.prune_strategy} post-frame pruning is itself an intervention and may remove different useful primitives.",
            "In fixed-schedule mode, targets are calibrated from earlier runs; in endogenous mode, targets are the lower realized post-update population.",
            "Ground-truth pose mapping-only evaluation; not a tracking or long-horizon result.",
        ],
    }
    (output / "dynamic_matched_map_results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# E1 dynamic matched-map-size pilot", "",
        f"Scene `{args.scene}`, seed {args.seed}, {summary['processed_frames']}/{actual_online} online frames matched. "
        + (f"Targets came from frozen schedule `{result['target_schedule_id']}`."
           if schedule_targets is not None else "Targets used the lower realized arm count each frame.")
        + f" Both arms were pruned by {args.prune_strategy} before continuing.", "",
        "| Metric (coverage − no-throttle) | Before count match | After count match |", "|:--|--:|--:|",
    ]
    for field in ("psnr", "ssim", "depth_l1"):
        before = summary[f"mean_pre_match_delta_coverage_minus_no_throttle_{field}"]
        after = summary[f"mean_delta_coverage_minus_no_throttle_{field}"]
        lines.append(f"| {field} | {before:+.6f} | {after:+.6f} |" if before is not None and after is not None
                     else f"| {field} | NA | NA |")
    for field in ("mature_update_coverage", "growth_pressure"):
        after = summary[f"mean_delta_coverage_minus_no_throttle_{field}"]
        lines.append(f"| {field} | — | {after:+.6f} |" if after is not None else f"| {field} | — | NA |")
    lines += [
        "", f"Matched count gate: **{'PASS' if summary['matched_count_gate_passed_every_frame'] else 'FAIL'}**; "
        f"mean map size {summary['mean_matched_gaussians']:.0f}; count-match pruning removed "
        f"{summary['total_pruned_no_throttle_for_matching']:,} no-throttle and "
        f"{summary['total_pruned_coverage_for_matching']:,} coverage-throttle Gaussians over the run.", "",
        f"Mean PSNR change caused by count-match pruning: no-throttle {summary['mean_matching_prune_delta_no_throttle_psnr']:+.4f} dB; "
        f"coverage-throttle {summary['mean_matching_prune_delta_coverage_throttle_psnr']:+.4f} dB.", "",
        f"Per-frame PSNR: coverage wins {summary['paired_frames_coverage_psnr_wins']}, loses {summary['paired_frames_coverage_psnr_losses']}.", "",
        "This pilot equalizes map cardinality after every paired update, but Gaussian identities and spatial coverage still differ. The lower realized map size defines an endogenous schedule, and opacity-based pruning is an additional treatment. One scene/seed is exploratory, not a general causal estimate.", "",
    ] + [f"- {item}" for item in result["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output / 'report.md'}")


if __name__ == "__main__":
    main()
