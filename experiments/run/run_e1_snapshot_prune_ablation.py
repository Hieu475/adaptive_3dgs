#!/usr/bin/env python3
"""One-step pruning-selector counterfactuals on unchanged natural map snapshots."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.run.run_e1_dynamic_matched_map import capture_rng, compact, make_pipeline, restore_rng
from experiments.run.run_frozen_benchmark import load_benchmark_sequence
from src.core.projection import world_to_camera
from src.core.rasterizer import render as rasterize_scene
from src.evaluation.reproducibility import create_provenance_manifest

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def finite_trace(pipeline):
    estimator = pipeline.importance_estimator
    model = pipeline.gaussian_model
    state = {
        "confidence": model._confidence,
        "positions": model.positions,
        "log_scales": model._scaling,
        "rotations": model._rotation,
        "opacity_logits": model._opacity,
        "covariance": model.build_covariance(),
        "running_depth_error": estimator._running_depth_error,
        "running_color_error": estimator._running_color_error,
        "running_normal_error": estimator._running_normal_error,
        "running_error_fast": estimator._running_error_fast,
        "running_error_slow": estimator._running_error_slow,
        "visibility_count": estimator._visibility_count,
        "screen_area": estimator._screen_areas,
    }
    out = {}
    for name, tensor in state.items():
        if tensor is None or tensor.numel() == 0:
            out[name] = {"count": 0, "finite_count": 0, "nonfinite_count": 0}
        else:
            finite = torch.isfinite(tensor)
            out[name] = {"count": int(tensor.numel()),
                         "finite_count": int(finite.sum()),
                         "nonfinite_count": int((~finite).sum())}
    camera_z = world_to_camera(model.positions, pipeline.current_pose)[:, 2]
    bad_area = (~torch.isfinite(estimator._screen_areas)
                if estimator._screen_areas is not None else torch.zeros_like(camera_z, dtype=torch.bool))
    behind = camera_z <= 0.1
    out["projection_diagnostic"] = {
        "behind_or_near_camera_count": int(behind.sum()),
        "screen_area_nonfinite_behind_or_near_count": int((bad_area & behind).sum()),
        "screen_area_nonfinite_in_front_count": int((bad_area & ~behind).sum()),
    }
    return out


@torch.no_grad()
def eval_kept(pipeline, frame, target, selector, random_seed):
    model = pipeline.gaussian_model
    n = int(model.num_gaussians)
    if n < target:
        raise RuntimeError(f"snapshot has {n} Gaussians, below requested target {target}")
    remove_n = n - int(target)
    if remove_n == 0:
        keep = torch.ones(n, dtype=torch.bool, device=model.positions.device)
        removed = torch.zeros(n, dtype=torch.bool, device=model.positions.device)
    elif selector == "lowest_opacity":
        opacity = model.opacities.squeeze(-1)
        remove = torch.topk(opacity, k=remove_n, largest=False).indices
        keep = torch.ones(n, dtype=torch.bool, device=opacity.device)
        keep[remove] = False
        removed = ~keep
    elif selector == "uniform_random":
        device = model.positions.device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(random_seed))
        remove = torch.rand(n, generator=generator, device=device).topk(remove_n, largest=False).indices
        keep = torch.ones(n, dtype=torch.bool, device=device)
        keep[remove] = False
        removed = ~keep
    else:
        raise ValueError(selector)
    pose = frame.get("pose", pipeline.current_pose).to(pipeline.device)
    H, W = frame["depth"].shape
    subset = model.get_optimization_subset(keep, extrinsics=pose)
    backend = pipeline.config.get("rendering", {}).get(
        "backend", "gsplat" if str(pipeline.device).startswith("cuda") else "reference")
    rendered = rasterize_scene(
        means3D=subset["means3D"], cov3D=subset["cov3D"], colors=subset["colors"],
        opacities=subset["opacities"], extrinsics=pose, intrinsics=pipeline.intrinsics,
        image_width=W, image_height=H,
        tile_size=pipeline.config["rendering"]["tile_size"], backend=backend)
    rgb, depth = frame["rgb"].to(pipeline.device), frame["depth"].to(pipeline.device)
    color, pred_depth = rendered["color"], rendered["depth"]
    valid = depth > 0
    mse = (((color[valid] - rgb[valid]) ** 2).mean() if valid.any()
           else ((color - rgb) ** 2).mean()) + 1e-8
    depth_l1 = float((pred_depth[valid] - depth[valid]).abs().mean()) if valid.any() else 0.0
    psnr = float(-10 * torch.log10(mse))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = color.mean(dim=(0, 1)), rgb.mean(dim=(0, 1))
    var1 = ((color - mu1) ** 2).mean(dim=(0, 1))
    var2 = ((rgb - mu2) ** 2).mean(dim=(0, 1))
    cov12 = ((color - mu1) * (rgb - mu2)).mean(dim=(0, 1))
    ssim = float((((2 * mu1 * mu2 + c1) * (2 * cov12 + c2)) /
                  ((mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2))).mean())
    opacity_all = model.opacities.squeeze(-1)
    ages = model.state_store.ages
    confidence = model._confidence.squeeze(-1)
    lifecycle = model._state
    removed_n = int(removed.sum())
    if removed_n:
        pos = model.positions
        lo, hi = pos.amin(dim=0), pos.amax(dim=0)
        bins = (((pos - lo) / (hi - lo).clamp_min(1e-6) * 4).long().clamp(0, 3))
        codes = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
        removed_ages = ages[removed]
        removed_lifecycle = lifecycle[removed]
        removed_confidence = confidence[removed]
        finite_confidence = torch.isfinite(removed_confidence)
        confidence_finite_count = int(finite_confidence.sum())
        spatial_occupancy = float(torch.unique(codes[removed]).numel() / 64.0)
        removed_diag = {
            "mean_opacity": float(opacity_all[removed].mean()),
            "median_opacity": float(opacity_all[removed].median()),
            "sum_opacity": float(opacity_all[removed].sum()),
            "mean_confidence": (float(removed_confidence[finite_confidence].mean())
                                if confidence_finite_count else None),
            "confidence_finite_count": confidence_finite_count,
            "confidence_nonfinite_fraction": float((~finite_confidence).float().mean()),
            "mean_age_frames": float(removed_ages.float().mean()),
            "mature_fraction_age_ge_10": float((removed_ages >= 10).float().mean()),
            "stable_or_frozen_fraction": float((removed_lifecycle >= 1).float().mean()),
            "spatial_occupancy_4cube": spatial_occupancy,
        }
    else:
        removed_diag = {k: 0.0 for k in (
            "mean_opacity", "median_opacity", "sum_opacity", "mean_confidence",
            "mean_age_frames", "mature_fraction_age_ge_10",
            "stable_or_frozen_fraction", "spatial_occupancy_4cube",
            "confidence_nonfinite_fraction")}
        removed_diag["confidence_finite_count"] = 0
    removed_diag["count"] = removed_n
    return {"psnr": psnr, "ssim": ssim, "depth_l1": depth_l1,
            "n_gaussians": int(keep.sum()), "pruned": remove_n,
            "removed_gaussian_diagnostics": removed_diag}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="replica_office1")
    ap.add_argument("--seed", type=int, default=57)
    ap.add_argument("--n_frames", type=int, default=119)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if str(args.device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    frames, intrinsics = load_benchmark_sequence(
        scene_name=args.scene, n_frames=args.n_frames + 1, H=240, W=320, device=args.device)
    online = min(args.n_frames, len(frames) - 1)
    no, cfg_no = make_pipeline("error_off", args.seed, args.budget_ms,
                               args.device, frames, intrinsics)
    cov, cfg_cov = make_pipeline("error_coverage", args.seed, args.budget_ms,
                                 args.device, frames, intrinsics)
    if no.gaussian_model.num_gaussians != cov.gaussian_model.num_gaussians:
        raise RuntimeError("Policies did not start with an equal map size")

    rows, t0 = [], time.perf_counter()
    for t in range(1, online + 1):
        frame = frames[t]
        rng = capture_rng(args.device)
        post_rng = None
        process_metrics = {}
        for name, pipe in (("no_throttle", no), ("coverage_throttle", cov)):
            restore_rng(rng)
            process_metrics[name] = pipe.process_frame(
                rgb=frame["rgb"], depth=frame["depth"], gt_pose=frame.get("pose"))
            if name == "no_throttle":
                post_rng = capture_rng(args.device)
        restore_rng(post_rng)
        counts = {"no_throttle": compact(no), "coverage_throttle": compact(cov)}
        finite = {"no_throttle": finite_trace(no), "coverage_throttle": finite_trace(cov)}
        target = min(counts.values())
        pre = {"no_throttle": no.eval_frame(frame["rgb"], frame["depth"], frame.get("pose", no.current_pose)),
               "coverage_throttle": cov.eval_frame(frame["rgb"], frame["depth"], frame.get("pose", cov.current_pose))}
        counterfactuals = {}
        for name, pipe in (("no_throttle", no), ("coverage_throttle", cov)):
            counterfactuals[name] = {}
            for selector_idx, selector in enumerate(("lowest_opacity", "uniform_random")):
                counterfactuals[name][selector] = eval_kept(
                    pipe, frame, target, selector,
                    random_seed=args.seed * 1000003 + t * 2 + selector_idx)
        rows.append({
            "online_frame": t, "natural_counts": counts, "snapshot_target": target,
            "finite_state": finite,
            "natural_metrics": {k: {m: float(pre[k][m]) for m in ("psnr", "ssim", "depth_l1")}
                                for k in pre},
            "counterfactuals": counterfactuals,
            "natural_process_metrics": {k: {
                "n_spawned": int(v.get("n_spawned", 0)),
                "mature_update_coverage": float(v.get("mature_update_coverage", 0.0)),
                "growth_pressure": float(v.get("growth_pressure", 0.0)),
            } for k, v in process_metrics.items()},
        })
        if t % 20 == 0 or t == online:
            print(f"[{t}/{online}] natural N={counts}; snapshot target={target:,}", flush=True)

    elapsed = time.perf_counter() - t0
    no.cleanup(); cov.cleanup()
    selectors = ("lowest_opacity", "uniform_random")
    deltas = {selector: {
        metric: [r["counterfactuals"]["coverage_throttle"][selector][metric]
                 - r["counterfactuals"]["no_throttle"][selector][metric] for r in rows]
        for metric in ("psnr", "ssim", "depth_l1")}
        for selector in selectors}
    prune_effect = {policy: {selector: {
        metric: [r["counterfactuals"][policy][selector][metric]
                 - r["natural_metrics"][policy][metric] for r in rows]
        for metric in ("psnr", "ssim", "depth_l1")}
        for selector in selectors}
        for policy in ("no_throttle", "coverage_throttle")}
    summary = {
        "processed_frames": len(rows), "requested_online_frames": online,
        "all_frame_targets_exact": all(
            r["counterfactuals"][p][s]["n_gaussians"] == r["snapshot_target"]
            for r in rows for p in ("no_throttle", "coverage_throttle") for s in selectors),
        "mean_natural_count_no_throttle": float(np.mean([r["natural_counts"]["no_throttle"] for r in rows])),
        "mean_natural_count_coverage_throttle": float(np.mean([r["natural_counts"]["coverage_throttle"] for r in rows])),
        "mean_snapshot_target": float(np.mean([r["snapshot_target"] for r in rows])),
        "first_nonfinite_frame_by_policy_and_state": {
            policy: {name: next((r["online_frame"] for r in rows
                                if r["finite_state"][policy][name]["nonfinite_count"] > 0), None)
                     for name in rows[0]["finite_state"][policy] if name != "projection_diagnostic"}
            for policy in ("no_throttle", "coverage_throttle")},
        "mean_counterfactual_delta_coverage_minus_no_throttle": {
            s: {m: float(np.mean(v)) for m, v in deltas[s].items()} for s in selectors},
        "mean_immediate_prune_effect_by_policy": {
            p: {s: {m: float(np.mean(v)) for m, v in prune_effect[p][s].items()} for s in selectors}
            for p in ("no_throttle", "coverage_throttle")},
        "wall_time_sec_excluding_preload": elapsed,
    }
    config = {"scene": args.scene, "seed": args.seed, "online_frames": online,
              "budget_ms": args.budget_ms, "selectors": selectors,
              "target_rule": "per-frame minimum natural post-update count across the two policies",
              "mapping_trajectory": "natural, no pruning applied to live pipelines"}
    manifest = create_provenance_manifest(
        config={"no_throttle": cfg_no, "coverage_throttle": cfg_cov, **config},
        dataset_name=args.scene,
        renderer_backend="gsplat" if str(args.device).startswith("cuda") else "reference",
        extra_metadata={"experiment": "E1_same_snapshot_prune_selector_ablation",
                        "status": "exploratory one-step counterfactual",
                        "pose_convention": "Replica ground-truth W2C extrinsics from Phase 10 loader"})
    result = {"experiment": "E1_same_snapshot_prune_selector_ablation",
              "status": "exploratory", "protocol_version": "e1-snapshot-prune-ablation-v1",
              "scene": args.scene, "seed": args.seed, "online_frames": online,
              "config": config, "summary": summary, "frames": rows, "provenance": manifest,
              "limitations": [
                  "Prune outcomes are one-step render counterfactuals; the live mapping pipelines are not pruned and continue naturally.",
                  "Target is endogenous per frame, selected from the lower natural population.",
                  "Same-snapshot selector comparisons do not test downstream optimization after pruning.",
                  "Ground-truth pose mapping-only evaluation; one scene/seed is exploratory.",
              ]}
    (out / "snapshot_prune_ablation.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Same-snapshot E1 prune-selector ablation", "",
        f"Scene `{args.scene}`, seed {args.seed}; {len(rows)}/{online} frames. The natural pipelines are never pruned; each selector renders a same-snapshot counterfactual at the per-frame lower natural count.", "",
        "| Selector | Mean PSNR Δ coverage−no-throttle | Mean Depth L1 Δ | Mean no-throttle immediate prune PSNR change | Mean coverage immediate prune PSNR change |", "|:--|--:|--:|--:|--:|",
    ]
    for selector in selectors:
        d = summary["mean_counterfactual_delta_coverage_minus_no_throttle"][selector]
        p = summary["mean_immediate_prune_effect_by_policy"]
        lines.append(f"| {selector} | {d['psnr']:+.3f} dB | {d['depth_l1']:+.5f} m | {p['no_throttle'][selector]['psnr']:+.3f} dB | {p['coverage_throttle'][selector]['psnr']:+.3f} dB |")
    lines += ["", f"Exact per-frame subset count gate: **{'PASS' if summary['all_frame_targets_exact'] else 'FAIL'}**; mean target {summary['mean_snapshot_target']:.0f} Gaussians.", "",
              "Confidence/importance state finite-count trace is saved per frame for root-cause analysis.", "",
              "This compares pruning selectors on identical per-policy snapshots and identical per-frame target counts, without allowing pruning to alter future frames. It measures immediate render damage only; it does not estimate the effect of carrying a pruned map into later optimization.", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(out / "report.md")


if __name__ == "__main__":
    main()
