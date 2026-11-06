#!/usr/bin/env python3
"""Compare direct/separable SSIM gradients through a real gsplat render.

Restore a full replay checkpoint, render one fixed RGB-D view once, then
differentiate both SSIM implementations and RGB-D objectives through that
same graph. The map and optimizer are not updated.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_phase13_frozen_benchmark import build_pipeline_config, load_phase10_sequence
from research.losses import color_loss, depth_loss, ssim_loss
from research.pipeline import OnlineReconstructionPipeline
from research.pipeline_checkpoint import load_pipeline_checkpoint
from research.rasterizer import render
from research.reproducibility import get_git_commit


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def summarize_gradient_pairs(named_params, direct, separable):
    rows, all_direct, all_separable = [], [], []
    for (name, _), gd, gs in zip(named_params, direct, separable):
        if gd is None and gs is None:
            rows.append({"name": name, "unused": True})
            continue
        if gd is None or gs is None:
            rows.append({"name": name, "unused_mismatch": True})
            continue
        gd, gs = gd.detach().float().reshape(-1), gs.detach().float().reshape(-1)
        delta = gs - gd
        nd, ns = torch.linalg.vector_norm(gd), torch.linalg.vector_norm(gs)
        rows.append({
            "name": name, "numel": int(gd.numel()),
            "finite": bool(torch.isfinite(gd).all() and torch.isfinite(gs).all()),
            "direct_l2": float(nd.item()), "separable_l2": float(ns.item()),
            "relative_l2_delta": float((torch.linalg.vector_norm(delta) / (nd + 1e-30)).item()),
            "cosine_similarity": float((torch.dot(gd, gs) / (nd * ns + 1e-30)).clamp(-1.0, 1.0).item()),
            "max_abs_delta": float(delta.abs().max().item()) if delta.numel() else 0.0,
        })
        all_direct.append(gd)
        all_separable.append(gs)
    if not all_direct:
        return {"global": {"numel": 0, "finite": True}, "parameters": rows}
    gd, gs = torch.cat(all_direct), torch.cat(all_separable)
    delta = gs - gd
    nd, ns = torch.linalg.vector_norm(gd), torch.linalg.vector_norm(gs)
    return {"global": {
        "numel": int(gd.numel()),
        "finite": bool(torch.isfinite(gd).all() and torch.isfinite(gs).all()),
        "direct_l2": float(nd.item()), "separable_l2": float(ns.item()),
        "relative_l2_delta": float((torch.linalg.vector_norm(delta) / (nd + 1e-30)).item()),
        "cosine_similarity": float((torch.dot(gd, gs) / (nd * ns + 1e-30)).clamp(-1.0, 1.0).item()),
        "max_abs_delta": float(delta.abs().max().item()),
    }, "parameters": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=("replica_room1", "replica_office1"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--snapshot_frame", type=int, required=True)
    ap.add_argument("--eval_frame", type=int, required=True)
    ap.add_argument("--checkpoint_dir", type=Path, required=True,
                    help="Experiment root containing snapshots/<scene>/seed*_frame*.checkpoint.pt")
    ap.add_argument("--micro_updates_per_frame", type=int, default=5000)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("This probe requires the production CUDA/gsplat path")

    frames, intrinsics = load_phase10_sequence(
        scene_name=args.scene, n_frames=args.eval_frame + 1, H=240, W=320,
        device=args.device)
    if args.eval_frame >= len(frames):
        raise ValueError(f"eval frame {args.eval_frame} unavailable; sequence has {len(frames)} frames")

    config = build_pipeline_config("ours", args.seed, args.budget_ms,
                                   320, 240, args.device)
    config["system"]["synchronized_stage_timing"] = True
    config["training"]["optimization_control"] = {
        "micro_updates_per_frame": args.micro_updates_per_frame,
    }
    pipeline = OnlineReconstructionPipeline(config=config, device=args.device)
    checkpoint = (args.checkpoint_dir / "snapshots" / args.scene /
                  f"seed{args.seed}_frame{args.snapshot_frame}.checkpoint.pt")
    metadata = load_pipeline_checkpoint(
        checkpoint, pipeline, scene=args.scene, seed=args.seed,
        snapshot_frame=args.snapshot_frame, restore_rng=True)

    frame = frames[args.eval_frame]
    rgb, depth = frame["rgb"].to(args.device), frame["depth"].to(args.device)
    pose = frame.get("pose", pipeline.current_pose).to(args.device)
    model = pipeline.gaussian_model
    pipeline.current_pose = pose
    cov3d = model.build_covariance()
    if model.sh_degree > 0:
        from research.rasterizer import _view_directions
        directions = _view_directions(model.positions, pose)
    else:
        directions = None
    rendered = render(
        means3D=model.positions, cov3D=cov3d,
        colors=model.get_colors(directions),
        opacities=model.opacities.squeeze(-1),
        extrinsics=pose, intrinsics=intrinsics,
        image_width=320, image_height=240,
        tile_size=config["rendering"]["tile_size"], backend="gsplat")
    pred_color, pred_depth = rendered["color"], rendered["depth"]
    valid_depth = (pred_depth > 0) & (depth > 0)
    weights = config["losses"]
    l_color, l_depth = color_loss(pred_color, rgb), depth_loss(pred_depth, depth, valid_depth)
    ssim_direct = ssim_loss(pred_color, rgb, implementation="direct")
    ssim_separable = ssim_loss(pred_color, rgb, implementation="separable")
    total_direct = weights["weight_color"] * l_color + weights["weight_depth"] * l_depth + weights["weight_ssim"] * ssim_direct
    total_separable = weights["weight_color"] * l_color + weights["weight_depth"] * l_depth + weights["weight_ssim"] * ssim_separable

    named_params = [(name, getattr(model, name)) for name in
                    ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest")
                    if getattr(model, name).requires_grad]
    params = [p for _, p in named_params]
    # Same gsplat forward graph for both implementations; only the loss filter changes.
    g_ssim_direct = torch.autograd.grad(ssim_direct, params, retain_graph=True, allow_unused=True)
    g_ssim_separable = torch.autograd.grad(ssim_separable, params, retain_graph=True, allow_unused=True)
    g_total_direct = torch.autograd.grad(total_direct, params, retain_graph=True, allow_unused=True)
    g_total_separable = torch.autograd.grad(total_separable, params, retain_graph=False, allow_unused=True)

    result = {
        "schema": "ssim-real-render-gradient-probe-v1",
        "scene": args.scene, "seed": args.seed,
        "snapshot_frame": args.snapshot_frame, "eval_frame": args.eval_frame,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "snapshot_map_sha256": metadata["map_digest"],
        "n_gaussians": int(model.num_gaussians), "renderer_backend": "gsplat",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "git_commit": get_git_commit(),
        "render_color_sha256": tensor_digest(pred_color),
        "render_depth_sha256": tensor_digest(pred_depth),
        "rgb_loss": float(l_color.detach().item()),
        "depth_loss": float(l_depth.detach().item()),
        "ssim_loss_direct": float(ssim_direct.detach().item()),
        "ssim_loss_separable": float(ssim_separable.detach().item()),
        "total_loss_direct": float(total_direct.detach().item()),
        "total_loss_separable": float(total_separable.detach().item()),
        "ssim_gradient": summarize_gradient_pairs(named_params, g_ssim_direct, g_ssim_separable),
        "training_objective_gradient": summarize_gradient_pairs(named_params, g_total_direct, g_total_separable),
        "scope": "Full-map gsplat render; no background-cache/subset scheduler; no optimizer step. Both losses differentiate through the same render graph.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "out": str(args.out), "snapshot_map_sha256": result["snapshot_map_sha256"],
        "n_gaussians": result["n_gaussians"],
        "ssim_loss_delta": result["ssim_loss_separable"] - result["ssim_loss_direct"],
        "ssim_gradient": result["ssim_gradient"]["global"],
        "training_objective_gradient": result["training_objective_gradient"]["global"],
    }, indent=2), flush=True)
    pipeline.cleanup()


if __name__ == "__main__":
    main()
