#!/usr/bin/env python3
"""Run an RGB-D qualitative comparison with a selected policy seed/frame."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_frozen_benchmark import build_pipeline_config, load_benchmark_sequence
from src.core.pipeline import OnlineReconstructionPipeline
from src.evaluation.reproducibility import set_seed


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_comparison(panel_data, out_path, frame, seed, device, renderer, metrics):
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.5), constrained_layout=True)
    panels = [
        ("GT RGB", panel_data["gt_rgb"], "rgb"),
        ("Error-only", panel_data["error_only_rgb"], "rgb"),
        ("OURS", panel_data["ours_rgb"], "rgb"),
        ("FULL", panel_data["full_rgb"], "rgb"),
        ("GT Depth (m)", panel_data["gt_depth"], "depth"),
        ("OURS Depth (m)", panel_data["ours_depth"], "depth"),
        ("Depth Error (m)", panel_data["depth_error"], "error"),
        ("Coverage Mask", panel_data["coverage_mask"], "mask"),
    ]
    valid = np.isfinite(panel_data["gt_depth"]) & (panel_data["gt_depth"] > 0)
    dmin, dmax = (np.percentile(panel_data["gt_depth"][valid], [2, 98])
                  if valid.any() else (0.0, 1.0))
    err = panel_data["depth_error"]
    err_valid = np.isfinite(err)
    emax = float(np.percentile(err[err_valid], 98)) if err_valid.any() else 1.0
    depth_artist = None
    error_artist = None
    mask_artist = None
    policy_for_title = {"Error-only": "error_only", "OURS": "ours", "FULL": "full"}
    for ax, (title, image, kind) in zip(axes.flat, panels):
        if kind == "rgb":
            ax.imshow(np.clip(image, 0, 1))
            if title in policy_for_title:
                score = metrics[policy_for_title[title]]["psnr_db"]
                ax.set_xlabel(f"PSNR {score:.2f} dB", fontsize=12, labelpad=3)
            else:
                ax.set_xlabel("Observed input", fontsize=12, labelpad=3)
        elif kind == "depth":
            depth_artist = ax.imshow(
                np.ma.masked_where(~valid, image), cmap="viridis",
                vmin=dmin, vmax=max(dmax, dmin + 1e-6))
        elif kind == "error":
            error_artist = ax.imshow(
                np.ma.masked_invalid(image), cmap="magma", vmin=0,
                vmax=max(emax, 1e-6))
        else:
            # 1=covered valid GT pixel, 0=valid but uncovered, 0.5=invalid GT depth.
            mask = np.where(np.isfinite(image), image, 0.5)
            mask_cmap = ListedColormap(["#20252B", "#9AA0A6", "#F5F7FA"])
            mask_artist = ax.imshow(mask, cmap=mask_cmap, vmin=0, vmax=1,
                                    interpolation="nearest")
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#D9D9D9")
            spine.set_linewidth(0.7)
    accelerator = "CUDA" if device.startswith("cuda") else "CPU"
    fig.suptitle(f"TUM fr2_xyz, frame {frame} (observed view) | seed {seed} | {accelerator} / {renderer}", fontsize=15)
    depth_bar = fig.colorbar(depth_artist, ax=[axes[1, 0], axes[1, 1]],
                             orientation="horizontal", shrink=0.78, pad=0.16)
    depth_bar.set_label("Depth (m)", fontsize=12)
    depth_bar.ax.tick_params(labelsize=10)
    error_bar = fig.colorbar(error_artist, ax=axes[1, 2], orientation="horizontal",
                             shrink=0.78, pad=0.16)
    error_bar.set_label("Absolute error (m)", fontsize=12)
    error_bar.ax.tick_params(labelsize=10)
    mask_bar = fig.colorbar(mask_artist, ax=axes[1, 3],
                            boundaries=[-0.25, 0.25, 0.75, 1.25],
                            ticks=[0, 0.5, 1], spacing="uniform", shrink=0.95,
                            orientation="horizontal", pad=0.16)
    mask_bar.set_ticklabels(["Missing", "Invalid GT", "Covered"])
    mask_bar.ax.tick_params(labelsize=9)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=150)
    parser.add_argument("--view", type=int, default=149)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget_ms", type=float, default=15.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results" / "qualitative_rgbd_gpu_frame149")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--render_saved_data", action="store_true",
                        help="Rebuild the panel from the saved NPZ without rerunning policies")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.render_saved_data:
        data_path = out_dir / f"rgbd_comparison_frame{args.view}_data.npz"
        metadata_path = out_dir / "run_metadata.json"
        stored = np.load(data_path)
        panel_data = {k: stored[k] for k in stored.files}
        old_metadata = json.loads(metadata_path.read_text())
        old_metadata.update({
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "python_executable": sys.executable,
            "conda_env": __import__("os").environ.get("CONDA_DEFAULT_ENV", "ai"),
            "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "gsplat_version": "1.5.3",
            "effective_configs": {
                policy: build_pipeline_config(policy, args.seed, args.budget_ms,
                                               W=320, H=240, device=old_metadata["device"])
                for policy in ("error_only", "ours", "full")
            },
        })
        pdf_path = out_dir / f"rgbd_comparison_frame{args.view}.pdf"
        save_comparison(panel_data, pdf_path, args.view, args.seed,
                        old_metadata["device"], old_metadata["renderer_backend"],
                        old_metadata["metrics"])
        report_figure = ROOT / "bao_cao" / "figures" / "qualitative_rgbd_frame149.pdf"
        report_figure.write_bytes(pdf_path.read_bytes())
        metadata_path.write_text(json.dumps(old_metadata, indent=2) + "\n")
        print(f"Rebuilt {pdf_path} from saved panel arrays")
        return
    if args.view >= args.n_frames:
        raise ValueError("--view must be smaller than --n_frames")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    device = args.device
    W, H = 320, 240
    frames, intrinsics = load_benchmark_sequence("tum_fr2_xyz", args.n_frames,
                                                H=H, W=W, device=device)
    frame = frames[args.view]
    outputs = {}
    metrics = {}
    for policy, key in (("error_only", "error_only"), ("ours", "ours"), ("full", "full")):
        print(f"Starting {policy}: {args.n_frames} frames, seed {args.seed}, {device}", flush=True)
        set_seed(args.seed)
        cfg = build_pipeline_config(policy=policy, seed=args.seed, budget_ms=args.budget_ms,
                                    W=W, H=H, device=device)
        pipe = OnlineReconstructionPipeline(config=cfg, device=device)
        pipe.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                        intrinsics=intrinsics, pose=frames[0].get("pose", torch.eye(4)))
        for i in range(1, args.view + 1):
            f = frames[i]
            m = pipe.process_frame(rgb=f["rgb"], depth=f["depth"], gt_pose=f.get("pose"))
            if i % 10 == 0 or i == args.view:
                print(f"  {policy} frame {i}/{args.view}: PSNR={float(m['psnr']):.2f} dB, N={int(m['n_gaussians'])}", flush=True)
        m = pipe.eval_frame(rgb=frame["rgb"], depth=frame["depth"], pose=frame["pose"],
                            return_images=True)
        im = m["images"]
        outputs[key] = {
            "rgb": as_numpy(im["rgb_prediction"]),
            "psnr_db": float(m["psnr"]),
            "gaussians": int(m["n_gaussians"]),
        }
        metrics[key] = {"psnr_db": float(m["psnr"]),
                        "ssim_standard": float(m["ssim_standard"]),
                        "n_gaussians": int(m["n_gaussians"])}
        if key == "ours":
            gt_depth = as_numpy(im["depth_ground_truth"]).astype(np.float32)
            pred_depth = as_numpy(im["depth_prediction"]).astype(np.float32)
            valid_gt = as_numpy(im["gt_valid_mask"]).astype(bool)
            supported = as_numpy(im["pred_depth_coverage_mask"]).astype(bool)
            depth_error = as_numpy(im["depth_absolute_error_gt_valid"]).astype(np.float32)
            coverage = np.where(valid_gt, supported.astype(np.float32), np.nan)
            outputs["rgbd"] = {
                "gt_rgb": as_numpy(im["rgb_ground_truth"]),
                "gt_depth": np.where(valid_gt, gt_depth, np.nan),
                "ours_depth": np.where(supported, pred_depth, np.nan),
                "depth_error": depth_error,
                "coverage_mask": coverage,
            }
        pipe.cleanup()
    panel_data = {
        "gt_rgb": outputs["rgbd"]["gt_rgb"],
        "error_only_rgb": outputs["error_only"]["rgb"],
        "ours_rgb": outputs["ours"]["rgb"],
        "full_rgb": outputs["full"]["rgb"],
        **{k: outputs["rgbd"][k] for k in ("gt_depth", "ours_depth", "depth_error", "coverage_mask")},
    }
    np.savez_compressed(out_dir / f"rgbd_comparison_frame{args.view}_data.npz", **panel_data)
    pdf_path = out_dir / f"rgbd_comparison_frame{args.view}.pdf"
    renderer = cfg["rendering"]["backend"]
    save_comparison(panel_data, pdf_path, args.view, args.seed, device, renderer, metrics)
    report_figure = ROOT / "bao_cao" / "figures" / "qualitative_rgbd_frame149.pdf"
    report_figure.parent.mkdir(parents=True, exist_ok=True)
    report_figure.write_bytes(pdf_path.read_bytes())
    report = {
        "scene": "tum_fr2_xyz",
        "frame": args.view,
        "n_frames_processed": args.view + 1,
        "seed": args.seed,
        "resolution": [W, H],
        "device": device,
        "renderer_backend": renderer,
        "gpu_name": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None,
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "python_executable": sys.executable,
        "conda_env": __import__("os").environ.get("CONDA_DEFAULT_ENV"),
        "scheduler_budget_ms": args.budget_ms,
        "scheduler_cost_per_gaussian_us": cfg["scheduler"]["cost_per_gaussian_us"],
        "protocol_note": "Single-seed qualitative run using the headline scene, policy configs, resolution, and scheduler budget. It is not the eight-seed metric aggregate.",
        "metrics": metrics,
        "panel": pdf_path.name,
        "report_figure": str(report_figure.relative_to(ROOT)),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
