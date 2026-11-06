#!/usr/bin/env python3
"""Qualitative renders: GT vs OURS vs error_only vs FULL at held viewpoints.

Runs short trajectories (default 50 frames, fr2_xyz), then renders selected
viewpoints with each policy's FINAL map via pipeline.eval_frame (no state
change) and saves PNGs + per-view PSNR to results/qualitative/.

Usage:
  python experiments/generate_qualitative_renders.py --n_frames 50 --views 10 25 49
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_phase13_frozen_benchmark import load_phase10_sequence, build_pipeline_config
from research.pipeline import OnlineReconstructionPipeline
from research.reproducibility import set_seed

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def save_png(tensor_hw3, path):
    arr = (tensor_hw3.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    if HAVE_CV2:
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        from PIL import Image
        Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="tum_fr2_xyz")
    ap.add_argument("--n_frames", type=int, default=50)
    ap.add_argument("--views", type=int, nargs="+", default=[10, 25, 49])
    ap.add_argument("--policies", nargs="+", default=["ours", "error_only", "full"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="results/qualitative")
    ap.add_argument("--depth_panel", action="store_true",
                    help="Save GT RGB, rendered RGB, GT depth and depth-error panel for OURS.")
    a = ap.parse_args()
    out = REPO_ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240
    frames, intr = load_phase10_sequence(scene_name=a.scene, n_frames=a.n_frames + 1,
                                         H=H, W=W, device=a.device)
    summary = {"scene": a.scene, "views": []}
    for pol in a.policies:
        set_seed(a.seed)
        cfg = build_pipeline_config(policy=pol, seed=a.seed, budget_ms=a.budget_ms,
                                    W=W, H=H, device=a.device)
        pipe = OnlineReconstructionPipeline(config=cfg, device=a.device)
        pipe.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                        intrinsics=intr, pose=frames[0].get("pose", torch.eye(4)))
        for t in range(1, len(frames)):
            f = frames[t]
            pipe.process_frame(rgb=f["rgb"], depth=f["depth"], gt_pose=f.get("pose"))
        for v in a.views:
            f = frames[v]
            want_images = a.depth_panel and pol == "ours"
            m = pipe.eval_frame(rgb=f["rgb"], depth=f["depth"], pose=f.get("pose"),
                                return_images=want_images)
            # Re-render for pixels (eval_frame doesn't return image; render again cheaply)
            with torch.no_grad():
                from research.rasterizer import render as rasterize_scene
                cov = pipe.gaussian_model.build_covariance()
                res = rasterize_scene(
                    means3D=pipe.gaussian_model.positions, cov3D=cov,
                    colors=pipe.gaussian_model.get_colors(),
                    opacities=pipe.gaussian_model.opacities.squeeze(-1),
                    extrinsics=f.get("pose").to(pipe.device), intrinsics=pipe.intrinsics,
                    image_width=W, image_height=H,
                    tile_size=pipe.config["rendering"]["tile_size"],
                    backend=pipe.config.get("rendering", {}).get("backend", "gsplat"))
            p = out / f"{pol}_view{v:02d}.png"
            save_png(res["color"], p)
            if want_images:
                image_data = m["images"]
                gt_rgb = image_data["rgb_ground_truth"].numpy()
                pred_rgb = image_data["rgb_prediction"].numpy()
                gt_depth = image_data["depth_ground_truth"].numpy()
                depth_error = image_data["depth_absolute_error_gt_valid"].numpy()
                valid_gt = np.isfinite(gt_depth) & (gt_depth > 0)
                err_valid = np.isfinite(depth_error) & valid_gt
                depth_vmin = float(np.nanpercentile(gt_depth[valid_gt], 2)) if valid_gt.any() else 0.0
                depth_vmax = float(np.nanpercentile(gt_depth[valid_gt], 98)) if valid_gt.any() else 1.0
                err_vmax = float(np.nanpercentile(depth_error[err_valid], 98)) if err_valid.any() else 1.0
                fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.2))
                axes[0, 0].imshow(np.clip(gt_rgb, 0, 1))
                axes[0, 0].set_title("GT RGB")
                axes[0, 1].imshow(np.clip(pred_rgb, 0, 1))
                axes[0, 1].set_title("Rendered RGB")
                depth_im = axes[1, 0].imshow(np.ma.masked_where(~valid_gt, gt_depth),
                                             cmap="viridis", vmin=depth_vmin, vmax=depth_vmax)
                axes[1, 0].set_title("GT depth (m)")
                fig.colorbar(depth_im, ax=axes[1, 0], fraction=.046, pad=.03)
                err_im = axes[1, 1].imshow(np.ma.masked_where(~err_valid, depth_error),
                                           cmap="magma", vmin=0, vmax=max(err_vmax, 1e-6))
                axes[1, 1].set_title("Absolute depth error (m)")
                fig.colorbar(err_im, ax=axes[1, 1], fraction=.046, pad=.03)
                for ax in axes.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.suptitle(f"TUM {a.scene}, frame {v}; OURS seed {a.seed}", fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, .95))
                fig.savefig(out / f"rgbd_ours_view{v:02d}.pdf", bbox_inches="tight")
                plt.close(fig)
            if pol == a.policies[0]:
                save_png(f["rgb"], out / f"gt_view{v:02d}.png")
                summary["views"].append({"view": v})
            for rec in summary["views"]:
                if rec["view"] == v:
                    rec[pol] = round(m["psnr"], 2)
            print(f"{pol} view={v}: PSNR={m['psnr']:.2f} N={m['n_gaussians']}", flush=True)
        pipe.cleanup()
    json.dump(summary, open(out / "qual_summary.json", "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
