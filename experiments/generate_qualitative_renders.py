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
            m = pipe.eval_frame(rgb=f["rgb"], depth=f["depth"], pose=f.get("pose"))
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
