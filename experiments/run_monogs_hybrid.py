#!/usr/bin/env python3
"""Hybrid system: MonoGS tracker for poses, our mapper for the map.

Needs a MonoGS checkout (https://github.com/muskie82/MonoGS) whose submodules are built in the active
environment, and the TUM data linked at <MonoGS>/datasets/tum. Per frame:
  1. MonoGS tracking: starting from the previous pose, optimize a 6-DoF correction (100 Adam iterations,
     MonoGS photometric+depth loss, MonoGS rasterizer) against the render of OUR current map. Our Gaussians
     are handed over through a thin adapter that exposes the attributes MonoGS' renderer reads.
  2. Our mapper (src.core.pipeline) maps the frame at the estimated pose (original budgeted system, K uniform
     steps, or time-targeted steps).
Frames with index % 5 == 0 (index > 0), the grid on which MonoGS evaluates, are tracked but never mapped; their PSNR
(final map, rendered from the estimated pose, MonoGS' own psnr function) is the held-out quality, as in MonoGS' eval. ATE is the Umeyama-aligned RMSE of the estimated against ground-truth camera centers.

usage (MonoGS env): python experiments/run_monogs_hybrid.py --monogs <path> --config <yaml> --cond k3 --n_frames 150
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monogs", type=Path, required=True)
    ap.add_argument("--config", required=True, help="MonoGS yaml, relative to the MonoGS checkout")
    ap.add_argument("--conds", nargs="+", default=["orig", "k3"], choices=["orig", "k3", "k6", "k10", "tts60", "k3_rp", "k10_rp", "k3_jr", "k3_jn", "k3_jn8", "k3_jr3b"])
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--holdout_every", type=int, default=5, help="frames with index %% holdout_every == 0 (index > 0) are tracked but never mapped")
    ap.add_argument("--track_iters", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "monogs_hybrid")
    return ap.parse_args()


def main():
    a = parse()
    os.chdir(a.monogs)
    sys.path.insert(0, str(a.monogs))
    sys.path.insert(1, str(ROOT))
    from munch import munchify

    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
    from utils.camera_utils import Camera
    from utils.config_utils import load_config
    from utils.dataset import load_dataset
    from utils.eval_utils import evaluate_evo
    from gaussian_splatting.utils.image_utils import psnr as monogs_psnr
    from utils.pose_utils import update_pose
    from utils.slam_utils import get_loss_tracking

    import experiments.run_vgg as vgg  # noqa: E402  (installs the time-target wrapper)
    from src.core.pipeline import OnlineReconstructionPipeline  # noqa: E402
    from src.evaluation.reproducibility import set_seed  # noqa: E402

    C0 = 0.28209479177387814
    cfg = load_config(a.config)
    cfg["Training"]["monocular"] = cfg["Dataset"]["sensor_type"] == "monocular"
    model_params, pipeline_params = munchify(cfg["model_params"]), munchify(cfg["pipeline_params"])
    model_params.sh_degree = 0
    dataset = load_dataset(model_params, model_params.source_path, config=cfg)
    n = min(a.n_frames, len(dataset))
    track_iters = a.track_iters or cfg["Training"]["tracking_itr_num"]
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    proj = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=dataset.fx, fy=dataset.fy, cx=dataset.cx, cy=dataset.cy,
                                W=dataset.width, H=dataset.height).transpose(0, 1).to("cuda")
    K = torch.tensor([[dataset.fx, 0, dataset.cx], [0, dataset.fy, dataset.cy], [0, 0, 1]], dtype=torch.float32, device="cuda")

    class MapView:
        """The attributes MonoGS' renderer reads, built from our Gaussian model (SH degree 0)."""
        active_sh_degree = 0
        max_sh_degree = 0

        def __init__(self, gm):
            with torch.no_grad():
                rot = gm._rotation
                self.get_xyz = gm.positions.detach()
                self.get_opacity = gm.opacities.detach().reshape(-1, 1)
                self.get_scaling = gm.scales.detach()
                self.get_rotation = (rot / (rot.norm(dim=-1, keepdim=True) + 1e-8)).detach()
                self.get_features = ((gm.get_colors(None) - 0.5) / C0).detach().unsqueeze(1)

    def w2c(cam):
        m = torch.eye(4, device="cuda")
        m[:3, :3], m[:3, 3] = cam.R, cam.T
        return m

    def track(pipe, vp, prev):
        """MonoGS tracking iterations against our current map."""
        vp.update_RT(prev.R, prev.T)
        opt = torch.optim.Adam([
            {"params": [vp.cam_rot_delta], "lr": cfg["Training"]["lr"]["cam_rot_delta"]},
            {"params": [vp.cam_trans_delta], "lr": cfg["Training"]["lr"]["cam_trans_delta"]},
            {"params": [vp.exposure_a], "lr": 0.01}, {"params": [vp.exposure_b], "lr": 0.01}])
        view = MapView(pipe.gaussian_model)
        for _ in range(track_iters):
            pkg = render(vp, view, pipeline_params, background)
            opt.zero_grad()
            loss = get_loss_tracking(cfg, pkg["render"], pkg["depth"], pkg["opacity"], vp)
            loss.backward()
            with torch.no_grad():
                opt.step()
                if update_pose(vp):
                    break

    spec = {"orig": ("error_off", {}), "k3": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}),
            "k6": ("tts", {"T_ms": 1.0, "n_min": 6, "n_max": 6, "n_init": 6}),
            "k10": ("tts", {"T_ms": 1.0, "n_min": 10, "n_max": 10, "n_init": 10}), "tts60": ("tts", {"T_ms": 60.0}),
            "k3_rp": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}), "k10_rp": ("tts", {"T_ms": 1.0, "n_min": 10, "n_max": 10, "n_init": 10}),
            "k3_jr": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}),
            "k3_jn": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}), "k3_jn8": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3}),
            "k3_jr3b": ("tts", {"T_ms": 1.0, "n_min": 3, "n_max": 3, "n_init": 3})}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    scene = Path(cfg["Dataset"]["dataset_path"]).name.replace("rgbd_dataset_freiburg", "fr")
    for name in a.conds:
        target = a.out_dir / f"{scene}_{name}_seed{a.seed}.json"
        if target.exists():
            continue
        set_seed(a.seed)
        cond, tt = spec[name]
        _, pcfg = vgg.config_for(cond, a.seed, 15.0, "cuda", {}, tt)
        if name == "k3_jr":  # joint replay: steps 2..K optimize the current frame with 2 replayed frames
            pcfg["training"]["replay"] = {"every": 1, "size": 32, "stride": 2, "joint": 2}
        if name == "k3_jr3b":  # joint replay with three replayed views, rendered with the current view in one call
            pcfg["training"]["replay"] = {"every": 1, "size": 32, "stride": 2, "joint": 3, "batched": True}
        if name in ("k3_jn", "k3_jn8"):  # joint replay over a pose-proximity keyframe window
            pcfg["training"]["replay"] = {"every": 1, "size": 32, "stride": 2, "joint": 2, "near": 4 if name == "k3_jn" else 8}
        if name.endswith("_rp"):  # keyframe replay, constants chosen on the development scenes
            pcfg["training"]["replay"] = {"every": 3, "size": 32, "stride": 2}
        pipe = OnlineReconstructionPipeline(config=pcfg, device="cuda")
        vgg.STATES.clear(); vgg.TT_STATES.clear()
        cams, est, gt, held, mapped_psnr, held_psnr, t_trk, t_map, held_idx = [], [], [], [], [], [], [], [], []
        for i in range(n):
            vp = Camera.init_from_dataset(dataset, i, proj)
            vp.compute_grad_mask(cfg)
            rgb = vp.original_image.permute(1, 2, 0).contiguous().cuda()
            depth = torch.from_numpy(vp.depth).float().cuda()
            if i == 0:
                vp.update_RT(vp.R_gt, vp.T_gt)
                pipe.initialize(rgb=rgb, depth=depth, intrinsics=K, pose=w2c(vp))
            else:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                track(pipe, vp, cams[-1])
                torch.cuda.synchronize(); t1 = time.perf_counter()
                pose = w2c(vp).detach()
                if i % a.holdout_every == 0:
                    held_idx.append((i, pose.clone()))
                    held_psnr.append(float(pipe.eval_frame(rgb=rgb, depth=depth, pose=pose)["psnr"]))
                else:
                    m = pipe.process_frame(rgb=rgb, depth=depth, gt_pose=pose)
                    mapped_psnr.append(float(m["psnr"]))
                torch.cuda.synchronize(); t2 = time.perf_counter()
                t_trk.append(t1 - t0); t_map.append(t2 - t1)
            cams.append(vp)
            est.append(np.linalg.inv(w2c(vp).cpu().numpy()))
            gt_pose = np.eye(4); gt_pose[:3, :3] = vp.R_gt.cpu().numpy(); gt_pose[:3, 3] = vp.T_gt.cpu().numpy()
            gt.append(np.linalg.inv(gt_pose))
            vp.clean()
        final_psnr = []
        for i, pose in held_idx:  # final map on the held-out frames, MonoGS' psnr with its pixel mask
            gt_img, _, _ = dataset[i]
            gt_img = gt_img.cuda()
            gt_depth = torch.from_numpy(dataset[i][1]).float().cuda()
            out = pipe.eval_frame(rgb=gt_img.permute(1, 2, 0).contiguous(), depth=gt_depth, pose=pose, return_images=True)
            img = torch.clamp(out["images"]["rgb_prediction"].permute(2, 0, 1).cuda(), 0.0, 1.0)
            mask = gt_img > 0
            final_psnr.append(float(monogs_psnr(img[mask].unsqueeze(0), gt_img[mask].unsqueeze(0))))
        plot_dir = a.out_dir / "plots"; plot_dir.mkdir(exist_ok=True)
        ate_all = evaluate_evo(gt, est, str(plot_dir), f"{scene}_{name}_all")
        ate_5 = evaluate_evo(gt[::5], est[::5], str(plot_dir), f"{scene}_{name}_every5")
        res = {"scene": scene, "cond": name, "seed": a.seed, "n_frames": n, "track_iters": track_iters,
               "ate_all_m": float(ate_all), "ate_every5_m": float(ate_5),
               "mapped_psnr": float(np.mean(mapped_psnr)), "heldout_psnr": float(np.mean(held_psnr)),
               "heldout_final_psnr": float(np.mean(final_psnr)),
               "track_ms": 1000 * float(np.mean(t_trk)), "map_ms": 1000 * float(np.mean(t_map)),
               "N_final": int(pipe.gaussian_model.num_gaussians), "n_heldout": len(held_psnr)}
        target.write_text(json.dumps(res, indent=1))
        print("DONE", json.dumps(res), flush=True)
        del pipe; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
