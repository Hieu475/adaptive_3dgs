"""Map-based (frame-to-model) pose tracker: render-and-align against the current 3DGS map.

Per frame: initialise from the constant-velocity prediction, then optimise a 6-DoF left-perturbation
of the W2C pose by Adam on a photometric + depth L1 loss between the map render and the observed
RGB-D frame (SplaTAM-style tracking, no loop closure / global BA). Map quality therefore feeds back
into pose quality. Poses are World-to-Camera like the rest of the pipeline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .rasterizer import render as rasterize_scene


def _se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """SE(3) exp map with a gradient that is finite and non-zero at xi = 0 (small-angle series)."""
    w, v = xi[:3], xi[3:]
    th2 = (w * w).sum() + 1e-12
    th = th2.sqrt()
    a = torch.sin(th) / th
    b = (1 - torch.cos(th)) / th2
    c = (1 - a) / th2
    z = torch.zeros((), device=xi.device, dtype=xi.dtype)
    K = torch.stack([torch.stack([z, -w[2], w[1]]), torch.stack([w[2], z, -w[0]]), torch.stack([-w[1], w[0], z])])
    I = torch.eye(3, device=xi.device, dtype=xi.dtype)
    R = I + a * K + b * (K @ K)
    V = I + b * K + c * (K @ K)
    top = torch.cat([R, (V @ v)[:, None]], dim=1)
    return torch.cat([top, torch.tensor([[0., 0., 0., 1.]], device=xi.device, dtype=xi.dtype)], dim=0)


class MapTracker:
    """Coarse-to-fine render-and-align tracker.

    ``levels``: downsampling factors (e.g. (4, 2, 1)); ``iters``: Adam steps per level; ``lr``: step size
    for the translation/rotation parameters at the finest level (coarser levels use proportionally larger steps).
    """

    def __init__(self, levels=(4, 2, 1), iters=(20, 15, 10), lr: float = 1e-3, depth_w: float = 1.0):
        self.levels, self.iters, self.lr, self.depth_w = tuple(levels), tuple(iters), lr, depth_w
        self.prev = None   # last W2C pose
        self.vel = None    # relative motion T_t @ T_{t-1}^-1

    def reset(self, pose: torch.Tensor):
        self.prev = pose.detach().float().clone()
        self.vel = torch.eye(4, device=pose.device)

    def track(self, pipe, rgb, depth, init=None):
        """``init``: optional W2C initial guess; else constant velocity."""
        dev = rgb.device
        T0 = init.detach().float().to(dev) if init is not None else self.vel @ self.prev
        gm = pipe.gaussian_model
        with torch.no_grad():
            cov = gm.build_covariance()
            dirs = None
            if gm.sh_degree > 0:
                from .rasterizer import _view_directions
                dirs = _view_directions(gm.positions, T0)
            col = gm.get_colors(dirs)
            pos, op = gm.positions, gm.opacities.squeeze(-1)
        backend = pipe.config.get("rendering", {}).get("backend", "gsplat")
        T_cur = T0
        for lvl, n_it in zip(self.levels, self.iters):
            H, W = depth.shape[0] // lvl, depth.shape[1] // lvl
            if lvl > 1:
                rgb_l = F.avg_pool2d(rgb.permute(2, 0, 1)[None], lvl)[0].permute(1, 2, 0)
                valid_full = (torch.isfinite(depth) & (depth > 0)).float()
                dz = torch.where(valid_full > 0, depth, torch.zeros_like(depth))
                cnt = F.avg_pool2d(valid_full[None, None], lvl)[0, 0]
                depth_l = F.avg_pool2d(dz[None, None], lvl)[0, 0] / cnt.clamp_min(1e-6)
                valid = cnt > 0.99
                K = pipe.intrinsics.clone().float()
                K[0, :] /= lvl; K[1, :] /= lvl
            else:
                rgb_l, depth_l, K = rgb, depth, pipe.intrinsics
                valid = torch.isfinite(depth) & (depth > 0)
            xi = torch.zeros(6, device=dev, requires_grad=True)
            opt = torch.optim.Adam([xi], lr=self.lr * (lvl ** 0.5))
            best, best_loss = T_cur, float("inf")
            base = T_cur
            for _ in range(n_it):
                opt.zero_grad()
                T = _se3_exp(xi) @ base
                r = rasterize_scene(means3D=pos, cov3D=cov, colors=col, opacities=op, extrinsics=T,
                                    intrinsics=K, image_width=W, image_height=H,
                                    tile_size=pipe.config["rendering"]["tile_size"], backend=backend)
                sup = valid & (r["alpha"] > 0.5)
                if sup.sum() < 50:
                    break
                loss = (r["color"] - rgb_l).abs().mean(-1)[sup].mean() + self.depth_w * (r["depth"] - depth_l).abs()[sup].mean()
                if float(loss) < best_loss:
                    best_loss, best = float(loss), T.detach().clone()
                loss.backward()
                opt.step()
            T_cur = best
        self.vel = T_cur @ torch.linalg.inv(self.prev)
        self.prev = T_cur.clone()
        return T_cur
