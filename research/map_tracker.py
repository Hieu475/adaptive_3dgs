"""Map-based (frame-to-model) pose tracker: render-and-align against the current 3DGS map.

Per frame: initialise from the constant-velocity prediction, then optimise a 6-DoF left-perturbation
of the W2C pose by Adam on a photometric + depth L1 loss between the map render and the observed
RGB-D frame (SplaTAM-style tracking, no loop closure / global BA). Map quality therefore feeds back
into pose quality. Poses are World-to-Camera like the rest of the pipeline.
"""
from __future__ import annotations

import torch

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
    def __init__(self, iters: int = 20, lr_rot: float = 2e-3, lr_trans: float = 2e-3, depth_w: float = 1.0):
        self.iters, self.lr_rot, self.lr_trans, self.depth_w = iters, lr_rot, lr_trans, depth_w
        self.prev = None   # last W2C pose
        self.vel = None    # relative motion T_t @ T_{t-1}^-1

    def reset(self, pose: torch.Tensor):
        self.prev = pose.detach().float().clone()
        self.vel = torch.eye(4, device=pose.device)

    def track(self, pipe, rgb, depth, init=None):
        """``init``: optional W2C initial guess (e.g. from the frame-to-frame tracker); else constant velocity."""
        dev = rgb.device
        H, W = depth.shape
        T0 = init.detach().float().to(dev) if init is not None else self.vel @ self.prev
        xi = torch.zeros(6, device=dev, requires_grad=True)
        opt = torch.optim.Adam([{"params": [xi]}], lr=self.lr_trans)
        valid = torch.isfinite(depth) & (depth > 0)
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
        best, best_loss = T0, float("inf")
        scale = torch.tensor([self.lr_rot / self.lr_trans] * 3 + [1.0] * 3, device=dev)
        for _ in range(self.iters):
            opt.zero_grad()
            T = _se3_exp(xi * scale) @ T0
            r = rasterize_scene(means3D=pos, cov3D=cov, colors=col, opacities=op, extrinsics=T,
                                intrinsics=pipe.intrinsics, image_width=W, image_height=H,
                                tile_size=pipe.config["rendering"]["tile_size"], backend=backend)
            sup = valid & (r["alpha"] > 0.5)
            if sup.sum() < 100:
                break
            l_c = (r["color"] - rgb).abs().mean(-1)[sup].mean()
            l_d = (r["depth"] - depth).abs()[sup].mean()
            loss = l_c + self.depth_w * l_d
            lv = float(loss)
            if lv < best_loss:
                best_loss, best = lv, T.detach().clone()
            loss.backward()
            opt.step()
        T_new = best
        self.vel = T_new @ torch.linalg.inv(self.prev)
        self.prev = T_new.clone()
        return T_new
