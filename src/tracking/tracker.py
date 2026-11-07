"""Dense RGB-D frame-to-frame tracker (baseline odometry, GT-free).

Replaces the former identity stub. Frame-to-model would be ideal, but a
frame-to-frame dense odometer is the honest minimal baseline: coarse-to-fine
Gauss-Newton on SE(3) over combined photometric + depth residuals with Huber
weights. No learned weights, no keyframes, no loop closure — label ATE from
this module as "baseline VO", never as a SLAM system result.

Conventions: poses are World-to-Camera (match the renderer). track_frame
returns the estimated absolute W2C pose of the current frame.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def _se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Exp map se(3) -> SE(3), xi = (omega(3), v(3))."""
    w, v = xi[:3], xi[3:]
    th = w.norm() + 1e-12
    K = torch.tensor([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]],
                     device=xi.device, dtype=xi.dtype)
    R = (torch.eye(3, device=xi.device, dtype=xi.dtype)
         + torch.sin(th) / th * K + (1 - torch.cos(th)) / (th * th) * (K @ K))
    V = (torch.eye(3, device=xi.device, dtype=xi.dtype)
         + (1 - torch.cos(th)) / (th * th) * K
         + (th - torch.sin(th)) / (th ** 3) * (K @ K))
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def _to_gray(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim == 3:
        g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        g = rgb
    return g.float()


def _downsample(img: torch.Tensor, levels: int) -> torch.Tensor:
    x = img
    for _ in range(levels):
        if x.ndim == 2:
            x = F.avg_pool2d(x[None, None], 2, stride=2)[0, 0]
        else:
            x = F.avg_pool2d(x.permute(2, 0, 1)[None], 2, stride=2)[0].permute(1, 2, 0)
    return x


class DenseRGBDTracker:
    """Frame-to-frame dense RGB-D odometry (baseline)."""

    def __init__(self, n_levels: int = 2, n_iters: Tuple[int, ...] = (8, 10),
                 huber_photo: float = 0.1, huber_depth: float = 0.05,
                 depth_weight: float = 1.0):
        self.n_levels = n_levels
        self.n_iters = n_iters
        self.huber_photo = huber_photo
        self.huber_depth = huber_depth
        self.depth_weight = depth_weight
        self._prev_gray: Optional[torch.Tensor] = None
        self._prev_depth: Optional[torch.Tensor] = None
        self._prev_pose: Optional[torch.Tensor] = None
        self._vel = torch.eye(4)

    def reset(self, rgb: torch.Tensor, depth: torch.Tensor, pose: torch.Tensor) -> None:
        self._prev_gray = _to_gray(rgb).detach()
        self._prev_depth = depth.detach().float()
        self._prev_pose = pose.detach().float()
        self._vel = torch.eye(4, dtype=self._prev_pose.dtype)

    def _align_level(self, gp: torch.Tensor, dp: torch.Tensor,
                     gc: torch.Tensor, dc: torch.Tensor,
                     K: torch.Tensor, T_init: torch.Tensor, n_iter: int,
                     max_px: int = 9000) -> torch.Tensor:
        """Gauss-Newton on SE(3) with analytic Jacobians (ESM-style sampling)."""
        dev, dt = gp.device, torch.float32
        H, W = gp.shape
        K = K.to(dev, dt)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt),
                                torch.arange(W, device=dev, dtype=dt), indexing="ij")
        # image gradients of the CURRENT frame (target of warp)
        gx = torch.zeros_like(gc); gy = torch.zeros_like(gc)
        gx[:, 1:-1] = (gc[:, 2:] - gc[:, :-2]) * 0.5
        gy[1:-1, :] = (gc[2:, :] - gc[:-2, :]) * 0.5
        T_rel = T_init.clone().to(dev, dt)
        for _ in range(n_iter):
            with torch.no_grad():
                R, t = T_rel[:3, :3], T_rel[:3, 3]
                X = torch.stack([(xs - cx) / fx * dp, (ys - cy) / fy * dp, dp], dim=-1)
                Xc = X @ R.T + t
                Zc = Xc[..., 2].clamp_min(1e-3)
                u = fx * Xc[..., 0] / Zc + cx
                v = fy * Xc[..., 1] / Zc + cy
                grid = torch.stack([2 * u / max(W - 1, 1) - 1,
                                    2 * v / max(H - 1, 1) - 1], dim=-1)[None]
                gn = lambda im, mode: F.grid_sample(
                    im[None, None], grid, mode=mode,
                    padding_mode="zeros", align_corners=True)[0, 0]
                Ic, Dc = gn(gc, "bilinear"), gn(dc, "nearest")
                Ix, Iy = gn(gx, "bilinear"), gn(gy, "bilinear")
                m = (dp > 0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1) & (Dc > 0)
                idx = torch.where(m.reshape(-1))[0]
                if idx.numel() < 300:
                    break
                if idx.numel() > max_px:
                    idx = idx[torch.randperm(idx.numel(), device=dev)[:max_px]]
                Xc_s = Xc.reshape(-1, 3)[idx]
                Z = Xc_s[:, 2].clamp_min(1e-3)
                # d[u,v]/dXc
                Ju = torch.stack([fx / Z, torch.zeros_like(Z), -fx * Xc_s[:, 0] / (Z * Z)], dim=-1)
                Jv = torch.stack([torch.zeros_like(Z), fy / Z, -fy * Xc_s[:, 1] / (Z * Z)], dim=-1)
                # dXc/dxi = [I | -skew(Xc)]
                Xx, Xy = Xc_s[:, 0], Xc_s[:, 1]
                Zc_col = Z
                # xi = (omega(0:3), v(3:6)); dXc/dxi = [-skew(Xc) | I].
                Xx, Xy, Xz = Xc_s[:, 0], Xc_s[:, 1], Z
                dX = torch.zeros((idx.numel(), 3, 6), device=dev, dtype=dt)
                dX[:, 0, 1] = Xz; dX[:, 0, 2] = -Xy; dX[:, 0, 3] = 1
                dX[:, 1, 0] = -Xz; dX[:, 1, 2] = Xx; dX[:, 1, 4] = 1
                dX[:, 2, 0] = Xy; dX[:, 2, 1] = -Xx; dX[:, 2, 5] = 1
                Ix_s, Iy_s = Ix.reshape(-1)[idx], Iy.reshape(-1)[idx]
                # photo jacobian: (Ix*Ju + Iy*Jv) @ dX  -> (M,6)
                A = Ix_s[:, None] * Ju + Iy_s[:, None] * Jv  # (M,3) dI/dXc
                Jp = (A[:, None, :] @ dX)[:, 0, :]
                # depth jacobian (drop target-gradient term): dZc/dxi = row3 of dX
                Jd = dX[:, 2, :]  # (M,6)
                rp = Ic.reshape(-1)[idx] - gp.reshape(-1)[idx]
                rd = Z - Dc.reshape(-1)[idx]
                wp = (self.huber_photo / rp.abs().clamp_min(1e-6)).clamp_max(1.0)
                wd = (self.huber_depth / rd.abs().clamp_min(1e-6)).clamp_max(1.0)
                J = torch.cat([Jp * wp[:, None],
                               Jd * (wd * self.depth_weight)[:, None]], dim=0)
                r = torch.cat([rp * wp, rd * wd * self.depth_weight])
                Hm = J.T @ J + 1e-4 * torch.eye(6, device=dev, dtype=dt) * (J.T @ J).diag().mean().clamp_min(1e-8)
                try:
                    delta = -torch.linalg.solve(Hm, J.T @ r)
                except Exception:
                    break
                if not torch.isfinite(delta).all() or delta.norm() > 0.5:
                    delta = delta.clamp(-0.05, 0.05)
                T_rel = _se3_exp(delta) @ T_rel
                if delta.norm() < 1e-5:
                    break
        return T_rel.detach()

    def track(self, rgb: torch.Tensor, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Estimate absolute W2C pose. First call after reset() inits velocity prior."""
        g = _to_gray(rgb)
        d = depth.float()
        assert self._prev_gray is not None, "call reset() with the first frame"
        T_init = self._vel.to(g.device, torch.float32)
        T_rel = T_init
        for lvl in range(self.n_levels - 1, -1, -1):
            s = 2 ** lvl
            Kl = K.clone().float()
            Kl[0, 0] /= s; Kl[1, 1] /= s; Kl[0, 2] /= s; Kl[1, 2] /= s
            T_rel = self._align_level(
                _downsample(self._prev_gray, lvl), _downsample(self._prev_depth, lvl),
                _downsample(g, lvl), _downsample(d, lvl), Kl, T_rel,
                self.n_iters[lvl] if lvl < len(self.n_iters) else self.n_iters[-1])
        T_cur = T_rel @ self._prev_pose.to(g.device, torch.float32)
        self._vel = T_rel.clone()
        self._prev_gray = g.detach()
        self._prev_depth = d.detach()
        self._prev_pose = T_cur.clone()
        return T_cur


class ICPTracker(DenseRGBDTracker):
    """Backward-compatible name. Now a real dense baseline (was identity stub)."""

    def track_frame(self, rgb: torch.Tensor, depth: torch.Tensor,
                    model=None, intrinsics=None) -> torch.Tensor:
        """Pipeline-compatible entry: first call anchors, later calls track."""
        if self._prev_gray is None:
            if intrinsics is None:
                raise ValueError("tracker needs intrinsics on first frame")
            self._K = intrinsics.detach().float()
            dev = rgb.device
            self.reset(rgb.to(dev), depth.to(dev),
                       torch.eye(4, dtype=torch.float32, device=dev))
            # anchor: unknown global frame; caller overrides with init pose
            return torch.eye(4, dtype=torch.float32, device=dev)
        K = intrinsics if intrinsics is not None else self._K
        return self.track(rgb, depth, K)
