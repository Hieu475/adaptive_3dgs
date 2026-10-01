"""RTG-SLAM (Peng et al., SIGGRAPH 2024) faithful re-implementation as budget-matched baseline.

Reference: RTG-SLAM compacts Gaussians into opaque / nearly-transparent roles,
adds Gaussians at newly-observed / high color-error / high depth-error pixels,
and optimizes ONLY unstable Gaussians (stable = low temporal error drift).
This is rule-based stable/unstable selection — the primary external baseline
our continuous utility / throttling must beat at MATCHED compute.

Interface matches research/benchmark_policies.py:
    select(state, budget_ms) -> selected_ids, info
so it can be dropped into run_final_confirmation.py / run_phase13_frozen_benchmark.py
without changing harness. No learned weights. No oracle access.

Differences from original CUDA SLAM system (disclosed):
- tracking is ground-truth TUM pose (same as internal policies here);
- rendering is gsplat/python backend in this repo, not RTG-SLAM's native rasterizer;
- therefore this is a POLICY baseline (selection logic), not a full-system reproduction.
Label in paper/tables as "RTG-SLAM-policy (reimpl., matched budget)".
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class RTGSLAMConfig:
    color_error_thresh: float = 0.1
    depth_error_thresh: float = 0.05
    opacity_thresh_stable: float = 0.8
    ema_beta: float = 0.9
    ema_drift_thresh: float = 1.2  # unstable if recent_ema / long_ema > thresh
    max_select: int = 100000
    # densification mirrors RTG-SLAM: add at new / high-error pixels, capped
    max_new_per_frame: int = 500


class RTGSLAMPolicy:
    """Stateless-per-call policy; caller maintains error EMA buffers in StateStore."""

    def __init__(self, config: RTGSLAMConfig | None = None):
        self.cfg = config or RTGSLAMConfig()

    def select(
        self,
        *,
        color_err: np.ndarray,   # [N] per-Gaussian photometric residual
        depth_err: np.ndarray,   # [N]
        opacity: np.ndarray,     # [N]
        ema_recent: np.ndarray,  # [N] short-horizon error EMA from StateStore
        ema_long: np.ndarray,    # [N] long-horizon error EMA
        cost_us: np.ndarray,     # [N] modeled per-Gaussian cost (µs)
        budget_ms: float,        # total optimize budget for this frame
    ) -> Tuple[np.ndarray, Dict]:
        n = len(color_err)
        assert len(depth_err) == n and len(opacity) == n
        # 1. unstable mask: high error OR drift (re-activation), and not confidently stable-opaque
        high_err = (color_err > self.cfg.color_error_thresh) | (depth_err > self.cfg.depth_error_thresh)
        drift = ema_recent / np.maximum(ema_long, 1e-8)
        unstable = high_err | (drift > self.cfg.ema_drift_thresh)
        # stable opaque Gaussians are frozen (RTG-SLAM compact mapping)
        stable_opaque = (opacity > self.cfg.opacity_thresh_stable) & (~high_err)
        candidate = np.where(unstable & (~stable_opaque))[0]
        # 2. rank candidates by max(color, depth) error — original heuristic order
        score = np.maximum(color_err[candidate], depth_err[candidate])
        order = candidate[np.argsort(-score)]
        # 3. greedy knapsack by cost until budget
        budget_us = budget_ms * 1000.0
        sel, used = [], 0.0
        for idx in order:
            c = float(cost_us[idx]) if idx < len(cost_us) else 0.5
            if used + c > budget_us:
                continue
            sel.append(int(idx))
            used += c
            if len(sel) >= self.cfg.max_select:
                break
        sel = np.asarray(sel, dtype=np.int64)
        info = {
            "policy": "rtg_slam_reimpl",
            "n_candidate": int(len(candidate)),
            "n_selected": int(len(sel)),
            "budget_used_us": float(used),
            "budget_total_us": float(budget_us),
        }
        return sel, info
