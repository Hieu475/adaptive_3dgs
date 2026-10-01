"""Novel-view holdout + ATE evaluation utilities (ICCV-ready protocol).

Protocol:
- Every ``holdout_every``-th frame (default 8) is HELD OUT from mapping:
  rendered via ``pipeline.eval_frame`` (no densify/schedule/optimize) and
  scored as holdout PSNR/SSIM. These views are unseen for optimization
  (online-holdout proxy; disclosed as such, not an independent novel
  trajectory).
- Train frames go through ``pipeline.process_frame`` as usual.
- ATE: RMSE over translation between estimated and GT poses. Current harness
  runs with GT poses (tracking decoupled; ``ICPTracker`` is a stub), so ATE
  is 0 by construction — reported explicitly as ``ate_gt_anchored`` with the
  limitation disclosed. Do NOT present as SLAM tracking result.

No pipeline imports here (keeps research/ dependency-clean); the loop lives in
``experiments/run_holdout_benchmark.py``.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch


def compute_ate(est_poses: List[torch.Tensor], gt_poses: List[torch.Tensor]) -> Dict[str, float]:
    """Translation RMSE (+max) between two SE(3) trajectories.

    No alignment (assumes shared frame, true for GT-anchored runs).
    Returns zeros for empty input.
    """
    if not est_poses or not gt_poses or len(est_poses) != len(gt_poses):
        return {"ate_rmse_m": 0.0, "ate_max_m": 0.0, "n_poses": 0}
    errs = []
    for e, g in zip(est_poses, gt_poses):
        e = e.detach().cpu() if isinstance(e, torch.Tensor) else torch.as_tensor(e)
        g = g.detach().cpu() if isinstance(g, torch.Tensor) else torch.as_tensor(g)
        errs.append(float((e[:3, 3] - g[:3, 3]).norm().item()))
    a = np.asarray(errs)
    return {"ate_rmse_m": float(np.sqrt((a ** 2).mean())), "ate_max_m": float(a.max()),
            "n_poses": int(len(a))}


def summarize_holdout(train_psnrs: List[float], hold_psnrs: List[float]) -> Dict[str, float]:
    """Mean train vs holdout PSNR + generalization gap (train - holdout)."""
    t = np.asarray(train_psnrs, dtype=float) if train_psnrs else np.zeros(0)
    h = np.asarray(hold_psnrs, dtype=float) if hold_psnrs else np.zeros(0)
    return {
        "train_psnr": float(t.mean()) if len(t) else 0.0,
        "holdout_psnr": float(h.mean()) if len(h) else 0.0,
        "gen_gap_db": float(t.mean() - h.mean()) if len(t) and len(h) else 0.0,
        "n_train": int(len(t)),
        "n_holdout": int(len(h)),
    }
