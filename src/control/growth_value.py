"""Online, label-free estimate of the value of recently spawned Gaussians.

Given the incoming frame (not yet used for any update) and its pose, render it twice with the
current map: once as is, once with the Gaussians created in the last ``window`` frames made
transparent. The PSNR difference is the counterfactual value, on an unseen frame, of those
Gaussians together with the updates they have received. Nothing is modified permanently.
"""
from typing import Dict, Optional

import torch


@torch.no_grad()
def spawn_value(pipe, rgb: torch.Tensor, depth: torch.Tensor, pose: torch.Tensor,
                window: int = 5) -> Optional[Dict[str, float]]:
    model = pipe.gaussian_model
    store = model.state_store
    n = model.num_gaussians
    born = store.creation_frames[:n]
    recent = (pipe.frame_count - born) < window
    n_recent = int(recent.sum().item())
    if n_recent == 0 or n_recent == n:
        return None
    full = pipe.eval_frame(rgb=rgb, depth=depth, pose=pose)
    opac = model._opacity
    saved = opac.data[:n][recent].clone()
    opac.data[:n][recent] = -20.0
    try:
        without = pipe.eval_frame(rgb=rgb, depth=depth, pose=pose)
    finally:
        opac.data[:n][recent] = saved
    return {"psnr_full": float(full["psnr"]), "psnr_without": float(without["psnr"]),
            "value_db": float(full["psnr"]) - float(without["psnr"]),
            "n_recent": n_recent, "n_total": n,
            "depth_full": float(full["depth_l1"]), "depth_without": float(without["depth_l1"])}
