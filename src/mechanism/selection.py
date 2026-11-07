"""Small, inspectable selection scores for mechanism control arms."""
from __future__ import annotations

import torch


def combined_residual_percentile(color_error: torch.Tensor,
                                 depth_error: torch.Tensor) -> torch.Tensor:
    """Sum within-cohort RGB and depth residual percentile ranks.

    Ranking each channel before combining avoids letting the different units
    of RGB and metric depth dictate the ordering. Ties are resolved stably by
    the caller's existing Gaussian order. Non-finite values receive the
    lowest rank so missing diagnostics do not attract updates.
    """
    color = torch.nan_to_num(color_error.float(), nan=float("-inf"),
                             posinf=float("-inf"), neginf=float("-inf"))
    depth = torch.nan_to_num(depth_error.float(), nan=float("-inf"),
                             posinf=float("-inf"), neginf=float("-inf"))
    if color.ndim != 1 or depth.ndim != 1 or color.shape != depth.shape:
        raise ValueError("color_error and depth_error must be matching 1D tensors")

    def percentile_rank(values: torch.Tensor) -> torch.Tensor:
        n = values.numel()
        if n == 0:
            return torch.empty_like(values)
        order = torch.argsort(values, stable=True)
        ranks = torch.empty(n, device=values.device, dtype=torch.float32)
        ranks[order] = torch.arange(n, device=values.device, dtype=torch.float32)
        return ranks / max(n - 1, 1)

    return percentile_rank(color) + percentile_rank(depth)
