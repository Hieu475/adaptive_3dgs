"""Value-gated growth (VGG): scene-adaptive control of the spawn allowance.

Every ``every`` frames (from ``start``) the controller measures, on the incoming frame, how much
the Gaussians spawned in the last ``window`` frames improve the render (``research.growth_value``),
normalised per 1000 Gaussians. An exponential moving average of that value drives a multiplicative
rule with hysteresis on the spawn-cap factor g in [g_min, 1]:

    value_ema < theta_lo  ->  g <- max(g_min, g / 2)     (new Gaussians are not paying off)
    value_ema > theta_hi  ->  g <- min(1,     2 g)       (new Gaussians help; let the map grow)

g is applied through the existing coverage-gate-free spawn cap (``densification.throttle_factor``).
No scene labels, no learned parameters; theta_lo/theta_hi are the only tuned constants.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .growth_value import spawn_value


@dataclass
class VGGParams:
    start: int = 10
    every: int = 2
    window: int = 5
    theta_lo: float = 0.08      # dB per 1000 recently spawned Gaussians
    theta_hi: float = 0.20
    g_min: float = 0.1
    ema: float = 0.7
    min_recent: int = 200       # ignore measurements based on very few spawned Gaussians
    down: float = 0.25          # multiplicative decrease of g
    up: float = 2.0             # multiplicative increase of g
    backoff_max: int = 8        # measurement interval may grow up to every*backoff_max when growth is clearly valuable
    backoff_factor: float = 2.0 # value_ema must exceed backoff_factor*theta_hi to back off


@dataclass
class VGGState:
    g: float = 1.0
    value_ema: Optional[float] = None
    history: List[dict] = field(default_factory=list)
    interval_mult: int = 1
    last_measure: int = -10**9


def step(pipe, rgb: torch.Tensor, depth: torch.Tensor, pose: Optional[torch.Tensor], p: VGGParams, st: VGGState) -> None:
    """Measure and update g; call before processing the incoming frame."""
    fc = int(pipe.frame_count)
    if pose is None:
        pose = pipe.current_pose
    interval = p.every * st.interval_mult
    if fc >= p.start and fc - st.last_measure >= interval:
        st.last_measure = fc
        v = spawn_value(pipe, rgb, depth, pose, p.window)
        if v is not None and v["n_recent"] >= p.min_recent:
            per_kg = v["value_db"] / v["n_recent"] * 1000.0
            st.value_ema = per_kg if st.value_ema is None else p.ema * per_kg + (1 - p.ema) * st.value_ema
            if st.value_ema < p.theta_lo:
                st.g = max(p.g_min, st.g * p.down)
            elif st.value_ema > p.theta_hi:
                st.g = min(1.0, st.g * p.up)
            clearly_valuable = st.g >= 1.0 and st.value_ema > p.backoff_factor * p.theta_hi
            st.interval_mult = min(p.backoff_max, st.interval_mult * 2) if clearly_valuable else 1
            st.history.append({"frame": fc, "per_kg": per_kg, "ema": st.value_ema, "g": st.g, "n_recent": v["n_recent"]})
    pipe.config["densification"]["throttle_factor"] = st.g
