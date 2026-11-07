"""Wall-clock-targeted optimization budget.

The modeled optimization budget is a poor proxy for real cost: it does not account for the
rendering time that grows with the map. This controller keeps the measured end-to-end frame time
near a target ``T_ms`` by scaling the modeled budget with a multiplier m in [1, m_max]:

    m <- clip(m * (T_ms / ema_frame_ms) ** gamma, 1, m_max)

so a compact, fast-to-render map automatically receives more optimization, and a large map
less. The spawn allowance stays at the base budget (extra compute cannot buy extra splats).

``mode="steps"`` controls the number of optimization steps per frame instead. Each step costs a fixed
render+backward pass regardless of how many Gaussians it updates, so the step count is the knob that
actually moves wall-clock time, while the modeled budget is held at a large constant.
"""
import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TTParams:
    T_ms: float = 100.0
    gamma: float = 0.5
    m_max: float = 16.0
    ema: float = 0.5
    warmup_frames: int = 3
    mode: str = "budget"      # "budget": scale the modeled budget; "steps": scale the optimization steps per frame
    n_init: float = 5.0       # steps mode: initial and minimum-to-maximum range of micro-steps
    n_min: float = 2.0
    n_max: float = 24.0
    m_fixed: float = 64.0     # steps mode: modeled budget multiplier, large enough never to bind


@dataclass
class TTState:
    m: float = 1.0
    n_steps: float = 5.0
    ema_ms: Optional[float] = None
    base_ms: Optional[float] = None
    n: int = 0
    t0: float = 0.0


def before(pipe, p: TTParams, st: TTState) -> None:
    if st.base_ms is None:
        st.base_ms = float(pipe.scheduler.gpu_budget_ms)
    if p.mode == "steps":
        if st.n == 0:
            st.n_steps = p.n_init
        pipe.scheduler.gpu_budget_ms = st.base_ms * p.m_fixed
        pipe.config.setdefault("training", {})["n_micro_steps"] = max(1, int(round(st.n_steps)))
    else:
        pipe.scheduler.gpu_budget_ms = st.base_ms * st.m
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    st.t0 = time.perf_counter()


def after(pipe, p: TTParams, st: TTState) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - st.t0) * 1000.0
    st.n += 1
    st.ema_ms = ms if st.ema_ms is None else p.ema * ms + (1 - p.ema) * st.ema_ms
    if p.mode == "steps":
        if st.n >= p.warmup_frames:
            st.n_steps = min(p.n_max, max(p.n_min, st.n_steps * (p.T_ms / st.ema_ms) ** p.gamma))
    elif st.n >= p.warmup_frames:
        st.m = min(p.m_max, max(1.0, st.m * (p.T_ms / st.ema_ms) ** p.gamma))
