import numpy as np
from research.baselines import RTGSLAMPolicy


def test_rtg_slam_respects_budget():
    rng = np.random.default_rng(42)
    n = 200
    pol = RTGSLAMPolicy()
    sel, info = pol.select(
        color_err=rng.random(n), depth_err=rng.random(n) * 0.1,
        opacity=np.full(n, 0.5), ema_recent=rng.random(n), ema_long=np.full(n, 0.5),
        cost_us=np.full(n, 50.0), budget_ms=2.0,
    )
    assert info["budget_used_us"] <= 2000.0 + 1e-6
    assert info["policy"] == "rtg_slam_reimpl"


def test_rtg_slam_freezes_stable_opaque():
    n = 10
    pol = RTGSLAMPolicy()
    sel, _ = pol.select(
        color_err=np.zeros(n), depth_err=np.zeros(n),
        opacity=np.full(n, 0.95), ema_recent=np.zeros(n), ema_long=np.ones(n),
        cost_us=np.full(n, 10.0), budget_ms=15.0,
    )
    assert len(sel) == 0  # all stable-opaque, nothing to optimize
