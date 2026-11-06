"""Unit tests for value-gated growth (VGG) and the wall-clock-targeted budget (TT)."""
from types import SimpleNamespace

import pytest

from research import growth_controller as gc
from research import time_budget as tb


def _fake_pipe(frame_count=0):
    return SimpleNamespace(frame_count=frame_count, current_pose=None,
                           config={"densification": {"throttle_factor": 1.0}},
                           scheduler=SimpleNamespace(gpu_budget_ms=15.0))


def _drive(monkeypatch, values, p=None, n_recent=1000):
    """Feed a sequence of per-1000-Gaussian values (dB/kG) to the controller, one per measurement."""
    p = p or gc.VGGParams(start=0, every=1, ema=1.0, min_recent=1, backoff_max=1)
    st = gc.VGGState()
    pipe = _fake_pipe()
    it = iter(values)

    def fake(pipe_, rgb, depth, pose, window):
        v = next(it)
        return {"value_db": v * n_recent / 1000.0, "n_recent": n_recent, "n_total": 10 * n_recent}

    monkeypatch.setattr(gc, "spawn_value", fake)
    gs = []
    for i, _ in enumerate(values):
        pipe.frame_count = i
        gc.step(pipe, None, None, None, p, st)
        gs.append(st.g)
        assert pipe.config["densification"]["throttle_factor"] == st.g
    return gs, st


def test_vgg_throttles_when_spawns_are_worthless(monkeypatch):
    gs, _ = _drive(monkeypatch, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert gs[0] == pytest.approx(0.25) and gs[1] == pytest.approx(0.1)   # down=0.25, floor g_min=0.1
    assert min(gs) == pytest.approx(0.1)


def test_vgg_recovers_when_spawns_become_valuable(monkeypatch):
    gs, _ = _drive(monkeypatch, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert gs[1] == pytest.approx(0.1)
    assert gs[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(gs[1:], gs[2:]))


def test_vgg_hysteresis_band_leaves_g_unchanged(monkeypatch):
    gs, _ = _drive(monkeypatch, [0.0, 0.14, 0.14, 0.14])      # 0.14 lies between theta_lo=0.08 and theta_hi=0.2
    assert gs[0] == pytest.approx(0.25)
    assert gs[1:] == [pytest.approx(0.25)] * 3


def test_vgg_ignores_measurements_from_few_spawns(monkeypatch):
    p = gc.VGGParams(start=0, every=1, ema=1.0, min_recent=500, backoff_max=1)
    gs, st = _drive(monkeypatch, [0.0, 0.0], p=p, n_recent=100)
    assert gs == [1.0, 1.0] and st.value_ema is None


def test_vgg_backoff_lengthens_the_measurement_interval(monkeypatch):
    p = gc.VGGParams(start=0, every=1, ema=1.0, min_recent=1, backoff_max=8)
    st = gc.VGGState()
    pipe = _fake_pipe()
    calls = []

    def fake(pipe_, rgb, depth, pose, window):
        calls.append(pipe_.frame_count)
        return {"value_db": 5.0, "n_recent": 1000, "n_total": 10000}        # 5 dB/kG: clearly valuable

    monkeypatch.setattr(gc, "spawn_value", fake)
    for i in range(40):
        pipe.frame_count = i
        gc.step(pipe, None, None, None, p, st)
    gaps = [b - a for a, b in zip(calls, calls[1:])]
    assert gaps[:4] == [2, 4, 8, 8] and max(gaps) == 8           # interval doubles up to every*backoff_max


def test_vgg_backoff_resets_when_value_drops(monkeypatch):
    p = gc.VGGParams(start=0, every=1, ema=1.0, min_recent=1, backoff_max=8)
    st = gc.VGGState()
    pipe = _fake_pipe()
    vals = iter([5.0, 5.0, 5.0, 0.0])

    monkeypatch.setattr(gc, "spawn_value", lambda *a, **k: {"value_db": next(vals), "n_recent": 1000, "n_total": 10000})
    for fc in (0, 2, 6, 14):
        pipe.frame_count = fc
        gc.step(pipe, None, None, None, p, st)
    assert st.interval_mult == 1 and st.g == pytest.approx(0.25)


def _tt_frames(monkeypatch, times_ms, p=None):
    p = p or tb.TTParams(T_ms=100.0, gamma=0.5, m_max=16.0, ema=1.0, warmup_frames=0)
    st, pipe = tb.TTState(), _fake_pipe()
    clock = {"t": 0.0}
    monkeypatch.setattr(tb.time, "perf_counter", lambda: clock["t"])
    monkeypatch.setattr(tb.torch.cuda, "is_available", lambda: False)
    ms_trace = []
    for ms in times_ms:
        tb.before(pipe, p, st)
        ms_trace.append(pipe.scheduler.gpu_budget_ms)
        clock["t"] += ms / 1000.0
        tb.after(pipe, p, st)
    return st, ms_trace


def test_time_target_raises_budget_when_frames_are_fast(monkeypatch):
    st, trace = _tt_frames(monkeypatch, [50.0] * 6)
    assert trace[0] == pytest.approx(15.0)
    assert all(b >= a for a, b in zip(trace, trace[1:])) and st.m > 1.0


def test_time_target_never_goes_below_the_base_budget(monkeypatch):
    st, trace = _tt_frames(monkeypatch, [400.0] * 10)
    assert st.m == pytest.approx(1.0) and min(trace) == pytest.approx(15.0)


def test_time_target_is_capped_by_m_max(monkeypatch):
    p = tb.TTParams(T_ms=100.0, gamma=1.0, m_max=4.0, ema=1.0, warmup_frames=0)
    st, _ = _tt_frames(monkeypatch, [1.0] * 10, p=p)
    assert st.m == pytest.approx(4.0)


def test_time_target_converges_for_a_linear_cost_model(monkeypatch):
    # frame time = 40 ms + 2 ms per multiplier unit; the fixed point for T=100 is m = 30 -> clipped to 16,
    # so use a steeper cost to get an interior fixed point: 40 + 15 m = 100 -> m = 4
    p = tb.TTParams(T_ms=100.0, gamma=0.5, m_max=16.0, ema=0.5, warmup_frames=0)
    st, pipe = tb.TTState(), _fake_pipe()
    clock = {"t": 0.0}
    monkeypatch.setattr(tb.time, "perf_counter", lambda: clock["t"])
    monkeypatch.setattr(tb.torch.cuda, "is_available", lambda: False)
    for _ in range(60):
        tb.before(pipe, p, st)
        clock["t"] += (40.0 + 15.0 * st.m) / 1000.0
        tb.after(pipe, p, st)
    assert st.m == pytest.approx(4.0, rel=0.05)
