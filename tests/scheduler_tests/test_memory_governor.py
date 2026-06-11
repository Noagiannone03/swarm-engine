"""Tests for the Fabi memory governor (parallax_utils.memory_governor).

Pure-logic tests with an injected clock → fully deterministic. The whole point
is to PROVE the anti-yo-yo behaviour: noisy input must never produce a change.
"""

import importlib

import pytest


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += dt


def _gov(monkeypatch, clock, **env):
    for k in list(env):
        monkeypatch.setenv(k, str(env[k]))
    monkeypatch.setenv("PARALLAX_MEMORY_GOVERNOR", env.get("PARALLAX_MEMORY_GOVERNOR", "1"))
    mod = importlib.import_module("parallax_utils.memory_governor")
    return mod.MemoryGovernor(now_fn=clock)


def test_stable_input_never_changes_after_init(monkeypatch):
    c = Clock()
    g = _gov(monkeypatch, c)
    first = g.observe(16.0)
    assert first.changed is True and first.advertised_gb == 16.0
    for _ in range(50):
        c.tick(10.0)
        d = g.observe(16.0)
        assert d.changed is False
        assert d.advertised_gb == 16.0


def test_oscillation_is_absorbed_no_yoyo(monkeypatch):
    c = Clock()
    g = _gov(monkeypatch, c)
    g.observe(10.0)  # advertised = 10
    changes = 0
    for i in range(100):
        c.tick(30.0)  # well past dwell, to prove it's the hysteresis (not dwell) absorbing
        raw = 8.0 if i % 2 == 0 else 12.0  # ±20% around 10
        d = g.observe(raw)
        changes += int(d.changed)
    assert changes == 0  # noise never moves the advertised budget
    assert g.state()["advertised_gb"] == 10.0


def test_sustained_drop_shrinks(monkeypatch):
    c = Clock()
    g = _gov(monkeypatch, c)
    g.observe(20.0)            # advertised = 20
    c.tick(700.0)              # machine up a while → dwell elapsed
    shrunk_at = None
    for i in range(8):
        c.tick(10.0)
        d = g.observe(10.0)    # sustained drop to 10
        if d.changed:
            shrunk_at = i
            break
    assert shrunk_at is not None and shrunk_at <= 5  # shrinks after a few low ticks
    assert g.state()["advertised_gb"] < 16.0


def test_brief_dip_does_not_shrink(monkeypatch):
    c = Clock()
    g = _gov(monkeypatch, c)
    g.observe(20.0)
    c.tick(700.0)
    # one or two low ticks then recovery — below the 3-tick streak → no change
    for raw in (10.0, 10.0, 20.0, 20.0, 20.0):
        c.tick(10.0)
        d = g.observe(raw)
        assert d.changed is False
    assert g.state()["advertised_gb"] == 20.0


def test_critical_bypasses_dwell(monkeypatch):
    c = Clock()
    g = _gov(monkeypatch, c)
    g.observe(20.0)            # advertised = 20, last_change = now
    c.tick(10.0)               # only 10s later — far inside the 600s dwell
    d = g.observe(20.0, available_ratio=0.05)  # CRITICAL
    assert d.pressure == "CRITICAL"
    assert d.changed is True
    assert d.advertised_gb < 16.0  # shrunk immediately (~20*0.7=14)
    assert d.admission_scale == 0.0


def test_grow_is_slow_and_stepped(monkeypatch):
    c = Clock()
    g = _gov(monkeypatch, c)
    g.observe(10.0)            # advertised = 10
    c.tick(700.0)              # dwell elapsed
    grew_at = None
    for i in range(40):
        c.tick(10.0)
        d = g.observe(20.0)    # plenty of headroom now
        if d.changed:
            grew_at = i
            break
    assert grew_at is not None and grew_at >= 29  # needed ~30 consecutive high ticks
    assert g.state()["advertised_gb"] <= 10.0 * 1.25 + 0.01  # grew by a bounded step, not to 20


def test_admission_scale_tracks_pressure(monkeypatch):
    c = Clock()
    g = _gov(monkeypatch, c)
    g.observe(16.0)
    assert g.observe(16.0, available_ratio=0.5).admission_scale == 1.0   # NORMAL
    assert g.observe(16.0, available_ratio=0.15).admission_scale == 0.5  # ELEVATED
    assert g.observe(16.0, available_ratio=0.05).admission_scale == 0.0  # CRITICAL
    assert g.observe(16.0, psi_avg10=30.0).admission_scale == 0.0        # CRITICAL via PSI


def test_quant_gate_blocks_subthreshold_change(monkeypatch):
    c = Clock()
    g = _gov(monkeypatch, c)
    g.observe(10.0)
    # direct test of the min-delta gate: 0.5 GB < step max(1.0, 10%)=1.0 → no change
    assert g._apply(10.5) is False
    assert g.state()["advertised_gb"] == 10.0
    # 2 GB >= step → applied
    assert g._apply(8.0) is True
    assert g.state()["advertised_gb"] == 8.0


def test_killswitch_passthrough(monkeypatch):
    c = Clock()
    g = _gov(monkeypatch, c, PARALLAX_MEMORY_GOVERNOR="0")
    assert g.enabled is False
    for raw in (20.0, 5.0, 20.0, 5.0):
        c.tick(10.0)
        d = g.observe(raw, available_ratio=0.05)
        assert d.advertised_gb == round(raw, 2)  # raw passthrough
        assert d.changed is False
        assert d.admission_scale == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
