"""Tests for the Fabi contribution gate (backend.server.contribution_gate).

Pure-logic tests using the in-memory lease fallback (no Redis). We construct a
fresh ContributionGate per test after setting the env, since the gate reads its
configuration in __init__.
"""

import importlib
import time

import pytest


def _make_gate(monkeypatch, **env):
    """Build a ContributionGate with a controlled environment (no Redis)."""
    for key in [
        "FABI_GATE",
        "FABI_GATE_LEASE_S",
        "FABI_GATE_ALLOWLIST",
        "FABI_GATE_REDIS_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    mod = importlib.import_module("backend.server.contribution_gate")
    gate = mod.ContributionGate()
    # Force the in-memory fallback to keep tests hermetic (no live Redis).
    gate._redis = None
    return gate


def test_disabled_by_default_is_open(monkeypatch):
    gate = _make_gate(monkeypatch)  # FABI_GATE unset → off
    assert gate.enabled is False
    assert gate.is_allowed(None) is True
    assert gate.is_allowed("whatever") is True
    # refresh is a no-op when disabled (must not raise)
    gate.refresh("tok", "node-1", "model-x")


def test_enabled_requires_a_lease(monkeypatch):
    gate = _make_gate(monkeypatch, FABI_GATE="on")
    assert gate.enabled is True
    assert gate.is_allowed(None) is False
    assert gate.is_allowed("") is False
    assert gate.is_allowed("unknown-token") is False


def test_refresh_then_allowed(monkeypatch):
    gate = _make_gate(monkeypatch, FABI_GATE="on")
    gate.refresh("tok-abc", "node-1", "Qwen/Qwen3-1.7B")
    assert gate.is_allowed("tok-abc") is True
    assert gate.is_allowed("other-tok") is False  # only the contributing account


def test_expired_lease_denied_and_purged(monkeypatch):
    gate = _make_gate(monkeypatch, FABI_GATE="on")
    gate.refresh("tok-exp", "node-1", "m")
    # Force expiry in the in-memory store (lease_s min is 10s, too slow to wait).
    h = next(iter(gate._mem))
    gate._mem[h] = time.time() - 1.0
    assert gate.is_allowed("tok-exp") is False
    assert gate._mem == {}  # expired entry purged on read


def test_allowlist_bypasses_lease(monkeypatch):
    gate = _make_gate(monkeypatch, FABI_GATE="on", FABI_GATE_ALLOWLIST="admin-1, admin-2")
    assert gate.is_allowed("admin-1") is True
    assert gate.is_allowed("admin-2") is True
    assert gate.is_allowed("not-admin") is False


def test_token_never_stored_in_clear(monkeypatch):
    gate = _make_gate(monkeypatch, FABI_GATE="on")
    secret = "super-secret-token"
    gate.refresh(secret, "node-1", "m")
    # The store key is a hash, never the raw token.
    assert secret not in gate._mem
    assert all(len(k) == 64 for k in gate._mem)  # sha256 hexdigest


def test_lease_seconds_floor(monkeypatch):
    gate = _make_gate(monkeypatch, FABI_GATE="on", FABI_GATE_LEASE_S="2")
    assert gate.lease_s == 10  # clamped to a sane minimum


def test_denial_payload_shape(monkeypatch):
    gate = _make_gate(monkeypatch, FABI_GATE="on")
    payload = gate.denial_payload("12D3KooWPEER")
    assert payload["error"]["code"] == "contribution_required"
    assert "12D3KooWPEER" in payload["error"]["join_command"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
