from __future__ import annotations

import hashlib
import threading
import time

import pytest

from swarm_protocol.registry import RouteAuthorityKey, RouteAuthorityKeyset
from swarm_protocol.route_authority import RouteAuthorityTrustStore


def keyset(
    *,
    generation: int,
    public_key: str,
    issued_at_ms: int = 1_000,
    expires_at_ms: int = 10_000,
) -> RouteAuthorityKeyset:
    return RouteAuthorityKeyset(
        generation=generation,
        issued_at_ms=issued_at_ms,
        expires_at_ms=expires_at_ms,
        keys=(
            RouteAuthorityKey(
                key_id=hashlib.sha256(bytes.fromhex(public_key)).hexdigest(),
                public_key=public_key,
                not_before_ms=issued_at_ms,
                not_after_ms=expires_at_ms,
            ),
        ),
    )


class RotatingRegistry:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.refresh_called = threading.Event()
        self.calls = 0

    def route_authorities(self):
        self.calls += 1
        if self.calls > 1:
            self.refresh_called.set()
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_trust_store_refreshes_off_hot_path_and_retains_valid_authenticated_cache():
    clock = [1_000]
    first = keyset(generation=1, public_key="11" * 32)
    second = keyset(generation=2, public_key="22" * 32)
    registry = RotatingRegistry((first, second, RuntimeError("registry offline")))
    store = RouteAuthorityTrustStore(
        registry,
        refresh_interval_ms=10,
        clock_ms=lambda: clock[0],
    )

    assert store.snapshot(clock[0]).generation == 1
    clock[0] = 1_010
    assert store.snapshot(clock[0]).generation == 1
    assert registry.refresh_called.wait(timeout=1)

    deadline = time.monotonic() + 1
    while store.snapshot(clock[0]).generation != 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert store.snapshot(clock[0]).generation == 2

    registry.refresh_called.clear()
    clock[0] = 1_020
    assert store.snapshot(clock[0]).generation == 2
    assert registry.refresh_called.wait(timeout=1)
    deadline = time.monotonic() + 1
    while store.last_refresh_error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "registry offline" in (store.last_refresh_error or "")
    assert store.snapshot(clock[0]).generation == 2

    with pytest.raises(ValueError, match="not currently valid"):
        store.snapshot(second.expires_at_ms)
