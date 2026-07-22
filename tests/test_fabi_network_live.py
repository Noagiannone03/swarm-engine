"""Opt-in qualification against an authenticated Iroh relay.

The normal test suite skips this module unless a relay URL and token source are
provided explicitly. Secrets are never embedded in test output or source.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest
from lattica import rpc_method, rpc_stream_iter

from fabi_network.rpc import IrohRpcRuntime


def _live_relay_credentials() -> tuple[str, str] | None:
    relay_url = os.environ.get("FABI_NETWORK_LIVE_RELAY_URL", "").strip()
    token = os.environ.get("FABI_NETWORK_LIVE_RELAY_TOKEN", "").strip()
    token_file = os.environ.get("FABI_NETWORK_LIVE_RELAY_TOKEN_FILE", "").strip()
    if token and token_file:
        raise ValueError("set only one live relay token source")
    if token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
        if token.startswith("IROH_RELAY_ACCESS_TOKEN="):
            token = token.removeprefix("IROH_RELAY_ACCESS_TOKEN=").strip()
    return (relay_url, token) if relay_url and token else None


_LIVE_RELAY = _live_relay_credentials()


@pytest.mark.skipif(_LIVE_RELAY is None, reason="live Iroh relay credentials are not configured")
def test_stream_cancel_releases_saturated_producer_and_reuses_connection():
    relay_url, relay_token = _LIVE_RELAY
    generator_closed = threading.Event()

    class FloodService:
        @rpc_stream_iter
        def flood(self, request):
            del request
            try:
                for _ in range(100_000):
                    yield b"x" * 4096
            finally:
                generator_closed.set()

        @rpc_method
        def ping(self, request):
            return request

    with tempfile.TemporaryDirectory() as root:
        server = IrohRpcRuntime(
            Path(root) / "server.key",
            relay_url,
            relay_token,
            force_relay=True,
        )
        client = IrohRpcRuntime(
            Path(root) / "client.key",
            relay_url,
            relay_token,
            force_relay=True,
        )
        try:
            server.register(FloodService())
            stub = client.stub(server.endpoint_id, FloodService)
            stream = stub.flood(None)
            time.sleep(1.0)

            started = time.monotonic()
            stream.cancel()

            assert generator_closed.wait(5.0)
            assert time.monotonic() - started < 5.0
            assert stub.ping({"after": "cancel"}).result(timeout=10) == {"after": "cancel"}
            paths = json.loads(client.paths(server.endpoint_id))
            selected = next(path for path in paths if path.get("selected"))
            assert selected["kind"] == "relay"
        finally:
            client.close()
            server.close()
