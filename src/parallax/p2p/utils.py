"""
Utility functions for P2P server.

This module contains utility functions for the P2P server.
"""

import asyncio
import os
from concurrent.futures import Future
from threading import Thread
from typing import Awaitable, Collection

try:
    import uvloop
except ImportError:  # uvloop does not support Windows.
    uvloop = None


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def mdns_enabled_for_topology(
    *,
    initial_peers: Collection[str] = (),
    relay_servers: Collection[str] = (),
) -> bool:
    """Preserve Lattica's hybrid discovery default unless explicitly overridden.

    mDNS and public bootstrap/relay discovery solve different topologies and
    intentionally coexist upstream.  Keeping mDNS enabled lets peers on the
    same LAN establish a direct path without relying on NAT hairpinning, while
    relay + DCUtR remains available when no local peer is discovered.

    The topology arguments remain part of this compatibility helper because
    existing callers pass them, but they must not silently disable discovery.
    """

    configured = os.environ.get("PARALLAX_ENABLE_MDNS", "").strip().lower()
    if configured in _TRUE_VALUES:
        return True
    if configured in _FALSE_VALUES:
        return False
    return True


def log_nat_traversal_preflight(lattica, logger) -> bool | None:
    """Log Lattica's UDP NAT classification without treating it as a verdict.

    Lattica currently derives this value from UDP STUN mappings.  A symmetric
    result can make UDP hole punching harder, but it cannot prove that every
    TCP/QUIC DCUtR attempt will fail.  Connectivity is therefore qualified by
    the real RPC probes after peers connect, not by terminating the process at
    this advisory preflight.
    """

    try:
        is_symmetric_nat = lattica.is_symmetric_nat()
    except Exception:
        logger.exception("Could not classify the UDP NAT; continuing with live connectivity probes")
        return None

    if is_symmetric_nat is None:
        logger.warning("UDP NAT classification unavailable; continuing with live connectivity probes")
    elif is_symmetric_nat:
        logger.warning(
            "Symmetric UDP NAT detected; continuing with relay-assisted DCUtR. "
            "Only peers that pass a real direct RPC probe will be eligible for routing."
        )
    else:
        logger.info("UDP NAT classification is compatible with standard hole punching")
    return is_symmetric_nat


def switch_to_uvloop() -> asyncio.AbstractEventLoop:
    """Stop any running event loop, then create and set a fresh loop."""
    try:
        # if we're in jupyter, get rid of its built-in event loop
        asyncio.get_event_loop().stop()
    except RuntimeError:
        pass  # this allows running DHT from background threads with no event loop
    if uvloop is not None:
        uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


class AsyncWorker:
    """
    Async worker class for Parallax.

    This class is used to run coroutines in a separate thread.
    """

    def __init__(self) -> None:
        self._event_thread = None
        self._event_loop_fut = None
        self._pid = None

    def _run_event_loop(self):
        try:
            loop = switch_to_uvloop()
            self._event_loop_fut.set_result(loop)
        except Exception as e:
            self._event_loop_fut.set_exception(e)
        loop.run_forever()

    def run_coroutine(self, coro: Awaitable, return_future: bool = False):
        """Run a coroutine in a separate thread."""
        if self._event_thread is None or self._pid != os.getpid():
            self._pid = os.getpid()
            self._event_loop_fut = Future()
            self._event_thread = Thread(target=self._run_event_loop, daemon=True)
            self._event_thread.start()

        loop = self._event_loop_fut.result()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future if return_future else future.result()
