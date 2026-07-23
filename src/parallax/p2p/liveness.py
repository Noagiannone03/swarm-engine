"""Shared liveness timings for the worker-to-scheduler control channel.

The worker currently sends one status update every ten seconds and allows one
update RPC to spend up to thirty seconds in flight.  The scheduler lease must
therefore outlive a complete failed RPC plus the retry interval; otherwise one
temporary transport stall is indistinguishable from a dead worker.
"""

from __future__ import annotations

import math

WORKER_HEARTBEAT_INTERVAL_SECONDS = 10.0
WORKER_HEARTBEAT_RPC_TIMEOUT_SECONDS = 30.0
MIN_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS = (
    WORKER_HEARTBEAT_RPC_TIMEOUT_SECONDS + 3 * WORKER_HEARTBEAT_INTERVAL_SECONDS
)
DEFAULT_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS = 120.0


def validate_scheduler_heartbeat_timeout(value: float) -> float:
    """Validate the product lease without constraining low-level test schedulers."""

    timeout = float(value)
    if not math.isfinite(timeout):
        raise ValueError("heartbeat timeout must be finite")
    if timeout < MIN_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS:
        raise ValueError(
            "heartbeat timeout must be at least "
            f"{MIN_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS:.0f}s: one worker update may wait "
            f"{WORKER_HEARTBEAT_RPC_TIMEOUT_SECONDS:.0f}s and retries every "
            f"{WORKER_HEARTBEAT_INTERVAL_SECONDS:.0f}s"
        )
    return timeout
