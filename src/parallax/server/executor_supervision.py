"""Lightweight, backend-independent executor supervision outcomes."""

from __future__ import annotations

from enum import Enum


class ExecutorSupervisionOutcome(str, Enum):
    """Reason why one executor generation stopped being supervised."""

    EXITED = "exited"
    RELOAD_REQUESTED = "reload_requested"
    MEMORY_SHUTDOWN_REQUESTED = "memory_shutdown_requested"


def failed_executor_outcome(
    *,
    has_memory_contract_failure: bool,
) -> ExecutorSupervisionOutcome | None:
    """Classify a failed executor without importing any model backend.

    A runtime memory-contract rejection is a recoverable control-plane event:
    the worker heartbeat must remain alive long enough to publish a measured
    lower context tier. Every other non-zero executor exit remains terminal.
    """

    if has_memory_contract_failure:
        return ExecutorSupervisionOutcome.RELOAD_REQUESTED
    return None
