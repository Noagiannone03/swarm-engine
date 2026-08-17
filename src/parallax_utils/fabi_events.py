"""Stable, best-effort worker lifecycle events for Fabi clients."""

from __future__ import annotations

import json
import time
from typing import Any


_PREFIX = "[FABI]"


def emit(event: str, **fields: Any) -> None:
    """Write one machine-readable event without affecting worker execution."""

    payload: dict[str, Any] = {"event": event, "ts": time.time()}
    for key, value in fields.items():
        try:
            json.dumps(value)
            payload[key] = value
        except (TypeError, ValueError):
            payload[key] = str(value)
    try:
        print(f"{_PREFIX} {json.dumps(payload, ensure_ascii=False)}", flush=True)
    except Exception:
        # Telemetry is additive. A closed stdout or encoding failure must never
        # demote, restart, or otherwise perturb a healthy worker generation.
        pass
