"""
Structured event emitter for the Fabi CLI.

Parallax already logs human-readable messages via the standard logger, but the
Fabi CLI needs to drive a TUI that reflects worker state in real time. Parsing
free-form log lines is fragile — formats change, log levels can be suppressed,
strings get translated. Instead we emit one-line JSON events on stdout with a
fixed `[FABI]` prefix that the CLI greps.

Design choices:
  - stdout (not stderr): the CLI tee's worker stdout into a ring buffer; using
    stdout makes events trivially capturable from Node's child_process.
  - One line per event: makes line-based parsing on the CLI side bulletproof,
    no need to track JSON structure across chunks.
  - Magic prefix `[FABI]` followed by a space and a JSON object: easy to grep,
    easy to ignore for humans tailing the logs.
  - `flush=True`: never let buffered stdout mask a phase transition; the cost
    is negligible compared to the once-per-second cadence we emit at.

Events are intentionally append-only and free-form: any new field added on
the Python side is forwarded as-is by the CLI parser. Older CLI versions just
ignore unknown fields.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any


_PREFIX = "[FABI]"


def emit(event: str, **fields: Any) -> None:
    """Emit one event to the Fabi CLI.

    `event` is a short snake_case verb describing what just happened. Extra
    keyword arguments are merged into the JSON payload. Non-serializable values
    are coerced to string so a bug in a caller never silently swallows the
    event.
    """
    payload: dict[str, Any] = {"event": event, "ts": time.time()}
    for k, v in fields.items():
        try:
            json.dumps(v)
            payload[k] = v
        except (TypeError, ValueError):
            payload[k] = str(v)
    try:
        # Use print's default newline and flush — no logger involvement so the
        # CLI sees the event regardless of log_level filtering.
        print(f"{_PREFIX} {json.dumps(payload, ensure_ascii=False)}", flush=True)
    except Exception:
        # Never let an emission failure crash the worker. The CLI degrades
        # gracefully — it just won't see this transition.
        pass
