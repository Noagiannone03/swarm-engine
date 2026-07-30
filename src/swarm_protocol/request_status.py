"""Bounded, reconnectable Request Agent phase events for local product UI."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

REQUEST_PHASES = frozenset(
    {
        "planning",
        "authorizing",
        "reserving",
        "prefilling",
        "decoding",
        "recovering",
        "replaying",
        "completed",
        "failed",
        "aborted",
        "released",
    }
)
TERMINAL_REQUEST_PHASES = frozenset({"completed", "failed", "aborted", "released"})


@dataclass(frozen=True)
class RequestPhaseEvent:
    event_id: int
    request_id: str
    phase: str
    occurred_at_ms: int
    epoch: int | None = None
    route_id: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "request_id": self.request_id,
            "phase": self.phase,
            "occurred_at_ms": self.occurred_at_ms,
        }
        if self.epoch is not None:
            payload["epoch"] = self.epoch
        if self.route_id is not None:
            payload["route_id"] = self.route_id
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class RequestPhaseFeed:
    """Thread-safe transition journal used only for observability.

    Request correctness remains in the durable recovery journal. This feed is
    intentionally bounded and exposes a current-state snapshot when a client
    reconnects after its requested event ID has fallen out of retention.
    """

    def __init__(self, *, max_events: int = 512) -> None:
        if max_events < 1:
            raise ValueError("request phase retention must be positive")
        self._events: deque[RequestPhaseEvent] = deque(maxlen=max_events)
        self._current: dict[str, RequestPhaseEvent] = {}
        self._next_event_id = 1
        self._closed = False
        self._condition = threading.Condition()

    def publish(
        self,
        request_id: str,
        phase: str,
        *,
        epoch: int | None = None,
        route_id: str | None = None,
        detail: str | None = None,
    ) -> RequestPhaseEvent:
        request_id = str(request_id).strip()
        if not request_id:
            raise ValueError("request phase event requires a request id")
        if phase not in REQUEST_PHASES:
            raise ValueError(f"unsupported Request Agent phase: {phase}")
        if epoch is not None and (isinstance(epoch, bool) or epoch < 0):
            raise ValueError("request phase epoch must be a non-negative integer")
        if route_id is not None:
            route_id = str(route_id).strip()
            if not route_id:
                raise ValueError("request phase route id must not be empty")
        if detail is not None:
            detail = str(detail).strip()[:512] or None

        with self._condition:
            if self._closed:
                raise RuntimeError("request phase feed is closed")
            previous = self._current.get(request_id)
            if (
                previous is not None
                and previous.phase == phase
                and previous.epoch == epoch
                and previous.route_id == route_id
                and previous.detail == detail
            ):
                return previous
            event = RequestPhaseEvent(
                event_id=self._next_event_id,
                request_id=request_id,
                phase=phase,
                occurred_at_ms=time.time_ns() // 1_000_000,
                epoch=epoch,
                route_id=route_id,
                detail=detail,
            )
            self._next_event_id += 1
            self._events.append(event)
            if phase in TERMINAL_REQUEST_PHASES:
                self._current.pop(request_id, None)
            else:
                self._current[request_id] = event
            self._condition.notify_all()
            return event

    def latest_event_id(self) -> int:
        with self._condition:
            return self._next_event_id - 1

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "last_event_id": self._next_event_id - 1,
                "active_requests": [
                    event.to_dict()
                    for _, event in sorted(self._current.items())
                ],
            }

    def read_after(self, event_id: int) -> tuple[tuple[RequestPhaseEvent, ...], bool]:
        """Return retained events after ``event_id`` and whether history has a gap."""

        if isinstance(event_id, bool) or event_id < 0:
            raise ValueError("last Request Agent event id must be a non-negative integer")
        with self._condition:
            oldest = self._events[0].event_id if self._events else self._next_event_id
            gap = event_id < oldest - 1
            if gap:
                return (), True
            return tuple(event for event in self._events if event.event_id > event_id), False

    def wait_after(
        self,
        event_id: int,
        *,
        timeout: float,
    ) -> tuple[tuple[RequestPhaseEvent, ...], bool]:
        if timeout < 0:
            raise ValueError("request phase wait timeout must not be negative")
        with self._condition:
            events, gap = self.read_after(event_id)
            if events or gap or self._closed or timeout == 0:
                return events, gap
            self._condition.wait(timeout)
            return self.read_after(event_id)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
