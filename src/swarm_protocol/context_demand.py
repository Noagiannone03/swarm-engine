"""Bounded, privacy-preserving workload demand for autonomous placement.

The request coordinator observes load because it already performs exact token
admission. It publishes only class aggregates; prompts, account identifiers and
request IDs never leave this process. Workers remain placement authorities.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from swarm_protocol.context_placement import ContextCapacityDemandMap, ContextClassDemand
from swarm_protocol.contracts import ModelManifest


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


class ContextDemandStore(Protocol):
    def publish_context_demand(self, demand: ContextCapacityDemandMap) -> None: ...


@dataclass(frozen=True)
class _TimedClass:
    at_ms: int
    context_tokens: int


@dataclass(frozen=True)
class _CompletedRequest:
    at_ms: int
    context_tokens: int
    service_time_ms: int


class ContextDemandWindow:
    """Convert exact admissions into a bounded multi-context demand snapshot.

    Desired concurrency follows Little's law (arrival rate × observed p95
    service time), then includes currently in-flight work and recent no-route
    pressure. This mirrors llm-d's request-rate/service-time/queue inputs while
    keeping Fabi's output deterministic and safe for a public DHT.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        region_id: str,
        *,
        window_ms: int = 5 * 60 * 1_000,
        target_independent_routes: int = 2,
        maximum_concurrent_slots: int = 32,
        maximum_events: int = 8_192,
    ) -> None:
        if not region_id or len(region_id) > 128:
            raise ValueError("demand region must contain 1 to 128 characters")
        if window_ms <= 0 or target_independent_routes <= 0:
            raise ValueError("demand window and route target must be positive")
        if maximum_concurrent_slots <= 0 or maximum_events <= 0:
            raise ValueError("demand bounds must be positive")
        self.manifest = manifest
        self.region_id = region_id
        self.window_ms = window_ms
        self.target_independent_routes = target_independent_routes
        self.maximum_concurrent_slots = maximum_concurrent_slots
        self.maximum_events = maximum_events
        self._admissions: deque[_TimedClass] = deque(maxlen=maximum_events)
        # Rejections are keyed by the private coordinator request id so HTTP,
        # route-reservation, or client retries cannot manufacture placement
        # pressure. Request ids never leave this process: snapshots contain
        # only per-class aggregates.
        self._rejections: OrderedDict[str, _TimedClass] = OrderedDict()
        self._completed: deque[_CompletedRequest] = deque(maxlen=maximum_events)
        self._inflight: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._lock = threading.RLock()

    def _class_for(self, required_context_tokens: int) -> int:
        if required_context_tokens <= 0:
            raise ValueError("required context tokens must be positive")
        for context_tokens in self.manifest.context_classes:
            if required_context_tokens <= context_tokens:
                return context_tokens
        raise ValueError("required context exceeds the signed model contract")

    def _prune_locked(self, now_ms: int) -> None:
        cutoff = now_ms - self.window_ms
        for events in (self._admissions, self._completed):
            while events and events[0].at_ms < cutoff:
                events.popleft()
        while self._rejections:
            request_id, event = next(iter(self._rejections.items()))
            if event.at_ms >= cutoff:
                break
            self._rejections.pop(request_id, None)

    def record_admission(
        self,
        request_id: str,
        *,
        required_context_tokens: int,
        now_ms: int,
    ) -> None:
        if not request_id or now_ms < 0:
            raise ValueError("admitted request identity and time must be valid")
        context_tokens = self._class_for(required_context_tokens)
        with self._lock:
            self._prune_locked(now_ms)
            if request_id in self._inflight:
                return
            self._admissions.append(_TimedClass(now_ms, context_tokens))
            self._inflight[request_id] = (context_tokens, now_ms)
            while len(self._inflight) > self.maximum_events:
                self._inflight.popitem(last=False)

    def record_no_route(
        self,
        request_id: str,
        *,
        required_context_tokens: int,
        now_ms: int,
    ) -> None:
        if not request_id or now_ms < 0:
            raise ValueError("rejected request identity and time must be valid")
        context_tokens = self._class_for(required_context_tokens)
        with self._lock:
            self._prune_locked(now_ms)
            if request_id in self._rejections:
                return
            self._rejections[request_id] = _TimedClass(now_ms, context_tokens)
            while len(self._rejections) > self.maximum_events:
                self._rejections.popitem(last=False)

    def record_completion(self, request_id: str, *, now_ms: int) -> None:
        if not request_id or now_ms < 0:
            raise ValueError("completed request identity and time must be valid")
        with self._lock:
            self._prune_locked(now_ms)
            current = self._inflight.pop(request_id, None)
            if current is None:
                return
            context_tokens, started_at_ms = current
            self._completed.append(
                _CompletedRequest(
                    at_ms=now_ms,
                    context_tokens=context_tokens,
                    service_time_ms=max(0, now_ms - started_at_ms),
                )
            )

    @staticmethod
    def _p95(values: list[int]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return float(ordered[index])

    def snapshot(self, *, now_ms: int, ttl_ms: int = 2 * 60 * 1_000) -> ContextCapacityDemandMap:
        if now_ms < 0 or ttl_ms <= 0 or ttl_ms > 5 * 60 * 1_000:
            raise ValueError("demand snapshot time or TTL is invalid")
        with self._lock:
            self._prune_locked(now_ms)
            admissions = tuple(self._admissions)
            rejections = tuple(self._rejections.values())
            completed = tuple(self._completed)
            inflight = tuple(self._inflight.values())

        classes: list[ContextClassDemand] = []
        for context_tokens in self.manifest.context_classes:
            admitted = sum(item.context_tokens == context_tokens for item in admissions)
            rejected = sum(item.context_tokens == context_tokens for item in rejections)
            active = sum(item[0] == context_tokens for item in inflight)
            service_times = [
                item.service_time_ms
                for item in completed
                if item.context_tokens == context_tokens
            ]
            p95_service_time_ms = self._p95(service_times)
            # A legitimate agentic turn may outlive the observation window.
            # Its admission has then expired when completion supplies the most
            # valuable service-time sample, so use the larger start/finish
            # count instead of erasing long-horizon demand.
            observed_requests = max(admitted, len(service_times))
            requests_per_minute = observed_requests * 60_000 / self.window_ms
            little_law_slots = math.ceil(
                requests_per_minute * p95_service_time_ms / 60_000
            )
            has_demand = observed_requests > 0 or rejected > 0 or active > 0
            desired_slots = (
                min(
                    self.maximum_concurrent_slots,
                    max(1, active, little_law_slots) + min(rejected, 4),
                )
                if has_demand
                else 0
            )
            desired_routes = self.target_independent_routes if has_demand else 0
            confidence = min(1.0, max(0.25, len(service_times) / 20)) if has_demand else 0.0
            layer_weight = (
                confidence * (1.0 + min(4.0, rejected / max(1, admitted)))
                if has_demand
                else 0.0
            )
            classes.append(
                ContextClassDemand(
                    context_tokens=context_tokens,
                    desired_independent_routes=desired_routes,
                    desired_concurrent_slots=desired_slots,
                    desired_replicas_by_layer=(desired_routes,) * self.manifest.num_layers,
                    demand_weight_by_layer=(layer_weight,) * self.manifest.num_layers,
                    admitted_requests_per_minute=requests_per_minute,
                    queued_requests=0,
                    no_route_rejections=rejected,
                    p95_service_time_ms=p95_service_time_ms,
                    confidence=confidence,
                )
            )
        return ContextCapacityDemandMap(
            model_swarm_id=self.manifest.model_swarm_id,
            region_id=self.region_id,
            issued_at_ms=now_ms,
            expires_at_ms=now_ms + ttl_ms,
            classes=tuple(classes),
        )


class ContextDemandAnnouncer:
    """Publish demand independently from request streaming and lease renewal."""

    def __init__(
        self,
        store: ContextDemandStore,
        region_id: str,
        *,
        clock_ms: Callable[[], int] = _system_clock_ms,
        publish_interval_s: float = 30.0,
        snapshot_ttl_ms: int = 2 * 60 * 1_000,
        start_thread: bool = True,
    ) -> None:
        if publish_interval_s <= 0:
            raise ValueError("demand publication interval must be positive")
        self._store = store
        self._region_id = region_id
        self._clock_ms = clock_ms
        self._publish_interval_s = publish_interval_s
        self._snapshot_ttl_ms = snapshot_ttl_ms
        self._windows: dict[str, ContextDemandWindow] = {}
        self._last_issued_at_ms: dict[str, int] = {}
        self._last_publish_at_ms: int | None = None
        self._last_error: dict[str, str] | None = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        if start_thread:
            self._thread = threading.Thread(
                target=self._run,
                name="SwarmV3ContextDemand",
                daemon=True,
            )
            self._thread.start()

    def _window(self, manifest: ModelManifest) -> ContextDemandWindow:
        with self._lock:
            current = self._windows.get(manifest.model_swarm_id)
            if current is not None:
                if current.manifest != manifest:
                    raise ValueError("model swarm id resolved to a different manifest")
                return current
            current = ContextDemandWindow(manifest, self._region_id)
            self._windows[manifest.model_swarm_id] = current
            return current

    def record_admission(
        self,
        request_id: str,
        manifest: ModelManifest,
        *,
        required_context_tokens: int,
    ) -> None:
        self._window(manifest).record_admission(
            request_id,
            required_context_tokens=required_context_tokens,
            now_ms=self._clock_ms(),
        )

    def record_no_route(
        self,
        request_id: str,
        manifest: ModelManifest,
        *,
        required_context_tokens: int,
    ) -> None:
        self._window(manifest).record_no_route(
            request_id,
            required_context_tokens=required_context_tokens,
            now_ms=self._clock_ms(),
        )

    def record_completion(self, request_id: str) -> None:
        now_ms = self._clock_ms()
        with self._lock:
            windows = tuple(self._windows.values())
        for window in windows:
            window.record_completion(request_id, now_ms=now_ms)

    def publish_once(self) -> int:
        now_ms = self._clock_ms()
        with self._lock:
            windows = tuple(self._windows.values())
        published = 0
        for window in windows:
            model_swarm_id = window.manifest.model_swarm_id
            with self._lock:
                if now_ms <= self._last_issued_at_ms.get(model_swarm_id, -1):
                    continue
            snapshot = window.snapshot(now_ms=now_ms, ttl_ms=self._snapshot_ttl_ms)
            self._store.publish_context_demand(snapshot)
            with self._lock:
                self._last_issued_at_ms[model_swarm_id] = now_ms
            published += 1
        with self._lock:
            if published:
                self._last_publish_at_ms = now_ms
            self._last_error = None
        return published

    def _run(self) -> None:
        while not self._stop_event.wait(self._publish_interval_s):
            try:
                self.publish_once()
            except Exception as exc:
                # Demand is advisory. Publication failure must not take down
                # route admission; the previous record expires naturally.
                with self._lock:
                    self._last_error = {
                        "code": type(exc).__name__,
                        "detail": str(exc)[:256],
                    }
                continue

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "mode": "advisory",
                "region_id": self._region_id,
                "models": sorted(self._windows),
                "last_publish_at_ms": self._last_publish_at_ms,
                "error": self._last_error,
            }

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
