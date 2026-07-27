"""Active request-route lifecycle for Fabi Swarm Protocol v3."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from swarm_protocol.contracts import RecoveryLevel, RequestContract
from swarm_protocol.coordinator import (
    CommittedRoute,
    ControlTransport,
    RouteReservationCoordinator,
    RouteReservationError,
)
from swarm_protocol.epochs import EpochAllocator, InMemoryEpochAllocator
from swarm_protocol.shadow import SchedulerProtocolV3Shadow

logger = logging.getLogger(__name__)


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _steady_clock_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class _ActiveRoute:
    committed: CommittedRoute
    active: bool = True
    next_renew_at_ms: int = 0
    lease_deadline_ms: int = 0
    consecutive_renewal_failures: int = 0
    last_renewal_error: str | None = None
    operation_lock: threading.Lock = field(default_factory=threading.Lock)


class ActiveRouteRuntime:
    """Plan, reserve, renew and release exact v3 routes.

    The maintenance thread is intentionally independent of HTTP streaming and
    worker status RPCs. A long prefill therefore cannot starve reservation
    renewal. Transient renewal failures are retried only while the last
    acknowledged lease is still provably valid; expiry or worker departure
    fences the route out of the data plane.
    """

    def __init__(
        self,
        *,
        planner: SchedulerProtocolV3Shadow,
        transport: ControlTransport,
        nodes_provider: Callable[[], list[Any]],
        clock_ms: Callable[[], int] = _system_clock_ms,
        steady_clock_ms: Callable[[], int] = _steady_clock_ms,
        prepare_ttl_ms: int = 5_000,
        plan_ttl_ms: int = 10_000,
        session_ttl_ms: int = 60_000,
        renew_interval_ms: int = 20_000,
        renew_retry_interval_ms: int = 2_000,
        lease_expiry_guard_ms: int = 1_000,
        coordinator: RouteReservationCoordinator | None = None,
        epoch_allocator: EpochAllocator | None = None,
    ) -> None:
        if planner.mode != "active":
            raise ValueError("active route runtime requires an active v3 planner")
        if prepare_ttl_ms <= 0 or plan_ttl_ms <= prepare_ttl_ms:
            raise ValueError("plan TTL must be greater than the positive prepare TTL")
        if renew_interval_ms <= 0 or session_ttl_ms <= renew_interval_ms * 2:
            raise ValueError("session TTL must exceed two renewal intervals")
        if renew_retry_interval_ms <= 0 or renew_retry_interval_ms >= renew_interval_ms:
            raise ValueError("renew retry interval must be positive and below renew interval")
        if lease_expiry_guard_ms <= 0:
            raise ValueError("lease expiry guard must be positive")
        self.planner = planner
        self.transport = transport
        self.nodes_provider = nodes_provider
        self._clock_ms = clock_ms
        self._steady_clock_ms = steady_clock_ms
        self.prepare_ttl_ms = prepare_ttl_ms
        self.plan_ttl_ms = plan_ttl_ms
        self.session_ttl_ms = session_ttl_ms
        self.renew_interval_ms = renew_interval_ms
        self.renew_retry_interval_ms = renew_retry_interval_ms
        self.lease_expiry_guard_ms = lease_expiry_guard_ms
        self.coordinator = coordinator or RouteReservationCoordinator(
            transport, clock_ms=clock_ms, session_ttl_ms=session_ttl_ms
        )
        self._renew_attempt_budget_ms = int(
            getattr(self.coordinator, "command_ttl_ms", prepare_ttl_ms)
        )
        if (
            self._renew_attempt_budget_ms <= 0
            or session_ttl_ms
            <= self._renew_attempt_budget_ms + self.lease_expiry_guard_ms
        ):
            raise ValueError("session TTL leaves no safe renewal retry window")
        self.epoch_allocator = epoch_allocator or InMemoryEpochAllocator()
        self._routes: dict[str, _ActiveRoute] = {}
        self._request_locks: dict[str, threading.Lock] = {}
        self._failures: deque[dict[str, object]] = deque(maxlen=64)
        self._lock = threading.RLock()
        self._capacity_changed = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            name="SwarmV3RouteLeases",
            daemon=True,
        )
        self._maintenance_thread.start()

    def _now_ms(self) -> int:
        now = int(self._clock_ms())
        if now < 0:
            raise RuntimeError("active route clock returned a negative timestamp")
        return now

    def _steady_now_ms(self) -> int:
        now = int(self._steady_clock_ms())
        if now < 0:
            raise RuntimeError("active route steady clock returned a negative timestamp")
        return now

    def reserve(
        self,
        *,
        request_id: str,
        prompt_tokens: int,
        reserved_output_tokens: int,
        recovery_level: RecoveryLevel = RecoveryLevel.RESTARTABLE,
    ) -> tuple[str, ...]:
        """Atomically acquire a complete route before returning worker ids."""

        request_key = str(request_id)
        with self._lock:
            request_lock = self._request_locks.setdefault(request_key, threading.Lock())
        with request_lock:
            with self._lock:
                current = self._routes.get(request_key)
                if current is not None and current.active:
                    return tuple(stage.worker_id for stage in current.committed.plan.stages)
            # The durable write happens before the token is put into a signed
            # plan. A failed planning attempt intentionally burns an epoch:
            # fencing tokens may have gaps but can never move backwards.
            epoch = self.epoch_allocator.next_epoch()

            nodes = self.nodes_provider()
            model_swarm_id = self.planner.ready_model_swarm_id(nodes)
            request = RequestContract(
                request_id=request_key,
                model_swarm_id=model_swarm_id,
                prompt_tokens=prompt_tokens,
                reserved_output_tokens=reserved_output_tokens,
                recovery_level=recovery_level,
            )
            now_ms = self._now_ms()
            planned = self.planner.plan_request(
                nodes,
                request=request,
                coordinator_id=self.transport.peer_id(),
                epoch=epoch,
                reservation_deadline_ms=now_ms + self.prepare_ttl_ms,
                plan_expires_at_ms=now_ms + self.plan_ttl_ms,
            )
            # Measure lease safety against the coordinator's monotonic progress,
            # not worker-produced wall-clock timestamps. Starting the local
            # deadline before the RPC is conservative: the worker can only
            # install its full TTL after this point.
            lease_started_at_ms = self._steady_now_ms()
            committed = self.coordinator.reserve(planned.plan)
            acknowledged_at_ms = self._steady_now_ms()
            lease_deadline_ms = lease_started_at_ms + self.session_ttl_ms
            if acknowledged_at_ms >= lease_deadline_ms - self.lease_expiry_guard_ms:
                try:
                    self.coordinator.release(committed)
                finally:
                    raise RouteReservationError(
                        "route session lease was acknowledged too close to expiry"
                    )
            active = _ActiveRoute(
                committed=committed,
                next_renew_at_ms=acknowledged_at_ms + self.renew_interval_ms,
                lease_deadline_ms=lease_deadline_ms,
            )
            with self._lock:
                self._routes[request_key] = active
            return tuple(stage.worker_id for stage in active.committed.plan.stages)

    def is_active(self, request_id: str) -> bool:
        with self._lock:
            route = self._routes.get(str(request_id))
            return (
                route is not None
                and route.active
                and self._steady_now_ms() < route.lease_deadline_ms
            )

    def has_active_routes(self) -> bool:
        with self._lock:
            now_ms = self._steady_now_ms()
            return any(
                route.active and now_ms < route.lease_deadline_ms
                for route in self._routes.values()
            )

    def authority(self, request_id: str) -> dict[str, object] | None:
        """Return the immutable data-plane fence for one active request."""

        with self._lock:
            route = self._routes.get(str(request_id))
            if (
                route is None
                or not route.active
                or self._steady_now_ms() >= route.lease_deadline_ms
            ):
                return None
            plan = route.committed.plan
            return {
                "route_id": plan.route_id,
                "epoch": plan.epoch,
            }

    def release(self, request_id: str) -> bool:
        request_key = str(request_id)
        with self._lock:
            route = self._routes.get(request_key)
        if route is None:
            return False
        with route.operation_lock:
            with self._lock:
                if not route.active:
                    self._routes.pop(request_key, None)
                    return False
                route.active = False
            try:
                self.coordinator.release(route.committed)
            finally:
                with self._capacity_changed:
                    self._routes.pop(request_key, None)
                    self._capacity_changed.notify_all()
        return True

    def wait_for_capacity(self, timeout: float) -> bool:
        """Wake route retries after any active reservation releases."""

        with self._capacity_changed:
            return self._capacity_changed.wait(timeout=max(0.0, timeout))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            steady_now_ms = self._steady_now_ms()
            routes = [
                {
                    "request_id": request_id,
                    "route_id": route.committed.plan.route_id,
                    "epoch": route.committed.plan.epoch,
                    "workers": [stage.worker_id for stage in route.committed.plan.stages],
                    # Remote lease timestamps use worker clocks. Expose only
                    # the remaining duration measured by the local steady
                    # clock, never a meaningless cross-machine wall timestamp.
                    "lease_expires_in_ms": max(
                        0, route.lease_deadline_ms - steady_now_ms
                    ),
                    "renewal_failures": route.consecutive_renewal_failures,
                    "last_renewal_error": route.last_renewal_error,
                }
                for request_id, route in sorted(self._routes.items())
                if route.active
            ]
            return {
                "mode": "active",
                "active_routes": routes,
                "recent_failures": list(self._failures),
                "session_ttl_ms": self.session_ttl_ms,
                "renew_interval_ms": self.renew_interval_ms,
                "renew_retry_interval_ms": self.renew_retry_interval_ms,
                "lease_expiry_guard_ms": self.lease_expiry_guard_ms,
            }

    def close(self) -> None:
        self._stop_event.set()
        self._maintenance_thread.join(timeout=2.0)
        with self._lock:
            request_ids = list(self._routes)
        for request_id in request_ids:
            try:
                self.release(request_id)
            except Exception:
                logger.warning(
                    "Failed to release v3 route %s during shutdown",
                    request_id,
                    exc_info=True,
                )

    def _maintenance_loop(self) -> None:
        interval_seconds = min(2.0, self.renew_interval_ms / 1000)
        while not self._stop_event.wait(interval_seconds):
            self.maintain_once()

    def maintain_once(self) -> None:
        """Check membership and renew due leases once (also deterministic in tests)."""

        live_worker_ids = getattr(self.planner, "live_worker_ids", None)
        active_workers = live_worker_ids() if live_worker_ids is not None else None
        if active_workers is None:
            active_workers = {
                str(node.node_id)
                for node in self.nodes_provider()
                if getattr(node, "is_active", False)
            }
        with self._lock:
            routes = list(self._routes.items())
        now_ms = self._steady_now_ms()
        for request_id, route in routes:
            route_workers = {stage.worker_id for stage in route.committed.plan.stages}
            departed = sorted(route_workers - active_workers)
            if departed:
                self._invalidate(
                    request_id,
                    route,
                    RuntimeError(f"route workers departed: {departed}"),
                )
            elif route.next_renew_at_ms <= now_ms:
                self._renew_one(request_id, route)

    def _invalidate(
        self,
        request_id: str,
        route: _ActiveRoute,
        exc: Exception,
    ) -> None:
        with route.operation_lock:
            with self._lock:
                if self._routes.get(request_id) is not route or not route.active:
                    return
                route.active = False
                self._failures.append(
                    {
                        "request_id": request_id,
                        "route_id": route.committed.plan.route_id,
                        "epoch": route.committed.plan.epoch,
                        "error": f"{type(exc).__name__}: {exc}"[:256],
                        "failed_at_ms": self._now_ms(),
                    }
                )
            try:
                self.coordinator.release(route.committed)
            except Exception:
                logger.warning(
                    "Failed to clean up lost v3 route %s",
                    request_id,
                    exc_info=True,
                )
            finally:
                with self._capacity_changed:
                    self._routes.pop(request_id, None)
                    self._capacity_changed.notify_all()

    def _renew_one(self, request_id: str, route: _ActiveRoute) -> None:
        renewal_error: Exception | None = None
        with route.operation_lock:
            with self._lock:
                if self._routes.get(request_id) is not route or not route.active:
                    return
                attempt_started_at_ms = self._steady_now_ms()
                if (
                    attempt_started_at_ms
                    + self._renew_attempt_budget_ms
                    + self.lease_expiry_guard_ms
                    >= route.lease_deadline_ms
                ):
                    renewal_error = TimeoutError(
                        "no safe lease window remains for another renewal attempt"
                    )
            if renewal_error is None:
                try:
                    renewed = self.coordinator.renew(
                        route.committed,
                        ttl_ms=self.session_ttl_ms,
                    )
                except Exception as exc:
                    now_ms = self._steady_now_ms()
                    with self._lock:
                        if self._routes.get(request_id) is not route or not route.active:
                            return
                        route.consecutive_renewal_failures += 1
                        route.last_renewal_error = f"{type(exc).__name__}: {exc}"[:256]
                        safe_retry_deadline = (
                            route.lease_deadline_ms
                            - self._renew_attempt_budget_ms
                            - self.lease_expiry_guard_ms
                        )
                        if now_ms + self.renew_retry_interval_ms < safe_retry_deadline:
                            route.next_renew_at_ms = (
                                now_ms + self.renew_retry_interval_ms
                            )
                            logger.warning(
                                "Protocol-v3 route %s lease renewal failed "
                                "(attempt %d); retrying before acknowledged expiry: %s",
                                request_id,
                                route.consecutive_renewal_failures,
                                exc,
                            )
                            return
                    renewal_error = exc
                else:
                    acknowledged_at_ms = self._steady_now_ms()
                    with self._lock:
                        if self._routes.get(request_id) is not route or not route.active:
                            return
                        if (
                            acknowledged_at_ms
                            >= route.lease_deadline_ms - self.lease_expiry_guard_ms
                        ):
                            renewal_error = TimeoutError(
                                "lease renewal acknowledgement arrived after the safe deadline"
                            )
                        else:
                            route.committed = renewed
                            route.lease_deadline_ms = (
                                attempt_started_at_ms + self.session_ttl_ms
                            )
                            route.next_renew_at_ms = (
                                acknowledged_at_ms + self.renew_interval_ms
                            )
                            route.consecutive_renewal_failures = 0
                            route.last_renewal_error = None
                            return
        assert renewal_error is not None
        logger.error(
            "Protocol-v3 route %s lost its session lease: %s",
            request_id,
            renewal_error,
        )
        self._invalidate(request_id, route, renewal_error)
