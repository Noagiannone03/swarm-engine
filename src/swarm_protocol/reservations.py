"""Worker-local KV admission for Fabi Swarm Protocol v3.

The discovery catalogue may be stale, so it is never allowed to reserve memory.  This table is
the worker-side authority used by PREPARE/COMMIT/RELEASE.  All transitions are idempotent, fenced
by request epoch, bounded by a TTL, and serialized under one local lock.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from swarm_protocol.contracts import LayerSpan, ReservationLease, ReservationState


class ReservationError(RuntimeError):
    """Base class for explicit admission failures."""


class CapacityUnavailable(ReservationError):
    """The exact local KV envelope cannot fit a reservation."""


class ReservationConflict(ReservationError):
    """A reservation id was reused with different immutable fields."""


class ReservationNotFound(ReservationError):
    """The requested reservation is unknown to this worker."""


class ReservationExpired(ReservationError):
    """The reservation TTL elapsed before the requested transition."""


class StaleEpoch(ReservationError):
    """The caller attempted to use an epoch older than the worker fence."""


class InvalidReservationTransition(ReservationError):
    """The requested state transition is not legal."""


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


class LocalReservationTable:
    """Thread-safe authority for one worker's allocatable KV bytes."""

    _CAPACITY_STATES = frozenset({ReservationState.PREPARED, ReservationState.COMMITTED})

    def __init__(
        self,
        *,
        worker_id: str,
        allocatable_kv_bytes: int,
        clock_ms: Callable[[], int] = _system_clock_ms,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        if allocatable_kv_bytes < 0:
            raise ValueError("allocatable_kv_bytes must be non-negative")
        self.worker_id = worker_id
        self.allocatable_kv_bytes = allocatable_kv_bytes
        self._clock_ms = clock_ms
        self._leases: dict[str, ReservationLease] = {}
        self._highest_epoch_by_request: dict[str, int] = {}
        self._lock = threading.RLock()

    def _now_ms(self) -> int:
        now = int(self._clock_ms())
        if now < 0:
            raise RuntimeError("reservation clock returned a negative timestamp")
        return now

    def _expire_locked(self, now_ms: int) -> None:
        for reservation_id, lease in tuple(self._leases.items()):
            if lease.state in self._CAPACITY_STATES and lease.expires_at_ms <= now_ms:
                self._leases[reservation_id] = lease.model_copy(
                    update={"state": ReservationState.EXPIRED}
                )

    def _used_bytes_locked(self) -> int:
        return sum(
            lease.exact_kv_bytes
            for lease in self._leases.values()
            if lease.state in self._CAPACITY_STATES
        )

    def _advance_epoch_locked(self, request_id: str, epoch: int) -> None:
        current = self._highest_epoch_by_request.get(request_id)
        if current is not None and epoch < current:
            raise StaleEpoch(
                f"request {request_id} epoch {epoch} is older than worker fence {current}"
            )
        if current is not None and epoch == current:
            return

        self._highest_epoch_by_request[request_id] = epoch
        for reservation_id, lease in tuple(self._leases.items()):
            if (
                lease.request_id == request_id
                and lease.epoch < epoch
                and lease.state in self._CAPACITY_STATES
            ):
                self._leases[reservation_id] = lease.model_copy(
                    update={"state": ReservationState.RELEASED}
                )

    @staticmethod
    def _same_prepare(
        lease: ReservationLease,
        *,
        request_id: str,
        route_id: str,
        epoch: int,
        effective_span: LayerSpan,
        exact_kv_bytes: int,
    ) -> bool:
        return (
            lease.request_id == request_id
            and lease.route_id == route_id
            and lease.epoch == epoch
            and lease.effective_span == effective_span
            and lease.exact_kv_bytes == exact_kv_bytes
        )

    @property
    def used_kv_bytes(self) -> int:
        with self._lock:
            self._expire_locked(self._now_ms())
            return self._used_bytes_locked()

    @property
    def available_kv_bytes(self) -> int:
        return self.allocatable_kv_bytes - self.used_kv_bytes

    def prepare(
        self,
        *,
        reservation_id: str,
        request_id: str,
        route_id: str,
        epoch: int,
        effective_span: LayerSpan,
        exact_kv_bytes: int,
        ttl_ms: int,
    ) -> ReservationLease:
        if not reservation_id or not request_id or not route_id:
            raise ValueError("reservation_id, request_id and route_id must not be empty")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if exact_kv_bytes <= 0 or ttl_ms <= 0:
            raise ValueError("exact_kv_bytes and ttl_ms must be positive")

        with self._lock:
            now_ms = self._now_ms()
            self._expire_locked(now_ms)
            existing = self._leases.get(reservation_id)
            if existing is not None:
                if not self._same_prepare(
                    existing,
                    request_id=request_id,
                    route_id=route_id,
                    epoch=epoch,
                    effective_span=effective_span,
                    exact_kv_bytes=exact_kv_bytes,
                ):
                    raise ReservationConflict(
                        f"reservation id {reservation_id} was reused with different fields"
                    )
                if existing.state == ReservationState.EXPIRED:
                    raise ReservationExpired(f"reservation {reservation_id} has expired")
                return existing

            self._advance_epoch_locked(request_id, epoch)
            used_bytes = self._used_bytes_locked()
            if used_bytes + exact_kv_bytes > self.allocatable_kv_bytes:
                raise CapacityUnavailable(
                    f"worker {self.worker_id} has {self.allocatable_kv_bytes - used_bytes} KV "
                    f"bytes available, requested {exact_kv_bytes}"
                )

            lease = ReservationLease(
                reservation_id=reservation_id,
                request_id=request_id,
                route_id=route_id,
                epoch=epoch,
                worker_id=self.worker_id,
                effective_span=effective_span,
                exact_kv_bytes=exact_kv_bytes,
                state=ReservationState.PREPARED,
                issued_at_ms=now_ms,
                expires_at_ms=now_ms + ttl_ms,
            )
            self._leases[reservation_id] = lease
            return lease

    def commit(self, reservation_id: str, *, epoch: int) -> ReservationLease:
        with self._lock:
            now_ms = self._now_ms()
            self._expire_locked(now_ms)
            lease = self._leases.get(reservation_id)
            if lease is None:
                raise ReservationNotFound(f"unknown reservation {reservation_id}")
            self._advance_epoch_locked(lease.request_id, epoch)
            if epoch != lease.epoch:
                raise StaleEpoch(
                    f"commit epoch {epoch} does not match reservation epoch {lease.epoch}"
                )
            if lease.state == ReservationState.EXPIRED:
                raise ReservationExpired(f"reservation {reservation_id} has expired")
            if lease.state == ReservationState.COMMITTED:
                return lease
            if lease.state != ReservationState.PREPARED:
                raise InvalidReservationTransition(
                    f"cannot commit reservation {reservation_id} in state {lease.state.value}"
                )
            committed = lease.model_copy(update={"state": ReservationState.COMMITTED})
            self._leases[reservation_id] = committed
            return committed

    def renew(self, reservation_id: str, *, epoch: int, ttl_ms: int) -> ReservationLease:
        if ttl_ms <= 0:
            raise ValueError("ttl_ms must be positive")
        with self._lock:
            now_ms = self._now_ms()
            self._expire_locked(now_ms)
            lease = self._leases.get(reservation_id)
            if lease is None:
                raise ReservationNotFound(f"unknown reservation {reservation_id}")
            self._advance_epoch_locked(lease.request_id, epoch)
            if epoch != lease.epoch:
                raise StaleEpoch(
                    f"renew epoch {epoch} does not match reservation epoch {lease.epoch}"
                )
            if lease.state == ReservationState.EXPIRED:
                raise ReservationExpired(f"reservation {reservation_id} has expired")
            if lease.state != ReservationState.COMMITTED:
                raise InvalidReservationTransition(
                    f"cannot renew reservation {reservation_id} in state {lease.state.value}"
                )
            renewed = lease.model_copy(update={"expires_at_ms": now_ms + ttl_ms})
            self._leases[reservation_id] = renewed
            return renewed

    def release(self, reservation_id: str, *, epoch: int) -> ReservationLease | None:
        """Release a reservation. Missing releases are successful and return ``None``."""

        with self._lock:
            now_ms = self._now_ms()
            self._expire_locked(now_ms)
            lease = self._leases.get(reservation_id)
            if lease is None:
                return None
            current_epoch = self._highest_epoch_by_request.get(lease.request_id)
            if current_epoch is not None and epoch < current_epoch:
                raise StaleEpoch(
                    f"release epoch {epoch} is older than worker fence {current_epoch}"
                )
            if epoch != lease.epoch:
                raise StaleEpoch(
                    f"release epoch {epoch} does not match reservation epoch {lease.epoch}"
                )
            if lease.state in {ReservationState.RELEASED, ReservationState.EXPIRED}:
                return lease
            released = lease.model_copy(update={"state": ReservationState.RELEASED})
            self._leases[reservation_id] = released
            return released

    def fence_request(self, request_id: str, *, epoch: int) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        with self._lock:
            self._expire_locked(self._now_ms())
            self._advance_epoch_locked(request_id, epoch)

    def get(self, reservation_id: str) -> ReservationLease | None:
        with self._lock:
            self._expire_locked(self._now_ms())
            return self._leases.get(reservation_id)

    def snapshot(self) -> tuple[ReservationLease, ...]:
        with self._lock:
            self._expire_locked(self._now_ms())
            return tuple(sorted(self._leases.values(), key=lambda lease: lease.reservation_id))
