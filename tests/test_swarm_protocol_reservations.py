from concurrent.futures import ThreadPoolExecutor

import pytest

from swarm_protocol import (
    CapacityUnavailable,
    InvalidReservationTransition,
    LayerSpan,
    LocalReservationTable,
    ReservationConflict,
    ReservationExpired,
    ReservationState,
    StaleEpoch,
)


class ManualClock:
    def __init__(self, now_ms: int = 1_000) -> None:
        self.now_ms = now_ms

    def __call__(self) -> int:
        return self.now_ms


def prepare(
    table: LocalReservationTable,
    reservation_id: str = "reservation-1",
    *,
    request_id: str = "request-1",
    route_id: str = "route-1",
    epoch: int = 0,
    exact_kv_bytes: int = 400,
    ttl_ms: int = 100,
):
    return table.prepare(
        reservation_id=reservation_id,
        request_id=request_id,
        route_id=route_id,
        epoch=epoch,
        effective_span=LayerSpan(start=0, end=4),
        exact_kv_bytes=exact_kv_bytes,
        ttl_ms=ttl_ms,
    )


def test_prepare_commit_renew_release_are_idempotent() -> None:
    clock = ManualClock()
    table = LocalReservationTable(worker_id="worker-1", allocatable_kv_bytes=1_000, clock_ms=clock)

    first = prepare(table)
    assert prepare(table) == first
    assert first.state == ReservationState.PREPARED
    assert table.used_kv_bytes == 400

    committed = table.commit(first.reservation_id, epoch=0)
    assert table.commit(first.reservation_id, epoch=0) == committed
    assert committed.state == ReservationState.COMMITTED

    clock.now_ms += 20
    renewed = table.renew(first.reservation_id, epoch=0, ttl_ms=200)
    assert renewed.expires_at_ms == clock.now_ms + 200

    released = table.release(first.reservation_id, epoch=0)
    assert released is not None and released.state == ReservationState.RELEASED
    assert table.release(first.reservation_id, epoch=0) == released
    assert table.used_kv_bytes == 0
    assert table.release("missing", epoch=0) is None


def test_prepare_rejects_identifier_reuse_with_different_fields() -> None:
    table = LocalReservationTable(worker_id="worker-1", allocatable_kv_bytes=1_000)
    prepare(table)
    with pytest.raises(ReservationConflict):
        prepare(table, exact_kv_bytes=401)


def test_expired_prepare_releases_capacity_and_cannot_commit() -> None:
    clock = ManualClock()
    table = LocalReservationTable(worker_id="worker-1", allocatable_kv_bytes=500, clock_ms=clock)
    lease = prepare(table, exact_kv_bytes=500, ttl_ms=50)
    clock.now_ms += 50

    assert table.used_kv_bytes == 0
    assert table.get(lease.reservation_id).state == ReservationState.EXPIRED
    with pytest.raises(ReservationExpired):
        table.commit(lease.reservation_id, epoch=0)

    replacement = prepare(table, "reservation-2", exact_kv_bytes=500)
    assert replacement.state == ReservationState.PREPARED


def test_capacity_check_is_atomic_under_concurrent_prepares() -> None:
    table = LocalReservationTable(worker_id="worker-1", allocatable_kv_bytes=1_000)

    def attempt(index: int) -> bool:
        try:
            prepare(
                table,
                f"reservation-{index}",
                request_id=f"request-{index}",
                route_id=f"route-{index}",
                exact_kv_bytes=700,
            )
            return True
        except CapacityUnavailable:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, range(2)))

    assert sorted(outcomes) == [False, True]
    assert table.used_kv_bytes == 700


def test_new_epoch_fences_and_releases_old_request_reservation() -> None:
    table = LocalReservationTable(worker_id="worker-1", allocatable_kv_bytes=1_000)
    old = prepare(table, exact_kv_bytes=900)
    table.commit(old.reservation_id, epoch=0)

    new = prepare(
        table,
        "reservation-new",
        request_id=old.request_id,
        route_id="route-new",
        epoch=1,
        exact_kv_bytes=900,
    )
    assert table.get(old.reservation_id).state == ReservationState.RELEASED
    assert new.state == ReservationState.PREPARED
    assert table.used_kv_bytes == 900

    with pytest.raises(StaleEpoch):
        table.release(old.reservation_id, epoch=0)


def test_prepare_rejects_epoch_older_than_explicit_fence() -> None:
    table = LocalReservationTable(worker_id="worker-1", allocatable_kv_bytes=1_000)
    table.fence_request("request-1", epoch=4)
    with pytest.raises(StaleEpoch):
        prepare(table, epoch=3)


def test_only_committed_reservations_can_be_renewed() -> None:
    table = LocalReservationTable(worker_id="worker-1", allocatable_kv_bytes=1_000)
    lease = prepare(table)
    with pytest.raises(InvalidReservationTransition):
        table.renew(lease.reservation_id, epoch=0, ttl_ms=100)
