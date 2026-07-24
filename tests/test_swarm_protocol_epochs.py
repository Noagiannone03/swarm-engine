from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from swarm_protocol.epochs import SqliteEpochAllocator, SqliteRequestEpochFence
from swarm_protocol.reservations import StaleEpoch


def test_sqlite_epoch_allocator_survives_restart_and_serializes_writers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scheduler-control.sqlite3"
    first = SqliteEpochAllocator(path, namespace="routes")
    second = SqliteEpochAllocator(path, namespace="routes")

    with ThreadPoolExecutor(max_workers=8) as pool:
        epochs = list(
            pool.map(
                lambda index: (first if index % 2 else second).next_epoch(),
                range(40),
            )
        )

    assert sorted(epochs) == list(range(1, 41))
    assert SqliteEpochAllocator(path, namespace="routes").current() == 40
    assert SqliteEpochAllocator(path, namespace="allocations").next_epoch() == 1


def test_sqlite_epoch_allocator_fails_closed_on_corrupt_state(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(b"this is not a sqlite database")

    with pytest.raises(Exception, match="database"):
        SqliteEpochAllocator(path)


def test_request_epoch_fence_survives_worker_restart_and_expires_bounded_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "worker-control.sqlite3"
    first = SqliteRequestEpochFence(path, cleanup_interval=1)
    assert (
        first.advance(
            coordinator_id="coordinator",
            request_id="request",
            epoch=8,
            retain_until_ms=2_000,
            now_ms=1_000,
        )
        == 8
    )

    restarted = SqliteRequestEpochFence(path, cleanup_interval=1)
    with pytest.raises(StaleEpoch, match="worker fence 8"):
        restarted.advance(
            coordinator_id="coordinator",
            request_id="request",
            epoch=7,
            retain_until_ms=2_000,
            now_ms=1_100,
        )

    # Once every signed message that established the fence is expired, keeping
    # the row forever provides no extra safety and would grow the database.
    assert (
        restarted.advance(
            coordinator_id="coordinator",
            request_id="request",
            epoch=1,
            retain_until_ms=3_000,
            now_ms=2_001,
        )
        == 1
    )
