"""Crash-durable fencing epochs for scheduler and worker control planes.

SQLite is deliberately used as the cross-platform durability primitive.  A
``BEGIN IMMEDIATE`` transaction serializes writers on Linux, macOS and Windows,
while ``synchronous=FULL`` makes a successful allocation durable before the
epoch is exposed on the wire.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from swarm_protocol.reservations import StaleEpoch


class EpochAllocator(Protocol):
    """Allocate monotonically increasing fencing tokens."""

    def current(self) -> int: ...

    def next_epoch(self) -> int: ...


class InMemoryEpochAllocator:
    """Process-local allocator used by isolated unit tests."""

    def __init__(self, initial: int = 0) -> None:
        if initial < 0:
            raise ValueError("initial epoch must be non-negative")
        self._epoch = initial
        self._lock = threading.Lock()

    def current(self) -> int:
        with self._lock:
            return self._epoch

    def next_epoch(self) -> int:
        with self._lock:
            self._epoch += 1
            return self._epoch


class SqliteEpochAllocator:
    """Host-local durable epoch allocator safe across processes and threads."""

    def __init__(self, path: str | Path, *, namespace: str = "scheduler") -> None:
        if not namespace:
            raise ValueError("epoch namespace must not be empty")
        self.path = Path(path)
        self.namespace = namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS epoch_counters (
                    namespace TEXT PRIMARY KEY,
                    epoch INTEGER NOT NULL CHECK(epoch >= 0)
                )
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO epoch_counters(namespace, epoch) VALUES (?, 0)",
                (self.namespace,),
            )
        finally:
            connection.close()

    def current(self) -> int:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT epoch FROM epoch_counters WHERE namespace = ?",
                (self.namespace,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise RuntimeError(f"missing durable epoch namespace {self.namespace!r}")
        return int(row[0])

    def next_epoch(self) -> int:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT epoch FROM epoch_counters WHERE namespace = ?",
                (self.namespace,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"missing durable epoch namespace {self.namespace!r}")
            epoch = int(row[0]) + 1
            connection.execute(
                "UPDATE epoch_counters SET epoch = ? WHERE namespace = ?",
                (epoch, self.namespace),
            )
            connection.execute("COMMIT")
            return epoch
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


class RequestEpochFence(Protocol):
    """Durably reject an older route plan for one logical request."""

    def advance(
        self,
        *,
        coordinator_id: str,
        request_id: str,
        epoch: int,
        retain_until_ms: int,
        now_ms: int,
    ) -> int: ...


class InMemoryRequestEpochFence:
    """Process-local request fence used when durable worker state is not configured."""

    def __init__(self) -> None:
        self._epochs: dict[tuple[str, str], tuple[int, int]] = {}
        self._lock = threading.Lock()

    def advance(
        self,
        *,
        coordinator_id: str,
        request_id: str,
        epoch: int,
        retain_until_ms: int,
        now_ms: int,
    ) -> int:
        _validate_fence_input(
            coordinator_id, request_id, epoch, retain_until_ms, now_ms
        )
        key = (coordinator_id, request_id)
        with self._lock:
            current = self._epochs.get(key)
            if current is not None and current[1] <= now_ms:
                current = None
                self._epochs.pop(key, None)
            if current is not None and epoch < current[0]:
                raise StaleEpoch(
                    f"request {request_id} epoch {epoch} is older than worker fence {current[0]}"
                )
            highest = epoch if current is None else max(epoch, current[0])
            retained = retain_until_ms if current is None else max(retain_until_ms, current[1])
            self._epochs[key] = (highest, retained)
            return highest


class SqliteRequestEpochFence:
    """Power-loss durable worker fence with bounded automatic retention."""

    def __init__(
        self,
        path: str | Path,
        *,
        cleanup_interval: int = 256,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if cleanup_interval <= 0:
            raise ValueError("cleanup interval must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cleanup_interval = cleanup_interval
        self._clock_ms = clock_ms
        self._writes = 0
        self._writes_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_epoch_fences (
                    coordinator_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    highest_epoch INTEGER NOT NULL CHECK(highest_epoch >= 0),
                    retain_until_ms INTEGER NOT NULL CHECK(retain_until_ms > 0),
                    PRIMARY KEY (coordinator_id, request_id)
                )
                """
            )
        finally:
            connection.close()

    def advance(
        self,
        *,
        coordinator_id: str,
        request_id: str,
        epoch: int,
        retain_until_ms: int,
        now_ms: int,
    ) -> int:
        _validate_fence_input(
            coordinator_id, request_id, epoch, retain_until_ms, now_ms
        )
        with self._writes_lock:
            self._writes += 1
            cleanup = self._writes % self.cleanup_interval == 0

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if cleanup:
                connection.execute(
                    "DELETE FROM request_epoch_fences WHERE retain_until_ms <= ?",
                    (now_ms,),
                )
            row = connection.execute(
                """
                SELECT highest_epoch, retain_until_ms
                FROM request_epoch_fences
                WHERE coordinator_id = ? AND request_id = ?
                """,
                (coordinator_id, request_id),
            ).fetchone()
            if row is not None and int(row[1]) <= now_ms:
                connection.execute(
                    """
                    DELETE FROM request_epoch_fences
                    WHERE coordinator_id = ? AND request_id = ?
                    """,
                    (coordinator_id, request_id),
                )
                row = None
            if row is not None and epoch < int(row[0]):
                raise StaleEpoch(
                    f"request {request_id} epoch {epoch} is older than worker fence {row[0]}"
                )
            highest = epoch if row is None else max(epoch, int(row[0]))
            retained = retain_until_ms if row is None else max(retain_until_ms, int(row[1]))
            connection.execute(
                """
                INSERT INTO request_epoch_fences(
                    coordinator_id, request_id, highest_epoch, retain_until_ms
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(coordinator_id, request_id) DO UPDATE SET
                    highest_epoch = excluded.highest_epoch,
                    retain_until_ms = excluded.retain_until_ms
                """,
                (coordinator_id, request_id, highest, retained),
            )
            connection.execute("COMMIT")
            return highest
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


def _validate_fence_input(
    coordinator_id: str,
    request_id: str,
    epoch: int,
    retain_until_ms: int,
    now_ms: int,
) -> None:
    if not coordinator_id or not request_id:
        raise ValueError("coordinator and request identities must not be empty")
    if epoch < 0 or now_ms < 0:
        raise ValueError("epoch and current time must be non-negative")
    if retain_until_ms <= now_ms:
        raise ValueError("request fence must be retained into the future")
