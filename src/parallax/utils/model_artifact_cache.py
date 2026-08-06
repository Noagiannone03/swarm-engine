"""Transactional, process-leased storage for Fabi's selective model packs.

Placement deliberately does not consult this cache.  Once a span has been
chosen from memory, context and swarm demand, this module proves that its exact
signed objects can be materialized without exhausting the host volume.

The lifecycle combines three established ideas:

* containerd-style leases keep in-use and in-flight content out of garbage
  collection;
* kubelet-style high/low watermarks avoid an eviction on every download; and
* GreedyDual-Size-Frequency retains objects that repeatedly avoid expensive
  downloads while dynamically ageing stale popularity.

Only Fabi-owned projection packs are removed here.  Hugging Face's shared blob
cache has its own reference graph and must only be pruned through
``huggingface_hub.scan_cache_dir().delete_revisions()``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Iterator, Protocol

import psutil
from filelock import FileLock

_IDENTITY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PACK_PATTERN = re.compile(
    r"^model-fabi-(?:layer-[0-9]{5}|embedding|output)\.safetensors$"
)
_RECEIPT_NAME = ".fabi-selective-artifacts.json"
_STATE_NAME = ".fabi-cache-v1.sqlite3"
_LOCK_NAME = ".fabi-cache-v1.lock"
_LEASES_NAME = ".leases-v1"
_CONTROL_WORKSPACE_BYTES = 1024 * 1024
_DEFAULT_MIN_FREE_FLOOR_BYTES = 1024**3
_DEFAULT_MIN_FREE_CAP_BYTES = 10 * 1024**3
_DEFAULT_HYSTERESIS_FLOOR_BYTES = 256 * 1024**2
_DEFAULT_HYSTERESIS_CAP_BYTES = 2 * 1024**3
_CACHE_ROOT_ENV = "FABI_MODEL_ARTIFACT_CACHE"
_CACHE_ROOTS_ENV = "FABI_MODEL_ARTIFACT_CACHE_ROOTS"


def default_model_artifact_cache_root() -> Path:
    """Return the backward-compatible primary Fabi projection cache."""

    configured = os.environ.get(_CACHE_ROOT_ENV)
    return (
        Path(configured).expanduser().resolve()
        if configured and configured.strip()
        else (Path.home() / ".cache" / "fabi" / "models").resolve()
    )


def configured_model_artifact_cache_roots() -> tuple[Path, ...]:
    """Return the primary cache plus explicitly authorized extra volumes.

    The multi-root value is a JSON string array so Windows drive-letter colons
    are never confused with a path separator.  A POSIX path-separated value is
    accepted for operator compatibility, but JSON is the product contract.
    """

    raw = os.environ.get(_CACHE_ROOTS_ENV, "").strip()
    extras: list[str] = []
    if raw:
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            if os.name == "nt":
                raise ValueError(f"{_CACHE_ROOTS_ENV} must be a JSON string array on Windows")
            decoded = raw.split(os.pathsep)
        if not isinstance(decoded, list) or not all(
            isinstance(item, str) and item.strip() for item in decoded
        ):
            raise ValueError(f"{_CACHE_ROOTS_ENV} must be a JSON string array")
        extras = decoded

    result: list[Path] = []
    seen: set[str] = set()
    for path in (default_model_artifact_cache_root(), *(Path(item) for item in extras)):
        resolved = path.expanduser().resolve()
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return tuple(result)


class ModelArtifactStorageError(RuntimeError):
    """The selected span cannot be stored without violating disk safety.

    The fields are deliberately machine-readable.  Executor subprocesses use
    them to report a recoverable placement constraint to the always-on P2P
    controller instead of turning disk pressure into a generic worker crash.
    """

    def __init__(
        self,
        message: str,
        *,
        artifact_identity: str,
        model_id: str,
        immutable_revision: str,
        snapshot: "CacheStorageSnapshot",
        missing_bytes: int,
    ) -> None:
        super().__init__(message)
        self.artifact_identity = artifact_identity
        self.model_id = model_id
        self.immutable_revision = immutable_revision
        self.snapshot = snapshot
        self.missing_bytes = missing_bytes

    def as_report(
        self,
        *,
        allocation_epoch: int | None,
        placement_generation: int | None,
        start_layer: int | None,
        end_layer: int | None,
    ) -> dict[str, object]:
        """Return the bounded cross-process failure contract."""

        snapshot = self.snapshot
        return {
            "kind": "artifact_storage",
            "allocation_epoch": allocation_epoch,
            "placement_generation": placement_generation,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "artifact_identity": self.artifact_identity,
            "model_id": self.model_id,
            "immutable_revision": self.immutable_revision,
            "required_content_growth_bytes": snapshot.required_content_growth_bytes,
            "required_growth_bytes": snapshot.required_growth_bytes,
            "missing_bytes": self.missing_bytes,
            "free_bytes": snapshot.free_bytes,
            "minimum_free_bytes": snapshot.minimum_free_bytes,
            "cleanup_hysteresis_bytes": snapshot.cleanup_hysteresis_bytes,
            "cache_bytes": snapshot.cache_bytes,
            "reserved_bytes": snapshot.reserved_bytes,
            "reclaimed_bytes": snapshot.reclaimed_bytes,
        }


class DiskUsage(Protocol):
    total: int
    used: int
    free: int


@dataclass(frozen=True)
class CacheObjectRequirement:
    """One deterministic file needed by a selected model projection."""

    relative_path: str
    size_bytes: int
    is_weight_pack: bool = False

    def __post_init__(self) -> None:
        normalized = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or normalized.is_absolute()
            or ".." in normalized.parts
            or str(normalized) != self.relative_path
        ):
            raise ValueError("cache object path must be a normalized relative POSIX path")
        if self.size_bytes < 0:
            raise ValueError("cache object size cannot be negative")
        if self.is_weight_pack and not _PACK_PATTERN.fullmatch(normalized.name):
            raise ValueError("weight pack name does not match the Fabi projection format")


@dataclass(frozen=True)
class CacheReservation:
    lease_id: str
    artifact_identity: str
    model_id: str
    immutable_revision: str
    required_objects: tuple[CacheObjectRequirement, ...]
    content_growth_bytes: int
    reserved_growth_bytes: int


@dataclass(frozen=True)
class CacheStorageSnapshot:
    total_bytes: int
    free_bytes: int
    minimum_free_bytes: int
    cleanup_hysteresis_bytes: int
    cache_bytes: int
    reserved_bytes: int
    required_content_growth_bytes: int
    required_growth_bytes: int
    reclaimed_bytes: int


@dataclass(frozen=True)
class CacheReservationPlan:
    """Non-destructive exact admission plan for one cache volume."""

    cache_root: Path
    snapshot: CacheStorageSnapshot
    reclaim_target_bytes: int
    reclaimable_bytes: int
    missing_bytes: int

    @property
    def can_reserve(self) -> bool:
        return self.missing_bytes == 0


def _parse_non_negative_bytes(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer byte count") from exc
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _atomic_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _tree_size(root: Path) -> int:
    total = 0
    try:
        entries = tuple(root.rglob("*"))
    except OSError:
        return 0
    for path in entries:
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


class ModelArtifactCache:
    """Manage exact selective packs under one Fabi-owned cache root."""

    def __init__(
        self,
        cache_root: Path,
        *,
        minimum_free_bytes: int | None = None,
        cleanup_hysteresis_bytes: int | None = None,
        maximum_cache_bytes: int | None = None,
        disk_usage: Callable[[Path], DiskUsage] = shutil.disk_usage,
        now_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.cache_root = cache_root.expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        usage = disk_usage(self.cache_root)
        automatic_minimum = max(
            _DEFAULT_MIN_FREE_FLOOR_BYTES,
            min(_DEFAULT_MIN_FREE_CAP_BYTES, usage.total // 50),
        )
        automatic_hysteresis = max(
            _DEFAULT_HYSTERESIS_FLOOR_BYTES,
            min(_DEFAULT_HYSTERESIS_CAP_BYTES, usage.total // 200),
        )
        configured_minimum = _parse_non_negative_bytes("FABI_MODEL_CACHE_MIN_FREE_BYTES")
        configured_hysteresis = _parse_non_negative_bytes(
            "FABI_MODEL_CACHE_HYSTERESIS_BYTES"
        )
        configured_maximum = _parse_non_negative_bytes("FABI_MODEL_CACHE_MAX_BYTES")
        self.minimum_free_bytes = (
            minimum_free_bytes
            if minimum_free_bytes is not None
            else configured_minimum
            if configured_minimum is not None
            else automatic_minimum
        )
        self.cleanup_hysteresis_bytes = (
            cleanup_hysteresis_bytes
            if cleanup_hysteresis_bytes is not None
            else configured_hysteresis
            if configured_hysteresis is not None
            else automatic_hysteresis
        )
        self.maximum_cache_bytes = (
            maximum_cache_bytes
            if maximum_cache_bytes is not None
            else configured_maximum
        )
        if self.minimum_free_bytes < 0 or self.cleanup_hysteresis_bytes < 0:
            raise ValueError("cache disk thresholds cannot be negative")
        if self.maximum_cache_bytes is not None and self.maximum_cache_bytes <= 0:
            raise ValueError("maximum cache bytes must be positive")
        self._disk_usage = disk_usage
        self._now_ns = now_ns
        self._lock = FileLock(str(self.cache_root / _LOCK_NAME))
        self._leases_root = self.cache_root / _LEASES_NAME
        self._leases_root.mkdir(parents=True, exist_ok=True)

    def plan(
        self,
        *,
        artifact_identity: str,
        required_objects: Iterable[CacheObjectRequirement],
    ) -> CacheReservationPlan:
        """Prove volume feasibility without deleting cached model content."""

        if not _IDENTITY_PATTERN.fullmatch(artifact_identity):
            raise ValueError("artifact identity must be a lowercase SHA-256")
        requirements = tuple(required_objects)
        if not requirements:
            raise ValueError("a cache reservation requires at least one object")
        if len({item.relative_path for item in requirements}) != len(requirements):
            raise ValueError("cache reservation contains duplicate object paths")
        with self._lock:
            return self._plan_locked(
                artifact_identity=artifact_identity,
                requirements=requirements,
            )

    def _plan_locked(
        self,
        *,
        artifact_identity: str,
        requirements: tuple[CacheObjectRequirement, ...],
    ) -> CacheReservationPlan:
        leases = self._live_leases_locked()
        reserved_bytes = sum(int(item.get("reserved_growth_bytes", 0)) for item in leases)
        reserved_content_bytes = sum(
            int(item.get("reserved_content_growth_bytes", 0)) for item in leases
        )
        projection_root = self.cache_root / artifact_identity
        content_growth = self._content_growth(projection_root, requirements)
        required_growth = content_growth + _CONTROL_WORKSPACE_BYTES
        usage = self._disk_usage(self.cache_root)
        cache_bytes = self._cache_size()
        reclaim_target = 0
        if usage.free - reserved_bytes - required_growth < self.minimum_free_bytes:
            reclaim_target = max(
                reclaim_target,
                self.minimum_free_bytes
                + self.cleanup_hysteresis_bytes
                + reserved_bytes
                + required_growth
                - usage.free,
            )
        if (
            self.maximum_cache_bytes is not None
            and cache_bytes + reserved_content_bytes + content_growth
            > self.maximum_cache_bytes
        ):
            quota_low_watermark = max(
                0,
                self.maximum_cache_bytes - self.cleanup_hysteresis_bytes,
            )
            reclaim_target = max(
                reclaim_target,
                cache_bytes
                + reserved_content_bytes
                + content_growth
                - quota_low_watermark,
            )
        protected = self._protected_objects(leases) | frozenset(
            (artifact_identity, item.relative_path)
            for item in requirements
            if item.is_weight_pack
        )
        reclaimable = self._reclaimable_bytes_locked(protected=protected)
        missing = max(reclaim_target - reclaimable, 0)
        return CacheReservationPlan(
            cache_root=self.cache_root,
            snapshot=CacheStorageSnapshot(
                total_bytes=usage.total,
                free_bytes=usage.free,
                minimum_free_bytes=self.minimum_free_bytes,
                cleanup_hysteresis_bytes=self.cleanup_hysteresis_bytes,
                cache_bytes=cache_bytes,
                reserved_bytes=reserved_bytes,
                required_content_growth_bytes=content_growth,
                required_growth_bytes=required_growth,
                reclaimed_bytes=0,
            ),
            reclaim_target_bytes=reclaim_target,
            reclaimable_bytes=reclaimable,
            missing_bytes=missing,
        )

    def reserve(
        self,
        *,
        artifact_identity: str,
        model_id: str,
        immutable_revision: str,
        required_objects: Iterable[CacheObjectRequirement],
    ) -> tuple[CacheReservation, CacheStorageSnapshot]:
        """Reserve exact net growth, evicting only unleased cold packs first."""

        if not _IDENTITY_PATTERN.fullmatch(artifact_identity):
            raise ValueError("artifact identity must be a lowercase SHA-256")
        requirements = tuple(required_objects)
        if not requirements:
            raise ValueError("a cache reservation requires at least one object")
        if len({item.relative_path for item in requirements}) != len(requirements):
            raise ValueError("cache reservation contains duplicate object paths")

        with self._lock:
            leases = self._live_leases_locked()
            reserved_bytes = sum(int(item.get("reserved_growth_bytes", 0)) for item in leases)
            reserved_content_bytes = sum(
                int(item.get("reserved_content_growth_bytes", 0)) for item in leases
            )
            projection_root = self.cache_root / artifact_identity
            content_growth = self._content_growth(projection_root, requirements)
            required_growth = content_growth + _CONTROL_WORKSPACE_BYTES
            usage = self._disk_usage(self.cache_root)
            cache_bytes = self._cache_size()

            free_pressure = (
                usage.free - reserved_bytes - required_growth < self.minimum_free_bytes
            )
            quota_pressure = (
                self.maximum_cache_bytes is not None
                and cache_bytes + reserved_content_bytes + content_growth
                > self.maximum_cache_bytes
            )
            reclaim_target = 0
            if free_pressure:
                reclaim_target = max(
                    reclaim_target,
                    self.minimum_free_bytes
                    + self.cleanup_hysteresis_bytes
                    + reserved_bytes
                    + required_growth
                    - usage.free,
                )
            if quota_pressure:
                assert self.maximum_cache_bytes is not None
                quota_low_watermark = max(
                    0,
                    self.maximum_cache_bytes - self.cleanup_hysteresis_bytes,
                )
                reclaim_target = max(
                    reclaim_target,
                    cache_bytes
                    + reserved_content_bytes
                    + content_growth
                    - quota_low_watermark,
                )

            protected = self._protected_objects(leases) | frozenset(
                (artifact_identity, item.relative_path)
                for item in requirements
                if item.is_weight_pack
            )
            reclaimed = self._evict_locked(reclaim_target, protected=protected)
            usage_after = self._disk_usage(self.cache_root)
            cache_after = self._cache_size()
            remaining_free = usage_after.free - reserved_bytes - required_growth
            remaining_cache = cache_after + reserved_content_bytes + content_growth
            if remaining_free < self.minimum_free_bytes or (
                self.maximum_cache_bytes is not None
                and remaining_cache > self.maximum_cache_bytes
            ):
                missing = max(self.minimum_free_bytes - remaining_free, 0)
                if self.maximum_cache_bytes is not None:
                    missing = max(missing, remaining_cache - self.maximum_cache_bytes)
                failure_snapshot = CacheStorageSnapshot(
                    total_bytes=usage_after.total,
                    free_bytes=usage_after.free,
                    minimum_free_bytes=self.minimum_free_bytes,
                    cleanup_hysteresis_bytes=self.cleanup_hysteresis_bytes,
                    cache_bytes=cache_after,
                    reserved_bytes=reserved_bytes,
                    required_content_growth_bytes=content_growth,
                    required_growth_bytes=required_growth,
                    reclaimed_bytes=reclaimed,
                )
                raise ModelArtifactStorageError(
                    "selected layer span needs "
                    f"{required_growth} additional bytes, but {missing} bytes cannot be "
                    "reclaimed without deleting active model packs or violating the disk reserve",
                    artifact_identity=artifact_identity,
                    model_id=model_id,
                    immutable_revision=immutable_revision,
                    snapshot=failure_snapshot,
                    missing_bytes=missing,
                )

            lease_id = uuid.uuid4().hex
            reservation = CacheReservation(
                lease_id=lease_id,
                artifact_identity=artifact_identity,
                model_id=model_id,
                immutable_revision=immutable_revision,
                required_objects=requirements,
                content_growth_bytes=content_growth,
                reserved_growth_bytes=required_growth,
            )
            _atomic_json(
                self._lease_path(lease_id),
                self._lease_payload(reservation, state="downloading"),
            )
            return reservation, CacheStorageSnapshot(
                total_bytes=usage_after.total,
                free_bytes=usage_after.free,
                minimum_free_bytes=self.minimum_free_bytes,
                cleanup_hysteresis_bytes=self.cleanup_hysteresis_bytes,
                cache_bytes=cache_after,
                reserved_bytes=reserved_bytes,
                required_content_growth_bytes=content_growth,
                required_growth_bytes=required_growth,
                reclaimed_bytes=reclaimed,
            )

    def commit(self, reservation: CacheReservation) -> None:
        """Record a verified cache hit and turn the download lease into an active lease."""

        with self._lock:
            lease_path = self._lease_path(reservation.lease_id)
            if not lease_path.is_file():
                raise RuntimeError("cache reservation disappeared before commit")
            required_packs = tuple(
                item for item in reservation.required_objects if item.is_weight_pack
            )
            projection_root = self.cache_root / reservation.artifact_identity
            for requirement in required_packs:
                path = projection_root / PurePosixPath(requirement.relative_path)
                if not path.is_file() or path.stat().st_size != requirement.size_bytes:
                    raise RuntimeError(
                        f"cannot commit missing cache pack {requirement.relative_path!r}"
                    )

            with self._database() as database:
                age = self._cache_age(database)
                now_ns = self._now_ns()
                for requirement in required_packs:
                    row = database.execute(
                        "SELECT frequency FROM objects WHERE identity = ? AND relative_path = ?",
                        (reservation.artifact_identity, requirement.relative_path),
                    ).fetchone()
                    frequency = int(row[0]) + 1 if row is not None else 1
                    # Signed pack bytes are also the bytes that must be fetched
                    # after a miss.  GDSF's cost/size term is therefore one;
                    # frequency plus dynamic age captures actual reuse without
                    # inventing a model-specific preference.
                    priority = age + frequency
                    database.execute(
                        """
                        INSERT INTO objects(
                            identity, relative_path, model_id, immutable_revision,
                            size_bytes, frequency, priority, last_access_ns
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(identity, relative_path) DO UPDATE SET
                            model_id = excluded.model_id,
                            immutable_revision = excluded.immutable_revision,
                            size_bytes = excluded.size_bytes,
                            frequency = excluded.frequency,
                            priority = excluded.priority,
                            last_access_ns = excluded.last_access_ns
                        """,
                        (
                            reservation.artifact_identity,
                            requirement.relative_path,
                            reservation.model_id,
                            reservation.immutable_revision,
                            requirement.size_bytes,
                            frequency,
                            priority,
                            now_ns,
                        ),
                    )

            current_pid = os.getpid()
            current_started_at = psutil.Process(current_pid).create_time()
            for path in self._leases_root.glob("*.json"):
                if path == lease_path:
                    continue
                payload = self._read_lease(path)
                if (
                    payload is not None
                    and payload.get("state") == "active"
                    and payload.get("pid") == current_pid
                    and math.isclose(
                        float(payload.get("process_started_at", -1)),
                        current_started_at,
                        abs_tol=0.01,
                    )
                ):
                    path.unlink(missing_ok=True)
            _atomic_json(lease_path, self._lease_payload(reservation, state="active"))

    def abort(self, reservation: CacheReservation) -> None:
        """Release a failed reservation; partial files remain GC-eligible."""

        with self._lock:
            self._lease_path(reservation.lease_id).unlink(missing_ok=True)

    def _lease_payload(self, reservation: CacheReservation, *, state: str) -> dict[str, object]:
        process = psutil.Process(os.getpid())
        return {
            "version": 1,
            "lease_id": reservation.lease_id,
            "state": state,
            "pid": process.pid,
            "process_started_at": process.create_time(),
            "artifact_identity": reservation.artifact_identity,
            "model_id": reservation.model_id,
            "immutable_revision": reservation.immutable_revision,
            "reserved_growth_bytes": (
                reservation.reserved_growth_bytes if state == "downloading" else 0
            ),
            "reserved_content_growth_bytes": (
                reservation.content_growth_bytes if state == "downloading" else 0
            ),
            "objects": [item.relative_path for item in reservation.required_objects],
            "updated_at_ns": self._now_ns(),
        }

    def _lease_path(self, lease_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", lease_id):
            raise ValueError("invalid cache lease identifier")
        return self._leases_root / f"{lease_id}.json"

    @staticmethod
    def _read_lease(path: Path) -> dict[str, object] | None:
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) and payload.get("version") == 1 else None

    @staticmethod
    def _lease_process_alive(payload: dict[str, object]) -> bool:
        try:
            pid = int(payload["pid"])
            expected_start = float(payload["process_started_at"])
            process = psutil.Process(pid)
            return process.is_running() and math.isclose(
                process.create_time(), expected_start, abs_tol=0.01
            )
        except (KeyError, TypeError, ValueError, psutil.Error):
            return False

    def _live_leases_locked(self) -> tuple[dict[str, object], ...]:
        live: list[dict[str, object]] = []
        for path in self._leases_root.glob("*.json"):
            payload = self._read_lease(path)
            if payload is None or not self._lease_process_alive(payload):
                path.unlink(missing_ok=True)
                continue
            live.append(payload)
        return tuple(live)

    @staticmethod
    def _protected_objects(
        leases: Iterable[dict[str, object]],
    ) -> frozenset[tuple[str, str]]:
        result: set[tuple[str, str]] = set()
        for lease in leases:
            identity = lease.get("artifact_identity")
            objects = lease.get("objects")
            if not isinstance(identity, str) or not isinstance(objects, list):
                continue
            result.update((identity, item) for item in objects if isinstance(item, str))
        return frozenset(result)

    @staticmethod
    def _content_growth(
        projection_root: Path,
        requirements: Iterable[CacheObjectRequirement],
    ) -> int:
        growth = 0
        for requirement in requirements:
            path = projection_root / PurePosixPath(requirement.relative_path)
            try:
                existing = path.stat().st_size if path.is_file() else 0
            except OSError:
                existing = 0
            growth += max(requirement.size_bytes - existing, 0)
        return growth

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        database = sqlite3.connect(self.cache_root / _STATE_NAME, timeout=30)
        try:
            database.execute("PRAGMA foreign_keys = ON")
            database.execute("PRAGMA synchronous = FULL")
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS objects(
                    identity TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    immutable_revision TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    frequency INTEGER NOT NULL,
                    priority REAL NOT NULL,
                    last_access_ns INTEGER NOT NULL,
                    PRIMARY KEY(identity, relative_path)
                )
                """
            )
            database.execute(
                "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value REAL NOT NULL)"
            )
            database.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES ('greedydual_age', 0)"
            )
            yield database
            database.commit()
        except Exception:
            database.rollback()
            raise
        finally:
            database.close()

    @staticmethod
    def _cache_age(database: sqlite3.Connection) -> float:
        row = database.execute(
            "SELECT value FROM metadata WHERE key = 'greedydual_age'"
        ).fetchone()
        return float(row[0]) if row is not None else 0.0

    def _projection_roots(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in self.cache_root.iterdir()
            if path.is_dir() and _IDENTITY_PATTERN.fullmatch(path.name)
        )

    def _cache_size(self) -> int:
        return sum(_tree_size(root) for root in self._projection_roots())

    def _inventory(self, database: sqlite3.Connection) -> list[tuple[float, int, str, str, int]]:
        inventory: list[tuple[float, int, str, str, int]] = []
        age = self._cache_age(database)
        present: set[tuple[str, str]] = set()
        for root in self._projection_roots():
            receipt = self._read_receipt(root / _RECEIPT_NAME)
            packs = receipt.get("packs", {}) if receipt is not None else {}
            model_id = str(receipt.get("model_id", "unknown")) if receipt else "unknown"
            revision = (
                str(receipt.get("immutable_revision", "unknown")) if receipt else "unknown"
            )
            names = {
                path.name
                for path in root.glob("model-fabi-*.safetensors")
                if _PACK_PATTERN.fullmatch(path.name)
            }
            if isinstance(packs, dict):
                names.update(name for name in packs if _PACK_PATTERN.fullmatch(str(name)))
            for name in names:
                path = root / name
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                key = (root.name, name)
                present.add(key)
                row = database.execute(
                    "SELECT priority, last_access_ns FROM objects "
                    "WHERE identity = ? AND relative_path = ?",
                    key,
                ).fetchone()
                if row is None:
                    priority = age + 1.0
                    last_access = path.stat().st_mtime_ns
                    database.execute(
                        """
                        INSERT INTO objects(
                            identity, relative_path, model_id, immutable_revision,
                            size_bytes, frequency, priority, last_access_ns
                        ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                        """,
                        (*key, model_id, revision, size, priority, last_access),
                    )
                else:
                    priority, last_access = float(row[0]), int(row[1])
                inventory.append((priority, last_access, root.name, name, size))
        rows = database.execute("SELECT identity, relative_path FROM objects").fetchall()
        database.executemany(
            "DELETE FROM objects WHERE identity = ? AND relative_path = ?",
            (tuple(row) for row in rows if tuple(row) not in present),
        )
        return inventory

    def _reclaimable_bytes_locked(
        self,
        *,
        protected: frozenset[tuple[str, str]],
    ) -> int:
        """Conservatively count bytes GC may remove on this volume.

        When every pack in a projection is unleased, eviction removes the
        complete Fabi-owned projection and its metadata.  For a partially
        protected projection only the exact unprotected pack sizes count; we
        deliberately do not guess how much a rewritten JSON receipt may save.
        """

        with self._database() as database:
            inventory = self._inventory(database)
        by_identity: dict[str, list[tuple[str, int]]] = {}
        for _, _, identity, name, size in inventory:
            by_identity.setdefault(identity, []).append((name, size))
        reclaimable = 0
        for identity, packs in by_identity.items():
            if any((identity, name) in protected for name, _ in packs):
                reclaimable += sum(
                    size for name, size in packs if (identity, name) not in protected
                )
            else:
                reclaimable += _tree_size(self.cache_root / identity)
        return reclaimable

    @staticmethod
    def _read_receipt(path: Path) -> dict[str, object] | None:
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _evict_locked(
        self,
        target_bytes: int,
        *,
        protected: frozenset[tuple[str, str]],
    ) -> int:
        if target_bytes <= 0:
            return 0
        before_free = self._disk_usage(self.cache_root).free
        with self._database() as database:
            inventory = sorted(self._inventory(database), key=lambda item: (item[0], item[1]))
            for priority, _, identity, name, _ in inventory:
                if (identity, name) in protected:
                    continue
                root = self.cache_root / identity
                projection_lock = FileLock(str(root.with_suffix(".lock")))
                with projection_lock:
                    receipt_path = root / _RECEIPT_NAME
                    receipt = self._read_receipt(receipt_path)
                    if receipt is not None and isinstance(receipt.get("packs"), dict):
                        packs = dict(receipt["packs"])
                        if name in packs:
                            packs.pop(name)
                            receipt["packs"] = packs
                            _atomic_json(receipt_path, receipt)
                    (root / name).unlink(missing_ok=True)
                    database.execute(
                        "DELETE FROM objects WHERE identity = ? AND relative_path = ?",
                        (identity, name),
                    )
                    database.execute(
                        "UPDATE metadata SET value = ? WHERE key = 'greedydual_age'",
                        (priority,),
                    )
                    if not any(root.glob("model-fabi-*.safetensors")):
                        shutil.rmtree(root, ignore_errors=True)
                reclaimed = max(self._disk_usage(self.cache_root).free - before_free, 0)
                if reclaimed >= target_bytes:
                    break
        return max(self._disk_usage(self.cache_root).free - before_free, 0)


class ModelArtifactCachePool:
    """Select one authorized volume after placement, before materialization."""

    def __init__(self, caches: Iterable[ModelArtifactCache]) -> None:
        self.caches = tuple(caches)
        if not self.caches:
            raise ValueError("a model artifact cache pool requires at least one volume")

    @classmethod
    def configured(cls) -> "ModelArtifactCachePool":
        caches: list[ModelArtifactCache] = []
        failures: list[str] = []
        for index, root in enumerate(configured_model_artifact_cache_roots()):
            # Extra roots come from an explicit directory grant.  If a
            # removable volume is absent, never recreate its mount path on the
            # system disk; simply leave that volume out of this transaction.
            if index > 0 and not root.is_dir():
                failures.append(f"{root}: authorized volume is not mounted")
                continue
            try:
                caches.append(ModelArtifactCache(root))
            except OSError as exc:
                failures.append(f"{root}: {exc}")
        if not caches:
            detail = "; ".join(failures) or "no configured cache root"
            raise OSError(f"none of Fabi's authorized cache volumes is writable: {detail}")
        return cls(caches)

    def reserve(
        self,
        *,
        artifact_identity: str,
        model_id: str,
        immutable_revision: str,
        required_objects: Iterable[CacheObjectRequirement],
    ) -> tuple[ModelArtifactCache, CacheReservation, CacheStorageSnapshot]:
        """Reserve the best exact feasible volume without changing span rank."""

        requirements = tuple(required_objects)
        plans = tuple(
            cache.plan(
                artifact_identity=artifact_identity,
                required_objects=requirements,
            )
            for cache in self.caches
        )
        feasible = sorted(
            (
                (cache, plan)
                for cache, plan in zip(self.caches, plans)
                if plan.can_reserve
            ),
            key=lambda item: (
                item[1].snapshot.required_content_growth_bytes,
                item[1].reclaim_target_bytes,
                -(
                    item[1].snapshot.free_bytes
                    - item[1].snapshot.minimum_free_bytes
                    - item[1].snapshot.reserved_bytes
                ),
                str(item[0].cache_root),
            ),
        )
        raced_failures: list[ModelArtifactStorageError] = []
        for cache, _ in feasible:
            try:
                reservation, snapshot = cache.reserve(
                    artifact_identity=artifact_identity,
                    model_id=model_id,
                    immutable_revision=immutable_revision,
                    required_objects=requirements,
                )
            except ModelArtifactStorageError as exc:
                raced_failures.append(exc)
                continue
            return cache, reservation, snapshot

        if raced_failures:
            raise min(raced_failures, key=lambda error: error.missing_bytes)
        closest = min(
            plans,
            key=lambda plan: (
                plan.missing_bytes,
                plan.snapshot.required_content_growth_bytes,
                str(plan.cache_root),
            ),
        )
        raise ModelArtifactStorageError(
            "selected layer span cannot fit on any authorized Fabi cache volume; "
            f"the closest volume {closest.cache_root} is missing {closest.missing_bytes} bytes",
            artifact_identity=artifact_identity,
            model_id=model_id,
            immutable_revision=immutable_revision,
            snapshot=closest.snapshot,
            missing_bytes=closest.missing_bytes,
        )
