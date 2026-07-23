"""Transport-independent discovery catalogue for Fabi Swarm Protocol v3.

Petals demonstrates that model-serving discovery must be soft state: workers publish short-lived,
monotonically versioned declarations and consumers compose routes from a local snapshot.  This
module defines those domain semantics without pretending that an in-process dictionary is a DHT.
The production rust-libp2p adapter and deterministic simulations both implement this boundary.

The catalogue is deliberately not an admission authority.  Capacity values are observations and
every planned route must still pass worker-local PREPARE/COMMIT before execution.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from swarm_protocol.contracts import (
    LinkMetric,
    ModelManifest,
    SpanLease,
    WorkerOffer,
)


class DiscoveryError(RuntimeError):
    """Base class for catalogue validation failures."""


class StaleCatalogRecord(DiscoveryError):
    """A publisher attempted to roll a logical record back to an older sequence."""


class CatalogRecordConflict(DiscoveryError):
    """One logical version was reused with different immutable contents."""


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(frozen=True)
class DiscoverySnapshot:
    """One coherent, immutable view used by a request-time route planner."""

    captured_at_ms: int
    manifests: tuple[ModelManifest, ...]
    offers: tuple[WorkerOffer, ...]
    leases: tuple[SpanLease, ...]
    links: tuple[LinkMetric, ...]

    def manifest(self, model_swarm_id: str) -> ModelManifest | None:
        return next(
            (item for item in self.manifests if item.model_swarm_id == model_swarm_id), None
        )


@runtime_checkable
class DiscoveryStore(Protocol):
    """Domain port implemented by local simulations and the native DHT adapter."""

    def publish_manifest(self, manifest: ModelManifest) -> bool:
        """Publish immutable model metadata; return whether the local view changed."""

    def publish_offer(self, offer: WorkerOffer) -> bool:
        """Publish the newest soft-state worker offer."""

    def publish_span_lease(self, lease: SpanLease) -> bool:
        """Publish the newest soft-state span lease for one worker/model pair."""

    def publish_link(self, metric: LinkMetric) -> bool:
        """Publish the newest directed link observation."""

    def snapshot(
        self, *, model_swarm_id: str | None = None, now_ms: int | None = None
    ) -> DiscoverySnapshot:
        """Read a deterministic snapshot containing only live, coherent soft state."""


class InMemoryDiscoveryStore:
    """Thread-safe reference implementation of the discovery semantics.

    This implementation is intentionally useful for unit tests, simulations, and shadow mode only.
    It keeps sequence watermarks after expired payloads are collected so delayed network messages
    cannot resurrect an older declaration during the process lifetime.
    """

    def __init__(self, *, clock_ms: Callable[[], int] = _system_clock_ms) -> None:
        self._clock_ms = clock_ms
        self._manifests: dict[str, ModelManifest] = {}
        self._offers: dict[str, WorkerOffer] = {}
        self._offer_watermarks: dict[str, int] = {}
        self._leases: dict[tuple[str, str], SpanLease] = {}
        self._lease_watermarks: dict[tuple[str, str], int] = {}
        self._links: dict[tuple[str, str], LinkMetric] = {}
        self._lock = threading.RLock()

    def _now_ms(self) -> int:
        now = int(self._clock_ms())
        if now < 0:
            raise RuntimeError("discovery clock returned a negative timestamp")
        return now

    @staticmethod
    def _publish_sequenced(
        *,
        logical_name: str,
        key: object,
        sequence: int,
        value: WorkerOffer | SpanLease,
        records: dict,
        watermarks: dict,
    ) -> bool:
        watermark = watermarks.get(key)
        existing = records.get(key)
        if watermark is not None and sequence < watermark:
            raise StaleCatalogRecord(
                f"{logical_name} sequence {sequence} is older than watermark {watermark}"
            )
        if watermark is not None and sequence == watermark:
            if existing is not None and existing == value:
                return False
            raise CatalogRecordConflict(
                f"{logical_name} sequence {sequence} was reused with different contents"
            )
        watermarks[key] = sequence
        records[key] = value
        return True

    def publish_manifest(self, manifest: ModelManifest) -> bool:
        key = manifest.model_swarm_id
        with self._lock:
            existing = self._manifests.get(key)
            if existing is None:
                self._manifests[key] = manifest
                return True
            if existing == manifest:
                return False
            # This should be cryptographically unreachable unless model_swarm_id generation or
            # validation is broken, so fail closed rather than choosing an arbitrary value.
            raise CatalogRecordConflict(f"model swarm id collision for {key}")

    def publish_offer(self, offer: WorkerOffer) -> bool:
        with self._lock:
            return self._publish_sequenced(
                logical_name=f"worker offer {offer.worker_id}",
                key=offer.worker_id,
                sequence=offer.offer_seq,
                value=offer,
                records=self._offers,
                watermarks=self._offer_watermarks,
            )

    def publish_span_lease(self, lease: SpanLease) -> bool:
        key = (lease.model_swarm_id, lease.worker_id)
        with self._lock:
            return self._publish_sequenced(
                logical_name=f"span lease {lease.model_swarm_id}/{lease.worker_id}",
                key=key,
                sequence=lease.lease_seq,
                value=lease,
                records=self._leases,
                watermarks=self._lease_watermarks,
            )

    def publish_link(self, metric: LinkMetric) -> bool:
        key = (metric.from_worker_id, metric.to_worker_id)
        with self._lock:
            existing = self._links.get(key)
            if existing is not None and metric.measured_at_ms < existing.measured_at_ms:
                raise StaleCatalogRecord(
                    f"link {key[0]}->{key[1]} measurement {metric.measured_at_ms} is older "
                    f"than {existing.measured_at_ms}"
                )
            if existing is not None and metric.measured_at_ms == existing.measured_at_ms:
                if existing == metric:
                    return False
                raise CatalogRecordConflict(
                    f"link {key[0]}->{key[1]} measurement timestamp was reused"
                )
            self._links[key] = metric
            return True

    def collect_expired(self, *, now_ms: int | None = None) -> int:
        """Drop expired payloads while retaining their anti-rollback watermarks."""

        captured_at = self._now_ms() if now_ms is None else now_ms
        if captured_at < 0:
            raise ValueError("now_ms must be non-negative")
        removed = 0
        with self._lock:
            for key, offer in tuple(self._offers.items()):
                if offer.expires_at_ms <= captured_at:
                    del self._offers[key]
                    removed += 1
            for key, lease in tuple(self._leases.items()):
                if lease.expires_at_ms <= captured_at:
                    del self._leases[key]
                    removed += 1
            for key, metric in tuple(self._links.items()):
                if metric.expires_at_ms <= captured_at:
                    del self._links[key]
                    removed += 1
        return removed

    def snapshot(
        self, *, model_swarm_id: str | None = None, now_ms: int | None = None
    ) -> DiscoverySnapshot:
        captured_at = self._now_ms() if now_ms is None else now_ms
        if captured_at < 0:
            raise ValueError("now_ms must be non-negative")

        with self._lock:
            manifests = tuple(
                sorted(
                    (
                        manifest
                        for key, manifest in self._manifests.items()
                        if model_swarm_id is None or key == model_swarm_id
                    ),
                    key=lambda item: item.model_swarm_id,
                )
            )
            offers = tuple(
                sorted(
                    (
                        offer
                        for offer in self._offers.values()
                        if offer.expires_at_ms > captured_at
                    ),
                    key=lambda item: item.worker_id,
                )
            )
            live_worker_ids = {offer.worker_id for offer in offers}
            leases = tuple(
                sorted(
                    (
                        lease
                        for lease in self._leases.values()
                        if lease.expires_at_ms > captured_at
                        and lease.worker_id in live_worker_ids
                        and (
                            model_swarm_id is None
                            or lease.model_swarm_id == model_swarm_id
                        )
                    ),
                    key=lambda item: (
                        item.model_swarm_id,
                        item.hosted_span.start,
                        item.hosted_span.end,
                        item.worker_id,
                    ),
                )
            )
            links = tuple(
                sorted(
                    (
                        metric
                        for metric in self._links.values()
                        if metric.expires_at_ms > captured_at
                        and metric.from_worker_id in live_worker_ids
                        and metric.to_worker_id in live_worker_ids
                    ),
                    key=lambda item: (item.from_worker_id, item.to_worker_id),
                )
            )

        return DiscoverySnapshot(
            captured_at_ms=captured_at,
            manifests=manifests,
            offers=offers,
            leases=leases,
            links=links,
        )
