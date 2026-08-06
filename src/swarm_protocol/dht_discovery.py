"""Production discovery adapter backed by Fabi's signed native Kademlia catalogue.

The native layer owns cryptographic verification, TTLs, Kademlia replication and Hivemind-style
membership-set merging.  This adapter owns domain serialization and coherent planner snapshots.
It deliberately embeds the offer, span lease and recent outgoing links in each membership entry,
matching Petals' proven ``ServerInfo`` pattern and avoiding an N+1 lookup per discovered worker.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from swarm_protocol.contracts import (
    LinkMetric,
    ModelManifest,
    ModelMemberAdvertisement,
    SpanLease,
    WorkerOffer,
)
from swarm_protocol.context_placement import ContextCapacityDemandMap
from swarm_protocol.discovery import DiscoveryError, DiscoverySnapshot, InMemoryDiscoveryStore

_CATALOG_TTL_MS = 4 * 60 * 1000
MAX_ADVERTISED_LINKS = 8


class NativeCatalogRecord(Protocol):
    kind: str
    logical_key: str
    publisher_endpoint_id: str
    discovery_peer_id: str
    sequence: int
    issued_at_ms: int
    expires_at_ms: int

    @property
    def payload(self) -> bytes: ...


class NativeCatalogNode(Protocol):
    endpoint_id: str

    def catalog_key(
        self,
        kind: str,
        model_swarm_id: str | None = None,
        target_endpoint_id: str | None = None,
        region_id: str | None = None,
        publisher_endpoint_id: str | None = None,
    ) -> str: ...

    def sign_catalog_record(
        self,
        kind: str,
        logical_key: str,
        discovery_peer_id: str,
        sequence: int,
        issued_at_ms: int,
        expires_at_ms: int,
        payload: bytes,
    ) -> bytes: ...

    def catalog_put(self, logical_key: str, encoded: bytes, quorum: int = 1) -> None: ...

    def catalog_get(self, logical_key: str) -> NativeCatalogRecord: ...

    def catalog_get_model_members(self, model_swarm_id: str) -> list[NativeCatalogRecord]: ...


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _json_payload(
    model: (
        ModelManifest
        | WorkerOffer
        | SpanLease
        | LinkMetric
        | ModelMemberAdvertisement
        | ContextCapacityDemandMap
    ),
) -> bytes:
    return model.model_dump_json().encode("utf-8")


class DhtDiscoveryStore:
    """Synchronous ``DiscoveryStore`` implementation over the native catalogue node.

    One instance represents one stable Iroh endpoint. Publication is serialized so membership
    versions remain strictly monotonic even when heartbeats and link probes finish concurrently.
    Reads query every deterministic model shard concurrently inside Rust and fail on a partial
    shard view instead of silently planning from incomplete coverage.
    """

    def __init__(
        self,
        node: NativeCatalogNode,
        discovery_peer_id: str,
        *,
        clock_ms: Callable[[], int] = _system_clock_ms,
        quorum: int = 1,
        trusted_demand_publishers: dict[str, str] | None = None,
    ) -> None:
        if not discovery_peer_id:
            raise ValueError("discovery_peer_id must not be empty")
        if quorum <= 0:
            raise ValueError("quorum must be positive")
        self._node = node
        self._discovery_peer_id = discovery_peer_id
        self._clock_ms = clock_ms
        self._quorum = quorum
        self._lock = threading.RLock()
        self._shadow = InMemoryDiscoveryStore(clock_ms=clock_ms)
        self._local_offer: WorkerOffer | None = None
        self._local_leases: dict[str, SpanLease] = {}
        self._local_links: dict[tuple[str, str], LinkMetric] = {}
        self._member_sequences: dict[str, int] = {}
        self._known_manifests: dict[str, ModelManifest] = {}
        self._known_offers: dict[str, WorkerOffer] = {}
        self._trusted_demand_publishers = dict(trusted_demand_publishers or {})

    def _now_ms(self) -> int:
        now = int(self._clock_ms())
        if now < 0:
            raise RuntimeError("discovery clock returned a negative timestamp")
        return now

    def _sign_and_put(
        self,
        *,
        kind: str,
        logical_key: str,
        sequence: int,
        issued_at_ms: int,
        expires_at_ms: int,
        payload: bytes,
    ) -> None:
        encoded = self._node.sign_catalog_record(
            kind,
            logical_key,
            self._discovery_peer_id,
            sequence,
            issued_at_ms,
            expires_at_ms,
            payload,
        )
        self._node.catalog_put(logical_key, encoded, self._quorum)

    def publish_manifest(self, manifest: ModelManifest) -> bool:
        with self._lock:
            now = self._now_ms()
            logical_key = self._node.catalog_key("model_manifest", manifest.model_swarm_id)
            self._sign_and_put(
                kind="model_manifest",
                logical_key=logical_key,
                sequence=now,
                issued_at_ms=now,
                expires_at_ms=now + _CATALOG_TTL_MS,
                payload=_json_payload(manifest),
            )
            changed = self._shadow.publish_manifest(manifest)
            self._known_manifests[manifest.model_swarm_id] = manifest
            return changed

    def publish_context_demand(self, demand: ContextCapacityDemandMap) -> None:
        """Publish bounded aggregate demand, never a worker placement command.

        Any endpoint may publish its own namespaced observation. Consumers
        explicitly pin the endpoint trusted for each region, so a third party
        cannot overwrite or impersonate that stream.
        """

        now = self._now_ms()
        if demand.issued_at_ms > now:
            raise DiscoveryError("context demand issue time is in the future")
        if demand.expires_at_ms <= now:
            raise DiscoveryError("cannot publish expired context demand")
        if demand.expires_at_ms - demand.issued_at_ms > _CATALOG_TTL_MS:
            raise DiscoveryError("context demand TTL exceeds the catalogue maximum")
        with self._lock:
            manifest = self._known_manifests.get(demand.model_swarm_id)
            if manifest is not None:
                demand.validate_for(manifest, now_ms=now)
            logical_key = self._node.catalog_key(
                "context_demand",
                demand.model_swarm_id,
                None,
                demand.region_id,
            )
            self._sign_and_put(
                kind="context_demand",
                logical_key=logical_key,
                sequence=demand.issued_at_ms,
                issued_at_ms=demand.issued_at_ms,
                expires_at_ms=demand.expires_at_ms,
                payload=_json_payload(demand),
            )

    def context_demand(
        self,
        manifest: ModelManifest,
        *,
        region_id: str,
        now_ms: int | None = None,
    ) -> ContextCapacityDemandMap | None:
        """Read one trusted region aggregate or return no advice.

        Missing, unreachable, stale or malformed advice is deliberately not a
        placement failure: a READY worker retains its last verified target and
        only a cold worker may use the minimal route bootstrap. A valid record
        must be signed by the endpoint pinned for the region and must exactly
        match the trusted model contract.
        """

        captured_at = self._now_ms() if now_ms is None else now_ms
        if captured_at < 0:
            raise ValueError("now_ms must be non-negative")
        publisher = self._trusted_demand_publishers.get(region_id)
        if publisher is None:
            return None
        logical_key = self._node.catalog_key(
            "context_demand",
            manifest.model_swarm_id,
            None,
            region_id,
            publisher,
        )
        try:
            record = self._node.catalog_get(logical_key)
        except Exception:
            return None
        if (
            record.kind != "context_demand"
            or record.publisher_endpoint_id != publisher
            or record.expires_at_ms <= captured_at
        ):
            return None
        try:
            demand = ContextCapacityDemandMap.model_validate_json(bytes(record.payload))
            demand.validate_for(manifest, now_ms=captured_at)
        except ValueError:
            return None
        if demand.region_id != region_id:
            return None
        return demand

    def publish_offer(self, offer: WorkerOffer) -> bool:
        if offer.endpoint_id != self._node.endpoint_id:
            raise DiscoveryError("worker offer endpoint does not match the native signing endpoint")
        with self._lock:
            logical_key = self._node.catalog_key("worker_offer")
            self._sign_and_put(
                kind="worker_offer",
                logical_key=logical_key,
                sequence=offer.offer_seq,
                issued_at_ms=offer.issued_at_ms,
                expires_at_ms=offer.expires_at_ms,
                payload=_json_payload(offer),
            )
            changed = self._shadow.publish_offer(offer)
            self._local_offer = offer
            self._known_offers[offer.worker_id] = offer
            for lease in self._local_leases.values():
                self._publish_membership_locked(lease)
            return changed

    def publish_span_lease(self, lease: SpanLease) -> bool:
        with self._lock:
            offer = self._local_offer
            if offer is None or offer.worker_id != lease.worker_id:
                raise DiscoveryError("publish a matching local worker offer before its span lease")
            logical_key = self._node.catalog_key("span_lease", lease.model_swarm_id)
            self._sign_and_put(
                kind="span_lease",
                logical_key=logical_key,
                sequence=lease.lease_seq,
                issued_at_ms=lease.issued_at_ms,
                expires_at_ms=lease.expires_at_ms,
                payload=_json_payload(lease),
            )
            self._publish_membership_locked(lease)
            changed = self._shadow.publish_span_lease(lease)
            self._local_leases[lease.model_swarm_id] = lease
            return changed

    def publish_link(self, metric: LinkMetric) -> bool:
        with self._lock:
            offer = self._local_offer
            if offer is None or offer.worker_id != metric.from_worker_id:
                raise DiscoveryError("link source does not match the local worker offer")
            target = self._known_offers.get(metric.to_worker_id)
            if target is None:
                raise DiscoveryError("target worker endpoint is unknown; refresh discovery first")
            logical_key = self._node.catalog_key("link_metric", None, target.endpoint_id)
            self._sign_and_put(
                kind="link_metric",
                logical_key=logical_key,
                sequence=metric.measured_at_ms,
                issued_at_ms=metric.measured_at_ms,
                expires_at_ms=metric.expires_at_ms,
                payload=_json_payload(metric),
            )
            changed = self._shadow.publish_link(metric)
            self._local_links[(metric.from_worker_id, metric.to_worker_id)] = metric
            for lease in self._local_leases.values():
                self._publish_membership_locked(lease)
            return changed

    def publish_advertisement(self, advertisement: ModelMemberAdvertisement) -> None:
        """Publish one coherent worker heartbeat without an N+1 catalogue read.

        The worker signed the complete source-owned advertisement. Target
        workers are deliberately not resolved here; readers retain only links
        whose target also appears in their coherent membership snapshot.
        """

        offer = advertisement.offer
        lease = advertisement.lease
        if offer.endpoint_id != self._node.endpoint_id:
            raise DiscoveryError(
                "worker advertisement endpoint does not match the signing endpoint"
            )
        if offer.worker_id != lease.worker_id:
            raise DiscoveryError("worker advertisement contains inconsistent identities")
        if len(advertisement.outgoing_links) > MAX_ADVERTISED_LINKS:
            raise DiscoveryError("worker advertisement exceeds the link bound")
        with self._lock:
            offer_key = self._node.catalog_key("worker_offer")
            self._sign_and_put(
                kind="worker_offer",
                logical_key=offer_key,
                sequence=offer.offer_seq,
                issued_at_ms=offer.issued_at_ms,
                expires_at_ms=offer.expires_at_ms,
                payload=_json_payload(offer),
            )
            lease_key = self._node.catalog_key("span_lease", lease.model_swarm_id)
            self._sign_and_put(
                kind="span_lease",
                logical_key=lease_key,
                sequence=lease.lease_seq,
                issued_at_ms=lease.issued_at_ms,
                expires_at_ms=lease.expires_at_ms,
                payload=_json_payload(lease),
            )
            self._local_offer = offer
            self._local_leases[lease.model_swarm_id] = lease
            self._local_links = {
                (metric.from_worker_id, metric.to_worker_id): metric
                for metric in advertisement.outgoing_links
            }
            self._known_offers[offer.worker_id] = offer
            self._shadow.publish_offer(offer)
            self._shadow.publish_span_lease(lease)
            self._publish_membership_locked(lease)

    def _publish_membership_locked(self, lease: SpanLease) -> None:
        offer = self._local_offer
        if offer is None:
            raise DiscoveryError("cannot publish model membership without a worker offer")
        now = self._now_ms()
        expires_at_ms = min(offer.expires_at_ms, lease.expires_at_ms)
        if expires_at_ms <= now:
            raise DiscoveryError("cannot publish an expired model membership")
        links = tuple(
            sorted(
                (
                    metric
                    for metric in self._local_links.values()
                    if metric.from_worker_id == offer.worker_id and metric.expires_at_ms > now
                ),
                key=lambda metric: (
                    metric.path_kind.value,
                    metric.loss_rate,
                    metric.rtt_ms,
                    int(metric.throughput_bytes_per_second is None),
                    -(metric.throughput_bytes_per_second or 0.0),
                    metric.to_worker_id,
                ),
            )
        )[:MAX_ADVERTISED_LINKS]
        advertisement = ModelMemberAdvertisement(
            offer=offer,
            lease=lease,
            outgoing_links=links,
        )
        previous = self._member_sequences.get(lease.model_swarm_id, -1)
        sequence = max(now, previous + 1)
        logical_key = self._node.catalog_key("model_member", lease.model_swarm_id)
        self._sign_and_put(
            kind="model_member",
            logical_key=logical_key,
            sequence=sequence,
            issued_at_ms=now,
            expires_at_ms=expires_at_ms,
            payload=_json_payload(advertisement),
        )
        self._member_sequences[lease.model_swarm_id] = sequence

    def _read_manifest(self, model_swarm_id: str) -> ModelManifest:
        logical_key = self._node.catalog_key("model_manifest", model_swarm_id)
        try:
            record = self._node.catalog_get(logical_key)
        except Exception as error:
            raise DiscoveryError(f"model manifest lookup failed for {model_swarm_id}") from error
        if record.kind != "model_manifest":
            raise DiscoveryError("catalogue returned a non-manifest record for a manifest key")
        manifest = ModelManifest.model_validate_json(bytes(record.payload))
        if manifest.model_swarm_id != model_swarm_id:
            raise DiscoveryError("manifest payload hash does not match its catalogue key")
        return manifest

    def snapshot(
        self, *, model_swarm_id: str | None = None, now_ms: int | None = None
    ) -> DiscoverySnapshot:
        captured_at = self._now_ms() if now_ms is None else now_ms
        if captured_at < 0:
            raise ValueError("now_ms must be non-negative")
        if model_swarm_id is None:
            with self._lock:
                model_ids = tuple(sorted(self._known_manifests))
            if not model_ids:
                raise DiscoveryError(
                    "global model enumeration belongs to the registry; request a model-specific snapshot"
                )
        else:
            model_ids = (model_swarm_id,)

        manifests: list[ModelManifest] = []
        advertisements: list[tuple[NativeCatalogRecord, ModelMemberAdvertisement]] = []
        for current_model_id in model_ids:
            manifest = self._read_manifest(current_model_id)
            manifests.append(manifest)
            try:
                records = self._node.catalog_get_model_members(current_model_id)
            except Exception as error:
                raise DiscoveryError(
                    f"model membership lookup failed for {current_model_id}"
                ) from error
            for record in records:
                if record.kind != "model_member" or record.expires_at_ms <= captured_at:
                    continue
                advertisement = ModelMemberAdvertisement.model_validate_json(bytes(record.payload))
                if advertisement.lease.model_swarm_id != current_model_id:
                    continue
                if advertisement.offer.endpoint_id != record.publisher_endpoint_id:
                    continue
                if (
                    advertisement.offer.expires_at_ms <= captured_at
                    or advertisement.lease.expires_at_ms <= captured_at
                ):
                    continue
                advertisements.append((record, advertisement))

        # A malicious endpoint may reuse another logical worker_id. Keep one deterministic newest
        # signed view per worker so route construction cannot contain an ambiguous identity.
        by_worker: dict[str, tuple[NativeCatalogRecord, ModelMemberAdvertisement]] = {}
        for candidate in advertisements:
            worker_id = candidate[1].offer.worker_id
            previous = by_worker.get(worker_id)
            if previous is None or (candidate[0].sequence, candidate[0].publisher_endpoint_id) > (
                previous[0].sequence,
                previous[0].publisher_endpoint_id,
            ):
                by_worker[worker_id] = candidate

        offers = tuple(
            sorted((item[1].offer for item in by_worker.values()), key=lambda x: x.worker_id)
        )
        leases = tuple(
            sorted(
                (item[1].lease for item in by_worker.values()),
                key=lambda x: (
                    x.model_swarm_id,
                    x.hosted_span.start,
                    x.hosted_span.end,
                    x.worker_id,
                ),
            )
        )
        links = tuple(
            sorted(
                (
                    metric
                    for _, advertisement in by_worker.values()
                    for metric in advertisement.outgoing_links
                    if metric.expires_at_ms > captured_at and metric.to_worker_id in by_worker
                ),
                key=lambda x: (x.from_worker_id, x.to_worker_id),
            )
        )
        with self._lock:
            self._known_manifests.update((item.model_swarm_id, item) for item in manifests)
            self._known_offers.update((item.worker_id, item) for item in offers)
        return DiscoverySnapshot(
            captured_at_ms=captured_at,
            manifests=tuple(sorted(manifests, key=lambda x: x.model_swarm_id)),
            offers=offers,
            leases=leases,
            links=links,
        )
