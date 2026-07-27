"""Read-only v3 route comparison beside the qualified Parallax scheduler."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from scheduling.node import node_is_routable
from swarm_protocol.contracts import (
    ModelManifest,
    ModelMemberAdvertisement,
    RecoveryLevel,
    RequestContract,
    SpanState,
)
from swarm_protocol.discovery import DiscoverySnapshot
from swarm_protocol.registry import ModelRegistryBundle, TrustedModelRegistry
from swarm_protocol.routing import ExactRoutePlanner, NoFeasibleRoute

logger = logging.getLogger(__name__)


class CatalogStore(Protocol):
    def publish_manifest(self, manifest: ModelManifest) -> bool: ...

    def snapshot(
        self,
        *,
        model_swarm_id: str | None = None,
        now_ms: int | None = None,
    ) -> DiscoverySnapshot: ...


class SchedulerProtocolV3Shadow:
    """Validate worker reports and expose the shared v3 route planner."""

    def __init__(self, registry: TrustedModelRegistry, *, mode: str = "shadow") -> None:
        if mode not in {"shadow", "active"}:
            raise ValueError("scheduler protocol-v3 mode must be 'shadow' or 'active'")
        self.registry = registry
        self.mode = mode
        self.planner = ExactRoutePlanner()
        self._lock = threading.RLock()
        self._bundles: dict[str, ModelRegistryBundle] = {}
        self._pending: set[str] = set()
        self._bundle_errors: dict[str, dict[str, str]] = {}
        self._bundle_retry_after: dict[str, float] = {}
        self._latest: dict[str, object] = {"mode": mode, "state": "waiting_workers"}
        self._last_log_key = ""
        self._catalog: CatalogStore | None = None
        self._catalog_status: dict[str, object] = {"state": "off"}
        self._catalog_snapshots: dict[str, DiscoverySnapshot] = {}
        self._catalog_reads_pending: set[str] = set()
        self._catalog_read_retry_after: dict[str, float] = {}
        self._catalog_manifest_pending: set[str] = set()
        self._catalog_manifest_publish_after: dict[str, float] = {}

    def attach_catalog(self, catalog: CatalogStore) -> None:
        with self._lock:
            self._catalog = catalog
            bundles = tuple(self._bundles.values())
            self._catalog_status = {"state": "waiting_manifest"}
        for bundle in bundles:
            self._publish_manifest_async(bundle)

    @classmethod
    def from_environment(cls) -> "SchedulerProtocolV3Shadow | None":
        mode = os.environ.get("FABI_SWARM_V3_MODE", "off").strip().lower()
        if mode in {"", "off", "disabled"}:
            return None
        if mode not in {"shadow", "active"}:
            raise ValueError("FABI_SWARM_V3_MODE supports only 'off', 'shadow', or 'active'")
        metadata_url = os.environ.get("FABI_MODEL_REGISTRY_METADATA_URL")
        targets_url = os.environ.get("FABI_MODEL_REGISTRY_TARGETS_URL")
        root_path = os.environ.get("FABI_MODEL_REGISTRY_ROOT")
        if not metadata_url or not targets_url or not root_path:
            raise ValueError("scheduler shadow mode requires registry URLs and a pinned root path")
        state_dir = Path(
            os.environ.get(
                "FABI_SWARM_V3_SCHEDULER_STATE_DIR",
                str(Path.home() / ".fabi" / "swarm-v3" / "scheduler-registry"),
            )
        )
        return cls(
            TrustedModelRegistry(
                state_dir,
                metadata_base_url=metadata_url,
                target_base_url=targets_url,
                bootstrap_root=Path(root_path).read_bytes(),
            ),
            mode=mode,
        )

    @property
    def latest(self) -> dict[str, object]:
        with self._lock:
            return dict(self._latest)

    def observe(
        self,
        nodes: list[Any],
        *,
        model_num_layers: int,
        planning_context_tokens: int,
        epoch: int,
    ) -> dict[str, object]:
        """Perform a CPU-only comparison and schedule registry I/O off-thread."""

        now_ms = time.time_ns() // 1_000_000
        advertisements, rejected_workers = self._advertisements(nodes)

        if not advertisements:
            return self._store(
                {
                    "mode": self.mode,
                    "state": "waiting_workers",
                    "accepted_workers": 0,
                    "rejected_workers": rejected_workers,
                }
            )

        swarm_counts: dict[str, int] = {}
        for advertisement in advertisements:
            swarm_id = advertisement.lease.model_swarm_id
            swarm_counts[swarm_id] = swarm_counts.get(swarm_id, 0) + 1
        model_swarm_id = min(
            swarm_counts,
            key=lambda swarm_id: (-swarm_counts[swarm_id], swarm_id),
        )
        advertisements = [
            advertisement
            for advertisement in advertisements
            if advertisement.lease.model_swarm_id == model_swarm_id
        ]

        with self._lock:
            bundle = self._bundles.get(model_swarm_id)
            bundle_error = self._bundle_errors.get(model_swarm_id)
            retry_allowed = time.monotonic() >= self._bundle_retry_after.get(model_swarm_id, 0)
            if bundle is None and model_swarm_id not in self._pending and retry_allowed:
                self._pending.add(model_swarm_id)
                threading.Thread(
                    target=self._fetch_bundle,
                    args=(model_swarm_id,),
                    name="SwarmV3ShadowRegistry",
                    daemon=True,
                ).start()
        if bundle is None:
            return self._store(
                {
                    "mode": self.mode,
                    "state": "registry_rejected" if bundle_error else "verifying_registry",
                    "model_swarm_id": model_swarm_id,
                    "accepted_workers": len(advertisements),
                    "registry_error": bundle_error,
                    "rejected_workers": rejected_workers,
                }
            )

        self._publish_manifest_async(bundle)
        self._refresh_catalog_async(model_swarm_id)

        prompt_tokens = max(1, planning_context_tokens - min(4096, planning_context_tokens - 1))
        output_tokens = planning_context_tokens - prompt_tokens
        request = RequestContract(
            request_id=f"shadow-{now_ms}",
            model_swarm_id=model_swarm_id,
            prompt_tokens=prompt_tokens,
            reserved_output_tokens=output_tokens,
            recovery_level=RecoveryLevel.RESTARTABLE,
        )
        offers = tuple(advertisement.offer for advertisement in advertisements)
        leases = tuple(advertisement.lease for advertisement in advertisements)
        links = tuple(
            link for advertisement in advertisements for link in advertisement.outgoing_links
        )
        blockers = []
        if len(leases) > 1 and not links:
            blockers.append("missing_link_goodput")
        try:
            planned = self.planner.plan(
                manifest=bundle.manifest,
                request=request,
                offers=offers,
                leases=leases,
                links=links,
                snapshot_time_ms=now_ms,
                coordinator_id="qualified-scheduler-shadow",
                reservation_deadline_ms=now_ms + 5_000,
                plan_expires_at_ms=now_ms + 10_000,
                epoch=epoch,
            )
        except NoFeasibleRoute as exc:
            return self._store(
                {
                    "mode": self.mode,
                    "state": "no_feasible_route",
                    "model_swarm_id": model_swarm_id,
                    "required_context_tokens": planning_context_tokens,
                    "accepted_workers": len(advertisements),
                    "blockers": blockers or [str(exc)],
                    "rejected_workers": rejected_workers,
                }
            )

        v3_route = tuple(stage.worker_id for stage in planned.plan.stages)
        legacy_routes = self._legacy_complete_routes(nodes, model_num_layers)
        agrees = v3_route in legacy_routes
        return self._store(
            {
                "mode": self.mode,
                "state": "agreement" if agrees else "divergence",
                "model_swarm_id": model_swarm_id,
                "required_context_tokens": planning_context_tokens,
                "v3_route": v3_route,
                "legacy_routes": legacy_routes,
                "projected_ttft_ms": (
                    planned.estimate.ttft_ms if planned.estimate.complete else None
                ),
                "projected_inter_token_ms": (
                    planned.estimate.inter_token_ms if planned.estimate.complete else None
                ),
                "performance_telemetry_complete": planned.estimate.complete,
                "rejected_workers": rejected_workers,
            }
        )

    @staticmethod
    def _advertisements(
        nodes: list[Any],
    ) -> tuple[list[ModelMemberAdvertisement], dict[str, str]]:
        advertisements = []
        rejected_workers: dict[str, str] = {}
        for node in nodes:
            report = getattr(node, "swarm_v3", None)
            if not node_is_routable(node):
                rejected_workers[str(node.node_id)] = (
                    "liveness_suspect"
                    if getattr(node, "liveness_state", "healthy") != "healthy"
                    else "scheduler_transition"
                )
                continue
            if not isinstance(report, dict) or report.get("state") not in {"ready", "warming"}:
                if isinstance(report, dict):
                    rejected_workers[str(node.node_id)] = str(report.get("state", "invalid"))
                continue
            try:
                advertisement = ModelMemberAdvertisement.model_validate(report.get("advertisement"))
            except Exception as exc:  # noqa: BLE001 - untrusted RPC boundary
                rejected_workers[str(node.node_id)] = f"invalid:{type(exc).__name__}"
                continue
            if advertisement.offer.worker_id != str(node.node_id):
                rejected_workers[str(node.node_id)] = "worker_identity_mismatch"
                continue
            advertisements.append(advertisement)
        return advertisements, rejected_workers

    def plan_request(
        self,
        nodes: list[Any],
        *,
        request: RequestContract,
        coordinator_id: str,
        epoch: int,
        reservation_deadline_ms: int,
        plan_expires_at_ms: int,
    ):
        """Plan from the same verified snapshot used by comparison mode."""

        advertisements, rejected_workers = self._advertisements(nodes)
        matching = [
            advertisement
            for advertisement in advertisements
            if advertisement.lease.model_swarm_id == request.model_swarm_id
        ]
        with self._lock:
            bundle = self._bundles.get(request.model_swarm_id)
            catalog = self._catalog
            catalog_snapshot = self._catalog_snapshots.get(request.model_swarm_id)
        if bundle is None:
            raise NoFeasibleRoute("trusted model bundle is not ready")
        if catalog is not None:
            self._refresh_catalog_async(request.model_swarm_id)
            if catalog_snapshot is None:
                raise NoFeasibleRoute("coherent DHT membership snapshot is not ready")
            if catalog_snapshot.manifest(request.model_swarm_id) != bundle.manifest:
                raise NoFeasibleRoute("DHT manifest does not match the trusted registry bundle")
            # The coherent DHT snapshot is the v3 membership authority.  Do not
            # intersect it with the legacy scheduler's local node table: doing
            # so would make an autonomous route depend on the central v2 view
            # and break multi-routing-node operation.
            offers = catalog_snapshot.offers
            leases = catalog_snapshot.leases
            links = catalog_snapshot.links
        else:
            if not matching:
                raise NoFeasibleRoute(
                    f"no verified workers for model swarm {request.model_swarm_id}; "
                    f"rejected={rejected_workers}"
                )
            offers = tuple(item.offer for item in matching)
            leases = tuple(item.lease for item in matching)
            links = tuple(link for item in matching for link in item.outgoing_links)
        return self.planner.plan(
            manifest=bundle.manifest,
            request=request,
            offers=offers,
            leases=leases,
            links=links,
            snapshot_time_ms=time.time_ns() // 1_000_000,
            coordinator_id=coordinator_id,
            reservation_deadline_ms=reservation_deadline_ms,
            plan_expires_at_ms=plan_expires_at_ms,
            epoch=epoch,
        )

    def _refresh_catalog_async(self, model_swarm_id: str) -> None:
        with self._lock:
            if self._catalog is None:
                return
            if model_swarm_id in self._catalog_reads_pending:
                return
            if time.monotonic() < self._catalog_read_retry_after.get(model_swarm_id, 0):
                return
            self._catalog_reads_pending.add(model_swarm_id)
        threading.Thread(
            target=self._refresh_catalog,
            args=(model_swarm_id,),
            name="SwarmV3CatalogSnapshot",
            daemon=True,
        ).start()

    def _refresh_catalog(self, model_swarm_id: str) -> None:
        with self._lock:
            catalog = self._catalog
        if catalog is None:
            return
        try:
            snapshot = catalog.snapshot(model_swarm_id=model_swarm_id)
        except Exception as exc:  # noqa: BLE001 - asynchronous network status boundary
            with self._lock:
                self._catalog_status = {
                    "state": "read_error",
                    "error": {
                        "code": type(exc).__name__,
                        "detail": str(exc)[:256],
                    },
                }
                self._catalog_read_retry_after[model_swarm_id] = time.monotonic() + 2
                self._catalog_reads_pending.discard(model_swarm_id)
            return
        with self._lock:
            self._catalog_snapshots[model_swarm_id] = snapshot
            self._catalog_status = {
                "state": "snapshot_ready",
                "model_swarm_id": model_swarm_id,
                "captured_at_ms": snapshot.captured_at_ms,
                "workers": len(snapshot.offers),
            }
            self._catalog_read_retry_after[model_swarm_id] = time.monotonic() + 2
            self._catalog_reads_pending.discard(model_swarm_id)

    def ready_model_swarm_id(self, nodes: list[Any]) -> str:
        """Return the dominant verified swarm whose trusted bundle is ready."""

        with self._lock:
            if self._catalog is not None:
                ready = {
                    swarm_id: len(snapshot.leases)
                    for swarm_id, snapshot in self._catalog_snapshots.items()
                    if swarm_id in self._bundles
                    and snapshot.manifest(swarm_id) == self._bundles[swarm_id].manifest
                    and snapshot.leases
                }
                if not ready:
                    raise NoFeasibleRoute(
                        "no DHT model swarm has both trusted metadata and live members"
                    )
                return min(ready, key=lambda swarm_id: (-ready[swarm_id], swarm_id))

        advertisements, _ = self._advertisements(nodes)
        counts: dict[str, int] = {}
        for advertisement in advertisements:
            swarm_id = advertisement.lease.model_swarm_id
            counts[swarm_id] = counts.get(swarm_id, 0) + 1
        with self._lock:
            ready = {
                swarm_id: count for swarm_id, count in counts.items() if swarm_id in self._bundles
            }
        if not ready:
            raise NoFeasibleRoute("no verified model swarm has a trusted registry bundle")
        return min(ready, key=lambda swarm_id: (-ready[swarm_id], swarm_id))

    def trusted_manifest(self, model_swarm_id: str) -> ModelManifest:
        """Return the immutable registry manifest used for route admission."""

        with self._lock:
            bundle = self._bundles.get(str(model_swarm_id))
            if bundle is None:
                raise NoFeasibleRoute("trusted model bundle is not ready")
            return bundle.manifest

    def live_worker_ids(self) -> frozenset[str] | None:
        """Return DHT liveness when the catalogue is authoritative.

        ``None`` means that discovery is still using the qualified scheduler
        membership and callers must retain their legacy liveness source.
        """

        with self._lock:
            if self._catalog is None:
                return None
            return frozenset(
                offer.worker_id
                for snapshot in self._catalog_snapshots.values()
                for offer in snapshot.offers
            )

    def ready_worker_ids(self) -> frozenset[str] | None:
        """Return workers with both an unexpired offer and a READY DHT lease."""

        with self._lock:
            if self._catalog is None:
                return None
            ready: set[str] = set()
            for snapshot in self._catalog_snapshots.values():
                offered = {offer.worker_id for offer in snapshot.offers}
                ready.update(
                    lease.worker_id
                    for lease in snapshot.leases
                    if lease.state is SpanState.READY
                    and lease.worker_id in offered
                )
            return frozenset(ready)

    def _fetch_bundle(self, model_swarm_id: str) -> None:
        try:
            bundle = self.registry.fetch(model_swarm_id)
        except Exception as exc:  # noqa: BLE001 - converted to shadow status
            with self._lock:
                self._bundle_errors[model_swarm_id] = {
                    "code": type(exc).__name__,
                    "detail": str(exc)[:256],
                }
                self._bundle_retry_after[model_swarm_id] = time.monotonic() + 30.0
                self._pending.discard(model_swarm_id)
            return
        with self._lock:
            self._bundles[model_swarm_id] = bundle
            self._bundle_errors.pop(model_swarm_id, None)
            self._bundle_retry_after.pop(model_swarm_id, None)
            self._pending.discard(model_swarm_id)
        self._publish_manifest_async(bundle)

    def _publish_manifest_async(self, bundle: ModelRegistryBundle) -> None:
        with self._lock:
            if self._catalog is None:
                return
            swarm_id = bundle.model_swarm_id
            if swarm_id in self._catalog_manifest_pending:
                return
            if time.monotonic() < self._catalog_manifest_publish_after.get(swarm_id, 0):
                return
            self._catalog_manifest_pending.add(swarm_id)
        threading.Thread(
            target=self._publish_manifest,
            args=(bundle,),
            name="SwarmV3ManifestPublisher",
            daemon=True,
        ).start()

    def _publish_manifest(self, bundle: ModelRegistryBundle) -> None:
        with self._lock:
            catalog = self._catalog
        if catalog is None:
            return
        try:
            catalog.publish_manifest(bundle.manifest)
        except Exception as exc:  # noqa: BLE001 - asynchronous network status boundary
            status: dict[str, object] = {
                "state": "error",
                "error": {
                    "code": type(exc).__name__,
                    "detail": str(exc)[:256],
                },
            }
        else:
            status = {
                "state": "published",
                "model_swarm_id": bundle.model_swarm_id,
            }
        with self._lock:
            self._catalog_status = status
            self._catalog_manifest_pending.discard(bundle.model_swarm_id)
            self._catalog_manifest_publish_after[bundle.model_swarm_id] = time.monotonic() + (
                120 if status["state"] == "published" else 5
            )

    @staticmethod
    def _legacy_complete_routes(nodes: list[Any], num_layers: int) -> tuple[tuple[str, ...], ...]:
        by_start: dict[int, list[Any]] = {}
        for node in nodes:
            start = getattr(node, "start_layer", None)
            end = getattr(node, "end_layer", None)
            if not node_is_routable(node) or start is None or end is None:
                continue
            by_start.setdefault(int(start), []).append(node)

        routes: list[tuple[str, ...]] = []

        def walk(position: int, path: tuple[str, ...]) -> None:
            if position == num_layers:
                routes.append(path)
                return
            for node in sorted(by_start.get(position, []), key=lambda item: str(item.node_id)):
                end = int(node.end_layer)
                if end > position and str(node.node_id) not in path:
                    walk(end, (*path, str(node.node_id)))

        walk(0, ())
        return tuple(sorted(routes))

    def _store(self, value: dict[str, object]) -> dict[str, object]:
        with self._lock:
            value = {**value, "catalog": dict(self._catalog_status)}
            log_key = repr(value)
            self._latest = value
            if log_key != self._last_log_key:
                logger.info("Protocol-v3 %s planner status: %s", self.mode, value)
                self._last_log_key = log_key
            return dict(value)
