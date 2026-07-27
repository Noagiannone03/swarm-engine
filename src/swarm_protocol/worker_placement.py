"""Non-blocking worker-side orchestration for autonomous v3 placement."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from swarm_protocol.contracts import (
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    LinkMetric,
    ModelManifest,
    ModelMemberAdvertisement,
    ReservationState,
    SpanLease,
    SpanState,
    WorkerOffer,
)
from swarm_protocol.discovery import DiscoverySnapshot
from swarm_protocol.execution import WorkerExecutionAdmission
from swarm_protocol.placement import (
    AutonomousPlacementPolicy,
    CapacityDemandMap,
    MaterializationPhase,
    PlacementAction,
    PlacementMaterializer,
)


class PlacementCatalog(Protocol):
    def snapshot(
        self,
        *,
        model_swarm_id: str | None = None,
        now_ms: int | None = None,
    ) -> DiscoverySnapshot: ...


class SpanStatePublisher(Protocol):
    def publish_bootstrap_state(
        self,
        advertisement: ModelMemberAdvertisement,
    ) -> None: ...

    def publish_span_state(
        self,
        advertisement: ModelMemberAdvertisement,
        state: SpanState,
    ) -> ModelMemberAdvertisement: ...


def autonomous_context_tiers(
    preferred_tokens: int,
    *,
    minimum_tokens: int = 4_096,
) -> tuple[int, ...]:
    """Return deterministic per-worker context tiers from preferred to minimum.

    Context is a property of a complete v3 route, not a swarm-wide constant.
    A constrained contributor may therefore materialize a smaller-context
    shard without downgrading unrelated routes.  Halving keeps the number of
    exact placement attempts bounded while the final configured minimum
    remains reachable even when it is not a power-of-two boundary.
    """

    if preferred_tokens <= 0 or minimum_tokens <= 0:
        raise ValueError("autonomous context tiers must be positive")
    floor = min(preferred_tokens, minimum_tokens)
    tiers: list[int] = []
    candidate = preferred_tokens
    while candidate > floor:
        tiers.append(candidate)
        candidate = max(floor, candidate // 2)
    tiers.append(floor)
    return tuple(dict.fromkeys(tiers))


class AutonomousWorkerPlacement:
    """Compose DHT policy, admission drain and the existing executor reload path."""

    def __init__(
        self,
        *,
        catalog: PlacementCatalog,
        admission: WorkerExecutionAdmission,
        state_publisher: SpanStatePublisher,
        reload_target: Callable[[LayerSpan, int], None],
        current_span: LayerSpan | None = None,
        policy: AutonomousPlacementPolicy | None = None,
    ) -> None:
        self._catalog = catalog
        self._admission = admission
        self._state_publisher = state_publisher
        self._policy = policy or AutonomousPlacementPolicy()
        self._materializer = PlacementMaterializer(
            drain=admission,
            reload_target=reload_target,
            current_span=current_span,
        )
        self._snapshot: DiscoverySnapshot | None = None
        self._snapshot_model_id: str | None = None
        self._read_pending = False
        self._retry_after = 0.0
        self._last_moved_at_ms: int | None = None
        self._announced_transition: tuple[object, ...] | None = None
        self._error: dict[str, str] | None = None
        self._context_tokens: int | None = None
        self._lock = threading.RLock()

    def bootstrap(
        self,
        *,
        offer: WorkerOffer,
        manifest: ModelManifest,
        context_tokens: int,
        kv_block_size: int,
        max_sessions: int,
        weight_hashes: tuple[str, ...],
        outgoing_links: tuple[LinkMetric, ...] = (),
    ) -> dict[str, object]:
        """Choose and announce a cold span before starting its executor.

        This follows Petals' JOINING lifecycle: an eventually-consistent intent
        is visible before expensive materialization starts, while route planning
        continues to admit READY leases only.
        """

        if context_tokens <= 0 or kv_block_size <= 0 or max_sessions <= 0:
            raise ValueError("bootstrap context, KV block size and sessions must be positive")
        if not weight_hashes:
            raise ValueError("bootstrap intent must bind signed weight identities")
        with self._lock:
            self._context_tokens = context_tokens

        state = self._materializer.snapshot()
        if state.phase is not MaterializationPhase.STANDBY:
            return self._status(state, decision="materializing", error=None)

        self._refresh_async(manifest.model_swarm_id)
        with self._lock:
            snapshot = (
                self._snapshot
                if self._snapshot_model_id == manifest.model_swarm_id
                else None
            )
            read_error = self._error
        if snapshot is None:
            return self._status(state, decision="waiting_catalog", error=read_error)
        if snapshot.manifest(manifest.model_swarm_id) != manifest:
            return self._status(
                state,
                decision="catalog_manifest_mismatch",
                error={"code": "TrustError", "detail": "DHT manifest differs from registry"},
            )

        decision = self._policy.choose(
            offer=offer,
            manifest=manifest,
            leases=snapshot.leases,
            demand=CapacityDemandMap.uniform(manifest.num_layers, desired_replicas=2),
            context_tokens=context_tokens,
            kv_block_size=kv_block_size,
            current_span=None,
            current_reservations=0,
            now_ms=time.time_ns() // 1_000_000,
        )
        if decision.action is PlacementAction.STANDBY:
            return self._status(state, decision=decision.reason, error=read_error)
        if decision.action is not PlacementAction.JOIN or decision.span is None:
            raise RuntimeError("cold placement policy returned an invalid transition")

        span = decision.span
        rounded_tokens = (
            (context_tokens + kv_block_size - 1) // kv_block_size * kv_block_size
        )
        allocatable_kv_bytes = rounded_tokens * sum(
            manifest.kv_bytes_per_token_by_layer[span.start : span.end]
        )
        now_ms = time.time_ns() // 1_000_000
        building = ModelMemberAdvertisement(
            offer=offer,
            lease=SpanLease(
                model_swarm_id=manifest.model_swarm_id,
                worker_id=offer.worker_id,
                hosted_span=span,
                effective_span_mode=EffectiveSpanMode.FIXED,
                state=SpanState.BUILDING,
                weight_hashes=weight_hashes,
                kv_geometry=KvGeometry(
                    block_size_tokens=kv_block_size,
                    bytes_per_token_by_layer=manifest.kv_bytes_per_token_by_layer,
                    allocatable_bytes=allocatable_kv_bytes,
                ),
                available_kv_bytes_snapshot=0,
                max_sessions=max_sessions,
                lease_seq=0,
                issued_at_ms=now_ms,
                expires_at_ms=now_ms + 45_000,
            ),
            outgoing_links=outgoing_links,
        )
        # Publish intent before loading so simultaneous cold joins can spread
        # across deficits instead of stampeding the same layers.
        self._state_publisher.publish_bootstrap_state(building)
        state = self._materializer.reconcile(decision)
        self._announced_transition = (state.phase, state.generation, state.target_span)
        return self._status(state, decision=decision.reason, error=read_error)

    def observe(
        self,
        *,
        advertisement: ModelMemberAdvertisement,
        manifest: ModelManifest,
        context_tokens: int,
    ) -> dict[str, object]:
        """Advance placement immediately; DHT reads always run off-thread."""

        if advertisement.lease.model_swarm_id != manifest.model_swarm_id:
            raise ValueError("placement advertisement and trusted manifest disagree")
        if context_tokens <= 0:
            raise ValueError("placement context contract must be positive")
        with self._lock:
            self._context_tokens = context_tokens

        state = self._materializer.snapshot()
        if (
            state.phase is MaterializationPhase.BUILDING
            and advertisement.lease.state is SpanState.READY
            and advertisement.lease.hosted_span == state.target_span
        ):
            self._admission.configure(advertisement)
            state = self._materializer.mark_ready(
                span=advertisement.lease.hosted_span,
                generation=state.generation,
            )
            self._last_moved_at_ms = time.time_ns() // 1_000_000
            self._announced_transition = None
        elif state.phase is MaterializationPhase.DRAINING:
            state = self._materializer.continue_after_drain()
        elif state.phase is MaterializationPhase.READY:
            self._admission.configure(advertisement)

        self._refresh_async(manifest.model_swarm_id)
        with self._lock:
            snapshot = (
                self._snapshot
                if self._snapshot_model_id == manifest.model_swarm_id
                else None
            )
            read_error = self._error

        state = self._materializer.snapshot()
        if state.phase in {MaterializationPhase.DRAINING, MaterializationPhase.BUILDING}:
            self._announce_transition_once(advertisement, state)
            return self._status(state, decision="materializing", error=read_error)
        if advertisement.lease.state is not SpanState.READY:
            return self._status(state, decision="waiting_executor_ready", error=read_error)
        if snapshot is None:
            return self._status(state, decision="waiting_catalog", error=read_error)
        if snapshot.manifest(manifest.model_swarm_id) != manifest:
            return self._status(
                state,
                decision="catalog_manifest_mismatch",
                error={"code": "TrustError", "detail": "DHT manifest differs from registry"},
            )

        active_reservations = sum(
            lease.state in {ReservationState.PREPARED, ReservationState.COMMITTED}
            for lease in self._admission.snapshot()
        )
        decision = self._policy.choose(
            offer=advertisement.offer,
            manifest=manifest,
            leases=snapshot.leases,
            demand=CapacityDemandMap.uniform(manifest.num_layers, desired_replicas=2),
            context_tokens=context_tokens,
            kv_block_size=advertisement.lease.kv_geometry.block_size_tokens,
            current_span=state.current_span,
            current_reservations=active_reservations,
            last_moved_at_ms=self._last_moved_at_ms,
            now_ms=time.time_ns() // 1_000_000,
        )
        if decision.action in {PlacementAction.JOIN, PlacementAction.MOVE}:
            state = self._materializer.reconcile(decision)
            self._announce_transition_once(advertisement, state)
        return self._status(state, decision=decision.reason, error=read_error)

    def mark_failed(self, *, generation: int, error: Exception) -> dict[str, object]:
        """Fence a backend failure and start the materializer's safe rollback."""

        state = self._materializer.mark_failed(generation=generation, error=error)
        return self._status(
            state,
            decision="rolling_back_previous_span",
            error={"code": type(error).__name__, "detail": str(error)[:256]},
        )

    def _announce_transition_once(
        self,
        advertisement: ModelMemberAdvertisement,
        state,
    ) -> None:
        key = (state.phase, state.generation, state.target_span)
        if self._announced_transition == key:
            return
        self._state_publisher.publish_span_state(advertisement, SpanState.DRAINING)
        self._announced_transition = key

    def _refresh_async(self, model_swarm_id: str) -> None:
        with self._lock:
            if self._read_pending or time.monotonic() < self._retry_after:
                return
            self._read_pending = True
        threading.Thread(
            target=self._refresh,
            args=(model_swarm_id,),
            name="SwarmV3WorkerPlacement",
            daemon=True,
        ).start()

    def _refresh(self, model_swarm_id: str) -> None:
        try:
            snapshot = self._catalog.snapshot(model_swarm_id=model_swarm_id)
        except Exception as exc:  # noqa: BLE001 - asynchronous discovery boundary
            with self._lock:
                self._error = {
                    "code": type(exc).__name__,
                    "detail": str(exc)[:256],
                }
                self._retry_after = time.monotonic() + 2
                self._read_pending = False
            return
        with self._lock:
            self._snapshot = snapshot
            self._snapshot_model_id = model_swarm_id
            self._error = None
            self._retry_after = time.monotonic() + 2
            self._read_pending = False

    def _status(self, state, *, decision: str, error) -> dict[str, object]:
        return {
            "mode": "autonomous",
            "context_tokens": self._context_tokens,
            "phase": state.phase.value,
            "generation": state.generation,
            "current_span": (
                None
                if state.current_span is None
                else [state.current_span.start, state.current_span.end]
            ),
            "target_span": (
                None
                if state.target_span is None
                else [state.target_span.start, state.target_span.end]
            ),
            "decision": decision,
            "error": error,
        }
