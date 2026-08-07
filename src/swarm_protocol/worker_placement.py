"""Non-blocking worker-side orchestration for autonomous v3 placement."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from swarm_protocol.contracts import (
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    LinkMetric,
    ModelManifest,
    ModelMemberAdvertisement,
    RecoveryLevel,
    RequestContract,
    ReservationState,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)
from swarm_protocol.discovery import DiscoverySnapshot
from swarm_protocol.context_placement import ContextCapacityDemandMap
from swarm_protocol.execution import WorkerExecutionAdmission
from swarm_protocol.placement import (
    AutonomousPlacementPolicy,
    CapacityDemandMap,
    MaterializationPhase,
    PlacementAction,
    PlacementDecision,
    PlacementMaterializer,
    SpanStaticBytes,
)
from swarm_protocol.routing import ExactRoutePlanner, RoutePlanningError

logger = logging.getLogger(__name__)

_TRANSITION_REFRESH_INTERVAL_SECONDS = 15.0


class PlacementCatalog(Protocol):
    def snapshot(
        self,
        *,
        model_swarm_id: str | None = None,
        now_ms: int | None = None,
    ) -> DiscoverySnapshot: ...

    def context_demand(
        self,
        manifest: ModelManifest,
        *,
        region_id: str,
        now_ms: int | None = None,
    ) -> ContextCapacityDemandMap | None: ...


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


@dataclass(frozen=True)
class AutonomousPeerTopology:
    """Bounded DHT-derived peers needed by one worker's serving edges.

    ``outbound_worker_ids`` are actively qualified by this worker.  The
    separate authorization set contains potential predecessors that may
    qualify an inbound edge.  Keeping these sets distinct avoids an O(N^2)
    active probe mesh while still allowing every adjacent pipeline edge and
    the mandatory tail-to-head decode closure to form without scheduler-owned
    assignments.
    """

    outbound_worker_ids: tuple[str, ...]
    authorized_worker_ids: tuple[str, ...]


def _can_transition(left: SpanLease, right: SpanLease) -> bool:
    """Return whether a forward route can switch from ``left`` to ``right``."""

    if left.worker_id == right.worker_id:
        return False
    left_span = left.hosted_span
    right_span = right.hosted_span
    if left.effective_span_mode is EffectiveSpanMode.FIXED:
        transition = left_span.end
        if right.effective_span_mode is EffectiveSpanMode.FIXED:
            return transition == right_span.start
        return right_span.start <= transition < right_span.end
    if right.effective_span_mode is EffectiveSpanMode.FIXED:
        transition = right_span.start
        return left_span.start < transition <= left_span.end
    # Elastic spans may meet at any strictly-progressing layer contained by
    # the right span and reachable within the left span.
    return max(left_span.start + 1, right_span.start) <= min(
        left_span.end,
        right_span.end - 1,
    )


def autonomous_peer_topology(
    snapshot: DiscoverySnapshot,
    *,
    worker_id: str,
    model_num_layers: int,
    max_outbound_peers: int = 8,
) -> AutonomousPeerTopology:
    """Derive a sparse executable link graph from a signed DHT snapshot.

    Petals discovers serving spans from its DHT and connects to the selected
    servers on demand.  Fabi's fixed Parallax pipeline additionally needs
    authenticated directed activation links before a request is admitted.
    Each worker therefore qualifies only plausible successors plus the
    tail-to-head closure, while authorizing plausible predecessors.  This is
    worker-owned topology; the scheduler does not nominate peers.
    """

    if not worker_id:
        raise ValueError("worker_id must not be empty")
    if model_num_layers <= 0:
        raise ValueError("model_num_layers must be positive")
    if max_outbound_peers <= 0:
        raise ValueError("max_outbound_peers must be positive")

    offers = {offer.worker_id: offer for offer in snapshot.offers}
    leases = {
        lease.worker_id: lease
        for lease in snapshot.leases
        if lease.state in {SpanState.BUILDING, SpanState.READY} and lease.worker_id in offers
    }
    current = leases.get(worker_id)
    if current is None:
        return AutonomousPeerTopology((), ())

    others = tuple(lease for peer, lease in leases.items() if peer != worker_id)

    def _rank(lease: SpanLease) -> tuple[int, int, int, str]:
        return (
            int(lease.state is not SpanState.READY),
            -lease.hosted_span.end,
            lease.hosted_span.start,
            lease.worker_id,
        )

    successors = sorted(
        (lease for lease in others if _can_transition(current, lease)),
        key=_rank,
    )
    predecessors = {lease.worker_id for lease in others if _can_transition(lease, current)}

    # Decode returns sampled tokens from a tail stage to a frontend-capable
    # head.  This closure is a real directed edge in Fabi's cyclic request
    # path, not an implicit scheduler hop.
    heads = sorted(
        (
            lease
            for lease in others
            if lease.hosted_span.start == 0
            and WorkerRole.FRONTEND in offers[lease.worker_id].supported_roles
        ),
        key=_rank,
    )
    tails = {lease.worker_id for lease in others if lease.hosted_span.end == model_num_layers}
    if current.hosted_span.end == model_num_layers:
        successors = heads + [
            lease for lease in successors if lease.worker_id not in {x.worker_id for x in heads}
        ]
    if current.hosted_span.start == 0 and WorkerRole.FRONTEND in offers[worker_id].supported_roles:
        predecessors.update(tails)

    outbound = tuple(dict.fromkeys(lease.worker_id for lease in successors))[:max_outbound_peers]
    # Authorizing a signed current model member is cheap and passive.  Do not
    # cap this set with the active probe budget: otherwise a valid predecessor
    # could select us while being rejected solely because of local ordering.
    authorized = tuple(sorted(predecessors))
    return AutonomousPeerTopology(outbound, authorized)


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


def reconciled_autonomous_context_limit(
    requested_tokens: int,
    supported_tokens: int,
    *,
    minimum_tokens: int = 4_096,
) -> int | None:
    """Keep the exact KV ceiling proved to fit by the live executor.

    Static placement uses the stable pre-load memory envelope, but only the
    initialized backend can measure its final workspace and KV footprint.  A
    failed cold join therefore reconciles the worker's own context claim to
    that measured limit instead of rounding it down to a demand-summary class
    or asking the legacy scheduler to resize an autonomous DHT generation.

    Context classes are cumulative aggregation buckets, not machine capacity
    limits.  The backend already reports ``supported_tokens`` in its concrete
    KV geometry, so replacing (for example) 30,752 with the 16,384 demand class
    would discard usable capacity without adding a safety invariant.
    """

    if requested_tokens <= 0 or minimum_tokens <= 0:
        raise ValueError("requested and minimum context tokens must be positive")
    if supported_tokens < 0:
        raise ValueError("supported context tokens cannot be negative")
    floor = min(requested_tokens, minimum_tokens)
    if supported_tokens < floor or supported_tokens >= requested_tokens:
        return None
    return supported_tokens


class AutonomousWorkerPlacement:
    """Compose DHT policy, admission drain and the existing executor reload path."""

    def __init__(
        self,
        *,
        catalog: PlacementCatalog,
        admission: WorkerExecutionAdmission,
        state_publisher: SpanStatePublisher,
        reload_target: Callable[[LayerSpan, int, int], None],
        current_span: LayerSpan | None = None,
        current_context_tokens: int | None = None,
        policy: AutonomousPlacementPolicy | None = None,
        topology_observer: Callable[[DiscoverySnapshot], None] | None = None,
        transition_refresh_interval_s: float = _TRANSITION_REFRESH_INTERVAL_SECONDS,
        demand_region_id: str | None = None,
        span_static_bytes: SpanStaticBytes | None = None,
        materialization_identity_hashes: tuple[str, ...] | None = None,
    ) -> None:
        if transition_refresh_interval_s <= 0:
            raise ValueError("transition refresh interval must be positive")
        self._catalog = catalog
        self._admission = admission
        self._state_publisher = state_publisher
        self._policy = policy or AutonomousPlacementPolicy()
        self._topology_observer = topology_observer
        self._materializer = PlacementMaterializer(
            drain=admission,
            reload_target=reload_target,
            current_span=current_span,
            current_context_tokens=current_context_tokens,
        )
        self._snapshot: DiscoverySnapshot | None = None
        self._snapshot_model_id: str | None = None
        self._read_pending = False
        self._retry_after = 0.0
        # A process that adopts an already loaded legacy span must not treat it
        # as infinitely old and immediately churn on its first DHT snapshot.
        self._last_moved_at_ms: int | None = (
            time.time_ns() // 1_000_000 if current_span is not None else None
        )
        self._announced_transition: tuple[object, ...] | None = None
        self._transition_advertisement: ModelMemberAdvertisement | None = None
        self._transition_renewal_thread: threading.Thread | None = None
        self._transition_refresh_interval_s = transition_refresh_interval_s
        self._transition_publish_lock = threading.RLock()
        self._error: dict[str, str] | None = None
        self._context_tokens: int | None = None
        self._qualified_context_limit_tokens: int | None = None
        self._demand_region_id = demand_region_id
        self._span_static_bytes = span_static_bytes
        self._materialization_identity_hashes = materialization_identity_hashes
        self._context_demand: ContextCapacityDemandMap | None = None
        self._context_demand_state = "off" if demand_region_id is None else "waiting"
        self._context_demand_status: dict[str, object] | None = None
        # Exact local materialization failures are hard constraints, never
        # placement preferences.  The normal score remains byte-for-byte
        # identical for every candidate that is physically feasible.
        self._storage_rejected_spans: set[LayerSpan] = set()
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
            if self._materialization_identity_hashes is None:
                self._materialization_identity_hashes = weight_hashes
            elif self._materialization_identity_hashes != weight_hashes:
                raise ValueError("bootstrap identity differs from the execution contract")
            self._context_tokens = context_tokens
            self._qualified_context_limit_tokens = max(
                context_tokens,
                self._qualified_context_limit_tokens or 0,
            )

        state = self._materializer.snapshot()
        if state.phase is not MaterializationPhase.STANDBY:
            return self._status(state, decision="materializing", error=None)

        self._refresh_async(manifest.model_swarm_id)
        with self._lock:
            snapshot = (
                self._snapshot if self._snapshot_model_id == manifest.model_swarm_id else None
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

        baseline = self._policy.choose(
            offer=offer,
            manifest=manifest,
            leases=snapshot.leases,
            # This is only the chicken-and-egg bootstrap needed before the
            # first legitimate request can create signed demand.  It is not a
            # steady-state replica policy.
            demand=CapacityDemandMap.uniform(manifest.num_layers, desired_replicas=1),
            context_tokens=context_tokens,
            kv_block_size=kv_block_size,
            current_span=None,
            current_reservations=0,
            excluded_spans=self._storage_exclusions(),
            span_static_bytes=self._span_static_bytes,
            now_ms=time.time_ns() // 1_000_000,
        )
        decision = self._select_context_demand(
            baseline=baseline,
            offer=offer,
            manifest=manifest,
            leases=snapshot.leases,
            context_tokens=context_tokens,
            kv_block_size=kv_block_size,
            current_span=None,
            current_reservations=0,
            cold_start=True,
            now_ms=time.time_ns() // 1_000_000,
        )
        if decision.action is PlacementAction.STANDBY:
            return self._status(state, decision=decision.reason, error=read_error)
        if decision.action is not PlacementAction.JOIN or decision.span is None:
            raise RuntimeError("cold placement policy returned an invalid transition")
        if decision.context_tokens is None:
            raise RuntimeError("cold placement policy omitted its context target")

        span = decision.span
        selected_context_tokens = decision.context_tokens
        with self._lock:
            self._context_tokens = selected_context_tokens
        rounded_tokens = (
            (selected_context_tokens + kv_block_size - 1) // kv_block_size * kv_block_size
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
                max_context_tokens=selected_context_tokens,
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
        self._track_transition(building, state)
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
            self._qualified_context_limit_tokens = max(
                context_tokens,
                self._qualified_context_limit_tokens or 0,
            )

        state = self._materializer.snapshot()
        if (
            state.phase is MaterializationPhase.BUILDING
            and advertisement.lease.state is SpanState.READY
            and advertisement.lease.hosted_span == state.target_span
        ):
            with self._transition_publish_lock:
                self._admission.configure(advertisement)
                state = self._materializer.mark_ready(
                    span=advertisement.lease.hosted_span,
                    context_tokens=advertisement.lease.max_context_tokens,
                    generation=state.generation,
                )
                self._last_moved_at_ms = time.time_ns() // 1_000_000
                with self._lock:
                    self._announced_transition = None
                    self._transition_advertisement = None
        elif state.phase is MaterializationPhase.DRAINING:
            with self._transition_publish_lock:
                state = self._materializer.continue_after_drain()
                self._announce_transition_once(advertisement, state, manifest)
        elif state.phase is MaterializationPhase.READY:
            self._admission.configure(advertisement)

        self._refresh_async(manifest.model_swarm_id)
        with self._lock:
            snapshot = (
                self._snapshot if self._snapshot_model_id == manifest.model_swarm_id else None
            )
            read_error = self._error

        state = self._materializer.snapshot()
        if state.phase in {MaterializationPhase.DRAINING, MaterializationPhase.BUILDING}:
            self._announce_transition_once(advertisement, state, manifest)
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
        serving_route_exists = self._route_exists(
            snapshot=snapshot,
            manifest=manifest,
            coordinator_id=advertisement.offer.worker_id,
            context_tokens=context_tokens,
        )
        serving_route_survives_movement = self._route_exists(
            snapshot=snapshot,
            manifest=manifest,
            coordinator_id=advertisement.offer.worker_id,
            context_tokens=context_tokens,
            exclude_worker_id=advertisement.offer.worker_id,
        )
        decision = PlacementDecision(
            action=PlacementAction.KEEP,
            span=state.current_span,
            required_memory_bytes=0,
            score=None,
            reason="holding_last_ready_target_without_valid_context_demand",
            context_tokens=state.current_context_tokens,
        )
        decision = self._select_context_demand(
            baseline=decision,
            offer=advertisement.offer,
            manifest=manifest,
            leases=snapshot.leases,
            context_tokens=context_tokens,
            kv_block_size=advertisement.lease.kv_geometry.block_size_tokens,
            current_span=state.current_span,
            current_reservations=active_reservations,
            cold_start=False,
            last_moved_at_ms=self._last_moved_at_ms,
            serving_route_exists=serving_route_exists,
            serving_route_survives_movement=serving_route_survives_movement,
            now_ms=time.time_ns() // 1_000_000,
        )
        if decision.action in {PlacementAction.JOIN, PlacementAction.MOVE}:
            with self._transition_publish_lock:
                if decision.context_tokens is None:
                    raise RuntimeError("adaptive placement omitted its context target")
                with self._lock:
                    self._context_tokens = decision.context_tokens
                state = self._materializer.reconcile(decision)
                self._announce_transition_once(advertisement, state, manifest)
        return self._status(state, decision=decision.reason, error=read_error)

    def downgrade_building_context(self, context_tokens: int) -> dict[str, object]:
        """Republish one non-routable cold join with a measured lower KV tier.

        The layer span and materialization generation remain unchanged: only
        the worker-local KV claim changes.  ``publish_span_state`` sequences
        the replacement lease, so concurrent readers cannot mistake it for
        the previous, larger BUILDING contract.
        """

        if context_tokens <= 0:
            raise ValueError("placement context contract must be positive")
        with self._transition_publish_lock:
            state = self._materializer.snapshot()
            if state.phase is not MaterializationPhase.BUILDING:
                raise RuntimeError("context downgrade requires a BUILDING placement")
            with self._lock:
                previous_context = self._context_tokens
                advertisement = self._transition_advertisement
            if previous_context is None or context_tokens >= previous_context:
                raise ValueError("context downgrade must reduce the current contract")
            if advertisement is None:
                raise RuntimeError("BUILDING placement has no renewable advertisement")

            state = self._materializer.downgrade_building_context(context_tokens)

            geometry = advertisement.lease.kv_geometry
            adjusted_geometry = geometry.model_copy(
                update={
                    "allocatable_bytes": geometry.required_bytes(
                        advertisement.lease.hosted_span,
                        context_tokens,
                    )
                }
            )
            adjusted = advertisement.model_copy(
                update={
                    "lease": advertisement.lease.model_copy(
                        update={
                            "max_context_tokens": context_tokens,
                            "kv_geometry": adjusted_geometry,
                        }
                    )
                }
            )
            transitioned = self._state_publisher.publish_span_state(
                adjusted,
                SpanState.BUILDING,
            )
            with self._lock:
                self._context_tokens = context_tokens
            self._track_transition(transitioned, state)
        return self._status(
            state,
            decision="measured_context_downgrade",
            error=None,
        )

    @staticmethod
    def _route_exists(
        *,
        snapshot: DiscoverySnapshot,
        manifest: ModelManifest,
        coordinator_id: str,
        context_tokens: int,
        exclude_worker_id: str | None = None,
    ) -> bool:
        """Prove that one executable route exists in an exact DHT snapshot."""

        reserved_output_tokens = min(4_096, context_tokens - 1)
        request = RequestContract(
            request_id=(
                f"placement-safety-{coordinator_id}-{exclude_worker_id or 'current'}-"
                f"{snapshot.captured_at_ms}"
            ),
            model_swarm_id=manifest.model_swarm_id,
            prompt_tokens=context_tokens - reserved_output_tokens,
            reserved_output_tokens=reserved_output_tokens,
            recovery_level=RecoveryLevel.RESTARTABLE,
        )
        try:
            ExactRoutePlanner().plan(
                manifest=manifest,
                request=request,
                offers=tuple(
                    offer for offer in snapshot.offers if offer.worker_id != exclude_worker_id
                ),
                leases=tuple(
                    lease for lease in snapshot.leases if lease.worker_id != exclude_worker_id
                ),
                links=tuple(
                    link
                    for link in snapshot.links
                    if exclude_worker_id not in {link.from_worker_id, link.to_worker_id}
                ),
                snapshot_time_ms=snapshot.captured_at_ms,
                coordinator_id=coordinator_id,
                reservation_deadline_ms=snapshot.captured_at_ms + 30_000,
                plan_expires_at_ms=snapshot.captured_at_ms + 60_000,
            )
        except RoutePlanningError:
            return False
        return True

    def mark_failed(self, *, generation: int, error: Exception) -> dict[str, object]:
        """Fence a backend failure and start the materializer's safe rollback."""

        state = self._materializer.mark_failed(generation=generation, error=error)
        return self._status(
            state,
            decision="rolling_back_previous_span",
            error={"code": type(error).__name__, "detail": str(error)[:256]},
        )

    def reject_storage_target(self, *, generation: int, error: Exception) -> dict[str, object]:
        """Exclude one exactly measured local disk failure and replan.

        Cache presence, download cost and popularity never enter the ranking.
        This exclusion is added only after the cache transaction proves that
        the selected span cannot fit even after safe garbage collection.
        """

        before = self._materializer.snapshot()
        if before.target_span is None:
            raise RuntimeError("storage failure has no materialization target")
        rejected_span = before.target_span
        state = self._materializer.reject_unavailable_target(
            generation=generation,
            error=error,
        )
        with self._lock:
            self._storage_rejected_spans.add(rejected_span)
            if state.phase is MaterializationPhase.STANDBY:
                self._announced_transition = None
                self._transition_advertisement = None
        decision = (
            "rolling_back_previous_span"
            if state.phase is MaterializationPhase.BUILDING
            else "replanning_after_storage_rejection"
        )
        return self._status(
            state,
            decision=decision,
            error={"code": type(error).__name__, "detail": str(error)[:256]},
        )

    def _storage_exclusions(self) -> frozenset[LayerSpan]:
        with self._lock:
            return frozenset(self._storage_rejected_spans)

    def _announce_transition_once(
        self,
        advertisement: ModelMemberAdvertisement,
        state,
        manifest: ModelManifest,
    ) -> None:
        key = (
            state.phase,
            state.generation,
            state.target_span,
            state.target_context_tokens,
        )
        with self._transition_publish_lock:
            with self._lock:
                if self._announced_transition == key:
                    return
            span_state = self._span_state_for_phase(state.phase)
            transition_advertisement = advertisement
            if state.phase is MaterializationPhase.BUILDING:
                if state.target_span is None or state.target_context_tokens is None:
                    raise RuntimeError("BUILDING placement has no complete target")
                rounded_tokens = (
                    (
                        state.target_context_tokens
                        + advertisement.lease.kv_geometry.block_size_tokens
                        - 1
                    )
                    // advertisement.lease.kv_geometry.block_size_tokens
                    * advertisement.lease.kv_geometry.block_size_tokens
                )
                allocatable_kv_bytes = rounded_tokens * sum(
                    manifest.kv_bytes_per_token_by_layer[
                        state.target_span.start : state.target_span.end
                    ]
                )
                transition_advertisement = advertisement.model_copy(
                    update={
                        "lease": advertisement.lease.model_copy(
                            update={
                                "hosted_span": state.target_span,
                                "weight_hashes": self._transition_hashes(manifest),
                                "max_context_tokens": state.target_context_tokens,
                                "kv_geometry": advertisement.lease.kv_geometry.model_copy(
                                    update={"allocatable_bytes": allocatable_kv_bytes}
                                ),
                            }
                        )
                    }
                )
            transitioned = self._state_publisher.publish_span_state(
                transition_advertisement,
                span_state,
            )
            self._track_transition(transitioned, state)

    @staticmethod
    def _span_state_for_phase(phase: MaterializationPhase) -> SpanState:
        if phase is MaterializationPhase.DRAINING:
            return SpanState.DRAINING
        if phase is MaterializationPhase.BUILDING:
            return SpanState.BUILDING
        raise ValueError(f"{phase.value} is not a renewable placement transition")

    def _track_transition(self, advertisement, state) -> None:
        """Keep non-routable placement intent alive during slow materialization.

        Model downloads and backend initialization can legitimately take much
        longer than one DHT TTL.  As in Petals' ModuleAnnouncerThread, renewal
        is therefore owned by a small independent daemon rather than by an
        executor heartbeat or a guessed loading deadline.
        """

        key = (
            state.phase,
            state.generation,
            state.target_span,
            state.target_context_tokens,
        )
        with self._lock:
            self._announced_transition = key
            self._transition_advertisement = advertisement
            if (
                self._transition_renewal_thread is not None
                and self._transition_renewal_thread.is_alive()
            ):
                return
            thread = threading.Thread(
                target=self._renew_transition,
                name="SwarmV3PlacementAnnouncer",
                daemon=True,
            )
            self._transition_renewal_thread = thread
            thread.start()

    def _renew_transition(self) -> None:
        while True:
            time.sleep(self._transition_refresh_interval_s)
            with self._transition_publish_lock:
                state = self._materializer.snapshot()
                if state.phase not in {
                    MaterializationPhase.DRAINING,
                    MaterializationPhase.BUILDING,
                }:
                    with self._lock:
                        self._transition_advertisement = None
                    continue
                key = (
                    state.phase,
                    state.generation,
                    state.target_span,
                    state.target_context_tokens,
                )
                with self._lock:
                    advertisement = self._transition_advertisement
                    announced = self._announced_transition
                if advertisement is None:
                    continue
                # A phase/generation change must first publish its own exact
                # transition. Never renew a stale DRAINING value as BUILDING.
                if announced != key:
                    continue
                span_state = self._span_state_for_phase(state.phase)
                try:
                    refreshed = self._state_publisher.publish_span_state(
                        advertisement,
                        span_state,
                    )
                except Exception:  # noqa: BLE001 - asynchronous DHT publication boundary
                    logger.warning("Autonomous transition renewal failed", exc_info=True)
                    continue
                with self._lock:
                    if self._announced_transition == key:
                        self._transition_advertisement = refreshed

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
            demand = None
            demand_state = "off"
            if self._demand_region_id is not None:
                manifest = snapshot.manifest(model_swarm_id)
                if manifest is None:
                    demand_state = "missing_manifest"
                else:
                    demand = self._catalog.context_demand(
                        manifest,
                        region_id=self._demand_region_id,
                        now_ms=snapshot.captured_at_ms,
                    )
                    demand_state = "valid" if demand is not None else "missing"
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
            self._context_demand = demand
            self._context_demand_state = demand_state
            self._error = None
            self._retry_after = time.monotonic() + 2
            self._read_pending = False
        if self._topology_observer is not None:
            try:
                self._topology_observer(snapshot)
            except Exception:  # noqa: BLE001 - telemetry callback boundary
                # Placement must continue from the verified snapshot.  The
                # missing link will keep route admission fail-closed and a
                # later refresh retries topology publication.
                logger.warning("Autonomous topology projection failed", exc_info=True)

    def _select_context_demand(
        self,
        *,
        baseline: PlacementDecision,
        offer: WorkerOffer,
        manifest: ModelManifest,
        leases: tuple[SpanLease, ...],
        context_tokens: int,
        kv_block_size: int,
        current_span: LayerSpan | None,
        current_reservations: int,
        cold_start: bool,
        now_ms: int,
        last_moved_at_ms: int | None = None,
        serving_route_exists: bool = False,
        serving_route_survives_movement: bool = False,
    ) -> PlacementDecision:
        """Apply trusted demand, or retain the sole cold-start/READY fallback."""

        with self._lock:
            demand = self._context_demand
            demand_state = self._context_demand_state
            qualified_context_limit_tokens = self._qualified_context_limit_tokens
        if demand is None:
            with self._lock:
                self._context_demand_status = {
                    "state": demand_state,
                    "applied": False,
                    "fallback": "cold_start" if cold_start else "hold_last_ready",
                }
            return baseline
        try:
            demand.validate_for(manifest, now_ms=now_ms)
            if qualified_context_limit_tokens is None:
                raise ValueError("worker has no qualified context limit")
            selected, utility = self._policy.choose_contextual(
                offer=offer,
                manifest=manifest,
                leases=leases,
                demand=demand,
                qualified_context_limit_tokens=qualified_context_limit_tokens,
                kv_block_size=kv_block_size,
                current_span=current_span,
                current_context_tokens=(None if cold_start else context_tokens),
                current_reservations=current_reservations,
                last_moved_at_ms=last_moved_at_ms,
                serving_route_exists=serving_route_exists,
                serving_route_survives_movement=serving_route_survives_movement,
                excluded_spans=self._storage_exclusions(),
                span_static_bytes=self._span_static_bytes,
                now_ms=now_ms,
            )
        except ValueError as exc:
            no_demand = str(exc) == "context demand snapshot has no locally qualified demand"
            status: dict[str, object] = {
                "state": "no_demand" if no_demand else "invalid",
                "applied": False,
                "fallback": "cold_start" if cold_start else "hold_last_ready",
                "error": str(exc)[:256],
            }
        else:
            status = {
                "state": "active",
                "applied": True,
                "region_id": demand.region_id,
                "demand_issued_at_ms": demand.issued_at_ms,
                "selected_action": selected.action.value,
                "selected_span": (
                    None if selected.span is None else [selected.span.start, selected.span.end]
                ),
                "selected_context_tokens": selected.context_tokens,
                "weighted_complete_routes": utility.weighted_complete_routes,
                "weighted_concurrent_slots": utility.weighted_concurrent_slots,
                "weighted_layer_deficit_filled": utility.weighted_layer_deficit_filled,
                "decision": selected.reason,
            }
        with self._lock:
            self._context_demand_status = status
        return selected if status["applied"] else baseline

    def _transition_hashes(self, manifest: ModelManifest) -> tuple[str, ...]:
        """Return the compact signed identity for the executor being built."""

        with self._lock:
            hashes = self._materialization_identity_hashes
        return hashes or (manifest.weight_collection_hash,)

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
            "current_context_tokens": state.current_context_tokens,
            "target_context_tokens": state.target_context_tokens,
            "decision": decision,
            "storage_rejected_spans": len(self._storage_exclusions()),
            "context_demand": self._context_demand_status,
            "error": error,
        }
