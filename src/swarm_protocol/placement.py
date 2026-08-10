"""Deterministic Petals-style autonomous span placement for protocol v3."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import networkx as nx

from swarm_protocol.context_placement import (
    CapacityDemandMap,
    ContextCapacityDemandMap,
    MemoryPlacementPoint,
)
from swarm_protocol.contracts import (
    LayerSpan,
    ModelManifest,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)

SpanStaticBytes = Callable[[LayerSpan], int | None]


class PlacementAction(str, Enum):
    JOIN = "join"
    KEEP = "keep"
    MOVE = "move"
    STANDBY = "standby"


class MaterializationPhase(str, Enum):
    STANDBY = "standby"
    READY = "ready"
    DRAINING = "draining"
    BUILDING = "building"
    FAILED = "failed"


class PlacementDrain(Protocol):
    def begin_drain(self) -> int: ...

    def draining_reservations(self) -> int: ...

    def cancel_drain(self) -> None: ...

    def finish_drain(self) -> None: ...


class SpanReloadTarget(Protocol):
    def __call__(self, span: LayerSpan, context_tokens: int, generation: int) -> None: ...


@dataclass(frozen=True)
class MaterializationSnapshot:
    phase: MaterializationPhase
    generation: int
    current_span: LayerSpan | None
    current_context_tokens: int | None
    target_span: LayerSpan | None
    target_context_tokens: int | None
    previous_span: LayerSpan | None
    previous_context_tokens: int | None
    error: str | None = None


class PlacementMaterializer:
    """Fence one fixed-backend span reload behind worker-local admission.

    Petals restarts its module container after announcing JOINING/OFFLINE. Fabi
    keeps that proven lifecycle but makes it explicit and generation-fenced:
    DRAINING closes PREPARE atomically, BUILDING triggers the existing executor
    reload path, and only a matching verified completion may publish READY.
    """

    def __init__(
        self,
        *,
        drain: PlacementDrain,
        reload_target: SpanReloadTarget,
        current_span: LayerSpan | None = None,
        current_context_tokens: int | None = None,
    ) -> None:
        if (current_span is None) != (current_context_tokens is None):
            raise ValueError("current span and context must be supplied together")
        if current_context_tokens is not None and current_context_tokens <= 0:
            raise ValueError("current context must be positive")
        self._drain = drain
        self._reload_target = reload_target
        self._phase = (
            MaterializationPhase.READY if current_span is not None else MaterializationPhase.STANDBY
        )
        self._generation = 0
        self._current_span = current_span
        self._current_context_tokens = current_context_tokens
        self._target_span: LayerSpan | None = None
        self._target_context_tokens: int | None = None
        self._previous_span: LayerSpan | None = None
        self._previous_context_tokens: int | None = None
        self._error: str | None = None
        self._lock = threading.RLock()

    def snapshot(self) -> MaterializationSnapshot:
        with self._lock:
            return MaterializationSnapshot(
                phase=self._phase,
                generation=self._generation,
                current_span=self._current_span,
                current_context_tokens=self._current_context_tokens,
                target_span=self._target_span,
                target_context_tokens=self._target_context_tokens,
                previous_span=self._previous_span,
                previous_context_tokens=self._previous_context_tokens,
                error=self._error,
            )

    def reconcile(self, decision: PlacementDecision) -> MaterializationSnapshot:
        """Advance a placement decision without ever overlapping generations."""

        with self._lock:
            if self._phase is MaterializationPhase.BUILDING:
                if (
                    decision.span != self._target_span
                    or decision.context_tokens != self._target_context_tokens
                ):
                    raise RuntimeError("cannot replace an in-flight materialization target")
                return self.snapshot()
            if decision.action is PlacementAction.KEEP:
                return self.snapshot()
            if decision.action is PlacementAction.STANDBY:
                if self._current_span is not None:
                    raise RuntimeError(
                        "a serving worker cannot enter standby without a safe target"
                    )
                self._phase = MaterializationPhase.STANDBY
                return self.snapshot()
            if decision.span is None:
                raise ValueError("join and move decisions require a target span")
            if decision.context_tokens is None or decision.context_tokens <= 0:
                raise ValueError("join and move decisions require a positive target context")
            if decision.action is PlacementAction.JOIN and self._current_span is not None:
                raise RuntimeError("a serving worker cannot execute a join decision")
            if decision.action is PlacementAction.MOVE and self._current_span is None:
                raise RuntimeError("a standby worker cannot execute a move decision")

            self._target_span = decision.span
            self._target_context_tokens = decision.context_tokens
            self._previous_span = self._current_span
            self._previous_context_tokens = self._current_context_tokens
            self._error = None
            if self._current_span is not None:
                self._phase = MaterializationPhase.DRAINING
                if self._drain.begin_drain() > 0:
                    return self.snapshot()
            self._start_reload_locked(decision.span, decision.context_tokens)
            return self.snapshot()

    def continue_after_drain(self) -> MaterializationSnapshot:
        with self._lock:
            if self._phase is not MaterializationPhase.DRAINING:
                return self.snapshot()
            if self._drain.draining_reservations() > 0:
                return self.snapshot()
            assert self._target_span is not None
            assert self._target_context_tokens is not None
            self._start_reload_locked(self._target_span, self._target_context_tokens)
            return self.snapshot()

    def _start_reload_locked(self, span: LayerSpan, context_tokens: int) -> None:
        if context_tokens <= 0:
            raise ValueError("reload context must be positive")
        self._generation += 1
        generation = self._generation
        self._phase = MaterializationPhase.BUILDING
        try:
            self._reload_target(span, context_tokens, generation)
        except Exception as exc:
            self._phase = MaterializationPhase.FAILED
            self._error = f"{type(exc).__name__}: {exc}"[:256]
            if self._previous_span == self._current_span and self._current_span is not None:
                self._drain.cancel_drain()
                self._phase = MaterializationPhase.READY
            raise
        self._current_span = None
        self._current_context_tokens = None

    def mark_ready(
        self,
        *,
        span: LayerSpan,
        context_tokens: int,
        generation: int,
    ) -> MaterializationSnapshot:
        """Accept only the verified completion for the current reload generation."""

        with self._lock:
            if (
                self._phase is not MaterializationPhase.BUILDING
                or generation != self._generation
                or span != self._target_span
                or context_tokens != self._target_context_tokens
            ):
                raise RuntimeError("stale or mismatched materialization completion")
            self._current_span = span
            self._current_context_tokens = context_tokens
            self._target_span = None
            self._target_context_tokens = None
            self._previous_span = None
            self._previous_context_tokens = None
            self._phase = MaterializationPhase.READY
            self._error = None
            self._drain.finish_drain()
            return self.snapshot()

    def downgrade_building_context(self, context_tokens: int) -> MaterializationSnapshot:
        """Reconcile a backend-measured lower KV ceiling for this generation."""

        if context_tokens <= 0:
            raise ValueError("measured context must be positive")
        with self._lock:
            if self._phase is not MaterializationPhase.BUILDING:
                raise RuntimeError("context downgrade requires a BUILDING placement")
            if self._target_context_tokens is None or context_tokens >= self._target_context_tokens:
                raise ValueError("context downgrade must reduce the current target")
            self._target_context_tokens = context_tokens
            return self.snapshot()

    def mark_failed(self, *, generation: int, error: Exception) -> MaterializationSnapshot:
        with self._lock:
            if generation != self._generation or self._phase is not MaterializationPhase.BUILDING:
                raise RuntimeError("stale materialization failure")
            self._phase = MaterializationPhase.FAILED
            self._error = f"{type(error).__name__}: {error}"[:256]
            if self._previous_span is not None:
                rollback_span = self._previous_span
                rollback_context_tokens = self._previous_context_tokens
                assert rollback_context_tokens is not None
                self._target_span = rollback_span
                self._target_context_tokens = rollback_context_tokens
                try:
                    self._start_reload_locked(rollback_span, rollback_context_tokens)
                except Exception:
                    # ``_start_reload_locked`` records the synchronous failure.
                    pass
            return self.snapshot()

    def reject_unavailable_target(
        self,
        *,
        generation: int,
        error: Exception,
    ) -> MaterializationSnapshot:
        """Reject one locally impossible target without killing a cold worker.

        A move still rolls back to its last verified span.  A cold join has no
        generation to restore, so it returns to STANDBY and lets the unchanged
        placement score select the best remaining physically feasible span.
        """

        with self._lock:
            if generation != self._generation or self._phase is not MaterializationPhase.BUILDING:
                raise RuntimeError("stale materialization failure")
            rejected = self._target_span
            if rejected is None:
                raise RuntimeError("building materialization has no target")
            self._error = f"{type(error).__name__}: {error}"[:256]
            if self._previous_span is not None:
                rollback_span = self._previous_span
                rollback_context_tokens = self._previous_context_tokens
                assert rollback_context_tokens is not None
                self._target_span = rollback_span
                self._target_context_tokens = rollback_context_tokens
                try:
                    self._start_reload_locked(rollback_span, rollback_context_tokens)
                except Exception:
                    pass
            else:
                self._phase = MaterializationPhase.STANDBY
                self._target_span = None
                self._target_context_tokens = None
                self._current_span = None
                self._current_context_tokens = None
                self._previous_span = None
                self._previous_context_tokens = None
            return self.snapshot()


@dataclass(frozen=True)
class PlacementScore:
    completes_fixed_route: int
    establishes_missing_frontend: int
    fixed_route_progress: int
    minimum_ready_coverage: int
    weighted_deficit_filled: float
    weighted_coverage: float
    span_length: int
    deterministic_tiebreaker: int

    def rank(self) -> tuple[int, int, int, int, float, float, int, int]:
        return (
            self.completes_fixed_route,
            self.establishes_missing_frontend,
            self.fixed_route_progress,
            self.minimum_ready_coverage,
            self.weighted_deficit_filled,
            self.weighted_coverage,
            self.span_length,
            self.deterministic_tiebreaker,
        )


@dataclass(frozen=True)
class PlacementDecision:
    action: PlacementAction
    span: LayerSpan | None
    required_memory_bytes: int
    score: PlacementScore | None
    reason: str
    context_tokens: int | None = None


@dataclass(frozen=True)
class ContextPlacementUtility:
    """Demand served by one local span/context choice.

    The fields deliberately remain explainable.  They are computed only from
    a trusted aggregate and signed/leased capacity; no prompt content or
    coordinator-owned layer assignment enters the worker decision.
    """

    weighted_complete_routes: float
    weighted_concurrent_slots: float
    weighted_layer_deficit_filled: float
    context_tokens: int
    span_length: int
    max_sessions: int

    @property
    def value(self) -> float:
        return (
            2.0 * self.weighted_complete_routes
            + self.weighted_concurrent_slots
            + self.weighted_layer_deficit_filled
        )

    def rank(self) -> tuple[float, float, float, float, int, int, int]:
        return (
            self.value,
            self.weighted_complete_routes,
            self.weighted_concurrent_slots,
            self.weighted_layer_deficit_filled,
            self.context_tokens,
            self.span_length,
            self.max_sessions,
        )


@dataclass(frozen=True)
class _ContextSupply:
    layer_coverage: tuple[int, ...]
    session_coverage: tuple[int, ...]
    prefix_boundaries: frozenset[int]
    suffix_boundaries: frozenset[int]


class AutonomousPlacementPolicy:
    """Choose a locally feasible span without creating a voluntary coverage hole.

    Capacity is a hard constraint derived from signed exact weight geometry and
    the target KV contract. Discovery only supplies eventually consistent
    demand hints; a worker still builds, verifies, warms and publishes READY
    locally before any route can use the result.
    """

    def __init__(
        self,
        *,
        minimum_improvement: float = 0.05,
        movement_cooldown_ms: int = 10 * 60 * 1_000,
        maximum_candidates: int = 65_536,
        maximum_exact_candidates: int = 32,
    ) -> None:
        if not 0 <= minimum_improvement < 1:
            raise ValueError("minimum improvement must be in [0, 1)")
        if (
            movement_cooldown_ms < 0
            or maximum_candidates <= 0
            or maximum_exact_candidates <= 0
        ):
            raise ValueError("cooldown must be non-negative and candidate bound positive")
        self.minimum_improvement = minimum_improvement
        self.movement_cooldown_ms = movement_cooldown_ms
        self.maximum_candidates = maximum_candidates
        self.maximum_exact_candidates = maximum_exact_candidates

    @staticmethod
    def _static_bytes(
        manifest: ModelManifest,
        span: LayerSpan,
        span_static_bytes: SpanStaticBytes | None,
    ) -> int | None:
        """Return signed static bytes, or ``None`` for an unsupported boundary.

        SafeTensors-backed executors use the manifest's exact resident weight
        geometry. Portable executors supply a plan-specific resolver whose
        result is the de-duplicated size of signed ONNX graph/external-data
        files. Runtime workspaces remain covered by the independently measured
        stable memory envelope and backend reserve.
        """

        value = (
            manifest.weight_bytes(span) if span_static_bytes is None else span_static_bytes(span)
        )
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("span static byte geometry must be a positive integer")
        return value

    @classmethod
    def _required_memory_bytes(
        cls,
        manifest: ModelManifest,
        span: LayerSpan,
        *,
        context_tokens: int,
        kv_block_size: int,
        span_static_bytes: SpanStaticBytes | None = None,
    ) -> int | None:
        if context_tokens <= 0 or kv_block_size <= 0:
            raise ValueError("context and KV block size must be positive")
        static_bytes = cls._static_bytes(manifest, span, span_static_bytes)
        if static_bytes is None:
            return None
        rounded_tokens = (context_tokens + kv_block_size - 1) // kv_block_size * kv_block_size
        kv_bytes = rounded_tokens * sum(manifest.kv_bytes_per_token_by_layer[span.start : span.end])
        return static_bytes + kv_bytes

    def feasible_spans(
        self,
        *,
        offer: WorkerOffer,
        manifest: ModelManifest,
        context_tokens: int,
        kv_block_size: int,
        span_static_bytes: SpanStaticBytes | None = None,
    ) -> tuple[tuple[LayerSpan, int], ...]:
        """Enumerate every exact contiguous span this worker can materialize."""

        if span_static_bytes is None and not manifest.weight_bytes_by_layer:
            raise ValueError("autonomous placement requires exact weight byte geometry")
        granularity = offer.execution_granularity_layers
        result: list[tuple[LayerSpan, int]] = []
        attempts = 0
        for start in range(0, manifest.num_layers, granularity):
            for end in range(start + granularity, manifest.num_layers + 1, granularity):
                attempts += 1
                if attempts > self.maximum_candidates:
                    raise ValueError("placement candidate count exceeds the configured bound")
                # ``FRONTEND`` means that the worker can own request ingress
                # and tokenization at layer zero.  A final pipeline stage does
                # not need that HTTP capability: vLLM/MLX materialize the
                # output norm and lm_head from ``end == num_layers`` itself.
                if start == 0 and WorkerRole.FRONTEND not in offer.supported_roles:
                    continue
                span = LayerSpan(start=start, end=end)
                required = self._required_memory_bytes(
                    manifest,
                    span,
                    context_tokens=context_tokens,
                    kv_block_size=kv_block_size,
                    span_static_bytes=span_static_bytes,
                )
                if required is None:
                    continue
                if required <= offer.stable_memory_envelope_bytes:
                    result.append((span, required))
                # Every additional layer is positive, except that endpoint
                # ownership appears only at the final boundary. Do not break:
                # tied endpoints can make the full-model delta non-monotonic.
        return tuple(result)

    def memory_frontier(
        self,
        *,
        offer: WorkerOffer,
        manifest: ModelManifest,
        qualified_context_limit_tokens: int,
        kv_block_size: int,
        context_targets: tuple[int, ...] | None = None,
        maximum_sessions: int = 32,
        span_static_bytes: SpanStaticBytes | None = None,
    ) -> tuple[MemoryPlacementPoint, ...]:
        """Return non-dominated exact memory choices across signed classes.

        This deliberately models only hard resident bytes.  Runtime throughput,
        network boundaries, cached downloads and profile confidence are added
        by the second-pass scorer; they must not weaken the memory invariant.
        """

        if (
            qualified_context_limit_tokens <= 0
            or qualified_context_limit_tokens > manifest.model_max_context_tokens
        ):
            raise ValueError("qualified backend context limit must fit the model contract")
        if kv_block_size <= 0 or maximum_sessions <= 0:
            raise ValueError("KV block size and session bound must be positive")

        targets = manifest.context_classes if context_targets is None else context_targets
        if not targets or any(value <= 0 for value in targets):
            raise ValueError("memory frontier context targets must be positive")
        targets = tuple(sorted(set(targets)))
        if targets[-1] > manifest.model_max_context_tokens:
            raise ValueError("memory frontier target exceeds the model contract")

        points: list[MemoryPlacementPoint] = []
        for context_tokens in targets:
            if context_tokens > qualified_context_limit_tokens:
                continue
            for span, _ in self.feasible_spans(
                offer=offer,
                manifest=manifest,
                context_tokens=context_tokens,
                kv_block_size=kv_block_size,
                span_static_bytes=span_static_bytes,
            ):
                weight_bytes = self._static_bytes(manifest, span, span_static_bytes)
                if weight_bytes is None:
                    continue
                rounded_tokens = (
                    (context_tokens + kv_block_size - 1) // kv_block_size * kv_block_size
                )
                kv_bytes_per_session = rounded_tokens * sum(
                    manifest.kv_bytes_per_token_by_layer[span.start : span.end]
                )
                if kv_bytes_per_session <= 0:
                    continue
                available_kv_bytes = offer.stable_memory_envelope_bytes - weight_bytes
                max_sessions = min(maximum_sessions, available_kv_bytes // kv_bytes_per_session)
                if max_sessions <= 0:
                    continue
                points.append(
                    MemoryPlacementPoint(
                        span=span,
                        context_tokens=context_tokens,
                        weight_bytes=weight_bytes,
                        kv_bytes_per_session=kv_bytes_per_session,
                        max_sessions=max_sessions,
                        memory_headroom_bytes=(
                            available_kv_bytes - max_sessions * kv_bytes_per_session
                        ),
                    )
                )

        frontier = [
            point
            for point in points
            if not any(other.dominates(point) for other in points if other is not point)
        ]
        return tuple(
            sorted(
                frontier,
                key=lambda point: (
                    point.span.start,
                    point.span.end,
                    point.context_tokens,
                    -point.max_sessions,
                ),
            )
        )

    @staticmethod
    def _coverage(
        manifest: ModelManifest,
        leases: tuple[SpanLease, ...],
        *,
        exclude_worker_id: str,
        minimum_context_tokens: int,
        states: frozenset[SpanState] = frozenset({SpanState.READY}),
    ) -> list[int]:
        if minimum_context_tokens <= 0:
            raise ValueError("coverage context must be positive")
        coverage = [0] * manifest.num_layers
        for lease in leases:
            if (
                lease.model_swarm_id != manifest.model_swarm_id
                or lease.worker_id == exclude_worker_id
                or lease.state not in states
                or lease.max_context_tokens < minimum_context_tokens
            ):
                continue
            for layer in range(lease.hosted_span.start, lease.hosted_span.end):
                coverage[layer] += 1
        return coverage

    @staticmethod
    def _session_coverage(
        manifest: ModelManifest,
        leases: tuple[SpanLease, ...],
        *,
        exclude_worker_id: str,
        minimum_context_tokens: int,
        states: frozenset[SpanState],
    ) -> list[int]:
        """Sum exact KV session supply per layer for the Petals-style screen."""

        if minimum_context_tokens <= 0:
            raise ValueError("session coverage context must be positive")
        coverage = [0] * manifest.num_layers
        for lease in leases:
            if (
                lease.model_swarm_id != manifest.model_swarm_id
                or lease.worker_id == exclude_worker_id
                or lease.state not in states
                or lease.max_context_tokens < minimum_context_tokens
            ):
                continue
            for layer in range(lease.hosted_span.start, lease.hosted_span.end):
                coverage[layer] += lease.max_sessions
        return coverage

    @staticmethod
    def _tiebreaker(worker_id: str, span: LayerSpan) -> int:
        digest = hashlib.sha256(
            f"fabi/placement/v3\0{worker_id}\0{span.start}\0{span.end}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "big")

    def _score(
        self,
        *,
        worker_id: str,
        span: LayerSpan,
        base_coverage: list[int],
        demand: CapacityDemandMap,
        prefix_boundaries: frozenset[int],
        suffix_boundaries: frozenset[int],
        model_num_layers: int,
        can_host_frontend: bool,
    ) -> PlacementScore:
        after = [
            count + int(span.start <= layer < span.end) for layer, count in enumerate(base_coverage)
        ]
        deficit_filled = sum(
            weight
            for layer, weight in enumerate(demand.demand_weight_by_layer)
            if span.start <= layer < span.end
            and base_coverage[layer] < demand.desired_replicas_by_layer[layer]
        )
        weighted_coverage = sum(
            min(count, demand.desired_replicas_by_layer[layer]) * weight
            for layer, (count, weight) in enumerate(zip(after, demand.demand_weight_by_layer))
        )
        completes_fixed_route = int(
            span.start in prefix_boundaries and span.end in suffix_boundaries
        )
        fixed_route_progress = max(
            span.end if span.start in prefix_boundaries else 0,
            model_num_layers - span.start if span.end in suffix_boundaries else 0,
        )
        return PlacementScore(
            completes_fixed_route=completes_fixed_route,
            establishes_missing_frontend=int(
                can_host_frontend and span.start == 0 and base_coverage[0] == 0
            ),
            fixed_route_progress=fixed_route_progress,
            minimum_ready_coverage=min(after),
            weighted_deficit_filled=deficit_filled,
            weighted_coverage=weighted_coverage,
            span_length=span.length,
            deterministic_tiebreaker=self._tiebreaker(worker_id, span),
        )

    @staticmethod
    def _fixed_route_boundaries(
        manifest: ModelManifest,
        leases: tuple[SpanLease, ...],
        *,
        exclude_worker_id: str,
        minimum_context_tokens: int,
    ) -> tuple[frozenset[int], frozenset[int]]:
        """Return exact prefix/suffix boundaries reachable through live intents."""

        if minimum_context_tokens <= 0:
            raise ValueError("route-boundary context must be positive")

        spans = tuple(
            lease.hosted_span
            for lease in leases
            if lease.model_swarm_id == manifest.model_swarm_id
            and lease.worker_id != exclude_worker_id
            and lease.state in {SpanState.BUILDING, SpanState.WARMING, SpanState.READY}
            and lease.max_context_tokens >= minimum_context_tokens
        )
        prefix = {0}
        suffix = {manifest.num_layers}
        changed = True
        while changed:
            changed = False
            for span in spans:
                if span.start in prefix and span.end not in prefix:
                    prefix.add(span.end)
                    changed = True
                if span.end in suffix and span.start not in suffix:
                    suffix.add(span.start)
                    changed = True
        return frozenset(prefix), frozenset(suffix)

    @staticmethod
    def _preserves_coverage(
        *,
        current_span: LayerSpan | None,
        candidate: LayerSpan,
        base_coverage: list[int],
    ) -> bool:
        if current_span is None or current_span == candidate:
            return True
        # Current fixed backends unload before loading a different span. The
        # candidate therefore cannot count as coverage during that gap.
        return all(
            base_coverage[layer] > 0 for layer in range(current_span.start, current_span.end)
        )

    def choose(
        self,
        *,
        offer: WorkerOffer,
        manifest: ModelManifest,
        leases: tuple[SpanLease, ...],
        demand: CapacityDemandMap,
        context_tokens: int,
        kv_block_size: int,
        current_span: LayerSpan | None = None,
        current_reservations: int = 0,
        last_moved_at_ms: int | None = None,
        serving_route_exists: bool = False,
        serving_route_survives_movement: bool = False,
        excluded_spans: frozenset[LayerSpan] = frozenset(),
        span_static_bytes: SpanStaticBytes | None = None,
        now_ms: int,
    ) -> PlacementDecision:
        if len(demand.desired_replicas_by_layer) != manifest.num_layers:
            raise ValueError("capacity demand map does not match the model")
        if current_reservations < 0 or now_ms < 0:
            raise ValueError("reservation count and current time must be non-negative")
        feasible = self.feasible_spans(
            offer=offer,
            manifest=manifest,
            context_tokens=context_tokens,
            kv_block_size=kv_block_size,
            span_static_bytes=span_static_bytes,
        )
        if not feasible:
            return PlacementDecision(
                action=PlacementAction.STANDBY,
                span=None,
                required_memory_bytes=0,
                score=None,
                reason="no_exact_span_fits_the_stable_memory_envelope",
                context_tokens=context_tokens,
            )
        if excluded_spans:
            feasible = tuple(item for item in feasible if item[0] not in excluded_spans)
            if not feasible:
                return PlacementDecision(
                    action=PlacementAction.STANDBY,
                    span=None,
                    required_memory_bytes=0,
                    score=None,
                    reason="no_exact_span_fits_local_artifact_storage",
                    context_tokens=context_tokens,
                )

        # BUILDING/WARMING leases are demand intents, like Petals' JOINING
        # modules. They spread simultaneous joins but never make a route
        # executable and never protect a serving worker from a coverage hole.
        planned_coverage = self._coverage(
            manifest,
            leases,
            exclude_worker_id=offer.worker_id,
            minimum_context_tokens=context_tokens,
            states=frozenset({SpanState.BUILDING, SpanState.WARMING, SpanState.READY}),
        )
        ready_coverage = self._coverage(
            manifest,
            leases,
            exclude_worker_id=offer.worker_id,
            minimum_context_tokens=context_tokens,
        )
        prefix_boundaries, suffix_boundaries = self._fixed_route_boundaries(
            manifest,
            leases,
            exclude_worker_id=offer.worker_id,
            minimum_context_tokens=context_tokens,
        )
        candidates = [
            (
                span,
                required,
                self._score(
                    worker_id=offer.worker_id,
                    span=span,
                    base_coverage=planned_coverage,
                    demand=demand,
                    prefix_boundaries=prefix_boundaries,
                    suffix_boundaries=suffix_boundaries,
                    model_num_layers=manifest.num_layers,
                    can_host_frontend=WorkerRole.FRONTEND in offer.supported_roles,
                ),
            )
            for span, required in feasible
            if self._preserves_coverage(
                current_span=current_span,
                candidate=span,
                base_coverage=ready_coverage,
            )
        ]
        if not candidates:
            if current_span is not None:
                current_required = self._required_memory_bytes(
                    manifest,
                    current_span,
                    context_tokens=context_tokens,
                    kv_block_size=kv_block_size,
                    span_static_bytes=span_static_bytes,
                )
                if current_required is None:
                    raise ValueError("current span is unsupported by the execution geometry")
                return PlacementDecision(
                    action=PlacementAction.KEEP,
                    span=current_span,
                    required_memory_bytes=current_required,
                    score=self._score(
                        worker_id=offer.worker_id,
                        span=current_span,
                        base_coverage=planned_coverage,
                        demand=demand,
                        prefix_boundaries=prefix_boundaries,
                        suffix_boundaries=suffix_boundaries,
                        model_num_layers=manifest.num_layers,
                        can_host_frontend=WorkerRole.FRONTEND in offer.supported_roles,
                    ),
                    reason="movement_would_remove_the_last_ready_coverage",
                    context_tokens=context_tokens,
                )
            return PlacementDecision(
                action=PlacementAction.STANDBY,
                span=None,
                required_memory_bytes=0,
                score=None,
                reason="no_candidate_preserves_ready_coverage",
                context_tokens=context_tokens,
            )

        best_span, best_required, best_score = max(
            candidates,
            key=lambda item: item[2].rank(),
        )
        if current_span is None:
            return PlacementDecision(
                action=PlacementAction.JOIN,
                span=best_span,
                required_memory_bytes=best_required,
                score=best_score,
                reason="fills_the_highest_verified_capacity_deficit",
                context_tokens=context_tokens,
            )
        current = next(
            (item for item in candidates if item[0] == current_span),
            None,
        )
        if current is None:
            current_required = self._required_memory_bytes(
                manifest,
                current_span,
                context_tokens=context_tokens,
                kv_block_size=kv_block_size,
                span_static_bytes=span_static_bytes,
            )
            if current_required is None:
                raise ValueError("current span is unsupported by the execution geometry")
            current_score = self._score(
                worker_id=offer.worker_id,
                span=current_span,
                base_coverage=planned_coverage,
                demand=demand,
                prefix_boundaries=prefix_boundaries,
                suffix_boundaries=suffix_boundaries,
                model_num_layers=manifest.num_layers,
                can_host_frontend=WorkerRole.FRONTEND in offer.supported_roles,
            )
        else:
            _, current_required, current_score = current
        if best_span == current_span:
            return PlacementDecision(
                action=PlacementAction.KEEP,
                span=current_span,
                required_memory_bytes=current_required,
                score=current_score,
                reason="current_span_is_still_the_best_stable_choice",
                context_tokens=context_tokens,
            )
        if current_reservations:
            return PlacementDecision(
                action=PlacementAction.KEEP,
                span=current_span,
                required_memory_bytes=current_required,
                score=current_score,
                reason="active_reservations_must_drain_before_movement",
                context_tokens=context_tokens,
            )
        # Petals only refuses a move that would break a swarm which is
        # currently connected. For Fabi, layer coverage alone is not enough:
        # the worker-side controller proves exact routes with compatible
        # context, endpoints and authenticated links (including the decode
        # closure). Applying that guard while the swarm is
        # already disjoint deadlocks duplicate spans in place: no worker may
        # move to fill the missing range because, by definition, there is no
        # complete route to preserve yet.  ``_preserves_coverage`` above still
        # proves that unloading this worker cannot remove the last READY copy
        # of any layer.
        if serving_route_exists and not serving_route_survives_movement:
            return PlacementDecision(
                action=PlacementAction.KEEP,
                span=current_span,
                required_memory_bytes=current_required,
                score=current_score,
                reason="movement_would_remove_the_last_executable_route",
                context_tokens=context_tokens,
            )
        if last_moved_at_ms is not None and now_ms - last_moved_at_ms < self.movement_cooldown_ms:
            return PlacementDecision(
                action=PlacementAction.KEEP,
                span=current_span,
                required_memory_bytes=current_required,
                score=current_score,
                reason="movement_cooldown_has_not_elapsed",
                context_tokens=context_tokens,
            )
        current_value = max(current_score.weighted_deficit_filled, 1.0)
        improvement = (
            best_score.weighted_deficit_filled - current_score.weighted_deficit_filled
        ) / current_value
        if (
            best_score.minimum_ready_coverage <= current_score.minimum_ready_coverage
            and improvement < self.minimum_improvement
        ):
            return PlacementDecision(
                action=PlacementAction.KEEP,
                span=current_span,
                required_memory_bytes=current_required,
                score=current_score,
                reason="durable_gain_is_below_the_movement_threshold",
                context_tokens=context_tokens,
            )
        return PlacementDecision(
            action=PlacementAction.MOVE,
            span=best_span,
            required_memory_bytes=best_required,
            score=best_score,
            reason="coverage_preserved_and_verified_gain_exceeds_hysteresis",
            context_tokens=context_tokens,
        )

    def _context_utility(
        self,
        *,
        offer: WorkerOffer,
        manifest: ModelManifest,
        leases: tuple[SpanLease, ...],
        demand: ContextCapacityDemandMap,
        span: LayerSpan,
        context_tokens: int,
        kv_block_size: int,
        maximum_sessions: int | None = None,
        span_static_bytes: SpanStaticBytes | None = None,
        supply_by_context: Mapping[int, _ContextSupply],
    ) -> ContextPlacementUtility:
        """Score one exact target across all cumulative demand classes."""

        rounded_tokens = (context_tokens + kv_block_size - 1) // kv_block_size * kv_block_size
        kv_bytes_per_session = rounded_tokens * sum(
            manifest.kv_bytes_per_token_by_layer[span.start : span.end]
        )
        static_bytes = self._static_bytes(manifest, span, span_static_bytes)
        available_kv_bytes = (
            0 if static_bytes is None else max(0, offer.stable_memory_envelope_bytes - static_bytes)
        )
        max_sessions = available_kv_bytes // kv_bytes_per_session if kv_bytes_per_session > 0 else 0
        if maximum_sessions is not None:
            if maximum_sessions <= 0:
                raise ValueError("session capacity must be positive")
            max_sessions = min(max_sessions, maximum_sessions)

        weighted_complete_routes = 0.0
        weighted_concurrent_slots = 0.0
        weighted_layer_deficit_filled = 0.0
        for class_demand in demand.classes:
            if class_demand.context_tokens > context_tokens:
                continue
            if (
                class_demand.desired_independent_routes == 0
                and class_demand.desired_concurrent_slots == 0
            ):
                continue
            class_weight = (
                sum(class_demand.demand_weight_by_layer)
                / manifest.num_layers
                * class_demand.confidence
            )
            if class_weight <= 0:
                continue
            supply = supply_by_context[class_demand.context_tokens]
            planned_coverage = supply.layer_coverage
            planned_session_coverage = supply.session_coverage
            completes_route = (
                span.start in supply.prefix_boundaries
                and span.end in supply.suffix_boundaries
            )
            if completes_route:
                replicas_before = min(planned_coverage)
                replicas_after = min(
                    count + int(span.start <= layer < span.end)
                    for layer, count in enumerate(planned_coverage)
                )
                weighted_complete_routes += class_weight * (
                    min(replicas_after, class_demand.desired_independent_routes)
                    - min(replicas_before, class_demand.desired_independent_routes)
                )
                slots_before = min(planned_session_coverage)
                slots_after = min(
                    count + (max_sessions if span.start <= layer < span.end else 0)
                    for layer, count in enumerate(planned_session_coverage)
                )
                weighted_concurrent_slots += class_weight * (
                    min(slots_after, class_demand.desired_concurrent_slots)
                    - min(slots_before, class_demand.desired_concurrent_slots)
                )
            weighted_layer_deficit_filled += (
                class_demand.confidence
                * sum(
                    weight
                    for layer, weight in enumerate(class_demand.demand_weight_by_layer)
                    if span.start <= layer < span.end
                    and planned_coverage[layer] < class_demand.desired_replicas_by_layer[layer]
                )
                / manifest.num_layers
            )

        return ContextPlacementUtility(
            weighted_complete_routes=weighted_complete_routes,
            weighted_concurrent_slots=weighted_concurrent_slots,
            weighted_layer_deficit_filled=weighted_layer_deficit_filled,
            context_tokens=context_tokens,
            span_length=span.length,
            max_sessions=max_sessions,
        )

    @staticmethod
    def _route_capacity(
        *,
        manifest: ModelManifest,
        leases: tuple[SpanLease, ...],
        exclude_worker_id: str,
        minimum_context_tokens: int,
        candidate_span: LayerSpan,
        candidate_sessions: int,
        session_capacity: bool,
    ) -> int:
        """Measure complete fixed-boundary capacity with maintained max-flow.

        Every hosted span is a directed edge between layer boundaries.  For
        redundancy each worker contributes one unit; for concurrency it
        contributes its exact KV session count. BUILDING/WARMING intents are
        included so simultaneous autonomous joins spread instead of all
        choosing the same temporary deficit.
        """

        graph = nx.DiGraph()
        graph.add_nodes_from(range(manifest.num_layers + 1))

        def add_span(span: LayerSpan, capacity: int) -> None:
            edge = (span.start, span.end)
            previous = graph.get_edge_data(*edge, default={}).get("capacity", 0)
            graph.add_edge(*edge, capacity=previous + capacity)

        for lease in leases:
            if (
                lease.model_swarm_id != manifest.model_swarm_id
                or lease.worker_id == exclude_worker_id
                or lease.state not in {SpanState.BUILDING, SpanState.WARMING, SpanState.READY}
                or lease.max_context_tokens < minimum_context_tokens
            ):
                continue
            add_span(lease.hosted_span, lease.max_sessions if session_capacity else 1)
        add_span(candidate_span, candidate_sessions if session_capacity else 1)
        return int(
            nx.maximum_flow_value(
                graph,
                0,
                manifest.num_layers,
                capacity="capacity",
            )
        )

    def _exact_context_utility(
        self,
        *,
        offer: WorkerOffer,
        manifest: ModelManifest,
        leases: tuple[SpanLease, ...],
        demand: ContextCapacityDemandMap,
        span: LayerSpan,
        context_tokens: int,
        approximate: ContextPlacementUtility,
    ) -> ContextPlacementUtility:
        """Replace local route guesses with capped end-to-end capacity."""

        weighted_complete_routes = 0.0
        weighted_concurrent_slots = 0.0
        for class_demand in demand.classes:
            if class_demand.context_tokens > context_tokens:
                continue
            if (
                class_demand.desired_independent_routes == 0
                and class_demand.desired_concurrent_slots == 0
            ):
                continue
            class_weight = (
                sum(class_demand.demand_weight_by_layer)
                / manifest.num_layers
                * class_demand.confidence
            )
            if class_weight <= 0:
                continue
            if class_demand.desired_independent_routes:
                routes = self._route_capacity(
                    manifest=manifest,
                    leases=leases,
                    exclude_worker_id=offer.worker_id,
                    minimum_context_tokens=class_demand.context_tokens,
                    candidate_span=span,
                    candidate_sessions=approximate.max_sessions,
                    session_capacity=False,
                )
                weighted_complete_routes += class_weight * min(
                    routes,
                    class_demand.desired_independent_routes,
                )
            if class_demand.desired_concurrent_slots:
                slots = self._route_capacity(
                    manifest=manifest,
                    leases=leases,
                    exclude_worker_id=offer.worker_id,
                    minimum_context_tokens=class_demand.context_tokens,
                    candidate_span=span,
                    candidate_sessions=approximate.max_sessions,
                    session_capacity=True,
                )
                weighted_concurrent_slots += class_weight * min(
                    slots,
                    class_demand.desired_concurrent_slots,
                )

        return ContextPlacementUtility(
            weighted_complete_routes=weighted_complete_routes,
            weighted_concurrent_slots=weighted_concurrent_slots,
            weighted_layer_deficit_filled=approximate.weighted_layer_deficit_filled,
            context_tokens=context_tokens,
            span_length=span.length,
            max_sessions=approximate.max_sessions,
        )

    def choose_contextual(
        self,
        *,
        offer: WorkerOffer,
        manifest: ModelManifest,
        leases: tuple[SpanLease, ...],
        demand: ContextCapacityDemandMap,
        qualified_context_limit_tokens: int,
        kv_block_size: int,
        current_span: LayerSpan | None = None,
        current_context_tokens: int | None = None,
        current_reservations: int = 0,
        last_moved_at_ms: int | None = None,
        serving_route_exists: bool = False,
        serving_route_survives_movement: bool = False,
        excluded_spans: frozenset[LayerSpan] = frozenset(),
        span_static_bytes: SpanStaticBytes | None = None,
        now_ms: int,
    ) -> tuple[PlacementDecision, ContextPlacementUtility]:
        """Choose a real span *and* context from trusted cumulative demand.

        The legacy single-context scorer remains a bounded first pass for each
        demanded context.  This second pass compares those finalists across
        the worker's memory frontier.  A READY worker is never moved because
        advice disappeared: callers invoke this method only for a valid signed
        snapshot and otherwise keep the last verified target.
        """

        demand.validate_for(manifest, now_ms=now_ms)
        if qualified_context_limit_tokens <= 0:
            raise ValueError("qualified context limit must be positive")
        if (current_span is None) != (current_context_tokens is None):
            raise ValueError("current span and context must be supplied together")

        demanded_classes = tuple(
            item
            for item in demand.classes
            if item.context_tokens <= qualified_context_limit_tokens
            and (item.desired_independent_routes > 0 or item.desired_concurrent_slots > 0)
        )
        if not demanded_classes:
            raise ValueError("context demand snapshot has no locally qualified demand")

        planned_states = frozenset({SpanState.BUILDING, SpanState.WARMING, SpanState.READY})
        supply_by_context: dict[int, _ContextSupply] = {}
        for class_demand in demanded_classes:
            prefix, suffix = self._fixed_route_boundaries(
                manifest,
                leases,
                exclude_worker_id=offer.worker_id,
                minimum_context_tokens=class_demand.context_tokens,
            )
            supply_by_context[class_demand.context_tokens] = _ContextSupply(
                layer_coverage=tuple(
                    self._coverage(
                        manifest,
                        leases,
                        exclude_worker_id=offer.worker_id,
                        minimum_context_tokens=class_demand.context_tokens,
                        states=planned_states,
                    )
                ),
                session_coverage=tuple(
                    self._session_coverage(
                        manifest,
                        leases,
                        exclude_worker_id=offer.worker_id,
                        minimum_context_tokens=class_demand.context_tokens,
                        states=planned_states,
                    )
                ),
                prefix_boundaries=prefix,
                suffix_boundaries=suffix,
            )

        frontier = self.memory_frontier(
            offer=offer,
            manifest=manifest,
            qualified_context_limit_tokens=min(
                qualified_context_limit_tokens,
                manifest.model_max_context_tokens,
            ),
            kv_block_size=kv_block_size,
            context_targets=tuple(item.context_tokens for item in demanded_classes),
            span_static_bytes=span_static_bytes,
        )
        finalists: list[tuple[PlacementDecision, ContextPlacementUtility]] = []
        for point in frontier:
            if point.span in excluded_spans:
                continue
            decision = PlacementDecision(
                action=(PlacementAction.JOIN if current_span is None else PlacementAction.MOVE),
                span=point.span,
                required_memory_bytes=point.required_memory_bytes,
                score=None,
                reason="trusted_context_demand_frontier_candidate",
                context_tokens=point.context_tokens,
            )
            utility = self._context_utility(
                offer=offer,
                manifest=manifest,
                leases=leases,
                demand=demand,
                span=point.span,
                context_tokens=point.context_tokens,
                kv_block_size=kv_block_size,
                maximum_sessions=point.max_sessions,
                span_static_bytes=span_static_bytes,
                supply_by_context=supply_by_context,
            )
            finalists.append((decision, utility))

        if current_span is not None and current_context_tokens is not None:
            current_lease = next(
                (
                    lease
                    for lease in leases
                    if lease.worker_id == offer.worker_id
                    and lease.hosted_span == current_span
                    and lease.state is SpanState.READY
                ),
                None,
            )
            current_utility = self._context_utility(
                offer=offer,
                manifest=manifest,
                leases=leases,
                demand=demand,
                span=current_span,
                context_tokens=current_context_tokens,
                kv_block_size=kv_block_size,
                maximum_sessions=(None if current_lease is None else current_lease.max_sessions),
                span_static_bytes=span_static_bytes,
                supply_by_context=supply_by_context,
            )
            finalists.append(
                (
                    PlacementDecision(
                        action=PlacementAction.KEEP,
                        span=current_span,
                        required_memory_bytes=0,
                        score=None,
                        reason="current_verified_span_context_candidate",
                        context_tokens=current_context_tokens,
                    ),
                    current_utility,
                )
            )

        if not finalists:
            raise ValueError("no demanded span/context target fits the stable memory envelope")

        # Petals' per-layer deficit is an efficient screen for a large
        # frontier.  The bounded finalists are then measured end-to-end so a
        # high-capacity duplicate cannot beat the actual bottleneck span.
        shortlisted = sorted(
            finalists,
            key=lambda item: (
                item[1].rank(),
                -item[0].span.start if item[0].span is not None else 0,
                (
                    self._tiebreaker(offer.worker_id, item[0].span)
                    if item[0].span is not None
                    else 0
                ),
            ),
            reverse=True,
        )[: self.maximum_exact_candidates]
        current_candidate = next(
            (item for item in finalists if item[0].action is PlacementAction.KEEP),
            None,
        )
        if current_candidate is not None and current_candidate not in shortlisted:
            shortlisted.append(current_candidate)
        finalists = [
            (
                decision,
                self._exact_context_utility(
                    offer=offer,
                    manifest=manifest,
                    leases=leases,
                    demand=demand,
                    span=decision.span,
                    context_tokens=utility.context_tokens,
                    approximate=utility,
                ),
            )
            for decision, utility in shortlisted
            if decision.span is not None
        ]

        selected, selected_utility = max(
            finalists,
            key=lambda item: (
                item[1].rank(),
                -item[0].span.start if item[0].span is not None else 0,
                self._tiebreaker(offer.worker_id, item[0].span) if item[0].span is not None else 0,
            ),
        )
        if current_span is None:
            return (
                PlacementDecision(
                    action=PlacementAction.JOIN,
                    span=selected.span,
                    required_memory_bytes=selected.required_memory_bytes,
                    score=selected.score,
                    reason="trusted_context_demand_selected_cold_target",
                    context_tokens=selected.context_tokens,
                ),
                selected_utility,
            )

        assert current_context_tokens is not None
        current_utility = next(
            utility
            for decision, utility in finalists
            if decision.action is PlacementAction.KEEP
            and decision.span == current_span
            and decision.context_tokens == current_context_tokens
        )
        if selected.span == current_span and selected.context_tokens == current_context_tokens:
            return (
                PlacementDecision(
                    action=PlacementAction.KEEP,
                    span=current_span,
                    required_memory_bytes=selected.required_memory_bytes,
                    score=selected.score,
                    reason="current_span_and_context_match_trusted_demand",
                    context_tokens=current_context_tokens,
                ),
                current_utility,
            )
        for class_demand in demanded_classes:
            if class_demand.context_tokens > current_context_tokens:
                continue
            ready_coverage = self._coverage(
                manifest,
                leases,
                exclude_worker_id=offer.worker_id,
                minimum_context_tokens=class_demand.context_tokens,
            )
            if any(
                ready_coverage[layer] == 0 for layer in range(current_span.start, current_span.end)
            ):
                return (
                    PlacementDecision(
                        action=PlacementAction.KEEP,
                        span=current_span,
                        required_memory_bytes=0,
                        score=None,
                        reason="context_movement_would_remove_last_ready_layer_coverage",
                        context_tokens=current_context_tokens,
                    ),
                    current_utility,
                )
        if current_reservations:
            return (
                PlacementDecision(
                    action=PlacementAction.KEEP,
                    span=current_span,
                    required_memory_bytes=0,
                    score=None,
                    reason="active_reservations_must_drain_before_context_movement",
                    context_tokens=current_context_tokens,
                ),
                current_utility,
            )
        if serving_route_exists and not serving_route_survives_movement:
            return (
                PlacementDecision(
                    action=PlacementAction.KEEP,
                    span=current_span,
                    required_memory_bytes=0,
                    score=None,
                    reason="context_movement_would_remove_the_last_executable_route",
                    context_tokens=current_context_tokens,
                ),
                current_utility,
            )
        if last_moved_at_ms is not None and now_ms - last_moved_at_ms < self.movement_cooldown_ms:
            return (
                PlacementDecision(
                    action=PlacementAction.KEEP,
                    span=current_span,
                    required_memory_bytes=0,
                    score=None,
                    reason="context_movement_cooldown_has_not_elapsed",
                    context_tokens=current_context_tokens,
                ),
                current_utility,
            )
        improvement = (selected_utility.value - current_utility.value) / max(
            current_utility.value,
            1.0,
        )
        if (
            selected_utility.rank() <= current_utility.rank()
            or improvement < self.minimum_improvement
        ):
            return (
                PlacementDecision(
                    action=PlacementAction.KEEP,
                    span=current_span,
                    required_memory_bytes=0,
                    score=None,
                    reason="context_demand_gain_is_below_the_movement_threshold",
                    context_tokens=current_context_tokens,
                ),
                current_utility,
            )
        return (
            PlacementDecision(
                action=PlacementAction.MOVE,
                span=selected.span,
                required_memory_bytes=selected.required_memory_bytes,
                score=selected.score,
                reason="trusted_context_demand_selected_new_target",
                context_tokens=selected.context_tokens,
            ),
            selected_utility,
        )
