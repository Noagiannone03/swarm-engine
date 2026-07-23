"""Deterministic Petals-style autonomous span placement for protocol v3."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from swarm_protocol.contracts import (
    LayerSpan,
    ModelManifest,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)


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
    def __call__(self, span: LayerSpan, generation: int) -> None: ...


@dataclass(frozen=True)
class MaterializationSnapshot:
    phase: MaterializationPhase
    generation: int
    current_span: LayerSpan | None
    target_span: LayerSpan | None
    previous_span: LayerSpan | None
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
    ) -> None:
        self._drain = drain
        self._reload_target = reload_target
        self._phase = (
            MaterializationPhase.READY
            if current_span is not None
            else MaterializationPhase.STANDBY
        )
        self._generation = 0
        self._current_span = current_span
        self._target_span: LayerSpan | None = None
        self._previous_span: LayerSpan | None = None
        self._error: str | None = None
        self._lock = threading.RLock()

    def snapshot(self) -> MaterializationSnapshot:
        with self._lock:
            return MaterializationSnapshot(
                phase=self._phase,
                generation=self._generation,
                current_span=self._current_span,
                target_span=self._target_span,
                previous_span=self._previous_span,
                error=self._error,
            )

    def reconcile(self, decision: PlacementDecision) -> MaterializationSnapshot:
        """Advance a placement decision without ever overlapping generations."""

        with self._lock:
            if self._phase is MaterializationPhase.BUILDING:
                if decision.span != self._target_span:
                    raise RuntimeError("cannot replace an in-flight materialization target")
                return self.snapshot()
            if decision.action is PlacementAction.KEEP:
                return self.snapshot()
            if decision.action is PlacementAction.STANDBY:
                if self._current_span is not None:
                    raise RuntimeError("a serving worker cannot enter standby without a safe target")
                self._phase = MaterializationPhase.STANDBY
                return self.snapshot()
            if decision.span is None:
                raise ValueError("join and move decisions require a target span")
            if (
                decision.action is PlacementAction.JOIN
                and self._current_span is not None
            ):
                raise RuntimeError("a serving worker cannot execute a join decision")
            if (
                decision.action is PlacementAction.MOVE
                and self._current_span is None
            ):
                raise RuntimeError("a standby worker cannot execute a move decision")

            self._target_span = decision.span
            self._previous_span = self._current_span
            self._error = None
            if self._current_span is not None:
                self._phase = MaterializationPhase.DRAINING
                if self._drain.begin_drain() > 0:
                    return self.snapshot()
            self._start_reload_locked(decision.span)
            return self.snapshot()

    def continue_after_drain(self) -> MaterializationSnapshot:
        with self._lock:
            if self._phase is not MaterializationPhase.DRAINING:
                return self.snapshot()
            if self._drain.draining_reservations() > 0:
                return self.snapshot()
            assert self._target_span is not None
            self._start_reload_locked(self._target_span)
            return self.snapshot()

    def _start_reload_locked(self, span: LayerSpan) -> None:
        self._generation += 1
        generation = self._generation
        self._phase = MaterializationPhase.BUILDING
        try:
            self._reload_target(span, generation)
        except Exception as exc:
            self._phase = MaterializationPhase.FAILED
            self._error = f"{type(exc).__name__}: {exc}"[:256]
            if self._previous_span == self._current_span and self._current_span is not None:
                self._drain.cancel_drain()
                self._phase = MaterializationPhase.READY
            raise
        self._current_span = None

    def mark_ready(self, *, span: LayerSpan, generation: int) -> MaterializationSnapshot:
        """Accept only the verified completion for the current reload generation."""

        with self._lock:
            if (
                self._phase is not MaterializationPhase.BUILDING
                or generation != self._generation
                or span != self._target_span
            ):
                raise RuntimeError("stale or mismatched materialization completion")
            self._current_span = span
            self._target_span = None
            self._previous_span = None
            self._phase = MaterializationPhase.READY
            self._error = None
            self._drain.finish_drain()
            return self.snapshot()

    def mark_failed(self, *, generation: int, error: Exception) -> MaterializationSnapshot:
        with self._lock:
            if generation != self._generation or self._phase is not MaterializationPhase.BUILDING:
                raise RuntimeError("stale materialization failure")
            self._phase = MaterializationPhase.FAILED
            self._error = f"{type(error).__name__}: {error}"[:256]
            if self._previous_span is not None:
                rollback_span = self._previous_span
                self._target_span = rollback_span
                try:
                    self._start_reload_locked(rollback_span)
                except Exception:
                    # ``_start_reload_locked`` records the synchronous failure.
                    pass
            return self.snapshot()


class CapacityDemandMap(BaseModel):
    """Desired independent READY replicas and relative demand per layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    desired_replicas_by_layer: tuple[int, ...]
    demand_weight_by_layer: tuple[float, ...]

    @model_validator(mode="after")
    def validate_shape(self) -> "CapacityDemandMap":
        if not self.desired_replicas_by_layer:
            raise ValueError("capacity demand map must contain model layers")
        if len(self.demand_weight_by_layer) != len(self.desired_replicas_by_layer):
            raise ValueError("capacity demand arrays must have identical lengths")
        if any(value <= 0 for value in self.desired_replicas_by_layer):
            raise ValueError("desired replica counts must be positive")
        if any(value <= 0 for value in self.demand_weight_by_layer):
            raise ValueError("demand weights must be positive")
        return self

    @classmethod
    def uniform(
        cls,
        num_layers: int,
        *,
        desired_replicas: int = 2,
    ) -> "CapacityDemandMap":
        if num_layers <= 0 or desired_replicas <= 0:
            raise ValueError("layer and replica counts must be positive")
        return cls(
            desired_replicas_by_layer=(desired_replicas,) * num_layers,
            demand_weight_by_layer=(1.0,) * num_layers,
        )


@dataclass(frozen=True)
class PlacementScore:
    minimum_ready_coverage: int
    weighted_deficit_filled: float
    weighted_coverage: float
    span_length: int
    deterministic_tiebreaker: int

    def rank(self) -> tuple[int, float, float, int, int]:
        return (
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
    ) -> None:
        if not 0 <= minimum_improvement < 1:
            raise ValueError("minimum improvement must be in [0, 1)")
        if movement_cooldown_ms < 0 or maximum_candidates <= 0:
            raise ValueError("cooldown must be non-negative and candidate bound positive")
        self.minimum_improvement = minimum_improvement
        self.movement_cooldown_ms = movement_cooldown_ms
        self.maximum_candidates = maximum_candidates

    @staticmethod
    def _required_memory_bytes(
        manifest: ModelManifest,
        span: LayerSpan,
        *,
        context_tokens: int,
        kv_block_size: int,
    ) -> int:
        if context_tokens <= 0 or kv_block_size <= 0:
            raise ValueError("context and KV block size must be positive")
        rounded_tokens = (context_tokens + kv_block_size - 1) // kv_block_size * kv_block_size
        kv_bytes = rounded_tokens * sum(manifest.kv_bytes_per_token_by_layer[span.start : span.end])
        return manifest.weight_bytes(span) + kv_bytes

    def feasible_spans(
        self,
        *,
        offer: WorkerOffer,
        manifest: ModelManifest,
        context_tokens: int,
        kv_block_size: int,
    ) -> tuple[tuple[LayerSpan, int], ...]:
        """Enumerate every exact contiguous span this worker can materialize."""

        if not manifest.weight_bytes_by_layer:
            raise ValueError("autonomous placement requires exact weight byte geometry")
        granularity = offer.execution_granularity_layers
        result: list[tuple[LayerSpan, int]] = []
        attempts = 0
        for start in range(0, manifest.num_layers, granularity):
            for end in range(start + granularity, manifest.num_layers + 1, granularity):
                attempts += 1
                if attempts > self.maximum_candidates:
                    raise ValueError("placement candidate count exceeds the configured bound")
                if (start == 0 or end == manifest.num_layers) and (
                    WorkerRole.FRONTEND not in offer.supported_roles
                ):
                    continue
                span = LayerSpan(start=start, end=end)
                required = self._required_memory_bytes(
                    manifest,
                    span,
                    context_tokens=context_tokens,
                    kv_block_size=kv_block_size,
                )
                if required <= offer.stable_memory_envelope_bytes:
                    result.append((span, required))
                # Every additional layer is positive, except that endpoint
                # ownership appears only at the final boundary. Do not break:
                # tied endpoints can make the full-model delta non-monotonic.
        return tuple(result)

    @staticmethod
    def _coverage(
        manifest: ModelManifest,
        leases: tuple[SpanLease, ...],
        *,
        exclude_worker_id: str,
    ) -> list[int]:
        coverage = [0] * manifest.num_layers
        for lease in leases:
            if (
                lease.model_swarm_id != manifest.model_swarm_id
                or lease.worker_id == exclude_worker_id
                or lease.state != SpanState.READY
            ):
                continue
            for layer in range(lease.hosted_span.start, lease.hosted_span.end):
                coverage[layer] += 1
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
        return PlacementScore(
            minimum_ready_coverage=min(after),
            weighted_deficit_filled=deficit_filled,
            weighted_coverage=weighted_coverage,
            span_length=span.length,
            deterministic_tiebreaker=self._tiebreaker(worker_id, span),
        )

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
        )
        if not feasible:
            return PlacementDecision(
                action=PlacementAction.STANDBY,
                span=None,
                required_memory_bytes=0,
                score=None,
                reason="no_exact_span_fits_the_stable_memory_envelope",
            )

        base_coverage = self._coverage(
            manifest,
            leases,
            exclude_worker_id=offer.worker_id,
        )
        candidates = [
            (
                span,
                required,
                self._score(
                    worker_id=offer.worker_id,
                    span=span,
                    base_coverage=base_coverage,
                    demand=demand,
                ),
            )
            for span, required in feasible
            if self._preserves_coverage(
                current_span=current_span,
                candidate=span,
                base_coverage=base_coverage,
            )
        ]
        if not candidates:
            if current_span is not None:
                current_required = self._required_memory_bytes(
                    manifest,
                    current_span,
                    context_tokens=context_tokens,
                    kv_block_size=kv_block_size,
                )
                return PlacementDecision(
                    action=PlacementAction.KEEP,
                    span=current_span,
                    required_memory_bytes=current_required,
                    score=self._score(
                        worker_id=offer.worker_id,
                        span=current_span,
                        base_coverage=base_coverage,
                        demand=demand,
                    ),
                    reason="movement_would_remove_the_last_ready_coverage",
                )
            return PlacementDecision(
                action=PlacementAction.STANDBY,
                span=None,
                required_memory_bytes=0,
                score=None,
                reason="no_candidate_preserves_ready_coverage",
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
            )
            current_score = self._score(
                worker_id=offer.worker_id,
                span=current_span,
                base_coverage=base_coverage,
                demand=demand,
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
            )
        if current_reservations:
            return PlacementDecision(
                action=PlacementAction.KEEP,
                span=current_span,
                required_memory_bytes=current_required,
                score=current_score,
                reason="active_reservations_must_drain_before_movement",
            )
        if last_moved_at_ms is not None and now_ms - last_moved_at_ms < self.movement_cooldown_ms:
            return PlacementDecision(
                action=PlacementAction.KEEP,
                span=current_span,
                required_memory_bytes=current_required,
                score=current_score,
                reason="movement_cooldown_has_not_elapsed",
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
            )
        return PlacementDecision(
            action=PlacementAction.MOVE,
            span=best_span,
            required_memory_bytes=best_required,
            score=best_score,
            reason="coverage_preserved_and_verified_gain_exceeds_hysteresis",
        )
