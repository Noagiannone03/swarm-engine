"""Context-aware demand and exact local capacity primitives for protocol v3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from swarm_protocol.contracts import LayerSpan, ModelManifest, SpanLease, SpanState


class CapacityDemandMap(BaseModel):
    """Legacy single-context layer demand used by the active V3 policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    desired_replicas_by_layer: tuple[int, ...]
    demand_weight_by_layer: tuple[float, ...]

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
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
    ) -> Self:
        if num_layers <= 0 or desired_replicas <= 0:
            raise ValueError("layer and replica counts must be positive")
        return cls(
            desired_replicas_by_layer=(desired_replicas,) * num_layers,
            demand_weight_by_layer=(1.0,) * num_layers,
        )


class ContextClassDemand(BaseModel):
    """Bounded aggregate demand for one cumulative context service class.

    This is placement telemetry, never raw prompts or per-user data. A route
    qualified for a larger class can also serve every smaller class; the
    multi-class scorer is therefore responsible for cumulative accounting.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_tokens: Annotated[int, Field(gt=0)]
    desired_independent_routes: Annotated[int, Field(ge=0)]
    desired_concurrent_slots: Annotated[int, Field(ge=0)] = 0
    desired_replicas_by_layer: tuple[int, ...]
    demand_weight_by_layer: tuple[float, ...]
    admitted_requests_per_minute: Annotated[float, Field(ge=0)] = 0.0
    queued_requests: Annotated[int, Field(ge=0)] = 0
    no_route_rejections: Annotated[int, Field(ge=0)] = 0
    p95_service_time_ms: Annotated[float, Field(ge=0)] = 0.0
    confidence: Annotated[float, Field(ge=0, le=1)] = 1.0

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if not self.desired_replicas_by_layer:
            raise ValueError("context demand class must contain model layers")
        if len(self.demand_weight_by_layer) != len(self.desired_replicas_by_layer):
            raise ValueError("context demand arrays must have identical lengths")
        if any(value < 0 for value in self.desired_replicas_by_layer):
            raise ValueError("desired replica counts must be non-negative")
        if any(not math.isfinite(value) or value < 0 for value in self.demand_weight_by_layer):
            raise ValueError("demand weights must be finite and non-negative")
        if not all(
            math.isfinite(value)
            for value in (
                self.admitted_requests_per_minute,
                self.p95_service_time_ms,
                self.confidence,
            )
        ):
            raise ValueError("context demand telemetry must be finite")
        if (
            self.desired_independent_routes or self.desired_concurrent_slots
        ) and not any(self.desired_replicas_by_layer):
            raise ValueError("a routed context class must request layer coverage")
        if any(self.desired_replicas_by_layer) and not any(self.demand_weight_by_layer):
            raise ValueError("requested layer coverage must carry a demand weight")
        return self

    def layer_screen(self) -> CapacityDemandMap:
        """Return the cheap Petals-style layer screen for this class."""

        if any(value <= 0 for value in self.desired_replicas_by_layer):
            raise ValueError("legacy layer screen cannot express zero-replica layers")
        if any(value <= 0 for value in self.demand_weight_by_layer):
            raise ValueError("legacy layer screen cannot express zero-weight layers")
        return CapacityDemandMap(
            desired_replicas_by_layer=self.desired_replicas_by_layer,
            demand_weight_by_layer=self.demand_weight_by_layer,
        )


class ContextCapacityDemandMap(BaseModel):
    """Versioned, expiring demand advice for one model and network region."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    demand_version: int = 1
    model_swarm_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    region_id: Annotated[str, Field(min_length=1, max_length=128)]
    issued_at_ms: Annotated[int, Field(ge=0)]
    expires_at_ms: Annotated[int, Field(gt=0)]
    classes: Annotated[tuple[ContextClassDemand, ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.demand_version != 1:
            raise ValueError("unsupported context demand version")
        if self.expires_at_ms <= self.issued_at_ms:
            raise ValueError("context demand snapshot must expire after issue")
        tokens = tuple(item.context_tokens for item in self.classes)
        if tokens != tuple(sorted(set(tokens))):
            raise ValueError("context demand classes must be strictly increasing and unique")
        widths = {len(item.desired_replicas_by_layer) for item in self.classes}
        if len(widths) != 1:
            raise ValueError("all context demand classes must describe the same model layers")
        return self

    def validate_for(self, manifest: ModelManifest, *, now_ms: int) -> None:
        """Fail closed when advice targets another contract or is stale."""

        if now_ms < 0:
            raise ValueError("current time must be non-negative")
        if self.model_swarm_id != manifest.model_swarm_id:
            raise ValueError("context demand snapshot targets another model swarm")
        if tuple(item.context_tokens for item in self.classes) != manifest.context_classes:
            raise ValueError("context demand classes do not match the signed model contract")
        if len(self.classes[0].desired_replicas_by_layer) != manifest.num_layers:
            raise ValueError("context demand snapshot does not match the model layer count")
        if now_ms >= self.expires_at_ms:
            raise ValueError("context demand snapshot has expired")

    def class_for(self, required_tokens: int) -> ContextClassDemand:
        """Return the smallest advertised class able to serve the request."""

        if required_tokens <= 0:
            raise ValueError("required context tokens must be positive")
        for item in self.classes:
            if item.context_tokens >= required_tokens:
                return item
        raise ValueError("required context exceeds the demand snapshot")


@dataclass(frozen=True)
class MemoryPlacementPoint:
    """One exact local span/context trade-off before runtime benchmarking."""

    span: LayerSpan
    context_tokens: int
    weight_bytes: int
    kv_bytes_per_session: int
    max_sessions: int
    memory_headroom_bytes: int

    @property
    def required_memory_bytes(self) -> int:
        return self.weight_bytes + self.kv_bytes_per_session

    def dominates(self, other: "MemoryPlacementPoint") -> bool:
        """Whether this point is never worse for the same coverage obligation.

        Different, non-containing layer ranges are intentionally incomparable:
        their usefulness depends on the distributed coverage graph.
        """

        covers = self.span.start <= other.span.start and self.span.end >= other.span.end
        no_worse = (
            covers
            and self.context_tokens >= other.context_tokens
            and self.max_sessions >= other.max_sessions
            and self.required_memory_bytes <= other.required_memory_bytes
        )
        strictly_better = (
            self.span != other.span
            or self.context_tokens > other.context_tokens
            or self.max_sessions > other.max_sessions
            or self.required_memory_bytes < other.required_memory_bytes
        )
        return no_worse and strictly_better


def cumulative_context_coverage(
    manifest: ModelManifest,
    leases: tuple[SpanLease, ...],
    *,
    exclude_worker_id: str = "",
    states: frozenset[SpanState] = frozenset({SpanState.READY}),
) -> dict[int, tuple[int, ...]]:
    """Count layer coverage for every signed cumulative context class."""

    coverage = {context: [0] * manifest.num_layers for context in manifest.context_classes}
    for lease in leases:
        if (
            lease.model_swarm_id != manifest.model_swarm_id
            or lease.worker_id == exclude_worker_id
            or lease.state not in states
        ):
            continue
        for context_tokens, by_layer in coverage.items():
            if lease.max_context_tokens < context_tokens:
                continue
            for layer in range(lease.hosted_span.start, lease.hosted_span.end):
                by_layer[layer] += 1
    return {context: tuple(by_layer) for context, by_layer in coverage.items()}
