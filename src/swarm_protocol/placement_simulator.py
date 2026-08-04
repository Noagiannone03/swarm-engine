"""Offline evaluation helpers for context-aware autonomous placement.

The simulator is deliberately outside the product control loop.  It measures
complete pipeline capacity with NetworkX max-flow so placement heuristics can
be compared against reproducible graphs without making a central solver an
authority over live workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from swarm_protocol.contracts import ModelManifest
from swarm_protocol.context_placement import (
    ContextCapacityDemandMap,
    MemoryPlacementPoint,
)


@dataclass(frozen=True)
class SimulatedPlacement:
    worker_id: str
    point: MemoryPlacementPoint

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("simulated placement requires a worker id")


@dataclass(frozen=True)
class ContextServiceCapacity:
    context_tokens: int
    concurrent_service_slots: int
    independent_worker_routes: int


@dataclass(frozen=True)
class CandidatePotential:
    """Explainable shadow score for one local placement option."""

    weighted_independent_route_gain: float
    weighted_concurrent_slot_gain: float
    weighted_layer_deficit_filled: float
    context_tokens: int
    span_length: int
    max_sessions: int

    @property
    def demand_gain(self) -> float:
        # Completing a route is more valuable than covering an isolated layer,
        # but a sufficiently scarce long class can still guide cold bootstrap
        # before its other stages have arrived.
        return (
            2.0 * self.weighted_independent_route_gain
            + self.weighted_concurrent_slot_gain
            + self.weighted_layer_deficit_filled
        )

    def rank(self) -> tuple[float, float, float, float, int, int, int]:
        return (
            self.demand_gain,
            self.weighted_independent_route_gain,
            self.weighted_concurrent_slot_gain,
            self.weighted_layer_deficit_filled,
            self.context_tokens,
            self.span_length,
            self.max_sessions,
        )


def _maximum_flow_value(
    *,
    num_layers: int,
    placements: tuple[SimulatedPlacement, ...],
    context_tokens: int,
    capacity_from_point: str,
) -> int:
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - exercised by minimal wheels
        raise RuntimeError(
            "placement simulation requires the optional 'simulation' dependencies"
        ) from exc

    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_layers + 1))
    for placement in placements:
        point = placement.point
        if point.context_tokens < context_tokens:
            continue
        edge = (point.span.start, point.span.end)
        capacity = 1 if capacity_from_point == "worker" else point.max_sessions
        previous = graph.get_edge_data(*edge, default={}).get("capacity", 0)
        graph.add_edge(*edge, capacity=previous + capacity)
    if not graph.edges:
        return 0
    return int(nx.maximum_flow_value(graph, 0, num_layers, capacity="capacity"))


def evaluate_context_service(
    manifest: ModelManifest,
    placements: tuple[SimulatedPlacement, ...],
) -> tuple[ContextServiceCapacity, ...]:
    """Measure complete-route supply for every signed cumulative class.

    ``concurrent_service_slots`` includes each worker's exact KV concurrency.
    ``independent_worker_routes`` caps every worker at one and therefore tracks
    route redundancy rather than counting several sessions on one failure
    domain as several independent routes.
    """

    worker_ids = [placement.worker_id for placement in placements]
    if len(worker_ids) != len(set(worker_ids)):
        raise ValueError("a simulated worker can materialize only one placement point")
    for placement in placements:
        if placement.point.span.end > manifest.num_layers:
            raise ValueError("simulated placement exceeds the model layer graph")
        if placement.point.context_tokens not in manifest.context_classes:
            raise ValueError("simulated placement uses an unsigned context class")

    return tuple(
        ContextServiceCapacity(
            context_tokens=context_tokens,
            concurrent_service_slots=_maximum_flow_value(
                num_layers=manifest.num_layers,
                placements=placements,
                context_tokens=context_tokens,
                capacity_from_point="sessions",
            ),
            independent_worker_routes=_maximum_flow_value(
                num_layers=manifest.num_layers,
                placements=placements,
                context_tokens=context_tokens,
                capacity_from_point="worker",
            ),
        )
        for context_tokens in manifest.context_classes
    )


def score_candidate_potential(
    manifest: ModelManifest,
    demand: ContextCapacityDemandMap,
    placements: tuple[SimulatedPlacement, ...],
    candidate: SimulatedPlacement,
    *,
    now_ms: int,
) -> CandidatePotential:
    """Score route gain plus weighted bootstrap progress for shadow tests."""

    demand.validate_for(manifest, now_ms=now_ms)
    if any(item.worker_id == candidate.worker_id for item in placements):
        raise ValueError("candidate worker already has a simulated placement")

    before = {
        item.context_tokens: item for item in evaluate_context_service(manifest, placements)
    }
    after = {
        item.context_tokens: item
        for item in evaluate_context_service(manifest, (*placements, candidate))
    }

    coverage: dict[int, list[int]] = {
        context_tokens: [0] * manifest.num_layers
        for context_tokens in manifest.context_classes
    }
    for placement in placements:
        for context_tokens, by_layer in coverage.items():
            if placement.point.context_tokens < context_tokens:
                continue
            for layer in range(placement.point.span.start, placement.point.span.end):
                by_layer[layer] += 1

    route_gain = 0.0
    slot_gain = 0.0
    deficit_filled = 0.0
    for class_demand in demand.classes:
        class_weight = (
            sum(class_demand.demand_weight_by_layer)
            / len(class_demand.demand_weight_by_layer)
            * class_demand.confidence
        )
        context_tokens = class_demand.context_tokens
        before_capacity = before[context_tokens]
        after_capacity = after[context_tokens]
        route_gain += class_weight * (
            min(
                after_capacity.independent_worker_routes,
                class_demand.desired_independent_routes,
            )
            - min(
                before_capacity.independent_worker_routes,
                class_demand.desired_independent_routes,
            )
        )
        slot_gain += class_weight * (
            min(after_capacity.concurrent_service_slots, class_demand.desired_concurrent_slots)
            - min(before_capacity.concurrent_service_slots, class_demand.desired_concurrent_slots)
        )
        if candidate.point.context_tokens < context_tokens:
            continue
        per_layer_gain = sum(
            weight
            for layer, weight in enumerate(class_demand.demand_weight_by_layer)
            if candidate.point.span.start <= layer < candidate.point.span.end
            and coverage[context_tokens][layer]
            < class_demand.desired_replicas_by_layer[layer]
        )
        deficit_filled += class_demand.confidence * per_layer_gain / manifest.num_layers

    return CandidatePotential(
        weighted_independent_route_gain=route_gain,
        weighted_concurrent_slot_gain=slot_gain,
        weighted_layer_deficit_filled=deficit_filled,
        context_tokens=candidate.point.context_tokens,
        span_length=candidate.point.span.length,
        max_sessions=candidate.point.max_sessions,
    )


def build_oracle_scenario(
    manifest: ModelManifest,
    demand: ContextCapacityDemandMap,
    worker_frontiers: Mapping[str, tuple[MemoryPlacementPoint, ...]],
    *,
    now_ms: int,
    weight_scale: int = 1_000,
) -> dict[str, Any]:
    """Serialize exact frontiers for the protobuf-isolated CP-SAT oracle."""

    demand.validate_for(manifest, now_ms=now_ms)
    if weight_scale <= 0:
        raise ValueError("oracle weight scale must be positive")

    workers = []
    for worker_id in sorted(worker_frontiers):
        if not worker_id:
            raise ValueError("oracle worker ids must be non-empty")
        points = worker_frontiers[worker_id]
        options = []
        option_ids: set[str] = set()
        for point in points:
            if point.span.end > manifest.num_layers:
                raise ValueError("oracle point exceeds the model layer graph")
            if point.context_tokens not in manifest.context_classes:
                raise ValueError("oracle point uses an unsigned context class")
            option_id = (
                f"l{point.span.start}-{point.span.end}"
                f"-c{point.context_tokens}-s{point.max_sessions}"
            )
            if option_id in option_ids:
                raise ValueError("oracle frontier contains duplicate placement options")
            option_ids.add(option_id)
            options.append(
                {
                    "option_id": option_id,
                    "start_layer": point.span.start,
                    "end_layer": point.span.end,
                    "context_tokens": point.context_tokens,
                    "max_sessions": point.max_sessions,
                }
            )
        workers.append({"worker_id": worker_id, "options": options})

    demands = []
    for item in demand.classes:
        class_weight = (
            sum(item.demand_weight_by_layer)
            / len(item.demand_weight_by_layer)
            * item.confidence
        )
        target_slots = max(item.desired_independent_routes, item.desired_concurrent_slots)
        demands.append(
            {
                "context_tokens": item.context_tokens,
                "target_slots": target_slots,
                "weight": max(1, round(class_weight * weight_scale)),
            }
        )

    return {
        "num_layers": manifest.num_layers,
        "context_classes": list(manifest.context_classes),
        "demands": demands,
        "workers": workers,
    }
