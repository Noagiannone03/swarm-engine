"""Offline evaluation helpers for context-aware autonomous placement.

The simulator is deliberately outside the product control loop.  It measures
complete pipeline capacity with NetworkX max-flow so placement heuristics can
be compared against reproducible graphs without making a central solver an
authority over live workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from swarm_protocol.contracts import BackendKind, ModelManifest, WorkerOffer, WorkerRole
from swarm_protocol.context_placement import (
    ContextCapacityDemandMap,
    MemoryPlacementPoint,
)
from swarm_protocol.placement import AutonomousPlacementPolicy


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


@dataclass(frozen=True)
class GreedyPlacementResult:
    placements: tuple[SimulatedPlacement, ...]
    service: tuple[ContextServiceCapacity, ...]


@dataclass(frozen=True)
class SyntheticWorkerEnvelope:
    """One explicit stable envelope for deterministic offline populations.

    This is intentionally not physical RAM. Benchmarks must feed the same
    post-OS, post-application stable envelope that a real worker would publish.
    """

    worker_id: str
    stable_memory_envelope_bytes: int
    qualified_context_limit_tokens: int
    backend: BackendKind = BackendKind.MLX
    can_host_frontend: bool = True
    execution_granularity_layers: int = 1

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("synthetic worker id must be non-empty")
        if self.stable_memory_envelope_bytes <= 0:
            raise ValueError("synthetic stable memory envelope must be positive")
        if self.qualified_context_limit_tokens <= 0:
            raise ValueError("synthetic qualified context limit must be positive")
        if self.execution_granularity_layers <= 0:
            raise ValueError("synthetic execution granularity must be positive")


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


def _placement_context_coverage(
    manifest: ModelManifest,
    placements: tuple[SimulatedPlacement, ...],
) -> dict[int, list[int]]:
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
    return coverage


def _weighted_layer_deficit_from_coverage(
    manifest: ModelManifest,
    demand: ContextCapacityDemandMap,
    coverage: Mapping[int, list[int]],
    point: MemoryPlacementPoint,
) -> float:
    deficit_filled = 0.0
    for class_demand in demand.classes:
        if point.context_tokens < class_demand.context_tokens:
            continue
        deficit_filled += class_demand.confidence * sum(
            weight
            for layer, weight in enumerate(class_demand.demand_weight_by_layer)
            if point.span.start <= layer < point.span.end
            and coverage[class_demand.context_tokens][layer]
            < class_demand.desired_replicas_by_layer[layer]
        ) / manifest.num_layers
    return deficit_filled


def weighted_layer_deficit_score(
    manifest: ModelManifest,
    demand: ContextCapacityDemandMap,
    placements: tuple[SimulatedPlacement, ...],
    point: MemoryPlacementPoint,
) -> float:
    """Cheap Petals-style first-pass score across cumulative classes."""

    return _weighted_layer_deficit_from_coverage(
        manifest,
        demand,
        _placement_context_coverage(manifest, placements),
        point,
    )


def build_synthetic_memory_frontiers(
    manifest: ModelManifest,
    workers: tuple[SyntheticWorkerEnvelope, ...],
    *,
    kv_block_size: int,
    maximum_sessions: int = 32,
) -> dict[str, tuple[MemoryPlacementPoint, ...]]:
    """Build exact local frontiers from explicit stable-memory scenarios."""

    if len({worker.worker_id for worker in workers}) != len(workers):
        raise ValueError("synthetic worker ids must be unique")
    policy = AutonomousPlacementPolicy()
    result: dict[str, tuple[MemoryPlacementPoint, ...]] = {}
    cached_frontiers: dict[
        tuple[int, int, BackendKind, bool, int],
        tuple[MemoryPlacementPoint, ...],
    ] = {}
    for worker in workers:
        cache_key = (
            worker.stable_memory_envelope_bytes,
            worker.qualified_context_limit_tokens,
            worker.backend,
            worker.can_host_frontend,
            worker.execution_granularity_layers,
        )
        cached = cached_frontiers.get(cache_key)
        if cached is not None:
            result[worker.worker_id] = cached
            continue
        roles = {WorkerRole.EXECUTOR}
        if worker.can_host_frontend:
            roles.add(WorkerRole.FRONTEND)
        offer = WorkerOffer(
            worker_id=worker.worker_id,
            endpoint_id=f"simulation-{worker.worker_id}",
            runtime_version="simulation",
            platform="simulation",
            backend=worker.backend,
            stable_memory_envelope_bytes=worker.stable_memory_envelope_bytes,
            execution_granularity_layers=worker.execution_granularity_layers,
            supported_roles=frozenset(roles),
            offer_seq=0,
            issued_at_ms=0,
            expires_at_ms=1,
        )
        frontier = policy.memory_frontier(
            offer=offer,
            manifest=manifest,
            qualified_context_limit_tokens=worker.qualified_context_limit_tokens,
            kv_block_size=kv_block_size,
            maximum_sessions=maximum_sessions,
        )
        cached_frontiers[cache_key] = frontier
        result[worker.worker_id] = frontier
    return result


def maximum_span_placements(
    manifest: ModelManifest,
    demand: ContextCapacityDemandMap,
    worker_frontiers: Mapping[str, tuple[MemoryPlacementPoint, ...]],
    *,
    now_ms: int,
) -> GreedyPlacementResult:
    """Baseline that maximizes resident layers before considering demand."""

    demand.validate_for(manifest, now_ms=now_ms)
    placements: list[SimulatedPlacement] = []
    for worker_id, frontier in worker_frontiers.items():
        if not worker_id:
            raise ValueError("simulated worker ids must be non-empty")
        if not frontier:
            continue
        current = tuple(placements)
        coverage = _placement_context_coverage(manifest, current)
        point = max(
            frontier,
            key=lambda candidate: (
                candidate.span.length,
                _weighted_layer_deficit_from_coverage(
                    manifest, demand, coverage, candidate
                ),
                candidate.context_tokens,
                candidate.max_sessions,
                -candidate.span.start,
            ),
        )
        placements.append(SimulatedPlacement(worker_id=worker_id, point=point))

    result = tuple(placements)
    return GreedyPlacementResult(
        placements=result,
        service=evaluate_context_service(manifest, result),
    )


def petals_fixed_context_placements(
    manifest: ModelManifest,
    demand: ContextCapacityDemandMap,
    worker_frontiers: Mapping[str, tuple[MemoryPlacementPoint, ...]],
    *,
    context_tokens: int,
    now_ms: int,
) -> GreedyPlacementResult:
    """Petals-style single-context baseline over exact local frontiers.

    Petals determines a server's block count from local capacity, then places
    that fixed-length span where supply is weakest. Fabi's input/output
    endpoints make feasible length depend on position, so this adaptation
    compares the resulting global minimum-supply vector across all feasible
    spans. It deliberately does not claim to model Petals throughput
    calibration or routing latency.
    """

    demand.validate_for(manifest, now_ms=now_ms)
    if context_tokens not in manifest.context_classes:
        raise ValueError("Petals baseline requires a signed context class")
    placements: list[SimulatedPlacement] = []
    for worker_id, frontier in worker_frontiers.items():
        if not worker_id:
            raise ValueError("simulated worker ids must be non-empty")
        eligible = tuple(
            point for point in frontier if point.context_tokens == context_tokens
        )
        if not eligible:
            continue
        coverage = _placement_context_coverage(manifest, tuple(placements))[context_tokens]

        # Petals compares sorted per-layer throughputs lexicographically. For
        # variable endpoint-aware lengths, compare the complete supply vector
        # after each join: this first improves the global bottleneck, then the
        # next weakest layer. Unit capacity is used because this memory-only
        # scenario has no measured throughput profile yet.
        point = max(
            eligible,
            key=lambda candidate: (
                tuple(
                    sorted(
                        value
                        + int(candidate.span.start <= layer < candidate.span.end)
                        for layer, value in enumerate(coverage)
                    )
                ),
                candidate.span.length,
                candidate.max_sessions,
                -candidate.span.start,
            ),
        )
        placements.append(SimulatedPlacement(worker_id=worker_id, point=point))

    result = tuple(placements)
    return GreedyPlacementResult(
        placements=result,
        service=evaluate_context_service(manifest, result),
    )


def _largest_remainder_layer_allocation(
    total_layers: int,
    memory_bytes: tuple[int, ...],
) -> tuple[int, ...]:
    """Mirror Exo's proportional allocation without importing its runtime."""

    if not memory_bytes:
        raise ValueError("Exo baseline requires at least one worker")
    if any(value <= 0 for value in memory_bytes):
        raise ValueError("Exo baseline memory values must be positive")
    if total_layers < len(memory_bytes):
        raise ValueError("Exo baseline requires at least one layer per worker")

    total_memory = sum(memory_bytes)
    numerators = tuple(value * total_layers for value in memory_bytes)
    result = [value // total_memory for value in numerators]
    by_remainder = sorted(
        range(len(memory_bytes)),
        key=lambda index: (numerators[index] % total_memory, -index),
        reverse=True,
    )
    for index in by_remainder[: total_layers - sum(result)]:
        result[index] += 1

    # This is the same minimum-one correction as Exo. It matters for large
    # heterogeneous cycles where a small contributor rounds down to zero.
    for index, allocated in enumerate(result):
        if allocated != 0:
            continue
        donor = max(range(len(result)), key=lambda item: (result[item], -item))
        if result[donor] <= 1:
            raise ValueError("Exo baseline cannot preserve one layer per worker")
        result[donor] -= 1
        result[index] = 1
    return tuple(result)


def exo_memory_proportional_fixed_context_placements(
    manifest: ModelManifest,
    demand: ContextCapacityDemandMap,
    worker_frontiers: Mapping[str, tuple[MemoryPlacementPoint, ...]],
    worker_memory_bytes: Mapping[str, int],
    *,
    context_tokens: int,
    now_ms: int,
) -> GreedyPlacementResult:
    """Exo-style one-cycle baseline adapted to Fabi's exact span contract.

    Exo allocates a contiguous pipeline proportionally to each node's reported
    available memory. It does not optimize multiple context classes or route
    redundancy. This baseline keeps that objective, fixes one signed context,
    and selects the closest feasible contiguous chain from Fabi's exact local
    frontiers. A dynamic program is required because Fabi's endpoint weights
    make some otherwise proportional boundaries infeasible.
    """

    demand.validate_for(manifest, now_ms=now_ms)
    if context_tokens not in manifest.context_classes:
        raise ValueError("Exo baseline requires a signed context class")
    worker_ids = tuple(worker_frontiers)
    if set(worker_ids) != set(worker_memory_bytes):
        raise ValueError("Exo baseline memory map must exactly match worker frontiers")
    targets = _largest_remainder_layer_allocation(
        manifest.num_layers,
        tuple(worker_memory_bytes[worker_id] for worker_id in worker_ids),
    )

    # cursor -> (rank, placements). The rank first minimizes deviation from
    # Exo's proportional split, then prefers the strongest KV bottleneck.
    states: dict[
        int,
        tuple[tuple[int, int, int], tuple[SimulatedPlacement, ...]],
    ] = {0: ((0, 0, 0), ())}
    for worker_id, target_layers in zip(worker_ids, targets, strict=True):
        next_states: dict[
            int,
            tuple[tuple[int, int, int], tuple[SimulatedPlacement, ...]],
        ] = {}
        for cursor, (rank, selected) in states.items():
            for point in worker_frontiers[worker_id]:
                if point.context_tokens != context_tokens or point.span.start != cursor:
                    continue
                candidate_rank = (
                    rank[0] - abs(point.span.length - target_layers),
                    min(rank[1], point.max_sessions) if selected else point.max_sessions,
                    rank[2] + point.max_sessions,
                )
                candidate = (*selected, SimulatedPlacement(worker_id, point))
                previous = next_states.get(point.span.end)
                if previous is None or candidate_rank > previous[0]:
                    next_states[point.span.end] = (candidate_rank, candidate)
        states = next_states
        if not states:
            break

    complete = states.get(manifest.num_layers)
    if complete is None:
        raise ValueError(
            "Exo proportional split cannot form an exact Fabi fixed-span route"
        )
    result = complete[1]
    return GreedyPlacementResult(
        placements=result,
        service=evaluate_context_service(manifest, result),
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
    return _score_candidate_potential_from_state(
        manifest,
        demand,
        placements,
        candidate,
        before=before,
        coverage=_placement_context_coverage(manifest, placements),
    )


def _score_candidate_potential_from_state(
    manifest: ModelManifest,
    demand: ContextCapacityDemandMap,
    placements: tuple[SimulatedPlacement, ...],
    candidate: SimulatedPlacement,
    *,
    before: Mapping[int, ContextServiceCapacity],
    coverage: Mapping[int, list[int]],
) -> CandidatePotential:
    """Score one candidate against state cached once for the joining worker."""

    after = {
        item.context_tokens: item
        for item in evaluate_context_service(manifest, (*placements, candidate))
    }

    route_gain = 0.0
    slot_gain = 0.0
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
    return CandidatePotential(
        weighted_independent_route_gain=route_gain,
        weighted_concurrent_slot_gain=slot_gain,
        weighted_layer_deficit_filled=_weighted_layer_deficit_from_coverage(
            manifest, demand, coverage, candidate.point
        ),
        context_tokens=candidate.point.context_tokens,
        span_length=candidate.point.span.length,
        max_sessions=candidate.point.max_sessions,
    )


def greedy_context_placements(
    manifest: ModelManifest,
    demand: ContextCapacityDemandMap,
    worker_frontiers: Mapping[str, tuple[MemoryPlacementPoint, ...]],
    *,
    now_ms: int,
    exact_candidate_bound: int = 32,
) -> GreedyPlacementResult:
    """Run the bounded two-pass decentralized heuristic in arrival order."""

    demand.validate_for(manifest, now_ms=now_ms)
    if exact_candidate_bound <= 0:
        raise ValueError("exact candidate bound must be positive")

    placements: list[SimulatedPlacement] = []
    for worker_id, frontier in worker_frontiers.items():
        if not worker_id:
            raise ValueError("simulated worker ids must be non-empty")
        if not frontier:
            continue
        current = tuple(placements)
        coverage = _placement_context_coverage(manifest, current)
        before = {
            item.context_tokens: item
            for item in evaluate_context_service(manifest, current)
        }
        cheap_ranked = sorted(
            [
                (
                    point,
                    _weighted_layer_deficit_from_coverage(
                        manifest, demand, coverage, point
                    ),
                )
                for point in frontier
            ],
            key=lambda item: (
                item[1],
                item[0].context_tokens,
                item[0].span.length,
                item[0].max_sessions,
                -item[0].span.start,
            ),
            reverse=True,
        )
        finalists = cheap_ranked[:exact_candidate_bound]
        candidates = [
            (
                SimulatedPlacement(worker_id=worker_id, point=point),
                cheap_score,
            )
            for point, cheap_score in finalists
        ]
        chosen, _ = max(
            candidates,
            key=lambda item: (
                _score_candidate_potential_from_state(
                    manifest,
                    demand,
                    current,
                    item[0],
                    before=before,
                    coverage=coverage,
                ).rank(),
                item[1],
                -item[0].point.span.start,
            ),
        )
        placements.append(chosen)

    result = tuple(placements)
    return GreedyPlacementResult(
        placements=result,
        service=evaluate_context_service(manifest, result),
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
        demands.append(
            {
                "context_tokens": item.context_tokens,
                "target_concurrent_slots": item.desired_concurrent_slots,
                "target_independent_routes": item.desired_independent_routes,
                "weight": max(1, round(class_weight * weight_scale)),
            }
        )

    return {
        "num_layers": manifest.num_layers,
        "context_classes": list(manifest.context_classes),
        "demands": demands,
        "workers": workers,
    }
