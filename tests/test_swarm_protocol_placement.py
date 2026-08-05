from __future__ import annotations

import pytest

from swarm_protocol import (
    AutonomousPlacementPolicy,
    BackendKind,
    CapacityDemandMap,
    ContextCapacityDemandMap,
    ContextClassDemand,
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    MaterializationPhase,
    ModelManifest,
    PlacementAction,
    PlacementDecision,
    PlacementMaterializer,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
    cumulative_context_coverage,
)

HASHES = tuple(character * 64 for character in "abcdef")
SWARM_NOW = 1_000


def manifest() -> ModelManifest:
    return ModelManifest(
        model_id="test/model",
        immutable_revision="revision",
        architecture_graph_hash=HASHES[0],
        tokenizer_hash=HASHES[1],
        weight_collection_hash=HASHES[2],
        weight_format="safetensors",
        quantization="bf16",
        dtype="bfloat16",
        num_layers=4,
        model_max_context_tokens=65_536,
        context_classes=(4_096, 8_192, 16_384, 32_768, 65_536),
        activation_bytes_per_token=128,
        kv_bytes_per_token_by_layer=(10,) * 4,
        weight_bytes_by_layer=(100,) * 4,
        input_endpoint_weight_bytes=100,
        output_endpoint_weight_bytes=100,
        shared_endpoint_weight_bytes=0,
        rope_context_contract_hash=HASHES[3],
        attention_kv_contract_hash=HASHES[4],
        prefill_contract_hash=HASHES[5],
        wire_protocol_version=1,
    )


def offer(
    worker_id: str = "joining",
    *,
    memory_bytes: int = 500,
    frontend: bool = True,
) -> WorkerOffer:
    roles = {WorkerRole.EXECUTOR}
    if frontend:
        roles.add(WorkerRole.FRONTEND)
    return WorkerOffer(
        worker_id=worker_id,
        endpoint_id=f"{worker_id}-endpoint",
        runtime_version="test",
        platform="test",
        backend=BackendKind.MLX,
        stable_memory_envelope_bytes=memory_bytes,
        supported_roles=roles,
        offer_seq=1,
        issued_at_ms=SWARM_NOW,
        expires_at_ms=SWARM_NOW + 60_000,
    )


def lease(
    model: ModelManifest,
    worker_id: str,
    start: int,
    end: int,
    *,
    state: SpanState = SpanState.READY,
    max_context_tokens: int = 65_536,
) -> SpanLease:
    return SpanLease(
        model_swarm_id=model.model_swarm_id,
        worker_id=worker_id,
        hosted_span=LayerSpan(start=start, end=end),
        effective_span_mode=EffectiveSpanMode.FIXED,
        state=state,
        weight_hashes=(HASHES[0],),
        max_context_tokens=max_context_tokens,
        kv_geometry=KvGeometry(
            block_size_tokens=1,
            bytes_per_token_by_layer=model.kv_bytes_per_token_by_layer,
            allocatable_bytes=10_000,
        ),
        available_kv_bytes_snapshot=10_000,
        max_sessions=1,
        lease_seq=1,
        issued_at_ms=SWARM_NOW,
        expires_at_ms=SWARM_NOW + 60_000,
    )


def test_joining_worker_fills_the_only_missing_contiguous_range():
    model = manifest()
    policy = AutonomousPlacementPolicy()
    decision = policy.choose(
        offer=offer(),
        manifest=model,
        leases=(
            lease(model, "head", 0, 2),
            lease(model, "head-copy", 0, 2),
        ),
        demand=CapacityDemandMap.uniform(4, desired_replicas=1),
        context_tokens=10,
        kv_block_size=1,
        now_ms=SWARM_NOW,
    )

    assert decision.action is PlacementAction.JOIN
    assert decision.span == LayerSpan(start=2, end=4)
    assert decision.required_memory_bytes == 500


def test_simultaneous_cold_join_counts_building_intent_for_demand_spreading():
    model = manifest()
    decision = AutonomousPlacementPolicy().choose(
        offer=offer("second-cold-worker"),
        manifest=model,
        leases=(lease(model, "first-cold-worker", 0, 2, state=SpanState.BUILDING),),
        demand=CapacityDemandMap.uniform(4, desired_replicas=1),
        context_tokens=10,
        kv_block_size=1,
        now_ms=SWARM_NOW,
    )

    assert decision.action is PlacementAction.JOIN
    assert decision.span == LayerSpan(start=2, end=4)


def test_fixed_cold_workers_form_a_complete_route_when_tail_arrives_first():
    model = manifest()
    policy = AutonomousPlacementPolicy()
    tail = policy.choose(
        offer=offer("tail-first", memory_bytes=500, frontend=False),
        manifest=model,
        leases=(),
        demand=CapacityDemandMap.uniform(4, desired_replicas=2),
        context_tokens=10,
        kv_block_size=1,
        now_ms=SWARM_NOW,
    )
    assert tail.action is PlacementAction.JOIN
    assert tail.span == LayerSpan(start=2, end=4)

    head = policy.choose(
        offer=offer("head-second", memory_bytes=500, frontend=True),
        manifest=model,
        leases=(
            lease(
                model,
                "tail-first",
                tail.span.start,
                tail.span.end,
                state=SpanState.BUILDING,
            ),
        ),
        demand=CapacityDemandMap.uniform(4, desired_replicas=2),
        context_tokens=10,
        kv_block_size=1,
        now_ms=SWARM_NOW + 1,
    )

    assert head.action is PlacementAction.JOIN
    assert head.span == LayerSpan(start=0, end=2)
    assert head.score is not None
    assert head.score.completes_fixed_route == 1


def test_first_frontend_worker_establishes_ingress_when_catalogue_is_empty():
    model = manifest()

    decision = AutonomousPlacementPolicy().choose(
        offer=offer("frontend-first", memory_bytes=500, frontend=True),
        manifest=model,
        leases=(),
        demand=CapacityDemandMap.uniform(4, desired_replicas=2),
        context_tokens=10,
        kv_block_size=1,
        now_ms=SWARM_NOW,
    )

    assert decision.action is PlacementAction.JOIN
    assert decision.span == LayerSpan(start=0, end=2)
    assert decision.score is not None
    assert decision.score.establishes_missing_frontend == 1


def test_cold_join_never_counts_its_own_historical_lease_as_replica():
    model = manifest()

    decision = AutonomousPlacementPolicy().choose(
        offer=offer("persistent-worker", memory_bytes=500, frontend=True),
        manifest=model,
        leases=(lease(model, "persistent-worker", 0, 2),),
        demand=CapacityDemandMap.uniform(4, desired_replicas=2),
        context_tokens=10,
        kv_block_size=1,
        now_ms=SWARM_NOW,
    )

    assert decision.action is PlacementAction.JOIN
    assert decision.span == LayerSpan(start=0, end=2)


def test_executor_without_http_frontend_can_fill_the_model_tail():
    model = manifest()
    policy = AutonomousPlacementPolicy()
    decision = policy.choose(
        offer=offer(frontend=False),
        manifest=model,
        leases=(lease(model, "head", 0, 2),),
        demand=CapacityDemandMap.uniform(4, desired_replicas=1),
        context_tokens=10,
        kv_block_size=1,
        now_ms=SWARM_NOW,
    )

    assert decision.action is PlacementAction.JOIN
    assert decision.span == LayerSpan(start=2, end=4)


def test_small_worker_is_evaluated_for_the_layer_it_can_really_host():
    model = manifest()
    policy = AutonomousPlacementPolicy()
    decision = policy.choose(
        offer=offer(memory_bytes=200),
        manifest=model,
        leases=(
            lease(model, "left", 0, 2),
            lease(model, "right", 3, 4),
        ),
        demand=CapacityDemandMap.uniform(4, desired_replicas=1),
        context_tokens=10,
        kv_block_size=1,
        now_ms=SWARM_NOW,
    )

    assert decision.action is PlacementAction.JOIN
    assert decision.span == LayerSpan(start=2, end=3)


def test_fixed_backend_never_moves_if_unload_would_open_a_coverage_hole():
    model = manifest()
    policy = AutonomousPlacementPolicy()
    decision = policy.choose(
        offer=offer("current", memory_bytes=500),
        manifest=model,
        leases=(
            lease(model, "current", 0, 2),
            lease(model, "other", 0, 1),
            lease(model, "tail", 2, 4),
        ),
        demand=CapacityDemandMap(
            desired_replicas_by_layer=(1, 1, 3, 3),
            demand_weight_by_layer=(1, 1, 10, 10),
        ),
        context_tokens=10,
        kv_block_size=1,
        current_span=LayerSpan(start=0, end=2),
        now_ms=SWARM_NOW,
    )

    assert decision.action is PlacementAction.KEEP
    assert decision.span == LayerSpan(start=0, end=2)
    assert decision.reason == "current_span_is_still_the_best_stable_choice"


def test_active_reservation_and_cooldown_prevent_oscillating_reload():
    model = manifest()
    leases = (
        lease(model, "current", 0, 2),
        lease(model, "replica", 0, 2),
        lease(model, "tail", 2, 4),
    )
    demand = CapacityDemandMap(
        desired_replicas_by_layer=(1, 1, 3, 3),
        demand_weight_by_layer=(1, 1, 10, 10),
    )
    policy = AutonomousPlacementPolicy(movement_cooldown_ms=10_000)
    common = dict(
        offer=offer("current", memory_bytes=500),
        manifest=model,
        leases=leases,
        demand=demand,
        context_tokens=10,
        kv_block_size=1,
        current_span=LayerSpan(start=0, end=2),
        serving_route_survives_movement=True,
        now_ms=20_000,
    )

    reserved = policy.choose(**common, current_reservations=1)
    cooling = policy.choose(**common, last_moved_at_ms=15_000)

    assert reserved.action is PlacementAction.KEEP
    assert reserved.reason == "active_reservations_must_drain_before_movement"
    assert cooling.action is PlacementAction.KEEP
    assert cooling.reason == "movement_cooldown_has_not_elapsed"


def test_placement_is_independent_from_catalogue_arrival_order():
    model = manifest()
    current = lease(model, "current", 0, 2)
    replica = lease(model, "replica", 0, 2)
    tail = lease(model, "tail", 2, 4)
    policy = AutonomousPlacementPolicy(movement_cooldown_ms=0)
    common = dict(
        offer=offer("current", memory_bytes=500),
        manifest=model,
        demand=CapacityDemandMap(
            desired_replicas_by_layer=(1, 1, 3, 3),
            demand_weight_by_layer=(1, 1, 10, 10),
        ),
        context_tokens=10,
        kv_block_size=1,
        current_span=LayerSpan(start=0, end=2),
        serving_route_survives_movement=True,
        now_ms=20_000,
    )

    forward = policy.choose(leases=(current, replica, tail), **common)
    reverse = policy.choose(leases=(tail, replica, current), **common)

    assert forward == reverse
    assert forward.action is PlacementAction.MOVE


def test_ready_worker_never_moves_without_a_complete_independent_route():
    model = manifest()
    decision = AutonomousPlacementPolicy(movement_cooldown_ms=0).choose(
        offer=offer("current", memory_bytes=500),
        manifest=model,
        leases=(
            lease(model, "current", 0, 2),
            lease(model, "head-copy", 0, 2),
            lease(model, "tail", 2, 4),
        ),
        demand=CapacityDemandMap(
            desired_replicas_by_layer=(1, 1, 3, 3),
            demand_weight_by_layer=(1, 1, 10, 10),
        ),
        context_tokens=10,
        kv_block_size=1,
        current_span=LayerSpan(start=0, end=2),
        serving_route_exists=True,
        serving_route_survives_movement=False,
        now_ms=20_000,
    )

    assert decision.action is PlacementAction.KEEP
    assert decision.reason == "movement_would_remove_the_last_executable_route"


def test_short_context_replica_cannot_displace_the_last_long_context_head():
    """Coverage is cumulative by context, not global across every lease.

    A fast 16k full-model replica must not make the only 32k head appear
    redundant. Otherwise that head may move by one layer, silently collapsing
    the long route even though every layer remains covered at *some* context.
    """

    model = manifest()
    current = lease(
        model,
        "long-head",
        0,
        2,
        max_context_tokens=20,
    )
    decision = AutonomousPlacementPolicy(
        minimum_improvement=0,
        movement_cooldown_ms=0,
    ).choose(
        offer=offer("long-head", memory_bytes=700),
        manifest=model,
        leases=(
            current,
            lease(model, "long-tail", 2, 4, max_context_tokens=20),
            lease(model, "short-replica", 0, 4, max_context_tokens=10),
        ),
        demand=CapacityDemandMap(
            desired_replicas_by_layer=(1, 1, 3, 3),
            demand_weight_by_layer=(1, 1, 10, 10),
        ),
        context_tokens=20,
        kv_block_size=1,
        current_span=current.hosted_span,
        serving_route_exists=True,
        serving_route_survives_movement=True,
        now_ms=20_000,
    )

    assert decision.action is PlacementAction.KEEP
    assert decision.span == LayerSpan(start=0, end=2)
    assert decision.reason == "current_span_is_still_the_best_stable_choice"


def test_disjoint_lab_topology_moves_redundant_head_to_uncovered_tail():
    base = manifest()
    model = base.model_copy(
        update={
            "num_layers": 36,
            "kv_bytes_per_token_by_layer": (10,) * 36,
            "weight_bytes_by_layer": (100,) * 36,
        }
    )
    decision = AutonomousPlacementPolicy(movement_cooldown_ms=0).choose(
        offer=offer("local", memory_bytes=2_100),
        manifest=model,
        leases=(
            lease(model, "local", 0, 10),
            lease(model, "mac-mini", 0, 32),
        ),
        demand=CapacityDemandMap.uniform(36, desired_replicas=2),
        context_tokens=10,
        kv_block_size=1,
        current_span=LayerSpan(start=0, end=10),
        serving_route_exists=False,
        serving_route_survives_movement=False,
        now_ms=20_000,
    )

    assert decision.action is PlacementAction.MOVE
    assert decision.span == LayerSpan(start=32, end=36)
    assert decision.reason == "coverage_preserved_and_verified_gain_exceeds_hysteresis"


def _context_demand(model: ModelManifest) -> ContextCapacityDemandMap:
    return ContextCapacityDemandMap(
        model_swarm_id=model.model_swarm_id,
        region_id="eu-west",
        issued_at_ms=1_000,
        expires_at_ms=61_000,
        classes=tuple(
            ContextClassDemand(
                context_tokens=context_tokens,
                desired_independent_routes=1 if context_tokens <= 32_768 else 0,
                desired_replicas_by_layer=(1,) * model.num_layers,
                demand_weight_by_layer=(1.0,) * model.num_layers,
            )
            for context_tokens in model.context_classes
        ),
    )


def test_context_demand_is_versioned_expiring_and_matches_the_signed_ladder():
    model = manifest()
    demand = _context_demand(model)

    demand.validate_for(model, now_ms=2_000)
    assert demand.class_for(16_385).context_tokens == 32_768
    assert demand.class_for(16_384).layer_screen() == CapacityDemandMap.uniform(
        4, desired_replicas=1
    )

    with pytest.raises(ValueError, match="expired"):
        demand.validate_for(model, now_ms=61_000)
    with pytest.raises(ValueError, match="exceeds"):
        demand.class_for(65_537)


def test_context_demand_rejects_unsorted_or_wrong_model_classes():
    model = manifest()
    demand = _context_demand(model)

    with pytest.raises(ValueError, match="strictly increasing"):
        ContextCapacityDemandMap.model_validate(
            {
                **demand.model_dump(),
                "classes": tuple(reversed(demand.classes)),
            }
        )

    missing_class = demand.model_copy(update={"classes": demand.classes[:-1]})
    with pytest.raises(ValueError, match="signed model contract"):
        missing_class.validate_for(model, now_ms=2_000)


def test_memory_frontier_exposes_exact_span_context_session_tradeoffs():
    model = manifest().model_copy(
        update={
            "model_max_context_tokens": 40,
            "context_classes": (10, 20, 40),
        }
    )
    points = AutonomousPlacementPolicy().memory_frontier(
        offer=offer(memory_bytes=700),
        manifest=model,
        qualified_context_limit_tokens=40,
        kv_block_size=1,
    )

    by_key = {(point.span, point.context_tokens): point for point in points}
    short = by_key[(LayerSpan(start=0, end=2), 10)]
    long = by_key[(LayerSpan(start=0, end=2), 20)]
    assert (short.weight_bytes, short.kv_bytes_per_session, short.max_sessions) == (300, 200, 2)
    assert (long.weight_bytes, long.kv_bytes_per_session, long.max_sessions) == (300, 400, 1)
    assert (LayerSpan(start=0, end=2), 40) not in by_key
    assert all(point.required_memory_bytes <= 700 for point in points)


def test_memory_frontier_never_offers_frontend_layers_to_executor_only_worker():
    model = manifest().model_copy(
        update={
            "model_max_context_tokens": 40,
            "context_classes": (10, 20, 40),
        }
    )

    points = AutonomousPlacementPolicy().memory_frontier(
        offer=offer(memory_bytes=700, frontend=False),
        manifest=model,
        qualified_context_limit_tokens=40,
        kv_block_size=1,
    )

    assert points
    assert all(point.span.start > 0 for point in points)


def test_memory_frontier_never_exceeds_the_qualified_backend_limit():
    model = manifest().model_copy(
        update={
            "model_max_context_tokens": 40,
            "context_classes": (10, 20, 40),
        }
    )

    points = AutonomousPlacementPolicy().memory_frontier(
        offer=offer(memory_bytes=700),
        manifest=model,
        qualified_context_limit_tokens=20,
        kv_block_size=1,
    )

    assert points
    assert all(point.context_tokens <= 20 for point in points)


def test_long_context_lease_counts_cumulatively_but_short_lease_does_not():
    model = manifest()
    long_lease = lease(model, "long", 0, 2).model_copy(update={"max_context_tokens": 65_536})
    short_lease = lease(model, "short", 2, 4).model_copy(update={"max_context_tokens": 16_384})

    coverage = cumulative_context_coverage(model, (long_lease, short_lease))

    assert coverage[4_096] == (1, 1, 1, 1)
    assert coverage[16_384] == (1, 1, 1, 1)
    assert coverage[32_768] == (1, 1, 0, 0)
    assert coverage[65_536] == (1, 1, 0, 0)


class FakeDrain:
    def __init__(self, reservations: int = 0) -> None:
        self.reservations = reservations
        self.draining = False

    def begin_drain(self) -> int:
        self.draining = True
        return self.reservations

    def draining_reservations(self) -> int:
        return self.reservations if self.draining else 0

    def cancel_drain(self) -> None:
        self.draining = False

    def finish_drain(self) -> None:
        assert self.reservations == 0
        self.draining = False


def move_decision(span: LayerSpan) -> PlacementDecision:
    return PlacementDecision(
        action=PlacementAction.MOVE,
        span=span,
        required_memory_bytes=1,
        score=None,
        reason="test",
    )


def test_materializer_drains_then_generation_fences_executor_reload():
    drain = FakeDrain(reservations=1)
    reloads = []
    materializer = PlacementMaterializer(
        drain=drain,
        reload_target=lambda span, generation: reloads.append((span, generation)),
        current_span=LayerSpan(start=0, end=2),
    )

    draining = materializer.reconcile(move_decision(LayerSpan(start=2, end=4)))
    assert draining.phase is MaterializationPhase.DRAINING
    assert reloads == []

    drain.reservations = 0
    building = materializer.continue_after_drain()
    assert building.phase is MaterializationPhase.BUILDING
    assert building.current_span is None
    assert reloads == [(LayerSpan(start=2, end=4), 1)]

    with pytest.raises(RuntimeError, match="stale"):
        materializer.mark_ready(span=LayerSpan(start=2, end=4), generation=0)
    ready = materializer.mark_ready(span=LayerSpan(start=2, end=4), generation=1)
    assert ready.phase is MaterializationPhase.READY
    assert ready.current_span == LayerSpan(start=2, end=4)
    assert not drain.draining


def test_materializer_rolls_back_failed_move_as_a_new_fenced_generation():
    drain = FakeDrain()
    reloads = []
    materializer = PlacementMaterializer(
        drain=drain,
        reload_target=lambda span, generation: reloads.append((span, generation)),
        current_span=LayerSpan(start=0, end=2),
    )
    materializer.reconcile(move_decision(LayerSpan(start=2, end=4)))

    rollback = materializer.mark_failed(generation=1, error=RuntimeError("load failed"))

    assert rollback.phase is MaterializationPhase.BUILDING
    assert rollback.target_span == LayerSpan(start=0, end=2)
    assert reloads == [
        (LayerSpan(start=2, end=4), 1),
        (LayerSpan(start=0, end=2), 2),
    ]
