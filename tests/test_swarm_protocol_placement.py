from __future__ import annotations

import pytest

from swarm_protocol import (
    AutonomousPlacementPolicy,
    BackendKind,
    CapacityDemandMap,
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


def offer(worker_id: str = "joining", *, memory_bytes: int = 500) -> WorkerOffer:
    return WorkerOffer(
        worker_id=worker_id,
        endpoint_id=f"{worker_id}-endpoint",
        runtime_version="test",
        platform="test",
        backend=BackendKind.MLX,
        stable_memory_envelope_bytes=memory_bytes,
        supported_roles={WorkerRole.EXECUTOR, WorkerRole.FRONTEND},
        offer_seq=1,
        issued_at_ms=SWARM_NOW,
        expires_at_ms=SWARM_NOW + 60_000,
    )


def lease(
    model: ModelManifest,
    worker_id: str,
    start: int,
    end: int,
) -> SpanLease:
    return SpanLease(
        model_swarm_id=model.model_swarm_id,
        worker_id=worker_id,
        hosted_span=LayerSpan(start=start, end=end),
        effective_span_mode=EffectiveSpanMode.FIXED,
        state=SpanState.READY,
        weight_hashes=(HASHES[0],),
        kv_geometry=KvGeometry(
            block_size_tokens=1,
            bytes_per_token_by_layer=(10,) * 4,
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
        now_ms=20_000,
    )

    forward = policy.choose(leases=(current, replica, tail), **common)
    reverse = policy.choose(leases=(tail, replica, current), **common)

    assert forward == reverse
    assert forward.action is PlacementAction.MOVE


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
