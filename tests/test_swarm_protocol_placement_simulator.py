from __future__ import annotations

from swarm_protocol import (
    ContextCapacityDemandMap,
    ContextClassDemand,
    LayerSpan,
    ModelManifest,
)
from swarm_protocol.context_placement import MemoryPlacementPoint
from swarm_protocol.context_demand import build_context_histogram
from swarm_protocol.placement_simulator import (
    SimulatedPlacement,
    SyntheticWorkerEnvelope,
    build_oracle_scenario,
    build_synthetic_memory_frontiers,
    evaluate_context_service,
    exo_memory_proportional_fixed_context_placements,
    greedy_context_placements,
    maximum_span_placements,
    petals_fixed_context_placements,
    score_candidate_potential,
)

HASHES = tuple(character * 64 for character in "abcdef")


def _demand_map(**kwargs) -> ContextCapacityDemandMap:
    classes = tuple(kwargs.pop("classes"))
    return ContextCapacityDemandMap(
        **kwargs,
        classes=classes,
        context_histogram=build_context_histogram(
            tuple(item.context_tokens for item in classes)
        ),
    )


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_id="test/simulator",
        immutable_revision="revision",
        architecture_graph_hash=HASHES[0],
        tokenizer_hash=HASHES[1],
        weight_collection_hash=HASHES[2],
        weight_format="safetensors",
        quantization="bf16",
        dtype="bfloat16",
        num_layers=4,
        model_max_context_tokens=65_536,
        context_classes=(16_384, 32_768, 65_536),
        activation_bytes_per_token=128,
        kv_bytes_per_token_by_layer=(10,) * 4,
        weight_bytes_by_layer=(100,) * 4,
        rope_context_contract_hash=HASHES[3],
        attention_kv_contract_hash=HASHES[4],
        prefill_contract_hash=HASHES[5],
        wire_protocol_version=1,
    )


def _placement(
    worker_id: str,
    start: int,
    end: int,
    *,
    context_tokens: int,
    sessions: int,
) -> SimulatedPlacement:
    return SimulatedPlacement(
        worker_id=worker_id,
        point=MemoryPlacementPoint(
            span=LayerSpan(start=start, end=end),
            context_tokens=context_tokens,
            weight_bytes=100,
            kv_bytes_per_session=100,
            max_sessions=sessions,
            memory_headroom_bytes=0,
        ),
    )


def test_max_flow_distinguishes_slots_redundancy_and_context_classes():
    model = _manifest()
    placements = (
        _placement("head-long", 0, 2, context_tokens=65_536, sessions=2),
        _placement("tail-short", 2, 4, context_tokens=16_384, sessions=3),
        _placement("tail-long", 2, 4, context_tokens=65_536, sessions=1),
    )

    by_context = {
        capacity.context_tokens: capacity
        for capacity in evaluate_context_service(model, placements)
    }

    assert by_context[16_384].concurrent_service_slots == 2
    assert by_context[16_384].independent_worker_routes == 1
    assert by_context[32_768].concurrent_service_slots == 1
    assert by_context[65_536].concurrent_service_slots == 1


def test_disconnected_layer_coverage_has_zero_service_capacity():
    model = _manifest()
    placements = (
        _placement("left", 0, 1, context_tokens=65_536, sessions=4),
        _placement("right", 2, 4, context_tokens=65_536, sessions=4),
    )

    assert all(
        capacity.concurrent_service_slots == 0
        for capacity in evaluate_context_service(model, placements)
    )


def test_coding_demand_prior_can_prefer_long_route_progress_over_short_route_trap():
    model = _manifest().model_copy(
        update={
            "model_max_context_tokens": 20,
            "context_classes": (10, 20),
        }
    )
    demand = _demand_map(
        model_swarm_id=model.model_swarm_id,
        region_id="eu-west",
        issued_at_ms=1_000,
        expires_at_ms=61_000,
        classes=(
            ContextClassDemand(
                context_tokens=10,
                desired_independent_routes=1,
                desired_concurrent_slots=1,
                desired_replicas_by_layer=(1,) * 4,
                demand_weight_by_layer=(1.0,) * 4,
            ),
            ContextClassDemand(
                context_tokens=20,
                desired_independent_routes=1,
                desired_concurrent_slots=1,
                desired_replicas_by_layer=(1,) * 4,
                demand_weight_by_layer=(10.0,) * 4,
            ),
        ),
    )
    short_full = _placement("adaptive", 0, 4, context_tokens=10, sessions=1)
    long_head = _placement("adaptive", 0, 2, context_tokens=20, sessions=1)

    short_score = score_candidate_potential(
        model, demand, (), short_full, now_ms=2_000
    )
    long_score = score_candidate_potential(
        model, demand, (), long_head, now_ms=2_000
    )

    assert short_score.weighted_independent_route_gain == 1
    assert long_score.weighted_independent_route_gain == 0
    assert long_score.rank() > short_score.rank()


def test_oracle_scenario_is_deterministic_and_uses_aggregated_demand_only():
    model = _manifest().model_copy(
        update={
            "model_max_context_tokens": 20,
            "context_classes": (10, 20),
        }
    )
    demand = _demand_map(
        model_swarm_id=model.model_swarm_id,
        region_id="eu-west",
        issued_at_ms=1_000,
        expires_at_ms=61_000,
        classes=(
            ContextClassDemand(
                context_tokens=10,
                desired_independent_routes=1,
                desired_concurrent_slots=2,
                desired_replicas_by_layer=(1,) * 4,
                demand_weight_by_layer=(1.0,) * 4,
            ),
            ContextClassDemand(
                context_tokens=20,
                desired_independent_routes=1,
                desired_concurrent_slots=3,
                desired_replicas_by_layer=(1,) * 4,
                demand_weight_by_layer=(5.0,) * 4,
                confidence=0.8,
            ),
        ),
    )
    frontiers = {
        "worker-b": (_placement("ignored", 2, 4, context_tokens=20, sessions=1).point,),
        "worker-a": (_placement("ignored", 0, 2, context_tokens=20, sessions=2).point,),
    }

    scenario = build_oracle_scenario(model, demand, frontiers, now_ms=2_000)

    assert [worker["worker_id"] for worker in scenario["workers"]] == ["worker-a", "worker-b"]
    assert scenario["demands"] == [
        {
            "context_tokens": 10,
            "target_concurrent_slots": 2,
            "target_independent_routes": 1,
            "weight": 1_000,
        },
        {
            "context_tokens": 20,
            "target_concurrent_slots": 3,
            "target_independent_routes": 1,
            "weight": 4_000,
        },
    ]
    assert set(scenario) == {"num_layers", "context_classes", "demands", "workers"}


def test_two_pass_greedy_bootstraps_the_weighted_agentic_route():
    model = _manifest().model_copy(
        update={
            "model_max_context_tokens": 20,
            "context_classes": (10, 20),
        }
    )
    demand = _demand_map(
        model_swarm_id=model.model_swarm_id,
        region_id="eu-west",
        issued_at_ms=1_000,
        expires_at_ms=61_000,
        classes=(
            ContextClassDemand(
                context_tokens=10,
                desired_independent_routes=1,
                desired_concurrent_slots=1,
                desired_replicas_by_layer=(1,) * 4,
                demand_weight_by_layer=(1.0,) * 4,
            ),
            ContextClassDemand(
                context_tokens=20,
                desired_independent_routes=1,
                desired_concurrent_slots=1,
                desired_replicas_by_layer=(1,) * 4,
                demand_weight_by_layer=(10.0,) * 4,
            ),
        ),
    )
    frontiers = {
        "adaptive": (
            _placement("ignored", 0, 4, context_tokens=10, sessions=1).point,
            _placement("ignored", 0, 2, context_tokens=20, sessions=1).point,
        ),
        "tail": (_placement("ignored", 2, 4, context_tokens=20, sessions=1).point,),
    }

    result = greedy_context_placements(model, demand, frontiers, now_ms=2_000)

    assert result.placements[0].point == frontiers["adaptive"][1]
    assert result.service[-1].context_tokens == 20
    assert result.service[-1].concurrent_service_slots == 1


def test_agentic_policy_spreads_an_explicit_8_16g_class_population_for_context():
    mib = 1024**2
    gib = 1024**3
    model = _manifest().model_copy(
        update={
            "model_id": "test/qwen3-4b-long-geometry",
            "num_layers": 36,
            "model_max_context_tokens": 65_536,
            "context_classes": (16_384, 40_960, 65_536),
            "kv_bytes_per_token_by_layer": (4_096,) * 36,
            "weight_bytes_by_layer": (200 * mib,) * 36,
            "input_endpoint_weight_bytes": 256 * mib,
            "output_endpoint_weight_bytes": 256 * mib,
        }
    )
    frontiers = build_synthetic_memory_frontiers(
        model,
        tuple(
            SyntheticWorkerEnvelope(f"stable-4g-{index}", 4 * gib, 65_536)
            for index in range(6)
        )
        + tuple(
            SyntheticWorkerEnvelope(f"stable-10g-{index}", 10 * gib, 65_536)
            for index in range(4)
        ),
        kv_block_size=16,
        maximum_sessions=4,
    )

    small = frontiers["stable-4g-0"]
    large = frontiers["stable-10g-0"]
    assert max(point.span.length for point in small) == 15
    assert any(
        point.context_tokens == 65_536 and point.span.length == 8 for point in small
    )
    assert any(
        point.context_tokens == 16_384 and point.span.length == 36 for point in large
    )
    assert all(point.required_memory_bytes <= 4 * gib for point in small)
    assert all(point.required_memory_bytes <= 10 * gib for point in large)

    demand = _demand_map(
        model_swarm_id=model.model_swarm_id,
        region_id="synthetic-eu",
        issued_at_ms=1_000,
        expires_at_ms=61_000,
        classes=tuple(
            ContextClassDemand(
                context_tokens=context_tokens,
                desired_independent_routes=2,
                desired_concurrent_slots=2,
                desired_replicas_by_layer=(2,) * 36,
                demand_weight_by_layer=(weight,) * 36,
            )
            for context_tokens, weight in (
                (16_384, 1.0),
                (40_960, 3.0),
                (65_536, 8.0),
            )
        ),
    )
    maximum_span = maximum_span_placements(model, demand, frontiers, now_ms=2_000)
    petals_short = petals_fixed_context_placements(
        model,
        demand,
        frontiers,
        context_tokens=16_384,
        now_ms=2_000,
    )
    petals_long = petals_fixed_context_placements(
        model,
        demand,
        frontiers,
        context_tokens=65_536,
        now_ms=2_000,
    )
    exo_long = exo_memory_proportional_fixed_context_placements(
        model,
        demand,
        frontiers,
        {
            worker_id: (4 * gib if worker_id.startswith("stable-4g") else 10 * gib)
            for worker_id in frontiers
        },
        context_tokens=65_536,
        now_ms=2_000,
    )
    context_aware = greedy_context_placements(model, demand, frontiers, now_ms=2_000)

    assert [item.concurrent_service_slots for item in maximum_span.service] == [4, 0, 0]
    assert [item.concurrent_service_slots for item in petals_short.service] == [4, 0, 0]
    petals_long_coverage = [0] * model.num_layers
    for placement in petals_long.placements:
        for layer in range(placement.point.span.start, placement.point.span.end):
            petals_long_coverage[layer] += 1
    assert min(petals_long_coverage) >= 2
    # Petals may execute a suffix of a hosted span. Fabi's current FIXED
    # backend contract may not, so layer coverage alone does not prove that
    # the exact span boundaries form a route.
    assert petals_long.service[-1].concurrent_service_slots == 0
    assert [placement.point.span.length for placement in exo_long.placements] == [
        2,
        2,
        2,
        2,
        2,
        2,
        6,
        6,
        6,
        6,
    ]
    assert exo_long.service[-1].concurrent_service_slots == 4
    assert exo_long.service[-1].independent_worker_routes == 1
    assert [item.concurrent_service_slots for item in context_aware.service] == [2, 2, 2]
    assert [item.independent_worker_routes for item in context_aware.service] == [
        2,
        2,
        2,
    ]
    assert all(
        placement.point.context_tokens == 65_536
        for placement in context_aware.placements
    )


def test_context_policy_beats_maximum_span_trap_for_agentic_demand():
    model = _manifest().model_copy(
        update={
            "model_max_context_tokens": 20,
            "context_classes": (10, 20),
        }
    )
    demand = _demand_map(
        model_swarm_id=model.model_swarm_id,
        region_id="eu-west",
        issued_at_ms=1_000,
        expires_at_ms=61_000,
        classes=(
            ContextClassDemand(
                context_tokens=10,
                desired_independent_routes=1,
                desired_concurrent_slots=1,
                desired_replicas_by_layer=(1,) * 4,
                demand_weight_by_layer=(1.0,) * 4,
            ),
            ContextClassDemand(
                context_tokens=20,
                desired_independent_routes=1,
                desired_concurrent_slots=1,
                desired_replicas_by_layer=(1,) * 4,
                demand_weight_by_layer=(10.0,) * 4,
            ),
        ),
    )
    frontiers = {
        "adaptive": (
            _placement("ignored", 0, 4, context_tokens=10, sessions=1).point,
            _placement("ignored", 0, 2, context_tokens=20, sessions=1).point,
        ),
        "tail": (_placement("ignored", 2, 4, context_tokens=20, sessions=1).point,),
    }

    maximum_span = maximum_span_placements(model, demand, frontiers, now_ms=2_000)
    context_aware = greedy_context_placements(model, demand, frontiers, now_ms=2_000)

    assert maximum_span.service[-1].concurrent_service_slots == 0
    assert context_aware.service[-1].concurrent_service_slots == 1
