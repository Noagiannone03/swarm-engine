from __future__ import annotations

from swarm_protocol import (
    ContextCapacityDemandMap,
    ContextClassDemand,
    LayerSpan,
    ModelManifest,
)
from swarm_protocol.context_placement import MemoryPlacementPoint
from swarm_protocol.placement_simulator import (
    SimulatedPlacement,
    build_oracle_scenario,
    evaluate_context_service,
    score_candidate_potential,
)

HASHES = tuple(character * 64 for character in "abcdef")


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
    demand = ContextCapacityDemandMap(
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
    demand = ContextCapacityDemandMap(
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
        {"context_tokens": 10, "target_slots": 2, "weight": 1_000},
        {"context_tokens": 20, "target_slots": 3, "weight": 4_000},
    ]
    assert set(scenario) == {"num_layers", "context_classes", "demands", "workers"}
