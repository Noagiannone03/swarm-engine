from __future__ import annotations

from swarm_protocol import ModelManifest
from swarm_protocol.context_demand import ContextDemandAnnouncer, ContextDemandWindow

HASHES = tuple(character * 64 for character in "abcdef")


def manifest() -> ModelManifest:
    return ModelManifest(
        model_id="test/agentic-model",
        immutable_revision="revision",
        architecture_graph_hash=HASHES[0],
        tokenizer_hash=HASHES[1],
        weight_collection_hash=HASHES[2],
        weight_format="safetensors",
        quantization="bf16",
        dtype="bfloat16",
        num_layers=4,
        model_max_context_tokens=65_536,
        context_classes=(4_096, 16_384, 65_536),
        activation_bytes_per_token=128,
        kv_bytes_per_token_by_layer=(10,) * 4,
        weight_bytes_by_layer=(100,) * 4,
        rope_context_contract_hash=HASHES[3],
        attention_kv_contract_hash=HASHES[4],
        prefill_contract_hash=HASHES[5],
        wire_protocol_version=1,
    )


def test_window_uses_exact_context_bucket_little_law_and_no_route_pressure():
    model = manifest()
    window = ContextDemandWindow(model, "eu-west")
    window.record_admission("request-1", required_context_tokens=12_220, now_ms=1_000)
    window.record_no_route(required_context_tokens=12_220, now_ms=2_000)
    window.record_completion("request-1", now_ms=61_000)

    demand = window.snapshot(now_ms=61_000)
    short, agentic, long = demand.classes
    assert short.desired_concurrent_slots == 0
    assert short.desired_replicas_by_layer == (0,) * model.num_layers
    assert agentic.admitted_requests_per_minute == 0.2
    assert agentic.p95_service_time_ms == 60_000
    assert agentic.no_route_rejections == 1
    assert agentic.desired_independent_routes == 2
    assert agentic.desired_concurrent_slots == 2
    assert agentic.demand_weight_by_layer[0] > agentic.confidence
    assert long.desired_concurrent_slots == 0
    demand.validate_for(model, now_ms=61_000)


def test_window_is_idempotent_for_duplicate_admission_and_tracks_live_long_request():
    model = manifest()
    window = ContextDemandWindow(model, "eu-west")
    window.record_admission("long", required_context_tokens=60_000, now_ms=1_000)
    window.record_admission("long", required_context_tokens=60_000, now_ms=2_000)

    demand = window.snapshot(now_ms=120_000)
    long = demand.classes[-1]
    assert long.admitted_requests_per_minute == 0.2
    assert long.desired_concurrent_slots == 1
    assert long.p95_service_time_ms == 0


def test_completion_keeps_service_signal_when_agentic_turn_outlives_window():
    model = manifest()
    window = ContextDemandWindow(model, "eu-west", window_ms=300_000)
    window.record_admission("long", required_context_tokens=60_000, now_ms=0)
    window.record_completion("long", now_ms=600_000)

    long = window.snapshot(now_ms=600_000).classes[-1]
    assert long.admitted_requests_per_minute == 0.2
    assert long.p95_service_time_ms == 600_000
    assert long.desired_concurrent_slots == 2


class FakeStore:
    def __init__(self) -> None:
        self.published = []

    def publish_context_demand(self, demand) -> None:
        self.published.append(demand)


def test_announcer_publishes_aggregates_only_and_skips_duplicate_clock_sequence():
    now = [1_000]
    store = FakeStore()
    announcer = ContextDemandAnnouncer(
        store,
        "eu-west",
        clock_ms=lambda: now[0],
        start_thread=False,
    )
    model = manifest()
    announcer.record_admission("private-request-id", model, required_context_tokens=12_220)

    assert announcer.publish_once() == 1
    assert announcer.status() == {
        "mode": "advisory",
        "region_id": "eu-west",
        "models": [model.model_swarm_id],
        "last_publish_at_ms": 1_000,
        "error": None,
    }
    assert announcer.publish_once() == 0
    payload = store.published[0].model_dump_json()
    assert "private-request-id" not in payload
    now[0] += 1
    announcer.record_completion("private-request-id")
    assert announcer.publish_once() == 1
    announcer.close()
