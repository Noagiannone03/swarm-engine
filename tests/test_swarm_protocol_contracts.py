import pytest
from pydantic import ValidationError

from swarm_protocol import (
    ArtifactDescriptor,
    ArtifactRole,
    BackendKind,
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    ModelManifest,
    PathKind,
    RecoveryLevel,
    RoutePlan,
    RouteStage,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64


def test_artifact_descriptor_rejects_ambiguous_paths() -> None:
    with pytest.raises(ValidationError, match="normalized"):
        ArtifactDescriptor(
            path="weights/../model.safetensors",
            size=1,
            sha256=HASH_A,
            media_type="application/vnd.safetensors",
            role=ArtifactRole.WEIGHT,
        )


def make_manifest() -> ModelManifest:
    return ModelManifest(
        model_id="Qwen/Qwen3-1.7B",
        immutable_revision="0123456789abcdef",
        architecture_graph_hash=HASH_A,
        tokenizer_hash=HASH_B,
        weight_collection_hash=HASH_C,
        weight_format="safetensors",
        quantization="bf16",
        dtype="bfloat16",
        num_layers=28,
        activation_bytes_per_token=4096,
        rope_context_contract_hash=HASH_D,
        attention_kv_contract_hash=HASH_E,
        prefill_contract_hash=HASH_F,
        wire_protocol_version=1,
    )


def test_model_swarm_id_is_deterministic_and_contract_sensitive() -> None:
    manifest = make_manifest()
    assert manifest.model_swarm_id == make_manifest().model_swarm_id

    changed = manifest.model_copy(update={"quantization": "int8"})
    assert changed.model_swarm_id != manifest.model_swarm_id


def test_worker_offer_accepts_non_executor_contribution() -> None:
    offer = WorkerOffer(
        worker_id="worker-small",
        endpoint_id="iroh-endpoint",
        runtime_version="3.0.0-dev",
        platform="darwin-arm64",
        backend=BackendKind.MLX,
        stable_memory_envelope_bytes=512 * 1024**2,
        supported_roles={WorkerRole.WEIGHT_SEEDER},
        offer_seq=1,
        issued_at_ms=1_000,
        expires_at_ms=2_000,
    )
    assert offer.supported_roles == frozenset({WorkerRole.WEIGHT_SEEDER})


def test_worker_offer_requires_a_real_role_and_future_expiry() -> None:
    base = dict(
        worker_id="worker",
        endpoint_id="endpoint",
        runtime_version="3.0.0-dev",
        platform="windows-amd64",
        backend=BackendKind.VLLM,
        stable_memory_envelope_bytes=1024,
        offer_seq=0,
        issued_at_ms=2_000,
        expires_at_ms=2_000,
    )
    with pytest.raises(ValidationError):
        WorkerOffer(**base, supported_roles=set())


def test_kv_geometry_rounds_per_worker_block_size() -> None:
    geometry = KvGeometry(
        block_size_tokens=16,
        bytes_per_token_per_layer=4096,
        allocatable_bytes=2 * 1024**3,
    )
    span = LayerSpan(start=4, end=28)
    assert geometry.rounded_tokens(16_316) == 16_320
    assert geometry.required_bytes(span, 16_316) == 16_320 * 4096 * 24


def test_span_lease_rejects_impossible_kv_snapshot() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        SpanLease(
            model_swarm_id=HASH_A,
            worker_id="worker",
            hosted_span=LayerSpan(start=0, end=4),
            effective_span_mode=EffectiveSpanMode.SUBSPAN,
            state=SpanState.READY,
            weight_hashes=(HASH_B,),
            kv_geometry=KvGeometry(
                block_size_tokens=16,
                bytes_per_token_per_layer=4096,
                allocatable_bytes=1024,
            ),
            available_kv_bytes_snapshot=1025,
            max_sessions=1,
            lease_seq=1,
            issued_at_ms=1_000,
            expires_at_ms=2_000,
        )


def make_route() -> RoutePlan:
    return RoutePlan(
        request_id="request-1",
        route_id="route-1",
        epoch=0,
        model_swarm_id=HASH_A,
        model_num_layers=28,
        prompt_tokens=12_220,
        reserved_output_tokens=4_096,
        recovery_level=RecoveryLevel.RESTARTABLE,
        coordinator_id="planner-1",
        reservation_deadline_ms=2_000,
        plan_expires_at_ms=3_000,
        stages=(
            RouteStage(
                worker_id="mac",
                endpoint_id="mac-endpoint",
                hosted_span=LayerSpan(start=0, end=28),
                effective_span=LayerSpan(start=0, end=4),
                path_to_next=PathKind.DIRECT,
                rounded_context_tokens=16_320,
                exact_kv_bytes=16_320 * 4096 * 4,
            ),
            RouteStage(
                worker_id="rtx",
                endpoint_id="rtx-endpoint",
                hosted_span=LayerSpan(start=4, end=28),
                effective_span=LayerSpan(start=4, end=28),
                path_to_next=PathKind.DIRECT,
                rounded_context_tokens=16_320,
                exact_kv_bytes=16_320 * 4096 * 24,
            ),
        ),
    )


def test_route_requires_an_exact_complete_cover() -> None:
    route = make_route()
    assert route.required_context_tokens == 16_316

    broken_stages = (
        route.stages[0],
        route.stages[1].model_copy(update={"effective_span": LayerSpan(start=5, end=28)}),
    )
    with pytest.raises(ValidationError, match="contiguous"):
        route.model_copy(update={"stages": broken_stages}, deep=True).model_validate(
            {**route.model_dump(), "stages": broken_stages}
        )


def test_route_rejects_context_larger_than_any_stage_reservation() -> None:
    route = make_route()
    small_stage = route.stages[0].model_copy(update={"rounded_context_tokens": 16_315})
    with pytest.raises(ValidationError, match="smaller"):
        RoutePlan.model_validate({**route.model_dump(), "stages": (small_stage, route.stages[1])})


def test_effective_span_must_be_hosted() -> None:
    with pytest.raises(ValidationError, match="contained"):
        RouteStage(
            worker_id="worker",
            endpoint_id="endpoint",
            hosted_span=LayerSpan(start=4, end=28),
            effective_span=LayerSpan(start=0, end=4),
            path_to_next=PathKind.RELAY,
            rounded_context_tokens=1024,
            exact_kv_bytes=1024,
        )
