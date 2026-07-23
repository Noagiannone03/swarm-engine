import pytest

from swarm_protocol import (
    BackendKind,
    EffectiveSpanMode,
    ExactRoutePlanner,
    KvGeometry,
    LayerSpan,
    LinkMetric,
    ModelManifest,
    NoFeasibleRoute,
    PathKind,
    RecoveryLevel,
    RequestContract,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)

HASHES = tuple(f"{digit:x}" * 64 for digit in range(1, 7))


def manifest(num_layers: int = 8) -> ModelManifest:
    return ModelManifest(
        model_id="fabi/test",
        immutable_revision="revision",
        architecture_graph_hash=HASHES[0],
        tokenizer_hash=HASHES[1],
        weight_collection_hash=HASHES[2],
        weight_format="safetensors",
        quantization="bf16",
        dtype="bfloat16",
        num_layers=num_layers,
        activation_bytes_per_token=4096,
        kv_bytes_per_token_by_layer=(512,) * num_layers,
        rope_context_contract_hash=HASHES[3],
        attention_kv_contract_hash=HASHES[4],
        prefill_contract_hash=HASHES[5],
        wire_protocol_version=1,
    )


def offer(worker_id: str, *, frontend: bool = True) -> WorkerOffer:
    roles = {WorkerRole.EXECUTOR}
    if frontend:
        roles.add(WorkerRole.FRONTEND)
    return WorkerOffer(
        worker_id=worker_id,
        endpoint_id=f"{worker_id}-endpoint",
        runtime_version="3.0.0-dev",
        platform="test",
        backend=BackendKind.MLX,
        stable_memory_envelope_bytes=16 * 1024**3,
        supported_roles=roles,
        offer_seq=1,
        issued_at_ms=1,
        expires_at_ms=10_000,
    )


def lease(
    model: ModelManifest,
    worker_id: str,
    start: int,
    end: int,
    *,
    kv_bytes: int = 10**12,
    mode: EffectiveSpanMode = EffectiveSpanMode.SUBSPAN,
    prefill_tps: float = 10_000,
    decode_tps: float = 100,
    expires_at_ms: int = 10_000,
) -> SpanLease:
    return SpanLease(
        model_swarm_id=model.model_swarm_id,
        worker_id=worker_id,
        hosted_span=LayerSpan(start=start, end=end),
        effective_span_mode=mode,
        state=SpanState.READY,
        weight_hashes=(HASHES[0],),
        measured_prefill_tokens_per_second=prefill_tps,
        measured_decode_tokens_per_second=decode_tps,
        kv_geometry=KvGeometry(
            block_size_tokens=16,
            bytes_per_token_per_layer=4096,
            allocatable_bytes=kv_bytes,
        ),
        available_kv_bytes_snapshot=kv_bytes,
        max_sessions=1,
        lease_seq=1,
        issued_at_ms=1,
        expires_at_ms=expires_at_ms,
    )


def link(
    source: str,
    target: str,
    *,
    path: PathKind = PathKind.DIRECT,
    rtt_ms: float = 2,
    throughput: float = 100 * 1024**2,
) -> LinkMetric:
    return LinkMetric(
        from_worker_id=source,
        to_worker_id=target,
        path_kind=path,
        rtt_ms=rtt_ms,
        throughput_bytes_per_second=throughput,
        measured_at_ms=1,
        expires_at_ms=10_000,
    )


def request(model: ModelManifest, *, prompt: int = 100, output: int = 20) -> RequestContract:
    return RequestContract(
        request_id="request",
        model_swarm_id=model.model_swarm_id,
        prompt_tokens=prompt,
        reserved_output_tokens=output,
        recovery_level=RecoveryLevel.RESTARTABLE,
    )


def plan(model, req, offers, leases, links):
    return ExactRoutePlanner().plan(
        manifest=model,
        request=req,
        offers=tuple(offers),
        leases=tuple(leases),
        links=tuple(links),
        snapshot_time_ms=100,
        coordinator_id="planner",
        reservation_deadline_ms=200,
        plan_expires_at_ms=300,
        route_id="route",
    )


def test_builds_exact_route_and_rounds_context_per_stage() -> None:
    model = manifest()
    result = plan(
        model,
        request(model, prompt=100, output=21),
        [offer("mac"), offer("rtx")],
        [lease(model, "mac", 0, 4), lease(model, "rtx", 4, 8)],
        [link("mac", "rtx"), link("rtx", "mac")],
    )
    assert [stage.worker_id for stage in result.plan.stages] == ["mac", "rtx"]
    assert [stage.effective_span for stage in result.plan.stages] == [
        LayerSpan(start=0, end=4),
        LayerSpan(start=4, end=8),
    ]
    assert {stage.rounded_context_tokens for stage in result.plan.stages} == {128}


def test_long_context_excludes_worker_with_insufficient_kv() -> None:
    model = manifest()
    req = request(model, prompt=100, output=28)
    one_layer_bytes = 128 * 4096
    result = plan(
        model,
        req,
        [offer("small"), offer("large")],
        [
            lease(model, "small", 0, 8, kv_bytes=one_layer_bytes),
            lease(model, "large", 0, 8, kv_bytes=one_layer_bytes * 8),
        ],
        [],
    )
    assert [stage.worker_id for stage in result.plan.stages] == ["large"]


def test_fixed_span_cannot_be_truncated_to_create_theoretical_route() -> None:
    model = manifest()
    with pytest.raises(NoFeasibleRoute):
        plan(
            model,
            request(model),
            [offer("fixed"), offer("tail")],
            [
                lease(
                    model,
                    "fixed",
                    0,
                    8,
                    kv_bytes=4 * 128 * 4096,
                    mode=EffectiveSpanMode.FIXED,
                ),
                lease(model, "tail", 4, 8),
            ],
            [link("fixed", "tail"), link("tail", "fixed")],
        )


def test_missing_reverse_closure_link_rejects_pipeline() -> None:
    model = manifest()
    with pytest.raises(NoFeasibleRoute):
        plan(
            model,
            request(model),
            [offer("head"), offer("tail")],
            [lease(model, "head", 0, 4), lease(model, "tail", 4, 8)],
            [link("head", "tail")],
        )


def test_prefers_direct_route_over_slower_relay_route() -> None:
    model = manifest()
    result = plan(
        model,
        request(model),
        [offer("head"), offer("direct"), offer("relay")],
        [
            lease(model, "head", 0, 4),
            lease(model, "direct", 4, 8),
            lease(model, "relay", 4, 8),
        ],
        [
            link("head", "direct"),
            link("direct", "head"),
            link("head", "relay", path=PathKind.RELAY, rtt_ms=80),
            link("relay", "head", path=PathKind.RELAY, rtt_ms=80),
        ],
    )
    assert [stage.worker_id for stage in result.plan.stages] == ["head", "direct"]


def test_expired_lease_is_not_routable() -> None:
    model = manifest()
    with pytest.raises(NoFeasibleRoute):
        plan(
            model,
            request(model),
            [offer("worker")],
            [lease(model, "worker", 0, 8, expires_at_ms=100)],
            [],
        )


def test_route_is_independent_from_discovery_order() -> None:
    model = manifest()
    offers = [offer("head"), offer("tail")]
    leases = [lease(model, "head", 0, 4), lease(model, "tail", 4, 8)]
    links = [link("head", "tail"), link("tail", "head")]

    forward = plan(model, request(model), offers, leases, links)
    reverse = plan(
        model,
        request(model),
        list(reversed(offers)),
        list(reversed(leases)),
        list(reversed(links)),
    )
    assert forward.plan.stages == reverse.plan.stages
