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
        model_max_context_tokens=65_536,
        context_classes=(4_096, 8_192, 16_384, 32_768, 65_536),
        activation_bytes_per_token=4096,
        kv_bytes_per_token_by_layer=(512,) * num_layers,
        rope_context_contract_hash=HASHES[3],
        attention_kv_contract_hash=HASHES[4],
        prefill_contract_hash=HASHES[5],
        wire_protocol_version=1,
    )


def offer(
    worker_id: str,
    *,
    frontend: bool = True,
    backend: BackendKind = BackendKind.MLX,
) -> WorkerOffer:
    roles = {WorkerRole.EXECUTOR}
    if frontend:
        roles.add(WorkerRole.FRONTEND)
    return WorkerOffer(
        worker_id=worker_id,
        endpoint_id=f"{worker_id}-endpoint",
        runtime_version="3.0.0-dev",
        platform="test",
        backend=backend,
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
    prefill_tps: float | None = 10_000,
    decode_tps: float | None = 100,
    max_context_tokens: int = 65_536,
    expires_at_ms: int = 10_000,
    execution_plan_identity_hash: str | None = None,
    activation_bytes_per_token: int | None = None,
) -> SpanLease:
    return SpanLease(
        model_swarm_id=model.model_swarm_id,
        worker_id=worker_id,
        hosted_span=LayerSpan(start=start, end=end),
        effective_span_mode=mode,
        state=SpanState.READY,
        weight_hashes=(HASHES[0],),
        execution_plan_identity_hash=execution_plan_identity_hash,
        activation_bytes_per_token=activation_bytes_per_token,
        measured_prefill_tokens_per_second=prefill_tps,
        measured_decode_tokens_per_second=decode_tps,
        max_context_tokens=max_context_tokens,
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


def request(
    model: ModelManifest,
    *,
    prompt: int = 100,
    output: int = 20,
    recovery: RecoveryLevel = RecoveryLevel.RESTARTABLE,
) -> RequestContract:
    return RequestContract(
        request_id="request",
        model_swarm_id=model.model_swarm_id,
        prompt_tokens=prompt,
        reserved_output_tokens=output,
        recovery_level=recovery,
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
        [offer("mac"), offer("rtx", frontend=False)],
        [lease(model, "mac", 0, 4), lease(model, "rtx", 4, 8)],
        [link("mac", "rtx"), link("rtx", "mac")],
    )
    assert [stage.worker_id for stage in result.plan.stages] == ["mac", "rtx"]
    assert [stage.effective_span for stage in result.plan.stages] == [
        LayerSpan(start=0, end=4),
        LayerSpan(start=4, end=8),
    ]
    assert {stage.rounded_context_tokens for stage in result.plan.stages} == {128}


def test_authenticated_links_remain_routable_without_fresh_goodput() -> None:
    model = manifest()
    unknown_goodput = [
        link("mac", "rtx").model_copy(
            update={
                "throughput_bytes_per_second": None,
                "throughput_measured_at_ms": None,
            }
        ),
        link("rtx", "mac").model_copy(
            update={
                "throughput_bytes_per_second": None,
                "throughput_measured_at_ms": None,
            }
        ),
    ]

    result = plan(
        model,
        request(model),
        [offer("mac"), offer("rtx", frontend=False)],
        [lease(model, "mac", 0, 4), lease(model, "rtx", 4, 8)],
        unknown_goodput,
    )

    assert [stage.worker_id for stage in result.plan.stages] == ["mac", "rtx"]
    assert result.estimate.complete is False


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


def test_per_session_context_ceiling_wins_over_large_aggregate_kv_capacity() -> None:
    model = manifest()
    req = request(model, prompt=16_000, output=721)

    with pytest.raises(NoFeasibleRoute):
        plan(
            model,
            req,
            [offer("rtx")],
            [
                lease(
                    model,
                    "rtx",
                    0,
                    8,
                    kv_bytes=10**12,
                    max_context_tokens=16_384,
                )
            ],
            [],
        )


def test_signed_model_context_limit_wins_over_worker_claim() -> None:
    model = manifest().model_copy(
        update={
            "model_max_context_tokens": 16_384,
            "context_classes": (4_096, 8_192, 16_384),
        }
    )
    req = request(model, prompt=16_000, output=721)

    with pytest.raises(NoFeasibleRoute, match="signed model context limit"):
        plan(
            model,
            req,
            [offer("worker")],
            [lease(model, "worker", 0, 8, max_context_tokens=65_536)],
            [],
        )


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


def test_route_estimate_counts_half_rtt_per_directed_edge() -> None:
    model = manifest()
    result = plan(
        model,
        request(model, prompt=100, output=20),
        [offer("head"), offer("tail", frontend=False)],
        [lease(model, "head", 0, 4), lease(model, "tail", 4, 8)],
        [
            link("head", "tail", rtt_ms=20),
            link("tail", "head", rtt_ms=20),
        ],
    )

    # Compute: 10 ms per hosted stage for prefill/decode. Network: one-way
    # latency is 10 ms per directed edge; activation transfer adds 3.90625
    # ms for prefill and 0.0390625 ms for each decode edge.
    assert result.estimate.ttft_ms == pytest.approx(33.90625)
    assert result.estimate.inter_token_ms == pytest.approx(40.078125)


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


def test_cold_worker_without_throughput_is_admissible_but_estimate_is_unknown() -> None:
    model = manifest()
    result = plan(
        model,
        request(model),
        [offer("cold")],
        [lease(model, "cold", 0, 8, prefill_tps=None, decode_tps=None)],
        [],
    )

    assert [stage.worker_id for stage in result.plan.stages] == ["cold"]
    assert result.estimate.complete is False


def test_recoverable_route_reserves_a_worker_disjoint_complete_backup() -> None:
    model = manifest()
    result = plan(
        model,
        request(model, recovery=RecoveryLevel.RECOVERABLE),
        [
            offer("primary-head"),
            offer("primary-tail", frontend=False),
            offer("backup-head"),
            offer("backup-tail", frontend=False),
        ],
        [
            lease(model, "primary-head", 0, 4),
            lease(model, "primary-tail", 4, 8),
            lease(model, "backup-head", 0, 4),
            lease(model, "backup-tail", 4, 8),
        ],
        [
            link("primary-head", "primary-tail", rtt_ms=1),
            link("primary-tail", "primary-head", rtt_ms=1),
            link("backup-head", "backup-tail", rtt_ms=20),
            link("backup-tail", "backup-head", rtt_ms=20),
        ],
    )

    assert [stage.worker_id for stage in result.plan.stages] == [
        "primary-head",
        "primary-tail",
    ]
    assert result.recovery_plan is not None
    assert [stage.worker_id for stage in result.recovery_plan.stages] == [
        "backup-head",
        "backup-tail",
    ]
    assert result.recovery_plan.route_id == "route-recovery"
    assert result.recovery_estimate is not None


def test_recoverable_route_fails_closed_without_disjoint_backup_coverage() -> None:
    model = manifest()
    with pytest.raises(NoFeasibleRoute, match="worker-disjoint"):
        plan(
            model,
            request(model, recovery=RecoveryLevel.RECOVERABLE),
            [offer("head"), offer("tail", frontend=False)],
            [lease(model, "head", 0, 4), lease(model, "tail", 4, 8)],
            [link("head", "tail"), link("tail", "head")],
        )


def test_portable_route_rejects_mixed_execution_plan_identities() -> None:
    model = manifest(num_layers=2)
    with pytest.raises(NoFeasibleRoute, match="no complete route"):
        plan(
            model,
            request(model),
            [
                offer("head", backend=BackendKind.SKIPPY),
                offer("tail", frontend=False, backend=BackendKind.SKIPPY),
            ],
            [
                lease(
                    model,
                    "head",
                    0,
                    1,
                    execution_plan_identity_hash="a" * 64,
                    activation_bytes_per_token=4096,
                ),
                lease(
                    model,
                    "tail",
                    1,
                    2,
                    execution_plan_identity_hash="b" * 64,
                    activation_bytes_per_token=4096,
                ),
            ],
            [link("head", "tail"), link("tail", "head")],
        )


def test_portable_route_uses_signed_activation_wire_geometry() -> None:
    model = manifest()
    metric = link("head", "tail", rtt_ms=0, throughput=1024)
    request_contract = request(model, prompt=10, output=1)

    f32 = ExactRoutePlanner()._link_cost(metric, request_contract, 4096)
    bf16 = ExactRoutePlanner()._link_cost(metric, request_contract, 2048)

    assert f32.ttft_ms == pytest.approx(bf16.ttft_ms * 2)
    assert f32.inter_token_ms == pytest.approx(bf16.inter_token_ms * 2)
