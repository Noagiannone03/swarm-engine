from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from swarm_protocol import (
    BackendKind,
    CatalogRecordConflict,
    EffectiveSpanMode,
    InMemoryDiscoveryStore,
    KvGeometry,
    LayerSpan,
    LinkMetric,
    ModelManifest,
    PathKind,
    SpanLease,
    SpanState,
    StaleCatalogRecord,
    WorkerOffer,
    WorkerRole,
)

HASHES = tuple(character * 64 for character in "abcdef")
GIB = 1024**3


def manifest(model_id: str = "Qwen/Qwen3-8B") -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        immutable_revision="0123456789abcdef",
        architecture_graph_hash=HASHES[0],
        tokenizer_hash=HASHES[1],
        weight_collection_hash=HASHES[2],
        weight_format="safetensors",
        quantization="bf16",
        dtype="bfloat16",
        num_layers=28,
        model_max_context_tokens=65_536,
        context_classes=(4_096, 8_192, 16_384, 32_768, 65_536),
        activation_bytes_per_token=4096,
        kv_bytes_per_token_by_layer=(512,) * 28,
        rope_context_contract_hash=HASHES[3],
        attention_kv_contract_hash=HASHES[4],
        prefill_contract_hash=HASHES[5],
        wire_protocol_version=1,
    )


def offer(worker_id: str, *, sequence: int, expires_at_ms: int = 10_000) -> WorkerOffer:
    return WorkerOffer(
        worker_id=worker_id,
        endpoint_id=f"{worker_id}-endpoint",
        runtime_version="3.0.0-dev",
        platform="darwin-arm64" if worker_id == "mac" else "windows-amd64",
        backend=BackendKind.MLX if worker_id == "mac" else BackendKind.VLLM,
        stable_memory_envelope_bytes=8 * GIB,
        supported_roles={WorkerRole.EXECUTOR, WorkerRole.FRONTEND},
        offer_seq=sequence,
        issued_at_ms=1_000 + sequence,
        expires_at_ms=expires_at_ms,
    )


def lease(
    model_swarm_id: str,
    worker_id: str,
    *,
    sequence: int,
    start: int = 0,
    end: int = 28,
    expires_at_ms: int = 10_000,
) -> SpanLease:
    return SpanLease(
        model_swarm_id=model_swarm_id,
        worker_id=worker_id,
        hosted_span=LayerSpan(start=start, end=end),
        effective_span_mode=EffectiveSpanMode.SUBSPAN,
        state=SpanState.READY,
        weight_hashes=(HASHES[0],),
        measured_prefill_tokens_per_second=100,
        measured_decode_tokens_per_second=20,
        max_context_tokens=65_536,
        kv_geometry=KvGeometry(
            block_size_tokens=16,
            bytes_per_token_per_layer=1024,
            allocatable_bytes=4 * GIB,
        ),
        available_kv_bytes_snapshot=4 * GIB,
        max_sessions=4,
        lease_seq=sequence,
        issued_at_ms=1_000 + sequence,
        expires_at_ms=expires_at_ms,
    )


def link(
    source: str,
    target: str,
    *,
    measured_at_ms: int,
    expires_at_ms: int = 10_000,
) -> LinkMetric:
    return LinkMetric(
        from_worker_id=source,
        to_worker_id=target,
        path_kind=PathKind.DIRECT,
        rtt_ms=5,
        throughput_bytes_per_second=100_000_000,
        measured_at_ms=measured_at_ms,
        expires_at_ms=expires_at_ms,
    )


def test_publication_is_idempotent_and_rejects_sequence_rollback() -> None:
    store = InMemoryDiscoveryStore()
    current = offer("mac", sequence=2)
    assert store.publish_offer(current) is True
    assert store.publish_offer(current) is False

    with pytest.raises(StaleCatalogRecord, match="older than watermark"):
        store.publish_offer(offer("mac", sequence=1))

    with pytest.raises(CatalogRecordConflict, match="reused"):
        store.publish_offer(current.model_copy(update={"endpoint_id": "impostor"}))


def test_new_span_lease_replaces_one_worker_span_without_touching_others() -> None:
    model = manifest()
    store = InMemoryDiscoveryStore()
    for worker_id in ("mac", "rtx"):
        store.publish_offer(offer(worker_id, sequence=1))
    store.publish_span_lease(lease(model.model_swarm_id, "mac", sequence=1, end=4))
    store.publish_span_lease(lease(model.model_swarm_id, "rtx", sequence=1, start=4))

    moved = lease(model.model_swarm_id, "mac", sequence=2, end=8)
    store.publish_span_lease(moved)
    snapshot = store.snapshot(model_swarm_id=model.model_swarm_id, now_ms=2_000)

    assert [(item.worker_id, item.hosted_span) for item in snapshot.leases] == [
        ("mac", LayerSpan(start=0, end=8)),
        ("rtx", LayerSpan(start=4, end=28)),
    ]


def test_snapshot_filters_expired_and_orphaned_soft_state() -> None:
    model = manifest()
    store = InMemoryDiscoveryStore()
    store.publish_manifest(model)
    store.publish_offer(offer("mac", sequence=1, expires_at_ms=2_000))
    store.publish_offer(offer("rtx", sequence=1))
    store.publish_span_lease(lease(model.model_swarm_id, "mac", sequence=1))
    store.publish_span_lease(lease(model.model_swarm_id, "rtx", sequence=1))
    store.publish_span_lease(lease(model.model_swarm_id, "orphan", sequence=1))
    store.publish_link(link("mac", "rtx", measured_at_ms=1_100))
    store.publish_link(link("rtx", "mac", measured_at_ms=1_100))

    snapshot = store.snapshot(model_swarm_id=model.model_swarm_id, now_ms=2_000)

    assert snapshot.manifest(model.model_swarm_id) == model
    assert [item.worker_id for item in snapshot.offers] == ["rtx"]
    assert [item.worker_id for item in snapshot.leases] == ["rtx"]
    assert snapshot.links == ()


def test_expiry_collection_does_not_allow_delayed_record_resurrection() -> None:
    store = InMemoryDiscoveryStore()
    store.publish_offer(offer("mac", sequence=5, expires_at_ms=2_000))
    assert store.collect_expired(now_ms=2_000) == 1
    assert store.snapshot(now_ms=2_000).offers == ()

    with pytest.raises(StaleCatalogRecord):
        store.publish_offer(offer("mac", sequence=4, expires_at_ms=10_000))

    assert store.publish_offer(offer("mac", sequence=6, expires_at_ms=10_000)) is True


def test_link_observations_are_monotonic_and_expire() -> None:
    store = InMemoryDiscoveryStore()
    store.publish_offer(offer("mac", sequence=1))
    store.publish_offer(offer("rtx", sequence=1))
    newest = link("mac", "rtx", measured_at_ms=1_500, expires_at_ms=3_000)
    assert store.publish_link(newest) is True
    assert store.publish_link(newest) is False

    with pytest.raises(StaleCatalogRecord):
        store.publish_link(link("mac", "rtx", measured_at_ms=1_400))
    with pytest.raises(CatalogRecordConflict):
        store.publish_link(newest.model_copy(update={"rtt_ms": 99}))

    assert store.snapshot(now_ms=2_999).links == (newest,)
    assert store.snapshot(now_ms=3_000).links == ()


def test_model_filter_and_order_are_deterministic() -> None:
    first = manifest("Qwen/Qwen3-8B")
    second = manifest("Qwen/Qwen3-14B")
    store = InMemoryDiscoveryStore()
    for item in (second, first):
        store.publish_manifest(item)
    for worker_id in ("zeta", "alpha", "middle"):
        store.publish_offer(offer(worker_id, sequence=1))
        store.publish_span_lease(lease(first.model_swarm_id, worker_id, sequence=1))
        store.publish_span_lease(lease(second.model_swarm_id, worker_id, sequence=1))

    snapshot = store.snapshot(model_swarm_id=first.model_swarm_id, now_ms=2_000)
    assert snapshot.manifests == (first,)
    assert [item.worker_id for item in snapshot.offers] == ["alpha", "middle", "zeta"]
    assert {item.model_swarm_id for item in snapshot.leases} == {first.model_swarm_id}


def test_concurrent_reordered_publication_converges_to_highest_sequence() -> None:
    store = InMemoryDiscoveryStore()

    def publish(sequence: int) -> None:
        try:
            store.publish_offer(offer("mac", sequence=sequence))
        except StaleCatalogRecord:
            pass

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(publish, range(1, 101)))

    assert store.snapshot(now_ms=2_000).offers[0].offer_seq == 100
