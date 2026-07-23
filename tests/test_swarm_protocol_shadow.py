import hashlib
import time
from types import SimpleNamespace

from swarm_protocol import (
    ArtifactDescriptor,
    ArtifactRole,
    BackendKind,
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    LinkMetric,
    ModelArtifactIndex,
    ModelManifest,
    ModelMemberAdvertisement,
    ModelRegistryBundle,
    PathKind,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
    artifact_collection_hash,
)
from swarm_protocol.shadow import SchedulerProtocolV3Shadow


def _bundle():
    descriptors = tuple(
        ArtifactDescriptor(
            path=path,
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            media_type="application/octet-stream",
            role=role,
        )
        for path, content, role in (
            ("config.json", b"config", ArtifactRole.ARCHITECTURE),
            ("model.safetensors", b"weights", ArtifactRole.WEIGHT),
            ("tokenizer.json", b"tokenizer", ArtifactRole.TOKENIZER),
        )
    )
    index = ModelArtifactIndex(
        model_id="test/model",
        immutable_revision="revision",
        artifacts=descriptors,
    )
    manifest = ModelManifest(
        model_id=index.model_id,
        immutable_revision=index.immutable_revision,
        architecture_graph_hash=artifact_collection_hash(index, ArtifactRole.ARCHITECTURE),
        tokenizer_hash=artifact_collection_hash(index, ArtifactRole.TOKENIZER),
        weight_collection_hash=artifact_collection_hash(index, ArtifactRole.WEIGHT),
        weight_format="safetensors",
        quantization="unquantized",
        dtype="bfloat16",
        num_layers=4,
        activation_bytes_per_token=128,
        kv_bytes_per_token_by_layer=(64,) * 4,
        rope_context_contract_hash="1" * 64,
        attention_kv_contract_hash="2" * 64,
        prefill_contract_hash="3" * 64,
        wire_protocol_version=1,
    )
    return ModelRegistryBundle(manifest=manifest, artifact_index=index)


class _Registry:
    def __init__(self, bundle):
        self.bundle = bundle

    def fetch(self, model_swarm_id):
        assert model_swarm_id == self.bundle.model_swarm_id
        return self.bundle


def _advertisement(bundle, worker_id, start, end, target, *, measured=True):
    now_ms = time.time_ns() // 1_000_000
    offer = WorkerOffer(
        worker_id=worker_id,
        endpoint_id=worker_id,
        runtime_version="test",
        platform="test",
        backend=BackendKind.MLX,
        stable_memory_envelope_bytes=8 * 1024**3,
        supported_roles=frozenset({WorkerRole.EXECUTOR, WorkerRole.FRONTEND}),
        offer_seq=1,
        issued_at_ms=now_ms,
        expires_at_ms=now_ms + 60_000,
    )
    lease = SpanLease(
        model_swarm_id=bundle.model_swarm_id,
        worker_id=worker_id,
        hosted_span=LayerSpan(start=start, end=end),
        effective_span_mode=EffectiveSpanMode.FIXED,
        state=SpanState.READY,
        weight_hashes=("a" * 64,),
        measured_prefill_tokens_per_second=1000 if measured else None,
        measured_decode_tokens_per_second=50 if measured else None,
        kv_geometry=KvGeometry(
            block_size_tokens=16,
            bytes_per_token_by_layer=(64,) * 4,
            allocatable_bytes=1024**3,
        ),
        available_kv_bytes_snapshot=1024**3,
        max_sessions=2,
        lease_seq=1,
        issued_at_ms=now_ms,
        expires_at_ms=now_ms + 60_000,
    )
    link = LinkMetric(
        from_worker_id=worker_id,
        to_worker_id=target,
        path_kind=PathKind.DIRECT,
        rtt_ms=2,
        throughput_bytes_per_second=100 * 1024**2,
        measured_at_ms=now_ms,
        expires_at_ms=now_ms + 60_000,
    )
    return ModelMemberAdvertisement(offer=offer, lease=lease, outgoing_links=(link,))


def _nodes(bundle, *, measured=True):
    mac = _advertisement(bundle, "mac", 0, 2, "rtx", measured=measured)
    rtx = _advertisement(bundle, "rtx", 2, 4, "mac", measured=measured)
    return [
        SimpleNamespace(
            node_id="mac",
            start_layer=0,
            end_layer=2,
            is_active=True,
            swarm_v3={"state": "ready", "advertisement": mac.model_dump(mode="json")},
        ),
        SimpleNamespace(
            node_id="rtx",
            start_layer=2,
            end_layer=4,
            is_active=True,
            swarm_v3={"state": "ready", "advertisement": rtx.model_dump(mode="json")},
        ),
    ]


def _observe_until_resolved(shadow, nodes):
    deadline = time.monotonic() + 2
    result = shadow.observe(
        nodes,
        model_num_layers=4,
        planning_context_tokens=512,
        epoch=1,
    )
    while result["state"] == "verifying_registry" and time.monotonic() < deadline:
        time.sleep(0.01)
        result = shadow.observe(
            nodes,
            model_num_layers=4,
            planning_context_tokens=512,
            epoch=1,
        )
    return result


def test_shadow_compares_v3_route_with_legacy_without_serving_it():
    bundle = _bundle()
    result = _observe_until_resolved(
        SchedulerProtocolV3Shadow(_Registry(bundle)),
        _nodes(bundle),
    )

    assert result["state"] == "agreement"
    assert result["v3_route"] == ("mac", "rtx")
    assert result["legacy_routes"] == (("mac", "rtx"),)


def test_shadow_explains_missing_executor_measurements():
    bundle = _bundle()
    result = _observe_until_resolved(
        SchedulerProtocolV3Shadow(_Registry(bundle)),
        _nodes(bundle, measured=False),
    )

    assert result["state"] == "no_feasible_route"
    assert result["blockers"] == ["missing_executor_throughput"]
