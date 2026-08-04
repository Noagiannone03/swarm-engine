import hashlib
import threading
import time
from pathlib import Path

from swarm_protocol import (
    ArtifactDescriptor,
    ArtifactRole,
    BackendKind,
    LayerSpan,
    ModelArtifactIndex,
    ModelManifest,
    ModelMemberAdvertisement,
    ModelRegistryBundle,
    artifact_collection_hash,
    SpanState,
)
from swarm_protocol.worker_integration import (
    WorkerProtocolV3Reporter,
    WorkerServingSnapshot,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _bundle(model_root: Path) -> ModelRegistryBundle:
    files = {
        "config.json": (b'{"num_hidden_layers":4}', ArtifactRole.ARCHITECTURE),
        "model.safetensors": (b"weights", ArtifactRole.WEIGHT),
        "tokenizer.json": (b"tokenizer", ArtifactRole.TOKENIZER),
    }
    descriptors = []
    for path, (content, role) in files.items():
        (model_root / path).write_bytes(content)
        descriptors.append(
            ArtifactDescriptor(
                path=path,
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                media_type="application/octet-stream",
                role=role,
            )
        )
    index = ModelArtifactIndex(
        model_id="test/model",
        immutable_revision=REVISION,
        artifacts=tuple(sorted(descriptors, key=lambda descriptor: descriptor.path)),
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
    def __init__(self, bundle, *, delay=0.0):
        self.bundle = bundle
        self.delay = delay

    def resolve(self, model_id, **kwargs):
        time.sleep(self.delay)
        assert model_id == self.bundle.manifest.model_id
        assert kwargs["immutable_revision"] == self.bundle.manifest.immutable_revision
        return self.bundle


def _serving(*, current_requests=0, max_sessions=2, measured=False, span=None):
    return WorkerServingSnapshot(
        worker_id="worker",
        endpoint_id="endpoint",
        model_id="test/model",
        immutable_revision=REVISION,
        span=span or LayerSpan(start=0, end=4),
        backend=BackendKind.MLX,
        stable_memory_envelope_bytes=8 * 1024**3,
        max_context_tokens=32_768,
        kv_cache_token_capacity=1024,
        kv_cache_block_size=16,
        max_sessions=max_sessions,
        is_ready=True,
        current_requests=current_requests,
        supports_frontend=True,
        measured_prefill_tokens_per_second=1000 if measured else None,
        measured_decode_tokens_per_second=50 if measured else None,
    )


def _wait_report(reporter, serving):
    deadline = time.monotonic() + 3
    report = reporter.snapshot(serving)
    while report["state"] == "verifying" and time.monotonic() < deadline:
        time.sleep(0.01)
        report = reporter.snapshot(serving)
    return report


def test_large_artifact_verification_never_blocks_heartbeat(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        "swarm_protocol.worker_integration._local_model_root",
        lambda *args: tmp_path,
    )
    reporter = WorkerProtocolV3Reporter(_Registry(bundle, delay=0.2))

    started = time.monotonic()
    report = reporter.snapshot(_serving())
    elapsed = time.monotonic() - started

    assert report["state"] == "verifying"
    assert elapsed < 0.05
    ready = _wait_report(reporter, _serving())
    assert ready["state"] == "ready"
    advertisement = ready["advertisement"]
    assert set(advertisement["offer"]["supported_roles"]) == {"executor", "frontend"}
    assert advertisement["lease"]["weight_hashes"]
    assert advertisement["lease"]["max_context_tokens"] == 32_768
    assert advertisement["lease"]["available_kv_bytes_snapshot"] == 1024 * 64 * 4


def test_active_legacy_request_makes_shadow_kv_advertisement_fail_closed(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        "swarm_protocol.worker_integration._local_model_root",
        lambda *args: tmp_path,
    )
    reporter = WorkerProtocolV3Reporter(_Registry(bundle))
    ready = _wait_report(reporter, _serving(current_requests=1))

    assert ready["state"] == "ready"
    assert ready["advertisement"]["lease"]["available_kv_bytes_snapshot"] == 0


def test_single_session_worker_advertises_measured_executor_throughput(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        "swarm_protocol.worker_integration._local_model_root",
        lambda *args: tmp_path,
    )
    reporter = WorkerProtocolV3Reporter(_Registry(bundle))
    ready = _wait_report(reporter, _serving(max_sessions=1, measured=True))

    lease = ready["advertisement"]["lease"]
    assert lease["measured_prefill_tokens_per_second"] == 1000
    assert lease["measured_decode_tokens_per_second"] == 50


def test_transition_refreshes_offer_and_lease_together(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        "swarm_protocol.worker_integration._local_model_root",
        lambda *args: tmp_path,
    )
    reporter = WorkerProtocolV3Reporter(_Registry(bundle))
    ready = _wait_report(reporter, _serving())
    advertisement = ModelMemberAdvertisement.model_validate(ready["advertisement"])

    transitioned = reporter.publish_span_state(advertisement, SpanState.BUILDING)

    assert transitioned.offer.offer_seq > advertisement.offer.offer_seq
    assert transitioned.offer.issued_at_ms >= advertisement.offer.issued_at_ms
    assert transitioned.offer.expires_at_ms >= advertisement.offer.expires_at_ms
    assert transitioned.lease.lease_seq > advertisement.lease.lease_seq
    assert transitioned.lease.issued_at_ms >= advertisement.lease.issued_at_ms
    assert transitioned.lease.expires_at_ms >= advertisement.lease.expires_at_ms
    assert transitioned.lease.state is SpanState.BUILDING
    assert transitioned.lease.available_kv_bytes_snapshot == 0


def test_reallocation_never_starts_concurrent_checkpoint_hashers(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        "swarm_protocol.worker_integration._local_model_root",
        lambda *args: tmp_path,
    )
    entered = threading.Event()
    release = threading.Event()

    class TrackingRegistry(_Registry):
        def __init__(self, value):
            super().__init__(value)
            self.calls = 0
            self.active = 0
            self.max_active = 0

        def resolve(self, model_id, **kwargs):
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            entered.set()
            release.wait(timeout=2)
            self.active -= 1
            return super().resolve(model_id, **kwargs)

    registry = TrackingRegistry(bundle)
    reporter = WorkerProtocolV3Reporter(registry)
    reporter.snapshot(_serving(span=LayerSpan(start=0, end=2)))
    assert entered.wait(timeout=1)
    for _ in range(10):
        reporter.snapshot(_serving(span=LayerSpan(start=2, end=4)))
    assert registry.calls == 1

    release.set()
    deadline = time.monotonic() + 2
    while registry.active and time.monotonic() < deadline:
        time.sleep(0.01)
    _wait_report(reporter, _serving(span=LayerSpan(start=2, end=4)))

    assert registry.calls == 2
    assert registry.max_active == 1
