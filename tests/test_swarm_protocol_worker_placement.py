from __future__ import annotations

import time

from swarm_protocol import (
    AutonomousPlacementPolicy,
    AutonomousWorkerPlacement,
    BackendKind,
    DiscoverySnapshot,
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    ModelManifest,
    ModelMemberAdvertisement,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)

HASHES = tuple(character * 64 for character in "abcdef")
NOW = 1_000


def model() -> ModelManifest:
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
        rope_context_contract_hash=HASHES[3],
        attention_kv_contract_hash=HASHES[4],
        prefill_contract_hash=HASHES[5],
        wire_protocol_version=1,
    )


def advertisement(
    manifest: ModelManifest,
    worker_id: str,
    start: int,
    end: int,
) -> ModelMemberAdvertisement:
    return ModelMemberAdvertisement(
        offer=WorkerOffer(
            worker_id=worker_id,
            endpoint_id=f"{worker_id}-endpoint",
            runtime_version="test",
            platform="test",
            backend=BackendKind.MLX,
            stable_memory_envelope_bytes=500,
            supported_roles={WorkerRole.EXECUTOR, WorkerRole.FRONTEND},
            offer_seq=1,
            issued_at_ms=NOW,
            expires_at_ms=NOW + 60_000,
        ),
        lease=SpanLease(
            model_swarm_id=manifest.model_swarm_id,
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
            issued_at_ms=NOW,
            expires_at_ms=NOW + 60_000,
        ),
    )


class FakeAdmission:
    def __init__(self) -> None:
        self.configured = []
        self.draining = False

    def configure(self, value):
        self.configured.append(value)

    def snapshot(self):
        return ()

    def begin_drain(self):
        self.draining = True
        return 0

    def draining_reservations(self):
        return 0

    def cancel_drain(self):
        self.draining = False

    def finish_drain(self):
        self.draining = False


class FakePublisher:
    def __init__(self) -> None:
        self.states = []

    def publish_span_state(self, value, state):
        self.states.append(state)
        return value


class FakeCatalog:
    def __init__(self, value) -> None:
        self.value = value

    def snapshot(self, *, model_swarm_id=None, now_ms=None):
        del now_ms
        assert model_swarm_id == self.value.manifests[0].model_swarm_id
        return self.value


def test_worker_placement_reads_dht_off_thread_and_drives_real_reload_fence():
    manifest = model()
    current = advertisement(manifest, "current", 0, 2)
    snapshot = DiscoverySnapshot(
        captured_at_ms=NOW,
        manifests=(manifest,),
        offers=(
            current.offer,
            advertisement(manifest, "replica", 0, 2).offer,
        ),
        leases=(
            current.lease,
            advertisement(manifest, "replica", 0, 2).lease,
        ),
        links=(),
    )
    admission = FakeAdmission()
    publisher = FakePublisher()
    reloads = []
    controller = AutonomousWorkerPlacement(
        catalog=FakeCatalog(snapshot),
        admission=admission,
        state_publisher=publisher,
        reload_target=lambda span, generation: reloads.append((span, generation)),
        current_span=current.lease.hosted_span,
        policy=AutonomousPlacementPolicy(movement_cooldown_ms=0),
    )

    first = controller.observe(
        advertisement=current,
        manifest=manifest,
        context_tokens=10,
    )
    assert first["decision"] in {
        "waiting_catalog",
        "fills_the_highest_verified_capacity_deficit",
        "coverage_preserved_and_verified_gain_exceeds_hysteresis",
    }
    deadline = time.monotonic() + 1
    while not reloads and time.monotonic() < deadline:
        time.sleep(0.01)
        controller.observe(
            advertisement=current,
            manifest=manifest,
            context_tokens=10,
        )

    assert reloads == [(LayerSpan(start=2, end=4), 1)]
    assert publisher.states == [SpanState.DRAINING]
    assert admission.draining

    ready = controller.observe(
        advertisement=advertisement(manifest, "current", 2, 4),
        manifest=manifest,
        context_tokens=10,
    )
    assert ready["phase"] == "ready"
    assert ready["current_span"] == [2, 4]
    assert not admission.draining
