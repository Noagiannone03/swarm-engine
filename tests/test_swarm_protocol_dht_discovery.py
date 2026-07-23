from __future__ import annotations

import base64
import json
from dataclasses import dataclass

import pytest

from swarm_protocol import (
    BackendKind,
    DhtDiscoveryStore,
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    LinkMetric,
    ModelManifest,
    ModelMemberAdvertisement,
    PathKind,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)
from swarm_protocol.discovery import DiscoveryError

HASHES = tuple(character * 64 for character in "abcdef")
GIB = 1024**3


def manifest() -> ModelManifest:
    return ModelManifest(
        model_id="Qwen/Qwen3-8B",
        immutable_revision="0123456789abcdef",
        architecture_graph_hash=HASHES[0],
        tokenizer_hash=HASHES[1],
        weight_collection_hash=HASHES[2],
        weight_format="safetensors",
        quantization="bf16",
        dtype="bfloat16",
        num_layers=28,
        activation_bytes_per_token=4096,
        rope_context_contract_hash=HASHES[3],
        attention_kv_contract_hash=HASHES[4],
        prefill_contract_hash=HASHES[5],
        wire_protocol_version=1,
    )


def offer(worker_id: str, endpoint_id: str, sequence: int = 1) -> WorkerOffer:
    return WorkerOffer(
        worker_id=worker_id,
        endpoint_id=endpoint_id,
        runtime_version="3.0.0-dev",
        platform="darwin-arm64" if worker_id == "mac" else "windows-amd64",
        backend=BackendKind.MLX if worker_id == "mac" else BackendKind.VLLM,
        stable_memory_envelope_bytes=8 * GIB,
        supported_roles={WorkerRole.EXECUTOR, WorkerRole.FRONTEND},
        offer_seq=sequence,
        issued_at_ms=1_000 + sequence,
        expires_at_ms=20_000,
    )


def lease(model_id: str, worker_id: str, start: int, end: int) -> SpanLease:
    return SpanLease(
        model_swarm_id=model_id,
        worker_id=worker_id,
        hosted_span=LayerSpan(start=start, end=end),
        effective_span_mode=EffectiveSpanMode.SUBSPAN,
        state=SpanState.READY,
        weight_hashes=(HASHES[0],),
        measured_prefill_tokens_per_second=100,
        measured_decode_tokens_per_second=20,
        kv_geometry=KvGeometry(
            block_size_tokens=16,
            bytes_per_token_per_layer=1024,
            allocatable_bytes=4 * GIB,
        ),
        available_kv_bytes_snapshot=4 * GIB,
        max_sessions=4,
        lease_seq=1,
        issued_at_ms=1_001,
        expires_at_ms=20_000,
    )


@dataclass
class FakeRecord:
    kind: str
    logical_key: str
    publisher_endpoint_id: str
    discovery_peer_id: str
    sequence: int
    issued_at_ms: int
    expires_at_ms: int
    payload: bytes


class FakeNativeCatalog:
    def __init__(self, endpoint_id: str) -> None:
        self.endpoint_id = endpoint_id
        self.records: dict[str, FakeRecord] = {}
        self.members: dict[str, dict[str, FakeRecord]] = {}

    def catalog_key(
        self,
        kind: str,
        model_swarm_id: str | None = None,
        target_endpoint_id: str | None = None,
    ) -> str:
        prefix = "fabi/swarm/v3"
        if kind == "model_manifest":
            return f"{prefix}/manifest/{model_swarm_id}"
        if kind == "worker_offer":
            return f"{prefix}/offer/{self.endpoint_id}"
        if kind == "span_lease":
            return f"{prefix}/span/{model_swarm_id}/{self.endpoint_id}"
        if kind == "link_metric":
            return f"{prefix}/link/{self.endpoint_id}/{target_endpoint_id}"
        if kind == "model_member":
            return f"{prefix}/member/{model_swarm_id}/00"
        raise AssertionError(kind)

    def sign_catalog_record(
        self,
        kind: str,
        logical_key: str,
        discovery_peer_id: str,
        sequence: int,
        issued_at_ms: int,
        expires_at_ms: int,
        payload: bytes,
    ) -> bytes:
        return json.dumps(
            {
                "kind": kind,
                "logical_key": logical_key,
                "publisher_endpoint_id": self.endpoint_id,
                "discovery_peer_id": discovery_peer_id,
                "sequence": sequence,
                "issued_at_ms": issued_at_ms,
                "expires_at_ms": expires_at_ms,
                "payload": base64.b64encode(payload).decode(),
            },
            sort_keys=True,
        ).encode()

    def catalog_put(self, logical_key: str, encoded: bytes, quorum: int = 1) -> None:
        assert quorum > 0
        raw = json.loads(encoded)
        record = FakeRecord(
            **{key: raw[key] for key in raw if key != "payload"},
            payload=base64.b64decode(raw["payload"]),
        )
        assert record.logical_key == logical_key
        if record.kind == "model_member":
            entries = self.members.setdefault(logical_key.split("/")[4], {})
            previous = entries.get(record.publisher_endpoint_id)
            if previous is None or record.sequence >= previous.sequence:
                entries[record.publisher_endpoint_id] = record
        else:
            previous = self.records.get(logical_key)
            if previous is None or record.sequence >= previous.sequence:
                self.records[logical_key] = record

    def catalog_get(self, logical_key: str) -> FakeRecord:
        return self.records[logical_key]

    def catalog_get_model_members(self, model_swarm_id: str) -> list[FakeRecord]:
        return list(self.members.get(model_swarm_id, {}).values())

    def inject_member(
        self,
        endpoint_id: str,
        model_id: str,
        advertisement: ModelMemberAdvertisement,
        sequence: int,
    ) -> None:
        record = FakeRecord(
            kind="model_member",
            logical_key=f"fabi/swarm/v3/member/{model_id}/01",
            publisher_endpoint_id=endpoint_id,
            discovery_peer_id=f"dht-{endpoint_id}",
            sequence=sequence,
            issued_at_ms=1_500,
            expires_at_ms=20_000,
            payload=advertisement.model_dump_json().encode(),
        )
        self.members.setdefault(model_id, {})[endpoint_id] = record


def test_dht_store_builds_complete_snapshot_from_signed_membership_values() -> None:
    model = manifest()
    native = FakeNativeCatalog("endpoint-mac")
    store = DhtDiscoveryStore(native, "dht-mac", clock_ms=lambda: 2_000)
    mac_offer = offer("mac", native.endpoint_id)
    mac_lease = lease(model.model_swarm_id, "mac", 0, 4)
    assert store.publish_manifest(model)
    assert store.publish_offer(mac_offer)
    assert store.publish_span_lease(mac_lease)

    rtx_offer = offer("rtx", "endpoint-rtx")
    rtx_lease = lease(model.model_swarm_id, "rtx", 4, 28)
    native.inject_member(
        rtx_offer.endpoint_id,
        model.model_swarm_id,
        ModelMemberAdvertisement(offer=rtx_offer, lease=rtx_lease),
        sequence=3,
    )
    snapshot = store.snapshot(model_swarm_id=model.model_swarm_id, now_ms=2_000)
    assert [item.worker_id for item in snapshot.offers] == ["mac", "rtx"]
    assert [
        (item.worker_id, item.hosted_span.start, item.hosted_span.end) for item in snapshot.leases
    ] == [
        ("mac", 0, 4),
        ("rtx", 4, 28),
    ]

    metric = LinkMetric(
        from_worker_id="mac",
        to_worker_id="rtx",
        path_kind=PathKind.DIRECT,
        rtt_ms=4,
        throughput_bytes_per_second=100_000_000,
        measured_at_ms=2_100,
        expires_at_ms=10_000,
    )
    assert store.publish_link(metric)
    assert store.snapshot(model_swarm_id=model.model_swarm_id, now_ms=2_200).links == (metric,)


def test_dht_store_rejects_offer_for_another_signing_endpoint() -> None:
    native = FakeNativeCatalog("endpoint-mac")
    store = DhtDiscoveryStore(native, "dht-mac", clock_ms=lambda: 2_000)
    with pytest.raises(DiscoveryError, match="signing endpoint"):
        store.publish_offer(offer("mac", "endpoint-impostor"))


def test_snapshot_drops_member_whose_signed_endpoint_and_offer_disagree() -> None:
    model = manifest()
    native = FakeNativeCatalog("endpoint-mac")
    store = DhtDiscoveryStore(native, "dht-mac", clock_ms=lambda: 2_000)
    store.publish_manifest(model)
    remote_offer = offer("rtx", "endpoint-claimed")
    native.inject_member(
        "endpoint-signer",
        model.model_swarm_id,
        ModelMemberAdvertisement(
            offer=remote_offer,
            lease=lease(model.model_swarm_id, "rtx", 0, 28),
        ),
        sequence=1,
    )
    snapshot = store.snapshot(model_swarm_id=model.model_swarm_id, now_ms=2_000)
    assert snapshot.offers == ()
    assert snapshot.leases == ()
