import hashlib
import time
from types import SimpleNamespace

import pytest

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
    RecoveryLevel,
    RequestContract,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
    artifact_collection_hash,
)
from swarm_protocol.discovery import DiscoverySnapshot
from swarm_protocol.routing import NoFeasibleRoute
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
    while result["state"] in {"verifying_registry", "waiting_catalog"} and time.monotonic() < deadline:
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


def test_shadow_routes_cold_workers_without_inventing_performance_estimates():
    bundle = _bundle()
    result = _observe_until_resolved(
        SchedulerProtocolV3Shadow(_Registry(bundle)),
        _nodes(bundle, measured=False),
    )

    assert result["state"] == "agreement"
    assert result["performance_telemetry_complete"] is False
    assert result["projected_ttft_ms"] is None
    assert result["projected_inter_token_ms"] is None


class _Catalog:
    def __init__(self, snapshot):
        self.value = snapshot

    def publish_manifest(self, manifest):
        assert manifest == self.value.manifests[0]
        return True

    def snapshot(self, *, model_swarm_id=None, now_ms=None):
        del now_ms
        assert model_swarm_id == self.value.manifests[0].model_swarm_id
        return self.value


def test_active_planning_uses_dht_membership_not_legacy_scheduler_nodes():
    bundle = _bundle()
    nodes = _nodes(bundle)
    advertisements = [
        ModelMemberAdvertisement.model_validate(node.swarm_v3["advertisement"])
        for node in nodes
    ]
    building = _advertisement(bundle, "building", 0, 1, "rtx")
    catalog_snapshot = DiscoverySnapshot(
        captured_at_ms=time.time_ns() // 1_000_000,
        manifests=(bundle.manifest,),
        offers=tuple(item.offer for item in advertisements) + (building.offer,),
        leases=tuple(item.lease for item in advertisements)
        + (building.lease.model_copy(update={"state": SpanState.BUILDING}),),
        links=tuple(link for item in advertisements for link in item.outgoing_links),
    )
    planner = SchedulerProtocolV3Shadow(_Registry(bundle), mode="active")
    planner.attach_catalog(_Catalog(catalog_snapshot))
    status = _observe_until_resolved(planner, nodes)
    assert status["state"] == "route_ready"
    assert status["v3_route"] == ("mac", "rtx")
    assert "legacy_routes" not in status
    assert status["catalog"]["topology"] == {
        "offers": 3,
        "leases": 3,
        "links": 2,
        "workers": [
            {
                "worker_id": "building",
                "endpoint_id": "building",
                "roles": ["executor", "frontend"],
                "span": [0, 1],
                "state": "building",
                "available_kv_bytes": 1024**3,
                "expires_at_ms": building.lease.expires_at_ms,
            },
            {
                "worker_id": "mac",
                "endpoint_id": "mac",
                "roles": ["executor", "frontend"],
                "span": [0, 2],
                "state": "ready",
                "available_kv_bytes": 1024**3,
                "expires_at_ms": advertisements[0].lease.expires_at_ms,
            },
            {
                "worker_id": "rtx",
                "endpoint_id": "rtx",
                "roles": ["executor", "frontend"],
                "span": [2, 4],
                "state": "ready",
                "available_kv_bytes": 1024**3,
                "expires_at_ms": advertisements[1].lease.expires_at_ms,
            },
        ],
        "directed_links": [
            {
                "from_worker_id": "mac",
                "to_worker_id": "rtx",
                "path_kind": "direct",
                "rtt_ms": 2.0,
                "expires_at_ms": advertisements[0].outgoing_links[0].expires_at_ms,
            },
            {
                "from_worker_id": "rtx",
                "to_worker_id": "mac",
                "path_kind": "direct",
                "rtt_ms": 2.0,
                "expires_at_ms": advertisements[1].outgoing_links[0].expires_at_ms,
            },
        ],
        "truncated": False,
    }
    deadline = time.monotonic() + 2
    while not planner.live_worker_ids() and time.monotonic() < deadline:
        time.sleep(0.01)

    for node in nodes:
        node.is_active = False
    now_ms = time.time_ns() // 1_000_000
    planned = planner.plan_request(
        nodes,
        request=RequestContract(
            request_id="dht-only",
            model_swarm_id=bundle.model_swarm_id,
            prompt_tokens=400,
            reserved_output_tokens=112,
            recovery_level=RecoveryLevel.RESTARTABLE,
        ),
        coordinator_id="coordinator",
        epoch=1,
        reservation_deadline_ms=now_ms + 5_000,
        plan_expires_at_ms=now_ms + 10_000,
    )

    assert tuple(stage.worker_id for stage in planned.plan.stages) == ("mac", "rtx")
    assert planner.ready_model_swarm_id(nodes) == bundle.model_swarm_id
    assert planner.live_worker_ids() == frozenset({"mac", "rtx", "building"})
    assert planner.ready_worker_ids() == frozenset({"mac", "rtx"})


def test_structural_planning_keeps_online_spans_visible_when_live_kv_is_busy():
    bundle = _bundle()
    nodes = _nodes(bundle)
    advertisements = [
        ModelMemberAdvertisement.model_validate(node.swarm_v3["advertisement"])
        for node in nodes
    ]
    busy_leases = tuple(
        item.lease.model_copy(update={"available_kv_bytes_snapshot": 0})
        for item in advertisements
    )
    snapshot = DiscoverySnapshot(
        captured_at_ms=time.time_ns() // 1_000_000,
        manifests=(bundle.manifest,),
        offers=tuple(item.offer for item in advertisements),
        leases=busy_leases,
        links=tuple(link for item in advertisements for link in item.outgoing_links),
    )
    planner = SchedulerProtocolV3Shadow(_Registry(bundle), mode="active")
    planner.attach_catalog(_Catalog(snapshot))
    _observe_until_resolved(planner, nodes)
    now_ms = time.time_ns() // 1_000_000
    request = RequestContract(
        request_id="busy-kv",
        model_swarm_id=bundle.model_swarm_id,
        prompt_tokens=400,
        reserved_output_tokens=112,
        recovery_level=RecoveryLevel.RESTARTABLE,
    )
    kwargs = {
        "nodes": nodes,
        "request": request,
        "coordinator_id": "coordinator",
        "epoch": 1,
        "reservation_deadline_ms": now_ms + 5_000,
        "plan_expires_at_ms": now_ms + 10_000,
    }

    with pytest.raises(NoFeasibleRoute):
        planner.plan_request(**kwargs)
    planned = planner.plan_request(**kwargs, structural=True)
    assert tuple(stage.worker_id for stage in planned.plan.stages) == ("mac", "rtx")


def test_active_status_fails_closed_when_rpc_workers_are_absent_from_dht():
    bundle = _bundle()
    planner = SchedulerProtocolV3Shadow(_Registry(bundle), mode="active")
    planner.attach_catalog(
        _Catalog(
            DiscoverySnapshot(
                captured_at_ms=time.time_ns() // 1_000_000,
                manifests=(bundle.manifest,),
                offers=(),
                leases=(),
                links=(),
            )
        )
    )

    status = _observe_until_resolved(planner, _nodes(bundle))

    assert status["state"] == "no_feasible_route"
    assert status["accepted_workers"] == 0
    assert "v3_route" not in status
