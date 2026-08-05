from __future__ import annotations

import time

from swarm_protocol import (
    AutonomousPlacementPolicy,
    AutonomousWorkerPlacement,
    BackendKind,
    ContextCapacityDemandMap,
    ContextClassDemand,
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
    autonomous_context_tiers,
    autonomous_peer_topology,
    next_autonomous_context_tier,
)

HASHES = tuple(character * 64 for character in "abcdef")
NOW = 1_000


def test_autonomous_context_tiers_are_bounded_and_include_configured_floor():
    assert autonomous_context_tiers(32_768) == (32_768, 16_384, 8_192, 4_096)
    assert autonomous_context_tiers(32_768, minimum_tokens=12_000) == (
        32_768,
        16_384,
        12_000,
    )
    assert autonomous_context_tiers(2_048) == (2_048,)


def test_autonomous_context_tiers_reject_non_positive_contracts():
    for preferred, minimum in ((0, 4_096), (32_768, 0), (-1, 4_096)):
        try:
            autonomous_context_tiers(preferred, minimum_tokens=minimum)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive context tier contract was accepted")


def test_measured_context_limit_selects_highest_strictly_lower_tier():
    assert next_autonomous_context_tier(32_768, 30_752) == 16_384
    assert next_autonomous_context_tier(32_768, 8_192) == 8_192
    assert next_autonomous_context_tier(4_096, 4_095) is None


def test_autonomous_topology_forms_sparse_forward_edges_and_decode_closure():
    manifest = model()
    head = advertisement(manifest, "head", 0, 1)
    middle = advertisement(manifest, "middle", 1, 3)
    tail = advertisement(manifest, "tail", 3, 4)
    unrelated = advertisement(manifest, "unrelated", 1, 2)
    snapshot = DiscoverySnapshot(
        captured_at_ms=NOW,
        manifests=(manifest,),
        offers=(head.offer, middle.offer, tail.offer, unrelated.offer),
        leases=(head.lease, middle.lease, tail.lease, unrelated.lease),
        links=(),
    )

    assert autonomous_peer_topology(
        snapshot,
        worker_id="head",
        model_num_layers=4,
    ).outbound_worker_ids == ("middle", "unrelated")
    assert autonomous_peer_topology(
        snapshot,
        worker_id="head",
        model_num_layers=4,
    ).authorized_worker_ids == ("tail",)
    assert autonomous_peer_topology(
        snapshot,
        worker_id="middle",
        model_num_layers=4,
    ).outbound_worker_ids == ("tail",)
    assert autonomous_peer_topology(
        snapshot,
        worker_id="middle",
        model_num_layers=4,
    ).authorized_worker_ids == ("head",)
    assert autonomous_peer_topology(
        snapshot,
        worker_id="tail",
        model_num_layers=4,
    ).outbound_worker_ids == ("head",)
    assert autonomous_peer_topology(
        snapshot,
        worker_id="tail",
        model_num_layers=4,
    ).authorized_worker_ids == ("middle",)


def test_autonomous_topology_never_probes_every_non_adjacent_member():
    manifest = model()
    current = advertisement(manifest, "current", 0, 1)
    non_adjacent = [advertisement(manifest, f"peer-{index}", 2, 4) for index in range(20)]
    snapshot = DiscoverySnapshot(
        captured_at_ms=NOW,
        manifests=(manifest,),
        offers=(current.offer, *(item.offer for item in non_adjacent)),
        leases=(current.lease, *(item.lease for item in non_adjacent)),
        links=(),
    )

    topology = autonomous_peer_topology(
        snapshot,
        worker_id="current",
        model_num_layers=4,
    )

    assert topology.outbound_worker_ids == ()
    assert set(topology.authorized_worker_ids) == {item.offer.worker_id for item in non_adjacent}


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
        model_max_context_tokens=65_536,
        context_classes=(4_096, 8_192, 16_384, 32_768, 65_536),
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
            max_context_tokens=65_536,
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
        self.bootstrap = []
        self.publications = []

    def publish_bootstrap_state(self, value):
        self.bootstrap.append(value)

    def publish_span_state(self, value, state):
        self.states.append(state)
        self.publications.append(value)
        return value.model_copy(update={"lease": value.lease.model_copy(update={"state": state})})


class FakeCatalog:
    def __init__(self, value, context_demand=None) -> None:
        self.value = value
        self.demand = context_demand

    def snapshot(self, *, model_swarm_id=None, now_ms=None):
        del now_ms
        assert model_swarm_id == self.value.manifests[0].model_swarm_id
        return self.value

    def context_demand(self, manifest, *, region_id, now_ms=None):
        del now_ms
        assert manifest == self.value.manifests[0]
        assert region_id == "eu-west"
        return self.demand


def context_demand(manifest: ModelManifest) -> ContextCapacityDemandMap:
    return ContextCapacityDemandMap(
        model_swarm_id=manifest.model_swarm_id,
        region_id="eu-west",
        issued_at_ms=500,
        expires_at_ms=60_000,
        classes=tuple(
            ContextClassDemand(
                context_tokens=tokens,
                desired_independent_routes=2,
                desired_concurrent_slots=2,
                desired_replicas_by_layer=(2,) * manifest.num_layers,
                demand_weight_by_layer=(1.0, 1.0, 4.0, 4.0),
                confidence=0.9,
            )
            for tokens in manifest.context_classes
        ),
    )


def test_worker_compares_signed_context_demand_without_applying_it():
    manifest = model()
    current = advertisement(manifest, "current", 0, 4)
    snapshot = DiscoverySnapshot(
        captured_at_ms=NOW,
        manifests=(manifest,),
        offers=(current.offer,),
        leases=(current.lease,),
        links=(),
    )
    controller = AutonomousWorkerPlacement(
        catalog=FakeCatalog(snapshot, context_demand(manifest)),
        admission=FakeAdmission(),
        state_publisher=FakePublisher(),
        reload_target=lambda span, generation: None,
        current_span=current.lease.hosted_span,
        demand_region_id="eu-west",
    )

    deadline = time.monotonic() + 1
    status = controller.observe(advertisement=current, manifest=manifest, context_tokens=4_096)
    while (
        (status["context_demand_shadow"] or {}).get("state") != "compared"
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
        status = controller.observe(
            advertisement=current,
            manifest=manifest,
            context_tokens=4_096,
        )

    comparison = status["context_demand_shadow"]
    assert comparison["state"] == "compared"
    assert comparison["applied"] is False
    assert comparison["region_id"] == "eu-west"
    assert comparison["context_class_tokens"] == 4_096


def test_worker_placement_reads_dht_off_thread_and_drives_real_reload_fence():
    manifest = model()
    current = advertisement(manifest, "current", 0, 2)
    independent_route = advertisement(manifest, "independent", 0, 4)
    snapshot = DiscoverySnapshot(
        captured_at_ms=NOW,
        manifests=(manifest,),
        offers=(
            current.offer,
            advertisement(manifest, "replica", 0, 2).offer,
            independent_route.offer,
        ),
        leases=(
            current.lease,
            advertisement(manifest, "replica", 0, 2).lease,
            independent_route.lease,
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
    assert publisher.states == [SpanState.BUILDING]
    assert admission.draining

    ready = controller.observe(
        advertisement=advertisement(manifest, "current", 2, 4),
        manifest=manifest,
        context_tokens=10,
    )
    assert ready["phase"] == "ready"
    assert ready["current_span"] == [2, 4]
    assert not admission.draining


def test_worker_placement_delivers_verified_snapshots_to_topology_observer():
    manifest = model()
    current = advertisement(manifest, "current", 0, 4)
    snapshot = DiscoverySnapshot(
        captured_at_ms=NOW,
        manifests=(manifest,),
        offers=(current.offer,),
        leases=(current.lease,),
        links=(),
    )
    observed = []
    controller = AutonomousWorkerPlacement(
        catalog=FakeCatalog(snapshot),
        admission=FakeAdmission(),
        state_publisher=FakePublisher(),
        reload_target=lambda span, generation: None,
        current_span=current.lease.hosted_span,
        topology_observer=observed.append,
    )

    controller.observe(advertisement=current, manifest=manifest, context_tokens=10)
    deadline = time.monotonic() + 1
    while not observed and time.monotonic() < deadline:
        time.sleep(0.01)

    assert observed == [snapshot]


def test_worker_placement_repairs_disconnected_coverage_from_a_redundant_span():
    manifest = model()
    current = advertisement(manifest, "current", 0, 2)
    replica = advertisement(manifest, "replica", 0, 2)
    snapshot = DiscoverySnapshot(
        captured_at_ms=NOW,
        manifests=(manifest,),
        offers=(current.offer, replica.offer),
        leases=(current.lease, replica.lease),
        links=(),
    )
    reloads = []
    controller = AutonomousWorkerPlacement(
        catalog=FakeCatalog(snapshot),
        admission=FakeAdmission(),
        state_publisher=FakePublisher(),
        reload_target=lambda span, generation: reloads.append((span, generation)),
        current_span=current.lease.hosted_span,
        policy=AutonomousPlacementPolicy(movement_cooldown_ms=0),
    )

    deadline = time.monotonic() + 1
    status = controller.observe(
        advertisement=current,
        manifest=manifest,
        context_tokens=10,
    )
    while status["decision"] == "waiting_catalog" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = controller.observe(
            advertisement=current,
            manifest=manifest,
            context_tokens=10,
        )

    assert status["decision"] == "coverage_preserved_and_verified_gain_exceeds_hysteresis"
    assert status["phase"] == "building"
    assert status["target_span"] == [2, 4]
    assert reloads == [(LayerSpan(start=2, end=4), 1)]


def test_cold_worker_announces_building_before_executor_reload():
    manifest = model()
    snapshot = DiscoverySnapshot(
        captured_at_ms=NOW,
        manifests=(manifest,),
        offers=(),
        leases=(),
        links=(),
    )
    admission = FakeAdmission()
    publisher = FakePublisher()
    events = []
    controller = AutonomousWorkerPlacement(
        catalog=FakeCatalog(snapshot),
        admission=admission,
        state_publisher=publisher,
        reload_target=lambda span, generation: events.append(("reload", span, generation)),
        current_span=None,
    )
    joining = advertisement(manifest, "cold", 0, 1)

    status = controller.bootstrap(
        offer=joining.offer,
        manifest=manifest,
        context_tokens=10,
        kv_block_size=1,
        max_sessions=1,
        weight_hashes=(HASHES[0],),
    )
    deadline = time.monotonic() + 1
    while status["phase"] != "building" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = controller.bootstrap(
            offer=joining.offer,
            manifest=manifest,
            context_tokens=10,
            kv_block_size=1,
            max_sessions=1,
            weight_hashes=(HASHES[0],),
        )

    assert status["phase"] == "building"
    assert status["context_tokens"] == 10
    assert len(publisher.bootstrap) == 1
    intent = publisher.bootstrap[0]
    assert intent.lease.state is SpanState.BUILDING
    assert intent.lease.available_kv_bytes_snapshot == 0
    assert events == [("reload", intent.lease.hosted_span, 1)]


def test_cold_worker_republishes_measured_lower_context_without_moving_layers():
    manifest = model()
    snapshot = DiscoverySnapshot(
        captured_at_ms=NOW,
        manifests=(manifest,),
        offers=(),
        leases=(),
        links=(),
    )
    publisher = FakePublisher()
    controller = AutonomousWorkerPlacement(
        catalog=FakeCatalog(snapshot),
        admission=FakeAdmission(),
        state_publisher=publisher,
        reload_target=lambda span, generation: None,
    )
    joining = advertisement(manifest, "cold-context", 0, 1)

    status = controller.bootstrap(
        offer=joining.offer,
        manifest=manifest,
        context_tokens=10,
        kv_block_size=1,
        max_sessions=1,
        weight_hashes=(HASHES[0],),
    )
    deadline = time.monotonic() + 1
    while status["phase"] != "building" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = controller.bootstrap(
            offer=joining.offer,
            manifest=manifest,
            context_tokens=10,
            kv_block_size=1,
            max_sessions=1,
            weight_hashes=(HASHES[0],),
        )

    original = publisher.bootstrap[0]
    downgraded = controller.downgrade_building_context(4)

    assert downgraded["decision"] == "measured_context_downgrade"
    assert downgraded["context_tokens"] == 4
    assert downgraded["generation"] == status["generation"]
    replacement = publisher.publications[-1]
    assert replacement.lease.hosted_span == original.lease.hosted_span
    assert replacement.lease.max_context_tokens == 4
    assert replacement.lease.kv_geometry.allocatable_bytes == (
        replacement.lease.kv_geometry.required_bytes(
            replacement.lease.hosted_span,
            4,
        )
    )


def test_cold_worker_renews_building_intent_until_executor_is_ready():
    manifest = model()
    snapshot = DiscoverySnapshot(
        captured_at_ms=NOW,
        manifests=(manifest,),
        offers=(),
        leases=(),
        links=(),
    )
    publisher = FakePublisher()
    controller = AutonomousWorkerPlacement(
        catalog=FakeCatalog(snapshot),
        admission=FakeAdmission(),
        state_publisher=publisher,
        reload_target=lambda span, generation: None,
        transition_refresh_interval_s=0.01,
    )
    joining = advertisement(manifest, "cold-renewed", 0, 1)

    status = controller.bootstrap(
        offer=joining.offer,
        manifest=manifest,
        context_tokens=10,
        kv_block_size=1,
        max_sessions=1,
        weight_hashes=(HASHES[0],),
    )
    deadline = time.monotonic() + 1
    while status["phase"] != "building" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = controller.bootstrap(
            offer=joining.offer,
            manifest=manifest,
            context_tokens=10,
            kv_block_size=1,
            max_sessions=1,
            weight_hashes=(HASHES[0],),
        )
    while len(publisher.states) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert publisher.states[:2] == [SpanState.BUILDING, SpanState.BUILDING]
    target = publisher.bootstrap[0].lease.hosted_span
    ready = advertisement(manifest, "cold-renewed", target.start, target.end)
    result = controller.observe(
        advertisement=ready,
        manifest=manifest,
        context_tokens=10,
    )
    assert result["phase"] == "ready"
    publications_at_ready = len(publisher.states)
    time.sleep(0.04)
    assert len(publisher.states) == publications_at_ready
