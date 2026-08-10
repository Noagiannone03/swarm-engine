from __future__ import annotations

import hashlib
import hmac
from types import SimpleNamespace

import pytest

from fabi_network.capability import RouteRecoveryPolicy
from swarm_protocol.contracts import (
    BackendKind,
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    ModelManifest,
    RecoveryLevel,
    RequestContract,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)
from swarm_protocol.control import ControlMessageKind, sign_control_contract
from swarm_protocol.coordinator import CommittedRoute
from swarm_protocol.discovery import InMemoryDiscoveryStore
from swarm_protocol.epochs import InMemoryEpochAllocator
from swarm_protocol.request_agent import (
    IssuedAdmission,
    RequestAgentAuthorityClient,
    RequestAgentAuthorityError,
    RequestAgentRouteRuntime,
    RoutePermitGrant,
)
from swarm_protocol.route_authority import RouteAdmissionEnvelope

COORDINATOR = "11" * 32
WORKER = "22" * 32
ACCOUNT = "33" * 32
PERMIT = "44" * 32
HASHES = tuple(f"{digit:x}" * 64 for digit in range(1, 7))


def manifest(*, revision="revision"):
    return ModelManifest(
        model_id="fabi/test",
        immutable_revision=revision,
        architecture_graph_hash=HASHES[0],
        tokenizer_hash=HASHES[1],
        weight_collection_hash=HASHES[2],
        weight_format="safetensors",
        quantization="bf16",
        dtype="bfloat16",
        num_layers=4,
        model_max_context_tokens=524_288,
        context_classes=(
            4_096,
            8_192,
            16_384,
            32_768,
            65_536,
            131_072,
            262_144,
            524_288,
        ),
        activation_bytes_per_token=4_096,
        kv_bytes_per_token_by_layer=(512,) * 4,
        rope_context_contract_hash=HASHES[3],
        attention_kv_contract_hash=HASHES[4],
        prefill_contract_hash=HASHES[5],
        wire_protocol_version=1,
    )


class CryptoTransport:
    def peer_id(self):
        return COORDINATOR

    def sign_control_payload(self, payload):
        return hmac.new(b"request-agent", payload, hashlib.sha512).digest()

    def verify_control_payload(self, signer_endpoint_id, payload, signature):
        assert signer_endpoint_id == COORDINATOR
        assert hmac.compare_digest(self.sign_control_payload(payload), signature)


class FakeAuthority:
    def __init__(self):
        self.permits = []
        self.capabilities = []
        self.released = []
        self.unmet_context_requests = []
        self.current_permit = None

    def observe_unmet_context_demand(self, **kwargs):
        self.unmet_context_requests.append(kwargs)
        return True

    def issue_permit(self, **kwargs):
        self.permits.append(kwargs)
        permit = RoutePermitGrant(
            permit_id=PERMIT,
            request_id=kwargs["request_id"],
            coordinator_endpoint_id=kwargs["coordinator_endpoint_id"],
            model_swarm_id=kwargs["model_swarm_id"],
            max_context_tokens=kwargs["max_context_tokens"],
            recovery_policies=kwargs["recovery_policies"],
            issued_at_ms=1_000,
            expires_at_ms=121_000,
        )
        self.current_permit = permit
        return permit

    def keepalive_permit(self, permit_id, *, ttl_ms, idempotency_key):
        assert permit_id == PERMIT
        assert ttl_ms > 0
        assert idempotency_key
        assert self.current_permit is not None
        self.current_permit = self.current_permit.model_copy(
            update={
                "authorization_generation": (self.current_permit.authorization_generation + 1),
                "expires_at_ms": self.current_permit.expires_at_ms + ttl_ms,
            }
        )
        return self.current_permit

    def issue_capability(self, *, permit_id, signed_plan, recovery_policy):
        self.capabilities.append((permit_id, signed_plan, recovery_policy))
        return IssuedAdmission(
            admission=RouteAdmissionEnvelope(
                signed_plan=signed_plan,
                authority_key_id="55" * 32,
                capability_token="signed-biscuit",
                permit_id=permit_id,
                account_id=ACCOUNT,
                authorization_generation=self.current_permit.authorization_generation,
                expires_at_ms=self.current_permit.expires_at_ms,
                recovery_policy=recovery_policy,
            ),
            expires_at_ms=11_000,
            root_revocation_id="66" * 64,
        )

    def release_permit(self, permit_id):
        self.released.append(permit_id)
        return True


class FakeCoordinator:
    def __init__(self, transport, authorizer, renewal_authorizer):
        self.transport = transport
        self.authorizer = authorizer
        self.renewal_authorizer = renewal_authorizer
        self.released = []
        self.renew_calls = []
        self.renew_errors = []
        self.on_renew = None

    def reserve(self, plan):
        signed = sign_control_contract(
            plan,
            kind=ControlMessageKind.ROUTE_PLAN,
            crypto=self.transport,
        )
        assert self.authorizer(signed).signed_plan == signed
        return CommittedRoute(plan=plan, leases=())

    def renew(self, route, *, ttl_ms):
        assert ttl_ms > 0
        signed = sign_control_contract(
            route.plan,
            kind=ControlMessageKind.ROUTE_PLAN,
            crypto=self.transport,
        )
        self.renewal_authorizer(signed)
        self.renew_calls.append((route.plan.route_id, ttl_ms))
        if self.on_renew is not None:
            self.on_renew()
        if self.renew_errors:
            raise self.renew_errors.pop(0)
        return route

    def release(self, route):
        self.released.append(route.plan.route_id)


def discovery(model):
    store = InMemoryDiscoveryStore(clock_ms=lambda: 1_000)
    store.publish_manifest(model)
    store.publish_offer(
        WorkerOffer(
            worker_id="worker",
            endpoint_id=WORKER,
            runtime_version="test",
            platform="test",
            backend=BackendKind.MLX,
            stable_memory_envelope_bytes=8 * 1024**3,
            supported_roles=frozenset({WorkerRole.EXECUTOR, WorkerRole.FRONTEND}),
            offer_seq=1,
            issued_at_ms=500,
            expires_at_ms=20_000,
        )
    )
    store.publish_span_lease(
        SpanLease(
            model_swarm_id=model.model_swarm_id,
            worker_id="worker",
            hosted_span=LayerSpan(start=0, end=4),
            effective_span_mode=EffectiveSpanMode.SUBSPAN,
            state=SpanState.READY,
            weight_hashes=(HASHES[0],),
            max_context_tokens=524_288,
            kv_geometry=KvGeometry(
                block_size_tokens=16,
                bytes_per_token_per_layer=512,
                allocatable_bytes=1024**3,
            ),
            available_kv_bytes_snapshot=1024**3,
            max_sessions=1,
            lease_seq=1,
            issued_at_ms=500,
            expires_at_ms=20_000,
        )
    )
    return store


def publish_complete_worker(store, model, *, worker_id: str, endpoint_id: str) -> None:
    store.publish_offer(
        WorkerOffer(
            worker_id=worker_id,
            endpoint_id=endpoint_id,
            runtime_version="test",
            platform="test",
            backend=BackendKind.MLX,
            stable_memory_envelope_bytes=8 * 1024**3,
            supported_roles=frozenset({WorkerRole.EXECUTOR, WorkerRole.FRONTEND}),
            offer_seq=1,
            issued_at_ms=500,
            expires_at_ms=20_000,
        )
    )
    store.publish_span_lease(
        SpanLease(
            model_swarm_id=model.model_swarm_id,
            worker_id=worker_id,
            hosted_span=LayerSpan(start=0, end=4),
            effective_span_mode=EffectiveSpanMode.SUBSPAN,
            state=SpanState.READY,
            weight_hashes=(HASHES[0],),
            max_context_tokens=524_288,
            kv_geometry=KvGeometry(
                block_size_tokens=16,
                bytes_per_token_per_layer=512,
                allocatable_bytes=1024**3,
            ),
            available_kv_bytes_snapshot=1024**3,
            max_sessions=1,
            lease_seq=1,
            issued_at_ms=500,
            expires_at_ms=20_000,
        )
    )


def test_request_agent_plans_from_dht_and_coordinates_with_its_own_identity():
    model = manifest()
    authority = FakeAuthority()
    coordinators = []

    def coordinator_factory(transport, authorizer, renewal_authorizer):
        coordinator = FakeCoordinator(transport, authorizer, renewal_authorizer)
        coordinators.append(coordinator)
        return coordinator

    runtime = RequestAgentRouteRuntime(
        transport=CryptoTransport(),
        discovery=discovery(model),
        registry=SimpleNamespace(fetch=lambda _model_id: SimpleNamespace(manifest=model)),
        authority=authority,
        epoch_allocator=InMemoryEpochAllocator(),
        coordinator_factory=coordinator_factory,
        clock_ms=lambda: 1_000,
        start_maintenance_thread=False,
    )
    request = RequestContract(
        request_id="request",
        model_swarm_id=model.model_swarm_id,
        prompt_tokens=1_000,
        reserved_output_tokens=200,
        recovery_level=RecoveryLevel.RESTARTABLE,
    )

    reservation = runtime.reserve(request)

    assert reservation.committed.plan.coordinator_id == COORDINATOR
    assert reservation.committed.plan.epoch == 1
    assert [stage.endpoint_id for stage in reservation.committed.plan.stages] == [WORKER]
    assert authority.permits[0]["max_context_tokens"] == 1_200
    assert authority.capabilities[0][0] == PERMIT
    assert runtime.reserve(request) is reservation
    renewed = runtime.renew(reservation, ttl_ms=60_000)
    assert renewed.committed == reservation.committed
    runtime.release(renewed)
    assert len(authority.permits) == 1
    assert coordinators[0].released == [reservation.committed.plan.route_id]
    assert authority.released == [PERMIT]


def test_request_agent_cold_replan_bans_failed_workers_and_reuses_permit():
    model = manifest()
    store = discovery(model)
    publish_complete_worker(
        store,
        model,
        worker_id="worker-z",
        endpoint_id="77" * 32,
    )
    authority = FakeAuthority()
    coordinators = []

    def coordinator_factory(transport, authorizer, renewal_authorizer):
        coordinator = FakeCoordinator(transport, authorizer, renewal_authorizer)
        coordinators.append(coordinator)
        return coordinator

    runtime = RequestAgentRouteRuntime(
        transport=CryptoTransport(),
        discovery=store,
        registry=SimpleNamespace(fetch=lambda _model_id: SimpleNamespace(manifest=model)),
        authority=authority,
        epoch_allocator=InMemoryEpochAllocator(),
        coordinator_factory=coordinator_factory,
        clock_ms=lambda: 1_000,
        start_maintenance_thread=False,
    )
    phases = []
    runtime.set_phase_observer(lambda request_id, phase: phases.append((request_id, phase)))
    request = RequestContract(
        request_id="request",
        model_swarm_id=model.model_swarm_id,
        prompt_tokens=1_000,
        reserved_output_tokens=200,
        recovery_level=RecoveryLevel.RESTARTABLE,
    )
    initial = runtime.reserve(request)

    replacement = runtime.replan_cold("request", failed_epoch=initial.committed.plan.epoch)

    assert [stage.worker_id for stage in initial.committed.plan.stages] == ["worker"]
    assert [stage.worker_id for stage in replacement.committed.plan.stages] == ["worker-z"]
    assert replacement.committed.plan.epoch > initial.committed.plan.epoch
    assert replacement.permit.permit_id == initial.permit.permit_id
    assert replacement.excluded_worker_ids == frozenset({"worker"})
    assert len(authority.permits) == 1
    assert len(authority.capabilities) == 2
    assert coordinators[0].released == [initial.committed.plan.route_id]
    assert authority.released == []
    assert runtime.status()["cold_replans"] == []
    assert phases == [
        ("request", "planning"),
        ("request", "authorizing"),
        ("request", "reserving"),
        ("request", "recovering"),
        ("request", "planning"),
        ("request", "authorizing"),
        ("request", "reserving"),
    ]
    assert runtime.release_request("request") is True
    assert authority.released == [PERMIT]


def test_request_agent_retains_failed_permit_until_clean_replan_error_is_released():
    model = manifest()
    authority = FakeAuthority()
    runtime = RequestAgentRouteRuntime(
        transport=CryptoTransport(),
        discovery=discovery(model),
        registry=SimpleNamespace(fetch=lambda _model_id: SimpleNamespace(manifest=model)),
        authority=authority,
        epoch_allocator=InMemoryEpochAllocator(),
        coordinator_factory=FakeCoordinator,
        clock_ms=lambda: 1_000,
        start_maintenance_thread=False,
    )
    request = RequestContract(
        request_id="request",
        model_swarm_id=model.model_swarm_id,
        prompt_tokens=1_000,
        reserved_output_tokens=200,
        recovery_level=RecoveryLevel.RESTARTABLE,
    )
    initial = runtime.reserve(request)

    with pytest.raises(Exception, match="no complete route"):
        runtime.replan_cold("request", failed_epoch=initial.committed.plan.epoch)

    assert runtime.active_reservation("request") is None
    assert runtime.status()["cold_replans"][0]["excluded_worker_count"] == 1
    assert authority.released == []
    assert runtime.release_request("request") is True
    assert authority.released == [PERMIT]


def test_request_agent_refuses_dht_manifest_not_authenticated_by_tuf():
    dht_model = manifest()
    trusted_model = manifest(revision="other")
    authority = FakeAuthority()
    runtime = RequestAgentRouteRuntime(
        transport=CryptoTransport(),
        discovery=discovery(dht_model),
        registry=SimpleNamespace(fetch=lambda _model_id: SimpleNamespace(manifest=trusted_model)),
        authority=authority,
        epoch_allocator=InMemoryEpochAllocator(),
        clock_ms=lambda: 1_000,
        start_maintenance_thread=False,
    )
    request = RequestContract(
        request_id="request",
        model_swarm_id=dht_model.model_swarm_id,
        prompt_tokens=100,
        reserved_output_tokens=20,
        recovery_level=RecoveryLevel.RESTARTABLE,
    )

    with pytest.raises(PermissionError, match="TUF-authenticated"):
        runtime.reserve(request)
    assert authority.permits == []


def test_request_agent_probes_exact_live_context_without_reserving():
    model = manifest()
    authority = FakeAuthority()
    runtime = RequestAgentRouteRuntime(
        transport=CryptoTransport(),
        discovery=discovery(model),
        registry=SimpleNamespace(fetch=lambda _model_id: SimpleNamespace(manifest=model)),
        authority=authority,
        epoch_allocator=InMemoryEpochAllocator(),
        clock_ms=lambda: 1_000,
        start_maintenance_thread=False,
    )

    assert runtime.max_supported_context_tokens(model.model_swarm_id, 600_000) == 524_288
    assert authority.permits == []
    assert runtime.status()["active_routes"] == []

    assert runtime.observe_unmet_context_demand(
        "long-request",
        model.model_swarm_id,
        400_000,
    )
    assert authority.unmet_context_requests == [
        {
            "request_id": "long-request",
            "model_swarm_id": model.model_swarm_id,
            "required_context_tokens": 400_000,
        }
    ]


def test_request_agent_reuses_readiness_snapshot_for_immediate_reservation():
    model = manifest()
    store = discovery(model)
    snapshot_calls = 0
    original_snapshot = store.snapshot

    def counted_snapshot(*, model_swarm_id=None, now_ms=None):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot(model_swarm_id=model_swarm_id, now_ms=now_ms)

    store.snapshot = counted_snapshot
    runtime = RequestAgentRouteRuntime(
        transport=CryptoTransport(),
        discovery=store,
        registry=SimpleNamespace(fetch=lambda _model_id: SimpleNamespace(manifest=model)),
        authority=FakeAuthority(),
        epoch_allocator=InMemoryEpochAllocator(),
        coordinator_factory=FakeCoordinator,
        clock_ms=lambda: 1_000,
        start_maintenance_thread=False,
    )
    request = RequestContract(
        request_id="cached-discovery-request",
        model_swarm_id=model.model_swarm_id,
        prompt_tokens=1_000,
        reserved_output_tokens=200,
        recovery_level=RecoveryLevel.RESTARTABLE,
    )

    assert runtime.max_supported_context_tokens(model.model_swarm_id, 600_000) == 524_288
    reservation = runtime.reserve(request)

    assert reservation.committed.plan.request_id == request.request_id
    assert snapshot_calls == 1
    runtime.release(reservation)


def test_cached_discovery_snapshot_never_extends_signed_worker_expiry():
    model = manifest()
    clock = MutableClock(1_000)
    runtime = RequestAgentRouteRuntime(
        transport=CryptoTransport(),
        discovery=discovery(model),
        registry=SimpleNamespace(fetch=lambda _model_id: SimpleNamespace(manifest=model)),
        authority=FakeAuthority(),
        epoch_allocator=InMemoryEpochAllocator(),
        coordinator_factory=FakeCoordinator,
        clock_ms=clock,
        steady_clock_ms=lambda: 1_000,
        start_maintenance_thread=False,
    )
    request = RequestContract(
        request_id="expired-cached-worker",
        model_swarm_id=model.model_swarm_id,
        prompt_tokens=1_000,
        reserved_output_tokens=200,
        recovery_level=RecoveryLevel.RESTARTABLE,
    )

    assert runtime.max_supported_context_tokens(model.model_swarm_id, 600_000) == 524_288
    clock.now_ms = 20_000

    with pytest.raises(Exception, match="no complete route"):
        runtime.reserve(request)


class MutableClock:
    def __init__(self, now_ms=0):
        self.now_ms = now_ms

    def __call__(self):
        return self.now_ms


def lease_runtime():
    model = manifest()
    authority = FakeAuthority()
    coordinators = []
    steady_clock = MutableClock()

    def coordinator_factory(transport, authorizer, renewal_authorizer):
        coordinator = FakeCoordinator(transport, authorizer, renewal_authorizer)
        coordinators.append(coordinator)
        return coordinator

    runtime = RequestAgentRouteRuntime(
        transport=CryptoTransport(),
        discovery=discovery(model),
        registry=SimpleNamespace(fetch=lambda _model_id: SimpleNamespace(manifest=model)),
        authority=authority,
        epoch_allocator=InMemoryEpochAllocator(),
        coordinator_factory=coordinator_factory,
        clock_ms=lambda: 1_000,
        steady_clock_ms=steady_clock,
        session_ttl_ms=60,
        renew_interval_ms=20,
        renew_retry_interval_ms=5,
        renew_attempt_budget_ms=10,
        lease_expiry_guard_ms=1,
        start_maintenance_thread=False,
    )
    request = RequestContract(
        request_id="lease-request",
        model_swarm_id=model.model_swarm_id,
        prompt_tokens=100,
        reserved_output_tokens=20,
        recovery_level=RecoveryLevel.RESTARTABLE,
    )
    reservation = runtime.reserve(request)
    return runtime, reservation, authority, coordinators[0], steady_clock


def test_request_agent_renews_due_lease_from_independent_maintenance_tick():
    runtime, reservation, _authority, coordinator, steady_clock = lease_runtime()

    steady_clock.now_ms = 19
    runtime.maintain_once()
    assert coordinator.renew_calls == []

    steady_clock.now_ms = 20
    runtime.maintain_once()

    assert coordinator.renew_calls == [(reservation.committed.plan.route_id, 60)]
    assert runtime.active_reservation("lease-request") is not None
    route_status = runtime.status()["active_routes"][0]
    assert route_status["lease_expires_in_ms"] == 60
    assert route_status["renewal_failures"] == 0


def test_explicit_control_acks_keep_a_silent_generation_alive_past_initial_ttl():
    runtime, reservation, authority, coordinator, steady_clock = lease_runtime()

    for now_ms in range(20, 221, 20):
        steady_clock.now_ms = now_ms
        runtime.maintain_once()

    assert len(coordinator.renew_calls) == 11
    assert all(
        call == (reservation.committed.plan.route_id, 60) for call in coordinator.renew_calls
    )
    assert authority.current_permit.authorization_generation == 11
    assert runtime.active_reservation("lease-request") is not None
    assert runtime.status()["active_routes"][0]["lease_expires_in_ms"] == 60


def test_request_agent_retries_transient_renewal_only_inside_safe_window():
    runtime, _reservation, _authority, coordinator, steady_clock = lease_runtime()
    coordinator.renew_errors.append(RuntimeError("temporary network failure"))

    steady_clock.now_ms = 20
    runtime.maintain_once()
    route_status = runtime.status()["active_routes"][0]
    assert route_status["renewal_failures"] == 1
    assert "temporary network failure" in route_status["last_renewal_error"]

    steady_clock.now_ms = 24
    runtime.maintain_once()
    assert len(coordinator.renew_calls) == 1

    steady_clock.now_ms = 25
    runtime.maintain_once()
    assert len(coordinator.renew_calls) == 2
    route_status = runtime.status()["active_routes"][0]
    assert route_status["lease_expires_in_ms"] == 60
    assert route_status["renewal_failures"] == 0


def test_request_agent_fails_closed_before_lease_retry_can_cross_expiry():
    runtime, reservation, authority, coordinator, steady_clock = lease_runtime()

    steady_clock.now_ms = 49
    runtime.maintain_once()

    assert runtime.active_reservation("lease-request") is None
    assert coordinator.renew_calls == []
    assert coordinator.released == [reservation.committed.plan.route_id]
    assert authority.released == []
    assert runtime.status()["cold_replans"][0]["failed_epoch"] == 1
    runtime.release_request("lease-request")
    assert authority.released == [PERMIT]
    failure = runtime.status()["recent_failures"][0]
    assert "no safe lease window" in failure["error"]


def test_request_agent_rejects_renewal_acknowledged_after_safe_deadline():
    runtime, reservation, authority, coordinator, steady_clock = lease_runtime()

    def delay_acknowledgement():
        steady_clock.now_ms += 59

    coordinator.on_renew = delay_acknowledgement
    steady_clock.now_ms = 20
    runtime.maintain_once()

    assert runtime.active_reservation("lease-request") is None
    assert coordinator.renew_calls == [(reservation.committed.plan.route_id, 60)]
    assert coordinator.released == [reservation.committed.plan.route_id]
    assert authority.released == []
    assert runtime.status()["cold_replans"][0]["failed_epoch"] == 1
    runtime.release_request("lease-request")
    assert authority.released == [PERMIT]
    failure = runtime.status()["recent_failures"][0]
    assert "acknowledgement arrived after the safe deadline" in failure["error"]


class FakeResponse:
    def __init__(self, status_code, payload, *, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.content = b"{}"

    def json(self):
        return self._payload


class FakeHttpSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response


def test_authority_http_client_bounds_security_and_preserves_retry_metadata():
    response = FakeResponse(
        429,
        {"error": {"code": "contribution_capacity_reached", "message": "busy"}},
        headers={"Retry-After": "2"},
    )
    session = FakeHttpSession(response)
    client = RequestAgentAuthorityClient(
        "https://authority.example",
        "ab" * 32,
        session=session,
    )

    with pytest.raises(RequestAgentAuthorityError) as caught:
        client.issue_permit(
            request_id="request",
            coordinator_endpoint_id=COORDINATOR,
            model_swarm_id="77" * 32,
            max_context_tokens=16_384,
            recovery_policies=(RouteRecoveryPolicy.REPLAN_COLD,),
        )
    assert caught.value.status_code == 429
    assert caught.value.code == "contribution_capacity_reached"
    assert caught.value.retry_after_seconds == 2
    issue_key = session.calls[0][2]["headers"]["Idempotency-Key"]
    assert issue_key.startswith("permit-issue:")
    assert len(issue_key) == len("permit-issue:") + 64
    assert session.calls[0][2]["json"]["request_id"] == "request"
    assert session.calls[0][2]["headers"]["Authorization"] == "Bearer " + "ab" * 32

    with pytest.raises(RequestAgentAuthorityError):
        client.issue_permit(
            request_id="request",
            coordinator_endpoint_id=COORDINATOR,
            model_swarm_id="77" * 32,
            max_context_tokens=16_384,
            recovery_policies=(RouteRecoveryPolicy.REPLAN_COLD,),
        )
    assert session.calls[1][2]["headers"]["Idempotency-Key"] == issue_key

    with pytest.raises(RequestAgentAuthorityError):
        client.issue_permit(
            request_id="request",
            coordinator_endpoint_id=COORDINATOR,
            model_swarm_id="77" * 32,
            max_context_tokens=16_393,
            recovery_policies=(RouteRecoveryPolicy.REPLAN_COLD,),
        )
    assert session.calls[2][2]["headers"]["Idempotency-Key"] != issue_key

    with pytest.raises(ValueError, match="HTTPS"):
        RequestAgentAuthorityClient("http://authority.example", "ab" * 32)
