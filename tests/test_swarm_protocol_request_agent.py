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

    def issue_permit(self, **kwargs):
        self.permits.append(kwargs)
        return RoutePermitGrant(
            permit_id=PERMIT,
            request_id=kwargs["request_id"],
            coordinator_endpoint_id=kwargs["coordinator_endpoint_id"],
            model_swarm_id=kwargs["model_swarm_id"],
            max_context_tokens=kwargs["max_context_tokens"],
            recovery_policies=kwargs["recovery_policies"],
            issued_at_ms=1_000,
            expires_at_ms=121_000,
        )

    def issue_capability(self, *, permit_id, signed_plan, recovery_policy):
        self.capabilities.append((permit_id, signed_plan, recovery_policy))
        return IssuedAdmission(
            admission=RouteAdmissionEnvelope(
                signed_plan=signed_plan,
                authority_key_id="55" * 32,
                capability_token="signed-biscuit",
                permit_id=permit_id,
                account_id=ACCOUNT,
                recovery_policy=recovery_policy,
            ),
            expires_at_ms=11_000,
            root_revocation_id="66" * 64,
        )

    def release_permit(self, permit_id):
        self.released.append(permit_id)
        return True


class FakeCoordinator:
    def __init__(self, transport, authorizer):
        self.transport = transport
        self.authorizer = authorizer
        self.released = []

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


def test_request_agent_plans_from_dht_and_coordinates_with_its_own_identity():
    model = manifest()
    authority = FakeAuthority()
    coordinators = []

    def coordinator_factory(transport, authorizer):
        coordinator = FakeCoordinator(transport, authorizer)
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
    assert session.calls[0][2]["headers"]["Idempotency-Key"] == "request"
    assert session.calls[0][2]["headers"]["Authorization"] == "Bearer " + "ab" * 32

    with pytest.raises(ValueError, match="HTTPS"):
        RequestAgentAuthorityClient("http://authority.example", "ab" * 32)
