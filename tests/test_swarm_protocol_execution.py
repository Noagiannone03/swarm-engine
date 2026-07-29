from __future__ import annotations

import hashlib
import hmac

import pytest

from swarm_protocol.contracts import (
    BackendKind,
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    ModelMemberAdvertisement,
    PathKind,
    RecoveryLevel,
    ReservationAction,
    ReservationCommand,
    ReservationState,
    RoutePlan,
    RouteStage,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)
from swarm_protocol.control import (
    ControlMessageKind,
    SignedControlMessage,
    sign_control_contract,
)
from swarm_protocol.execution import (
    ExecutionAdmissionError,
    ServingContractBusy,
    WorkerExecutionAdmission,
)
from swarm_protocol.epochs import SqliteRequestEpochFence
from swarm_protocol.reservations import CapacityUnavailable, StaleEpoch
from swarm_protocol.route_authority import (
    CapabilityRouteAuthority,
    RouteAdmissionEnvelope,
    route_plan_digest,
)

WORKER_ENDPOINT = "11" * 32
COORDINATOR_ENDPOINT = "22" * 32
ATTACKER_ENDPOINT = "33" * 32
SWARM_ID = "44" * 32
AUTHORITY_KEY_ID = "77" * 32
ACCOUNT_ID = "88" * 32
PERMIT_ID = "99" * 32


class FakeCrypto:
    keys = {
        WORKER_ENDPOINT: b"worker",
        COORDINATOR_ENDPOINT: b"coordinator",
        ATTACKER_ENDPOINT: b"attacker",
    }

    def __init__(self, endpoint_id: str):
        self.endpoint_id = endpoint_id

    def peer_id(self) -> str:
        return self.endpoint_id

    def sign_control_payload(self, payload: bytes) -> bytes:
        return hmac.new(self.keys[self.endpoint_id], payload, hashlib.sha512).digest()

    def verify_control_payload(
        self,
        signer_endpoint_id: str,
        payload: bytes,
        signature: bytes,
    ) -> None:
        expected = hmac.new(self.keys[signer_endpoint_id], payload, hashlib.sha512).digest()
        if not hmac.compare_digest(expected, signature):
            raise ValueError("invalid signature")


def member(*, now: int, allocatable_bytes: int = 100_000) -> ModelMemberAdvertisement:
    span = LayerSpan(start=0, end=4)
    return ModelMemberAdvertisement(
        offer=WorkerOffer(
            worker_id="worker",
            endpoint_id=WORKER_ENDPOINT,
            runtime_version="test",
            platform="test",
            backend=BackendKind.MLX,
            stable_memory_envelope_bytes=1_000_000,
            supported_roles=frozenset({WorkerRole.EXECUTOR, WorkerRole.FRONTEND}),
            offer_seq=1,
            issued_at_ms=now,
            expires_at_ms=now + 60_000,
        ),
        lease=SpanLease(
            model_swarm_id=SWARM_ID,
            worker_id="worker",
            hosted_span=span,
            effective_span_mode=EffectiveSpanMode.FIXED,
            state=SpanState.READY,
            weight_hashes=("55" * 32,),
            kv_geometry=KvGeometry(
                block_size_tokens=16,
                bytes_per_token_by_layer=(4, 4, 4, 4),
                allocatable_bytes=allocatable_bytes,
            ),
            available_kv_bytes_snapshot=allocatable_bytes,
            max_sessions=1,
            lease_seq=1,
            issued_at_ms=now,
            expires_at_ms=now + 60_000,
        ),
    )


def plan(
    *,
    now: int,
    epoch: int = 1,
    context_tokens: int = 1000,
    coordinator_endpoint_id: str = COORDINATOR_ENDPOINT,
) -> RoutePlan:
    rounded = ((context_tokens + 15) // 16) * 16
    return RoutePlan(
        request_id="request",
        route_id=f"route-{epoch}",
        epoch=epoch,
        model_swarm_id=SWARM_ID,
        model_num_layers=4,
        prompt_tokens=context_tokens - 100,
        reserved_output_tokens=100,
        stages=(
            RouteStage(
                worker_id="worker",
                endpoint_id=WORKER_ENDPOINT,
                hosted_span=LayerSpan(start=0, end=4),
                effective_span=LayerSpan(start=0, end=4),
                path_to_next=PathKind.DIRECT,
                rounded_context_tokens=rounded,
                exact_kv_bytes=rounded * 16,
            ),
        ),
        recovery_level=RecoveryLevel.RESTARTABLE,
        coordinator_id=coordinator_endpoint_id,
        reservation_deadline_ms=now + 5_000,
        plan_expires_at_ms=now + 10_000,
    )


def command(
    *,
    now: int,
    action: ReservationAction,
    route_id: str = "route-1",
    epoch: int = 1,
    ttl_ms: int | None = None,
) -> ReservationCommand:
    return ReservationCommand(
        action=action,
        reservation_id=None if action == ReservationAction.FENCE else f"{route_id}:worker",
        request_id="request",
        route_id=route_id,
        epoch=epoch,
        ttl_ms=ttl_ms,
        issued_at_ms=now,
        expires_at_ms=now + 1_000,
    )


def controller(now: list[int], *, request_epoch_fence=None) -> WorkerExecutionAdmission:
    result = WorkerExecutionAdmission(
        worker_id="worker",
        endpoint_id=WORKER_ENDPOINT,
        coordinator_endpoint_id=COORDINATOR_ENDPOINT,
        crypto=FakeCrypto(WORKER_ENDPOINT),
        clock_ms=lambda: now[0],
        request_epoch_fence=request_epoch_fence,
    )
    result.configure(member(now=now[0]))
    return result


def signed_plan(route: RoutePlan) -> SignedControlMessage:
    return sign_control_contract(
        route,
        kind=ControlMessageKind.ROUTE_PLAN,
        crypto=FakeCrypto(COORDINATOR_ENDPOINT),
    )


def capability_envelope(
    route: RoutePlan,
    *,
    capability_token: str = "signed-biscuit",
) -> RouteAdmissionEnvelope:
    signed = signed_plan(route)
    return RouteAdmissionEnvelope(
        signed_plan=signed,
        authority_key_id=AUTHORITY_KEY_ID,
        capability_token=capability_token,
        permit_id=PERMIT_ID,
        account_id=ACCOUNT_ID,
        recovery_policy="replan_cold",
    )


def signed_command(value: ReservationCommand) -> SignedControlMessage:
    return sign_control_contract(
        value,
        kind=ControlMessageKind.RESERVATION_COMMAND,
        crypto=FakeCrypto(COORDINATOR_ENDPOINT),
    )


def commit_route(admission: WorkerExecutionAdmission, *, now: int, epoch: int = 1) -> RoutePlan:
    route = plan(now=now, epoch=epoch)
    admission.prepare(signed_plan(route), caller_endpoint_id=COORDINATOR_ENDPOINT)
    admission.apply_command(
        signed_command(
            command(
                now=now,
                action=ReservationAction.COMMIT,
                route_id=route.route_id,
                epoch=epoch,
            )
        ),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )
    return route


def test_signed_prepare_commit_renew_release_is_exact_and_idempotent():
    now = [1_000]
    admission = controller(now)

    prepared = admission.prepare(
        signed_plan(plan(now=now[0])),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )
    duplicate = admission.prepare(
        signed_plan(plan(now=now[0])),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )
    assert prepared.payload == duplicate.payload
    assert admission.snapshot()[0].state == ReservationState.PREPARED

    committed = admission.apply_command(
        signed_command(command(now=now[0], action=ReservationAction.COMMIT)),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )
    assert committed is not None
    assert admission.snapshot()[0].state == ReservationState.COMMITTED

    admission.apply_command(
        signed_command(command(now=now[0], action=ReservationAction.RENEW, ttl_ms=30_000)),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )
    assert admission.snapshot()[0].expires_at_ms == now[0] + 30_000

    admission.apply_command(
        signed_command(command(now=now[0], action=ReservationAction.RELEASE)),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )
    assert admission.snapshot()[0].state == ReservationState.RELEASED


def test_admission_rejects_wrong_caller_tampering_and_impossible_kv():
    now = [1_000]
    admission = controller(now)
    valid = signed_plan(plan(now=now[0]))

    with pytest.raises(PermissionError, match="configured route coordinator"):
        admission.prepare(valid, caller_endpoint_id=ATTACKER_ENDPOINT)

    tampered = valid.model_copy(update={"payload": valid.payload + b" "})
    with pytest.raises(ValueError, match="invalid signature"):
        admission.prepare(tampered, caller_endpoint_id=COORDINATOR_ENDPOINT)

    too_large = plan(now=now[0], context_tokens=10_000)
    with pytest.raises(CapacityUnavailable):
        admission.prepare(
            signed_plan(too_large),
            caller_endpoint_id=COORDINATOR_ENDPOINT,
        )


def test_fixed_backend_rejects_effective_subspan_and_expired_plan():
    now = [1_000]
    admission = controller(now)
    route = plan(now=now[0])
    subspan = route.stages[0].model_copy(
        update={
            "effective_span": LayerSpan(start=0, end=3),
            "exact_kv_bytes": route.stages[0].rounded_context_tokens * 12,
        }
    )
    tail = RouteStage(
        worker_id="tail",
        endpoint_id="66" * 32,
        hosted_span=LayerSpan(start=3, end=4),
        effective_span=LayerSpan(start=3, end=4),
        path_to_next=PathKind.DIRECT,
        rounded_context_tokens=route.stages[0].rounded_context_tokens,
        exact_kv_bytes=route.stages[0].rounded_context_tokens * 4,
    )
    route = route.model_copy(update={"stages": (subspan, tail)})

    with pytest.raises(ExecutionAdmissionError, match="fixed-span"):
        admission.prepare(signed_plan(route), caller_endpoint_id=COORDINATOR_ENDPOINT)

    expired = plan(now=now[0])
    now[0] = expired.reservation_deadline_ms
    with pytest.raises(ExecutionAdmissionError, match="expired"):
        admission.prepare(signed_plan(expired), caller_endpoint_id=COORDINATOR_ENDPOINT)


def test_hosted_span_generation_cannot_change_while_kv_is_owned_and_epochs_fence():
    now = [1_000]
    admission = controller(now)
    admission.prepare(
        signed_plan(plan(now=now[0])),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )

    changed = member(now=now[0], allocatable_bytes=200_000)
    with pytest.raises(ServingContractBusy):
        admission.configure(changed)

    admission.apply_command(
        signed_command(
            command(
                now=now[0],
                action=ReservationAction.FENCE,
                route_id="route-2",
                epoch=2,
            )
        ),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )
    with pytest.raises(StaleEpoch):
        admission.apply_command(
            signed_command(command(now=now[0], action=ReservationAction.COMMIT)),
            caller_endpoint_id=COORDINATOR_ENDPOINT,
        )
    assert admission.snapshot()[0].state == ReservationState.RELEASED
    admission.configure(changed)


def test_worker_restart_durably_rejects_replayed_older_plan(tmp_path):
    now = [1_000]
    fence_path = tmp_path / "worker-control.sqlite3"
    first = controller(
        now,
        request_epoch_fence=SqliteRequestEpochFence(fence_path),
    )
    first.prepare(
        signed_plan(plan(now=now[0], epoch=4)),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )

    restarted = controller(
        now,
        request_epoch_fence=SqliteRequestEpochFence(fence_path),
    )
    with pytest.raises(StaleEpoch, match="worker fence 4"):
        restarted.prepare(
            signed_plan(plan(now=now[0], epoch=3)),
            caller_endpoint_id=COORDINATOR_ENDPOINT,
        )
    restarted.prepare(
        signed_plan(plan(now=now[0], epoch=5)),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )


def test_data_plane_requires_committed_route_exact_fence_and_authenticated_hop():
    now = [1_000]
    admission = controller(now)
    route = commit_route(admission, now=now[0])
    routing_table = ("worker",)

    admission.authorize_frontend(
        request_id=route.request_id,
        route_id=route.route_id,
        epoch=route.epoch,
        routing_table=routing_table,
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )
    admission.authorize_forward(
        request_id=route.request_id,
        route_id=route.route_id,
        epoch=route.epoch,
        routing_table=routing_table,
        caller_endpoint_id=WORKER_ENDPOINT,
    )

    with pytest.raises(PermissionError, match="coordinator"):
        admission.authorize_frontend(
            request_id=route.request_id,
            route_id=route.route_id,
            epoch=route.epoch,
            routing_table=routing_table,
            caller_endpoint_id=ATTACKER_ENDPOINT,
        )
    with pytest.raises(ExecutionAdmissionError, match="fence"):
        admission.authorize_forward(
            request_id=route.request_id,
            route_id=route.route_id,
            epoch=route.epoch + 1,
            routing_table=routing_table,
            caller_endpoint_id=WORKER_ENDPOINT,
        )


def test_released_route_allows_bounded_peer_abort_but_not_more_execution():
    now = [1_000]
    admission = controller(now)
    route = commit_route(admission, now=now[0])
    admission.apply_command(
        signed_command(command(now=now[0], action=ReservationAction.RELEASE)),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )

    with pytest.raises(ExecutionAdmissionError, match="committed"):
        admission.authorize_frontend(
            request_id=route.request_id,
            route_id=route.route_id,
            epoch=route.epoch,
            routing_table=("worker",),
            caller_endpoint_id=COORDINATOR_ENDPOINT,
        )
    admission.authorize_route_peer(
        request_id=route.request_id,
        route_id=route.route_id,
        epoch=route.epoch,
        routing_table=("worker",),
        caller_endpoint_id=WORKER_ENDPOINT,
    )


def test_drain_atomically_rejects_new_routes_but_allows_existing_release():
    now = [1_000]
    admission = controller(now)
    route = commit_route(admission, now=now[0])

    assert admission.begin_drain() == 1
    with pytest.raises(ExecutionAdmissionError, match="draining"):
        admission.prepare(
            signed_plan(plan(now=now[0], epoch=2)),
            caller_endpoint_id=COORDINATOR_ENDPOINT,
        )

    admission.apply_command(
        signed_command(
            command(
                now=now[0],
                action=ReservationAction.RELEASE,
                route_id=route.route_id,
                epoch=route.epoch,
            )
        ),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )
    assert admission.draining_reservations() == 0
    admission.cancel_drain()
    admission.prepare(
        signed_plan(plan(now=now[0], epoch=2)),
        caller_endpoint_id=COORDINATOR_ENDPOINT,
    )


def test_capability_authority_binds_dynamic_coordinator_to_exact_plan_and_context():
    now = [1_000]
    verified = []

    def verifier(public_key, token, context, revoked):
        verified.append((public_key, token, context, revoked))
        return "ab" * 32

    admission = WorkerExecutionAdmission(
        worker_id="worker",
        endpoint_id=WORKER_ENDPOINT,
        route_authority=CapabilityRouteAuthority(
            authority_public_keys={AUTHORITY_KEY_ID: "authority-public-key"},
            revoked_identifiers=lambda: ("cd" * 64,),
            verifier=verifier,
        ),
        crypto=FakeCrypto(WORKER_ENDPOINT),
        clock_ms=lambda: now[0],
    )
    admission.configure(member(now=now[0]))
    route = plan(now=now[0])
    envelope = capability_envelope(route)

    admission.prepare(envelope, caller_endpoint_id=COORDINATOR_ENDPOINT)

    assert len(verified) == 1
    public_key, token, context, revoked = verified[0]
    assert public_key == "authority-public-key"
    assert token == "signed-biscuit"
    assert revoked == ("cd" * 64,)
    assert context.route_plan_digest == route_plan_digest(envelope.signed_plan)
    assert context.coordinator_endpoint_id == COORDINATOR_ENDPOINT
    assert context.required_context_tokens == route.required_context_tokens
    assert context.epoch == route.epoch
    assert context.account_id == ACCOUNT_ID
    assert context.permit_id == PERMIT_ID


def test_dynamic_admission_rejects_bare_plans_wrong_callers_and_unbound_fences():
    now = [1_000]
    authority = CapabilityRouteAuthority(
        authority_public_keys={AUTHORITY_KEY_ID: "authority-public-key"},
        verifier=lambda *_: "ab" * 32,
    )
    admission = WorkerExecutionAdmission(
        worker_id="worker",
        endpoint_id=WORKER_ENDPOINT,
        route_authority=authority,
        crypto=FakeCrypto(WORKER_ENDPOINT),
        clock_ms=lambda: now[0],
    )
    admission.configure(member(now=now[0]))
    route = plan(now=now[0])
    envelope = capability_envelope(route)

    with pytest.raises(PermissionError, match="requires an authority capability"):
        admission.prepare(
            envelope.signed_plan,
            caller_endpoint_id=COORDINATOR_ENDPOINT,
        )
    with pytest.raises(PermissionError, match="caller"):
        admission.prepare(envelope, caller_endpoint_id=ATTACKER_ENDPOINT)

    admission.prepare(envelope, caller_endpoint_id=COORDINATOR_ENDPOINT)
    with pytest.raises(PermissionError, match="no authorized route"):
        admission.apply_command(
            signed_command(
                command(
                    now=now[0],
                    action=ReservationAction.FENCE,
                    route_id="unknown-route",
                    epoch=2,
                )
            ),
            caller_endpoint_id=COORDINATOR_ENDPOINT,
        )


def test_fixed_authority_rejects_a_signed_plan_claiming_another_coordinator():
    now = [1_000]
    admission = controller(now)
    route = plan(
        now=now[0],
        coordinator_endpoint_id=ATTACKER_ENDPOINT,
    )

    with pytest.raises(PermissionError, match="does not match"):
        admission.prepare(
            signed_plan(route),
            caller_endpoint_id=COORDINATOR_ENDPOINT,
        )
