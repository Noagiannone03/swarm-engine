from __future__ import annotations

import hashlib
import hmac
from concurrent.futures import Future

import pytest

from swarm_protocol.contracts import (
    BackendKind,
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    ModelMemberAdvertisement,
    PathKind,
    RecoveryLevel,
    ReservationState,
    RoutePlan,
    RouteStage,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)
from swarm_protocol.coordinator import (
    RouteReservationCoordinator,
    RouteReservationError,
)
from swarm_protocol.execution import WorkerExecutionAdmission
from swarm_protocol.execution_rpc import control_message_to_wire

COORDINATOR = "10" * 32
HEAD_ENDPOINT = "20" * 32
TAIL_ENDPOINT = "30" * 32
SWARM_ID = "40" * 32


class SharedCrypto:
    keys = {
        COORDINATOR: b"coordinator",
        HEAD_ENDPOINT: b"head",
        TAIL_ENDPOINT: b"tail",
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


def ready_member(
    *,
    worker_id: str,
    endpoint_id: str,
    span: LayerSpan,
    now: int,
    allocatable_bytes: int = 100_000,
) -> ModelMemberAdvertisement:
    return ModelMemberAdvertisement(
        offer=WorkerOffer(
            worker_id=worker_id,
            endpoint_id=endpoint_id,
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
            worker_id=worker_id,
            hosted_span=span,
            effective_span_mode=EffectiveSpanMode.FIXED,
            state=SpanState.READY,
            weight_hashes=("50" * 32,),
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


def route(now: int) -> RoutePlan:
    rounded = 1_008
    return RoutePlan(
        request_id="request",
        route_id="route",
        epoch=1,
        model_swarm_id=SWARM_ID,
        model_num_layers=4,
        prompt_tokens=900,
        reserved_output_tokens=100,
        stages=(
            RouteStage(
                worker_id="head",
                endpoint_id=HEAD_ENDPOINT,
                hosted_span=LayerSpan(start=0, end=2),
                effective_span=LayerSpan(start=0, end=2),
                path_to_next=PathKind.DIRECT,
                rounded_context_tokens=rounded,
                exact_kv_bytes=rounded * 8,
            ),
            RouteStage(
                worker_id="tail",
                endpoint_id=TAIL_ENDPOINT,
                hosted_span=LayerSpan(start=2, end=4),
                effective_span=LayerSpan(start=2, end=4),
                path_to_next=PathKind.DIRECT,
                rounded_context_tokens=rounded,
                exact_kv_bytes=rounded * 8,
            ),
        ),
        recovery_level=RecoveryLevel.RESTARTABLE,
        coordinator_id=COORDINATOR,
        reservation_deadline_ms=now + 5_000,
        plan_expires_at_ms=now + 10_000,
    )


class AdmissionStub:
    def __init__(
        self,
        admission: WorkerExecutionAdmission,
        *,
        fail_prepare: bool = False,
        fail_commit: bool = False,
    ):
        self.admission = admission
        self.fail_prepare = fail_prepare
        self.fail_commit = fail_commit
        self.native_timeouts = []

    def with_timeout(self, timeout_seconds):
        self.native_timeouts.append(timeout_seconds)
        return self

    @staticmethod
    def _future(function):
        future = Future()
        try:
            future.set_result(function())
        except Exception as exc:
            future.set_exception(exc)
        return future

    def prepare(self, message):
        if self.fail_prepare:
            return self._future(lambda: (_ for _ in ()).throw(RuntimeError("prepare failed")))
        return self._future(
            lambda: control_message_to_wire(
                self.admission.prepare(message, caller_endpoint_id=COORDINATOR)
            )
        )

    def command(self, message):
        if self.fail_commit:
            self.fail_commit = False
            return self._future(lambda: (_ for _ in ()).throw(RuntimeError("commit failed")))
        return self._future(
            lambda: (
                control_message_to_wire(result)
                if (
                    result := self.admission.apply_command(
                        message,
                        caller_endpoint_id=COORDINATOR,
                    )
                )
                is not None
                else None
            )
        )


class InProcessTransport(SharedCrypto):
    def __init__(self, stubs):
        super().__init__(COORDINATOR)
        self.stubs = stubs

    def stub(self, peer_id, _service):
        return self.stubs[peer_id]


def lab(now: list[int], *, tail_bytes: int = 100_000):
    head = WorkerExecutionAdmission(
        worker_id="head",
        endpoint_id=HEAD_ENDPOINT,
        coordinator_endpoint_id=COORDINATOR,
        crypto=SharedCrypto(HEAD_ENDPOINT),
        clock_ms=lambda: now[0],
    )
    tail = WorkerExecutionAdmission(
        worker_id="tail",
        endpoint_id=TAIL_ENDPOINT,
        coordinator_endpoint_id=COORDINATOR,
        crypto=SharedCrypto(TAIL_ENDPOINT),
        clock_ms=lambda: now[0],
    )
    head.configure(
        ready_member(
            worker_id="head",
            endpoint_id=HEAD_ENDPOINT,
            span=LayerSpan(start=0, end=2),
            now=now[0],
        )
    )
    tail.configure(
        ready_member(
            worker_id="tail",
            endpoint_id=TAIL_ENDPOINT,
            span=LayerSpan(start=2, end=4),
            now=now[0],
            allocatable_bytes=tail_bytes,
        )
    )
    return head, tail


def test_coordinator_commits_and_releases_every_stage():
    now = [1_000]
    head, tail = lab(now)
    transport = InProcessTransport(
        {
            HEAD_ENDPOINT: AdmissionStub(head),
            TAIL_ENDPOINT: AdmissionStub(tail),
        }
    )
    coordinator = RouteReservationCoordinator(transport, clock_ms=lambda: now[0])

    committed = coordinator.reserve(route(now[0]))

    assert [lease.state for lease in committed.leases] == [
        ReservationState.COMMITTED,
        ReservationState.COMMITTED,
    ]
    assert {lease.expires_at_ms for lease in committed.leases} == {61_000}
    assert transport.stubs[HEAD_ENDPOINT].native_timeouts == [5.0, 5.0, 5.0]
    assert transport.stubs[TAIL_ENDPOINT].native_timeouts == [5.0, 5.0, 5.0]
    coordinator.release(committed)
    assert transport.stubs[HEAD_ENDPOINT].native_timeouts == [5.0, 5.0, 5.0, 5.0]
    assert transport.stubs[TAIL_ENDPOINT].native_timeouts == [5.0, 5.0, 5.0, 5.0]
    assert head.snapshot()[0].state == ReservationState.RELEASED
    assert tail.snapshot()[0].state == ReservationState.RELEASED


def test_coordinator_fences_every_superseded_stage_with_newer_epoch():
    now = [1_000]
    head, tail = lab(now)
    transport = InProcessTransport(
        {
            HEAD_ENDPOINT: AdmissionStub(head),
            TAIL_ENDPOINT: AdmissionStub(tail),
        }
    )
    coordinator = RouteReservationCoordinator(transport, clock_ms=lambda: now[0])
    committed = coordinator.reserve(route(now[0]))

    coordinator.fence(committed, epoch=2)

    assert all(lease.state == ReservationState.RELEASED for lease in head.snapshot())
    assert all(lease.state == ReservationState.RELEASED for lease in tail.snapshot())
    with pytest.raises(ValueError, match="newer"):
        coordinator.fence(committed, epoch=1)


def test_prepare_failure_releases_parallel_successes():
    now = [1_000]
    head, tail = lab(now)
    transport = InProcessTransport(
        {
            HEAD_ENDPOINT: AdmissionStub(head),
            TAIL_ENDPOINT: AdmissionStub(tail, fail_prepare=True),
        }
    )
    coordinator = RouteReservationCoordinator(transport, clock_ms=lambda: now[0])

    with pytest.raises(RouteReservationError, match="prepare failed"):
        coordinator.reserve(route(now[0]))

    assert head.snapshot()[0].state == ReservationState.RELEASED
    assert tail.snapshot() == ()


def test_partial_commit_failure_releases_all_stages_before_traffic():
    now = [1_000]
    head, tail = lab(now)
    transport = InProcessTransport(
        {
            HEAD_ENDPOINT: AdmissionStub(head),
            TAIL_ENDPOINT: AdmissionStub(tail, fail_commit=True),
        }
    )
    coordinator = RouteReservationCoordinator(transport, clock_ms=lambda: now[0])

    with pytest.raises(RouteReservationError, match="commit failed"):
        coordinator.reserve(route(now[0]))

    assert head.snapshot()[0].state == ReservationState.RELEASED
    assert tail.snapshot()[0].state == ReservationState.RELEASED


def test_route_coordinator_identity_is_fenced_before_any_rpc():
    now = [1_000]
    head, tail = lab(now)
    coordinator = RouteReservationCoordinator(
        InProcessTransport(
            {
                HEAD_ENDPOINT: AdmissionStub(head),
                TAIL_ENDPOINT: AdmissionStub(tail),
            }
        ),
        clock_ms=lambda: now[0],
    )
    wrong = route(now[0]).model_copy(update={"coordinator_id": "60" * 32})

    with pytest.raises(RouteReservationError, match="does not match"):
        coordinator.reserve(wrong)

    assert head.snapshot() == ()
    assert tail.snapshot() == ()


def test_parallel_results_share_one_command_deadline(monkeypatch):
    now = [1_000]
    head, tail = lab(now)
    coordinator = RouteReservationCoordinator(
        InProcessTransport(
            {
                HEAD_ENDPOINT: AdmissionStub(head),
                TAIL_ENDPOINT: AdmissionStub(tail),
            }
        ),
        clock_ms=lambda: now[0],
        command_ttl_ms=5_000,
    )
    observed_timeouts = []

    class RecordingFuture:
        def result(self, *, timeout):
            observed_timeouts.append(timeout)
            return object()

    monotonic = iter((10.0, 10.0, 12.0))
    monkeypatch.setattr("swarm_protocol.coordinator.time.monotonic", lambda: next(monotonic))

    coordinator._resolve_all(
        [
            (route(now[0]).stages[0], RecordingFuture()),
            (route(now[0]).stages[1], RecordingFuture()),
        ],
        operation="prepare",
    )

    assert observed_timeouts == [5.0, 3.0]
