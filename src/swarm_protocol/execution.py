"""Worker-local execution admission for signed protocol-v3 route plans.

This module deliberately owns no network or model backend. It validates an
authenticated coordinator's plan against the worker's verified hosted span,
then delegates exact KV ownership to ``LocalReservationTable``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from swarm_protocol.contracts import (
    EffectiveSpanMode,
    ModelMemberAdvertisement,
    ReservationAction,
    ReservationCommand,
    ReservationLease,
    ReservationState,
    RoutePlan,
    RouteStage,
    SpanState,
)
from swarm_protocol.control import (
    ControlCrypto,
    ControlMessageKind,
    SignedControlMessage,
    sign_control_contract,
    verify_control_contract,
)
from swarm_protocol.reservations import LocalReservationTable


class ExecutionAdmissionError(RuntimeError):
    """A route cannot safely use this worker's current serving generation."""


class ServingContractBusy(ExecutionAdmissionError):
    """A different hosted-span generation was offered while reservations exist."""


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


class WorkerExecutionAdmission:
    """Authoritative worker-side PREPARE/COMMIT/RELEASE state machine."""

    def __init__(
        self,
        *,
        worker_id: str,
        endpoint_id: str,
        coordinator_endpoint_id: str,
        crypto: ControlCrypto,
        clock_ms: Callable[[], int] = _system_clock_ms,
    ) -> None:
        if not worker_id or not endpoint_id or not coordinator_endpoint_id:
            raise ValueError("worker, endpoint, and coordinator identities are required")
        if crypto.peer_id() != endpoint_id:
            raise ValueError("execution crypto identity does not match the worker endpoint")
        self.worker_id = worker_id
        self.endpoint_id = endpoint_id
        self.coordinator_endpoint_id = coordinator_endpoint_id
        self.crypto = crypto
        self._clock_ms = clock_ms
        self._advertisement: ModelMemberAdvertisement | None = None
        self._reservations: LocalReservationTable | None = None
        self._contract_key: tuple[object, ...] | None = None
        self._plans_by_route_id: dict[str, RoutePlan] = {}
        self._highest_epoch_by_request: dict[str, int] = {}
        self._lock = threading.RLock()

    def _now_ms(self) -> int:
        now = int(self._clock_ms())
        if now < 0:
            raise RuntimeError("execution admission clock returned a negative timestamp")
        return now

    def configure(self, advertisement: ModelMemberAdvertisement | dict[str, object]) -> None:
        """Install one locally verified hosted-span contract.

        Repeated heartbeat advertisements for the same executor generation keep
        the existing reservation table. A different span/KV generation may only
        replace it after all local reservations have left capacity-owning states.
        """

        member = (
            advertisement
            if isinstance(advertisement, ModelMemberAdvertisement)
            else ModelMemberAdvertisement.model_validate(advertisement)
        )
        if member.offer.worker_id != self.worker_id or member.lease.worker_id != self.worker_id:
            raise ExecutionAdmissionError(
                "advertisement worker identity does not match this worker"
            )
        if member.offer.endpoint_id != self.endpoint_id:
            raise ExecutionAdmissionError(
                "advertisement endpoint identity does not match this worker"
            )
        if member.lease.state not in {SpanState.WARMING, SpanState.READY}:
            raise ExecutionAdmissionError("only warming or ready spans can configure admission")

        lease = member.lease
        contract_key = (
            lease.model_swarm_id,
            lease.hosted_span,
            lease.effective_span_mode,
            lease.kv_geometry,
        )
        with self._lock:
            if contract_key == self._contract_key:
                self._advertisement = member
                return
            if self._reservations is not None and self._reservations.used_kv_bytes:
                raise ServingContractBusy(
                    "cannot replace a hosted-span generation while it owns KV reservations"
                )
            self._reservations = LocalReservationTable(
                worker_id=self.worker_id,
                allocatable_kv_bytes=lease.kv_geometry.allocatable_bytes,
                clock_ms=self._clock_ms,
            )
            self._advertisement = member
            self._contract_key = contract_key
            self._plans_by_route_id.clear()
            self._highest_epoch_by_request.clear()

    def _require_caller(self, caller_endpoint_id: str) -> None:
        if caller_endpoint_id != self.coordinator_endpoint_id:
            raise PermissionError("only the configured route coordinator may control reservations")

    def _ready_contract(
        self, now_ms: int
    ) -> tuple[ModelMemberAdvertisement, LocalReservationTable]:
        member = self._advertisement
        table = self._reservations
        if member is None or table is None:
            raise ExecutionAdmissionError("worker has no verified serving contract")
        if member.lease.state != SpanState.READY:
            raise ExecutionAdmissionError("worker hosted span is not ready")
        if member.offer.expires_at_ms <= now_ms or member.lease.expires_at_ms <= now_ms:
            raise ExecutionAdmissionError("worker serving contract has expired")
        return member, table

    def _local_stage(
        self,
        plan: RoutePlan,
        member: ModelMemberAdvertisement,
    ) -> RouteStage:
        if plan.model_swarm_id != member.lease.model_swarm_id:
            raise ExecutionAdmissionError("route plan targets a different model swarm")
        matches = [stage for stage in plan.stages if stage.worker_id == self.worker_id]
        if len(matches) != 1:
            raise ExecutionAdmissionError("route plan must contain this worker exactly once")
        stage = matches[0]
        if stage.endpoint_id != self.endpoint_id:
            raise ExecutionAdmissionError("route stage endpoint does not match this worker")
        if stage.hosted_span != member.lease.hosted_span:
            raise ExecutionAdmissionError("route stage does not match the verified hosted span")
        if (
            member.lease.effective_span_mode == EffectiveSpanMode.FIXED
            and stage.effective_span != member.lease.hosted_span
        ):
            raise ExecutionAdmissionError("fixed-span backend cannot execute a route subspan")
        expected_bytes = member.lease.kv_geometry.required_bytes(
            stage.effective_span,
            plan.required_context_tokens,
        )
        if stage.exact_kv_bytes != expected_bytes:
            raise ExecutionAdmissionError("route stage KV bytes do not match local geometry")
        return stage

    def prepare(
        self,
        signed_plan: SignedControlMessage | dict[str, object],
        *,
        caller_endpoint_id: str,
    ) -> SignedControlMessage:
        self._require_caller(caller_endpoint_id)
        now_ms = self._now_ms()
        member, table = self._ready_contract(now_ms)
        plan = verify_control_contract(
            signed_plan,
            expected_kind=ControlMessageKind.ROUTE_PLAN,
            expected_signer_endpoint_id=self.coordinator_endpoint_id,
            contract_type=RoutePlan,
            crypto=self.crypto,
        )
        if plan.reservation_deadline_ms <= now_ms or plan.plan_expires_at_ms <= now_ms:
            raise ExecutionAdmissionError("route plan reservation window has expired")
        stage = self._local_stage(plan, member)
        reservation = table.prepare(
            reservation_id=f"{plan.route_id}:{self.worker_id}",
            request_id=plan.request_id,
            route_id=plan.route_id,
            epoch=plan.epoch,
            effective_span=stage.effective_span,
            exact_kv_bytes=stage.exact_kv_bytes,
            ttl_ms=plan.reservation_deadline_ms - now_ms,
        )
        with self._lock:
            highest = self._highest_epoch_by_request.get(plan.request_id, -1)
            if plan.epoch > highest:
                self._plans_by_route_id = {
                    route_id: existing
                    for route_id, existing in self._plans_by_route_id.items()
                    if existing.request_id != plan.request_id
                }
                self._highest_epoch_by_request[plan.request_id] = plan.epoch
            self._plans_by_route_id[plan.route_id] = plan
        return sign_control_contract(
            reservation,
            kind=ControlMessageKind.RESERVATION_LEASE,
            crypto=self.crypto,
        )

    def apply_command(
        self,
        signed_command: SignedControlMessage | dict[str, object],
        *,
        caller_endpoint_id: str,
    ) -> SignedControlMessage | None:
        self._require_caller(caller_endpoint_id)
        now_ms = self._now_ms()
        _, table = self._ready_contract(now_ms)
        command = verify_control_contract(
            signed_command,
            expected_kind=ControlMessageKind.RESERVATION_COMMAND,
            expected_signer_endpoint_id=self.coordinator_endpoint_id,
            contract_type=ReservationCommand,
            crypto=self.crypto,
        )
        if command.expires_at_ms <= now_ms or command.issued_at_ms > now_ms + 30_000:
            raise ExecutionAdmissionError("reservation command is expired or issued in the future")

        lease: ReservationLease | None
        if command.action == ReservationAction.FENCE:
            table.fence_request(command.request_id, epoch=command.epoch)
            return None
        assert command.reservation_id is not None
        if command.action == ReservationAction.COMMIT:
            lease = table.commit(command.reservation_id, epoch=command.epoch)
        elif command.action == ReservationAction.RENEW:
            assert command.ttl_ms is not None
            lease = table.renew(
                command.reservation_id,
                epoch=command.epoch,
                ttl_ms=command.ttl_ms,
            )
        elif command.action == ReservationAction.RELEASE:
            lease = table.release(command.reservation_id, epoch=command.epoch)
        else:  # pragma: no cover - exhaustive enum defense
            raise ExecutionAdmissionError(f"unsupported action {command.action}")
        if lease is None:
            return None
        if lease.request_id != command.request_id or lease.route_id != command.route_id:
            raise ExecutionAdmissionError("reservation command identity does not match its lease")
        return sign_control_contract(
            lease,
            kind=ControlMessageKind.RESERVATION_LEASE,
            crypto=self.crypto,
        )

    def snapshot(self) -> tuple[ReservationLease, ...]:
        with self._lock:
            return () if self._reservations is None else self._reservations.snapshot()

    def _lookup_plan(
        self,
        *,
        request_id: str,
        route_id: str,
        epoch: int,
        routing_table: tuple[str, ...],
    ) -> RoutePlan:
        with self._lock:
            plan = self._plans_by_route_id.get(route_id)
            highest_epoch = self._highest_epoch_by_request.get(request_id)
        if plan is None:
            raise ExecutionAdmissionError("data plane references an unknown route")
        if (
            plan.request_id != request_id
            or plan.epoch != epoch
            or highest_epoch != epoch
            or tuple(stage.worker_id for stage in plan.stages) != routing_table
        ):
            raise ExecutionAdmissionError("data-plane route fence does not match its signed plan")
        return plan

    def _authorize_plan(
        self,
        *,
        request_id: str,
        route_id: str,
        epoch: int,
        routing_table: tuple[str, ...],
    ) -> RoutePlan:
        now_ms = self._now_ms()
        _, table = self._ready_contract(now_ms)
        plan = self._lookup_plan(
            request_id=request_id,
            route_id=route_id,
            epoch=epoch,
            routing_table=routing_table,
        )
        reservation = table.get(f"{route_id}:{self.worker_id}")
        if (
            reservation is None
            or reservation.request_id != request_id
            or reservation.epoch != epoch
            or reservation.state != ReservationState.COMMITTED
        ):
            raise ExecutionAdmissionError("data-plane route has no committed local reservation")
        return plan

    def authorize_frontend(
        self,
        *,
        request_id: str,
        route_id: str,
        epoch: int,
        routing_table: tuple[str, ...],
        caller_endpoint_id: str,
    ) -> None:
        """Authorize the coordinator's initial request at the route head."""

        self._require_caller(caller_endpoint_id)
        plan = self._authorize_plan(
            request_id=request_id,
            route_id=route_id,
            epoch=epoch,
            routing_table=routing_table,
        )
        if plan.stages[0].worker_id != self.worker_id:
            raise ExecutionAdmissionError("frontend request did not reach the route head")

    def authorize_forward(
        self,
        *,
        request_id: str,
        route_id: str,
        epoch: int,
        routing_table: tuple[str, ...],
        caller_endpoint_id: str,
    ) -> None:
        """Authorize one fenced pipeline hop from the preceding route stage."""

        plan = self._authorize_plan(
            request_id=request_id,
            route_id=route_id,
            epoch=epoch,
            routing_table=routing_table,
        )
        local_index = next(
            index for index, stage in enumerate(plan.stages) if stage.worker_id == self.worker_id
        )
        predecessor = plan.stages[(local_index - 1) % len(plan.stages)]
        if caller_endpoint_id != predecessor.endpoint_id:
            raise PermissionError("pipeline forward did not come from the signed predecessor")

    def authorize_route_peer(
        self,
        *,
        request_id: str,
        route_id: str,
        epoch: int,
        routing_table: tuple[str, ...],
        caller_endpoint_id: str,
    ) -> None:
        """Authorize route-scoped control emitted by any signed route member."""

        plan = self._lookup_plan(
            request_id=request_id,
            route_id=route_id,
            epoch=epoch,
            routing_table=routing_table,
        )
        if caller_endpoint_id not in {stage.endpoint_id for stage in plan.stages}:
            raise PermissionError("route control did not come from a signed route member")
