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
from swarm_protocol.epochs import InMemoryRequestEpochFence, RequestEpochFence
from swarm_protocol.reservations import LocalReservationTable
from swarm_protocol.route_authority import (
    AuthorizedRoutePlan,
    FixedCoordinatorRouteAuthority,
    RouteAdmissionEnvelope,
    RoutePlanAuthority,
)


class ExecutionAdmissionError(RuntimeError):
    """A route cannot safely use this worker's current serving generation."""


class ServingContractBusy(ExecutionAdmissionError):
    """A different hosted-span generation was offered while reservations exist."""


_MAX_PLAN_FUTURE_MS = 60_000
_FENCE_CLOCK_SKEW_MS = 30_000


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


class WorkerExecutionAdmission:
    """Authoritative worker-side PREPARE/COMMIT/RELEASE state machine."""

    def __init__(
        self,
        *,
        worker_id: str,
        endpoint_id: str,
        coordinator_endpoint_id: str | None = None,
        route_authority: RoutePlanAuthority | None = None,
        crypto: ControlCrypto,
        clock_ms: Callable[[], int] = _system_clock_ms,
        request_epoch_fence: RequestEpochFence | None = None,
    ) -> None:
        if not worker_id or not endpoint_id:
            raise ValueError("worker and endpoint identities are required")
        if route_authority is None:
            if not coordinator_endpoint_id:
                raise ValueError("a route authority or coordinator identity is required")
            route_authority = FixedCoordinatorRouteAuthority(coordinator_endpoint_id)
        elif coordinator_endpoint_id is not None:
            raise ValueError("configure either route_authority or coordinator_endpoint_id")
        if crypto.peer_id() != endpoint_id:
            raise ValueError("execution crypto identity does not match the worker endpoint")
        self.worker_id = worker_id
        self.endpoint_id = endpoint_id
        self.coordinator_endpoint_id = coordinator_endpoint_id
        self.route_authority = route_authority
        self.crypto = crypto
        self._clock_ms = clock_ms
        self._request_epoch_fence = request_epoch_fence or InMemoryRequestEpochFence()
        self._advertisement: ModelMemberAdvertisement | None = None
        self._reservations: LocalReservationTable | None = None
        self._contract_key: tuple[object, ...] | None = None
        self._routes_by_route_id: dict[str, AuthorizedRoutePlan] = {}
        self._highest_epoch_by_request: dict[str, int] = {}
        self._draining = False
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
            self._routes_by_route_id.clear()

    def _ready_contract(
        self, now_ms: int
    ) -> tuple[ModelMemberAdvertisement, LocalReservationTable]:
        with self._lock:
            member = self._advertisement
            table = self._reservations
            draining = self._draining
            if member is None or table is None:
                raise ExecutionAdmissionError("worker has no verified serving contract")
            if draining:
                raise ExecutionAdmissionError("worker hosted span is draining")
            if member.lease.state != SpanState.READY:
                raise ExecutionAdmissionError("worker hosted span is not ready")
            if member.offer.expires_at_ms <= now_ms or member.lease.expires_at_ms <= now_ms:
                raise ExecutionAdmissionError("worker serving contract has expired")
            return member, table

    def _configured_table(self) -> LocalReservationTable:
        with self._lock:
            if self._reservations is None:
                raise ExecutionAdmissionError("worker has no verified serving contract")
            return self._reservations

    def begin_drain(self) -> int:
        """Atomically close admission and return capacity-owning reservations."""

        with self._lock:
            self._draining = True
            if self._reservations is None:
                return 0
            return sum(
                lease.state in {ReservationState.PREPARED, ReservationState.COMMITTED}
                for lease in self._reservations.snapshot()
            )

    def draining_reservations(self) -> int:
        with self._lock:
            if not self._draining or self._reservations is None:
                return 0
            return sum(
                lease.state in {ReservationState.PREPARED, ReservationState.COMMITTED}
                for lease in self._reservations.snapshot()
            )

    def cancel_drain(self) -> None:
        """Reopen the unchanged READY generation after an aborted movement."""

        with self._lock:
            member = self._advertisement
            if member is None or member.lease.state != SpanState.READY:
                raise ExecutionAdmissionError("cannot cancel drain without an unchanged ready span")
            self._draining = False

    def finish_drain(self) -> None:
        """Open admission after the replacement generation is verified READY."""

        with self._lock:
            member = self._advertisement
            if member is None or member.lease.state != SpanState.READY:
                raise ExecutionAdmissionError("replacement serving contract is not ready")
            if self._reservations is not None and self._reservations.used_kv_bytes:
                raise ServingContractBusy("replacement generation still owns old KV reservations")
            self._draining = False

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
        signed_plan: SignedControlMessage | RouteAdmissionEnvelope | dict[str, object],
        *,
        caller_endpoint_id: str,
    ) -> SignedControlMessage:
        with self._lock:
            now_ms = self._now_ms()
            member, table = self._ready_contract(now_ms)
            authorized_route = self.route_authority.authorize_route(
                signed_plan,
                caller_endpoint_id=caller_endpoint_id,
                crypto=self.crypto,
                now_ms=now_ms,
            )
            plan = authorized_route.plan
            if plan.reservation_deadline_ms <= now_ms or plan.plan_expires_at_ms <= now_ms:
                raise ExecutionAdmissionError("route plan reservation window has expired")
            if plan.plan_expires_at_ms > now_ms + _MAX_PLAN_FUTURE_MS:
                raise ExecutionAdmissionError("route plan expiry exceeds the worker replay bound")
            stage = self._local_stage(plan, member)
            self._request_epoch_fence.advance(
                coordinator_id=authorized_route.coordinator_endpoint_id,
                request_id=plan.request_id,
                epoch=plan.epoch,
                retain_until_ms=plan.plan_expires_at_ms + _FENCE_CLOCK_SKEW_MS,
                now_ms=now_ms,
            )
            reservation = table.prepare(
                reservation_id=f"{plan.route_id}:{self.worker_id}",
                request_id=plan.request_id,
                route_id=plan.route_id,
                epoch=plan.epoch,
                effective_span=stage.effective_span,
                exact_kv_bytes=stage.exact_kv_bytes,
                ttl_ms=plan.reservation_deadline_ms - now_ms,
            )
            highest = self._highest_epoch_by_request.get(plan.request_id, -1)
            if plan.epoch > highest:
                self._routes_by_route_id = {
                    route_id: existing
                    for route_id, existing in self._routes_by_route_id.items()
                    if existing.plan.request_id != plan.request_id
                }
                self._highest_epoch_by_request[plan.request_id] = plan.epoch
            self._routes_by_route_id[plan.route_id] = authorized_route
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
        now_ms = self._now_ms()
        table = self._configured_table()
        renewal_admission = None
        if isinstance(signed_command, dict) and "signed_command" in signed_command:
            renewal_admission = signed_command.get("admission")
            signed_command = signed_command["signed_command"]
        envelope = (
            signed_command
            if isinstance(signed_command, SignedControlMessage)
            else SignedControlMessage.model_validate(signed_command)
        )
        unsigned_command = ReservationCommand.model_validate_json(envelope.payload)
        with self._lock:
            route = self._routes_by_route_id.get(unsigned_command.route_id)
        expected_coordinator = (
            route.coordinator_endpoint_id if route is not None else caller_endpoint_id
        )
        if caller_endpoint_id != expected_coordinator:
            raise PermissionError("reservation command did not come from the route coordinator")
        if route is None and not self.route_authority.authorize_unbound_control(caller_endpoint_id):
            raise PermissionError("reservation command references no authorized route")
        command = verify_control_contract(
            envelope,
            expected_kind=ControlMessageKind.RESERVATION_COMMAND,
            expected_signer_endpoint_id=expected_coordinator,
            contract_type=ReservationCommand,
            crypto=self.crypto,
        )
        if route is not None and command.request_id != route.plan.request_id:
            raise ExecutionAdmissionError(
                "reservation command request does not match its authorized route"
            )
        if command.expires_at_ms <= now_ms or command.issued_at_ms > now_ms + 30_000:
            raise ExecutionAdmissionError("reservation command is expired or issued in the future")
        refreshed_route = None
        if command.action == ReservationAction.RENEW and route is not None:
            if route.permit_id is not None:
                if renewal_admission is None:
                    raise PermissionError(
                        "dynamic route renewal requires a fresh authority capability"
                    )
                refreshed_route = self.route_authority.authorize_route(
                    renewal_admission,
                    caller_endpoint_id=caller_endpoint_id,
                    crypto=self.crypto,
                    now_ms=now_ms,
                )
                if (
                    refreshed_route.plan != route.plan
                    or refreshed_route.route_plan_digest != route.route_plan_digest
                    or refreshed_route.permit_id != route.permit_id
                    or refreshed_route.account_id != route.account_id
                    or refreshed_route.recovery_policy != route.recovery_policy
                ):
                    raise PermissionError("renewal capability changed the authorized route")
                previous_generation = route.authorization_generation
                refreshed_generation = refreshed_route.authorization_generation
                if (
                    previous_generation is None
                    or refreshed_generation is None
                    or refreshed_generation < previous_generation
                ):
                    raise PermissionError("renewal capability generation moved backwards")
                assert command.ttl_ms is not None
                capability_expiry = refreshed_route.capability_expires_at_ms
                if capability_expiry is None or now_ms + command.ttl_ms > capability_expiry:
                    raise PermissionError("worker lease would outlive the authority capability")
            elif renewal_admission is not None:
                raise PermissionError("fixed route renewal does not accept a capability")
        elif renewal_admission is not None:
            raise PermissionError("authority capability is only valid on route renewal")
        self._request_epoch_fence.advance(
            coordinator_id=expected_coordinator,
            request_id=command.request_id,
            epoch=command.epoch,
            retain_until_ms=command.expires_at_ms + _FENCE_CLOCK_SKEW_MS,
            now_ms=now_ms,
        )

        lease: ReservationLease | None
        if command.action == ReservationAction.FENCE:
            table.fence_request(command.request_id, epoch=command.epoch)
            with self._lock:
                highest = self._highest_epoch_by_request.get(command.request_id, -1)
                if command.epoch > highest:
                    self._highest_epoch_by_request[command.request_id] = command.epoch
                    self._routes_by_route_id = {
                        route_id: authorized
                        for route_id, authorized in self._routes_by_route_id.items()
                        if authorized.plan.request_id != command.request_id
                    }
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
            if refreshed_route is not None:
                with self._lock:
                    self._routes_by_route_id[command.route_id] = refreshed_route
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

    def _lookup_authorized_route(
        self,
        *,
        request_id: str,
        route_id: str,
        epoch: int,
        routing_table: tuple[str, ...],
    ) -> AuthorizedRoutePlan:
        with self._lock:
            authorized = self._routes_by_route_id.get(route_id)
            highest_epoch = self._highest_epoch_by_request.get(request_id)
        if authorized is None:
            raise ExecutionAdmissionError("data plane references an unknown route")
        plan = authorized.plan
        if (
            plan.request_id != request_id
            or plan.epoch != epoch
            or highest_epoch != epoch
            or tuple(stage.worker_id for stage in plan.stages) != routing_table
        ):
            raise ExecutionAdmissionError("data-plane route fence does not match its signed plan")
        return authorized

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
        authorized = self._lookup_authorized_route(
            request_id=request_id,
            route_id=route_id,
            epoch=epoch,
            routing_table=routing_table,
        )
        plan = authorized.plan
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

        plan = self._authorize_plan(
            request_id=request_id,
            route_id=route_id,
            epoch=epoch,
            routing_table=routing_table,
        )
        authorized = self._lookup_authorized_route(
            request_id=request_id,
            route_id=route_id,
            epoch=epoch,
            routing_table=routing_table,
        )
        if caller_endpoint_id != authorized.coordinator_endpoint_id:
            raise PermissionError("frontend request did not come from the route coordinator")
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

        authorized = self._lookup_authorized_route(
            request_id=request_id,
            route_id=route_id,
            epoch=epoch,
            routing_table=routing_table,
        )
        if caller_endpoint_id not in {stage.endpoint_id for stage in authorized.plan.stages}:
            raise PermissionError("route control did not come from a signed route member")
