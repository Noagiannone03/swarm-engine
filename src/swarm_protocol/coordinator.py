"""Request-scoped prepare/commit coordinator for protocol-v3 routes."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

from swarm_protocol.contracts import (
    ReservationAction,
    ReservationCommand,
    ReservationLease,
    ReservationState,
    RoutePlan,
    RouteStage,
)
from swarm_protocol.control import (
    ControlCrypto,
    ControlMessageKind,
    SignedControlMessage,
    sign_control_contract,
    verify_control_contract,
)
from swarm_protocol.execution_rpc import (
    WorkerExecutionControlService,
    control_message_to_wire,
    route_admission_to_wire,
    route_renewal_to_wire,
)
from swarm_protocol.route_authority import RouteAdmissionEnvelope


class RouteReservationError(RuntimeError):
    """A complete route could not be atomically admitted."""


class ControlTransport(ControlCrypto):
    def stub(self, peer_id: str, service: type[object]) -> object: ...


RouteAdmissionAuthorizer = Callable[[SignedControlMessage], RouteAdmissionEnvelope]
RouteRenewalAuthorizer = Callable[[SignedControlMessage], RouteAdmissionEnvelope]


@dataclass(frozen=True)
class CommittedRoute:
    plan: RoutePlan
    leases: tuple[ReservationLease, ...]


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


class RouteReservationCoordinator:
    """Run bounded two-phase admission across every stage of one exact route."""

    def __init__(
        self,
        transport: ControlTransport,
        *,
        clock_ms: Callable[[], int] = _system_clock_ms,
        command_ttl_ms: int = 5_000,
        session_ttl_ms: int = 60_000,
        admission_authorizer: RouteAdmissionAuthorizer | None = None,
        renewal_authorizer: RouteRenewalAuthorizer | None = None,
    ) -> None:
        if command_ttl_ms <= 0:
            raise ValueError("command TTL must be positive")
        if session_ttl_ms <= command_ttl_ms:
            raise ValueError("session TTL must be greater than the command TTL")
        self.transport = transport
        self.endpoint_id = transport.peer_id()
        self._clock_ms = clock_ms
        self.command_ttl_ms = command_ttl_ms
        self.session_ttl_ms = session_ttl_ms
        self.admission_authorizer = admission_authorizer
        self.renewal_authorizer = renewal_authorizer

    def _now_ms(self) -> int:
        now = int(self._clock_ms())
        if now < 0:
            raise RuntimeError("route coordinator clock returned a negative timestamp")
        return now

    def _stub(self, stage: RouteStage):
        stub = self.transport.stub(stage.endpoint_id, WorkerExecutionControlService)
        with_timeout = getattr(stub, "with_timeout", None)
        if with_timeout is not None:
            # Future.result(timeout=...) only bounds the local waiter. Bound the
            # underlying QUIC request as well so a timed-out mutation cannot be
            # applied minutes later after a route has already been fenced.
            stub = with_timeout(self.command_ttl_ms / 1000)
        return stub

    def _verify_lease(
        self,
        raw: dict[str, object],
        *,
        stage: RouteStage,
        plan: RoutePlan,
        expected_state: ReservationState,
    ) -> ReservationLease:
        lease = verify_control_contract(
            raw,
            expected_kind=ControlMessageKind.RESERVATION_LEASE,
            expected_signer_endpoint_id=stage.endpoint_id,
            contract_type=ReservationLease,
            crypto=self.transport,
        )
        expected_id = f"{plan.route_id}:{stage.worker_id}"
        if (
            lease.reservation_id != expected_id
            or lease.request_id != plan.request_id
            or lease.route_id != plan.route_id
            or lease.epoch != plan.epoch
            or lease.worker_id != stage.worker_id
            or lease.effective_span != stage.effective_span
            or lease.exact_kv_bytes != stage.exact_kv_bytes
            or lease.state != expected_state
        ):
            raise RouteReservationError(
                f"worker {stage.worker_id} returned a reservation outside the route contract"
            )
        return lease

    def _resolve_all(
        self,
        pending: list[tuple[RouteStage, Future | object]],
        *,
        operation: str,
    ) -> list[tuple[RouteStage, object]]:
        deadline = time.monotonic() + self.command_ttl_ms / 1000
        resolved = []
        for stage, result in pending:
            try:
                value = (
                    result.result(timeout=max(0.0, deadline - time.monotonic()))
                    if hasattr(result, "result")
                    else result
                )
            except TimeoutError as exc:
                raise RouteReservationError(
                    f"{operation} timed out waiting for worker {stage.worker_id}"
                ) from exc
            except Exception as exc:
                raise RouteReservationError(
                    f"{operation} failed on worker {stage.worker_id}: {exc}"
                ) from exc
            resolved.append((stage, value))
        return resolved

    def _command(
        self,
        *,
        plan: RoutePlan,
        stage: RouteStage,
        action: ReservationAction,
        ttl_ms: int | None = None,
    ) -> SignedControlMessage:
        now_ms = self._now_ms()
        command = ReservationCommand(
            action=action,
            reservation_id=(
                None if action == ReservationAction.FENCE else f"{plan.route_id}:{stage.worker_id}"
            ),
            request_id=plan.request_id,
            route_id=plan.route_id,
            epoch=plan.epoch,
            ttl_ms=ttl_ms,
            issued_at_ms=now_ms,
            expires_at_ms=now_ms + self.command_ttl_ms,
        )
        return sign_control_contract(
            command,
            kind=ControlMessageKind.RESERVATION_COMMAND,
            crypto=self.transport,
        )

    def _send_command(
        self,
        *,
        plan: RoutePlan,
        stages: tuple[RouteStage, ...],
        action: ReservationAction,
        ttl_ms: int | None = None,
        admission: RouteAdmissionEnvelope | None = None,
    ) -> list[tuple[RouteStage, object]]:
        pending = []
        for stage in stages:
            command = self._command(
                plan=plan,
                stage=stage,
                action=action,
                ttl_ms=ttl_ms,
            )
            wire = (
                route_renewal_to_wire(command, admission)
                if admission is not None
                else control_message_to_wire(command)
            )
            pending.append((stage, self._stub(stage).command(wire)))
        return self._resolve_all(pending, operation=action.value)

    def reserve(self, plan: RoutePlan) -> CommittedRoute:
        """Prepare every stage, then commit every stage before returning traffic authority."""

        if plan.coordinator_id != self.endpoint_id:
            raise RouteReservationError("route plan coordinator does not match this endpoint")
        now_ms = self._now_ms()
        if plan.reservation_deadline_ms <= now_ms or plan.plan_expires_at_ms <= now_ms:
            raise RouteReservationError("route plan has expired before reservation")

        signed_plan = sign_control_contract(
            plan,
            kind=ControlMessageKind.ROUTE_PLAN,
            crypto=self.transport,
        )
        try:
            prepare_message = (
                control_message_to_wire(signed_plan)
                if self.admission_authorizer is None
                else route_admission_to_wire(self.admission_authorizer(signed_plan))
            )
            pending = [
                (
                    stage,
                    self._stub(stage).prepare(prepare_message),
                )
                for stage in plan.stages
            ]
            for stage, raw in self._resolve_all(pending, operation="prepare"):
                self._verify_lease(
                    raw,
                    stage=stage,
                    plan=plan,
                    expected_state=ReservationState.PREPARED,
                )
        except Exception as exc:
            try:
                # RELEASE is idempotent for an unknown reservation, so contact
                # every stage: some parallel PREPARE calls may have succeeded
                # even if result collection observed another failure first.
                self._send_command(
                    plan=plan,
                    stages=plan.stages,
                    action=ReservationAction.RELEASE,
                )
            except Exception:
                # PREPARED leases are short lived. The original failure is
                # more useful; expiry remains the final cleanup authority.
                pass
            raise RouteReservationError(f"route prepare failed: {exc}") from exc

        try:
            committed = self._send_command(
                plan=plan,
                stages=plan.stages,
                action=ReservationAction.COMMIT,
            )
            committed_leases = tuple(
                self._verify_lease(
                    raw,
                    stage=stage,
                    plan=plan,
                    expected_state=ReservationState.COMMITTED,
                )
                for stage, raw in committed
            )
        except Exception as exc:
            try:
                self._send_command(
                    plan=plan,
                    stages=plan.stages,
                    action=ReservationAction.RELEASE,
                )
            except Exception:
                pass
            raise RouteReservationError(f"route commit failed: {exc}") from exc

        # COMMIT deliberately does not extend the short PREPARE deadline.
        # Acquire the long-lived session lease on every stage before the data
        # plane is allowed to see the route.
        committed_route = CommittedRoute(plan=plan, leases=committed_leases)
        try:
            return self.renew(committed_route, ttl_ms=self.session_ttl_ms)
        except Exception as exc:
            try:
                self.release(committed_route)
            except Exception:
                pass
            raise RouteReservationError(f"route session lease failed: {exc}") from exc

    def renew(self, route: CommittedRoute, *, ttl_ms: int) -> CommittedRoute:
        if ttl_ms <= 0:
            raise ValueError("renew TTL must be positive")
        admission = None
        if self.renewal_authorizer is not None:
            signed_plan = sign_control_contract(
                route.plan,
                kind=ControlMessageKind.ROUTE_PLAN,
                crypto=self.transport,
            )
            admission = self.renewal_authorizer(signed_plan)
        renewed = self._send_command(
            plan=route.plan,
            stages=route.plan.stages,
            action=ReservationAction.RENEW,
            ttl_ms=ttl_ms,
            admission=admission,
        )
        leases = tuple(
            self._verify_lease(
                raw,
                stage=stage,
                plan=route.plan,
                expected_state=ReservationState.COMMITTED,
            )
            for stage, raw in renewed
        )
        return CommittedRoute(plan=route.plan, leases=leases)

    def release(self, route: CommittedRoute) -> None:
        self._send_command(
            plan=route.plan,
            stages=route.plan.stages,
            action=ReservationAction.RELEASE,
        )

    def fence(self, route: CommittedRoute, *, epoch: int) -> None:
        """Best-effort fence a superseded route with a strictly newer epoch."""

        if epoch <= route.plan.epoch:
            raise ValueError("fencing epoch must be newer than the route epoch")
        fenced_plan = route.plan.model_copy(update={"epoch": epoch})
        self._send_command(
            plan=fenced_plan,
            stages=fenced_plan.stages,
            action=ReservationAction.FENCE,
        )
