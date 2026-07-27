from __future__ import annotations

from types import SimpleNamespace

from swarm_protocol.active import ActiveRouteRuntime
from swarm_protocol.contracts import (
    LayerSpan,
    PathKind,
    RecoveryLevel,
    ReservationLease,
    ReservationState,
    RoutePlan,
    RouteStage,
)
from swarm_protocol.coordinator import CommittedRoute

SWARM_ID = "40" * 32
COORDINATOR_ID = "10" * 32
WORKER_ENDPOINT = "20" * 32


class FakeTransport:
    def peer_id(self):
        return COORDINATOR_ID


class FakePlanner:
    mode = "active"

    def __init__(self):
        self.epochs = []

    def ready_model_swarm_id(self, nodes):
        assert nodes
        return SWARM_ID

    def plan_request(
        self,
        nodes,
        *,
        request,
        coordinator_id,
        epoch,
        reservation_deadline_ms,
        plan_expires_at_ms,
    ):
        del nodes
        self.epochs.append(epoch)
        return SimpleNamespace(
            plan=RoutePlan(
                request_id=request.request_id,
                route_id=f"route-{epoch}",
                epoch=epoch,
                model_swarm_id=request.model_swarm_id,
                model_num_layers=2,
                prompt_tokens=request.prompt_tokens,
                reserved_output_tokens=request.reserved_output_tokens,
                stages=(
                    RouteStage(
                        worker_id="worker",
                        endpoint_id=WORKER_ENDPOINT,
                        hosted_span=LayerSpan(start=0, end=2),
                        effective_span=LayerSpan(start=0, end=2),
                        path_to_next=PathKind.DIRECT,
                        rounded_context_tokens=128,
                        exact_kv_bytes=1024,
                    ),
                ),
                recovery_level=RecoveryLevel.RESTARTABLE,
                coordinator_id=coordinator_id,
                reservation_deadline_ms=reservation_deadline_ms,
                plan_expires_at_ms=plan_expires_at_ms,
            )
        )


def committed(plan: RoutePlan, expires_at_ms: int) -> CommittedRoute:
    return CommittedRoute(
        plan=plan,
        leases=(
            ReservationLease(
                reservation_id=f"{plan.route_id}:worker",
                request_id=plan.request_id,
                route_id=plan.route_id,
                epoch=plan.epoch,
                worker_id="worker",
                effective_span=LayerSpan(start=0, end=2),
                exact_kv_bytes=1024,
                state=ReservationState.COMMITTED,
                issued_at_ms=1_000,
                expires_at_ms=expires_at_ms,
            ),
        ),
    )


class FakeCoordinator:
    def __init__(self, now):
        self.now = now
        self.reserved = []
        self.renewed = []
        self.released = []
        self.fail_renew = False

    def reserve(self, plan):
        self.reserved.append(plan)
        return committed(plan, self.now[0] + 60_000)

    def renew(self, route, *, ttl_ms):
        if self.fail_renew:
            raise RuntimeError("worker unreachable")
        self.renewed.append(route)
        return committed(route.plan, self.now[0] + ttl_ms)

    def release(self, route):
        self.released.append(route)


def runtime(now, *, nodes=None, wall_clock=None):
    planner = FakePlanner()
    coordinator = FakeCoordinator(now)
    current_nodes = [SimpleNamespace(node_id="worker", is_active=True)] if nodes is None else nodes
    active = ActiveRouteRuntime(
        planner=planner,
        transport=FakeTransport(),
        nodes_provider=lambda: current_nodes,
        clock_ms=lambda: (wall_clock or now)[0],
        steady_clock_ms=lambda: now[0],
        prepare_ttl_ms=100,
        plan_ttl_ms=300,
        session_ttl_ms=600,
        renew_interval_ms=200,
        renew_retry_interval_ms=50,
        lease_expiry_guard_ms=25,
        coordinator=coordinator,
    )
    return active, planner, coordinator, current_nodes


def test_active_runtime_routes_only_after_complete_reservation():
    now = [1_000]
    active, planner, coordinator, _ = runtime(now)
    try:
        assert active.reserve(
            request_id="request",
            prompt_tokens=100,
            reserved_output_tokens=20,
        ) == ("worker",)
        assert active.is_active("request")
        assert planner.epochs == [1]
        assert len(coordinator.reserved) == 1
        assert active.authority("request") == {"route_id": "route-1", "epoch": 1}

        assert active.release("request")
        assert not active.is_active("request")
        assert active.authority("request") is None
        assert len(coordinator.released) == 1
    finally:
        active.close()


def test_active_runtime_lease_ignores_wall_clock_jumps():
    steady = [1_000]
    wall = [1_000]
    active, _, _, _ = runtime(steady, wall_clock=wall)
    try:
        active.reserve(
            request_id="request",
            prompt_tokens=100,
            reserved_output_tokens=20,
        )

        wall[0] += 24 * 60 * 60 * 1_000

        assert active.is_active("request")
        assert active.snapshot()["active_routes"][0]["lease_expires_in_ms"] == 600
    finally:
        active.close()


def test_active_runtime_retries_transient_renewal_failure_before_expiry():
    now = [1_000]
    active, _, coordinator, _ = runtime(now)
    try:
        active.reserve(
            request_id="request",
            prompt_tokens=100,
            reserved_output_tokens=20,
        )
        now[0] += 201
        active.maintain_once()
        assert len(coordinator.renewed) == 1
        assert active.is_active("request")

        coordinator.fail_renew = True
        now[0] += 201
        active.maintain_once()
        assert active.is_active("request")
        assert len(coordinator.released) == 0
        assert active.snapshot()["active_routes"][0]["renewal_failures"] == 1

        coordinator.fail_renew = False
        now[0] += 51
        active.maintain_once()
        assert active.is_active("request")
        assert len(coordinator.renewed) == 2
        assert active.snapshot()["active_routes"][0]["renewal_failures"] == 0
    finally:
        active.close()


def test_active_runtime_fences_when_no_safe_renewal_window_remains():
    now = [1_000]
    active, _, coordinator, _ = runtime(now)
    try:
        active.reserve(
            request_id="request",
            prompt_tokens=100,
            reserved_output_tokens=20,
        )
        coordinator.fail_renew = True
        now[0] += 201
        active.maintain_once()
        assert active.is_active("request")

        now[0] = 1_476
        active.maintain_once()
        assert not active.is_active("request")
        assert len(coordinator.released) == 1
        assert active.snapshot()["recent_failures"][0]["epoch"] == 1
    finally:
        active.close()


def test_active_runtime_invalidates_departed_worker_and_advances_retry_epoch():
    now = [1_000]
    active, planner, coordinator, nodes = runtime(now)
    try:
        active.reserve(
            request_id="request",
            prompt_tokens=100,
            reserved_output_tokens=20,
        )
        nodes[0].is_active = False
        active.maintain_once()
        assert not active.is_active("request")
        assert len(coordinator.released) == 1

        nodes[0].is_active = True
        active.reserve(
            request_id="request",
            prompt_tokens=100,
            reserved_output_tokens=20,
        )
        assert planner.epochs == [1, 2]
    finally:
        active.close()
