from __future__ import annotations

from types import SimpleNamespace

import pytest

from swarm_protocol.active import ActiveRouteRuntime
from swarm_protocol.contracts import (
    LayerSpan,
    ModelManifest,
    PathKind,
    RecoveryLevel,
    ReservationLease,
    ReservationState,
    RoutePlan,
    RouteStage,
)
from swarm_protocol.coordinator import CommittedRoute
from swarm_protocol.routing import NoFeasibleRoute

SWARM_ID = "40" * 32
COORDINATOR_ID = "10" * 32
WORKER_ENDPOINT = "20" * 32
RECOVERY_ENDPOINT = "30" * 32
HASHES = tuple(f"{digit:x}" * 64 for digit in range(1, 7))


class FakeTransport:
    def peer_id(self):
        return COORDINATOR_ID


class FakePlanner:
    mode = "active"

    def __init__(self):
        self.epochs = []
        self.fail_planning = False
        self.max_context_tokens = None

    def ready_model_swarm_id(self, nodes):
        assert nodes
        return SWARM_ID

    def trusted_manifest(self, model_swarm_id):
        assert model_swarm_id == SWARM_ID
        return ModelManifest(
            model_id="fabi/test",
            immutable_revision="revision",
            architecture_graph_hash=HASHES[0],
            tokenizer_hash=HASHES[1],
            weight_collection_hash=HASHES[2],
            weight_format="safetensors",
            quantization="bf16",
            dtype="bfloat16",
            num_layers=2,
            activation_bytes_per_token=4096,
            kv_bytes_per_token_by_layer=(512, 512),
            rope_context_contract_hash=HASHES[3],
            attention_kv_contract_hash=HASHES[4],
            prefill_contract_hash=HASHES[5],
            wire_protocol_version=1,
        )

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
        if self.fail_planning:
            raise NoFeasibleRoute("no complete route")
        if (
            self.max_context_tokens is not None
            and request.required_context_tokens > self.max_context_tokens
        ):
            raise NoFeasibleRoute("context exceeds route capacity")
        self.epochs.append(epoch)
        primary = RoutePlan(
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
                    rounded_context_tokens=max(128, request.required_context_tokens),
                    exact_kv_bytes=1024,
                ),
            ),
            recovery_level=request.recovery_level,
            coordinator_id=coordinator_id,
            reservation_deadline_ms=reservation_deadline_ms,
            plan_expires_at_ms=plan_expires_at_ms,
        )
        recovery = (
            primary.model_copy(
                update={
                    "route_id": f"route-{epoch}-recovery",
                    "stages": (
                        RouteStage(
                            worker_id="recovery-worker",
                            endpoint_id=RECOVERY_ENDPOINT,
                            hosted_span=LayerSpan(start=0, end=2),
                            effective_span=LayerSpan(start=0, end=2),
                            path_to_next=PathKind.DIRECT,
                            rounded_context_tokens=max(
                                128,
                                request.required_context_tokens,
                            ),
                            exact_kv_bytes=1024,
                        ),
                    ),
                }
            )
            if request.recovery_level == RecoveryLevel.RECOVERABLE
            else None
        )
        return SimpleNamespace(plan=primary, recovery_plan=recovery)


def committed(plan: RoutePlan, expires_at_ms: int) -> CommittedRoute:
    stage = plan.stages[0]
    return CommittedRoute(
        plan=plan,
        leases=(
            ReservationLease(
                reservation_id=f"{plan.route_id}:{stage.worker_id}",
                request_id=plan.request_id,
                route_id=plan.route_id,
                epoch=plan.epoch,
                worker_id=stage.worker_id,
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
        self.fenced = []
        self.fail_renew = False
        self.fail_renew_route = None
        self.fail_reserve_route = None

    def reserve(self, plan):
        if plan.route_id == self.fail_reserve_route:
            raise RuntimeError("backup capacity disappeared")
        self.reserved.append(plan)
        return committed(plan, self.now[0] + 60_000)

    def renew(self, route, *, ttl_ms):
        if self.fail_renew or route.plan.route_id == self.fail_renew_route:
            raise RuntimeError("worker unreachable")
        self.renewed.append(route)
        return committed(route.plan, self.now[0] + ttl_ms)

    def release(self, route):
        self.released.append(route)

    def fence(self, route, *, epoch):
        self.fenced.append((route, epoch))


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
        assert active.authority("request") == {
            "route_id": "route-1",
            "epoch": 1,
            "recovery_level": "restartable",
        }
        snapshot = active.snapshot()["active_routes"][0]
        assert snapshot["prompt_tokens"] == 100
        assert snapshot["reserved_output_tokens"] == 20
        assert snapshot["required_context_tokens"] == 120

        assert active.release("request")
        assert not active.is_active("request")
        assert active.authority("request") is None
        assert len(coordinator.released) == 1
    finally:
        active.close()


def test_readiness_probe_plans_without_reserving_or_advancing_epoch():
    now = [1_000]
    active, planner, coordinator, _ = runtime(now)
    try:
        assert active.route_available(16_384)
        assert planner.epochs == [0]
        assert coordinator.reserved == []

        # The bounded cache avoids rebuilding the same large DHT graph for
        # every IDE status poll.
        planner.fail_planning = True
        assert active.route_available(16_384)
        now[0] += 1_001
        assert not active.route_available(16_384)
        assert coordinator.reserved == []
    finally:
        active.close()


def test_max_supported_context_uses_exact_route_planner_capacity():
    now = [1_000]
    active, planner, coordinator, _ = runtime(now)
    planner.max_context_tokens = 32_896
    try:
        assert active.max_supported_context_tokens(40_960) == 32_896
        assert coordinator.reserved == []
        assert set(planner.epochs) == {0}
    finally:
        active.close()


def test_recoverable_runtime_reserves_renews_and_releases_backup_route():
    now = [1_000]
    nodes = [
        SimpleNamespace(node_id="worker", is_active=True),
        SimpleNamespace(node_id="recovery-worker", is_active=True),
    ]
    active, _, coordinator, _ = runtime(now, nodes=nodes)
    try:
        assert active.reserve(
            request_id="request",
            prompt_tokens=100,
            reserved_output_tokens=20,
            recovery_level=RecoveryLevel.RECOVERABLE,
        ) == ("worker",)
        assert [route.route_id for route in coordinator.reserved] == [
            "route-1",
            "route-1-recovery",
        ]
        snapshot = active.snapshot()["active_routes"][0]
        assert snapshot["recovery_route_id"] == "route-1-recovery"
        assert snapshot["recovery_workers"] == ["recovery-worker"]
        context = active.execution_context("request")
        assert context is not None
        assert context.manifest.model_id == "fabi/test"
        assert context.primary_plan.route_id == "route-1"
        assert context.recovery_plan is not None
        assert context.recovery_plan.route_id == "route-1-recovery"

        now[0] += 201
        active.maintain_once()
        assert [route.plan.route_id for route in coordinator.renewed] == [
            "route-1",
            "route-1-recovery",
        ]

        assert active.release("request")
        assert [route.plan.route_id for route in coordinator.released] == [
            "route-1-recovery",
            "route-1",
        ]
    finally:
        active.close()


def test_recoverable_runtime_rolls_back_primary_when_backup_prepare_fails():
    now = [1_000]
    active, _, coordinator, _ = runtime(now)
    coordinator.fail_reserve_route = "route-1-recovery"
    try:
        with pytest.raises(RuntimeError, match="backup capacity"):
            active.reserve(
                request_id="request",
                prompt_tokens=100,
                reserved_output_tokens=20,
                recovery_level=RecoveryLevel.RECOVERABLE,
            )
        assert not active.is_active("request")
        assert [route.plan.route_id for route in coordinator.released] == ["route-1"]
    finally:
        active.close()


def test_lost_backup_downgrades_recovery_without_killing_healthy_primary():
    now = [1_000]
    nodes = [
        SimpleNamespace(node_id="worker", is_active=True),
        SimpleNamespace(node_id="recovery-worker", is_active=True),
    ]
    active, _, coordinator, _ = runtime(now, nodes=nodes)
    try:
        active.reserve(
            request_id="request",
            prompt_tokens=100,
            reserved_output_tokens=20,
            recovery_level=RecoveryLevel.RECOVERABLE,
        )
        nodes[1].is_active = False

        active.maintain_once()

        assert active.is_active("request")
        assert active.authority("request")["recovery_level"] == "restartable"
        context = active.execution_context("request")
        assert context is not None
        assert context.recovery_plan is None
        assert context.effective_recovery_level == RecoveryLevel.RESTARTABLE
        assert [route.plan.route_id for route in coordinator.released] == ["route-1-recovery"]
        degradation = active.snapshot()["recent_recovery_degradations"][0]
        assert degradation["recovery_route_id"] == "route-1-recovery"
    finally:
        active.close()


def test_departed_primary_is_retained_only_until_reserved_backup_promotion():
    now = [1_000]
    nodes = [
        SimpleNamespace(node_id="worker", is_active=True),
        SimpleNamespace(node_id="recovery-worker", is_active=True),
    ]
    active, _, coordinator, _ = runtime(now, nodes=nodes)
    try:
        active.reserve(
            request_id="request",
            prompt_tokens=100,
            reserved_output_tokens=20,
            recovery_level=RecoveryLevel.RECOVERABLE,
        )
        nodes[0].is_active = False

        active.maintain_once()

        assert not active.is_active("request")
        assert active.authority("request") is None
        pending = active.snapshot()["active_routes"][0]
        assert pending["primary_failure"] is not None
        assert pending["recovery_route_id"] == "route-1-recovery"

        promoted = active.promote_recovery("request", failed_epoch=1)

        assert promoted.primary_plan.route_id == "route-1-recovery-promotion-2"
        assert promoted.primary_plan.epoch == 2
        assert promoted.recovery_plan is None
        assert promoted.effective_recovery_level == RecoveryLevel.RESTARTABLE
        assert active.is_active("request")
        assert active.authority("request") == {
            "route_id": "route-1-recovery-promotion-2",
            "epoch": 2,
            "recovery_level": "restartable",
        }
        assert [plan.route_id for plan in coordinator.reserved] == [
            "route-1",
            "route-1-recovery",
            "route-1-recovery-promotion-2",
        ]
        assert [(route.plan.route_id, epoch) for route, epoch in coordinator.fenced] == [
            ("route-1", 2)
        ]
    finally:
        active.close()


def test_promotion_failure_invalidates_primary_and_backup_capacity():
    now = [1_000]
    nodes = [
        SimpleNamespace(node_id="worker", is_active=True),
        SimpleNamespace(node_id="recovery-worker", is_active=True),
    ]
    active, _, coordinator, _ = runtime(now, nodes=nodes)
    try:
        active.reserve(
            request_id="request",
            prompt_tokens=100,
            reserved_output_tokens=20,
            recovery_level=RecoveryLevel.RECOVERABLE,
        )
        coordinator.fail_reserve_route = "route-1-recovery-promotion-2"

        with pytest.raises(RuntimeError, match="promotion failed"):
            active.promote_recovery("request", failed_epoch=1)

        assert not active.is_active("request")
        assert active.execution_context("request") is None
        assert [route.plan.route_id for route in coordinator.released] == [
            "route-1-recovery",
            "route-1",
        ]
    finally:
        active.close()


def test_backup_renewal_failure_downgrades_after_primary_renewal():
    now = [1_000]
    nodes = [
        SimpleNamespace(node_id="worker", is_active=True),
        SimpleNamespace(node_id="recovery-worker", is_active=True),
    ]
    active, _, coordinator, _ = runtime(now, nodes=nodes)
    try:
        active.reserve(
            request_id="request",
            prompt_tokens=100,
            reserved_output_tokens=20,
            recovery_level=RecoveryLevel.RECOVERABLE,
        )
        coordinator.fail_renew_route = "route-1-recovery"
        now[0] += 201

        active.maintain_once()

        assert active.is_active("request")
        assert [route.plan.route_id for route in coordinator.renewed] == ["route-1"]
        assert [route.plan.route_id for route in coordinator.released] == ["route-1-recovery"]
        assert active.snapshot()["active_routes"][0]["effective_recovery_level"] == ("restartable")
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
