import time
from types import SimpleNamespace

import pytest

from backend.server.contribution_gate import (
    ContributionGate,
    ContributionPermitDenied,
    account_hash,
)
from backend.server.route_permits import RoutePermitConflict, SqliteRoutePermitLedger
from backend.server.rpc_connection_handler import node_log_summary
from fabi_network.capability import RouteRecoveryPolicy
from swarm_protocol import (
    BackendKind,
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    ModelMemberAdvertisement,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)

CREDENTIAL = "ab" * 32
COORDINATOR = "12" * 32
MODEL = "34" * 32


def worker(
    credential=CREDENTIAL,
    *,
    ready=True,
    allocated=True,
    heartbeat=None,
    kv_capacity=32768,
):
    return SimpleNamespace(
        account_hash=account_hash(credential),
        is_active=ready,
        start_layer=0 if allocated else None,
        end_layer=2 if allocated else None,
        last_heartbeat=time.time() if heartbeat is None else heartbeat,
        effective_kv_cache_token_capacity=kv_capacity,
    )


def scheduler(*nodes, serving=True, timeout=30):
    return SimpleNamespace(
        node_manager=SimpleNamespace(active_nodes=list(nodes)),
        heartbeat_timeout=timeout,
        serving_ready=lambda: serving,
    )


def autonomous_worker(credential=CREDENTIAL, *, state="ready"):
    now_ms = time.time_ns() // 1_000_000
    worker_id = "autonomous-worker"
    advertisement = ModelMemberAdvertisement(
        offer=WorkerOffer(
            worker_id=worker_id,
            endpoint_id=worker_id,
            runtime_version="test",
            platform="test",
            backend=BackendKind.MLX,
            stable_memory_envelope_bytes=1_000_000,
            supported_roles=frozenset({WorkerRole.EXECUTOR, WorkerRole.FRONTEND}),
            offer_seq=1,
            issued_at_ms=now_ms,
            expires_at_ms=now_ms + 60_000,
        ),
        lease=SpanLease(
            model_swarm_id="11" * 32,
            worker_id=worker_id,
            hosted_span=LayerSpan(start=0, end=1),
            effective_span_mode=EffectiveSpanMode.FIXED,
            state=SpanState.READY,
            weight_hashes=("22" * 32,),
            kv_geometry=KvGeometry(
                block_size_tokens=16,
                bytes_per_token_by_layer=(16,),
                allocatable_bytes=1_000_000,
            ),
            available_kv_bytes_snapshot=1_000_000,
            max_sessions=1,
            lease_seq=1,
            issued_at_ms=now_ms,
            expires_at_ms=now_ms + 60_000,
        ),
    )
    return SimpleNamespace(
        node_id=worker_id,
        account_hash=account_hash(credential),
        is_active=True,
        liveness_state="healthy",
        uses_autonomous_placement=True,
        last_heartbeat=time.time(),
        effective_kv_cache_token_capacity=32_768,
        swarm_v3={
            "placement_mode": "autonomous",
            "state": state,
            "advertisement": advertisement.model_dump(mode="json"),
        },
    )


def test_gate_is_open_by_default(monkeypatch):
    monkeypatch.delenv("FABI_GATE", raising=False)
    gate = ContributionGate()
    assert gate.status(None, None).allowed is True
    assert gate.status(None, None).reason == "gate_disabled"


def test_gate_requires_a_valid_local_credential(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    gate = ContributionGate()
    assert gate.status(None, scheduler()).reason == "missing_credential"
    assert gate.status("not-a-secret", scheduler()).reason == "invalid_credential"


def test_only_ready_allocated_measured_workers_are_eligible(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    gate = ContributionGate()

    assert gate.status(CREDENTIAL, scheduler(worker())).allowed is True
    assert gate.status(CREDENTIAL, scheduler(worker(ready=False))).reason == "no_eligible_worker"
    assert (
        gate.status(CREDENTIAL, scheduler(worker(allocated=False))).reason == "no_eligible_worker"
    )
    assert (
        gate.status(CREDENTIAL, scheduler(worker(kv_capacity=None))).reason == "no_eligible_worker"
    )
    stale = worker(heartbeat=time.time() - 31)
    assert gate.status(CREDENTIAL, scheduler(stale)).reason == "no_eligible_worker"


def test_account_must_own_the_ready_worker(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    gate = ContributionGate()
    other = "cd" * 32
    assert gate.status(other, scheduler(worker())).reason == "no_eligible_worker"


def test_complete_swarm_is_a_separate_admission_precondition(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    gate = ContributionGate()
    status = gate.status(CREDENTIAL, scheduler(worker(), serving=False))
    assert status.allowed is False
    assert status.reason == "swarm_not_ready"
    assert status.eligible_workers == 1


def test_autonomous_contributor_is_bound_to_ready_dht_membership(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    gate = ContributionGate()
    node = autonomous_worker()
    live_ids = {node.node_id}
    sched = SimpleNamespace(
        node_manager=SimpleNamespace(nodes=[node], active_nodes=[]),
        heartbeat_timeout=30,
        external_ready_worker_ids=lambda: frozenset(live_ids),
        product_serving_ready=lambda: True,
        serving_ready=lambda: False,
    )

    assert gate.status(CREDENTIAL, sched).allowed is True
    live_ids.clear()
    assert gate.status(CREDENTIAL, sched).reason == "no_eligible_worker"


def test_autonomous_contributor_must_publish_verified_ready_lease(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    gate = ContributionGate()
    node = autonomous_worker(state="warming")
    sched = SimpleNamespace(
        node_manager=SimpleNamespace(nodes=[node], active_nodes=[]),
        heartbeat_timeout=30,
        external_ready_worker_ids=lambda: frozenset({node.node_id}),
        product_serving_ready=lambda: True,
        serving_ready=lambda: False,
    )

    assert gate.status(CREDENTIAL, sched).reason == "no_eligible_worker"


def test_gate_reports_account_scoped_memory_standby_without_peer_identity(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    gate = ContributionGate()
    node = autonomous_worker(state="waiting_contract")
    node.swarm_v3.update(
        placement={
            "phase": "standby",
            "decision": "no_exact_span_fits_the_stable_memory_envelope",
        },
        capacity={"usable_memory_bytes": 178_421_760},
    )
    sched = SimpleNamespace(
        node_manager=SimpleNamespace(nodes=[node], active_nodes=[]),
        heartbeat_timeout=30,
        external_ready_worker_ids=lambda: frozenset(),
        product_serving_ready=lambda: True,
        serving_ready=lambda: False,
    )

    status = gate.status(CREDENTIAL, sched)
    assert status.worker_state == "insufficient_memory"
    assert status.worker_usable_memory_bytes == 178_421_760
    assert "autonomous-worker" not in str(status.public_payload(enabled=True))


def test_v3_membership_authority_never_accepts_legacy_scheduler_worker(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    gate = ContributionGate()
    node = worker()
    node.node_id = "stale-scheduler-worker"
    sched = SimpleNamespace(
        node_manager=SimpleNamespace(nodes=[node], active_nodes=[node]),
        heartbeat_timeout=30,
        external_ready_worker_ids=lambda: frozenset({node.node_id}),
        product_serving_ready=lambda: True,
        serving_ready=lambda: True,
    )

    assert gate.status(CREDENTIAL, sched).reason == "no_eligible_worker"


def test_one_concurrent_request_per_ready_worker(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    monkeypatch.setenv("FABI_GATE_REQUESTS_PER_WORKER", "1")
    gate = ContributionGate()
    live = scheduler(worker())

    first = gate.admit(CREDENTIAL, live)
    assert first.allowed is True
    assert gate.admit(CREDENTIAL, live).status.reason == "capacity_reached"
    live.serving_ready = lambda: False
    assert gate.status(CREDENTIAL, live).reason == "capacity_reached"
    gate.release(first)
    live.serving_ready = lambda: True
    assert gate.admit(CREDENTIAL, live).allowed is True


def test_route_permit_and_gateway_share_one_contribution_slot(monkeypatch, tmp_path):
    monkeypatch.setenv("FABI_GATE", "on")
    gate = ContributionGate()
    ledger = SqliteRoutePermitLedger(tmp_path / "permits.sqlite3")
    gate.bind_route_permit_ledger(ledger)
    live = scheduler(worker())
    policies = frozenset({RouteRecoveryPolicy.REPLAN_COLD})

    permit = gate.issue_route_permit(
        CREDENTIAL,
        live,
        request_id="request",
        idempotency_key="permit-request",
        coordinator_endpoint_id=COORDINATOR,
        model_swarm_id=MODEL,
        max_context_tokens=16_384,
        recovery_policies=policies,
        ttl_ms=60_000,
    )
    assert gate.status(CREDENTIAL, live).reason == "capacity_reached"
    assert gate.admit(CREDENTIAL, live).status.reason == "capacity_reached"

    retry = gate.issue_route_permit(
        CREDENTIAL,
        live,
        request_id="request",
        idempotency_key="permit-request",
        coordinator_endpoint_id=COORDINATOR,
        model_swarm_id=MODEL,
        max_context_tokens=16_384,
        recovery_policies=policies,
        ttl_ms=60_000,
    )
    assert retry == permit
    with pytest.raises(RoutePermitConflict):
        gate.issue_route_permit(
            CREDENTIAL,
            live,
            request_id="request",
            idempotency_key="permit-request",
            coordinator_endpoint_id=COORDINATOR,
            model_swarm_id=MODEL,
            max_context_tokens=32_768,
            recovery_policies=policies,
            ttl_ms=60_000,
        )

    assert ledger.release_owned(permit.permit_id, account_hash(CREDENTIAL))
    gateway = gate.admit(CREDENTIAL, live)
    assert gateway.allowed is True
    with pytest.raises(ContributionPermitDenied, match="capacity_reached"):
        gate.issue_route_permit(
            CREDENTIAL,
            live,
            request_id="other",
            idempotency_key="permit-other",
            coordinator_endpoint_id=COORDINATOR,
            model_swarm_id=MODEL,
            max_context_tokens=16_384,
            recovery_policies=policies,
            ttl_ms=60_000,
        )


def test_worker_rpc_log_summary_never_contains_account_credential():
    message = {
        "node_id": "peer-a",
        "account_token": CREDENTIAL,
        "hardware": {"gpu_name": "M4", "device": "mlx"},
    }
    rendered = repr(node_log_summary(message))
    assert "peer-a" in rendered
    assert CREDENTIAL not in rendered
    assert "account_token" not in rendered
