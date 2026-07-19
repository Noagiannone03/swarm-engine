import time
from types import SimpleNamespace

from backend.server.contribution_gate import ContributionGate, account_hash
from backend.server.rpc_connection_handler import node_log_summary


CREDENTIAL = "ab" * 32


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
    assert gate.status(CREDENTIAL, scheduler(worker(allocated=False))).reason == "no_eligible_worker"
    assert gate.status(CREDENTIAL, scheduler(worker(kv_capacity=None))).reason == "no_eligible_worker"
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


def test_one_concurrent_request_per_ready_worker(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    monkeypatch.setenv("FABI_GATE_REQUESTS_PER_WORKER", "1")
    gate = ContributionGate()
    live = scheduler(worker())

    first = gate.admit(CREDENTIAL, live)
    assert first.allowed is True
    assert gate.admit(CREDENTIAL, live).status.reason == "capacity_reached"
    gate.release(first)
    assert gate.admit(CREDENTIAL, live).allowed is True


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
