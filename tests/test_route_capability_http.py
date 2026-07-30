from __future__ import annotations

import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.server.contribution_gate import ContributionGate, account_hash
from backend.server.route_capability_api import (
    RequestAgentAuthority,
    router,
    set_request_agent_authority,
)
from backend.server.route_permits import SqliteRoutePermitLedger
from swarm_protocol.control import ControlMessageKind, SignedControlMessage

CREDENTIAL = "12" * 32
OTHER_CREDENTIAL = "13" * 32
COORDINATOR = "22" * 32
MODEL = "33" * 32
app = FastAPI()
app.include_router(router)


def live_scheduler():
    node = SimpleNamespace(
        account_hash=account_hash(CREDENTIAL),
        is_active=True,
        start_layer=0,
        end_layer=4,
        last_heartbeat=time.time(),
        effective_kv_cache_token_capacity=32_768,
    )
    return SimpleNamespace(
        node_manager=SimpleNamespace(active_nodes=[node]),
        heartbeat_timeout=30,
        serving_ready=lambda: True,
    )


class FakeCapabilities:
    def __init__(self, account_id):
        self.account_id = account_id
        self.calls = []

    def issue(self, signed_plan, **kwargs):
        if kwargs["account_id"] != self.account_id:
            raise PermissionError("route permit belongs to a different account")
        self.calls.append((signed_plan, kwargs))
        return {"capability": "persisted"}


def install_authority(monkeypatch, tmp_path):
    monkeypatch.setenv("FABI_GATE", "on")
    gate = ContributionGate()
    ledger = SqliteRoutePermitLedger(tmp_path / "permits.sqlite3")
    capabilities = FakeCapabilities(account_hash(CREDENTIAL))
    authority = RequestAgentAuthority(
        gate=gate,
        ledger=ledger,
        capabilities=capabilities,
        scheduler_provider=live_scheduler,
        model_swarm_id_provider=lambda: MODEL,
        max_context_tokens_provider=lambda: 32_768,
    )
    set_request_agent_authority(authority)
    return authority, capabilities


def permit_payload(*, context=16_384):
    return {
        "coordinator_endpoint_id": COORDINATOR,
        "model_swarm_id": MODEL,
        "max_context_tokens": context,
        "recovery_policies": ["replan_cold"],
        "ttl_ms": 60_000,
    }


def auth_headers(*, credential=CREDENTIAL, request_id="request"):
    return {
        "Authorization": f"Bearer {credential}",
        "Idempotency-Key": request_id,
    }


def test_permit_endpoint_is_idempotent_account_scoped_and_shares_capacity(monkeypatch, tmp_path):
    install_authority(monkeypatch, tmp_path)
    client = TestClient(app)
    try:
        first = client.post(
            "/v1/swarm/route-permits",
            headers=auth_headers(),
            json=permit_payload(),
        )
        assert first.status_code == 200
        retry = client.post(
            "/v1/swarm/route-permits",
            headers=auth_headers(),
            json=permit_payload(),
        )
        assert retry.status_code == 200
        assert retry.json() == first.json()

        conflict = client.post(
            "/v1/swarm/route-permits",
            headers=auth_headers(),
            json=permit_payload(context=32_768),
        )
        assert conflict.status_code == 422
        assert conflict.json()["error"]["code"] == "idempotency_key_reused"

        capacity = client.post(
            "/v1/swarm/route-permits",
            headers=auth_headers(request_id="other"),
            json=permit_payload(),
        )
        assert capacity.status_code == 429

        permit_id = first.json()["permit_id"]
        assert (
            client.delete(
                f"/v1/swarm/route-permits/{permit_id}",
                headers=auth_headers(credential=OTHER_CREDENTIAL),
            ).status_code
            == 404
        )
        assert client.delete(
            f"/v1/swarm/route-permits/{permit_id}",
            headers=auth_headers(),
        ).json() == {"released": True}
    finally:
        set_request_agent_authority(None)


def test_permit_requires_idempotency_and_capability_rechecks_account(monkeypatch, tmp_path):
    _, capabilities = install_authority(monkeypatch, tmp_path)
    client = TestClient(app)
    try:
        missing_key = client.post(
            "/v1/swarm/route-permits",
            headers={"Authorization": f"Bearer {CREDENTIAL}"},
            json=permit_payload(),
        )
        assert missing_key.status_code == 400
        assert missing_key.json()["error"]["code"] == "idempotency_key_required"

        permit = client.post(
            "/v1/swarm/route-permits",
            headers=auth_headers(),
            json=permit_payload(),
        ).json()
        signed = SignedControlMessage(
            kind=ControlMessageKind.ROUTE_PLAN,
            signer_endpoint_id=COORDINATOR,
            payload=b"\x8a{}",
            signature=bytes(range(64)),
        )
        capability_payload = {
            "permit_id": permit["permit_id"],
            "signed_plan": signed.model_dump(mode="json"),
            "recovery_policy": "replan_cold",
        }
        denied = client.post(
            "/v1/swarm/route-capabilities",
            headers={"Authorization": f"Bearer {OTHER_CREDENTIAL}"},
            json=capability_payload,
        )
        assert denied.status_code == 403
        accepted = client.post(
            "/v1/swarm/route-capabilities",
            headers={"Authorization": f"Bearer {CREDENTIAL}"},
            json=capability_payload,
        )
        assert accepted.status_code == 200
        assert accepted.json() == {"capability": "persisted"}
        assert len(capabilities.calls) == 1
        assert capabilities.calls[0][0] == signed
    finally:
        set_request_agent_authority(None)


def test_permit_keepalive_rechecks_contribution_and_is_idempotent(monkeypatch, tmp_path):
    authority, _capabilities = install_authority(monkeypatch, tmp_path)
    client = TestClient(app)
    try:
        permit = client.post(
            "/v1/swarm/route-permits",
            headers=auth_headers(),
            json=permit_payload(),
        ).json()
        permit_id = permit["permit_id"]
        keepalive_headers = {
            "Authorization": f"Bearer {CREDENTIAL}",
            "Idempotency-Key": "keepalive-0",
        }
        first = client.post(
            f"/v1/swarm/route-permits/{permit_id}/keepalive",
            headers=keepalive_headers,
            json={"ttl_ms": 60_000},
        )
        assert first.status_code == 200
        assert first.json()["authorization_generation"] == 1
        retry = client.post(
            f"/v1/swarm/route-permits/{permit_id}/keepalive",
            headers=keepalive_headers,
            json={"ttl_ms": 60_000},
        )
        assert retry.json() == first.json()
        original_retry = client.post(
            "/v1/swarm/route-permits",
            headers=auth_headers(),
            json=permit_payload(),
        )
        assert original_retry.status_code == 200
        assert original_retry.json()["authorization_generation"] == 1

        foreign = client.post(
            f"/v1/swarm/route-permits/{permit_id}/keepalive",
            headers={
                "Authorization": f"Bearer {OTHER_CREDENTIAL}",
                "Idempotency-Key": "foreign",
            },
            json={"ttl_ms": 60_000},
        )
        assert foreign.status_code == 404

        authority._scheduler_provider = lambda: SimpleNamespace(
            node_manager=SimpleNamespace(active_nodes=[]),
            heartbeat_timeout=30,
            serving_ready=lambda: True,
        )
        stopped = client.post(
            f"/v1/swarm/route-permits/{permit_id}/keepalive",
            headers={
                "Authorization": f"Bearer {CREDENTIAL}",
                "Idempotency-Key": "keepalive-1",
            },
            json={"ttl_ms": 60_000},
        )
        assert stopped.status_code == 403
        assert stopped.json()["error"]["code"] == "contribution_required"
    finally:
        set_request_agent_authority(None)
