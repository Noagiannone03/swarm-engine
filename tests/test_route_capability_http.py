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
from fabi_network.capability import RouteRecoveryPolicy
from swarm_protocol.control import ControlMessageKind, SignedControlMessage
from swarm_protocol.request_agent import RequestAgentAuthorityClient

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
    unmet_context_requests = []
    admitted_context_requests = []
    renewed_context_requests = []
    completed_context_requests = []
    authority = RequestAgentAuthority(
        gate=gate,
        ledger=ledger,
        capabilities=capabilities,
        scheduler_provider=live_scheduler,
        model_swarm_id_provider=lambda: MODEL,
        max_context_tokens_provider=lambda: 32_768,
        unmet_context_observer=lambda request_id, required: (
            unmet_context_requests.append((request_id, required)) or True
        ),
        context_admission_observer=lambda request_id, model_id, required, expires: (
            admitted_context_requests.append((request_id, model_id, required, expires)) or True
        ),
        context_renewal_observer=lambda request_id, model_id, required, expires: (
            renewed_context_requests.append((request_id, model_id, required, expires)) or True
        ),
        context_completion_observer=lambda request_id, model_id: (
            completed_context_requests.append((request_id, model_id)) or True
        ),
    )
    authority.context_observations = (
        admitted_context_requests,
        renewed_context_requests,
        completed_context_requests,
    )
    set_request_agent_authority(authority)
    return authority, capabilities, unmet_context_requests


def permit_payload(*, request_id="request", context=16_384):
    return {
        "request_id": request_id,
        "coordinator_endpoint_id": COORDINATOR,
        "model_swarm_id": MODEL,
        "max_context_tokens": context,
        "recovery_policies": ["replan_cold"],
        "ttl_ms": 60_000,
    }


def auth_headers(*, credential=CREDENTIAL, idempotency_key="permit-issue"):
    return {
        "Authorization": f"Bearer {credential}",
        "Idempotency-Key": idempotency_key,
    }


def test_permit_endpoint_is_idempotent_account_scoped_and_shares_capacity(monkeypatch, tmp_path):
    authority, _, _ = install_authority(monkeypatch, tmp_path)
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
            headers=auth_headers(idempotency_key="other-permit-issue"),
            json=permit_payload(request_id="other"),
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
        admissions, _, completions = authority.context_observations
        assert [(item[0], item[1], item[2]) for item in admissions] == [
            ("request", MODEL, 16_384),
            ("request", MODEL, 16_384),
        ]
        assert completions == [("request", MODEL)]

        corrected = client.post(
            "/v1/swarm/route-permits",
            headers=auth_headers(idempotency_key="corrected-token-budget"),
            json=permit_payload(context=16_393),
        )
        assert corrected.status_code == 200
        assert corrected.json()["request_id"] == "request"
        assert corrected.json()["max_context_tokens"] == 16_393
        assert corrected.json()["permit_id"] != permit_id
    finally:
        set_request_agent_authority(None)


def test_request_agent_reuses_exact_contract_key_but_rekeys_token_correction(
    monkeypatch,
    tmp_path,
):
    install_authority(monkeypatch, tmp_path)
    transport = TestClient(app)
    client = RequestAgentAuthorityClient(
        "http://127.0.0.1",
        CREDENTIAL,
        session=transport,
    )
    kwargs = {
        "request_id": "opencode-request",
        "coordinator_endpoint_id": COORDINATOR,
        "model_swarm_id": MODEL,
        "max_context_tokens": 16_316,
        "recovery_policies": (RouteRecoveryPolicy.REPLAN_COLD,),
        "ttl_ms": 60_000,
    }
    try:
        first = client.issue_permit(**kwargs)
        assert client.issue_permit(**kwargs) == first
        assert client.release_permit(first.permit_id)

        corrected = client.issue_permit(
            **{
                **kwargs,
                "max_context_tokens": 16_325,
            }
        )
        assert corrected.request_id == first.request_id
        assert corrected.permit_id != first.permit_id
        assert corrected.max_context_tokens == 16_325
    finally:
        set_request_agent_authority(None)


def test_context_demand_is_authenticated_and_available_to_request_agent_client(
    monkeypatch,
    tmp_path,
):
    _authority, _capabilities, observed = install_authority(monkeypatch, tmp_path)
    transport = TestClient(app)
    client = RequestAgentAuthorityClient(
        "http://127.0.0.1",
        CREDENTIAL,
        session=transport,
    )
    try:
        assert client.observe_unmet_context_demand(
            request_id="long-request",
            model_swarm_id=MODEL,
            required_context_tokens=40_960,
        )
        assert observed == [("long-request", 40_960)]

        foreign = transport.post(
            "/v1/swarm/context-demand",
            headers={"Authorization": f"Bearer {OTHER_CREDENTIAL}"},
            json={
                "request_id": "foreign",
                "model_swarm_id": MODEL,
                "required_context_tokens": 40_960,
            },
        )
        assert foreign.status_code == 403
        assert foreign.json()["error"]["code"] == "contribution_required"

        wrong_model = transport.post(
            "/v1/swarm/context-demand",
            headers={"Authorization": f"Bearer {CREDENTIAL}"},
            json={
                "request_id": "wrong-model",
                "model_swarm_id": "44" * 32,
                "required_context_tokens": 40_960,
            },
        )
        assert wrong_model.status_code == 403
        assert wrong_model.json()["error"]["code"] == "model_not_authorized"
        assert observed == [("long-request", 40_960)]
    finally:
        set_request_agent_authority(None)


def test_permit_requires_idempotency_and_capability_rechecks_account(monkeypatch, tmp_path):
    _, capabilities, _ = install_authority(monkeypatch, tmp_path)
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
    authority, _capabilities, _ = install_authority(monkeypatch, tmp_path)
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
        _, renewals, _ = authority.context_observations
        assert [(item[0], item[1], item[2]) for item in renewals] == [
            ("request", MODEL, 16_384),
            ("request", MODEL, 16_384),
        ]
        assert all(item[3] == first.json()["expires_at_ms"] for item in renewals)
        original_retry = client.post(
            "/v1/swarm/route-permits",
            headers=auth_headers(),
            json=permit_payload(),
        )
        assert original_retry.status_code == 200
        assert original_retry.json()["authorization_generation"] == 1

        # The in-flight request may consume the route's only free session.  A
        # permit keepalive renews that existing lease; it must not ask whether
        # a second request could be admitted through a fresh route snapshot.
        busy_scheduler = live_scheduler()
        busy_scheduler.serving_ready = lambda: False
        authority._scheduler_provider = lambda: busy_scheduler
        busy = client.post(
            f"/v1/swarm/route-permits/{permit_id}/keepalive",
            headers={
                "Authorization": f"Bearer {CREDENTIAL}",
                "Idempotency-Key": "keepalive-busy-route",
            },
            json={"ttl_ms": 60_000},
        )
        assert busy.status_code == 200
        assert busy.json()["authorization_generation"] == 2

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
