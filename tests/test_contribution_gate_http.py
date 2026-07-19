import json
import time
from types import SimpleNamespace

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.server.contribution_gate import ContributionGate, account_hash


CREDENTIAL = "12" * 32
AUTH = {"Authorization": f"Bearer {CREDENTIAL}"}


def live_scheduler():
    node = SimpleNamespace(
        account_hash=account_hash(CREDENTIAL),
        is_active=True,
        start_layer=0,
        end_layer=28,
        last_heartbeat=time.time(),
        effective_kv_cache_token_capacity=32768,
    )
    return SimpleNamespace(
        node_manager=SimpleNamespace(active_nodes=[node]),
        heartbeat_timeout=30,
        serving_ready=lambda: True,
    )


def install_gate(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    gate = ContributionGate()
    monkeypatch.setattr(backend_main, "get_gate", lambda: gate)
    monkeypatch.setattr(
        backend_main,
        "scheduler_manage",
        SimpleNamespace(scheduler=live_scheduler()),
    )
    return gate


def test_status_endpoint_is_account_scoped(monkeypatch):
    install_gate(monkeypatch)
    client = TestClient(backend_main.app)

    assert client.get("/v1/contribution/status").json()["allowed"] is False
    status = client.get("/v1/contribution/status", headers=AUTH).json()
    assert status == {
        "enabled": True,
        "allowed": True,
        "reason": "eligible",
        "eligible_workers": 1,
        "active_requests": 0,
        "max_concurrent_requests": 1,
    }


def test_chat_rejects_non_contributors_before_inference(monkeypatch):
    install_gate(monkeypatch)
    called = False

    async def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return JSONResponse({})

    monkeypatch.setattr(backend_main.request_handler, "v1_chat_completions", should_not_run)
    response = TestClient(backend_main.app).post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "contribution_required"
    assert called is False


def test_non_stream_admission_is_released(monkeypatch):
    gate = install_gate(monkeypatch)

    async def answer(*_args, **_kwargs):
        return JSONResponse({"ok": True})

    monkeypatch.setattr(backend_main.request_handler, "v1_chat_completions", answer)
    response = TestClient(backend_main.app).post(
        "/v1/chat/completions", headers=AUTH, json={"messages": []}
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert gate.status(CREDENTIAL, backend_main.scheduler_manage.scheduler).active_requests == 0


def test_stream_admission_lives_until_body_finishes(monkeypatch):
    gate = install_gate(monkeypatch)

    async def answer(*_args, **_kwargs):
        async def body():
            yield b"data: " + json.dumps({"ok": True}).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    monkeypatch.setattr(backend_main.request_handler, "v1_chat_completions", answer)
    response = TestClient(backend_main.app).post(
        "/v1/chat/completions", headers=AUTH, json={"messages": [], "stream": True}
    )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert gate.status(CREDENTIAL, backend_main.scheduler_manage.scheduler).active_requests == 0
