from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.server.openai_compat import encode_http_response_envelope
from backend.server.request_agent_frontend import (
    RequestAgentOpenAIManager,
    _loopback_host,
    _verified_frontend_assets,
    create_request_agent_app,
)
from swarm_protocol.contracts import ArtifactRole
from fabi_network.capability import RouteRecoveryPolicy

MODEL_SWARM_ID = "11" * 32
ENDPOINT_ID = "22" * 32
API_CREDENTIAL = "33" * 32
AUTH_HEADERS = {"Authorization": f"Bearer {API_CREDENTIAL}"}


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert messages
        assert kwargs["tokenize"] is True
        return [10, 20, 30]


class FakeResponse:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.cancelled = False

    def __iter__(self):
        return iter(self._chunks)

    def cancel(self):
        self.cancelled = True


class FakeCompletionStub:
    def __init__(self):
        self.requests = []
        self.aborts = []

    def tokenize_chat(self, request):
        assert request["request"]["messages"]
        return {"ok": True, "tokens": [10, 20, 30], "count": 3}

    def chat_completion(self, request):
        self.requests.append(request)
        if request.get("stream"):
            return FakeResponse(
                [
                    b'data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"ok"}}]}\n\n',
                    b"data: [DONE]\n\n",
                ]
            )
        body = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
        return FakeResponse(
            [
                encode_http_response_envelope(
                    status_code=200,
                    content_type="application/json",
                    body=body,
                )
            ]
        )

    def abort_completion(self, request):
        self.aborts.append(request)
        return {"aborted": True}


class FakeTransport:
    def __init__(self, stub):
        self.stub_instance = stub
        self.calls = []

    def stub(self, endpoint_id, service_type):
        self.calls.append((endpoint_id, service_type))
        return self.stub_instance


class FakeRuntime:
    def __init__(self):
        manifest = SimpleNamespace(
            model_swarm_id=MODEL_SWARM_ID,
            model_id="fabi/test-model",
        )
        self.registry = SimpleNamespace(
            fetch=lambda model_swarm_id: SimpleNamespace(manifest=manifest)
        )
        self.stub = FakeCompletionStub()
        self.transport = FakeTransport(self.stub)
        self.active = {}
        self.requests = []
        self.released = []
        self.closed = False

    def max_supported_context_tokens(self, model_swarm_id, upper_bound):
        assert model_swarm_id == MODEL_SWARM_ID
        return min(4096, upper_bound)

    def reserve(self, request):
        self.requests.append(request)
        plan = SimpleNamespace(
            request_id=request.request_id,
            route_id=f"route-{request.request_id}",
            epoch=1,
            stages=(SimpleNamespace(worker_id="worker-head", endpoint_id=ENDPOINT_ID),),
        )
        reservation = SimpleNamespace(
            committed=SimpleNamespace(plan=plan),
            recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        )
        self.active[request.request_id] = reservation
        return reservation

    def active_reservation(self, request_id):
        return self.active.get(request_id)

    def release(self, reservation):
        request_id = reservation.committed.plan.request_id
        self.released.append(request_id)
        self.active.pop(request_id, None)

    def status(self):
        return {"mode": "request_agent", "active_routes": []}

    def close(self):
        self.closed = True


def manager_and_runtime():
    runtime = FakeRuntime()
    manager = RequestAgentOpenAIManager(
        runtime,
        MODEL_SWARM_ID,
        tokenizer=FakeTokenizer(),
        model_context_limit=8192,
        completion_service_type=object,
    )
    return manager, runtime


def test_frontend_materializes_only_tuf_signed_runtime_assets(monkeypatch, tmp_path):
    files = {
        "config.json": b'{"max_position_embeddings":4096}',
        "tokenizer/tokenizer.json": b'{"version":"1.0"}',
    }
    descriptors = []
    for logical_path, content in files.items():
        path = tmp_path / logical_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        descriptors.append(
            SimpleNamespace(
                path=logical_path,
                role=(
                    ArtifactRole.ARCHITECTURE
                    if logical_path == "config.json"
                    else ArtifactRole.TOKENIZER
                ),
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    bundle = SimpleNamespace(
        manifest=SimpleNamespace(model_id="fabi/test", immutable_revision="revision"),
        artifact_index=SimpleNamespace(artifacts=tuple(descriptors)),
    )
    monkeypatch.setattr(
        "backend.server.request_agent_frontend.download_model_file",
        lambda repo_id, filename, local_files_only, revision: tmp_path / filename,
    )

    assert _verified_frontend_assets(bundle, local_files_only=True) == tmp_path.resolve()
    (tmp_path / "config.json").write_text('{"max_position_embeddings":8192}')
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        _verified_frontend_assets(bundle, local_files_only=True)


def test_local_request_agent_executes_non_streaming_openai_request():
    manager, runtime = manager_and_runtime()
    app = create_request_agent_app(manager, api_credential=API_CREDENTIAL)

    with TestClient(app) as client:
        models = client.get("/v1/models", headers=AUTH_HEADERS)
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={
                "model": "fabi-swarm",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 32,
            },
        )

    assert models.json()["data"][0]["id"] == "fabi/test-model"
    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert len(runtime.requests) == 1
    assert runtime.requests[0].prompt_tokens == 3
    assert runtime.requests[0].reserved_output_tokens == 32
    assert runtime.released == [runtime.requests[0].request_id]
    downstream = runtime.stub.requests[0]
    assert downstream["model"] == "fabi/test-model"
    assert downstream["vllm_xargs"]["parallax_routing_table"] == ["worker-head"]
    assert downstream["vllm_xargs"]["fabi_route_epoch"] == 1
    assert runtime.closed is True


def test_local_request_agent_preserves_openai_sse_and_releases_after_done():
    manager, runtime = manager_and_runtime()
    app = create_request_agent_app(manager, api_credential=API_CREDENTIAL)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={
                "model": "fabi-swarm",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 8,
                "stream": True,
            },
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b'"content":"ok"' in body
    assert body.endswith(b"data: [DONE]\n\n")
    assert len(runtime.released) == 1


def test_local_request_agent_rejects_invalid_json_and_non_loopback_bind():
    manager, _runtime = manager_and_runtime()
    app = create_request_agent_app(manager, api_credential=API_CREDENTIAL)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=b"[",
            headers={**AUTH_HEADERS, "content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request_error"
    with pytest.raises(Exception, match="loopback"):
        _loopback_host("0.0.0.0")


def test_local_request_agent_rejects_missing_account_credential():
    manager, _runtime = manager_and_runtime()
    app = create_request_agent_app(manager, api_credential=API_CREDENTIAL)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
