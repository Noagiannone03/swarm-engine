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
from swarm_protocol.recovery import RecoveryState
from swarm_protocol.recovery_sqlite import SqliteRecoveryJournal

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


class FailoverTransport:
    def __init__(self, stubs):
        self.stubs = dict(stubs)

    def stub(self, endpoint_id, service_type):
        del service_type
        return self.stubs[endpoint_id]


class FailoverStub(FakeCompletionStub):
    def __init__(self, *, chat_chunks=(), replay_chunks=()):
        super().__init__()
        self.chat_chunks = tuple(chat_chunks)
        self.replay_chunks = tuple(replay_chunks)
        self.replay_requests = []

    def chat_completion(self, request):
        self.requests.append(request)
        return FakeResponse(self.chat_chunks)

    def replay_generation(self, request):
        self.replay_requests.append(request)
        return FakeResponse(self.replay_chunks)


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


class FailoverRuntime:
    def __init__(self, primary_stub, replacement_stub):
        self.manifest = SimpleNamespace(
            model_swarm_id=MODEL_SWARM_ID,
            model_id="fabi/test-model",
            immutable_revision="model-commit",
            tokenizer_hash="44" * 32,
            dtype="bfloat16",
            prefill_contract_hash="55" * 32,
            attention_kv_contract_hash="66" * 32,
        )
        self.registry = SimpleNamespace(
            fetch=lambda model_swarm_id: SimpleNamespace(manifest=self.manifest)
        )
        self.transport = FailoverTransport(
            {
                ENDPOINT_ID: primary_stub,
                "77" * 32: replacement_stub,
            }
        )
        self.active = {}
        self.replans = []
        self.released = []

    @staticmethod
    def _plan(request, *, worker_id, endpoint_id, route_id, epoch):
        return SimpleNamespace(
            request_id=request.request_id,
            model_swarm_id=request.model_swarm_id,
            route_id=route_id,
            epoch=epoch,
            prompt_tokens=request.prompt_tokens,
            required_context_tokens=request.required_context_tokens,
            stages=(SimpleNamespace(worker_id=worker_id, endpoint_id=endpoint_id),),
        )

    def max_supported_context_tokens(self, model_swarm_id, upper_bound):
        assert model_swarm_id == MODEL_SWARM_ID
        return min(4096, upper_bound)

    def reserve(self, request):
        plan = self._plan(
            request,
            worker_id="worker-primary",
            endpoint_id=ENDPOINT_ID,
            route_id=f"route-primary-{request.request_id}",
            epoch=1,
        )
        reservation = SimpleNamespace(
            committed=SimpleNamespace(plan=plan),
            recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        )
        self.active[request.request_id] = reservation
        return reservation

    def replan_cold(self, request_id, *, failed_epoch):
        current = self.active[request_id]
        assert failed_epoch == current.committed.plan.epoch
        request = SimpleNamespace(
            request_id=request_id,
            model_swarm_id=current.committed.plan.model_swarm_id,
            prompt_tokens=current.committed.plan.prompt_tokens,
            required_context_tokens=current.committed.plan.required_context_tokens,
        )
        plan = self._plan(
            request,
            worker_id="worker-replacement",
            endpoint_id="77" * 32,
            route_id=f"route-replacement-{request_id}",
            epoch=2,
        )
        replacement = SimpleNamespace(
            committed=SimpleNamespace(plan=plan),
            recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        )
        self.active[request_id] = replacement
        self.replans.append((request_id, failed_epoch))
        return replacement

    def active_reservation(self, request_id):
        return self.active.get(request_id)

    def release_request(self, request_id):
        if self.active.pop(request_id, None) is None:
            return False
        self.released.append(request_id)
        return True

    def status(self):
        return {"mode": "request_agent", "active_routes": []}

    def close(self):
        self.active.clear()


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


def test_local_request_agent_replans_replays_and_resumes_exactly_once(tmp_path):
    primary = FailoverStub(
        chat_chunks=(
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}],'
            b'"prompt_token_ids":[10,20,30]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
            b'"token_ids":[40],"finish_reason":null}]}\n\n',
        )
    )
    replacement = FailoverStub(
        replay_chunks=(
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}],'
            b'"prompt_token_ids":[10,20,30]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
            b'"token_ids":[40],"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":" continued"},'
            b'"token_ids":[41],"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"token_ids":[],'
            b'"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n",
        )
    )
    runtime = FailoverRuntime(primary, replacement)
    journal = SqliteRecoveryJournal(tmp_path / "recovery.sqlite3")
    manager = RequestAgentOpenAIManager(
        runtime,
        MODEL_SWARM_ID,
        tokenizer=FakeTokenizer(),
        model_context_limit=8192,
        completion_service_type=object,
        recovery_journal=journal,
    )
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
                "temperature": 0,
            },
        ) as response:
            body = b"".join(response.iter_bytes())

    request_id = response.headers["x-request-id"]
    snapshot = journal.get(request_id)
    assert response.status_code == 200
    assert body.count(b'"content":"partial"') == 1
    assert body.count(b'"content":" continued"') == 1
    assert body.endswith(b"data: [DONE]\n\n")
    assert runtime.replans == [(request_id, 1)]
    assert runtime.released == [request_id]
    assert replacement.replay_requests[0]["original_prompt_token_ids"] == [10, 20, 30]
    assert replacement.replay_requests[0]["committed_output_token_ids"] == [40]
    assert snapshot is not None
    assert snapshot.state == RecoveryState.COMPLETED
    assert snapshot.route_ids == (
        f"route-primary-{request_id}",
        f"route-replacement-{request_id}",
    )
    assert snapshot.committed_output_token_ids == (40, 41)
    journal.close()


def test_local_request_agent_recovers_when_primary_dies_during_prefill(tmp_path):
    primary = FailoverStub(chat_chunks=())
    replacement = FailoverStub(
        replay_chunks=(
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}],'
            b'"prompt_token_ids":[10,20,30]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"recovered"},'
            b'"token_ids":[41],"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n",
        )
    )
    runtime = FailoverRuntime(primary, replacement)
    journal = SqliteRecoveryJournal(tmp_path / "recovery.sqlite3")
    manager = RequestAgentOpenAIManager(
        runtime,
        MODEL_SWARM_ID,
        tokenizer=FakeTokenizer(),
        model_context_limit=8192,
        completion_service_type=object,
        recovery_journal=journal,
    )
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
                "temperature": 0,
            },
        ) as response:
            body = b"".join(response.iter_bytes())

    request_id = response.headers["x-request-id"]
    snapshot = journal.get(request_id)
    assert body.count(b'"content":"recovered"') == 1
    assert body.endswith(b"data: [DONE]\n\n")
    assert runtime.replans == [(request_id, 1)]
    assert replacement.replay_requests[0]["committed_output_token_ids"] == []
    assert snapshot is not None
    assert snapshot.state == RecoveryState.COMPLETED
    assert snapshot.committed_output_token_ids == (41,)
    journal.close()


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
