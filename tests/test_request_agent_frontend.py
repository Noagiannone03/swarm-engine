from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.server.openai_compat import encode_http_response_envelope
from backend.server.request_agent_frontend import (
    RequestAgentOpenAIManager,
    _ReadyFileServer,
    _bound_base_url,
    _encode_status_sse,
    _loopback_host,
    _with_sse_keepalive,
    _write_ready_file,
    _verified_frontend_assets,
    create_request_agent_app,
)
from swarm_protocol.contracts import ArtifactRole, RecoveryLevel, RequestContract
from fabi_network.capability import RouteRecoveryPolicy
from swarm_protocol.recovery import RecoveryState
from swarm_protocol.recovery_sqlite import SqliteRecoveryJournal
from swarm_protocol.speculative import (
    SpeculativeSampling,
    SpeculativeStrategy,
    SpeculativeVerifyResponse,
    SpeculativeVerifyWindow,
    SpeculativeWindowFence,
)

MODEL_SWARM_ID = "11" * 32
ENDPOINT_ID = "22" * 32
API_CREDENTIAL = "33" * 32
AUTH_HEADERS = {"Authorization": f"Bearer {API_CREDENTIAL}"}


def test_openai_sse_keepalive_does_not_cancel_a_quiet_source() -> None:
    async def collect() -> list[bytes]:
        async def delayed_source():
            await asyncio.sleep(0.035)
            yield b'data: {"token":"ready"}\n\n'

        return [
            chunk
            async for chunk in _with_sse_keepalive(
                delayed_source(),
                interval_seconds=0.01,
            )
        ]

    chunks = asyncio.run(collect())

    assert chunks[-1] == b'data: {"token":"ready"}\n\n'
    assert chunks[:-1]
    assert set(chunks[:-1]) == {b": keepalive\n\n"}


def test_openai_sse_keepalive_propagates_consumer_cancellation() -> None:
    async def exercise() -> bool:
        closed = asyncio.Event()

        async def blocked_source():
            try:
                await asyncio.Event().wait()
                yield b"unreachable"
            finally:
                closed.set()

        async def consume() -> None:
            async for _chunk in _with_sse_keepalive(
                blocked_source(),
                interval_seconds=30.0,
            ):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return closed.is_set()

    assert asyncio.run(exercise()) is True


def test_request_agent_frontend_module_is_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.server.request_agent_frontend",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "usage: fabi-request-agent" in completed.stdout


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


class FailingReplayStub(FailoverStub):
    """Replay stream that loses its transport after a deterministic prefix."""

    def replay_generation(self, request):
        self.replay_requests.append(request)
        chunks = self.replay_chunks

        class _FailingReplayResponse(FakeResponse):
            def __iter__(self):
                yield from chunks
                raise ConnectionError("replacement worker disappeared during replay")

        return _FailingReplayResponse(())


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
        self.unmet_context_requests = []
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
            committed=SimpleNamespace(plan=plan, route_plan_digest="a" * 64),
            recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        )
        self.active[request.request_id] = reservation
        return reservation

    def observe_unmet_context_demand(
        self,
        request_id,
        model_swarm_id,
        required_context_tokens,
    ):
        self.unmet_context_requests.append((request_id, model_swarm_id, required_context_tokens))
        return True

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
            committed=SimpleNamespace(plan=plan, route_plan_digest="a" * 64),
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
            committed=SimpleNamespace(plan=plan, route_plan_digest="b" * 64),
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


class RepeatedFailoverRuntime(FailoverRuntime):
    """Deterministic three-route harness for a failure during cold replay."""

    def __init__(self, primary_stub, first_replacement_stub, second_replacement_stub):
        super().__init__(primary_stub, first_replacement_stub)
        self.transport.stubs["88" * 32] = second_replacement_stub

    def replan_cold(self, request_id, *, failed_epoch):
        current = self.active[request_id]
        assert failed_epoch == current.committed.plan.epoch
        next_epoch = failed_epoch + 1
        if next_epoch == 2:
            worker_id = "worker-replacement-one"
            endpoint_id = "77" * 32
        elif next_epoch == 3:
            worker_id = "worker-replacement-two"
            endpoint_id = "88" * 32
        else:  # pragma: no cover - the test owns exactly two replacements
            raise RuntimeError("no additional replacement route")
        request = SimpleNamespace(
            request_id=request_id,
            model_swarm_id=current.committed.plan.model_swarm_id,
            prompt_tokens=current.committed.plan.prompt_tokens,
            required_context_tokens=current.committed.plan.required_context_tokens,
        )
        plan = self._plan(
            request,
            worker_id=worker_id,
            endpoint_id=endpoint_id,
            route_id=f"route-replacement-{next_epoch}-{request_id}",
            epoch=next_epoch,
        )
        replacement = SimpleNamespace(
            committed=SimpleNamespace(plan=plan, route_plan_digest="a" * 64),
            recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        )
        self.active[request_id] = replacement
        self.replans.append((request_id, failed_epoch))
        return replacement


class FakeRecoveryCheckpoints:
    def __init__(self):
        self.observed = []
        self.restored = []
        self.finished = []
        self.closed = False

    def observe_committed(self, snapshot, plan):
        self.observed.append((snapshot.committed_position, plan.epoch))
        return True

    def checkpoint_for_recovery(self, *, request_id, failed_epoch, replay_token_ids):
        assert failed_epoch == 1
        assert replay_token_ids == (10, 20, 30, 40)
        return SimpleNamespace(request_id=request_id, token_count=3)

    def restore(self, checkpoint, *, snapshot, plan):
        self.restored.append((checkpoint.token_count, snapshot.epoch, plan.epoch))
        return True

    def finish_request(self, request_id):
        self.finished.append(request_id)

    def status(self):
        return {"enabled": True, "ready_requests": 1}

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


def test_readiness_refresh_coalesces_concurrent_dht_snapshots():
    manager, runtime = manager_and_runtime()
    calls = 0
    calls_lock = threading.Lock()

    def slow_capacity(model_swarm_id, upper_bound):
        nonlocal calls
        assert model_swarm_id == MODEL_SWARM_ID
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return min(4096, upper_bound)

    runtime.max_supported_context_tokens = slow_capacity
    results: list[int] = []
    threads = [
        threading.Thread(target=lambda: results.append(manager.max_supported_context_tokens()))
        for _ in range(4)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [4096, 4096, 4096, 4096]
    assert calls == 1
    manager.close()


def test_local_request_agent_reports_context_rejected_before_route_planning():
    manager, runtime = manager_and_runtime()
    app = create_request_agent_app(manager, api_credential=API_CREDENTIAL)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={
                "model": "fabi-swarm",
                "messages": [{"role": "user", "content": "large"}],
                "max_completion_tokens": 4096,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert runtime.requests == []
    assert runtime.unmet_context_requests == [
        (response.headers["x-request-id"], MODEL_SWARM_ID, 4099)
    ]


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
    checkpoints = FakeRecoveryCheckpoints()
    manager = RequestAgentOpenAIManager(
        runtime,
        MODEL_SWARM_ID,
        tokenizer=FakeTokenizer(),
        model_context_limit=8192,
        completion_service_type=object,
        recovery_journal=journal,
        recovery_checkpoints=checkpoints,
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
    assert checkpoints.observed == [(1, 1), (2, 2)]
    assert checkpoints.restored == [(3, 2, 2)]
    assert checkpoints.finished == [request_id]
    assert checkpoints.closed is True
    phase_events, gap = manager.request_phases.read_after(0)
    assert gap is False
    phases = [event.phase for event in phase_events if event.request_id == request_id]
    assert phases == [
        "planning",
        "prefilling",
        "decoding",
        "recovering",
        "replaying",
        "decoding",
        "completed",
    ]
    journal.close()


def test_speculative_target_tokens_are_durable_before_frontend_publication(tmp_path):
    runtime = FailoverRuntime(FailoverStub(), FailoverStub())
    runtime.speculative_fence = SpeculativeWindowFence()
    journal = SqliteRecoveryJournal(tmp_path / "recovery.sqlite3")
    manager = RequestAgentOpenAIManager(
        runtime,
        MODEL_SWARM_ID,
        tokenizer=FakeTokenizer(),
        model_context_limit=8192,
        completion_service_type=object,
        recovery_journal=journal,
    )
    request = RequestContract(
        request_id="speculative-request",
        model_swarm_id=MODEL_SWARM_ID,
        prompt_tokens=3,
        reserved_output_tokens=8,
        recovery_level=RecoveryLevel.RESTARTABLE,
    )
    reservation = runtime.reserve(request)
    plan = reservation.committed.plan
    started = manager.begin_generation_journal_before_prefill(
        request.request_id,
        prompt_token_ids=(10, 20, 30),
        request_data={"stream": True, "temperature": 0},
    )
    manager.commit_generation_prefill(request.request_id, epoch=plan.epoch)
    window = SpeculativeVerifyWindow(
        request_id=request.request_id,
        route_id=plan.route_id,
        epoch=plan.epoch,
        route_plan_digest=reservation.committed.route_plan_digest,
        window_id=1,
        base_committed_position=0,
        input_position=len(started.spec.prompt_token_ids),
        starts_epoch=True,
        input_tokens=(30, 40),
        proposal_tokens=(40,),
        strategy=SpeculativeStrategy.NGRAM_SUFFIX,
        proposer_id="mesh-longest-suffix",
        proposer_version="0.75.1",
        sampling=SpeculativeSampling(seed=0, temperature=0),
        reserved_context_tokens=plan.required_context_tokens,
    )
    runtime.speculative_fence.admit(window)
    response = SpeculativeVerifyResponse(
        request_id=request.request_id,
        route_id=plan.route_id,
        epoch=plan.epoch,
        route_plan_digest=reservation.committed.route_plan_digest,
        window_id=window.window_id,
        input_position=window.input_position,
        input_token_count=len(window.input_tokens),
        verified_position=window.input_position + len(window.input_tokens),
        predicted_tokens=(40, 41),
    )

    unpublished = manager.settle_speculative_response(response, max_commit_tokens=8)
    durable = journal.get(request.request_id)

    assert unpublished.committed_tokens == (40, 41)
    assert durable is not None
    assert durable.committed_output_token_ids == unpublished.committed_tokens
    assert runtime.speculative_fence.pending_count(request.request_id) == 0
    manager.close()
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


def test_local_request_agent_replans_again_when_replacement_dies_during_replay(tmp_path):
    primary = FailoverStub(
        chat_chunks=(
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}],'
            b'"prompt_token_ids":[10,20,30]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
            b'"token_ids":[40],"finish_reason":null}]}\n\n',
        )
    )
    first_replacement = FailingReplayStub(
        replay_chunks=(
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}],'
            b'"prompt_token_ids":[10,20,30]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
            b'"token_ids":[40],"finish_reason":null}]}\n\n',
        )
    )
    second_replacement = FailoverStub(
        replay_chunks=(
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}],'
            b'"prompt_token_ids":[10,20,30]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
            b'"token_ids":[40],"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":" continued"},'
            b'"token_ids":[41],"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n",
        )
    )
    runtime = RepeatedFailoverRuntime(
        primary,
        first_replacement,
        second_replacement,
    )
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
    assert runtime.replans == [(request_id, 1), (request_id, 2)]
    assert first_replacement.replay_requests[0]["committed_output_token_ids"] == [40]
    assert second_replacement.replay_requests[0]["committed_output_token_ids"] == [40]
    assert snapshot is not None
    assert snapshot.state == RecoveryState.COMPLETED
    assert snapshot.epoch == 3
    assert snapshot.route_ids == (
        f"route-primary-{request_id}",
        f"route-replacement-2-{request_id}",
        f"route-replacement-3-{request_id}",
    )
    assert snapshot.committed_output_token_ids == (40, 41)
    phase_events, gap = manager.request_phases.read_after(0)
    assert gap is False
    phases = [event.phase for event in phase_events if event.request_id == request_id]
    assert phases == [
        "planning",
        "prefilling",
        "decoding",
        "recovering",
        "replaying",
        "decoding",
        "recovering",
        "replaying",
        "decoding",
        "completed",
    ]
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
    assert _encode_status_sse("request-phase", 7, {"phase": "recovering"}) == (
        b'id: 7\nevent: request-phase\ndata: {"phase":"recovering"}\n\n'
    )
    with pytest.raises(Exception, match="loopback"):
        _loopback_host("0.0.0.0")


def test_request_agent_ready_file_uses_actual_bound_port_and_owner_only_mode(tmp_path):
    listener = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 43127))
    server = SimpleNamespace(
        servers=[SimpleNamespace(sockets=[listener])],
    )
    ready_file = tmp_path / "request-agent.json"

    base_url = _bound_base_url(server, "127.0.0.1")
    _write_ready_file(ready_file, base_url=base_url)

    assert json.loads(ready_file.read_text()) == {
        "schema_version": 1,
        "pid": os.getpid(),
        "base_url": "http://127.0.0.1:43127",
    }
    if os.name != "nt":
        assert ready_file.stat().st_mode & 0o077 == 0


def test_request_agent_bound_url_brackets_ipv6_and_rejects_ambiguous_listeners():
    server = SimpleNamespace(
        servers=[
            SimpleNamespace(sockets=[SimpleNamespace(getsockname=lambda: ("::1", 43128, 0, 0))])
        ],
    )
    assert _bound_base_url(server, "::1") == "http://[::1]:43128"

    server.servers[0].sockets.append(SimpleNamespace(getsockname=lambda: ("::1", 43129, 0, 0)))
    with pytest.raises(RuntimeError, match="exactly one listener"):
        _bound_base_url(server, "::1")


def test_request_agent_server_publishes_bound_port_only_while_serving(tmp_path):
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ready"}

    ready_file = tmp_path / "request-agent.json"
    server = _ReadyFileServer(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"),
        ready_file=ready_file,
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    try:
        while not ready_file.exists() and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_file.exists()
        base_url = json.loads(ready_file.read_text())["base_url"]
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            assert json.load(response) == {"status": "ready"}
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert not ready_file.exists()


def test_local_request_agent_rejects_missing_account_credential():
    manager, _runtime = manager_and_runtime()
    app = create_request_agent_app(manager, api_credential=API_CREDENTIAL)

    with TestClient(app) as client:
        unauthorized_events = client.get("/v1/request-agent/events")
        invalid_cursor = client.get(
            "/v1/request-agent/events",
            headers={**AUTH_HEADERS, "Last-Event-ID": "not-an-integer"},
        )
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert unauthorized_events.status_code == 401
    assert invalid_cursor.status_code == 400
    assert invalid_cursor.json()["error"]["code"] == "invalid_last_event_id"
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
