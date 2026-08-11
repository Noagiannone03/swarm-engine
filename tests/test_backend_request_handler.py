import asyncio
import json
import threading
import time
from types import SimpleNamespace

import anyio
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.server.constants import NODE_STATUS_AVAILABLE, NODE_STATUS_WAITING
from backend.server.context_admission import ContextBudget
from backend.server.openai_compat import encode_http_response_envelope
from backend.server.request_handler import RequestHandler
from swarm_protocol.contracts import RecoveryLevel
from swarm_protocol.recovery import RecoveryConflict, RecoveryState


class DummySchedulerManage:
    scheduler = None

    def get_model_name(self):
        return "Qwen/Qwen3-0.6B"


class AuthoritySchedulerManage(DummySchedulerManage):
    def get_route_authority(self, request_id):
        assert request_id == "scheduler-req"
        return {"route_id": "route-7", "epoch": 7}


class ForwardingSchedulerManage(DummySchedulerManage):
    def __init__(
        self,
        status=NODE_STATUS_AVAILABLE,
        routing_table=None,
        context_budget=ContextBudget(prompt_tokens=5, max_output_tokens=128),
        max_context=4096,
    ):
        self.status = status
        self.routing_table = ["node-a"] if routing_table is None else routing_table
        self.released = []
        self.capacity_waits = 0
        self.context_budget = context_budget
        self.max_context = max_context
        self.routing_requests = []
        self.unmet_context_requests = []
        self.active_routes = set()

    def get_schedule_status(self):
        return self.status

    def build_context_budget(self, request_data):
        return self.context_budget

    def max_supported_context_tokens(self):
        return self.max_context

    def observe_unmet_context_demand(self, request_id, required_context_tokens):
        self.unmet_context_requests.append((request_id, required_context_tokens))
        return True

    def get_routing_table(
        self,
        request_id,
        received_ts,
        required_context_tokens=0,
        *,
        prompt_tokens=None,
        reserved_output_tokens=None,
        recovery_level=RecoveryLevel.RESTARTABLE,
    ):
        self.routing_requests.append(
            (
                request_id,
                required_context_tokens,
                prompt_tokens,
                reserved_output_tokens,
                recovery_level,
            )
        )
        if self.routing_table:
            self.active_routes.add(str(request_id))
        return self.routing_table

    def release_routing_table(self, request_id):
        self.released.append(request_id)
        self.active_routes.discard(str(request_id))
        return True

    def is_routing_table_active(self, request_id):
        return str(request_id) in self.active_routes

    def wait_for_routing_capacity(self, timeout):
        self.capacity_waits += 1
        self.routing_table = ["node-a"]
        return True


class V3ForwardingSchedulerManage(ForwardingSchedulerManage):
    def get_route_authority(self, request_id):
        return {"route_id": f"route-{request_id}", "epoch": 7}


class ExactTokenForwardingSchedulerManage(V3ForwardingSchedulerManage):
    def requires_exact_frontend_tokenization(self):
        return True


class RecoveryForwardingSchedulerManage(V3ForwardingSchedulerManage):
    def __init__(self):
        super().__init__(
            context_budget=ContextBudget(
                prompt_tokens=3,
                max_output_tokens=16,
                prompt_token_ids=(10, 20, 30),
            )
        )
        self.journal_calls = []
        self.committed_output_token_ids = []
        self.promotions = 0

    def preferred_recovery_level(self, request_data):
        assert request_data["temperature"] == 0
        return RecoveryLevel.RECOVERABLE

    def should_capture_generation_tokens(self, request_id, request_data):
        assert request_id == "recoverable-stream"
        return True

    def begin_generation_journal(self, request_id, **kwargs):
        self.journal_calls.append(("begin", request_id, kwargs))
        return SimpleNamespace(epoch=7)

    def commit_generation_prefill(self, request_id, *, epoch):
        self.journal_calls.append(("prefill", request_id, epoch))

    def commit_generation_tokens(self, request_id, *, epoch, token_ids):
        self.journal_calls.append(("tokens", request_id, epoch, token_ids))
        self.committed_output_token_ids.extend(token_ids)

    def finish_generation_journal(self, request_id, *, epoch, state, failure=None):
        self.journal_calls.append(("finish", request_id, epoch, state, failure))

    def promote_generation_recovery(self, request_id, *, failed_epoch):
        self.journal_calls.append(("promote", request_id, failed_epoch))
        self.promotions += 1
        if self.promotions > 1:
            raise RecoveryConflict("request has no reserved recovery route")
        recovering = SimpleNamespace(
            epoch=8,
            spec=SimpleNamespace(prompt_token_ids=(10, 20, 30)),
            committed_output_token_ids=tuple(self.committed_output_token_ids),
        )
        promoted = SimpleNamespace(
            primary_plan=SimpleNamespace(
                route_id="route-recovery-8",
                stages=(SimpleNamespace(worker_id="node-b"),),
            )
        )
        return recovering, promoted

    def build_generation_replay_request(
        self,
        request_id,
        *,
        original_request,
        model_name,
    ):
        self.journal_calls.append(
            ("build-replay", request_id, tuple(self.committed_output_token_ids))
        )
        replay_chat_request = dict(original_request)
        replay_chat_request["model"] = model_name
        replay_chat_request["request_id"] = request_id
        replay_chat_request["return_token_ids"] = True
        replay_chat_request["include_reasoning"] = True
        replay_chat_request["vllm_xargs"] = {
            "parallax_routing_table": ["node-b"],
            "parallax_scheduler_request_id": request_id,
            "fabi_route_id": "route-recovery-8",
            "fabi_route_epoch": 8,
        }
        return "node-b", {
            "authority_request_id": request_id,
            "request": replay_chat_request,
            "original_prompt_token_ids": [10, 20, 30],
            "committed_output_token_ids": list(self.committed_output_token_ids),
        }

    def complete_generation_replay(self, request_id, *, epoch):
        self.journal_calls.append(("replay-complete", request_id, epoch))


class ImmediateResult:
    def result(self, timeout=None):
        return {"ok": True, "timeout": timeout}


class StaticStub:
    def __init__(self, chunks):
        self.chunks = chunks
        self.request = None
        self.response = None
        self.abort_requests = []
        self.replay_request = None

    def chat_completion(self, request):
        self.request = request
        self.response = CancellableResponse(self.chunks)
        return self.response

    def abort_completion(self, request):
        self.abort_requests.append(request)
        return ImmediateResult()

    def replay_generation(self, request):
        self.replay_request = request
        self.response = CancellableResponse(self.chunks)
        return self.response


class ValueResult:
    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        del timeout
        return self.value


class TokenizingStub(StaticStub):
    def __init__(self, chunks, token_ids):
        super().__init__(chunks)
        self.token_ids = list(token_ids)
        self.tokenize_requests = []

    def tokenize_chat(self, request):
        self.tokenize_requests.append(request)
        return ValueResult(
            {
                "ok": True,
                "request_id": request["request_id"],
                "tokens": self.token_ids,
                "count": len(self.token_ids),
                "max_model_len": 4096,
            }
        )


class CancellableResponse:
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.cancelled = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def cancel(self):
        self.cancelled = True


class FailingResponse(CancellableResponse):
    def __init__(self, error):
        super().__init__(())
        self.error = error

    def __next__(self):
        raise self.error


class BlockingCancellableResponse:
    def __init__(self):
        self.cancelled = False
        self._cancelled = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        self._cancelled.wait(timeout=5)
        return b'{"choices":[]}'

    def cancel(self):
        self.cancelled = True
        self._cancelled.set()


class BlockingStub:
    def __init__(self):
        self.response = BlockingCancellableResponse()
        self.request = None
        self.abort_requests = []

    def chat_completion(self, request):
        self.request = request
        return self.response

    def abort_completion(self, request):
        self.abort_requests.append(request)
        return ImmediateResult()


def test_prepare_backend_request_uses_vllm_xargs_for_parallax_metadata():
    handler = RequestHandler()
    handler.set_scheduler_manage(DummySchedulerManage())
    request_data = {
        "model": "client-model",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
        ],
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "response_format": {"type": "json_object"},
        "stream_options": {"include_usage": True},
        "max_completion_tokens": 42,
        "extra_future_field": {"kept": True},
        "rid": "old-rid",
        "routing_table": ["old-node"],
        "vllm_xargs": {
            "user_arg": 1,
            "parallax_routing_table": ["user-node"],
            "parallax_scheduler_request_id": "user-req",
        },
    }

    backend_request = handler._prepare_backend_request(
        request_data,
        "scheduler-req",
        ["node-a", "node-b"],
    )

    assert "rid" not in backend_request
    assert "routing_table" not in backend_request
    assert backend_request["request_id"] == "scheduler-req"
    assert backend_request["model"] == "Qwen/Qwen3-0.6B"
    assert backend_request["messages"] == request_data["messages"]
    assert backend_request["tools"] == request_data["tools"]
    assert backend_request["tool_choice"] == request_data["tool_choice"]
    assert backend_request["response_format"] == request_data["response_format"]
    assert backend_request["stream_options"] == request_data["stream_options"]
    assert backend_request["max_completion_tokens"] == 42
    assert backend_request["extra_future_field"] == {"kept": True}
    assert backend_request["vllm_xargs"] == {
        "user_arg": 1,
        "parallax_routing_table": ["node-a", "node-b"],
        "parallax_scheduler_request_id": "scheduler-req",
    }
    assert request_data["routing_table"] == ["old-node"]
    assert request_data["vllm_xargs"]["parallax_routing_table"] == ["user-node"]


def test_prepare_backend_request_propagates_active_v3_route_fence():
    handler = RequestHandler()
    handler.set_scheduler_manage(AuthoritySchedulerManage())

    backend_request = handler._prepare_backend_request(
        {"messages": [{"role": "user", "content": "hello"}]},
        "scheduler-req",
        ["node-a"],
    )

    assert backend_request["vllm_xargs"] == {
        "parallax_routing_table": ["node-a"],
        "parallax_scheduler_request_id": "scheduler-req",
        "fabi_route_id": "route-7",
        "fabi_route_epoch": 7,
    }


def test_prepare_backend_request_replaces_client_selected_request_id():
    handler = RequestHandler()
    handler.set_scheduler_manage(AuthoritySchedulerManage())

    backend_request = handler._prepare_backend_request(
        {
            "request_id": "client-controlled",
            "messages": [{"role": "user", "content": "hello"}],
        },
        "scheduler-req",
        ["node-a"],
    )

    assert backend_request["request_id"] == "scheduler-req"


def test_forward_request_returns_openai_error_when_scheduler_not_ready():
    handler = RequestHandler()
    handler.set_scheduler_manage(ForwardingSchedulerManage(status=NODE_STATUS_WAITING))

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}]},
            "scheduler-req",
            1.0,
        )
    )

    payload = json.loads(response.body)
    assert response.status_code == 503
    assert payload["error"]["message"] == "Server is not ready"
    assert payload["error"]["type"] == "server_unavailable"
    assert payload["error"]["code"] == "server_not_ready"


def test_blocking_route_discovery_does_not_stall_the_asyncio_event_loop():
    class SlowRoutingManager(ForwardingSchedulerManage):
        def get_routing_table(self, *args, **kwargs):
            time.sleep(0.05)
            return super().get_routing_table(*args, **kwargs)

    async def exercise():
        handler = RequestHandler()
        manager = SlowRoutingManager()
        handler.set_scheduler_manage(manager)
        handler.stubs["node-a"] = StaticStub([b'{"choices":[]}'])
        loop_progressed = asyncio.Event()

        async def observe_loop_progress():
            await asyncio.sleep(0.01)
            loop_progressed.set()

        response, _ = await asyncio.gather(
            handler.v1_chat_completions(
                {"messages": [{"role": "user", "content": "hello"}]},
                "slow-discovery-request",
                1.0,
            ),
            observe_loop_progress(),
        )
        return response, loop_progressed.is_set()

    response, loop_progressed = asyncio.run(exercise())

    assert response.status_code == 200
    assert loop_progressed is True


def test_blocking_route_release_does_not_stall_the_asyncio_event_loop():
    class SlowReleaseManager(ForwardingSchedulerManage):
        def __init__(self):
            super().__init__()
            self.release_started = threading.Event()
            self.release_finished = threading.Event()

        def release_routing_table(self, request_id):
            self.release_started.set()
            time.sleep(0.05)
            try:
                return super().release_routing_table(request_id)
            finally:
                self.release_finished.set()

    async def exercise():
        handler = RequestHandler()
        manager = SlowReleaseManager()
        handler.set_scheduler_manage(manager)
        handler.stubs["node-a"] = StaticStub([b'{"choices":[]}'])

        async def observe_release():
            while not manager.release_started.is_set():
                await asyncio.sleep(0)
            await asyncio.sleep(0.01)
            return not manager.release_finished.is_set()

        response, loop_progressed_during_release = await asyncio.gather(
            handler.v1_chat_completions(
                {"messages": [{"role": "user", "content": "hello"}]},
                "slow-release-request",
                1.0,
            ),
            observe_release(),
        )
        return response, loop_progressed_during_release

    response, loop_progressed_during_release = asyncio.run(exercise())

    assert response.status_code == 200
    assert loop_progressed_during_release is True


def test_route_release_is_shielded_from_anyio_level_cancellation():
    handler = RequestHandler()
    scheduler_manage = ForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)

    async def cancel_then_release():
        with anyio.CancelScope() as cancelled_scope:
            cancelled_scope.cancel()
            await handler._release_route("cancelled-scope-request")

    anyio.run(cancel_then_release)

    assert scheduler_manage.released == ["cancelled-scope-request"]


def test_forward_request_returns_openai_error_when_pipelines_are_busy():
    handler = RequestHandler()
    handler.MAX_ROUTING_RETRY = 1
    handler.set_scheduler_manage(ForwardingSchedulerManage(routing_table=[]))

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}]},
            "scheduler-req",
            1.0,
        )
    )

    payload = json.loads(response.body)
    assert response.status_code == 429
    assert payload["error"]["type"] == "rate_limit_error"
    assert payload["error"]["code"] == "rate_limit_exceeded"


def test_forward_request_rejects_context_larger_than_every_pipeline():
    handler = RequestHandler()
    scheduler_manage = ForwardingSchedulerManage(
        context_budget=ContextBudget(prompt_tokens=32768, max_output_tokens=4096),
        max_context=32768,
    )
    handler.set_scheduler_manage(scheduler_manage)

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "large context"}]},
            "oversized-req",
            1.0,
        )
    )

    payload = json.loads(response.body)
    assert response.status_code == 400
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "context_length_exceeded"
    assert "32768 prompt + 4096 maximum output" in payload["error"]["message"]
    assert scheduler_manage.routing_requests == []
    assert scheduler_manage.unmet_context_requests == [("oversized-req", 36864)]


def test_context_rejection_is_unchanged_when_demand_observer_fails():
    class FailingDemandObserver(ForwardingSchedulerManage):
        def observe_unmet_context_demand(self, request_id, required_context_tokens):
            del request_id, required_context_tokens
            raise RuntimeError("DHT unavailable")

    handler = RequestHandler()
    handler.set_scheduler_manage(
        FailingDemandObserver(
            context_budget=ContextBudget(prompt_tokens=32768, max_output_tokens=4096),
            max_context=32768,
        )
    )

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "large context"}]},
            "observer-failure",
            1.0,
        )
    )

    payload = json.loads(response.body)
    assert response.status_code == 400
    assert payload["error"]["code"] == "context_length_exceeded"


def test_forward_request_routes_with_exact_required_context():
    handler = RequestHandler()
    scheduler_manage = ForwardingSchedulerManage(
        context_budget=ContextBudget(prompt_tokens=28672, max_output_tokens=4096),
        max_context=65536,
    )
    handler.set_scheduler_manage(scheduler_manage)
    handler.stubs["node-a"] = StaticStub([b'{"choices":[]}'])

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "large context"}]},
            "accepted-req",
            1.0,
        )
    )

    assert response.status_code == 200
    assert scheduler_manage.routing_requests == [
        (
            "accepted-req",
            32768,
            28672,
            4096,
            RecoveryLevel.RESTARTABLE,
        )
    ]


def test_active_v3_replans_with_qualified_frontend_token_ids():
    handler = RequestHandler()
    scheduler_manage = ExactTokenForwardingSchedulerManage(
        context_budget=ContextBudget(
            prompt_tokens=2,
            max_output_tokens=16,
            prompt_token_ids=(10, 20),
        ),
        max_context=4096,
    )
    handler.set_scheduler_manage(scheduler_manage)
    stub = TokenizingStub([b'{"choices":[]}'], [10, 20, 30])
    handler.stubs["node-a"] = stub

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}]},
            "exact-token-req",
            1.0,
        )
    )

    assert response.status_code == 200
    assert scheduler_manage.routing_requests == [
        (
            "exact-token-req",
            18,
            2,
            16,
            RecoveryLevel.RESTARTABLE,
        ),
        (
            "exact-token-req",
            19,
            3,
            16,
            RecoveryLevel.RESTARTABLE,
        ),
    ]
    assert scheduler_manage.released == ["exact-token-req", "exact-token-req"]
    assert len(stub.tokenize_requests) == 2
    assert stub.request is not None


def test_active_v3_rejects_frontend_tokenizer_disagreement():
    handler = RequestHandler()
    scheduler_manage = ExactTokenForwardingSchedulerManage(
        context_budget=ContextBudget(
            prompt_tokens=2,
            max_output_tokens=16,
            prompt_token_ids=(10, 20),
        ),
        max_context=4096,
    )
    handler.set_scheduler_manage(scheduler_manage)

    class ChangingTokenizingStub(TokenizingStub):
        def tokenize_chat(self, request):
            self.token_ids = [10, 20, 30] if not self.tokenize_requests else [10, 20, 31]
            return super().tokenize_chat(request)

    stub = ChangingTokenizingStub([b'{"choices":[]}'], [])
    handler.stubs["node-a"] = stub

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}]},
            "mismatched-tokenizer-req",
            1.0,
        )
    )

    assert response.status_code == 503
    assert json.loads(response.body)["error"]["code"] == "frontend_tokenizer_mismatch"
    assert stub.request is None


def test_forward_request_wakes_when_capacity_becomes_available():
    handler = RequestHandler()
    handler.MAX_ROUTING_RETRY = 2
    scheduler_manage = ForwardingSchedulerManage(routing_table=[])
    handler.set_scheduler_manage(scheduler_manage)
    handler.stubs["node-a"] = StaticStub([b'{"choices":[]}'])

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}]},
            "waiting-req",
            1.0,
        )
    )

    assert response.status_code == 200
    assert scheduler_manage.capacity_waits == 1


def test_forward_request_preserves_non_stream_downstream_status_and_content_type():
    handler = RequestHandler()
    scheduler_manage = ForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    body = (
        b'{"error":{"message":"bad request","type":"invalid_request_error",'
        b'"param":null,"code":"bad_request"}}'
    )
    handler.stubs["node-a"] = StaticStub(
        [
            encode_http_response_envelope(
                status_code=400,
                content_type="application/json; charset=utf-8",
                body=body,
            )
        ]
    )

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}]},
            "scheduler-req",
            1.0,
        )
    )

    assert response.status_code == 400
    assert response.body == body
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert scheduler_manage.released == ["scheduler-req"]


def test_non_stream_disconnect_cancels_rpc_and_releases_route():
    handler = RequestHandler()
    scheduler_manage = V3ForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    stub = BlockingStub()
    handler.stubs["node-a"] = stub

    async def disconnected():
        return True

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}]},
            "disconnected-req",
            1.0,
            disconnected,
        )
    )

    assert response.status_code == 499
    assert stub.response.cancelled
    assert len(stub.abort_requests) == 1
    assert stub.abort_requests[0]["request_id"] == "disconnected-req"
    assert scheduler_manage.released == ["disconnected-req"]


def test_non_stream_worker_loss_cancels_rpc_and_returns_openai_error():
    handler = RequestHandler()
    scheduler_manage = ForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    stub = BlockingStub()
    handler.stubs["node-a"] = stub

    async def lose_route():
        task = asyncio.create_task(
            handler.v1_chat_completions(
                {"messages": [{"role": "user", "content": "hello"}]},
                "lost-worker-req",
                1.0,
            )
        )
        while "lost-worker-req" not in scheduler_manage.active_routes:
            await asyncio.sleep(0)
        scheduler_manage.active_routes.remove("lost-worker-req")
        return await task

    response = asyncio.run(lose_route())

    payload = json.loads(response.body)
    assert response.status_code == 502
    assert payload["error"]["type"] == "upstream_error"
    assert payload["error"]["code"] == "upstream_worker_lost"
    assert stub.response.cancelled
    assert scheduler_manage.released == ["lost-worker-req"]


def test_streaming_request_releases_route_after_completion():
    handler = RequestHandler()
    scheduler_manage = ForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    stub = StaticStub([b'data: {"choices":[]}\n\n', b"data: [DONE]\n\n"])
    handler.stubs["node-a"] = stub

    async def consume_stream():
        response = await handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            "stream-req",
            1.0,
        )
        return b"".join([chunk async for chunk in response.body_iterator])

    body = asyncio.run(consume_stream())

    assert body.endswith(b"data: [DONE]\n\n")
    assert stub.response.cancelled
    assert stub.abort_requests == []
    assert scheduler_manage.released == ["stream-req"]


def test_streaming_downstream_http_error_is_one_well_framed_sse_error():
    handler = RequestHandler()
    scheduler_manage = V3ForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    body = (
        b'{"error":{"message":"maximum context length is 16384 tokens",'
        b'"type":"invalid_request_error","param":"messages",'
        b'"code":"context_length_exceeded"}}'
    )
    stub = StaticStub(
        [
            encode_http_response_envelope(
                status_code=400,
                content_type="application/json; charset=utf-8",
                body=body,
            )
        ]
    )
    handler.stubs["node-a"] = stub

    async def consume_stream():
        response = await handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "too long"}], "stream": True},
            "rejected-stream-req",
            1.0,
        )
        return b"".join([chunk async for chunk in response.body_iterator])

    wire = asyncio.run(consume_stream())
    events = [line.removeprefix(b"data: ") for line in wire.splitlines() if line]

    assert len(events) == 2
    error = json.loads(events[0])
    assert error["error"] == {
        "message": "maximum context length is 16384 tokens",
        "type": "invalid_request_error",
        "param": "messages",
        "code": "context_length_exceeded",
    }
    assert events[1] == b"[DONE]"
    assert stub.abort_requests == []
    assert scheduler_manage.released == ["rejected-stream-req"]


def test_recoverable_stream_commits_official_tokens_before_sanitized_sse():
    handler = RequestHandler()
    scheduler_manage = RecoveryForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    wire = b"".join(
        [
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}],'
            b'"prompt_token_ids":[10,20,30]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"reasoning":"hidden"},'
            b'"token_ids":[40],"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"ok"},'
            b'"token_ids":[41],"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    handler.stubs["node-a"] = StaticStub([wire[:23], wire[23:91], wire[91:]])

    async def consume_stream():
        response = await handler.v1_chat_completions(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "temperature": 0,
                "include_reasoning": False,
            },
            "recoverable-stream",
            1.0,
        )
        return b"".join([chunk async for chunk in response.body_iterator])

    body = asyncio.run(consume_stream())

    assert handler.stubs["node-a"].request["return_token_ids"] is True
    assert handler.stubs["node-a"].request["include_reasoning"] is True
    assert b"prompt_token_ids" not in body
    assert b"token_ids" not in body
    assert b"hidden" not in body
    assert b'"content":"ok"' in body
    assert body.endswith(b"data: [DONE]\n\n")
    assert scheduler_manage.journal_calls == [
        (
            "begin",
            "recoverable-stream",
            {
                "engine_prompt_token_ids": (10, 20, 30),
                "expected_prompt_token_ids": (10, 20, 30),
                "request_data": {
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                    "temperature": 0,
                    "include_reasoning": False,
                },
            },
        ),
        ("prefill", "recoverable-stream", 7),
        ("tokens", "recoverable-stream", 7, (40,)),
        ("tokens", "recoverable-stream", 7, (41,)),
        ("finish", "recoverable-stream", 7, RecoveryState.COMPLETED, None),
    ]
    assert scheduler_manage.routing_requests[0][-1] == RecoveryLevel.RECOVERABLE
    assert scheduler_manage.released == ["recoverable-stream"]


def test_recoverable_stream_promotes_and_resumes_without_repeating_committed_prefix():
    handler = RequestHandler()
    scheduler_manage = RecoveryForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    primary_wire = b"".join(
        [
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}],'
            b'"prompt_token_ids":[10,20,30]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
            b'"token_ids":[40],"finish_reason":null}]}\n\n',
        ]
    )
    replay_wire = b"".join(
        [
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}],'
            b'"prompt_token_ids":[10,20,30]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
            b'"token_ids":[40],"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":" continued"},'
            b'"token_ids":[41],"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"token_ids":[],"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    primary_stub = StaticStub([primary_wire])
    recovery_stub = StaticStub([replay_wire[:67], replay_wire[67:]])
    handler.stubs["node-a"] = primary_stub
    handler.stubs["node-b"] = recovery_stub

    async def consume_stream():
        response = await handler.v1_chat_completions(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "temperature": 0,
                "include_reasoning": False,
            },
            "recoverable-stream",
            1.0,
        )
        return b"".join([chunk async for chunk in response.body_iterator])

    body = asyncio.run(consume_stream())

    assert body.count(b'"content":"partial"') == 1
    assert body.count(b'"content":" continued"') == 1
    assert b"prompt_token_ids" not in body
    assert b"token_ids" not in body
    assert body.endswith(b"data: [DONE]\n\n")
    assert primary_stub.response.cancelled
    assert recovery_stub.replay_request["authority_request_id"] == "recoverable-stream"
    assert recovery_stub.replay_request["original_prompt_token_ids"] == [10, 20, 30]
    assert recovery_stub.replay_request["committed_output_token_ids"] == [40]
    assert recovery_stub.replay_request["request"]["vllm_xargs"]["fabi_route_epoch"] == 8
    assert scheduler_manage.journal_calls == [
        (
            "begin",
            "recoverable-stream",
            {
                "engine_prompt_token_ids": (10, 20, 30),
                "expected_prompt_token_ids": (10, 20, 30),
                "request_data": {
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                    "temperature": 0,
                    "include_reasoning": False,
                },
            },
        ),
        ("prefill", "recoverable-stream", 7),
        ("tokens", "recoverable-stream", 7, (40,)),
        ("promote", "recoverable-stream", 7),
        ("build-replay", "recoverable-stream", (40,)),
        ("replay-complete", "recoverable-stream", 8),
        ("tokens", "recoverable-stream", 8, (41,)),
        ("tokens", "recoverable-stream", 8, ()),
        ("finish", "recoverable-stream", 8, RecoveryState.COMPLETED, None),
    ]
    assert scheduler_manage.released == ["recoverable-stream"]


def test_second_worker_loss_fails_cleanly_after_reserved_backup_is_consumed():
    handler = RequestHandler()
    scheduler_manage = RecoveryForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    primary_stub = StaticStub(
        [
            b'data: {"choices":[],"prompt_token_ids":[10,20,30]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
            b'"token_ids":[40],"finish_reason":null}]}\n\n'
        ]
    )
    recovery_stub = StaticStub(
        [
            b'data: {"choices":[],"prompt_token_ids":[10,20,30]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
            b'"token_ids":[40],"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":" next"},'
            b'"token_ids":[41],"finish_reason":null}]}\n\n'
        ]
    )
    handler.stubs["node-a"] = primary_stub
    handler.stubs["node-b"] = recovery_stub

    async def consume_stream():
        response = await handler.v1_chat_completions(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "temperature": 0,
            },
            "recoverable-stream",
            1.0,
        )
        return b"".join([chunk async for chunk in response.body_iterator])

    body = asyncio.run(consume_stream())
    payloads = [line.removeprefix(b"data: ") for line in body.splitlines() if line]
    error = json.loads(payloads[-2])

    assert body.count(b'"content":"partial"') == 1
    assert body.count(b'"content":" next"') == 1
    assert error["error"]["code"] == "upstream_worker_lost"
    assert payloads[-1] == b"[DONE]"
    assert scheduler_manage.promotions == 2
    assert scheduler_manage.journal_calls[-1] == (
        "finish",
        "recoverable-stream",
        8,
        RecoveryState.FAILED,
        "upstream ended without [DONE]",
    )
    assert scheduler_manage.released == ["recoverable-stream"]


def test_recoverable_stream_fails_closed_on_prompt_token_mismatch():
    handler = RequestHandler()
    scheduler_manage = RecoveryForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)

    def reject_mismatch(request_id, **kwargs):
        raise RecoveryConflict("engine prompt token IDs differ")

    scheduler_manage.begin_generation_journal = reject_mismatch
    handler.stubs["node-a"] = StaticStub(
        [
            b'data: {"choices":[],"prompt_token_ids":[99]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    async def consume_stream():
        response = await handler.v1_chat_completions(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "temperature": 0,
            },
            "recoverable-stream",
            1.0,
        )
        return b"".join([chunk async for chunk in response.body_iterator])

    body = asyncio.run(consume_stream())
    events = [line.removeprefix(b"data: ") for line in body.splitlines() if line]
    error = json.loads(events[0])

    assert error["error"]["code"] == "recovery_contract_violation"
    assert events[-1] == b"[DONE]"
    assert scheduler_manage.released == ["recoverable-stream"]


def test_incomplete_upstream_stream_emits_error_and_terminal_event():
    handler = RequestHandler()
    scheduler_manage = V3ForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    stub = StaticStub([b'data: {"choices":[]}\n\n'])
    handler.stubs["node-a"] = stub

    async def consume_stream():
        response = await handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            "incomplete-stream-req",
            1.0,
        )
        return b"".join([chunk async for chunk in response.body_iterator])

    body = asyncio.run(consume_stream())

    events = [line.removeprefix(b"data: ") for line in body.splitlines() if line]
    error = json.loads(events[-2])
    assert error["error"]["type"] == "upstream_error"
    assert error["error"]["code"] == "upstream_worker_lost"
    assert events[-1] == b"[DONE]"
    assert stub.response.cancelled
    assert stub.abort_requests == [
        {
            "request_id": "incomplete-stream-req",
            "vllm_xargs": {
                "parallax_routing_table": ["node-a"],
                "parallax_scheduler_request_id": "incomplete-stream-req",
                "fabi_route_id": "route-incomplete-stream-req",
                "fabi_route_epoch": 7,
            },
        }
    ]
    assert scheduler_manage.released == ["incomplete-stream-req"]


def test_streaming_rpc_failure_emits_typed_error_and_terminal_event():
    handler = RequestHandler()
    scheduler_manage = V3ForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    stub = StaticStub([])

    def fail_chat_completion(request):
        stub.request = request
        stub.response = FailingResponse(TimeoutError("transport stalled"))
        return stub.response

    stub.chat_completion = fail_chat_completion
    handler.stubs["node-a"] = stub

    async def consume_stream():
        response = await handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            "failed-stream-req",
            1.0,
        )
        return b"".join([chunk async for chunk in response.body_iterator])

    body = asyncio.run(consume_stream())

    events = [line.removeprefix(b"data: ") for line in body.splitlines() if line]
    error = json.loads(events[-2])
    assert error["error"]["type"] == "upstream_error"
    assert error["error"]["code"] == "upstream_stream_failed"
    assert events[-1] == b"[DONE]"
    assert stub.response.cancelled
    assert scheduler_manage.released == ["failed-stream-req"]


def test_streaming_worker_loss_cancels_rpc_emits_error_and_releases_route():
    handler = RequestHandler()
    scheduler_manage = ForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    stub = BlockingStub()
    handler.stubs["node-a"] = stub

    async def consume_after_route_loss():
        response = await handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            "lost-stream-worker-req",
            1.0,
        )
        consume_task = asyncio.create_task(anext(response.body_iterator))
        while "lost-stream-worker-req" not in scheduler_manage.active_routes:
            await asyncio.sleep(0)
        scheduler_manage.active_routes.remove("lost-stream-worker-req")
        first_chunk = await consume_task
        remaining = b"".join([chunk async for chunk in response.body_iterator])
        return first_chunk + remaining

    body = asyncio.run(consume_after_route_loss())

    events = [line.removeprefix(b"data: ") for line in body.splitlines() if line]
    error = json.loads(events[0])
    assert error["error"]["type"] == "upstream_error"
    assert error["error"]["code"] == "upstream_worker_lost"
    assert events[-1] == b"[DONE]"
    assert stub.response.cancelled
    assert scheduler_manage.released == ["lost-stream-worker-req"]


def test_streaming_client_disconnect_cancels_rpc_without_emitting_error():
    handler = RequestHandler()
    scheduler_manage = V3ForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    stub = BlockingStub()
    handler.stubs["node-a"] = stub

    async def disconnected():
        return True

    async def consume_stream():
        response = await handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            "disconnected-stream-req",
            1.0,
            disconnected,
        )
        return b"".join([chunk async for chunk in response.body_iterator])

    body = asyncio.run(consume_stream())

    assert body == b""
    assert stub.response.cancelled
    assert len(stub.abort_requests) == 1
    assert stub.abort_requests[0]["request_id"] == "disconnected-stream-req"
    assert scheduler_manage.released == ["disconnected-stream-req"]


def test_streaming_body_task_cancellation_aborts_before_releasing_route():
    handler = RequestHandler()
    scheduler_manage = V3ForwardingSchedulerManage()
    handler.set_scheduler_manage(scheduler_manage)
    stub = BlockingStub()
    handler.stubs["node-a"] = stub

    async def cancel_body_task():
        response = await handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            "cancelled-body-req",
            1.0,
        )
        next_chunk = asyncio.create_task(anext(response.body_iterator))
        while "cancelled-body-req" not in scheduler_manage.active_routes or stub.request is None:
            await asyncio.sleep(0)
        next_chunk.cancel()
        try:
            await next_chunk
        except asyncio.CancelledError:
            pass

    asyncio.run(cancel_body_task())

    assert stub.response.cancelled
    assert len(stub.abort_requests) == 1
    assert stub.abort_requests[0]["request_id"] == "cancelled-body-req"
    assert scheduler_manage.released == ["cancelled-body-req"]


def test_openai_models_returns_empty_list_without_scheduler(monkeypatch):
    monkeypatch.setattr(backend_main, "scheduler_manage", None)

    response = TestClient(backend_main.app).get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def test_openai_models_returns_scheduler_model(monkeypatch):
    monkeypatch.setattr(backend_main, "scheduler_manage", DummySchedulerManage())

    response = TestClient(backend_main.app).get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen3-0.6B",
                "object": "model",
                "created": 0,
                "owned_by": "parallax",
            }
        ],
    }


def test_openai_chat_rejects_non_object_body():
    response = TestClient(backend_main.app).post("/v1/chat/completions", json=["not-object"])

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["message"] == "Request body must be a JSON object"
    assert payload["error"]["type"] == "invalid_request_error"
