import asyncio
import json
import threading

from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.server.constants import NODE_STATUS_AVAILABLE, NODE_STATUS_WAITING
from backend.server.context_admission import ContextBudget
from backend.server.openai_compat import encode_http_response_envelope
from backend.server.request_handler import RequestHandler


class DummySchedulerManage:
    scheduler = None

    def get_model_name(self):
        return "Qwen/Qwen3-0.6B"


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
        self.active_routes = set()

    def get_schedule_status(self):
        return self.status

    def build_context_budget(self, request_data):
        return self.context_budget

    def max_supported_context_tokens(self):
        return self.max_context

    def get_routing_table(self, request_id, received_ts, required_context_tokens=0):
        self.routing_requests.append((request_id, required_context_tokens))
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


class StaticStub:
    def __init__(self, chunks):
        self.chunks = chunks
        self.request = None
        self.response = None

    def chat_completion(self, request):
        self.request = request
        self.response = CancellableResponse(self.chunks)
        return self.response


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

    def chat_completion(self, request):
        return self.response


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
    assert scheduler_manage.routing_requests == [("accepted-req", 32768)]


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
    scheduler_manage = ForwardingSchedulerManage()
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
    assert scheduler_manage.released == ["stream-req"]


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
    scheduler_manage = ForwardingSchedulerManage()
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
    assert scheduler_manage.released == ["disconnected-stream-req"]


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
