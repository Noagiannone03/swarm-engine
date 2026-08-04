from __future__ import annotations

from types import SimpleNamespace

import pytest

import parallax.p2p.server as p2p_server
import parallax.server.vllm_rust_frontend as rust_frontend
from backend.server.openai_compat import decode_http_response_envelope
from parallax.p2p.server import TransformerConnectionHandler
from parallax.server.vllm_rust_frontend import launch_vllm_rust_frontend


class FakeIrohTransport:
    def peer_id(self):
        return "worker-endpoint"


class FakeAdmission:
    def __init__(self):
        self.calls = []

    def authorize_frontend(self, **kwargs):
        self.calls.append(kwargs)


class FakeHttpResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "count": 3,
            "max_model_len": 4096,
            "tokens": [10, 20, 30],
        }


class FakeStreamResponse(FakeHttpResponse):
    headers = {"content-type": "text/event-stream"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def iter_bytes(self):
        yield b'data: {"choices":[{"token_ids":[42]}]}\n\n'
        yield b"data: [DONE]\n\n"


class FakeRejectedStreamResponse(FakeStreamResponse):
    status_code = 400
    headers = {"content-type": "application/json; charset=utf-8"}
    body = (
        b'{"error":{"message":"maximum context length is 16384 tokens",'
        b'"type":"invalid_request_error","param":"messages",'
        b'"code":"context_length_exceeded"}}'
    )

    def read(self):
        return self.body

    def iter_bytes(self):
        pytest.fail("a rejected local HTTP stream must not leak raw JSON into RPC streaming")


class FakeHttpClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.posts = []
        self.streams = []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, *, json):
        self.posts.append((url, json))
        return FakeHttpResponse()

    def stream(self, method, url, *, json):
        self.streams.append((method, url, json))
        return FakeStreamResponse()


class FakeRejectedHttpClient(FakeHttpClient):
    def stream(self, method, url, *, json):
        self.streams.append((method, url, json))
        return FakeRejectedStreamResponse()


def make_handler(admission=None):
    return TransformerConnectionHandler(
        lattica=None,
        recv_from_peer_addr="inproc://recv",
        send_to_peer_addr="inproc://send",
        block_start_index=0,
        block_end_index=1,
        http_port=3000,
        iroh_transport=FakeIrohTransport(),
        execution_admission=admission,
    )


def test_abort_completion_is_route_fenced_and_calls_official_vllm_api(monkeypatch):
    admission = FakeAdmission()
    handler = make_handler(admission)
    FakeHttpClient.instances.clear()
    monkeypatch.setattr(p2p_server, "authenticated_rpc_peer_id", lambda: "coordinator")
    monkeypatch.setattr(p2p_server.httpx, "Client", FakeHttpClient)

    result = handler.abort_completion(
        {
            "request_id": "scheduler-request",
            "vllm_xargs": {
                "fabi_route_id": "route-7",
                "fabi_route_epoch": 7,
                "parallax_routing_table": ["worker-a", "worker-b"],
            },
        }
    )

    assert result == {"aborted": True, "request_id": "scheduler-request"}
    assert admission.calls == [
        {
            "request_id": "scheduler-request",
            "route_id": "route-7",
            "epoch": 7,
            "routing_table": ("worker-a", "worker-b"),
            "caller_endpoint_id": "coordinator",
        }
    ]
    assert FakeHttpClient.instances[0].posts == [
        (
            "http://localhost:3000/abort_requests",
            {"request_ids": ["chatcmpl-scheduler-request"]},
        )
    ]


def test_abort_completion_fails_closed_without_active_v3():
    handler = make_handler()

    with pytest.raises(PermissionError, match="active protocol v3"):
        handler.abort_completion({"request_id": "request", "vllm_xargs": {}})


def test_chat_tokenization_is_route_fenced_and_uses_official_frontend(monkeypatch):
    admission = FakeAdmission()
    handler = make_handler(admission)
    FakeHttpClient.instances.clear()
    monkeypatch.setattr(p2p_server, "authenticated_rpc_peer_id", lambda: "coordinator")
    monkeypatch.setattr(p2p_server.httpx, "Client", FakeHttpClient)

    result = handler.tokenize_chat(
        {
            "request_id": "scheduler-request",
            "vllm_xargs": {
                "fabi_route_id": "route-7",
                "fabi_route_epoch": 7,
                "parallax_routing_table": ["worker-a", "worker-b"],
            },
            "request": {
                "model": "Qwen/Qwen3-4B",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [],
                "chat_template_kwargs": {"enable_thinking": False},
                "max_completion_tokens": 128,
                "stream": True,
            },
        }
    )

    assert result == {
        "ok": True,
        "request_id": "scheduler-request",
        "tokens": [10, 20, 30],
        "count": 3,
        "max_model_len": 4096,
    }
    assert admission.calls == [
        {
            "request_id": "scheduler-request",
            "route_id": "route-7",
            "epoch": 7,
            "routing_table": ("worker-a", "worker-b"),
            "caller_endpoint_id": "coordinator",
        }
    ]
    assert FakeHttpClient.instances[0].posts == [
        (
            "http://localhost:3000/tokenize",
            {
                "model": "Qwen/Qwen3-4B",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [],
                "chat_template_kwargs": {"enable_thinking": False},
                "return_token_strs": False,
            },
        )
    ]


def test_chat_tokenization_rejects_unsupported_forced_tool_choice(monkeypatch):
    handler = make_handler(FakeAdmission())
    monkeypatch.setattr(p2p_server, "authenticated_rpc_peer_id", lambda: "coordinator")

    result = handler.tokenize_chat(
        {
            "request_id": "request",
            "vllm_xargs": {
                "fabi_route_id": "route",
                "fabi_route_epoch": 1,
                "parallax_routing_table": ["worker"],
            },
            "request": {
                "messages": [{"role": "user", "content": "hello"}],
                "tool_choice": "required",
            },
        }
    )

    assert result["ok"] is False
    assert result["status_code"] == 400
    assert "tool_choice" in result["error"]


def test_generation_replay_is_route_fenced_and_uses_qualified_chat_api(monkeypatch):
    admission = FakeAdmission()
    handler = make_handler(admission)
    FakeHttpClient.instances.clear()
    monkeypatch.setattr(p2p_server, "authenticated_rpc_peer_id", lambda: "coordinator")
    monkeypatch.setattr(p2p_server.httpx, "Client", FakeHttpClient)
    request = {
        "authority_request_id": "scheduler-request",
        "request": {
            "request_id": "scheduler-request",
            "model": "Qwen/Qwen3-4B",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "max_completion_tokens": 16,
            "temperature": 0,
            "vllm_xargs": {
                "fabi_route_id": "route-8",
                "fabi_route_epoch": 8,
                "parallax_routing_table": ["backup-a", "backup-b"],
            },
        },
        "original_prompt_token_ids": [10, 20, 30],
        "committed_output_token_ids": [40],
    }

    chunks = list(handler.replay_generation(request))

    assert chunks == [
        b'data: {"choices":[{"token_ids":[42]}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert admission.calls == [
        {
            "request_id": "scheduler-request",
            "route_id": "route-8",
            "epoch": 8,
            "routing_table": ("backup-a", "backup-b"),
            "caller_endpoint_id": "coordinator",
        }
    ]
    assert FakeHttpClient.instances[0].streams == [
        (
            "POST",
            "http://localhost:3000/inference/v1/chat-replay",
            {key: value for key, value in request.items() if key != "authority_request_id"},
        )
    ]


def test_generation_replay_rejects_unbounded_or_non_streaming_input(monkeypatch):
    handler = make_handler(FakeAdmission())
    monkeypatch.setattr(p2p_server, "authenticated_rpc_peer_id", lambda: "coordinator")

    chunks = list(
        handler.replay_generation(
            {
                "authority_request_id": "request",
                "request": {
                    "request_id": "request",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "max_completion_tokens": 1,
                    "vllm_xargs": {
                        "fabi_route_id": "route",
                        "fabi_route_epoch": 1,
                        "parallax_routing_table": ["worker"],
                    },
                },
                "original_prompt_token_ids": [1],
                "committed_output_token_ids": [],
            }
        )
    )

    assert len(chunks) == 1
    envelope = decode_http_response_envelope(chunks[0])
    assert envelope is not None
    status_code, _, body = envelope
    assert status_code == 502
    assert b"generation_replay_failed" in body


def test_streaming_chat_preserves_local_http_error_in_transport_envelope(monkeypatch):
    handler = make_handler()
    FakeHttpClient.instances.clear()
    monkeypatch.setattr(p2p_server.httpx, "Client", FakeRejectedHttpClient)

    chunks = list(
        handler.chat_completion(
            {
                "messages": [{"role": "user", "content": "too long"}],
                "stream": True,
            }
        )
    )

    assert len(chunks) == 1
    envelope = decode_http_response_envelope(chunks[0])
    assert envelope is not None
    status_code, content_type, body = envelope
    assert status_code == 400
    assert content_type == "application/json; charset=utf-8"
    assert b"context_length_exceeded" in body


def test_active_v3_frontend_enables_local_vllm_abort_route(monkeypatch):
    listener = SimpleNamespace(
        fileno=lambda: 12,
        getsockname=lambda: ("::1", 3000),
        close=lambda: None,
    )
    process = SimpleNamespace(poll=lambda: None)
    captured = {}

    monkeypatch.setenv("FABI_SWARM_V3_MODE", "active")
    monkeypatch.setattr(rust_frontend, "resolve_vllm_rs_binary", lambda: "/tmp/vllm-rs")
    monkeypatch.setattr(rust_frontend, "_bind_listener_socket", lambda host, port: listener)
    monkeypatch.setattr(rust_frontend.time, "sleep", lambda _: None)

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return process

    monkeypatch.setattr(rust_frontend.subprocess, "Popen", fake_popen)
    args = SimpleNamespace(
        host="localhost",
        port=3000,
        executor_input_ipc="ipc:///tmp/input",
        executor_output_ipc="ipc:///tmp/output",
        model_path="model",
        served_model_name=None,
        max_sequence_length=4096,
    )

    launched = launch_vllm_rust_frontend(args)

    assert launched.process is process
    assert captured["env"]["VLLM_SERVER_DEV_MODE"] == "1"


def test_active_v3_frontend_refuses_non_loopback_admin_routes(monkeypatch):
    monkeypatch.setenv("FABI_SWARM_V3_MODE", "active")
    monkeypatch.setattr(rust_frontend, "resolve_vllm_rs_binary", lambda: "/tmp/vllm-rs")
    monkeypatch.setattr(
        rust_frontend,
        "_bind_listener_socket",
        lambda host, port: pytest.fail("must reject before binding"),
    )
    args = SimpleNamespace(
        host="0.0.0.0",
        port=3000,
        executor_input_ipc="ipc:///tmp/input",
        executor_output_ipc="ipc:///tmp/output",
        model_path="model",
        served_model_name=None,
        max_sequence_length=4096,
    )

    with pytest.raises(RuntimeError, match="loopback-only"):
        launch_vllm_rust_frontend(args)
