from __future__ import annotations

from types import SimpleNamespace

import pytest

import parallax.p2p.server as p2p_server
import parallax.server.vllm_rust_frontend as rust_frontend
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
    def raise_for_status(self):
        return None


class FakeHttpClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.posts = []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, *, json):
        self.posts.append((url, json))
        return FakeHttpResponse()


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
