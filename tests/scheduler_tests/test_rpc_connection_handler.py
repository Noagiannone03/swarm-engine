"""Regression tests for scheduler RPC node join/update behavior."""

from __future__ import annotations

import importlib

from backend.server.rpc_connection_handler import RPCConnectionHandler
from scheduling.scheduler import Scheduler

from .test_utils import build_model_info


def _handler(scheduler: Scheduler) -> RPCConnectionHandler:
    handler = RPCConnectionHandler.__new__(RPCConnectionHandler)
    handler.scheduler = scheduler
    handler.http_port = 0
    return handler


def _node_message(node_id: str, *, account_token: str | None = None) -> dict:
    message = {
        "node_id": node_id,
        "hardware": {
            "node_id": node_id,
            "num_gpus": 1,
            "tflops_fp16": 7.1,
            "gpu_name": "Apple M3",
            "memory_gb": 4.0,
            "memory_bandwidth_gbps": 100.0,
            "device": "mlx",
        },
        "kvcache_mem_ratio": 0.25,
        "param_mem_ratio": 0.65,
        "max_concurrent_requests": 1,
        "max_sequence_length": 16384,
        "rtt_to_nodes": {},
        "status": "joining",
        "is_active": False,
        "last_refit_time": 0.0,
    }
    if account_token is not None:
        message["account_token"] = account_token
    return message


def _reset_gate(monkeypatch):
    monkeypatch.setenv("FABI_GATE", "on")
    monkeypatch.delenv("FABI_GATE_REDIS_URL", raising=False)
    monkeypatch.delenv("FABI_GATE_ALLOWLIST", raising=False)
    mod = importlib.import_module("backend.server.contribution_gate")
    mod._gate_singleton = None
    return mod


def test_node_join_returns_standby_ack_when_no_layers_are_assigned(monkeypatch):
    model = build_model_info(28)
    scheduler = Scheduler(model, [], strategy="dp", routing_strategy="dp", min_nodes_bootstrapping=1)
    handler = _handler(scheduler)
    monkeypatch.setattr(handler, "wait_layer_allocation", lambda *_args, **_kwargs: {})

    response = handler.node_join(_node_message("mac-standby"))

    assert response["node_id"] == "mac-standby"
    assert response["standby"] is True
    assert response["status"] == "standby"
    assert response["model_name"] == model.mlx_model_name
    assert "start_layer" not in response
    assert "end_layer" not in response


def test_node_join_refreshes_contribution_gate_for_standby_node(monkeypatch):
    gate_mod = _reset_gate(monkeypatch)
    model = build_model_info(28)
    scheduler = Scheduler(model, [], strategy="dp", routing_strategy="dp", min_nodes_bootstrapping=1)
    handler = _handler(scheduler)
    monkeypatch.setattr(handler, "wait_layer_allocation", lambda *_args, **_kwargs: {})

    token = "local-account-token"
    response = handler.node_join(_node_message("mac-standby", account_token=token))

    assert response["standby"] is True
    assert gate_mod.get_gate().is_allowed(token) is True
    assert gate_mod.get_gate().is_allowed("other-token") is False
