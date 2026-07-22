from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.server import scheduler_manage as scheduler_manage_module
from backend.server.scheduler_manage import SchedulerManage
from backend.server.server_args import parse_args


def test_backend_scheduler_defaults_to_dynamic_dp(monkeypatch):
    monkeypatch.setattr("sys.argv", ["backend"])

    args = parse_args()

    assert args.allocation_strategy == "dp"
    assert args.routing_strategy == "dp"


def test_backend_scheduler_rejects_unknown_strategy(monkeypatch):
    monkeypatch.setattr("sys.argv", ["backend", "--routing-strategy", "unknown"])

    with pytest.raises(SystemExit):
        parse_args()


def test_scheduler_manager_forwards_dynamic_dp_configuration():
    model_info = object()
    manager = SchedulerManage(allocation_strategy="dp", routing_strategy="dp")

    with (
        patch.object(scheduler_manage_module, "get_model_info", return_value=model_info),
        patch.object(scheduler_manage_module, "Scheduler") as scheduler_class,
        patch.object(scheduler_manage_module.threading, "Thread") as thread_class,
    ):
        manager._start_scheduler("Qwen/Qwen3-1.7B", 2)

    scheduler_class.assert_called_once_with(
        model_info,
        [],
        min_nodes_bootstrapping=2,
        enable_weight_refit=False,
        weight_refit_mode="disk",
        strategy="dp",
        routing_strategy="dp",
        planning_context_tokens=16_384,
        preferred_context_tokens=32_768,
        require_exact_weight_metadata=True,
    )
    thread_class.return_value.start.assert_called_once_with()


def test_cluster_node_info_exposes_frontend_capability():
    manager = SchedulerManage()
    node = SimpleNamespace(
        node_id="windows-worker",
        is_active=False,
        supports_frontend=False,
        hardware=SimpleNamespace(
            num_gpus=1,
            gpu_name="RTX 4080 SUPER",
            memory_gb=16.0,
        ),
    )

    assert manager.build_node_info(node)["supports_frontend"] is False


def test_cluster_node_info_exposes_measured_kv_capacity():
    manager = SchedulerManage()
    node = SimpleNamespace(
        node_id="measured-worker",
        is_active=True,
        supports_frontend=True,
        max_sequence_length=32768,
        max_requests=4,
        kv_cache_token_capacity=65536,
        kv_cache_block_size=64,
        reserved_context_tokens=10048,
        remaining_context_tokens=55488,
        direct_peer_ids={"next-worker"},
        rtt_to_nodes={"next-worker": 12.5},
        hardware=SimpleNamespace(num_gpus=1, gpu_name="RTX", memory_gb=16.0),
    )

    info = manager.build_node_info(node)

    assert info["kv_cache_telemetry_ready"] is True
    assert info["kv_cache_token_capacity"] == 65536
    assert info["kv_cache_block_size"] == 64
    assert info["reserved_context_tokens"] == 10048
    assert info["remaining_context_tokens"] == 55488
    assert info["direct_link_telemetry_ready"] is True
    assert info["direct_peer_ids"] == ["next-worker"]
    assert info["rtt_to_nodes_ms"] == {"next-worker": 12.5}


def test_scheduler_starts_and_reuses_iroh_rpc_handler(monkeypatch):
    registered = []
    transport = SimpleNamespace(
        register=registered.append,
        peer_id=lambda: "scheduler-endpoint",
    )
    manager = SchedulerManage(http_port=3001)
    first_scheduler = object()
    manager.scheduler = first_scheduler
    monkeypatch.setattr(
        scheduler_manage_module.IrohTransport,
        "from_environment",
        lambda role: transport,
    )

    manager._start_iroh()

    assert manager.get_peer_id() == "scheduler-endpoint"
    assert len(registered) == 1
    assert registered[0].scheduler is first_scheduler
    assert registered[0].http_port == 3001

    second_scheduler = object()
    manager.scheduler = second_scheduler
    manager._start_iroh()

    assert len(registered) == 1
    assert registered[0].scheduler is second_scheduler


def test_context_tokenizer_is_canonical_cached_and_offline_aware():
    manager = SchedulerManage(use_hfcache=True)
    manager.model_name = "Qwen/Qwen3-0.6B"
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: {"input_ids": [1, 2, 3]}
    )

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as loader:
        first = manager.build_context_budget({"messages": [{"role": "user", "content": "hello"}]})
        second = manager.build_context_budget({"messages": [{"role": "user", "content": "again"}]})

    assert first.prompt_tokens == second.prompt_tokens == 3
    loader.assert_called_once_with(
        "Qwen/Qwen3-0.6B",
        trust_remote_code=True,
        local_files_only=True,
    )
