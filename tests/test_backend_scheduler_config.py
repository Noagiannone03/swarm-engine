import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.server import scheduler_manage as scheduler_manage_module
from backend.server.scheduler_manage import SchedulerManage
from backend.server.server_args import parse_args
from parallax.p2p.liveness import DEFAULT_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS
from swarm_protocol.contracts import RecoveryLevel
from swarm_protocol.recovery import (
    RecoveryState,
    SamplingReplayContract,
    SamplingReplayMode,
)


def test_backend_scheduler_defaults_to_dynamic_dp(monkeypatch):
    monkeypatch.setattr("sys.argv", ["backend"])

    args = parse_args()

    assert args.allocation_strategy == "dp"
    assert args.routing_strategy == "dp"
    assert args.heartbeat_timeout == DEFAULT_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS


def test_backend_scheduler_reads_and_validates_heartbeat_timeout(monkeypatch):
    monkeypatch.setenv("PARALLAX_HEARTBEAT_TIMEOUT", "180")
    monkeypatch.setattr("sys.argv", ["backend"])

    assert parse_args().heartbeat_timeout == 180

    monkeypatch.setenv("PARALLAX_HEARTBEAT_TIMEOUT", "30")
    with pytest.raises(SystemExit):
        parse_args()


def test_backend_scheduler_rejects_unknown_strategy(monkeypatch):
    monkeypatch.setattr("sys.argv", ["backend", "--routing-strategy", "unknown"])

    with pytest.raises(SystemExit):
        parse_args()


def test_scheduler_manager_rejects_unknown_v3_mode(monkeypatch):
    monkeypatch.setenv("FABI_SWARM_V3_MODE", "maybe")

    with pytest.raises(ValueError, match="off, shadow, or active"):
        SchedulerManage()


def test_scheduler_manager_rejects_unknown_recovery_policy(monkeypatch):
    monkeypatch.setenv("FABI_SWARM_V3_RECOVERY", "sometimes")

    with pytest.raises(ValueError, match="off, prefer, or require"):
        SchedulerManage()


def test_scheduler_prefers_recovery_only_for_exact_streaming_sampling():
    manager = SchedulerManage()
    manager.active_v3_routes = object()

    assert (
        manager.preferred_recovery_level({"stream": True, "temperature": 0})
        == RecoveryLevel.RECOVERABLE
    )
    assert (
        manager.preferred_recovery_level({"stream": False, "temperature": 0})
        == RecoveryLevel.RESTARTABLE
    )
    assert (
        manager.preferred_recovery_level({"stream": True, "temperature": 0.7})
        == RecoveryLevel.RESTARTABLE
    )


def test_scheduler_promotes_route_before_fencing_recovery_journal():
    manager = SchedulerManage()
    calls = []
    promoted = SimpleNamespace(primary_plan=SimpleNamespace(epoch=8, route_id="backup-promotion-8"))
    manager.active_v3_routes = SimpleNamespace(
        promote_recovery=lambda request_id, failed_epoch: (
            calls.append(("route", request_id, failed_epoch)) or promoted
        ),
        release=lambda request_id: calls.append(("release", request_id)),
    )
    snapshot = SimpleNamespace(epoch=7)
    recovering = SimpleNamespace(epoch=8)
    manager.recovery_journal = SimpleNamespace(
        get=lambda request_id: snapshot,
        begin_recovery=lambda request_id, **kwargs: (
            calls.append(("journal", request_id, kwargs)) or recovering
        ),
    )

    assert manager.promote_generation_recovery("request", failed_epoch=7) == (
        recovering,
        promoted,
    )
    assert calls == [
        ("route", "request", 7),
        (
            "journal",
            "request",
            {
                "failed_epoch": 7,
                "new_epoch": 8,
                "replacement_route_id": "backup-promotion-8",
            },
        ),
    ]


def test_scheduler_releases_promoted_route_if_journal_fence_fails():
    manager = SchedulerManage()
    released = []
    manager.active_v3_routes = SimpleNamespace(
        promote_recovery=lambda request_id, failed_epoch: SimpleNamespace(
            primary_plan=SimpleNamespace(epoch=8, route_id="backup-promotion-8")
        ),
        release=released.append,
    )

    def fail_begin(*args, **kwargs):
        raise RuntimeError("journal unavailable")

    manager.recovery_journal = SimpleNamespace(
        get=lambda request_id: SimpleNamespace(epoch=7),
        begin_recovery=fail_begin,
    )

    with pytest.raises(RuntimeError, match="journal unavailable"):
        manager.promote_generation_recovery("request", failed_epoch=7)
    assert released == ["request"]


def test_scheduler_builds_exact_route_fenced_chat_replay_request():
    manager = SchedulerManage()
    sampling = SamplingReplayContract(
        params_json='{"min_tokens":4,"temperature":0,"top_p":0.9}',
        params_hash=hashlib.sha256(b'{"min_tokens":4,"temperature":0,"top_p":0.9}').hexdigest(),
        mode=SamplingReplayMode.GREEDY,
    )
    snapshot = SimpleNamespace(
        state=RecoveryState.RECOVERING,
        epoch=8,
        route_ids=("primary", "backup-promotion-8"),
        committed_position=2,
        committed_output_token_ids=(40, 50),
        replay_token_ids=(10, 20, 30, 40, 50),
        spec=SimpleNamespace(
            model_swarm_id="ab" * 32,
            prompt_token_ids=(10, 20, 30),
            reserved_context_tokens=19,
            sampling=sampling,
        ),
    )
    plan = SimpleNamespace(
        epoch=8,
        route_id="backup-promotion-8",
        model_swarm_id="ab" * 32,
        stages=(
            SimpleNamespace(worker_id="backup-head"),
            SimpleNamespace(worker_id="backup-tail"),
        ),
    )
    manager.recovery_journal = SimpleNamespace(get=lambda request_id: snapshot)
    manager.active_v3_routes = SimpleNamespace(
        execution_context=lambda request_id: SimpleNamespace(
            primary_plan=plan,
            manifest=SimpleNamespace(model_id="Qwen/Qwen3-4B"),
        )
    )

    original = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "temperature": 0,
        "min_tokens": 4,
        "max_completion_tokens": 16,
        "tools": [{"type": "function", "function": {"name": "read"}}],
    }
    head, request = manager.build_generation_replay_request(
        "request",
        original_request=original,
        model_name="Qwen/Qwen3-4B",
    )

    assert head == "backup-head"
    assert request == {
        "authority_request_id": "request",
        "request": {
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "temperature": 0,
            "min_tokens": 2,
            "top_p": 0.9,
            "max_completion_tokens": 14,
            "tools": [{"type": "function", "function": {"name": "read"}}],
            "request_id": "request",
            "model": "Qwen/Qwen3-4B",
            "vllm_xargs": {
                "parallax_routing_table": ["backup-head", "backup-tail"],
                "parallax_scheduler_request_id": "request",
                "fabi_route_id": "backup-promotion-8",
                "fabi_route_epoch": 8,
            },
        },
        "original_prompt_token_ids": [10, 20, 30],
        "committed_output_token_ids": [40, 50],
    }
    assert original["min_tokens"] == 4


def test_scheduler_completes_replay_with_the_journal_checksum_and_rng_position():
    manager = SchedulerManage()
    snapshot = SimpleNamespace(sequence_checksum="cd" * 32, rng_position=0)
    calls = []
    manager.recovery_journal = SimpleNamespace(
        get=lambda request_id: snapshot,
        complete_replay=lambda request_id, **kwargs: calls.append((request_id, kwargs)),
    )

    manager.complete_generation_replay("request", epoch=8)

    assert calls == [
        (
            "request",
            {
                "epoch": 8,
                "sequence_checksum": "cd" * 32,
                "rng_position": 0,
            },
        )
    ]


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
        heartbeat_timeout=DEFAULT_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS,
        planning_context_tokens=16_384,
        preferred_context_tokens=32_768,
        require_exact_weight_metadata=True,
        epoch_allocator=manager.epoch_allocator,
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


def test_cluster_node_info_exposes_bounded_v3_worker_error():
    manager = SchedulerManage()
    node = SimpleNamespace(
        node_id="rejected-worker",
        is_active=True,
        supports_frontend=False,
        swarm_v3={
            "state": "rejected",
            "error": {
                "code": "ArtifactVerificationError",
                "detail": "x" * 400,
                "internal": "must not cross the status boundary",
            },
        },
        hardware=SimpleNamespace(num_gpus=1, gpu_name="RTX", memory_gb=16.0),
    )

    info = manager.build_node_info(node)

    assert info["swarm_v3_state"] == "rejected"
    assert info["swarm_v3_error"] == {
        "code": "ArtifactVerificationError",
        "detail": "x" * 256,
    }


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


def test_active_v3_mode_fails_closed_without_verified_planner(monkeypatch):
    monkeypatch.setenv("FABI_SWARM_V3_MODE", "active")
    manager = SchedulerManage()
    manager.scheduler = SimpleNamespace(swarm_v3_shadow=None)
    manager.iroh_transport = SimpleNamespace()

    with pytest.raises(RuntimeError, match="planner failed"):
        manager._start_active_v3_routes()


def test_active_v3_scheduler_requires_persistent_epoch_storage(monkeypatch):
    monkeypatch.setenv("FABI_SWARM_V3_MODE", "active")
    monkeypatch.delenv("FABI_SWARM_V3_EPOCH_DB", raising=False)
    manager = SchedulerManage()

    with pytest.raises(RuntimeError, match="persistent storage"):
        manager._start_scheduler("Qwen/Qwen3-1.7B", 1)


def test_active_v3_routing_receives_exact_token_budget(monkeypatch):
    monkeypatch.setenv("FABI_SWARM_V3_MODE", "active")
    calls = []
    active = SimpleNamespace(
        reserve=lambda **kwargs: calls.append(kwargs) or ("head", "tail"),
    )
    manager = SchedulerManage()
    manager.scheduler = SimpleNamespace(
        serving_ready=lambda: True,
        _admission_paused=False,
    )
    manager.active_v3_routes = active

    route = manager.get_routing_table(
        "request",
        1.0,
        16_316,
        prompt_tokens=12_220,
        reserved_output_tokens=4_096,
    )

    assert route == ["head", "tail"]
    assert calls == [
        {
            "request_id": "request",
            "prompt_tokens": 12_220,
            "reserved_output_tokens": 4_096,
            "recovery_level": RecoveryLevel.RESTARTABLE,
        }
    ]


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
