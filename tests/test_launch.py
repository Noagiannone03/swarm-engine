from argparse import Namespace

import pytest

from parallax.launch import (
    MemoryPressureGuard,
    _build_memory_pressure_guards,
    _prepare_engine_core_generation,
    _update_args_from_shared_state,
    _wait_executors_check_layer_change,
)
from parallax.p2p.server import ServerState
from parallax.server.memory_budget import GIB, MemoryPressureController
from parallax.utils.shared_state import SharedState


def test_scheduler_model_name_is_kept_as_alias_for_local_weights():
    args = Namespace(
        model_path="/models/Qwen3-0.6B-bf16",
        tp_size=1,
        enable_weight_refit=False,
        weight_refit_mode=None,
    )
    shared_state = SharedState(
        {
            "model_name": "Qwen/Qwen3-0.6B",
            "block_start_index": 0,
            "block_end_index": 2,
            "tp_size": 1,
            "enable_weight_refit": False,
            "weight_refit_mode": None,
        }
    )

    _update_args_from_shared_state(args, shared_state, force_update=False)

    assert args.model_path == "/models/Qwen3-0.6B-bf16"
    assert args.served_model_name == "Qwen/Qwen3-0.6B"


def test_local_model_path_survives_scheduler_reallocation():
    args = Namespace(
        model_path="/models/Qwen3-0.6B-MLX-4bit",
        tp_size=1,
        enable_weight_refit=False,
        weight_refit_mode=None,
    )
    shared_state = SharedState(
        {
            "model_name": "Qwen/Qwen3-0.6B",
            "block_start_index": 0,
            "block_end_index": 2,
            "tp_size": 1,
            "enable_weight_refit": False,
            "weight_refit_mode": None,
        }
    )

    _update_args_from_shared_state(args, shared_state, force_update=False)
    shared_state.update(block_start_index=3, block_end_index=12)
    _update_args_from_shared_state(args, shared_state, force_update=True)

    assert args.model_path == "/models/Qwen3-0.6B-MLX-4bit"
    assert args.served_model_name == "Qwen/Qwen3-0.6B"
    assert (args.start_layer, args.end_layer) == (3, 12)


def test_worker_without_local_model_path_tracks_scheduler_model():
    args = Namespace(
        model_path=None,
        tp_size=1,
        enable_weight_refit=False,
        weight_refit_mode=None,
    )
    shared_state = SharedState(
        {
            "model_name": "Qwen/Qwen3-0.6B",
            "block_start_index": 0,
            "block_end_index": 2,
            "tp_size": 1,
            "enable_weight_refit": False,
            "weight_refit_mode": None,
        }
    )

    _update_args_from_shared_state(args, shared_state, force_update=False)
    shared_state.update(
        model_name="Qwen/Qwen3-1.7B",
        block_start_index=2,
        block_end_index=28,
    )
    _update_args_from_shared_state(args, shared_state, force_update=True)

    assert args.model_path == "Qwen/Qwen3-1.7B"
    assert args.served_model_name == "Qwen/Qwen3-1.7B"


def test_explicit_model_alias_wins_before_manual_scheduler_assignment_arrives():
    args = Namespace(
        model_path="/models/Qwen3-0.6B-bf16",
        served_model_name="Qwen/Qwen3-0.6B",
        tp_size=1,
        enable_weight_refit=False,
        weight_refit_mode=None,
    )
    shared_state = SharedState(
        {
            "model_name": "/models/Qwen3-0.6B-bf16",
            "block_start_index": 0,
            "block_end_index": 2,
            "tp_size": 1,
            "enable_weight_refit": False,
            "weight_refit_mode": None,
        }
    )

    _update_args_from_shared_state(args, shared_state, force_update=False)

    assert args.served_model_name == "Qwen/Qwen3-0.6B"


def test_engine_core_endpoints_are_rotated_for_each_executor_generation(monkeypatch):
    batches = iter(
        [
            ["ipc:///tmp/generation-1-input", "ipc:///tmp/generation-1-output"],
            ["ipc:///tmp/generation-2-input", "ipc:///tmp/generation-2-output"],
        ]
    )
    monkeypatch.setattr("parallax.launch.create_local_zmq_endpoints", lambda count: next(batches))
    args = Namespace(
        recv_from_peer_addr="ipc:///tmp/stable-peer-input",
        send_to_peer_addr="ipc:///tmp/stable-peer-output",
    )
    shared_state = SharedState({})

    _prepare_engine_core_generation(args, shared_state, frontend_required=True)
    first_pair = (args.executor_input_ipc, args.executor_output_ipc)
    _prepare_engine_core_generation(args, shared_state, frontend_required=True)

    assert first_pair == (
        "ipc:///tmp/generation-1-input",
        "ipc:///tmp/generation-1-output",
    )
    assert (args.executor_input_ipc, args.executor_output_ipc) == (
        "ipc:///tmp/generation-2-input",
        "ipc:///tmp/generation-2-output",
    )
    assert args.recv_from_peer_addr == "ipc:///tmp/stable-peer-input"
    assert args.send_to_peer_addr == "ipc:///tmp/stable-peer-output"
    assert shared_state.get("frontend_required") is True
    assert shared_state.get("frontend_alive") is False


def test_frontend_exit_marks_generation_unready():
    class RunningExecutor:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            del timeout

    class DeadFrontend:
        def is_alive(self):
            return False

    shared_state = SharedState(
        {
            "status": ServerState.READY.value,
            "frontend_required": True,
            "frontend_alive": True,
        }
    )

    with pytest.raises(RuntimeError, match="frontend exited"):
        _wait_executors_check_layer_change(
            shared_state,
            [RunningExecutor()],
            DeadFrontend(),
        )

    assert shared_state.get_status() == ServerState.INITIALIZING.value
    assert shared_state.get("frontend_alive") is False


class _FiniteExecutor:
    def __init__(self, iterations):
        self.iterations = iterations
        self.join_count = 0

    def is_alive(self):
        return self.join_count < self.iterations

    def join(self, timeout=None):
        del timeout
        self.join_count += 1


def test_memory_warning_pauses_then_resumes_without_layer_reallocation(monkeypatch):
    monkeypatch.setattr("parallax.launch.DEFAULT_PRESSURE_POLL_SECONDS", 0)
    samples = iter([4 * GIB, 4 * GIB, 4 * GIB, 6 * GIB, 6 * GIB])
    controller = MemoryPressureController(
        system_reserve_bytes=6 * GIB,
        warning_samples=3,
        recovery_samples=2,
    )
    shared_state = SharedState.create()
    shared_state.set_status(ServerState.READY.value)

    changed = _wait_executors_check_layer_change(
        shared_state,
        [_FiniteExecutor(iterations=5)],
        memory_pressure_guards=[MemoryPressureGuard("host", controller, lambda: next(samples))],
    )

    assert changed is False
    assert shared_state.get_status() == ServerState.READY.value
    assert shared_state.get("memory_pressure") == "normal"
    assert shared_state.get_layer_allocation_changed() is False
    assert shared_state.get("_memory_shutdown_requested") is False


def test_critical_memory_requests_one_shutdown_after_drain(monkeypatch):
    monkeypatch.setattr("parallax.launch.DEFAULT_PRESSURE_POLL_SECONDS", 0)
    controller = MemoryPressureController(
        system_reserve_bytes=6 * GIB,
        critical_samples=1,
    )
    shared_state = SharedState.create()
    shared_state.set_status(ServerState.READY.value)

    changed = _wait_executors_check_layer_change(
        shared_state,
        [_FiniteExecutor(iterations=10)],
        memory_pressure_guards=[MemoryPressureGuard("host", controller, lambda: GIB)],
    )

    assert changed is False
    assert shared_state.get_status() == ServerState.INITIALIZING.value
    assert shared_state.get("memory_pressure") == "critical"
    assert shared_state.get("_memory_shutdown_requested") is True
    assert shared_state.get_layer_allocation_changed() is False


def test_worst_resource_controls_admission_until_every_resource_recovers(monkeypatch):
    monkeypatch.setattr("parallax.launch.DEFAULT_PRESSURE_POLL_SECONDS", 0)
    host_samples = iter([6 * GIB] * 5)
    cuda_samples = iter([GIB, GIB, GIB, 2 * GIB, 2 * GIB])
    host = MemoryPressureController(system_reserve_bytes=6 * GIB)
    cuda = MemoryPressureController(
        system_reserve_bytes=2 * GIB,
        warning_samples=3,
        recovery_samples=2,
    )
    shared_state = SharedState.create()
    shared_state.set_status(ServerState.READY.value)

    changed = _wait_executors_check_layer_change(
        shared_state,
        [_FiniteExecutor(iterations=5)],
        memory_pressure_guards=[
            MemoryPressureGuard("host", host, lambda: next(host_samples)),
            MemoryPressureGuard("cuda:0", cuda, lambda: next(cuda_samples)),
        ],
    )

    assert changed is False
    assert shared_state.get_status() == ServerState.READY.value
    assert shared_state.get("memory_pressure") == "normal"
    resources = shared_state.get("memory_pressure_resources")
    assert resources["host"]["level"] == "normal"
    assert resources["cuda:0"]["level"] == "normal"


def test_transient_sensor_failure_keeps_last_stable_state(monkeypatch):
    monkeypatch.setattr("parallax.launch.DEFAULT_PRESSURE_POLL_SECONDS", 0)
    controller = MemoryPressureController(system_reserve_bytes=6 * GIB)
    shared_state = SharedState.create()
    shared_state.set_status(ServerState.READY.value)

    def fail_sample():
        raise OSError("temporary counter failure")

    changed = _wait_executors_check_layer_change(
        shared_state,
        [_FiniteExecutor(iterations=1)],
        memory_pressure_guards=[MemoryPressureGuard("host", controller, fail_sample)],
    )

    assert changed is False
    assert shared_state.get_status() == ServerState.READY.value
    assert shared_state.get("memory_pressure_resources")["host"] == {
        "level": "normal",
        "sample_error": True,
        "reserve_bytes": 6 * GIB,
    }


def test_cuda_guard_is_created_for_every_visible_device(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    free = {0: 8 * GIB, 1: 12 * GIB}
    total = {0: 16 * GIB, 1: 24 * GIB}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free[device], total[device]))

    guards = _build_memory_pressure_guards()
    cuda_guards = [guard for guard in guards if guard.name.startswith("cuda:")]

    assert [guard.name for guard in cuda_guards] == ["cuda:0", "cuda:1"]
    assert [guard.available_reader() for guard in cuda_guards] == [8 * GIB, 12 * GIB]
