import threading
import time
from argparse import Namespace

import pytest

from parallax.launch import (
    MemoryPressureGuard,
    _build_memory_pressure_guards,
    _consume_initial_autonomous_reload,
    _prepare_engine_core_generation,
    _update_args_from_shared_state,
    _wait_executors_check_layer_change,
    _wait_for_contract_replan,
    _wait_for_initial_layer_allocation,
    _wait_for_v3_placement_rollback,
)
from parallax.p2p.server import ServerState
from parallax.server.memory_budget import GIB, MemoryPressureController
from parallax.utils.shared_state import SharedState


class _ProcessState:
    def __init__(self, alive=True):
        self.alive = alive

    def is_alive(self):
        return self.alive


def test_unassigned_worker_remains_joining_until_complete_span_arrives(monkeypatch):
    shared_state = SharedState(
        {
            "block_start_index": None,
            "block_end_index": None,
            "model_name": None,
        }
    )
    polls = []

    def allocate_after_first_poll(delay):
        polls.append(delay)
        shared_state.update(
            block_start_index=4,
            block_end_index=28,
            model_name="Qwen/Qwen3-4B",
        )

    monkeypatch.setattr("parallax.launch.time.sleep", allocate_after_first_poll)

    _wait_for_initial_layer_allocation(
        shared_state,
        _ProcessState(),
        timeout_seconds=0,
        poll_seconds=0.25,
    )

    assert polls == [0.25]


def test_waiting_worker_fails_if_p2p_controller_exits():
    shared_state = SharedState(
        {
            "block_start_index": None,
            "block_end_index": None,
            "model_name": None,
        }
    )

    with pytest.raises(RuntimeError, match="P2P controller exited"):
        _wait_for_initial_layer_allocation(
            shared_state,
            _ProcessState(alive=False),
            timeout_seconds=0,
            poll_seconds=0,
        )


def test_initial_allocation_operator_deadline_is_optional(monkeypatch):
    shared_state = SharedState(
        {
            "block_start_index": None,
            "block_end_index": None,
            "model_name": None,
        }
    )
    times = iter([0.0, 2.0])
    monkeypatch.setattr("parallax.launch.time.monotonic", lambda: next(times))

    with pytest.raises(RuntimeError, match="within 1s"):
        _wait_for_initial_layer_allocation(
            shared_state,
            _ProcessState(),
            timeout_seconds=1,
            poll_seconds=0,
        )


def test_cold_join_consumes_only_its_initial_reload_wakeup():
    cold = SharedState(
        {
            "swarm_v3_placement_phase": "building",
            "swarm_v3_placement_generation": 1,
            "_layer_allocation_changed": True,
        }
    )
    legacy = SharedState(
        {
            "swarm_v3_placement_phase": "legacy",
            "swarm_v3_placement_generation": 0,
            "_layer_allocation_changed": True,
        }
    )

    assert _consume_initial_autonomous_reload(cold) is True
    assert cold.get_layer_allocation_changed() is False
    assert _consume_initial_autonomous_reload(legacy) is False
    assert legacy.get_layer_allocation_changed() is True


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


def test_model_context_limit_clamps_each_generation_without_losing_worker_capability():
    args = Namespace(
        model_path=None,
        max_sequence_length=65536,
        tp_size=1,
        enable_weight_refit=False,
        weight_refit_mode=None,
    )
    shared_state = SharedState(
        {
            "model_name": "Qwen/Qwen3-1.7B",
            "model_max_sequence_length": 40960,
            "block_start_index": 0,
            "block_end_index": 28,
            "tp_size": 1,
            "enable_weight_refit": False,
            "weight_refit_mode": None,
        }
    )

    _update_args_from_shared_state(args, shared_state, force_update=False)
    assert args.max_sequence_length == 40960

    shared_state.update(
        model_name="long-context/model",
        model_max_sequence_length=131072,
    )
    _update_args_from_shared_state(args, shared_state, force_update=True)

    assert args.max_sequence_length == 65536
    assert args._worker_max_sequence_length == 65536


def test_scheduler_allocation_context_caps_runtime_model_length():
    args = Namespace(
        model_path=None,
        max_sequence_length=65536,
        tp_size=1,
        enable_weight_refit=False,
        weight_refit_mode=None,
    )
    shared_state = SharedState(
        {
            "model_name": "Qwen/Qwen3-4B",
            "model_max_sequence_length": 40960,
            "planned_context_tokens": 32768,
            "allocation_epoch": 2,
            "block_start_index": 5,
            "block_end_index": 36,
            "tp_size": 1,
            "enable_weight_refit": False,
            "weight_refit_mode": None,
        }
    )

    _update_args_from_shared_state(args, shared_state, force_update=False)

    assert args.max_sequence_length == 32768
    assert args.planned_context_tokens == 32768
    assert args._worker_max_sequence_length == 65536


def test_later_allocation_epoch_recomputes_context_from_stable_worker_ceiling():
    args = Namespace(
        model_path=None,
        max_sequence_length=65536,
        tp_size=1,
        enable_weight_refit=False,
        weight_refit_mode=None,
    )
    shared_state = SharedState(
        {
            "model_name": "Qwen/Qwen3-4B",
            "model_max_sequence_length": 40960,
            "planned_context_tokens": 16384,
            "allocation_epoch": 2,
            "block_start_index": 5,
            "block_end_index": 36,
            "tp_size": 1,
            "enable_weight_refit": False,
            "weight_refit_mode": None,
        }
    )

    _update_args_from_shared_state(args, shared_state, force_update=False)
    assert args.max_sequence_length == 16384

    shared_state.update(planned_context_tokens=32768, allocation_epoch=3)
    _update_args_from_shared_state(args, shared_state, force_update=True)

    assert args.max_sequence_length == 32768
    assert args._worker_max_sequence_length == 65536


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


def test_executor_failure_is_not_treated_as_a_normal_worker_shutdown():
    class FailedExecutor:
        pid = 1234
        exitcode = 1

        @staticmethod
        def is_alive():
            return False

    shared_state = SharedState.create()
    shared_state.set_status(ServerState.READY.value)

    with pytest.raises(RuntimeError, match=r"1234.*1"):
        _wait_executors_check_layer_change(shared_state, [FailedExecutor()])

    assert shared_state.get_status() == ServerState.INITIALIZING.value
    assert shared_state.get("frontend_alive") is False


def test_memory_contract_failure_keeps_heartbeat_generation_alive_for_replan():
    class FailedExecutor:
        pid = 1234
        exitcode = 1

        @staticmethod
        def is_alive():
            return False

    shared_state = SharedState.create()
    shared_state.update(
        status=ServerState.READY.value,
        memory_contract_failure={
            "kind": "kv_materialization",
            "allocation_epoch": 3,
            "requested_tokens": 32_768,
            "supported_tokens": 24_000,
        },
    )

    assert _wait_executors_check_layer_change(shared_state, [FailedExecutor()]) is True
    assert shared_state.get_status() == ServerState.INITIALIZING.value


def test_autonomous_contract_replan_does_not_require_scheduler_epoch_change():
    shared_state = SharedState(
        {
            "_layer_allocation_changed": True,
            "allocation_epoch": 76,
            "swarm_v3_context_failure": None,
        }
    )

    _wait_for_contract_replan(
        shared_state,
        _ProcessState(),
        require_newer_scheduler_epoch=False,
    )


def test_autonomous_contract_replan_surfaces_terminal_local_capacity_failure():
    shared_state = SharedState(
        {
            "_layer_allocation_changed": False,
            "swarm_v3_context_failure": {
                "code": "NoSupportedContextTier",
                "detail": "measured ceiling below 4096",
            },
        }
    )

    with pytest.raises(RuntimeError, match="below 4096"):
        _wait_for_contract_replan(
            shared_state,
            _ProcessState(),
            require_newer_scheduler_epoch=False,
        )


def test_v3_load_failure_waits_for_a_new_fenced_rollback_generation():
    shared_state = SharedState.create()
    shared_state.update(
        swarm_v3_placement_generation=3,
        swarm_v3_placement_phase="building",
        swarm_v3_previous_start_layer=0,
        swarm_v3_previous_end_layer=2,
    )

    def acknowledge_failure():
        while shared_state.get("swarm_v3_placement_error") is None:
            time.sleep(0.001)
        shared_state.update(
            swarm_v3_placement_generation=4,
            block_start_index=0,
            block_end_index=2,
        )

    thread = threading.Thread(target=acknowledge_failure)
    thread.start()
    assert _wait_for_v3_placement_rollback(
        shared_state,
        failed_generation=3,
        detail="synthetic load error",
        timeout=1,
    )
    thread.join()

    assert shared_state.get("swarm_v3_placement_error") == {
        "generation": 3,
        "detail": "synthetic load error",
    }
    assert shared_state.get("swarm_v3_placement_generation") == 4


def test_v3_load_failure_without_previous_verified_span_fails_closed():
    shared_state = SharedState.create()
    shared_state.update(
        swarm_v3_placement_generation=1,
        swarm_v3_placement_phase="building",
    )

    assert not _wait_for_v3_placement_rollback(
        shared_state,
        failed_generation=1,
        detail="cold join failed",
        timeout=0,
    )
    assert shared_state.get("swarm_v3_placement_error") is None


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
