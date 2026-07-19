from argparse import Namespace

import pytest

from parallax.launch import (
    _prepare_engine_core_generation,
    _update_args_from_shared_state,
    _wait_executors_check_layer_change,
)
from parallax.p2p.server import ServerState
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
