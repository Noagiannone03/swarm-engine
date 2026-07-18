from argparse import Namespace

from parallax.launch import _update_args_from_shared_state
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
