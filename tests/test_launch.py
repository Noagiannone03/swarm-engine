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
