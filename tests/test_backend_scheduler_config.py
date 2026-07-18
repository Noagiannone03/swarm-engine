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
    )
    thread_class.return_value.start.assert_called_once_with()
