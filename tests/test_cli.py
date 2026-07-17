from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from parallax import cli


def test_serve_command_launches_local_server_without_scheduler(tmp_path):
    launch_script = tmp_path / "src" / "parallax" / "launch.py"
    launch_script.parent.mkdir(parents=True)
    launch_script.touch()

    args = Namespace(model_path="Qwen/Qwen3-0.6B", skip_upload=True)

    with (
        patch.object(cli, "check_python_version"),
        patch.object(cli, "get_project_root", return_value=Path(tmp_path)),
        patch.object(cli.sys, "executable", "/repo/.venv/bin/python"),
        patch.object(cli, "_execute_with_graceful_shutdown") as execute,
    ):
        cli.serve_command(args, ["--log-level", "DEBUG", "--port", "3005"])

    cmd = execute.call_args.args[0]
    env = execute.call_args.kwargs["env"]

    assert cmd == [
        "/repo/.venv/bin/python",
        str(launch_script),
        "--model-path",
        "Qwen/Qwen3-0.6B",
        "--log-level",
        "DEBUG",
        "--port",
        "3005",
    ]
    assert "--scheduler-addr" not in cmd
    assert env["SGLANG_ENABLE_JIT_DEEPGEMM"] == "0"


def test_join_command_injects_long_context_runtime_defaults(tmp_path):
    launch_script = tmp_path / "src" / "parallax" / "launch.py"
    launch_script.parent.mkdir(parents=True)
    launch_script.touch()

    args = Namespace(scheduler_addr="auto", skip_upload=True, use_relay=False)

    with (
        patch.object(cli, "check_python_version"),
        patch.object(cli, "get_project_root", return_value=Path(tmp_path)),
        patch.object(cli.sys, "executable", "/repo/.venv/bin/python"),
        patch.object(cli, "_execute_with_graceful_shutdown") as execute,
    ):
        cli.join_command(args, ["--log-level", "DEBUG"])

    cmd = execute.call_args.args[0]
    env = execute.call_args.kwargs["env"]

    assert cmd == [
        "/repo/.venv/bin/python",
        str(launch_script),
        "--max-num-tokens-per-batch",
        "8192",
        "--max-sequence-length",
        "65536",
        "--max-batch-size",
        "2",
        "--kv-block-size",
        "32",
        "--scheduler-addr",
        "auto",
        "--log-level",
        "DEBUG",
    ]
    assert env["SGLANG_ENABLE_JIT_DEEPGEMM"] == "0"


def test_join_command_preserves_explicit_worker_limits(tmp_path):
    launch_script = tmp_path / "src" / "parallax" / "launch.py"
    launch_script.parent.mkdir(parents=True)
    launch_script.touch()

    args = Namespace(scheduler_addr="auto", skip_upload=True, use_relay=False)
    explicit = [
        "--max-sequence-length",
        "32768",
        "--max-batch-size=1",
        "--enable-prefix-cache",
    ]

    with (
        patch.object(cli, "check_python_version"),
        patch.object(cli, "get_project_root", return_value=Path(tmp_path)),
        patch.object(cli.sys, "executable", "/repo/.venv/bin/python"),
        patch.object(cli, "_execute_with_graceful_shutdown") as execute,
    ):
        cli.join_command(args, explicit)

    cmd = execute.call_args.args[0]
    assert cmd.count("--max-sequence-length") == 1
    assert cmd[cmd.index("--max-sequence-length") + 1] == "32768"
    assert "--max-batch-size=1" in cmd
    assert "--enable-prefix-cache" in cmd


def test_main_dispatches_serve_command_with_passthrough_args():
    with (
        patch.object(
            cli.sys,
            "argv",
            [
                "parallax",
                "serve",
                "--model-path",
                "Qwen/Qwen3-0.6B",
                "--log-level",
                "DEBUG",
            ],
        ),
        patch.object(cli, "serve_command") as serve_command,
    ):
        cli.main()

    args, passthrough_args = serve_command.call_args.args
    assert args.command == "serve"
    assert args.model_path == "Qwen/Qwen3-0.6B"
    assert passthrough_args == ["--log-level", "DEBUG"]


def test_process_group_popen_kwargs_are_native():
    assert cli._process_group_popen_kwargs("posix") == {"start_new_session": True}
    assert cli._process_group_popen_kwargs("nt") == {"creationflags": 0x00000200}
