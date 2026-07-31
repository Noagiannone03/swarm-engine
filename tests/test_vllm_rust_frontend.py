import json
from types import SimpleNamespace

import pytest

from parallax.server import vllm_rust_frontend


def test_resolve_vllm_rs_binary_prefers_path(monkeypatch, tmp_path):
    binary = tmp_path / "vllm-rs"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    monkeypatch.setattr(vllm_rust_frontend.shutil, "which", lambda name: str(binary))

    assert vllm_rust_frontend.resolve_vllm_rs_binary() == str(binary)


def test_resolve_vllm_rs_binary_falls_back_to_python_bin(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    other_bin_dir = tmp_path / "other-bin"
    bin_dir.mkdir()
    other_bin_dir.mkdir()
    python = bin_dir / "python"
    binary = bin_dir / "vllm-rs"
    python.write_text("")
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    monkeypatch.setattr(vllm_rust_frontend.shutil, "which", lambda name: None)
    monkeypatch.setattr(vllm_rust_frontend.sysconfig, "get_path", lambda name: str(other_bin_dir))
    monkeypatch.setattr(vllm_rust_frontend.sys, "executable", str(python))

    assert vllm_rust_frontend.resolve_vllm_rs_binary() == str(binary)


def test_resolve_vllm_rs_binary_falls_back_to_scripts_dir(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    scripts_dir = tmp_path / "scripts"
    bin_dir.mkdir()
    scripts_dir.mkdir()
    python = bin_dir / "python"
    binary = scripts_dir / "vllm-rs"
    python.write_text("")
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    monkeypatch.setattr(vllm_rust_frontend.shutil, "which", lambda name: None)
    monkeypatch.setattr(vllm_rust_frontend.sysconfig, "get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr(vllm_rust_frontend.sys, "executable", str(python))

    assert vllm_rust_frontend.resolve_vllm_rs_binary() == str(binary)


def test_resolve_vllm_rs_binary_raises_when_missing(monkeypatch, tmp_path):
    python = tmp_path / "python"
    scripts_dir = tmp_path / "scripts"
    python.write_text("")
    scripts_dir.mkdir()

    monkeypatch.setattr(vllm_rust_frontend.shutil, "which", lambda name: None)
    monkeypatch.setattr(vllm_rust_frontend.sysconfig, "get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr(vllm_rust_frontend.sys, "executable", str(python))

    with pytest.raises(vllm_rust_frontend.VllmRustFrontendNotFound, match="./install.sh"):
        vllm_rust_frontend.resolve_vllm_rs_binary()


def test_frontend_is_available_on_windows_with_portable_binary(monkeypatch):
    monkeypatch.setattr(vllm_rust_frontend.os, "name", "nt")
    monkeypatch.setattr(vllm_rust_frontend, "resolve_vllm_rs_binary", lambda: "vllm-rs.exe")

    assert vllm_rust_frontend.vllm_rust_frontend_available() is True


def test_frontend_is_available_with_posix_runtime_and_binary(monkeypatch):
    monkeypatch.setattr(vllm_rust_frontend.os, "name", "posix")
    monkeypatch.setattr(vllm_rust_frontend, "resolve_vllm_rs_binary", lambda: "/bin/vllm-rs")

    assert vllm_rust_frontend.vllm_rust_frontend_available() is True


def test_runtime_args_default_to_language_model_only():
    args = SimpleNamespace(model_path="mlx-community/MiniMax-M3-4bit", max_sequence_length=None)

    runtime_args = json.loads(vllm_rust_frontend._runtime_args_json(args))

    assert runtime_args == {
        "model_tag": "mlx-community/MiniMax-M3-4bit",
        "language_model_only": True,
    }


def test_runtime_args_include_max_model_len_when_configured():
    args = SimpleNamespace(model_path="Qwen/Qwen3-0.6B", max_sequence_length=4096)

    runtime_args = json.loads(vllm_rust_frontend._runtime_args_json(args))

    assert runtime_args == {
        "model_tag": "Qwen/Qwen3-0.6B",
        "language_model_only": True,
        "max_model_len": 4096,
    }


def test_runtime_args_forward_official_engine_ready_timeout(monkeypatch):
    args = SimpleNamespace(model_path="Qwen/Qwen3-0.6B", max_sequence_length=4096)
    monkeypatch.setenv("VLLM_ENGINE_READY_TIMEOUT_S", "3600")

    runtime_args = json.loads(vllm_rust_frontend._runtime_args_json(args))

    assert runtime_args["engine_ready_timeout_secs"] == 3600


@pytest.mark.parametrize("value", ["slow", "-1"])
def test_runtime_args_reject_invalid_engine_ready_timeout(monkeypatch, value):
    args = SimpleNamespace(model_path="Qwen/Qwen3-0.6B", max_sequence_length=4096)
    monkeypatch.setenv("VLLM_ENGINE_READY_TIMEOUT_S", value)

    with pytest.raises(ValueError, match="non-negative integer"):
        vllm_rust_frontend._runtime_args_json(args)


def test_runtime_args_alias_local_model_path_to_scheduler_model_name():
    args = SimpleNamespace(
        model_path="/models/Qwen3-0.6B-bf16",
        served_model_name="Qwen/Qwen3-0.6B",
        max_sequence_length=4096,
    )

    runtime_args = json.loads(vllm_rust_frontend._runtime_args_json(args))

    assert runtime_args == {
        "model_tag": "/models/Qwen3-0.6B-bf16",
        "served_model_name": ["Qwen/Qwen3-0.6B"],
        "language_model_only": True,
        "max_model_len": 4096,
    }


def test_windows_frontend_binds_tcp_without_posix_fd_inheritance(monkeypatch):
    launched = {}

    class FakeProcess:
        returncode = None

        @staticmethod
        def poll():
            return None

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(vllm_rust_frontend.os, "name", "nt")
    monkeypatch.setattr(
        vllm_rust_frontend, "resolve_vllm_rs_binary", lambda: r"C:\Fabi\vllm-rs.exe"
    )
    monkeypatch.setattr(vllm_rust_frontend.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(vllm_rust_frontend.time, "sleep", lambda _: None)
    args = SimpleNamespace(
        host="127.0.0.1",
        port=19080,
        executor_input_ipc="tcp://127.0.0.1:19081",
        executor_output_ipc="tcp://127.0.0.1:19082",
        model_path="Qwen/Qwen3-0.6B",
        max_sequence_length=4096,
    )

    process = vllm_rust_frontend.launch_vllm_rust_frontend(args)

    assert process.listen_fd is None
    assert "--listen-fd" not in launched["command"]
    address_index = launched["command"].index("--listen-address") + 1
    assert launched["command"][address_index] == "127.0.0.1:19080"
    assert "pass_fds" not in launched["kwargs"]


def test_windows_frontend_normalizes_default_localhost_for_socket_addr(monkeypatch):
    launched = {}

    class FakeProcess:
        returncode = None

        @staticmethod
        def poll():
            return None

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(vllm_rust_frontend.os, "name", "nt")
    monkeypatch.setattr(
        vllm_rust_frontend, "resolve_vllm_rs_binary", lambda: r"C:\Fabi\vllm-rs.exe"
    )
    monkeypatch.setattr(vllm_rust_frontend.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(vllm_rust_frontend.time, "sleep", lambda _: None)
    args = SimpleNamespace(
        host="localhost",
        port=3001,
        executor_input_ipc="tcp://127.0.0.1:19081",
        executor_output_ipc="tcp://127.0.0.1:19082",
        model_path="Qwen/Qwen3-0.6B",
        max_sequence_length=4096,
    )

    vllm_rust_frontend.launch_vllm_rust_frontend(args)

    address_index = launched["command"].index("--listen-address") + 1
    assert launched["command"][address_index] == "127.0.0.1:3001"


def test_windows_frontend_formats_ipv6_socket_addr():
    assert vllm_rust_frontend._numeric_listen_address("::1", 3001) == "[::1]:3001"
