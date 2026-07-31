"""Subprocess wrapper for the official vLLM Rust frontend binary."""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import socket
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class VllmRustFrontendNotFound(RuntimeError):
    """Raised when `vllm-rs` cannot be resolved from PATH."""


@dataclass
class VllmRustFrontendProcess:
    process: subprocess.Popen
    listen_fd: Optional[int]
    host: str
    port: int

    def is_alive(self) -> bool:
        return self.process.poll() is None


def resolve_vllm_rs_binary() -> str:
    """Resolve the official vLLM Rust frontend binary from PATH."""
    binary = shutil.which("vllm-rs")
    if binary is not None:
        return binary

    executable_name = "vllm-rs.exe" if os.name == "nt" else "vllm-rs"
    candidates = [
        Path(sysconfig.get_path("scripts")) / executable_name,
        Path(sys.executable).parent / executable_name,
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    raise VllmRustFrontendNotFound(
        "Unable to find `vllm-rs` on PATH or next to the active Python interpreter. "
        "Run `./install.sh`, then activate `.venv` or add `.venv/bin` to PATH."
    )


def vllm_rust_frontend_available() -> bool:
    """Return whether this runtime can host Parallax's HTTP frontend."""
    try:
        resolve_vllm_rs_binary()
    except VllmRustFrontendNotFound:
        return False
    return True


def _bind_listener_socket(host: str, port: int) -> socket.socket:
    addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    last_error: Optional[OSError] = None
    for family, socktype, proto, _, sockaddr in addrinfos:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            sock.set_inheritable(True)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    assert last_error is not None
    raise last_error


def _numeric_listen_address(host: str, port: int) -> str:
    """Resolve a host to the numeric SocketAddr syntax expected by vllm-rs."""
    normalized_host = host.strip()
    if normalized_host.lower() == "localhost":
        # Fabi's loopback clients use IPv4 today. Pinning localhost avoids an
        # OS-dependent choice of ::1 which can make a later 127.0.0.1 health
        # probe fail even though both names denote the local machine.
        normalized_host = "127.0.0.1"

    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        addrinfos = socket.getaddrinfo(normalized_host, port, type=socket.SOCK_STREAM)
        if not addrinfos:
            raise OSError(f"Unable to resolve frontend listen host: {host}")
        normalized_host = str(addrinfos[0][4][0])
        address = ipaddress.ip_address(normalized_host)

    if address.version == 6:
        return f"[{address.compressed}]:{port}"
    return f"{address.compressed}:{port}"


def _runtime_args_json(args) -> str:
    runtime_args = {
        "model_tag": args.model_path,
        "language_model_only": True,
    }
    # The official Rust frontend parses ``--args-json`` with serde. Clap's
    # ``VLLM_ENGINE_READY_TIMEOUT_S`` binding is therefore bypassed on this
    # Python-supervised path (see vLLM ``parse_runtime_args_json``). Forward the
    # maintained vLLM setting explicitly so slow cold model downloads can use a
    # deliberate startup window instead of always dying at the 600 s default.
    ready_timeout = os.environ.get("VLLM_ENGINE_READY_TIMEOUT_S")
    if ready_timeout is not None:
        try:
            ready_timeout_seconds = int(ready_timeout)
        except ValueError as exc:
            raise ValueError("VLLM_ENGINE_READY_TIMEOUT_S must be a non-negative integer") from exc
        if ready_timeout_seconds < 0:
            raise ValueError("VLLM_ENGINE_READY_TIMEOUT_S must be a non-negative integer")
        runtime_args["engine_ready_timeout_secs"] = ready_timeout_seconds
    served_model_name = getattr(args, "served_model_name", None)
    if served_model_name and served_model_name != args.model_path:
        runtime_args["served_model_name"] = [served_model_name]
    if getattr(args, "max_sequence_length", None) is not None:
        runtime_args["max_model_len"] = int(args.max_sequence_length)
    return json.dumps(runtime_args, separators=(",", ":"))


def launch_vllm_rust_frontend(args) -> VllmRustFrontendProcess:
    """Launch `vllm-rs frontend` as Parallax's only HTTP frontend."""
    binary = resolve_vllm_rs_binary()
    child_env = os.environ.copy()
    if child_env.get("FABI_SWARM_V3_MODE", "").strip().lower() == "active":
        if args.host.strip().lower() not in {"localhost", "127.0.0.1", "::1"}:
            raise RuntimeError(
                "Protocol-v3 explicit abort requires the vLLM frontend to remain loopback-only"
            )
        # vLLM v0.24 exposes its maintained /abort_requests engine control
        # endpoint in dev mode. The listener remains loopback-only, while Fabi
        # exposes a separate route-fenced RPC to the authenticated coordinator.
        child_env["VLLM_SERVER_DEV_MODE"] = "1"

    runtime_args = _runtime_args_json(args)

    cmd = [
        binary,
        "frontend",
    ]
    listener: Optional[socket.socket] = None
    listen_fd: Optional[int] = None
    bound_port = int(args.port)
    popen_kwargs = {"env": child_env}
    if os.name == "posix":
        listener = _bind_listener_socket(args.host, args.port)
        listen_fd = listener.fileno()
        bound_port = int(listener.getsockname()[1])
        cmd.extend(["--listen-fd", str(listen_fd)])
        popen_kwargs["pass_fds"] = (listen_fd,)
    else:
        # The portable vLLM frontend binds TCP itself on Windows. Passing a
        # socket HANDLE through subprocess is not equivalent to POSIX fd
        # inheritance; direct bind is the native, race-free primitive.
        cmd.extend(["--listen-address", _numeric_listen_address(str(args.host), bound_port)])
    cmd.extend(
        [
            "--input-address",
            args.executor_input_ipc,
            "--output-address",
            args.executor_output_ipc,
            "--engine-count",
            "1",
            "--args-json",
            runtime_args,
        ]
    )

    logger.info(
        "Launching vLLM Rust frontend on %s:%s with input=%s output=%s",
        args.host,
        bound_port,
        args.executor_input_ipc,
        args.executor_output_ipc,
    )
    process = subprocess.Popen(cmd, **popen_kwargs)

    # On POSIX the child inherited the listener fd; close the parent's copy so
    # shutdown fully releases the port when the Rust frontend exits.
    if listener is not None:
        listener.close()
    time.sleep(0.05)
    if process.poll() is not None:
        raise RuntimeError(f"vLLM Rust frontend exited early with code {process.returncode}")

    if bound_port != args.port:
        args.port = bound_port

    return VllmRustFrontendProcess(
        process=process,
        listen_fd=listen_fd,
        host=args.host,
        port=bound_port,
    )


def stop_vllm_rust_frontend(frontend_process: Optional[VllmRustFrontendProcess]):
    """Terminate the Rust frontend subprocess."""
    if frontend_process is None:
        return None

    process = frontend_process.process
    if process.poll() is not None:
        return frontend_process

    logger.debug("Terminating vLLM Rust frontend subprocess %s", process.pid)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("vLLM Rust frontend did not exit after termination; killing it")
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    return frontend_process
