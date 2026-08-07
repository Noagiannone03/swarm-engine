"""Backend-initialized capacity admission for community workers.

The placement controller runs in a lightweight P2P process, while model
execution happens in a separate process.  Measuring free memory only in the
P2P process misses the Python/backend context that the executor is about to
create.  This module keeps a backend-initialized probe alive while an
unassigned worker advertises capacity, so the advertised envelope includes
that real baseline cost.

Capacity may recover while the worker is still in STANDBY.  Downward changes
are applied immediately; upward changes need several consecutive samples.
Once placement enters BUILDING, the launch controller freezes the envelope
for that immutable executor generation.
"""

from __future__ import annotations

import importlib
import math
import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from parallax.server.server_info import detect_node_hardware
from parallax.server.backend_capabilities import (
    DeviceKind,
    canonical_device_for_rank,
    device_kind,
    require_executor_backend,
)
from parallax.utils.shared_state import SharedState
from parallax_utils.logging_config import get_logger, set_log_level

logger = get_logger(__name__)

DEFAULT_CAPACITY_SAMPLE_SECONDS = 1.0
DEFAULT_CAPACITY_RISE_SAMPLES = 3


@dataclass(frozen=True)
class StableCapacityObservation:
    stable_bytes: int
    changed: bool


class StableCapacityTracker:
    """Conservatively stabilize a live available-memory signal.

    A loss of capacity is safety-sensitive and therefore takes effect on the
    first observation.  A gain is accepted only after repeated observations
    at or above the candidate value.  This avoids flapping DHT offers when an
    application briefly releases reusable pages, without using elapsed time as
    a proxy for worker health.
    """

    def __init__(self, *, rise_samples: int = DEFAULT_CAPACITY_RISE_SAMPLES):
        if rise_samples <= 0:
            raise ValueError("rise_samples must be positive")
        self.rise_samples = int(rise_samples)
        self._stable_bytes: int | None = None
        self._rise_floor: int | None = None
        self._rise_count = 0

    def observe(self, capacity_bytes: int) -> StableCapacityObservation:
        value = max(0, int(capacity_bytes))
        previous = self._stable_bytes
        if previous is None or value < previous:
            self._stable_bytes = value
            self._rise_floor = None
            self._rise_count = 0
        elif value > previous:
            self._rise_floor = (
                value if self._rise_floor is None else min(self._rise_floor, value)
            )
            self._rise_count += 1
            if self._rise_count >= self.rise_samples:
                self._stable_bytes = self._rise_floor
                self._rise_floor = None
                self._rise_count = 0
        else:
            self._rise_floor = None
            self._rise_count = 0
        assert self._stable_bytes is not None
        return StableCapacityObservation(
            stable_bytes=self._stable_bytes,
            changed=self._stable_bytes != previous,
        )


def _configured_sample_seconds() -> float:
    raw = os.environ.get(
        "FABI_CAPACITY_SAMPLE_SECONDS", str(DEFAULT_CAPACITY_SAMPLE_SECONDS)
    ).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("FABI_CAPACITY_SAMPLE_SECONDS must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("FABI_CAPACITY_SAMPLE_SECONDS must be finite and positive")
    return value


def _configured_rise_samples() -> int:
    raw = os.environ.get(
        "FABI_CAPACITY_RISE_SAMPLES", str(DEFAULT_CAPACITY_RISE_SAMPLES)
    ).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("FABI_CAPACITY_RISE_SAMPLES must be an integer") from exc
    if value <= 0:
        raise ValueError("FABI_CAPACITY_RISE_SAMPLES must be positive")
    return value


def _initialize_backend_runtime(
    gpu_backend: str, execution_device: str | None = None
) -> str:
    """Initialize the same device runtime imported by the future executor."""

    if execution_device is None:
        # Keep heavyweight Torch/MLX device discovery out of scheduler and
        # contract-only imports. The capacity subprocess needs it only when the
        # installer did not pin an execution provider explicitly.
        from parallax.utils.utils import get_current_device

        device = get_current_device()
    else:
        device = execution_device
    kind = device_kind(device)
    device = canonical_device_for_rank(device, 0)
    require_executor_backend(device, gpu_backend)
    if kind is DeviceKind.MLX:
        import mlx.core as mx

        # Importing the executor pulls in the actual MLX model/cache stack.
        importlib.import_module("parallax.server.executor.mlx_executor")
        mx.distributed.init()
        mx.eval(mx.zeros(1))
        return device
    if kind in {DeviceKind.CUDA, DeviceKind.XPU}:
        import torch

        runtime = torch.cuda if kind is DeviceKind.CUDA else torch.xpu
        runtime.init()
        module = (
            "parallax.server.executor.vllm_executor"
            if gpu_backend == "vllm"
            else "parallax.server.executor.sglang_executor"
        )
        importlib.import_module(module)
        # Materialize a context on every visible device before mem_get_info.
        for index in range(runtime.device_count()):
            with runtime.device(index):
                torch.empty(1, device=f"{kind.value}:{index}")
        return kind.value
    if kind is DeviceKind.DIRECTML:
        try:
            import onnxruntime as ort
        except (ImportError, OSError) as exc:
            raise RuntimeError("ONNX Runtime DirectML is not installed") from exc
        if "DmlExecutionProvider" not in tuple(ort.get_available_providers()):
            raise RuntimeError("ONNX Runtime did not expose DmlExecutionProvider")
        # Import the real executor/runner stack before the first measurement.
        # The live DXGI query performed by the detector then observes this
        # process, while the later immutable placement subtracts exact signed
        # stage and KV geometry.
        importlib.import_module("parallax.server.executor.onnx_executor")
        return device
    raise RuntimeError(f"Unsupported inference device for capacity probe: {device}")


def run_runtime_capacity_probe(
    gpu_backend: str,
    shared_state_dict: dict,
    log_level: str = "INFO",
    execution_device: str | None = None,
    *,
    detector: Callable[[str | None, str | None], dict[str, Any]] = detect_node_hardware,
) -> None:
    """Publish stable live capacity until placement freezes the contract."""

    set_log_level(log_level)
    state = SharedState(shared_state_dict)
    try:
        device = _initialize_backend_runtime(gpu_backend, execution_device)
        tracker = StableCapacityTracker(rise_samples=_configured_rise_samples())
        sequence = 0
        state.update(
            capacity_probe_state="sampling",
            capacity_probe_error=None,
            capacity_probe_device=device,
            execution_device=device,
        )
        sample_seconds = _configured_sample_seconds()
        while not state.get("capacity_contract_frozen", False) and not state.get(
            "capacity_probe_stop", False
        ):
            hardware = detector(None, device)
            advertised = hardware.get("usable_memory_bytes")
            if advertised is None:
                raise RuntimeError("backend did not expose a usable-memory envelope")
            observation = tracker.observe(int(advertised))
            if observation.changed or state.get("capacity_hardware") is None:
                sequence += 1
                published = dict(hardware)
                published["usable_memory_bytes"] = observation.stable_bytes
                published["capacity_sequence"] = sequence
                published["capacity_observed_at_ms"] = time.time_ns() // 1_000_000
                state.update(
                    capacity_probe_state="ready",
                    capacity_hardware=published,
                    capacity_probe_sequence=sequence,
                )
                logger.info(
                    "Runtime capacity probe published %.2f GiB on %s (sample %d)",
                    observation.stable_bytes / 1024**3,
                    device,
                    sequence,
                )
            time.sleep(sample_seconds)
    except Exception as exc:  # noqa: BLE001 - subprocess status boundary
        state.update(
            capacity_probe_state="failed",
            capacity_probe_error={
                "code": type(exc).__name__,
                "detail": str(exc)[:512],
            },
        )
        logger.exception("Runtime capacity probe failed")


def launch_runtime_capacity_probe(args, shared_state: SharedState) -> multiprocessing.Process:
    """Start the held backend probe used by autonomous cold placement."""

    shared_state.update(
        capacity_probe_state="starting",
        capacity_probe_error=None,
        capacity_probe_stop=False,
        capacity_contract_frozen=False,
    )
    process = multiprocessing.Process(
        target=run_runtime_capacity_probe,
        args=(
            args.gpu_backend,
            shared_state.dict,
            args.log_level,
            getattr(args, "execution_device", None),
        ),
    )
    process.start()
    return process


def freeze_and_stop_runtime_capacity_probe(
    shared_state: SharedState,
    process: multiprocessing.Process | None,
    *,
    join_timeout_seconds: float = 10.0,
) -> None:
    """Freeze the selected envelope and hand its runtime slot to the executor."""

    shared_state.update(capacity_contract_frozen=True, capacity_probe_stop=True)
    if process is None:
        return
    process.join(timeout=join_timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
    shared_state.set("capacity_probe_state", "frozen")
