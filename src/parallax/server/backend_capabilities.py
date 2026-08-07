"""Execution-device capabilities shared by worker subsystems.

Parallax historically inferred the tensor implementation from ad-hoc checks
such as ``device.startswith("cuda")``.  That made every non-CUDA device look
like MLX and prevented maintained Torch backends such as Intel XPU from ever
reaching the executor.  This module is the single fail-closed boundary between
hardware devices, tensor runtimes, and layer-executor engines.

The registry describes code-path compatibility, not product qualification.
A release still has to package the matching dependencies and pass a live
hardware qualification before an installer advertises that accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DeviceKind(str, Enum):
    CUDA = "cuda"
    XPU = "xpu"
    MLX = "mlx"
    WINML = "winml"
    DIRECTML = "directml"
    OPENVINO = "openvino"
    QNN = "qnn"
    CPU = "cpu"


class TensorRuntime(str, Enum):
    TORCH = "torch"
    MLX = "mlx"
    NUMPY = "numpy"


@dataclass(frozen=True)
class DeviceCapability:
    kind: DeviceKind
    tensor_runtime: TensorRuntime
    executor_backends: frozenset[str]
    accelerator: bool


_DEVICE_CAPABILITIES = {
    DeviceKind.CUDA: DeviceCapability(
        kind=DeviceKind.CUDA,
        tensor_runtime=TensorRuntime.TORCH,
        executor_backends=frozenset({"sglang", "vllm"}),
        accelerator=True,
    ),
    DeviceKind.XPU: DeviceCapability(
        kind=DeviceKind.XPU,
        tensor_runtime=TensorRuntime.TORCH,
        executor_backends=frozenset({"sglang"}),
        accelerator=True,
    ),
    DeviceKind.MLX: DeviceCapability(
        kind=DeviceKind.MLX,
        tensor_runtime=TensorRuntime.MLX,
        executor_backends=frozenset({"mlx"}),
        accelerator=True,
    ),
    # ONNX Runtime execution providers exchange host tensors through NumPy.
    # They remain distinct device kinds because provider availability,
    # packaging, memory telemetry, and model compatibility differ. Executor
    # sets stay empty until the signed stage runner passes live qualification;
    # detection alone must never let a laptop advertise READY.
    DeviceKind.WINML: DeviceCapability(
        kind=DeviceKind.WINML,
        tensor_runtime=TensorRuntime.NUMPY,
        executor_backends=frozenset(),
        accelerator=True,
    ),
    DeviceKind.DIRECTML: DeviceCapability(
        kind=DeviceKind.DIRECTML,
        tensor_runtime=TensorRuntime.NUMPY,
        executor_backends=frozenset({"onnxruntime"}),
        accelerator=True,
    ),
    DeviceKind.OPENVINO: DeviceCapability(
        kind=DeviceKind.OPENVINO,
        tensor_runtime=TensorRuntime.NUMPY,
        executor_backends=frozenset(),
        accelerator=True,
    ),
    DeviceKind.QNN: DeviceCapability(
        kind=DeviceKind.QNN,
        tensor_runtime=TensorRuntime.NUMPY,
        executor_backends=frozenset(),
        accelerator=True,
    ),
    # Torch CPU tensors are valid on the wire and in common tensor utilities,
    # but Fabi does not yet claim a qualified CPU layer executor.  Keeping the
    # engine set empty prevents an unbenchmarked host from joining as READY.
    DeviceKind.CPU: DeviceCapability(
        kind=DeviceKind.CPU,
        tensor_runtime=TensorRuntime.TORCH,
        executor_backends=frozenset(),
        accelerator=False,
    ),
}


def device_kind(device: str | None) -> DeviceKind:
    """Return the normalized kind for a concrete device such as ``xpu:0``."""

    if device is None:
        raise ValueError("execution device is required")
    normalized = str(device).strip().lower()
    base, separator, index = normalized.partition(":")
    try:
        kind = DeviceKind(base)
    except ValueError as exc:
        raise ValueError(f"Unsupported execution device: {device}") from exc
    if separator and (not index.isdigit() or kind in {DeviceKind.MLX, DeviceKind.CPU}):
        raise ValueError(f"Invalid execution device: {device}")
    return kind


def capability_for_device(device: str | None) -> DeviceCapability:
    return _DEVICE_CAPABILITIES[device_kind(device)]


def tensor_runtime_for_device(device: str | None) -> TensorRuntime:
    return capability_for_device(device).tensor_runtime


def is_torch_device(device: str | None) -> bool:
    return tensor_runtime_for_device(device) is TensorRuntime.TORCH


def canonical_device_for_rank(device: str, rank: int) -> str:
    """Attach a rank only to indexed accelerator families.

    An explicit index is preserved.  This keeps a selected multi-adapter device
    stable across the executor, serializer, and backend memory probe.
    """

    kind = device_kind(device)
    normalized = str(device).strip().lower()
    if ":" in normalized or kind in {DeviceKind.MLX, DeviceKind.CPU}:
        return normalized
    if rank < 0:
        raise ValueError("device rank must be non-negative")
    return f"{kind.value}:{rank}"


def require_executor_backend(device: str, requested_backend: str) -> str:
    """Resolve and validate the executor engine for a device.

    MLX has its own executor and therefore ignores the legacy ``gpu_backend``
    default.  Other devices must explicitly map to a maintained engine.
    """

    capability = capability_for_device(device)
    backend = "mlx" if capability.kind is DeviceKind.MLX else str(requested_backend).strip().lower()
    if backend not in capability.executor_backends:
        supported = ", ".join(sorted(capability.executor_backends)) or "none qualified"
        raise ValueError(
            f"Executor backend '{backend}' is not supported on {capability.kind.value}; "
            f"supported: {supported}"
        )
    return backend


def torch_xpu_is_available(torch_module: Any) -> bool:
    xpu = getattr(torch_module, "xpu", None)
    if xpu is None:
        return False
    try:
        return bool(xpu.is_available())
    except (AttributeError, RuntimeError):
        return False


def normalize_ort_provider(provider: str) -> DeviceKind | None:
    """Map an ONNX Runtime provider/device name to a Fabi device kind.

    WinML can expose hardware through the provider-v2 device API instead of
    the legacy provider list, hence the intentionally small set of normalized
    aliases accepted here.
    """

    normalized = str(provider).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "windowsml": DeviceKind.WINML,
        "winml": DeviceKind.WINML,
        "dmlexecutionprovider": DeviceKind.DIRECTML,
        "dml": DeviceKind.DIRECTML,
        "directml": DeviceKind.DIRECTML,
        "openvinoexecutionprovider": DeviceKind.OPENVINO,
        "openvino": DeviceKind.OPENVINO,
        "qnnexecutionprovider": DeviceKind.QNN,
        "qnn": DeviceKind.QNN,
    }
    return aliases.get(normalized)


def detect_best_device(
    *,
    torch_module: Any,
    metal_available: bool,
    ort_providers: tuple[str, ...] = (),
) -> str:
    """Choose the best local device from initialized, maintained runtimes.

    CUDA and Intel XPU keep priority when their native Torch runtimes are
    actually available.  An ONNX provider is considered only when its package
    reports it; package presence alone is never treated as usable hardware.
    """

    if metal_available:
        return DeviceKind.MLX.value
    try:
        if bool(torch_module.cuda.is_available()):
            return DeviceKind.CUDA.value
    except (AttributeError, RuntimeError):
        pass
    if torch_xpu_is_available(torch_module):
        return DeviceKind.XPU.value
    normalized_providers = {
        kind for provider in ort_providers if (kind := normalize_ort_provider(provider))
    }
    for kind in (
        DeviceKind.WINML,
        DeviceKind.QNN,
        DeviceKind.OPENVINO,
        DeviceKind.DIRECTML,
    ):
        if kind in normalized_providers:
            return kind.value
    return DeviceKind.CPU.value
