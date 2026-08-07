"""
ServerInfo that will be announce to DHT and used for client's routing.
    HardwareInfo: Detects and summarizes hardware information, RAM and FLOPs
We haven't used other info, will wait until DHT implemented.
"""

import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

from parallax.server.backend_capabilities import DeviceKind, device_kind, torch_xpu_is_available
from parallax.server.memory_budget import (
    calculate_directml_memory_budget,
    configured_directml_reserve_bytes,
    current_cuda_memory_budget,
    current_mlx_memory_budget,
    current_xpu_memory_budget,
)

if TYPE_CHECKING:
    from mlx import nn

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some envs
    torch = None

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class HardwareInfo:
    """Generic hardware summary for a peer."""

    total_ram_gb: float
    chip: str
    tflops_fp16: float
    num_gpus: int

    def dumps(self) -> Dict[str, Any]:
        """Serializes the HardwareInfo object to a dictionary."""
        return asdict(self)

    @classmethod
    def loads(cls, obj: Dict[str, Any]) -> "HardwareInfo":
        """Deserializes a dictionary into a HardwareInfo object."""
        return cls(**obj)

    @staticmethod
    def detect(execution_device: str | None = None) -> "HardwareInfo":
        """Dispatch to the correct subclass for the current machine.

        Prefers CUDA when available; falls back to Apple silicon on macOS.
        """
        requested_kind = device_kind(execution_device) if execution_device else None
        if requested_kind is DeviceKind.DIRECTML:
            return DirectMlHardwareInfo.detect(_execution_device_index(execution_device))
        if requested_kind is DeviceKind.CUDA:
            return NvidiaHardwareInfo.detect()
        if requested_kind is DeviceKind.XPU:
            return IntelXpuHardwareInfo.detect()
        if requested_kind is DeviceKind.MLX:
            return AppleSiliconHardwareInfo.detect()
        if requested_kind is not None:
            raise NotImplementedError(
                f"No qualified hardware probe for {requested_kind.value}"
            )
        if torch is not None and torch.cuda.is_available():
            return NvidiaHardwareInfo.detect()
        if torch is not None and torch_xpu_is_available(torch):
            return IntelXpuHardwareInfo.detect()
        if platform.system() == "Darwin" and platform.machine().startswith("arm"):
            return AppleSiliconHardwareInfo.detect()
        raise NotImplementedError("Unsupported hardware; add a subclass.")


@dataclass
class AppleSiliconHardwareInfo(HardwareInfo):
    """HardwareInfo specialised for Apple silicon (M-series)."""

    # From cpu-monkey.com
    _APPLE_PEAK_FP16: ClassVar[Dict[str, float]] = {
        "M1": 4.58,
        "M1 Pro": 10.6,
        "M1 Max": 21.2,
        "M2": 7.1,
        "M2 Pro": 11.36,
        "M2 Max": 26.98,
        "M2 Ultra": 53.96,
        "M3": 7.1,
        "M3 Pro": 9.94,
        "M3 Max": 28.4,
        "M3 Ultra": 57.34,
        "M4": 8.52,
        "M4 Pro": 17.04,
        "M4 Max": 34.08,
        "M5": 9.37,
        "M5 Pro": 18.74,
        "M5 Max": 37.49,
        "M5 Ultra": 74.98,
    }

    @classmethod
    def detect(cls) -> "AppleSiliconHardwareInfo":
        if psutil:
            total_gb = psutil.virtual_memory().total / 2**30
        else:
            total_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30

        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()

        short_name = chip.rsplit("Apple ", maxsplit=1)[-1]
        # For github action, we need to remove the "(Virtual)" suffix
        if short_name.endswith(" (Virtual)"):
            short_name = short_name.rsplit(" (Virtual)", maxsplit=1)[0]
        try:
            flops = cls._APPLE_PEAK_FP16[short_name]
        except KeyError as e:
            raise RuntimeError(
                f"Unknown Apple silicon chip '{short_name}' detected. "
                "Please add it to the _APPLE_PEAK_FP16 dictionary."
            ) from e

        return cls(num_gpus=1, total_ram_gb=round(total_gb, 1), chip=chip, tflops_fp16=flops)


@dataclass
class NvidiaHardwareInfo(HardwareInfo):
    """HardwareInfo specialised for NVIDIA CUDA devices.

    Captures peak FP16 TFLOPS and memory bandwidth using a best-effort mapping
    from device name. VRAM is reported via CUDA device properties.
    """

    vram_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0

    # Best-effort device database; can be extended as needed
    _GPU_DB: ClassVar[Dict[str, Dict[str, float]]] = {
        # key: substring to match in CUDA device name (case-insensitive)
        "a100-80g": {"tflops_fp16": 312.0, "bandwidth_gbps": 2039.0},
        "a100 80": {"tflops_fp16": 312.0, "bandwidth_gbps": 2039.0},
        "a100-40g": {"tflops_fp16": 312.0, "bandwidth_gbps": 1935.0},
        "a100 40": {"tflops_fp16": 312.0, "bandwidth_gbps": 1935.0},
        "rtx 5090": {"tflops_fp16": 104.8, "bandwidth_gbps": 1792.0},
        "rtx 4090": {"tflops_fp16": 82.6, "bandwidth_gbps": 1008.0},
    }

    @classmethod
    def _match_gpu_specs(cls, name: str, vram_gb: float) -> Dict[str, float]:
        key = name.lower()
        # Specialize A100 by VRAM size when name is generic
        if "a100" in key and ("80" in key or vram_gb >= 60):
            return cls._GPU_DB.get("a100-80g", {"tflops_fp16": 312.0, "bandwidth_gbps": 2039.0})
        if "a100" in key and ("40" in key or vram_gb < 60):
            return cls._GPU_DB.get("a100-40g", {"tflops_fp16": 312.0, "bandwidth_gbps": 1935.0})
        for sub, spec in cls._GPU_DB.items():
            if sub in key:
                return spec
        # Conservative fallback when unknown
        return {"tflops_fp16": 50.0, "bandwidth_gbps": 600.0}

    @classmethod
    def detect(cls) -> "NvidiaHardwareInfo":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; cannot detect NVIDIA hardware")

        device_count = torch.cuda.device_count()
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        name = getattr(props, "name", f"cuda:{device_index}")
        total_vram_gb = round(props.total_memory / (1024**3), 1)

        # Host RAM (for completeness)
        if psutil:
            total_gb = psutil.virtual_memory().total / 2**30
        else:
            total_gb = 0.0

        spec = cls._match_gpu_specs(name, total_vram_gb)
        return cls(
            num_gpus=device_count,
            total_ram_gb=round(total_gb, 1),
            chip=name,
            tflops_fp16=float(spec["tflops_fp16"]),
            vram_gb=total_vram_gb,
            memory_bandwidth_gbps=float(spec["bandwidth_gbps"]),
        )


@dataclass
class IntelXpuHardwareInfo(HardwareInfo):
    """Hardware summary backed only by PyTorch's maintained XPU APIs."""

    device_memory_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0

    @classmethod
    def detect(cls) -> "IntelXpuHardwareInfo":
        if torch is None or not torch_xpu_is_available(torch):
            raise RuntimeError("Intel XPU not available")
        device_count = int(torch.xpu.device_count())
        device_index = int(torch.xpu.current_device())
        props = torch.xpu.get_device_properties(device_index)
        name = getattr(props, "name", f"xpu:{device_index}")
        total_memory = int(getattr(props, "total_memory", 0))
        if total_memory <= 0:
            _, total_memory = torch.xpu.mem_get_info(device_index)
        host_total = psutil.virtual_memory().total / 2**30 if psutil else 0.0
        # PyTorch does not expose a portable peak-FP16 or bandwidth contract for
        # Arc/Core Ultra. Leave telemetry unknown instead of inventing a SKU
        # estimate; V3 routing learns execution throughput from live leases.
        return cls(
            num_gpus=device_count,
            total_ram_gb=round(host_total, 1),
            chip=str(name),
            tflops_fp16=0.0,
            device_memory_gb=round(total_memory / 2**30, 1),
            memory_bandwidth_gbps=0.0,
        )


def _execution_device_index(execution_device: str | None) -> int:
    normalized = str(execution_device or "").strip().lower()
    _, separator, raw_index = normalized.partition(":")
    return int(raw_index) if separator else 0


def _directml_ep_device(device_index: int, ort_module=None):
    """Resolve a DirectML provider id to its authoritative DXGI adapter id."""

    if ort_module is None:
        try:
            import onnxruntime as ort_module
        except (ImportError, OSError) as exc:
            raise RuntimeError("ONNX Runtime DirectML is not installed") from exc
    try:
        devices = tuple(ort_module.get_ep_devices())
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "ONNX Runtime does not expose execution-provider device metadata"
        ) from exc
    matches = []
    for candidate in devices:
        if str(getattr(candidate, "ep_name", "")) != "DmlExecutionProvider":
            continue
        options = dict(getattr(candidate, "ep_options", {}) or {})
        try:
            provider_device_id = int(options["device_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if provider_device_id == int(device_index):
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(
            f"DirectML device {device_index} has no unique ONNX Runtime metadata entry"
        )
    metadata = dict(getattr(matches[0].device, "metadata", {}) or {})
    try:
        dxgi_adapter_index = int(metadata["DxgiAdapterNumber"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("DirectML device metadata has no DXGI adapter number") from exc
    return matches[0], dxgi_adapter_index


def _query_dxgi_video_memory(adapter_index: int):
    try:
        import fabi_network_native
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "the qualified Fabi native wheel is required for DirectML memory telemetry"
        ) from exc
    try:
        return fabi_network_native.query_dxgi_video_memory(int(adapter_index))
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError("DXGI video-memory budget query failed") from exc


@dataclass
class DirectMlHardwareInfo(HardwareInfo):
    """DirectML adapter identity backed by ORT metadata and live DXGI APIs."""

    execution_device_index: int = 0
    dxgi_adapter_index: int = 0
    vendor_id: int = 0
    device_id: int = 0
    dedicated_video_memory_bytes: int = 0
    dedicated_system_memory_bytes: int = 0
    shared_system_memory_bytes: int = 0
    unified_memory: bool = False
    device_memory_gb: float = 0.0

    @classmethod
    def detect(cls, execution_device_index: int = 0) -> "DirectMlHardwareInfo":
        ep_device, dxgi_adapter_index = _directml_ep_device(execution_device_index)
        dxgi = _query_dxgi_video_memory(dxgi_adapter_index)
        hardware = ep_device.device
        metadata = dict(getattr(hardware, "metadata", {}) or {})
        if int(getattr(hardware, "vendor_id", -1)) != int(dxgi.vendor_id):
            raise RuntimeError("ONNX Runtime and DXGI disagree on DirectML vendor id")
        if int(getattr(hardware, "device_id", -1)) != int(dxgi.device_id):
            raise RuntimeError("ONNX Runtime and DXGI disagree on DirectML device id")
        ort_description = str(metadata.get("Description", "")).strip()
        if ort_description and ort_description != str(dxgi.description).strip():
            raise RuntimeError("ONNX Runtime and DXGI disagree on DirectML adapter")
        host_total = int(psutil.virtual_memory().total) if psutil else 0
        dedicated = int(dxgi.dedicated_video_memory)
        shared = int(dxgi.shared_system_memory)
        unified = dedicated <= 0
        device_total = dedicated if dedicated > 0 else int(dxgi.local_budget)
        return cls(
            total_ram_gb=round(host_total / 2**30, 1),
            chip=str(dxgi.description),
            # No maintained cross-vendor API reports a comparable peak FP16
            # value. V3 routing learns measured throughput from live leases.
            tflops_fp16=0.0,
            num_gpus=1,
            execution_device_index=int(execution_device_index),
            dxgi_adapter_index=int(dxgi_adapter_index),
            vendor_id=int(dxgi.vendor_id),
            device_id=int(dxgi.device_id),
            dedicated_video_memory_bytes=dedicated,
            dedicated_system_memory_bytes=int(dxgi.dedicated_system_memory),
            shared_system_memory_bytes=shared,
            unified_memory=unified,
            device_memory_gb=round(device_total / 2**30, 1),
        )


def detect_node_hardware(
    node_id: Optional[str], execution_device: str | None = None
) -> Dict[str, Any]:
    """Detect local hardware and return a dict for scheduling.

    Returns a dictionary with keys compatible with `NodeHardwareInfo` builder:
    - node_id: The peer/node id
    - tflops_fp16: Peak FP16 TFLOPS
    - memory_gb: Device memory in GB (VRAM for CUDA, total RAM for Apple)
    - memory_bandwidth_gbps: Estimated memory bandwidth in GB/s
    """
    try:
        hw = (
            HardwareInfo.detect()
            if execution_device is None
            else HardwareInfo.detect(execution_device)
        )
    except NotImplementedError:
        # Unsupported runtimes must not gain an invented accelerator or memory
        # envelope. They remain visible but cannot enter autonomous placement.
        return {
            "node_id": node_id,
            "num_gpus": 0,
            "tflops_fp16": 0.0,
            "gpu_name": "Unsupported",
            "memory_gb": 0.0,
            "usable_memory_bytes": 0,
            "memory_bandwidth_gbps": 0.0,
            "device": execution_device or "unsupported",
        }

    if isinstance(hw, NvidiaHardwareInfo):
        usable_memory_bytes = None
        device_available_memory_bytes = None
        device_reserve_bytes = None
        try:
            budgets = [
                current_cuda_memory_budget(torch, device)
                for device in range(torch.cuda.device_count())
            ]
            usable_memory_bytes = sum(budget.usable_bytes for budget in budgets)
            device_available_memory_bytes = sum(budget.available_bytes for budget in budgets)
            device_reserve_bytes = sum(budget.device_reserve_bytes for budget in budgets)
        except Exception:
            # Older CUDA/PyTorch environments remain compatible, but qualified
            # workers expose cudaMemGetInfo-backed capacity.
            pass
        return {
            "node_id": node_id,
            "num_gpus": hw.num_gpus,
            "tflops_fp16": hw.tflops_fp16,
            "gpu_name": hw.chip,
            "memory_gb": hw.vram_gb,
            "usable_memory_bytes": usable_memory_bytes,
            "device_available_memory_bytes": device_available_memory_bytes,
            "device_reserve_bytes": device_reserve_bytes,
            "memory_bandwidth_gbps": hw.memory_bandwidth_gbps,
            "device": "cuda",
        }
    if isinstance(hw, IntelXpuHardwareInfo):
        budgets = [
            current_xpu_memory_budget(torch, device)
            for device in range(torch.xpu.device_count())
        ]
        return {
            "node_id": node_id,
            "num_gpus": hw.num_gpus,
            "tflops_fp16": hw.tflops_fp16,
            "gpu_name": hw.chip,
            "memory_gb": hw.device_memory_gb,
            "usable_memory_bytes": sum(budget.usable_bytes for budget in budgets),
            "device_available_memory_bytes": sum(
                budget.available_bytes for budget in budgets
            ),
            "device_reserve_bytes": sum(
                budget.device_reserve_bytes for budget in budgets
            ),
            "memory_bandwidth_gbps": hw.memory_bandwidth_gbps,
            "device": "xpu",
        }
    if isinstance(hw, AppleSiliconHardwareInfo):
        # Use unified memory size as memory_gb; bandwidth rough estimate per family
        est_bandwidth = 100.0
        usable_memory_bytes = None
        system_available_memory_bytes = None
        system_reserve_bytes = None
        try:
            import mlx.core as mx

            budget = current_mlx_memory_budget(mx, psutil_module=psutil)
            # Placement accounts model weights and KV, not allocations already
            # owned by the initialized runtime. Publishing the process limit
            # would count that baseline twice when the executor sizes its KV.
            usable_memory_bytes = budget.additional_bytes
            system_available_memory_bytes = budget.available_bytes
            system_reserve_bytes = budget.system_reserve_bytes
        except Exception:
            # Detection remains best-effort for environments without Metal. A
            # real Apple worker has MLX and will publish the pressure-aware cap.
            pass
        return {
            "node_id": node_id,
            "num_gpus": hw.num_gpus,
            "tflops_fp16": hw.tflops_fp16,
            "gpu_name": hw.chip,
            "memory_gb": hw.total_ram_gb,
            "usable_memory_bytes": usable_memory_bytes,
            "system_available_memory_bytes": system_available_memory_bytes,
            "system_reserve_bytes": system_reserve_bytes,
            "device_process_limit_bytes": (
                None if usable_memory_bytes is None else budget.process_limit_bytes
            ),
            "memory_bandwidth_gbps": est_bandwidth,
            "device": "mlx",
        }
    if isinstance(hw, DirectMlHardwareInfo):
        dxgi = _query_dxgi_video_memory(hw.dxgi_adapter_index)
        if int(dxgi.vendor_id) != hw.vendor_id or int(dxgi.device_id) != hw.device_id:
            raise RuntimeError("DirectML adapter changed during capacity sampling")
        host_available = (
            int(psutil.virtual_memory().available)
            if hw.unified_memory and psutil
            else None
        )
        total = (
            hw.dedicated_video_memory_bytes
            if not hw.unified_memory
            else int(dxgi.local_budget)
        )
        budget = calculate_directml_memory_budget(
            total_bytes=total,
            local_budget_bytes=int(dxgi.local_budget),
            local_current_usage_bytes=int(dxgi.local_current_usage),
            device_reserve_bytes=configured_directml_reserve_bytes(total),
            unified_memory=hw.unified_memory,
            host_available_bytes=host_available,
        )
        return {
            "node_id": node_id,
            "num_gpus": 1,
            "tflops_fp16": 0.0,
            "gpu_name": hw.chip,
            "memory_gb": hw.device_memory_gb,
            "usable_memory_bytes": budget.usable_bytes,
            "device_available_memory_bytes": budget.available_bytes,
            "device_reserve_bytes": budget.device_reserve_bytes,
            "dxgi_local_budget_bytes": int(dxgi.local_budget),
            "dxgi_local_current_usage_bytes": int(dxgi.local_current_usage),
            "dxgi_non_local_budget_bytes": int(dxgi.non_local_budget),
            "dxgi_non_local_current_usage_bytes": int(dxgi.non_local_current_usage),
            "dxgi_adapter_index": hw.dxgi_adapter_index,
            "directml_unified_memory": hw.unified_memory,
            "memory_bandwidth_gbps": 0.0,
            "device": f"directml:{hw.execution_device_index}",
        }
    # Generic fallback
    return {
        "node_id": node_id,
        "num_gpus": hw.num_gpus,
        "tflops_fp16": hw.tflops_fp16,
        "gpu_name": hw.chip,
        "memory_gb": hw.total_ram_gb,
        "usable_memory_bytes": 0,
        "memory_bandwidth_gbps": 0.0,
        "device": execution_device or "unsupported",
    }


@dataclass
class ShardedModelInfo:
    """
    Detailed information about the specific model shard hosted by a server.
    """

    model_name: str
    start_layer: int
    end_layer: int
    parameter_count: int = 0
    memory_consumption_mb: float = 0.0

    def dumps(self) -> Dict[str, Any]:
        """Serializes the HardwareInfo object to a dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def loads(cls, data: Dict[str, Any]) -> "ShardedModelInfo":
        """Deserializes a dictionary into a HardwareInfo object."""
        return cls(**data)

    @classmethod
    def from_sharded_model(
        cls,
        sharded_model_instance: "nn.Module",  # Instance of your ShardedModel
    ) -> "ShardedModelInfo":
        """
        Constructs ShardedModelInfo from a loaded ShardedModel instance.
        Assumes sharded_model_instance has start_layer, end_layer, and model_id_original attributes.
        """
        # These dependencies only exist in the MLX runtime. Keeping them local
        # lets scheduler and CUDA workers import hardware metadata without
        # installing an emulated MLX stack.
        import mlx.core as mx
        from mlx.utils import tree_reduce
        from mlx_lm.tuner.utils import get_total_parameters

        # Calculate parameter count
        count = get_total_parameters(sharded_model_instance)

        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
            sharded_model_instance.parameters(),
            0,
        )
        memory_mb = round(model_bytes / (1024 * 1024), 2)

        return cls(
            model_name=sharded_model_instance.model_id,  # Use the cleaned name
            start_layer=sharded_model_instance.start_layer,
            end_layer=sharded_model_instance.end_layer,
            parameter_count=count,
            memory_consumption_mb=memory_mb,
        )
