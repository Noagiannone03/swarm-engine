"""
ServerInfo that will be announce to DHT and used for client's routing.
    HardwareInfo: Detects and summarizes hardware information, RAM and FLOPs
We haven't used other info, will wait until DHT implemented.
"""

import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Dict, Optional

from parallax_utils.cuda_memory import (
    configure_torch_cuda_memory_limit,
    resolve_cuda_memory_budget,
)

try:
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_reduce
    from mlx_lm.tuner.utils import get_total_parameters
except Exception:  # pragma: no cover - MLX is optional on non-Apple hosts
    mx = None
    nn = None
    tree_reduce = None
    get_total_parameters = None

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some envs
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


# Default headroom kept for the OS, GUI, and other userland processes on
# machines with unified memory (Apple silicon) or CPU-only Linux nodes.
# macOS memory pressure depends on swap/wired/cache, not only free RAM; a
# fixed reserve plus an "available now" check keeps personal machines usable.
_DEFAULT_SYSTEM_RESERVE_GB = 6.0
_DEFAULT_AVAILABLE_RESERVE_GB = 2.0

# Cap on the fraction of physical memory we ever report as usable, even after
# subtracting the reserve. Activations and transient buffers grow beyond what
# the scheduler's param/kvcache ratios account for.
_DEFAULT_USABLE_MEMORY_FRACTION = 0.45

# MLX should not wire every byte we report as schedulable. This limit is
# applied to the Metal working-set limit, leaving room for Python, tokenizers,
# networking, UI apps, and transient allocations.
_DEFAULT_MLX_WIRED_MEMORY_FRACTION = 0.9


def _read_env_float(
    key: str,
    default: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r (not a number)", key, raw)
        return default
    if minimum is not None and value < minimum:
        logger.warning("%s=%s clamped to %s minimum", key, value, minimum)
        return minimum
    if maximum is not None and value > maximum:
        logger.warning("%s=%s clamped to %s maximum", key, value, maximum)
        return maximum
    return value


def _dynamic_system_reserve_gb(total_gb: float) -> float:
    """Return a sane reserve for interactive machines.

    The env value remains the default, but the reserve scales up on bigger
    Apple Silicon machines where users are likely to run browsers/IDEs
    alongside Fabi. Dedicated nodes can set PARALLAX_SYSTEM_RESERVE_GB lower.
    """
    env_default = _DEFAULT_SYSTEM_RESERVE_GB
    if total_gb <= 18:
        env_default = 5.0
    elif total_gb <= 36:
        env_default = 7.0
    elif total_gb <= 72:
        env_default = 10.0
    else:
        env_default = 14.0
    return _read_env_float("PARALLAX_SYSTEM_RESERVE_GB", env_default, minimum=0.0)


def _minimum_interactive_budget_gb(total_gb: float, available_gb: Optional[float]) -> float:
    """Lower bound for useful contribution on personal shared-memory hosts.

    The scheduler interprets memory_gb as the whole worker budget, then spends
    only param_mem_ratio + kvcache_mem_ratio of it. Reporting 1 GB makes a
    connected Mac effectively useless and can prevent pipeline formation. Keep
    a small but useful floor on 16 GB Macs, while still backing off if the OS is
    already under severe memory pressure.
    """
    ratio = (available_gb / total_gb) if available_gb is not None and total_gb > 0 else 1.0
    if ratio < 0.15:
        return 2.0
    if total_gb <= 18:
        return 4.0
    if total_gb <= 36:
        return 6.0
    if total_gb <= 72:
        return 8.0
    return 10.0


def _memory_pressure_cap_gb(total_gb: float, available_gb: Optional[float]) -> Optional[float]:
    """Estimate a cap from currently available memory.

    psutil's available memory is the best portable signal we have from Python:
    on macOS it accounts for reclaimable memory better than "free" memory.
    When the machine is already under pressure we shrink the advertised budget
    aggressively instead of letting the worker push the OS into swap.
    """
    if available_gb is None:
        return None

    reserve = _read_env_float(
        "PARALLAX_AVAILABLE_RESERVE_GB",
        _DEFAULT_AVAILABLE_RESERVE_GB,
        minimum=0.0,
    )
    ratio = available_gb / total_gb if total_gb > 0 else 0.0
    cap = available_gb - reserve

    # Approximate Activity Monitor's yellow/red pressure behavior: as
    # available memory shrinks, preserve a larger portion for the OS.
    if ratio < 0.15:
        cap *= 0.35
    elif ratio < 0.25:
        cap *= 0.55
    elif ratio < 0.40:
        cap *= 0.75

    return max(_minimum_interactive_budget_gb(total_gb, available_gb), cap)


def _resolve_usable_memory_gb(
    total_gb: float,
    *,
    available_gb: Optional[float] = None,
    recommended_gb: Optional[float] = None,
) -> float:
    """Convert a raw "physical memory" reading into what we should report
    to the scheduler as memory_gb on shared-memory hosts.

    The scheduler currently treats memory_gb as if the worker owned the
    machine: it allocates param_mem_ratio + kvcache_mem_ratio (defaults
    0.65 + 0.25 = 0.9) of that figure to model weights and KV cache. On a
    laptop or developer workstation that assumption causes the worker to
    swap out the OS, freeze the desktop, and trip the OOM killer. We
    adjust by:

      1. Honoring PARALLAX_WORKER_MEMORY_GB when explicitly set
      2. Subtracting a reserve for the OS (PARALLAX_SYSTEM_RESERVE_GB)
      3. Capping by a usable fraction (PARALLAX_USABLE_MEMORY_FRACTION)
      4. Capping by current memory pressure when psutil is available, with
         a useful floor for interactive shared-memory Macs
      5. Capping by Metal's recommended working set when available
      6. Clamping to a 1 GB floor so we never report nonsense

    Operators dedicating a machine to Parallax should set the reserve to
    a small value (e.g. 1) and bump the fraction towards 1.0.
    """
    explicit = os.environ.get("PARALLAX_WORKER_MEMORY_GB", "").strip()
    if explicit:
        return _read_env_float(
            "PARALLAX_WORKER_MEMORY_GB",
            default=max(1.0, total_gb * 0.5),
            minimum=1.0,
            maximum=max(1.0, total_gb),
        )

    reserve_gb = _dynamic_system_reserve_gb(total_gb)
    fraction = _read_env_float(
        "PARALLAX_USABLE_MEMORY_FRACTION",
        _DEFAULT_USABLE_MEMORY_FRACTION,
        minimum=0.05,
        maximum=1.0,
    )
    candidates = [
        total_gb - reserve_gb,
        total_gb * fraction,
    ]
    pressure_cap = _memory_pressure_cap_gb(total_gb, available_gb)
    if pressure_cap is not None:
        candidates.append(pressure_cap)
    if recommended_gb is not None and recommended_gb > 0:
        candidates.append(recommended_gb * fraction)

    usable = min(candidates)
    return round(max(1.0, usable), 2)


def _recommended_metal_memory_gb() -> Optional[float]:
    if mx is None:
        return None
    try:
        info = mx.metal.device_info()
        value = info.get("max_recommended_working_set_size")
        if isinstance(value, (int, float)) and value > 0:
            return float(value) / 2**30
    except Exception:
        return None
    return None


def resolve_mlx_wired_limit_bytes(total_gb: Optional[float] = None) -> int:
    """Return the MLX wired-memory limit in bytes.

    MLX docs expose set_wired_limit() for macOS and recommend keeping it below
    total memory. We set it to the same conservative budget advertised to the
    scheduler, optionally multiplied by PARALLAX_MLX_WIRED_MEMORY_FRACTION.
    """
    if total_gb is None:
        if psutil:
            vm = psutil.virtual_memory()
            total_gb = vm.total / 2**30
            available_gb = vm.available / 2**30
        else:
            total_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
            available_gb = None
    else:
        available_gb = None

    explicit = os.environ.get("PARALLAX_MLX_WIRED_LIMIT_GB", "").strip()
    if explicit:
        limit_gb = _read_env_float(
            "PARALLAX_MLX_WIRED_LIMIT_GB",
            default=max(1.0, total_gb * 0.5),
            minimum=1.0,
            maximum=max(1.0, total_gb - 0.5),
        )
    else:
        usable_gb = _resolve_usable_memory_gb(
            total_gb,
            available_gb=available_gb,
            recommended_gb=_recommended_metal_memory_gb(),
        )
        fraction = _read_env_float(
            "PARALLAX_MLX_WIRED_MEMORY_FRACTION",
            _DEFAULT_MLX_WIRED_MEMORY_FRACTION,
            minimum=0.1,
            maximum=1.0,
        )
        limit_gb = usable_gb * fraction

    # MLX requires a wired limit strictly smaller than total memory.
    limit_gb = min(limit_gb, max(1.0, total_gb - 0.5))
    return int(max(1.0, limit_gb) * 2**30)


def resolve_mlx_budget_bytes(total_gb: Optional[float] = None) -> int:
    """Memory budget to use for MLX cache sizing."""
    if total_gb is None and psutil:
        total_gb = psutil.virtual_memory().total / 2**30
    return resolve_mlx_wired_limit_bytes(total_gb)


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
    def detect() -> "HardwareInfo":
        """Dispatch to the correct subclass for the current machine.

        Prefers CUDA when available; falls back to Apple silicon on macOS.
        """
        if torch is not None and torch.cuda.is_available():
            return NvidiaHardwareInfo.detect()
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
        "M4": 8.52,
        "M4 Pro": 17.04,
        "M4 Max": 34.08,
    }

    @classmethod
    def detect(cls) -> "AppleSiliconHardwareInfo":
        if psutil:
            vm = psutil.virtual_memory()
            physical_gb = vm.total / 2**30
            available_gb = vm.available / 2**30
        else:
            physical_gb = (
                int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
            )
            available_gb = None

        # Apple silicon shares one pool of RAM between the CPU, the GPU,
        # the OS, and every userland process. Reporting the raw physical
        # total to the scheduler causes it to allocate weights + KV cache
        # at ~90% of total memory, which on a personal Mac (browser +
        # IDE + system services) reliably swaps the machine to a freeze.
        # _resolve_usable_memory_gb() applies a configurable headroom.
        total_gb = _resolve_usable_memory_gb(
            physical_gb,
            available_gb=available_gb,
            recommended_gb=_recommended_metal_memory_gb(),
        )
        if total_gb < physical_gb:
            logger.info(
                "Apple silicon: reporting %.1f GB usable out of %.1f GB physical "
                "(available=%s GB; set PARALLAX_WORKER_MEMORY_GB, "
                "PARALLAX_SYSTEM_RESERVE_GB, or PARALLAX_USABLE_MEMORY_FRACTION to tune)",
                total_gb,
                physical_gb,
                f"{available_gb:.1f}" if available_gb is not None else "unknown",
            )

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
    # Worker-ENFORCED budget for weights+KV (bytes, summed across GPUs) and the
    # raw device total. ``usable_vram_bytes`` is the same ceiling the worker
    # installs via torch.cuda.set_per_process_memory_fraction, so the scheduler
    # can size layer counts against the exact limit the allocator will admit.
    usable_vram_bytes: float = 0.0
    total_vram_bytes: float = 0.0

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

        budgets = configure_torch_cuda_memory_limit(torch)
        device_count = torch.cuda.device_count()
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        name = getattr(props, "name", f"cuda:{device_index}")
        total_vram_gb = round(props.total_memory / (1024**3), 1)
        free_vram_gb = None
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
            free_vram_gb = free_bytes / (1024**3)
        except Exception:
            pass
        if budgets:
            current_budget = next(
                (b for b in budgets if b.device_index == device_index),
                budgets[0],
            )
        else:
            current_budget = resolve_cuda_memory_budget(
                device_index=device_index,
                total_gb=total_vram_gb,
                free_gb=free_vram_gb,
            )

        # Host RAM (for completeness)
        if psutil:
            total_gb = psutil.virtual_memory().total / 2**30
        else:
            total_gb = 0.0

        spec = cls._match_gpu_specs(name, total_vram_gb)
        if current_budget.usable_gb < total_vram_gb:
            logger.info(
                "CUDA: reporting %.1f GB usable out of %.1f GB VRAM on %s "
                "(set PARALLAX_WORKER_MEMORY_GB or PARALLAX_CUDA_USABLE_MEMORY_FRACTION to tune)",
                current_budget.usable_gb,
                total_vram_gb,
                name,
            )
        return cls(
            num_gpus=device_count,
            total_ram_gb=round(total_gb, 1),
            chip=name,
            tflops_fp16=float(spec["tflops_fp16"]),
            vram_gb=current_budget.usable_gb,
            memory_bandwidth_gbps=float(spec["bandwidth_gbps"]),
            # Summed across GPUs: the scheduler treats memory as a single budget.
            # Assumes homogeneous devices (the common multi-GPU case); the
            # current-device budget is representative.
            usable_vram_bytes=current_budget.usable_gb * device_count * 1024**3,
            total_vram_bytes=current_budget.total_gb * device_count * 1024**3,
        )


_HARDWARE_CACHE: Optional[Dict[str, Any]] = None


def detect_node_hardware(node_id: Optional[str]) -> Dict[str, Any]:
    """Detect local hardware and return a dict for scheduling (cached).

    Computed ONCE and cached: hardware doesn't change over a process's life, and
    the CUDA budget (usable_gb / usable_memory_bytes) must stay STABLE across
    heartbeats. Recomputing each heartbeat reads the *instantaneous* free VRAM,
    which legitimately drops once our own weights load — that made the advertised
    budget shrink, the scheduler think the node got smaller (re-allocation
    churn), and the memory governor misfire. The enforced per-process cap is set
    at startup, so a cold-start snapshot is the correct, stable figure.
    """
    global _HARDWARE_CACHE
    if _HARDWARE_CACHE is not None:
        cached = dict(_HARDWARE_CACHE)
        cached["node_id"] = node_id
        return cached
    _HARDWARE_CACHE = _detect_node_hardware_uncached(node_id)
    cached = dict(_HARDWARE_CACHE)
    cached["node_id"] = node_id
    return cached


def _detect_node_hardware_uncached(node_id: Optional[str]) -> Dict[str, Any]:
    """Detect local hardware and return a dict for scheduling.

    Returns a dictionary with keys compatible with `NodeHardwareInfo` builder:
    - node_id: The peer/node id
    - tflops_fp16: Peak FP16 TFLOPS
    - memory_gb: Device memory in GB (VRAM for CUDA, total RAM for Apple)
    - memory_bandwidth_gbps: Estimated memory bandwidth in GB/s
    """
    try:
        hw = HardwareInfo.detect()
    except NotImplementedError:
        # Fallback for hosts the dispatcher doesn't recognize (typically
        # Linux without CUDA). Upstream hardcoded 16 GB here, which lied
        # to the scheduler on machines with less RAM and produced OOM
        # kills the operator could not diagnose. Use psutil when
        # available and apply the usable-memory adjustment so the
        # scheduler never asks for more than the box can actually give.
        if psutil:
            vm = psutil.virtual_memory()
            physical_gb = vm.total / 2**30
            available_gb = vm.available / 2**30
            memory_gb = _resolve_usable_memory_gb(
                physical_gb,
                available_gb=available_gb,
            )
            logger.info(
                "Unknown hardware fallback: reporting %.1f GB usable out of %.1f GB physical "
                "(available=%.1f GB)",
                memory_gb,
                physical_gb,
                available_gb,
            )
        else:
            memory_gb = 8.0
            logger.warning(
                "Unknown hardware and psutil unavailable; reporting %.1f GB by default. "
                "Install psutil or set PARALLAX_SYSTEM_RESERVE_GB / "
                "PARALLAX_USABLE_MEMORY_FRACTION to control allocation.",
                memory_gb,
            )
        return {
            "node_id": node_id,
            "num_gpus": 1,
            "tflops_fp16": 50.0,
            "gpu_name": "Unknown",
            "memory_gb": memory_gb,
            "memory_bandwidth_gbps": 100.0,
            "device": "Unknown",
            # No GPU: the usable budget IS the (already overhead-adjusted) RAM.
            "usable_memory_bytes": memory_gb * 1024**3,
            "total_memory_bytes": memory_gb * 1024**3,
        }

    if isinstance(hw, NvidiaHardwareInfo):
        return {
            "node_id": node_id,
            "num_gpus": hw.num_gpus,
            "tflops_fp16": hw.tflops_fp16,
            "gpu_name": hw.chip,
            "memory_gb": hw.vram_gb,
            "memory_bandwidth_gbps": hw.memory_bandwidth_gbps,
            "device": "cuda",
            # The same per-process VRAM ceiling the worker enforces at load.
            "usable_memory_bytes": hw.usable_vram_bytes,
            "total_memory_bytes": hw.total_vram_bytes,
        }
    if isinstance(hw, AppleSiliconHardwareInfo):
        # Use unified memory size as memory_gb; bandwidth rough estimate per family
        est_bandwidth = 100.0
        # The MLX wired-memory limit is the budget the worker actually enforces
        # for model + KV — sized below total RAM to leave room for the OS/apps.
        try:
            usable_bytes = float(resolve_mlx_wired_limit_bytes(hw.total_ram_gb))
        except Exception:
            usable_bytes = hw.total_ram_gb * 1024**3
        return {
            "node_id": node_id,
            "num_gpus": hw.num_gpus,
            "tflops_fp16": hw.tflops_fp16,
            "gpu_name": hw.chip,
            "memory_gb": hw.total_ram_gb,
            "memory_bandwidth_gbps": est_bandwidth,
            "device": "mlx",
            "usable_memory_bytes": usable_bytes,
            "total_memory_bytes": hw.total_ram_gb * 1024**3,
        }
    # Generic fallback
    return {
        "node_id": node_id,
        "num_gpus": hw.num_gpus,
        "tflops_fp16": hw.tflops_fp16,
        "gpu_name": "Unknown",
        "memory_gb": 16.0,
        "memory_bandwidth_gbps": 100.0,
        "device": "Unknown",
        "usable_memory_bytes": 16.0 * 1024**3,
        "total_memory_bytes": 16.0 * 1024**3,
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
        cls, sharded_model_instance: Any  # Instance of your ShardedModel
    ) -> "ShardedModelInfo":
        """
        Constructs ShardedModelInfo from a loaded ShardedModel instance.
        Assumes sharded_model_instance has start_layer, end_layer, and model_id_original attributes.
        """
        if mx is None or tree_reduce is None or get_total_parameters is None:
            raise RuntimeError("MLX is required to inspect sharded model memory")

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
