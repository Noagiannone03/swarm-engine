"""CUDA memory budgeting helpers for workstation-safe inference."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CUDA_SYSTEM_RESERVE_GB = 1.5
_DEFAULT_CUDA_AVAILABLE_RESERVE_GB = 0.75
_DEFAULT_CUDA_USABLE_MEMORY_FRACTION = 0.82
_MIN_CUDA_BUDGET_GB = 1.0
_FABI_WINDOWS_CUDA_VERSION = "12.6"

# A process must keep using the admission budget measured before it loads model
# weights. Recomputing from live free VRAM later counts this process's own model
# allocations as external pressure and shrinks the KV budget a second time.
_cuda_process_budget_bytes: dict[int, int] = {}


@dataclass(frozen=True)
class CudaMemoryBudget:
    device_index: int
    total_gb: float
    free_gb: Optional[float]
    usable_gb: float
    allocator_fraction: float


def configure_windows_cuda_environment(
    *,
    environ: Optional[dict[str, str]] = None,
    platform: Optional[str] = None,
    program_files: Optional[str] = None,
) -> Optional[str]:
    """Expose an installed CUDA toolkit to Windows JIT backends.

    vLLM-Windows discovers CUDA through the standard ``CUDA_PATH`` / ``CUDA_HOME`` /
    ``CUDA_ROOT`` variables. FlashInfer additionally requires ``CUDA_LIB_PATH``.
    NVIDIA's installer creates a versioned toolkit directory, but processes that
    were already running (notably Fabi) do not inherit its new environment. Resolve
    the official install layout once before importing either backend and populate
    all four names consistently.
    """

    if (platform or sys.platform) != "win32":
        return None
    env = environ if environ is not None else os.environ
    candidates = [
        env.get(name, "").strip()
        for name in ("CUDA_LIB_PATH", "CUDA_PATH", "CUDA_HOME", "CUDA_ROOT")
    ]
    root = Path(
        program_files
        or env.get("ProgramFiles", "").strip()
        or r"C:\Program Files"
    ) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    requested = env.get("PARALLAX_CUDA_TOOLKIT_VERSION", _FABI_WINDOWS_CUDA_VERSION).strip()
    candidates.append(str(root / f"v{requested}"))
    if root.is_dir():
        candidates.extend(str(path) for path in sorted(root.glob("v*"), reverse=True))

    cuda_root = next(
        (
            Path(candidate).resolve()
            for candidate in candidates
            if candidate
            and (Path(candidate) / "bin").is_dir()
            and (Path(candidate) / "include").is_dir()
        ),
        None,
    )
    if cuda_root is None:
        return None

    resolved = str(cuda_root)
    for name in ("CUDA_LIB_PATH", "CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        env[name] = resolved
    cuda_bin = str(cuda_root / "bin")
    path_entries = env.get("PATH", "").split(os.pathsep)
    if cuda_bin.casefold() not in {entry.casefold() for entry in path_entries if entry}:
        env["PATH"] = os.pathsep.join([cuda_bin, *path_entries])
    logger.info("Using CUDA toolkit at %s", resolved)
    return resolved


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


def resolve_cuda_memory_budget(
    *,
    device_index: int = 0,
    total_gb: float,
    free_gb: Optional[float] = None,
) -> CudaMemoryBudget:
    """Return the VRAM budget Parallax should advertise and enforce.

    The scheduler spends most of reported ``memory_gb`` on weights and KV cache.
    Reporting raw VRAM on a gaming/workstation Windows or Linux box can evict
    the desktop, browser, or other GPU workloads. Keep headroom by default, with
    explicit environment overrides for dedicated inference nodes.
    """

    explicit = os.environ.get("PARALLAX_WORKER_MEMORY_GB", "").strip()
    if explicit:
        usable_gb = _read_env_float(
            "PARALLAX_WORKER_MEMORY_GB",
            default=max(_MIN_CUDA_BUDGET_GB, total_gb * 0.5),
            minimum=_MIN_CUDA_BUDGET_GB,
            maximum=max(_MIN_CUDA_BUDGET_GB, total_gb),
        )
    else:
        reserve_gb = _read_env_float(
            "PARALLAX_CUDA_SYSTEM_RESERVE_GB",
            _DEFAULT_CUDA_SYSTEM_RESERVE_GB,
            minimum=0.0,
            maximum=max(0.0, total_gb - _MIN_CUDA_BUDGET_GB),
        )
        usable_fraction = _read_env_float(
            "PARALLAX_CUDA_USABLE_MEMORY_FRACTION",
            _DEFAULT_CUDA_USABLE_MEMORY_FRACTION,
            minimum=0.05,
            maximum=1.0,
        )
        candidates = [
            total_gb - reserve_gb,
            total_gb * usable_fraction,
        ]
        if free_gb is not None and free_gb > 0:
            available_reserve_gb = _read_env_float(
                "PARALLAX_CUDA_AVAILABLE_RESERVE_GB",
                _DEFAULT_CUDA_AVAILABLE_RESERVE_GB,
                minimum=0.0,
            )
            candidates.append(free_gb - available_reserve_gb)
        usable_gb = min(candidates)

    usable_gb = round(max(_MIN_CUDA_BUDGET_GB, min(total_gb, usable_gb)), 2)
    allocator_fraction = _read_env_float(
        "PARALLAX_CUDA_ALLOCATOR_FRACTION",
        usable_gb / max(total_gb, 0.01),
        minimum=0.05,
        maximum=1.0,
    )
    allocator_fraction = min(1.0, max(0.05, allocator_fraction))
    return CudaMemoryBudget(
        device_index=device_index,
        total_gb=round(total_gb, 2),
        free_gb=round(free_gb, 2) if free_gb is not None else None,
        usable_gb=usable_gb,
        allocator_fraction=allocator_fraction,
    )


def cuda_allocator_budget_bytes(device_index: int = 0, torch_module=None) -> Optional[int]:
    """Return this process's stable VRAM admission budget for ``device_index``.

    :func:`configure_torch_cuda_memory_limit` snapshots the budget before model
    loading and optionally enforces it through PyTorch's allocator. Keeping the
    snapshot is essential: live free VRAM later includes this process's own model
    and workspace allocations. Returns ``None`` if CUDA or PyTorch is unavailable.
    """

    cached = _cuda_process_budget_bytes.get(device_index)
    if cached is not None:
        return cached
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception:
            return None
    try:
        if not torch_module.cuda.is_available():
            return None
        props = torch_module.cuda.get_device_properties(device_index)
        total_bytes = int(props.total_memory)
        try:
            free_bytes, _ = torch_module.cuda.mem_get_info(device_index)
            free_gb = float(free_bytes) / (1024**3)
        except Exception:
            free_gb = None
        budget = resolve_cuda_memory_budget(
            device_index=device_index,
            total_gb=total_bytes / (1024**3),
            free_gb=free_gb,
        )
        return int(budget.usable_gb * 1024**3)
    except Exception:
        return None


def available_kv_cache_bytes(
    device_index: int,
    kv_cache_memory_fraction: float,
    torch_module=None,
) -> Optional[int]:
    """Bytes available for the KV-cache pool, RESPECTING the per-process cap.

    vLLM normally sizes the KV pool from *physical* free VRAM
    (``torch.cuda.mem_get_info``). On a node whose allocator is capped below the
    card's real VRAM (the common case here: a pipeline stage sharing a GPU, or a
    workstation-safe budget), that over-counts free memory and vLLM requests a KV
    pool larger than the cap admits → CUDA OOM the moment it touches it.

    This instead bounds the pool by ``budget − actually_allocated`` (what this
    process can still use), then applies ``kv_cache_memory_fraction`` as the usual
    safety margin for activation spikes / fragmentation. PyTorch's *reserved*
    bytes deliberately include reusable allocator cache and must not be counted
    as permanently occupied; doing so applies the safety margin twice and can
    reject contexts that fit in the worker's advertised budget. Falls back to the
    physical-free computation when the budget can't be resolved. Returns ``None``
    if torch is unavailable.
    """
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception:
            return None
    try:
        physical_free, _ = torch_module.cuda.mem_get_info(device_index)
    except Exception:
        return None

    cap = cuda_allocator_budget_bytes(device_index, torch_module)
    if cap is None:
        return int(physical_free * kv_cache_memory_fraction)
    try:
        allocated = int(torch_module.cuda.memory_allocated(device_index))
    except Exception:
        allocated = 0
    remaining_in_cap = max(0, cap - allocated)
    headroom = min(int(physical_free), remaining_in_cap)
    return int(headroom * kv_cache_memory_fraction)


def configure_torch_cuda_memory_limit(torch_module=None) -> list[CudaMemoryBudget]:
    """Apply PyTorch CUDA allocator limits for all visible devices.

    Returns the budgets that were applied. If CUDA is unavailable or PyTorch is
    not installed, returns an empty list.
    """

    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception:
            return []

    try:
        if not torch_module.cuda.is_available():
            return []
    except Exception:
        return []

    budgets: list[CudaMemoryBudget] = []
    for device_index in range(torch_module.cuda.device_count()):
        props = torch_module.cuda.get_device_properties(device_index)
        total_gb = float(props.total_memory) / (1024**3)
        free_gb: Optional[float]
        try:
            free_bytes, _total_bytes = torch_module.cuda.mem_get_info(device_index)
            free_gb = float(free_bytes) / (1024**3)
        except Exception:
            free_gb = None

        budget = resolve_cuda_memory_budget(
            device_index=device_index,
            total_gb=total_gb,
            free_gb=free_gb,
        )
        # We always COMPUTE and report this budget so the scheduler can size layer
        # counts against it. Whether we also HARD-ENFORCE it via
        # set_per_process_memory_fraction is opt-in (PARALLAX_ENFORCE_CUDA_CAP).
        #
        # Upstream parallax never installs a per-process cap: the GPU backend
        # (sglang's mem_fraction_static / vLLM's gpu_memory_utilization) sizes its
        # own KV pool from the device's free memory. A hard torch cap is invisible
        # to those backends — they still target the full card, then OOM the moment
        # the pool crosses the cap. So the cap is for genuinely SHARED machines
        # (a workstation also running a desktop/browser); on a dedicated inference
        # node it only fights the backend. Default: report, don't enforce.
        enforce = os.environ.get("PARALLAX_ENFORCE_CUDA_CAP", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if enforce:
            try:
                torch_module.cuda.set_per_process_memory_fraction(
                    budget.allocator_fraction,
                    device_index,
                )
            except Exception as exc:
                logger.warning(
                    "Unable to set CUDA memory fraction on device %s: %s", device_index, exc
                )
        usable_bytes = int(budget.usable_gb * 1024**3)
        if enforce:
            usable_bytes = min(
                usable_bytes,
                int(budget.allocator_fraction * props.total_memory),
            )
        _cuda_process_budget_bytes.setdefault(device_index, usable_bytes)
        logger.info(
            "CUDA device %s budget: %.2f GB usable of %.2f GB total "
            "(free=%s GB, allocator_fraction=%.2f, enforced=%s)",
            device_index,
            budget.usable_gb,
            budget.total_gb,
            f"{budget.free_gb:.2f}" if budget.free_gb is not None else "unknown",
            budget.allocator_fraction,
            enforce,
        )
        budgets.append(budget)
    return budgets
