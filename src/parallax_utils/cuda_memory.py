"""CUDA memory budgeting helpers for workstation-safe inference."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CUDA_SYSTEM_RESERVE_GB = 1.5
_DEFAULT_CUDA_AVAILABLE_RESERVE_GB = 0.75
_DEFAULT_CUDA_USABLE_MEMORY_FRACTION = 0.82
_MIN_CUDA_BUDGET_GB = 1.0


@dataclass(frozen=True)
class CudaMemoryBudget:
    device_index: int
    total_gb: float
    free_gb: Optional[float]
    usable_gb: float
    allocator_fraction: float


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
    """Return this process's ENFORCED VRAM ceiling in bytes for ``device_index``.

    This is the exact limit installed by :func:`configure_torch_cuda_memory_limit`
    through ``torch.cuda.set_per_process_memory_fraction`` — i.e. the most this
    process is ever allowed to allocate on the device. Returns ``None`` if CUDA
    or PyTorch is unavailable.
    """
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
        return int(budget.allocator_fraction * total_bytes)
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

    This instead bounds the pool by ``cap − already_reserved`` (what this process
    can still allocate), then applies ``kv_cache_memory_fraction`` as the usual
    safety margin for activation spikes / fragmentation. Falls back to the
    physical-free computation when the cap can't be resolved (no behaviour change
    on uncapped single-model nodes). Returns ``None`` if torch is unavailable.
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
        reserved = int(torch_module.cuda.memory_reserved(device_index))
    except Exception:
        reserved = 0
    remaining_in_cap = max(0, cap - reserved)
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
        try:
            torch_module.cuda.set_per_process_memory_fraction(
                budget.allocator_fraction,
                device_index,
            )
            logger.info(
                "CUDA device %s budget: %.2f GB usable of %.2f GB total "
                "(free=%s GB, allocator_fraction=%.2f)",
                device_index,
                budget.usable_gb,
                budget.total_gb,
                f"{budget.free_gb:.2f}" if budget.free_gb is not None else "unknown",
                budget.allocator_fraction,
            )
        except Exception as exc:
            logger.warning("Unable to set CUDA memory fraction on device %s: %s", device_index, exc)
        budgets.append(budget)
    return budgets
