"""Memory budgets for inference workers that share RAM with the desktop.

Apple silicon uses one physical pool for the OS, applications, model weights,
Metal buffers, and the KV cache.  Basing allocations on total RAM (or only on
MLX's recommended working set) can therefore make an otherwise healthy worker
thrash the user's machine.  These helpers keep one explicit system reserve and
also account for memory that is actually available when the worker starts.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

GIB = 1024**3
MIB = 1024**2
DEFAULT_SYSTEM_RESERVE_GB = 6.0
DEFAULT_MLX_CACHE_LIMIT_MB = 256.0
DEFAULT_PRESSURE_POLL_SECONDS = 1.0
DEFAULT_PRESSURE_DRAIN_SECONDS = 30.0


@dataclass(frozen=True)
class MlxMemoryBudget:
    total_bytes: int
    available_bytes: int
    active_bytes: int
    system_reserve_bytes: int
    max_working_set_bytes: int
    process_limit_bytes: int
    additional_bytes: int
    cache_limit_bytes: int


class MemoryPressureLevel(str, Enum):
    """Stable worker states derived from noisy system-memory samples."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class MemoryPressureObservation:
    level: MemoryPressureLevel
    changed: bool
    available_bytes: int


class MemoryPressureController:
    """Debounce system pressure without continuously resizing a live worker.

    Sampling is intentionally faster than acting.  A warning only pauses new
    admission; it never changes the layer contract.  Sustained critical
    pressure requests one drain-and-restart, after which startup computes a new
    immutable envelope from the then-current availability.  There is no
    automatic upward resize in a worker generation.
    """

    def __init__(
        self,
        *,
        system_reserve_bytes: int,
        warning_fraction: float = 0.75,
        critical_fraction: float = 0.40,
        warning_samples: int = 3,
        critical_samples: int = 2,
        recovery_samples: int = 15,
        drain_timeout_seconds: float = DEFAULT_PRESSURE_DRAIN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        reserve = max(0, int(system_reserve_bytes))
        if not 0 <= critical_fraction < warning_fraction <= 1:
            raise ValueError("pressure fractions must satisfy 0 <= critical < warning <= 1")
        if min(warning_samples, critical_samples, recovery_samples) <= 0:
            raise ValueError("pressure sample counts must be positive")
        self.system_reserve_bytes = reserve
        self.warning_threshold_bytes = int(reserve * warning_fraction)
        self.critical_threshold_bytes = int(reserve * critical_fraction)
        self.recovery_threshold_bytes = reserve
        self.warning_samples = int(warning_samples)
        self.critical_samples = int(critical_samples)
        self.recovery_samples = int(recovery_samples)
        self.drain_timeout_seconds = max(0.0, float(drain_timeout_seconds))
        self._clock = clock
        self.level = MemoryPressureLevel.NORMAL
        self._warning_count = 0
        self._critical_count = 0
        self._recovery_count = 0
        self._critical_since: Optional[float] = None

    def observe(self, available_bytes: int) -> MemoryPressureObservation:
        available = max(0, int(available_bytes))
        previous = self.level

        if self.level is MemoryPressureLevel.NORMAL:
            self._critical_count = (
                self._critical_count + 1 if available < self.critical_threshold_bytes else 0
            )
            self._warning_count = (
                self._warning_count + 1 if available < self.warning_threshold_bytes else 0
            )
            if self._critical_count >= self.critical_samples:
                self._enter_critical()
            elif self._warning_count >= self.warning_samples:
                self.level = MemoryPressureLevel.WARNING
                self._recovery_count = 0
        elif self.level is MemoryPressureLevel.WARNING:
            self._critical_count = (
                self._critical_count + 1 if available < self.critical_threshold_bytes else 0
            )
            self._recovery_count = (
                self._recovery_count + 1 if available >= self.recovery_threshold_bytes else 0
            )
            if self._critical_count >= self.critical_samples:
                self._enter_critical()
            elif self._recovery_count >= self.recovery_samples:
                self.level = MemoryPressureLevel.NORMAL
                self._warning_count = 0
                self._critical_count = 0
                self._recovery_count = 0

        return MemoryPressureObservation(
            level=self.level,
            changed=self.level is not previous,
            available_bytes=available,
        )

    def _enter_critical(self) -> None:
        self.level = MemoryPressureLevel.CRITICAL
        self._critical_since = self._clock()

    def should_shutdown(self, current_requests: int) -> bool:
        """Return true once drained, or after the bounded drain deadline."""

        if self.level is not MemoryPressureLevel.CRITICAL:
            return False
        if int(current_requests) <= 0:
            return True
        assert self._critical_since is not None
        return self._clock() - self._critical_since >= self.drain_timeout_seconds


def _positive_env_bytes(name: str, unit: int) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, raw)
        return None
    if value < 0:
        logger.warning("Ignoring negative %s=%r", name, raw)
        return None
    return int(value * unit)


def configured_system_reserve_bytes(total_bytes: int) -> int:
    """Return the RAM kept away from MLX for the OS and foreground apps."""

    configured = _positive_env_bytes("PARALLAX_SYSTEM_RESERVE_GB", GIB)
    reserve = configured if configured is not None else int(DEFAULT_SYSTEM_RESERVE_GB * GIB)
    return min(max(0, reserve), max(0, int(total_bytes)))


def calculate_mlx_memory_budget(
    *,
    total_bytes: int,
    available_bytes: int,
    active_bytes: int,
    max_working_set_bytes: int,
    system_reserve_bytes: int,
    explicit_process_limit_bytes: Optional[int] = None,
    cache_limit_bytes: int = int(DEFAULT_MLX_CACHE_LIMIT_MB * MIB),
) -> MlxMemoryBudget:
    """Calculate a hard MLX process cap from current system availability.

    ``available_bytes`` is the amount the OS says can be allocated without
    swapping.  ``active_bytes`` is added back because MLX allocations already
    made by this process reduce that value.  This keeps repeated calculations
    stable while still reacting to memory consumed by other applications.
    """

    total = max(0, int(total_bytes))
    available = min(total, max(0, int(available_bytes)))
    active = max(0, int(active_bytes))
    working_set = max(0, int(max_working_set_bytes))
    reserve = min(total, max(0, int(system_reserve_bytes)))

    safe_total_cap = max(0, total - reserve)
    pressure_cap = active + max(0, available - reserve)
    caps = [safe_total_cap, pressure_cap]
    if working_set > 0:
        caps.append(working_set)
    if explicit_process_limit_bytes is not None:
        caps.append(max(0, int(explicit_process_limit_bytes)))

    process_limit = max(active, min(caps))
    additional = max(0, min(process_limit - active, available - reserve))
    cache_limit = min(max(0, int(cache_limit_bytes)), max(0, process_limit // 20))
    return MlxMemoryBudget(
        total_bytes=total,
        available_bytes=available,
        active_bytes=active,
        system_reserve_bytes=reserve,
        max_working_set_bytes=working_set,
        process_limit_bytes=process_limit,
        additional_bytes=additional,
        cache_limit_bytes=cache_limit,
    )


def current_mlx_memory_budget(
    mx, *, psutil_module=None, process_limit_cap_bytes: Optional[int] = None
) -> MlxMemoryBudget:
    """Read live MLX/system counters and return the current safe budget."""

    if psutil_module is None:
        import psutil as psutil_module  # type: ignore[no-redef]

    memory = psutil_module.virtual_memory()
    device_info = mx.device_info()
    total = int(getattr(memory, "total", 0) or device_info.get("memory_size", 0))
    available = int(getattr(memory, "available", 0))
    active = int(mx.get_active_memory())
    working_set = int(device_info.get("max_recommended_working_set_size", 0))
    explicit_limit = _positive_env_bytes("PARALLAX_MLX_MEMORY_LIMIT_GB", GIB)
    if process_limit_cap_bytes is not None:
        explicit_limit = (
            max(0, int(process_limit_cap_bytes))
            if explicit_limit is None
            else min(explicit_limit, max(0, int(process_limit_cap_bytes)))
        )
    configured_cache = _positive_env_bytes("PARALLAX_MLX_CACHE_LIMIT_MB", MIB)
    return calculate_mlx_memory_budget(
        total_bytes=total,
        available_bytes=available,
        active_bytes=active,
        max_working_set_bytes=working_set,
        system_reserve_bytes=configured_system_reserve_bytes(total),
        explicit_process_limit_bytes=explicit_limit,
        cache_limit_bytes=(
            configured_cache
            if configured_cache is not None
            else int(DEFAULT_MLX_CACHE_LIMIT_MB * MIB)
        ),
    )


def configure_mlx_memory_limits(mx, *, psutil_module=None) -> MlxMemoryBudget:
    """Apply the safe process/wired/cache limits supported by official MLX."""

    budget = current_mlx_memory_budget(mx, psutil_module=psutil_module)
    if budget.process_limit_bytes <= 0:
        raise RuntimeError(
            "No memory is safely available for MLX after preserving the system reserve"
        )

    mx.set_memory_limit(budget.process_limit_bytes)
    mx.set_wired_limit(budget.process_limit_bytes)
    mx.set_cache_limit(budget.cache_limit_bytes)
    logger.info(
        "MLX memory budget: limit=%.2f GB, additional=%.2f GB, available=%.2f GB, "
        "system_reserve=%.2f GB, cache_limit=%.0f MB",
        budget.process_limit_bytes / GIB,
        budget.additional_bytes / GIB,
        budget.available_bytes / GIB,
        budget.system_reserve_bytes / GIB,
        budget.cache_limit_bytes / MIB,
    )
    return budget
