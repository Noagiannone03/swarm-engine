from types import SimpleNamespace

import pytest

from parallax.server.memory_budget import (
    GIB,
    MemoryPressureController,
    MemoryPressureLevel,
    adaptive_system_reserve_bytes,
    calculate_cuda_memory_budget,
    calculate_mlx_memory_budget,
    configure_mlx_memory_limits,
    current_mlx_memory_budget,
)


def gb(value: float) -> int:
    return int(value * GIB)


def test_budget_uses_live_available_memory_not_total_ram():
    budget = calculate_mlx_memory_budget(
        total_bytes=gb(16),
        available_bytes=gb(10),
        active_bytes=0,
        max_working_set_bytes=gb(12),
        system_reserve_bytes=gb(2),
    )

    assert budget.process_limit_bytes == gb(8)
    assert budget.additional_bytes == gb(8)


def test_adaptive_system_reserve_tracks_pressure_without_a_fixed_six_gb_floor():
    assert adaptive_system_reserve_bytes(gb(8), gb(4)) == gb(1.25)
    assert adaptive_system_reserve_bytes(gb(8), gb(2.4)) == gb(1.5)
    assert adaptive_system_reserve_bytes(gb(8), gb(1.2)) == gb(2)
    assert adaptive_system_reserve_bytes(gb(16), gb(10)) == gb(2)
    assert adaptive_system_reserve_bytes(gb(16), gb(4.5)) == gb(2.5)
    assert adaptive_system_reserve_bytes(gb(16), gb(3)) == gb(3)
    assert adaptive_system_reserve_bytes(gb(64), gb(40)) == gb(6.4)


def test_cuda_budget_uses_global_free_vram_and_keeps_driver_reserve():
    budget = calculate_cuda_memory_budget(
        total_bytes=gb(16),
        available_bytes=gb(11),
        device_reserve_bytes=gb(1.5),
    )

    assert budget.total_bytes == gb(16)
    assert budget.available_bytes == gb(11)
    assert budget.usable_bytes == gb(9.5)


def test_cuda_budget_refuses_capacity_when_other_apps_consume_the_reserve():
    budget = calculate_cuda_memory_budget(
        total_bytes=gb(16),
        available_bytes=gb(1),
        device_reserve_bytes=gb(1.5),
    )

    assert budget.usable_bytes == 0


def test_budget_is_stable_after_its_own_allocations_reduce_available_memory():
    budget = calculate_mlx_memory_budget(
        total_bytes=gb(16),
        available_bytes=gb(7),
        active_bytes=gb(3),
        max_working_set_bytes=gb(12),
        system_reserve_bytes=gb(6),
    )

    assert budget.process_limit_bytes == gb(4)
    assert budget.additional_bytes == gb(1)


def test_budget_never_allocates_more_when_system_is_already_below_reserve():
    budget = calculate_mlx_memory_budget(
        total_bytes=gb(16),
        available_bytes=gb(3),
        active_bytes=gb(3),
        max_working_set_bytes=gb(12),
        system_reserve_bytes=gb(6),
    )

    assert budget.process_limit_bytes == gb(3)
    assert budget.additional_bytes == 0


def test_explicit_process_limit_can_only_reduce_the_adaptive_cap():
    budget = calculate_mlx_memory_budget(
        total_bytes=gb(64),
        available_bytes=gb(50),
        active_bytes=0,
        max_working_set_bytes=gb(48),
        system_reserve_bytes=gb(8),
        explicit_process_limit_bytes=gb(20),
    )

    assert budget.process_limit_bytes == gb(20)


class FakeMlx:
    def __init__(self):
        self.calls = []

    def device_info(self):
        return {
            "memory_size": gb(16),
            "max_recommended_working_set_size": gb(12),
        }

    def get_active_memory(self):
        return 0

    def set_memory_limit(self, value):
        self.calls.append(("memory", value))

    def set_wired_limit(self, value):
        self.calls.append(("wired", value))

    def set_cache_limit(self, value):
        self.calls.append(("cache", value))


def test_configure_applies_one_cap_to_memory_and_wired_limits(monkeypatch):
    monkeypatch.delenv("PARALLAX_SYSTEM_RESERVE_GB", raising=False)
    mlx = FakeMlx()
    psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=gb(16), available=gb(10)))

    budget = configure_mlx_memory_limits(mlx, psutil_module=psutil)

    assert budget.process_limit_bytes == gb(8)
    assert mlx.calls[0:2] == [("memory", gb(8)), ("wired", gb(8))]
    assert mlx.calls[2][0] == "cache"
    assert mlx.calls[2][1] <= 256 * 1024**2

    # Releasing memory after startup cannot grow this worker generation.
    mlx.get_active_memory = lambda: gb(3)
    psutil.virtual_memory = lambda: SimpleNamespace(total=gb(16), available=gb(10))
    later = current_mlx_memory_budget(
        mlx,
        psutil_module=psutil,
        process_limit_cap_bytes=budget.process_limit_bytes,
    )
    assert later.process_limit_bytes == gb(8)
    assert later.additional_bytes == gb(5)


def test_worker_generation_keeps_its_startup_reserve_tier(monkeypatch):
    """The worker's own weights must not look like new desktop pressure."""

    monkeypatch.delenv("PARALLAX_SYSTEM_RESERVE_GB", raising=False)
    mlx = FakeMlx()
    psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=gb(16), available=gb(4.5))
    )
    startup = configure_mlx_memory_limits(mlx, psutil_module=psutil)
    assert startup.system_reserve_bytes == gb(2.5)
    assert startup.process_limit_bytes == gb(2)

    # Loading 1.5 GiB makes macOS availability cross the next adaptive tier.
    # The reserve is immutable inside this generation, while the live
    # availability still bounds the remaining 0.5 GiB.
    mlx.get_active_memory = lambda: gb(1.5)
    psutil.virtual_memory = lambda: SimpleNamespace(total=gb(16), available=gb(3))
    loaded = current_mlx_memory_budget(
        mlx,
        psutil_module=psutil,
        process_limit_cap_bytes=startup.process_limit_bytes,
        system_reserve_bytes=startup.system_reserve_bytes,
    )
    assert loaded.system_reserve_bytes == gb(2.5)
    assert loaded.process_limit_bytes == gb(2)
    assert loaded.additional_bytes == gb(0.5)


def test_explicit_system_reserve_override_still_wins(monkeypatch):
    monkeypatch.setenv("PARALLAX_SYSTEM_RESERVE_GB", "6")
    mlx = FakeMlx()
    psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=gb(16), available=gb(10)))

    budget = configure_mlx_memory_limits(mlx, psutil_module=psutil)

    assert budget.system_reserve_bytes == gb(6)
    assert budget.process_limit_bytes == gb(4)


def test_configure_refuses_to_start_when_no_safe_memory_remains(monkeypatch):
    monkeypatch.setenv("PARALLAX_SYSTEM_RESERVE_GB", "10")
    mlx = FakeMlx()
    psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=gb(16), available=gb(8)))

    with pytest.raises(RuntimeError, match="No memory is safely available"):
        configure_mlx_memory_limits(mlx, psutil_module=psutil)


def test_pressure_controller_debounces_warning_and_recovers_slowly():
    controller = MemoryPressureController(
        system_reserve_bytes=gb(6),
        warning_samples=3,
        recovery_samples=4,
    )

    assert controller.observe(gb(4)).level is MemoryPressureLevel.NORMAL
    assert controller.observe(gb(5)).level is MemoryPressureLevel.NORMAL
    assert controller.observe(gb(4)).level is MemoryPressureLevel.NORMAL
    assert controller.observe(gb(4)).level is MemoryPressureLevel.NORMAL
    warning = controller.observe(gb(4))
    assert warning.level is MemoryPressureLevel.WARNING
    assert warning.changed is True

    for _ in range(3):
        assert controller.observe(gb(6)).level is MemoryPressureLevel.WARNING
    recovered = controller.observe(gb(6))
    assert recovered.level is MemoryPressureLevel.NORMAL
    assert recovered.changed is True


def test_pressure_controller_critical_is_one_way_and_drains_with_deadline():
    now = [100.0]
    controller = MemoryPressureController(
        system_reserve_bytes=gb(6),
        critical_samples=2,
        drain_timeout_seconds=30,
        clock=lambda: now[0],
    )

    controller.observe(gb(2))
    critical = controller.observe(gb(2))
    assert critical.level is MemoryPressureLevel.CRITICAL
    assert controller.should_shutdown(current_requests=1) is False

    # Even a green sample cannot make a critical generation grow/resume.
    assert controller.observe(gb(12)).level is MemoryPressureLevel.CRITICAL
    now[0] += 30
    assert controller.should_shutdown(current_requests=1) is True


def test_pressure_controller_stops_immediately_after_requests_drain():
    controller = MemoryPressureController(
        system_reserve_bytes=gb(6),
        critical_samples=1,
    )
    controller.observe(gb(1))

    assert controller.should_shutdown(current_requests=0) is True
