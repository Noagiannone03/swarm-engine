from types import SimpleNamespace

from parallax.server import server_info


def test_cuda_node_advertises_aggregate_live_capacity_for_visible_devices(monkeypatch):
    hardware = server_info.NvidiaHardwareInfo(
        total_ram_gb=32,
        chip="Test CUDA GPU",
        tflops_fp16=50,
        num_gpus=2,
        vram_gb=16,
        memory_bandwidth_gbps=600,
    )
    budgets = {
        0: SimpleNamespace(
            usable_bytes=8_000,
            available_bytes=9_500,
            device_reserve_bytes=1_500,
        ),
        1: SimpleNamespace(
            usable_bytes=7_000,
            available_bytes=8_500,
            device_reserve_bytes=1_500,
        ),
    }
    monkeypatch.setattr(server_info.HardwareInfo, "detect", lambda: hardware)
    monkeypatch.setattr(server_info.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        server_info,
        "current_cuda_memory_budget",
        lambda _torch, device: budgets[device],
    )

    info = server_info.detect_node_hardware("node-1")

    assert info["usable_memory_bytes"] == 15_000
    assert info["device_available_memory_bytes"] == 18_000
    assert info["device_reserve_bytes"] == 3_000


def test_mlx_node_advertises_only_memory_additional_to_initialized_runtime(monkeypatch):
    hardware = server_info.AppleSiliconHardwareInfo(
        total_ram_gb=16,
        chip="Apple M4",
        tflops_fp16=8.52,
        num_gpus=1,
    )
    budget = SimpleNamespace(
        additional_bytes=3_000,
        process_limit_bytes=3_750,
        available_bytes=5_000,
        system_reserve_bytes=2_000,
    )
    monkeypatch.setattr(server_info.HardwareInfo, "detect", lambda: hardware)
    monkeypatch.setattr(server_info, "current_mlx_memory_budget", lambda *_args, **_kwargs: budget)

    info = server_info.detect_node_hardware("node-1")

    assert info["usable_memory_bytes"] == 3_000
    assert info["device_process_limit_bytes"] == 3_750
