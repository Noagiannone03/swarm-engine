import sys
from types import ModuleType, SimpleNamespace

from parallax.server import server_info


def _fake_directml_ep(*, provider_id=0, dxgi_index=3):
    device = SimpleNamespace(
        vendor_id=0x8086,
        device_id=0x1234,
        metadata={
            "Description": "Intel Test Graphics",
            "DxgiAdapterNumber": str(dxgi_index),
            "Discrete": "0",
        },
    )
    return SimpleNamespace(
        ep_name="DmlExecutionProvider",
        ep_options={"device_id": str(provider_id)},
        device=device,
    )


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
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
    monkeypatch.setattr(server_info, "torch", fake_torch)
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
    fake_mlx = ModuleType("mlx")
    fake_mlx_core = ModuleType("mlx.core")
    fake_mlx.core = fake_mlx_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)
    monkeypatch.setattr(server_info, "current_mlx_memory_budget", lambda *_args, **_kwargs: budget)

    info = server_info.detect_node_hardware("node-1")

    assert info["usable_memory_bytes"] == 3_000
    assert info["device_process_limit_bytes"] == 3_750


def test_xpu_node_advertises_aggregate_live_capacity_without_invented_performance(monkeypatch):
    hardware = server_info.IntelXpuHardwareInfo(
        total_ram_gb=32,
        chip="Intel Arc Test",
        tflops_fp16=0,
        num_gpus=2,
        device_memory_gb=8,
        memory_bandwidth_gbps=0,
    )
    budgets = {
        0: SimpleNamespace(
            usable_bytes=5_000,
            available_bytes=5_500,
            device_reserve_bytes=500,
        ),
        1: SimpleNamespace(
            usable_bytes=4_000,
            available_bytes=4_500,
            device_reserve_bytes=500,
        ),
    }
    monkeypatch.setattr(server_info.HardwareInfo, "detect", lambda: hardware)
    fake_torch = SimpleNamespace(xpu=SimpleNamespace(device_count=lambda: 2))
    monkeypatch.setattr(server_info, "torch", fake_torch)
    monkeypatch.setattr(
        server_info,
        "current_xpu_memory_budget",
        lambda _torch, device: budgets[device],
    )

    info = server_info.detect_node_hardware("node-xpu")

    assert info["device"] == "xpu"
    assert info["usable_memory_bytes"] == 9_000
    assert info["device_available_memory_bytes"] == 10_000
    assert info["tflops_fp16"] == 0
    assert info["memory_bandwidth_gbps"] == 0


def test_directml_provider_id_resolves_exact_dxgi_adapter():
    selected, adapter_index = server_info._directml_ep_device(
        2,
        SimpleNamespace(
            get_ep_devices=lambda: (
                _fake_directml_ep(provider_id=0, dxgi_index=1),
                _fake_directml_ep(provider_id=2, dxgi_index=7),
            )
        ),
    )

    assert selected.ep_options["device_id"] == "2"
    assert adapter_index == 7


def test_directml_node_uses_dxgi_budget_and_host_cap_for_uma(monkeypatch):
    hardware = server_info.DirectMlHardwareInfo(
        total_ram_gb=16,
        chip="Intel Test Graphics",
        tflops_fp16=0,
        num_gpus=1,
        execution_device_index=2,
        dxgi_adapter_index=7,
        vendor_id=0x8086,
        device_id=0x1234,
        dedicated_video_memory_bytes=0,
        shared_system_memory_bytes=8_000,
        unified_memory=True,
        device_memory_gb=8,
    )
    dxgi = SimpleNamespace(
        vendor_id=0x8086,
        device_id=0x1234,
        local_budget=7_000,
        local_current_usage=1_000,
        non_local_budget=2_000,
        non_local_current_usage=500,
    )
    monkeypatch.setattr(server_info.HardwareInfo, "detect", lambda _device=None: hardware)
    monkeypatch.setattr(server_info, "_query_dxgi_video_memory", lambda _index: dxgi)
    monkeypatch.setattr(
        server_info,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=16_000, available=4_000)),
    )
    monkeypatch.setattr(server_info, "configured_directml_reserve_bytes", lambda _total: 500)

    info = server_info.detect_node_hardware("node-dml", "directml:2")

    assert info["device"] == "directml:2"
    assert info["dxgi_adapter_index"] == 7
    assert info["device_available_memory_bytes"] == 4_000
    assert info["usable_memory_bytes"] == 3_500
    assert info["dxgi_local_budget_bytes"] == 7_000
    assert info["dxgi_non_local_budget_bytes"] == 2_000
    assert info["tflops_fp16"] == 0


def test_unsupported_hardware_never_receives_invented_capacity(monkeypatch):
    def unsupported(_device=None):
        raise NotImplementedError

    monkeypatch.setattr(server_info.HardwareInfo, "detect", unsupported)

    info = server_info.detect_node_hardware("node-unknown", "openvino:0")

    assert info["num_gpus"] == 0
    assert info["usable_memory_bytes"] == 0
    assert info["memory_gb"] == 0
