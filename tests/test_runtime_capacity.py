import sys
from types import SimpleNamespace

from parallax.server import runtime_capacity
from parallax.server.executor.base_executor import BaseExecutor


class RecordingSharedState:
    def __init__(self):
        self.values = {}

    def update(self, **values):
        self.values.update(values)


def test_runtime_geometry_uses_mlx_physical_blocks():
    executor = SimpleNamespace(cache_manager=SimpleNamespace(num_gpu_blocks=123, block_size=32))

    assert BaseExecutor._runtime_kv_cache_geometry(executor) == (3936, 32)


def test_runtime_geometry_uses_sglang_token_allocator():
    executor = SimpleNamespace(
        model_runner=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(size=77777),
            page_size=64,
        )
    )

    assert BaseExecutor._runtime_kv_cache_geometry(executor) == (77777, 64)


def test_runtime_geometry_uses_vllm_shared_block_pool_without_group_division():
    executor = SimpleNamespace(
        model_runner=SimpleNamespace(
            kv_cache_config=SimpleNamespace(num_blocks=1000, kv_cache_groups=[1, 2]),
            cache_config=SimpleNamespace(block_size=16),
        )
    )

    assert BaseExecutor._runtime_kv_cache_geometry(executor) == (16000, 16)


def test_runtime_capacity_publication_uses_initialized_backend_values():
    shared_state = RecordingSharedState()
    executor = SimpleNamespace(
        cache_manager=SimpleNamespace(num_gpu_blocks=500, block_size=32),
        scheduler=SimpleNamespace(max_batch_size=6),
        shared_state=shared_state,
    )

    BaseExecutor._publish_runtime_capacity(executor)

    assert shared_state.values == {
        "kv_cache_token_capacity": 16000,
        "kv_cache_block_size": 32,
        "max_concurrent_requests": 6,
    }


def test_directml_capacity_probe_honors_explicit_device(monkeypatch):
    imported = []
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: (
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            )
        ),
    )
    monkeypatch.setattr(
        runtime_capacity.importlib,
        "import_module",
        lambda name: imported.append(name),
    )

    selected = runtime_capacity._initialize_backend_runtime(
        "onnxruntime", "directml:2"
    )

    assert selected == "directml:2"
    assert imported == ["parallax.server.executor.onnx_executor"]


def test_skippy_capacity_uses_native_backend_memory_without_estimation(monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(
        runtime_capacity,
        "_SKIPPY_CAPACITY_DEVICE",
        {
            "name": "NVIDIA RTX",
            "device_id": "CUDA0",
            "kind": "gpu",
            "memory_free": 10 * gib,
            "memory_total": 16 * gib,
            "caps": 7,
            "execution_device": "cuda:0",
        },
    )

    hardware = runtime_capacity._skippy_node_hardware("worker-1", "cuda:0")

    assert hardware["device_available_memory_bytes"] == 10 * gib
    assert hardware["device_reserve_bytes"] == 512 * 1024**2
    assert hardware["usable_memory_bytes"] == 10 * gib - 512 * 1024**2
    assert hardware["skippy_backend_device"] == "CUDA0"
    assert hardware["skippy_backend_caps"] == 7
