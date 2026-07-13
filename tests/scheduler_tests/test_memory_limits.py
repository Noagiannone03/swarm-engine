from parallax.server.server_info import _resolve_usable_memory_gb
from parallax_utils.cuda_memory import resolve_cuda_memory_budget
from parallax_utils.utils import derive_max_batch_size
from scheduling.node import Node, NodeHardwareInfo

from .test_utils import build_model_info


def test_shared_memory_budget_shrinks_under_pressure(monkeypatch):
    monkeypatch.delenv("PARALLAX_WORKER_MEMORY_GB", raising=False)
    monkeypatch.delenv("PARALLAX_SYSTEM_RESERVE_GB", raising=False)
    monkeypatch.delenv("PARALLAX_USABLE_MEMORY_FRACTION", raising=False)
    monkeypatch.delenv("PARALLAX_AVAILABLE_RESERVE_GB", raising=False)

    idle_budget = _resolve_usable_memory_gb(16, available_gb=10, recommended_gb=14)
    pressured_budget = _resolve_usable_memory_gb(16, available_gb=3, recommended_gb=14)

    assert idle_budget == 6.3
    assert pressured_budget < idle_budget
    assert pressured_budget == 4.0


def test_shared_memory_budget_backs_off_under_severe_pressure(monkeypatch):
    monkeypatch.delenv("PARALLAX_WORKER_MEMORY_GB", raising=False)
    monkeypatch.delenv("PARALLAX_SYSTEM_RESERVE_GB", raising=False)
    monkeypatch.delenv("PARALLAX_USABLE_MEMORY_FRACTION", raising=False)
    monkeypatch.delenv("PARALLAX_AVAILABLE_RESERVE_GB", raising=False)

    assert _resolve_usable_memory_gb(16, available_gb=2, recommended_gb=14) == 2.0


def test_explicit_worker_memory_override_wins(monkeypatch):
    monkeypatch.setenv("PARALLAX_WORKER_MEMORY_GB", "5")

    assert _resolve_usable_memory_gb(16, available_gb=1, recommended_gb=14) == 5.0


def test_cuda_budget_reserves_vram_for_desktop_hosts(monkeypatch):
    monkeypatch.delenv("PARALLAX_WORKER_MEMORY_GB", raising=False)
    monkeypatch.delenv("PARALLAX_CUDA_SYSTEM_RESERVE_GB", raising=False)
    monkeypatch.delenv("PARALLAX_CUDA_AVAILABLE_RESERVE_GB", raising=False)
    monkeypatch.delenv("PARALLAX_CUDA_USABLE_MEMORY_FRACTION", raising=False)
    monkeypatch.delenv("PARALLAX_CUDA_ALLOCATOR_FRACTION", raising=False)

    budget = resolve_cuda_memory_budget(total_gb=24, free_gb=22)

    assert budget.usable_gb == 19.68
    assert round(budget.allocator_fraction, 2) == 0.82


def test_cuda_budget_backs_off_when_vram_is_already_used(monkeypatch):
    monkeypatch.delenv("PARALLAX_WORKER_MEMORY_GB", raising=False)
    monkeypatch.delenv("PARALLAX_CUDA_SYSTEM_RESERVE_GB", raising=False)
    monkeypatch.delenv("PARALLAX_CUDA_AVAILABLE_RESERVE_GB", raising=False)
    monkeypatch.delenv("PARALLAX_CUDA_USABLE_MEMORY_FRACTION", raising=False)
    monkeypatch.delenv("PARALLAX_CUDA_ALLOCATOR_FRACTION", raising=False)

    budget = resolve_cuda_memory_budget(total_gb=24, free_gb=8)

    assert budget.usable_gb == 7.25
    assert round(budget.allocator_fraction, 2) == 0.30


def test_explicit_worker_memory_override_also_controls_cuda(monkeypatch):
    monkeypatch.setenv("PARALLAX_WORKER_MEMORY_GB", "10")
    monkeypatch.delenv("PARALLAX_CUDA_ALLOCATOR_FRACTION", raising=False)

    budget = resolve_cuda_memory_budget(total_gb=24, free_gb=4)

    assert budget.usable_gb == 10
    assert round(budget.allocator_fraction, 2) == 0.42


def test_derive_max_batch_size_treats_zero_cache_as_capacity_limit():
    assert (
        derive_max_batch_size(
            requested_max_batch_size=8,
            max_sequence_len=32768,
            max_tokens_in_cache=0,
        )
        == 1
    )


def test_node_max_requests_is_clamped_by_kv_capacity():
    model = build_model_info(36)
    hardware = NodeHardwareInfo(
        node_id="tiny",
        num_gpus=1,
        tflops_fp16=8.0,
        gpu_name="tiny",
        memory_gb=1.0,
        memory_bandwidth_gbps=100.0,
        device="mlx",
    )
    node = Node(
        node_id="tiny",
        hardware=hardware,
        model_info=model,
        max_concurrent_requests=8,
        max_sequence_length=32768,
        kvcache_mem_ratio=0.25,
    )
    node.set_layer_allocation(0, 36)

    assert node.max_requests == 1


def test_live_zero_kv_headroom_is_a_hard_context_limit():
    model = build_model_info(12)
    hardware = NodeHardwareInfo(
        node_id="busy",
        num_gpus=1,
        tflops_fp16=8.0,
        gpu_name="busy",
        memory_gb=24.0,
        memory_bandwidth_gbps=100.0,
        device="cuda",
    )
    node = Node(
        node_id="busy",
        hardware=hardware,
        model_info=model,
        max_sequence_length=65536,
        reported_kv_free_tokens=0,
    )
    node.set_layer_allocation(0, 12)

    assert node.max_context_tokens == 0
