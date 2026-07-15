import pytest

from parallax.server.capacity_profile import (
    build_capacity_profile_from_tensor_sizes,
    calibrate_profile_from_runtime,
    constrain_profile_to_memory,
    profile_allows_range,
)
from scheduling.layer_allocation import DynamicProgrammingLayerAllocator, GreedyLayerAllocator
from scheduling.model_info import ModelInfo
from scheduling.node import Node, NodeHardwareInfo
from scheduling.node_management import NodeState
from scheduling.scheduler import Scheduler

from .test_utils import build_node_management


def _model(*, tied: bool = False) -> ModelInfo:
    return ModelInfo(
        model_name="test/model",
        mlx_model_name="test/model",
        head_size=2,
        hidden_dim=4,
        intermediate_dim=8,
        num_attention_heads=1,
        num_kv_heads=1,
        vocab_size=10,
        num_layers=3,
        cache_bytes_per_element=2,
        tie_embedding=tied,
    )


def test_worker_profile_preserves_non_uniform_layer_and_endpoint_costs():
    profile = build_capacity_profile_from_tensor_sizes(
        model_name="test/model",
        tensor_sizes={
            "model.embed_tokens.weight": 50,
            "model.layers.0.weight": 100,
            "model.layers.1.weight": 200,
            "model.layers.2.weight": 300,
            "model.norm.weight": 10,
            "lm_head.weight": 40,
        },
        model_info=_model(),
        usable_memory_bytes=500,
        target_context_tokens=10,
        runtime_reserve_bytes=0,
        backend="cuda",
        generated_at=123.0,
    )

    # KV geometry is 8 bytes/token/layer, hence 80 bytes/layer at 10 tokens.
    # The exact (non-averaged) tensors make each start position different.
    assert profile["layer_weight_bytes"] == [100, 200, 300]
    assert profile["max_end_by_start"] == [1, 2, 3]
    assert profile_allows_range(profile, 2, 3)
    assert not profile_allows_range(profile, 0, 2)


def test_tied_embedding_is_not_double_counted_on_single_worker_pipeline():
    profile = build_capacity_profile_from_tensor_sizes(
        model_name="test/tied",
        tensor_sizes={
            "model.embed_tokens.weight": 50,
            "model.layers.0.weight": 10,
            "model.layers.1.weight": 10,
            "model.layers.2.weight": 10,
            "model.norm.weight": 10,
        },
        model_info=_model(tied=True),
        usable_memory_bytes=340,
        target_context_tokens=10,
        runtime_reserve_bytes=0,
        backend="cuda",
    )

    # 50 tied embedding + 30 layers + 10 norm + 240 KV = 330 bytes.
    assert profile["max_end_by_start"][0] == 3


def test_profile_rejects_unclassified_architecture_specific_weights():
    with pytest.raises(ValueError, match="unsupported non-layer tensors"):
        build_capacity_profile_from_tensor_sizes(
            model_name="test/unknown",
            tensor_sizes={
                "model.layers.0.weight": 10,
                "model.layers.1.weight": 10,
                "model.layers.2.weight": 10,
                "model.some_new_global_tensor": 10,
            },
            model_info=_model(),
            usable_memory_bytes=1_000,
            target_context_tokens=10,
            runtime_reserve_bytes=0,
            backend="cuda",
        )


def _contract(model: ModelInfo, max_ends: list[int]) -> dict:
    return {
        "protocol_version": 1,
        "model_name": model.model_name,
        "backend": "cuda",
        "state": "provisional",
        "num_layers": model.num_layers,
        "target_context_tokens": 4096,
        "max_end_by_start": max_ends,
        "http_frontend": {
            "available": True,
            "protocol": "vllm-engine-core-v1",
            "implementation": "test",
        },
    }


def _contract_node(node_id: str, model: ModelInfo, max_ends: list[int]) -> Node:
    hardware = NodeHardwareInfo(node_id, 1, 10.0, "test", 8.0, 100.0, "cuda")
    return Node(
        node_id=node_id,
        hardware=hardware,
        model_info=model,
        capacity_protocol_version=1,
        capacity_profile=_contract(model, max_ends),
    )


def test_protocol_worker_has_zero_capacity_until_contract_is_valid():
    model = _model()
    node = _contract_node("worker", model, [1, 2, 3])
    node.capacity_profile = None
    assert node.get_decoder_layer_capacity() == 0
    assert node.max_end_layer(0) == 0

    node.capacity_profile = _contract(model, [1, 2, 3])
    assert node.get_decoder_layer_capacity() == 1
    assert node.can_host_range(1, 2)
    assert not node.can_host_range(1, 3)


def test_dp_allocator_consumes_start_dependent_worker_ranges():
    model = _model()
    first = _contract_node("first", model, [1, 1, 2])
    tail = _contract_node("tail", model, [0, 3, 3])
    manager = build_node_management([first, tail])
    allocator = DynamicProgrammingLayerAllocator(
        model_info=model,
        node_management=manager,
        dynamic_pipelines_router=True,
    )

    assert allocator.allocate_from_standby()
    assert (first.start_layer, first.end_layer) == (0, 1)
    assert (tail.start_layer, tail.end_layer) == (1, 3)


@pytest.mark.parametrize(
    "allocator_type", [DynamicProgrammingLayerAllocator, GreedyLayerAllocator]
)
def test_contract_worker_without_frontend_cannot_own_layer_zero(allocator_type):
    model = _model()
    cuda = _contract_node("windows", model, [3, 3, 3])
    cuda.capacity_profile["http_frontend"] = {
        "available": False,
        "protocol": "vllm-engine-core-v1",
    }
    unix_head = _contract_node("mac", model, [1, 2, 3])
    manager = build_node_management([cuda, unix_head])
    allocator = allocator_type(
        model_info=model,
        node_management=manager,
        dynamic_pipelines_router=True,
    )

    assert allocator.allocate_from_standby()
    assert (unix_head.start_layer, unix_head.end_layer) == (0, 1)
    assert (cuda.start_layer, cuda.end_layer) == (1, 3)


def test_runtime_kv_measurement_can_only_shrink_provisional_contract():
    profile = build_capacity_profile_from_tensor_sizes(
        model_name="test/model",
        tensor_sizes={
            "model.embed_tokens.weight": 10,
            "model.layers.0.weight": 10,
            "model.layers.1.weight": 10,
            "model.layers.2.weight": 10,
            "model.norm.weight": 10,
            "lm_head.weight": 10,
        },
        model_info=_model(),
        usable_memory_bytes=10_000,
        target_context_tokens=100,
        runtime_reserve_bytes=100,
        backend="cuda",
    )
    calibrated = calibrate_profile_from_runtime(
        profile,
        start_layer=0,
        end_layer=3,
        kv_capacity_tokens=20,
    )

    assert calibrated["state"] == "runtime_calibrated"
    assert calibrated["usable_memory_bytes"] <= profile["usable_memory_bytes"]
    assert all(
        new_end <= old_end
        for new_end, old_end in zip(calibrated["max_end_by_start"], profile["max_end_by_start"])
    )

    pressured = constrain_profile_to_memory(profile, 1_000)
    assert pressured["usable_memory_bytes"] == 1_000
    assert all(
        new_end <= old_end
        for new_end, old_end in zip(pressured["max_end_by_start"], profile["max_end_by_start"])
    )


def test_bootstrapped_scheduler_keeps_unprofiled_join_safe_until_heartbeat():
    model = _model()
    existing = _contract_node("existing", model, [3, 3, 3])
    scheduler = Scheduler(
        model,
        [existing],
        strategy="dp",
        routing_strategy="dp",
        min_nodes_bootstrapping=1,
    )
    assert scheduler.bootstrap()

    joining = _contract_node("joining", model, [3, 3, 3])
    joining.capacity_profile = None
    scheduler.join(joining)
    assert scheduler.node_manager.state_of(joining.node_id) == NodeState.STANDBY
    assert joining.start_layer is None

    scheduler.enqueue_node_update(
        joining.node_id,
        capacity_protocol_version=1,
        capacity_profile=_contract(model, [3, 3, 3]),
    )
    scheduler._process_node_updates()
    assert scheduler.node_manager.state_of(joining.node_id) == NodeState.ACTIVE
    assert (joining.start_layer, joining.end_layer) == (0, 3)


def test_dp_reconnect_restores_recent_shard_after_capacity_negotiation():
    model = _model()
    head = _contract_node("mac-head", model, [1, 2, 3])
    tail = _contract_node("windows-tail", model, [0, 3, 3])
    tail.capacity_profile["http_frontend"] = {
        "available": False,
        "protocol": "vllm-engine-core-v1",
    }
    scheduler = Scheduler(
        model,
        [head, tail],
        strategy="dp",
        routing_strategy="dp",
        min_nodes_bootstrapping=1,
    )
    assert scheduler.bootstrap()
    assert (head.start_layer, head.end_layer) == (0, 1)
    assert (tail.start_layer, tail.end_layer) == (1, 3)

    scheduler.enqueue_leave(tail.node_id)
    scheduler._process_leaves()
    assert not scheduler.has_full_pipeline()

    replacement = _contract_node("windows-tail", model, [0, 3, 3])
    replacement.capacity_profile = None
    scheduler.enqueue_join(replacement)
    scheduler._process_joins()
    assert scheduler.node_manager.state_of(replacement.node_id) == NodeState.STANDBY
    assert replacement.start_layer is None

    negotiated = _contract(model, [0, 3, 3])
    negotiated["http_frontend"] = {
        "available": False,
        "protocol": "vllm-engine-core-v1",
    }
    scheduler.enqueue_node_update(
        replacement.node_id,
        capacity_protocol_version=1,
        capacity_profile=negotiated,
    )
    scheduler._process_node_updates()

    assert (replacement.start_layer, replacement.end_layer) == (1, 3)
    assert scheduler.node_manager.state_of(replacement.node_id) == NodeState.ACTIVE
    assert scheduler.has_full_pipeline()
