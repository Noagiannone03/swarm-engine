from scheduling.scheduler import Scheduler

from .test_utils import build_model_info, build_node, set_rtt_from_coords


def _allocated_scheduler():
    model = build_model_info(12)
    head = build_node("head", model, mem_gb=80.0)
    tail = build_node("tail", model, mem_gb=80.0)
    set_rtt_from_coords([head, tail])
    scheduler = Scheduler(
        model,
        [head, tail],
        strategy="dp",
        routing_strategy="dp",
        min_nodes_bootstrapping=1,
    )
    scheduler.layer_allocator.allocate(head, 0, 6)
    scheduler.layer_allocator.allocate(tail, 6, 12)
    return scheduler, head, tail


def test_homogeneous_pipeline_negotiates_smallest_chunk_size():
    scheduler, head, tail = _allocated_scheduler()
    head.preferred_chunked_prefill_size = 2048
    head.chunked_prefill_size = 1024
    tail.preferred_chunked_prefill_size = 1024
    tail.chunked_prefill_size = 1024

    assert scheduler.negotiated_chunked_prefill_size() == 1024
    assert scheduler.prefill_contract_ready()
    assert scheduler.serving_ready()


def test_vllm_stage_disables_chunking_for_the_entire_pipeline():
    scheduler, head, tail = _allocated_scheduler()
    head.preferred_chunked_prefill_size = 1024
    head.chunked_prefill_size = 1024
    tail.supports_chunked_prefill = False
    tail.preferred_chunked_prefill_size = 1024
    tail.chunked_prefill_size = 0

    assert scheduler.negotiated_chunked_prefill_size() == 0
    assert not scheduler.prefill_contract_ready()
    assert not scheduler.serving_ready()
    assert not scheduler.serving_ready()

    head.chunked_prefill_size = 0

    assert scheduler.prefill_contract_ready()
    assert scheduler.serving_ready()


def test_unknown_legacy_worker_fails_closed():
    scheduler, head, tail = _allocated_scheduler()
    head.preferred_chunked_prefill_size = 0
    head.chunked_prefill_size = 0
    tail.supports_chunked_prefill = False
    tail.preferred_chunked_prefill_size = None
    tail.chunked_prefill_size = None

    assert scheduler.negotiated_chunked_prefill_size() == 0
    assert not scheduler.prefill_contract_ready()


def test_orphan_vllm_shard_does_not_reload_complete_mlx_pipeline():
    model = build_model_info(12)
    mlx = build_node("mlx", model, mem_gb=80.0)
    orphan_vllm = build_node("vllm-orphan", model, mem_gb=80.0)
    set_rtt_from_coords([mlx, orphan_vllm])
    scheduler = Scheduler(
        model,
        [mlx, orphan_vllm],
        strategy="dp",
        routing_strategy="dp",
        min_nodes_bootstrapping=1,
    )
    scheduler.layer_allocator.allocate(mlx, 0, 12)
    scheduler.layer_allocator.allocate(orphan_vllm, 1, 12)

    mlx.preferred_chunked_prefill_size = 1024
    mlx.chunked_prefill_size = 1024
    orphan_vllm.supports_chunked_prefill = False
    orphan_vllm.preferred_chunked_prefill_size = 1024
    orphan_vllm.chunked_prefill_size = 0

    assert scheduler.node_manager.full_pipeline_node_ids(12) == {mlx.node_id}
    assert scheduler.negotiated_chunked_prefill_size() == 1024
    assert scheduler.chunked_prefill_size_for_node(orphan_vllm.node_id) == 0
    assert scheduler.prefill_contract_ready()


def test_prefill_contract_lowers_but_does_not_upgrade_within_generation():
    model = build_model_info(12)
    mlx = build_node("mlx", model, mem_gb=80.0)
    vllm_head = build_node("vllm-head", model, mem_gb=80.0)
    vllm_tail = build_node("vllm-tail", model, mem_gb=80.0)
    set_rtt_from_coords([mlx, vllm_head, vllm_tail])
    scheduler = Scheduler(
        model,
        [mlx, vllm_head, vllm_tail],
        strategy="dp",
        routing_strategy="dp",
        min_nodes_bootstrapping=1,
    )
    scheduler.layer_allocator.allocate(mlx, 0, 12)
    mlx.preferred_chunked_prefill_size = 1024
    mlx.chunked_prefill_size = 1024

    assert scheduler.negotiated_chunked_prefill_size() == 1024

    scheduler.layer_allocator.allocate(vllm_head, 0, 1)
    scheduler.layer_allocator.allocate(vllm_tail, 1, 12)
    for node in (vllm_head, vllm_tail):
        node.supports_chunked_prefill = False
        node.preferred_chunked_prefill_size = 1024
        node.chunked_prefill_size = 0

    assert scheduler.negotiated_chunked_prefill_size() == 0

    scheduler.layer_allocator.deallocate(vllm_head)
    scheduler.layer_allocator.deallocate(vllm_tail)

    # Removing the restrictive route is only a performance opportunity.  Keep
    # the compatible live contract until an explicit allocation generation.
    assert scheduler.negotiated_chunked_prefill_size() == 0
    assert not scheduler.serving_ready()
