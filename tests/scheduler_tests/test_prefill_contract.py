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
    assert not scheduler.serving_ready()
