"""
Minimal tests for the Scheduler orchestrator.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from scheduling.node import RequestSignal
from scheduling.scheduler import Scheduler

from .test_utils import build_model_info, build_node, set_rtt_from_coords


def test_scheduler_rejects_unknown_allocation_and_routing_strategies():
    model = build_model_info(12)

    with pytest.raises(ValueError, match="Unsupported layer allocation strategy"):
        Scheduler(model, [], strategy="unknown")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported request routing strategy"):
        Scheduler(model, [], routing_strategy="unknown")  # type: ignore[arg-type]


def test_scheduler_initialize_and_dispatch():
    """Allocate, then enqueue one request and dispatch it."""
    model = build_model_info(12)
    n1 = build_node("a100-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    n2 = build_node("a100-1", model, tflops=312.0, mem_gb=80.0, x=1, y=0)
    set_rtt_from_coords([n1, n2])

    # Use scheduler bootstrap so RR pipelines are registered in NodeManager.
    sched = Scheduler(
        model, [n1, n2], strategy="greedy", routing_strategy="rr", min_nodes_bootstrapping=1
    )
    ok = sched.bootstrap()
    assert ok
    allocs = sched.node_manager.list_node_allocations(model.num_layers)
    assert allocs, "Allocator should assign at least one pipeline"
    # Check coverage equals total layers
    total = sum(e - s for _, s, e in allocs)
    assert total >= model.num_layers

    # Push a request and dispatch
    req = RequestSignal(request_id="req-1")
    sched.receive_request(req)
    assignment = sched.dispatch_next_request()
    assert assignment is not None
    req_id, path, latency = assignment
    assert req_id == req.request_id
    assert path, "Path should be non-empty"
    assert latency >= 0.0


def test_scheduler_forwards_required_context_to_router(monkeypatch):
    model = build_model_info(12)
    sched = Scheduler(model, [], routing_strategy="dp")
    observed = {}

    class RecordingRouter:
        def max_supported_context_tokens(self):
            return 65536

        def find_optimal_path(self, **kwargs):
            observed.update(kwargs)
            return [], float("inf")

    sched.request_router = RecordingRouter()
    monkeypatch.setattr(sched, "serving_ready", lambda: True)
    request = RequestSignal(request_id="large-request", required_context_tokens=32768)
    sched.receive_request(request)

    sched.dispatch_next_request()

    assert observed == {
        "last_refit_time": 0.0,
        "required_context_tokens": 32768,
    }


def test_scheduler_caps_worker_context_by_model_contract():
    model = build_model_info(12)
    model.max_context_length = 32768
    node = build_node("single", model, mem_gb=400.0)
    node.max_sequence_length = 65536
    sched = Scheduler(
        model,
        [node],
        strategy="greedy",
        routing_strategy="rr",
        min_nodes_bootstrapping=1,
    )
    assert sched.bootstrap()

    assert sched.max_supported_context_tokens() == 32768
    request = RequestSignal(request_id="beyond-model-limit", required_context_tokens=32769)
    sched.receive_request(request)
    _, path, latency = sched.dispatch_next_request()

    assert path == []
    assert latency == float("inf")


def test_scheduler_releases_route_without_waiting_for_worker_heartbeat():
    model = build_model_info(12)
    node = build_node("single", model, mem_gb=400.0)
    node.max_concurrent_requests = 1
    sched = Scheduler(
        model,
        [node],
        strategy="greedy",
        routing_strategy="rr",
        min_nodes_bootstrapping=1,
    )
    assert sched.bootstrap()

    first_id = uuid.uuid4()
    first = RequestSignal(request_id=first_id)
    sched.receive_request(first)
    assert sched.dispatch_next_request() is not None
    assert node.routing_load == 1

    node.current_requests = 1  # A worker heartbeat may remain stale after completion.
    second = RequestSignal(request_id="second")
    sched.receive_request(second)
    _, busy_path, _ = sched.dispatch_next_request()
    assert busy_path == []

    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter = pool.submit(sched.wait_for_routing_capacity, 1.0)
        time.sleep(0.01)
        assert not waiter.done()
        assert sched.release_request(str(first_id))
        assert waiter.result(timeout=0.5)

    assert not sched.release_request(str(first_id))
    assert node.routing_load == 0

    third = RequestSignal(request_id="third")
    sched.receive_request(third)
    _, available_path, _ = sched.dispatch_next_request()
    assert available_path == [node.node_id]


def test_cancelled_pending_request_cannot_reserve_recovered_pipeline():
    model = build_model_info(12)
    node = build_node("recovered", model, mem_gb=400.0)
    node.max_concurrent_requests = 1
    sched = Scheduler(
        model,
        [node],
        strategy="greedy",
        routing_strategy="rr",
        min_nodes_bootstrapping=1,
    )
    assert sched.bootstrap()

    abandoned = RequestSignal(
        request_id="timed-out-http-request",
        required_context_tokens=16316,
    )
    sched.receive_request(abandoned)
    assert not sched.cancel_request_signal(abandoned)

    result = sched.dispatch_next_request()

    assert result == (abandoned.request_id, [], float("inf"))
    assert abandoned.cancelled
    assert node.reserved_requests == 0
    assert node.reserved_context_tokens == 0


def test_cancelling_signal_releases_route_when_dispatch_wins_race():
    model = build_model_info(12)
    node = build_node("dispatch-won", model, mem_gb=400.0)
    node.max_concurrent_requests = 1
    sched = Scheduler(
        model,
        [node],
        strategy="greedy",
        routing_strategy="rr",
        min_nodes_bootstrapping=1,
    )
    assert sched.bootstrap()

    request = RequestSignal(request_id="dispatched-before-timeout", required_context_tokens=4096)
    sched.receive_request(request)
    assert sched.dispatch_next_request() is not None
    assert node.reserved_requests == 1

    assert sched.cancel_request_signal(request)
    assert node.reserved_requests == 0
    assert node.reserved_context_tokens == 0


def test_scheduler_reserves_measured_kv_blocks_and_releases_them():
    model = build_model_info(12)
    node = build_node("measured", model, mem_gb=400.0)
    node.max_concurrent_requests = 4
    node.max_sequence_length = 32768
    node.kv_cache_token_capacity = 16384
    node.kv_cache_block_size = 64
    sched = Scheduler(
        model,
        [node],
        strategy="greedy",
        routing_strategy="rr",
        min_nodes_bootstrapping=1,
    )
    assert sched.bootstrap()
    assert sched.max_supported_context_tokens() == 16384

    sched.receive_request(RequestSignal(request_id="large", required_context_tokens=10000))
    _, first_path, _ = sched.dispatch_next_request()
    assert first_path == [node.node_id]
    assert node.reserved_context_tokens == 10048

    sched.receive_request(RequestSignal(request_id="does-not-fit", required_context_tokens=6400))
    _, busy_path, _ = sched.dispatch_next_request()
    assert busy_path == []

    assert sched.release_request("large")
    assert node.reserved_context_tokens == 0
    sched.receive_request(RequestSignal(request_id="now-fits", required_context_tokens=6400))
    _, available_path, _ = sched.dispatch_next_request()
    assert available_path == [node.node_id]


def test_scheduler_never_admits_context_without_runtime_telemetry():
    model = build_model_info(12)
    node = build_node("legacy", model, mem_gb=400.0)
    node.kv_cache_token_capacity = None
    node.kv_cache_block_size = None
    sched = Scheduler(
        model,
        [node],
        strategy="greedy",
        routing_strategy="rr",
        min_nodes_bootstrapping=1,
    )
    assert sched.bootstrap()
    assert sched.max_supported_context_tokens() == 0

    sched.receive_request(RequestSignal(request_id="requires-telemetry", required_context_tokens=1))
    _, path, latency = sched.dispatch_next_request()

    assert path == []
    assert latency == float("inf")
    assert node.reserved_context_tokens == 0


def test_concurrent_dispatch_cannot_double_reserve_last_kv_blocks():
    model = build_model_info(12)
    node = build_node("atomic", model, mem_gb=400.0)
    node.max_concurrent_requests = 4
    node.max_sequence_length = 32768
    node.kv_cache_token_capacity = 16384
    node.kv_cache_block_size = 64
    sched = Scheduler(
        model,
        [node],
        strategy="greedy",
        routing_strategy="rr",
        min_nodes_bootstrapping=1,
    )
    assert sched.bootstrap()
    for request_id in ("concurrent-a", "concurrent-b"):
        sched.receive_request(RequestSignal(request_id=request_id, required_context_tokens=10000))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: sched.dispatch_next_request(), range(2)))

    paths = [result[1] for result in results if result is not None]
    assert sorted(bool(path) for path in paths) == [False, True]
    assert node.reserved_requests == 1
    assert node.reserved_context_tokens == 10048


def test_layer_reallocation_invalidates_old_executor_kv_geometry():
    model = build_model_info(12)
    node = build_node("reallocated", model)
    node.start_layer = 0
    node.end_layer = 6

    node.set_layer_allocation(0, 12)

    assert node.kv_cache_token_capacity is None
    assert node.kv_cache_block_size is None
    assert node.static_context_capacity == 0


def test_scheduler_join_and_leave():
    """New node can join and be assigned; leave removes it and may rebalance."""
    model = build_model_info(12)
    n1 = build_node("a100-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    n2 = build_node("a100-1", model, tflops=312.0, mem_gb=80.0, x=1, y=0)
    set_rtt_from_coords([n1, n2])
    sched = Scheduler(
        model, [n1, n2], strategy="greedy", routing_strategy="dp", min_nodes_bootstrapping=1
    )

    # Join a new node
    n3 = build_node("rtx4090-x", model, tflops=82.6, mem_gb=24.0, x=0, y=1)
    sched.enqueue_join(n3)
    sched._process_joins()
    assert n3.start_layer is not None and n3.end_layer is not None

    # Leave
    sched.enqueue_leave(n3.node_id)
    sched._process_leaves()
    assert n3 not in sched.node_manager.active_nodes
    assert n3 not in sched.node_manager.standby_nodes


def test_scheduler_bootstrap_wait_and_dynamic_events():
    """Scheduler waits for min nodes, bootstraps, then handles join/leave events."""
    model = build_model_info(12)
    # Start with no nodes assigned yet; bootstrap needs 2
    n1 = build_node("a100-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    sched = Scheduler(model, [], strategy="dp", routing_strategy="dp", min_nodes_bootstrapping=2)

    # Enqueue one join; should not bootstrap yet (insufficient nodes)
    sched.enqueue_join(n1)
    # Process events once (simulate part of event loop)
    sched._process_joins()  # type: ignore[attr-defined]
    assert sched.node_manager.num_nodes == 1
    assert not sched.node_manager.has_full_pipeline(model.num_layers)

    # Add second node and process join; now bootstrap should succeed
    n2 = build_node("5090-1", model, tflops=165.0, mem_gb=32.0, x=1, y=0)
    sched.enqueue_join(n2)
    sched._process_joins()  # type: ignore[attr-defined]
    # RTTs are needed for DP routing strategy
    set_rtt_from_coords(sched.node_manager.nodes)
    ok = sched.bootstrap()
    assert ok
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Dynamic join after bootstrap should assign immediately
    n3 = build_node("rtx4090-x", model, tflops=82.6, mem_gb=24.0, x=0, y=1)
    sched.enqueue_join(n3)
    sched._process_joins()  # type: ignore[attr-defined]
    assert n3.start_layer is not None and n3.end_layer is not None
    print(sched.node_manager.list_node_allocations(model.num_layers))

    # Leave a non-critical node; if still full pipeline, no global rebalance forced
    remaining_before = sched.node_manager.has_full_pipeline(model.num_layers)
    sched.enqueue_leave(n3.node_id)
    sched._process_leaves()  # type: ignore[attr-defined]
    assert sched.node_manager.has_full_pipeline(model.num_layers) == remaining_before

    print(sched.node_manager.list_node_allocations(model.num_layers))

    for node in list(sched.node_manager.nodes):
        if node.start_layer is not None and node.end_layer is not None:
            sched.layer_allocator.deallocate(node)  # type: ignore[attr-defined]
    # Re-allocate only first node to make pipeline incomplete
    sched.layer_allocator.allocate(sched.node_manager.nodes[0], 0, model.num_layers - 1)  # type: ignore[attr-defined]
    # Now leave that node to break coverage and trigger global rebalance path
    core_id = sched.node_manager.nodes[0].node_id
    sched.enqueue_leave(core_id)
    sched._process_leaves()  # type: ignore[attr-defined]


def test_bootstrap_starts_a_fresh_heartbeat_lease_for_waiting_nodes():
    """A worker must not expire immediately after waiting for the cluster bootstrap."""
    model = build_model_info(12)
    first = build_node("first", model, mem_gb=400.0)
    second = build_node("second", model, mem_gb=400.0)
    first.last_heartbeat = time.time() - 60.0

    sched = Scheduler(
        model,
        [first, second],
        strategy="dp",
        routing_strategy="dp",
        min_nodes_bootstrapping=2,
        heartbeat_timeout=30.0,
    )

    before_bootstrap = time.time()
    assert sched.bootstrap()
    assert all(node.last_heartbeat >= before_bootstrap for node in sched.node_manager.active_nodes)

    sched.checking_node_heartbeat()
    assert sched._pending_leaves.empty()  # type: ignore[attr-defined]


def test_automatic_rejoin_discards_stale_worker_layer_assignment():
    """A reconnecting automatic worker is reallocated from clean STANDBY state."""
    model = build_model_info(12)
    first = build_node("first", model, mem_gb=400.0)
    sched = Scheduler(
        model,
        [first],
        strategy="dp",
        routing_strategy="dp",
        min_nodes_bootstrapping=1,
    )
    assert sched.bootstrap()

    rejoining = build_node("rejoining", model, mem_gb=400.0)
    rejoining.start_layer = 0
    rejoining.end_layer = model.num_layers
    assert not rejoining.manual_layer_assignment

    sched.enqueue_join(rejoining)
    sched._process_joins()  # type: ignore[attr-defined]

    assert rejoining in sched.node_manager.active_nodes
    assert rejoining.start_layer == 0
    assert rejoining.end_layer == model.num_layers


def test_scheduler_snapshot_handles_unallocated_standby_nodes():
    """A joined standby node has no layer allocation yet; snapshot must not divide by zero."""
    model = build_model_info(12)
    n1 = build_node("standby-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    sched = Scheduler(model, [], strategy="dp", routing_strategy="rr", min_nodes_bootstrapping=2)

    sched.enqueue_join(n1)
    sched._process_joins()  # type: ignore[attr-defined]

    snapshot = sched.alloc_log_snapshot
    assert "failed to build allocation snapshot" not in snapshot
    assert "Standby nodes (1)" in snapshot
    assert "standby-0" in snapshot
    assert "latency     inf ms" in snapshot


def test_rr_manual_pipeline_registers_after_all_stages_are_ready():
    """Manual RR allocations become routable only after every stage is ready."""
    model = build_model_info(12)
    head = build_node("manual-head", model, x=0, y=0)
    tail = build_node("manual-tail", model, x=1, y=0)
    set_rtt_from_coords([head, tail])

    head.manual_layer_assignment = True
    head.start_layer = 0
    head.end_layer = 2
    head.is_active = False
    tail.manual_layer_assignment = True
    tail.start_layer = 2
    tail.end_layer = model.num_layers
    tail.is_active = False

    sched = Scheduler(model, [], routing_strategy="rr", min_nodes_bootstrapping=2)
    sched.enqueue_join(head)
    sched.enqueue_join(tail)
    sched._process_joins()  # type: ignore[attr-defined]

    assert sched._bootstrapped_event.is_set()  # type: ignore[attr-defined]
    assert not sched.request_router.routing_ready()
    assert sched.node_manager.get_registered_pipeline_node_ids() == {}

    sched.enqueue_node_update(head.node_id, is_active=True)
    sched._process_node_updates()  # type: ignore[attr-defined]
    assert sched.node_manager.get_registered_pipeline_node_ids() == {}

    sched.enqueue_node_update(tail.node_id, is_active=True)
    sched._process_node_updates()  # type: ignore[attr-defined]

    assert sched.request_router.routing_ready()
    assert sched.node_manager.get_registered_pipeline_node_ids() == {
        0: [head.node_id, tail.node_id]
    }


def test_scheduler_single_node_leave_then_rejoin_reassigns_layers():
    """With one node, after leave then re-join, layers should be re-assigned.

    Reproduction of observed issue: when `min_nodes_bootstrapping=1`, after killing the
    only node (leave) and re-joining it, the scheduler fails to re-assign layers.
    This test encodes the expected behavior (should re-assign), so it currently fails.
    """
    model = build_model_info(12)

    # Start with a single capable node and bootstrap successfully
    n1 = build_node("solo-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    set_rtt_from_coords([n1])
    sched = Scheduler(model, [n1], strategy="dp", min_nodes_bootstrapping=1)
    ok = sched.bootstrap()
    assert ok
    assert n1.start_layer is not None and n1.end_layer is not None

    # Simulate node leave (e.g., the process was killed)
    sched.enqueue_leave(n1.node_id)
    sched._process_leaves()  # type: ignore[attr-defined]

    assert n1 not in sched.node_manager.nodes
    assert not sched.node_manager.has_full_pipeline(model.num_layers)

    # Re-join the (same) node id; scheduler should re-assign layers
    n1_rejoin = build_node("solo-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    sched.enqueue_join(n1_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]

    # Expected behavior: after re-join with min_nodes_bootstrapping=1, layers are assigned again
    assert n1_rejoin.start_layer is not None and n1_rejoin.end_layer is not None, (
        "After re-join, single node should be assigned a full layer range"
    )


def test_scheduler_three_nodes_sequential_join_leave_rejoin():
    """Test scheduler with 28-layer model, 3 nodes each capable of 22 layers.

    Scenario:
    - 28-layer model
    - n1, n2, n3 all can host 22 layers
    - min_nodes_bootstrapping=2
    - n1, n2, n3 join sequentially
    - n1 leaves and rejoins
    - n2 leaves and rejoins
    - n3 leaves and rejoins
    """
    model = build_model_info(28)

    # Create nodes that can each host 22 layers
    # Calculation: 100GB can host 16 layers, so 22 layers need ~137.5GB
    # Using 150GB to ensure capacity for 22 layers with some margin
    n1 = build_node("n1", model, tflops=312.0, mem_gb=138.0, x=0, y=0)
    n2 = build_node("n2", model, tflops=312.0, mem_gb=138.0, x=1, y=0)
    n3 = build_node("n3", model, tflops=312.0, mem_gb=138.0, x=2, y=0)

    # Verify nodes can host 22 layers
    assert n1.get_decoder_layer_capacity() >= 22, "n1 should be able to host 22 layers"
    assert n2.get_decoder_layer_capacity() >= 22, "n2 should be able to host 22 layers"
    assert n3.get_decoder_layer_capacity() >= 22, "n3 should be able to host 22 layers"

    # Initialize scheduler with min_nodes_bootstrapping=2, no nodes initially
    sched = Scheduler(model, [], strategy="dp", routing_strategy="dp", min_nodes_bootstrapping=2)

    # Step 1: n1 joins (not enough nodes yet)
    sched.enqueue_join(n1)
    sched._process_joins()  # type: ignore[attr-defined]
    assert sched.node_manager.num_nodes == 1
    assert sched.node_manager.num_standby_nodes == 1
    assert not sched.node_manager.has_full_pipeline(model.num_layers)

    # Step 2: n2 joins (now we have 2 nodes, should bootstrap)
    sched.enqueue_join(n2)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.node_manager.nodes)
    ok = sched.bootstrap()
    assert ok, "Bootstrap should succeed with 2 nodes"
    assert sched.node_manager.has_full_pipeline(model.num_layers)
    assert sched.node_manager.num_active_nodes == 2
    assert sched.node_manager.num_standby_nodes == 0

    # Step 3: n3 joins (dynamic join after bootstrap)
    sched.enqueue_join(n3)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.node_manager.nodes)
    assert n3.start_layer is not None and n3.end_layer is not None
    assert sched.node_manager.num_nodes == 3
    assert sched.node_manager.num_active_nodes == 3
    assert sched.node_manager.num_standby_nodes == 0
    print(sched.node_manager.list_node_allocations(model.num_layers))

    # Step 4: n1 leaves and rejoins
    n1_id = n1.node_id
    sched.enqueue_leave(n1_id)
    sched._process_leaves()  # type: ignore[attr-defined]
    assert n1 not in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 2
    print(sched.node_manager.list_node_allocations(model.num_layers))
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Rejoin n1
    n1_rejoin = build_node("n1", model, tflops=312.0, mem_gb=138.0, x=0, y=0)
    sched.enqueue_join(n1_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.node_manager.nodes)
    assert n1_rejoin.start_layer is not None and n1_rejoin.end_layer is not None
    assert sched.node_manager.num_nodes == 3
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Step 5: n2 leaves and rejoins
    n2_id = n2.node_id
    sched.enqueue_leave(n2_id)
    sched._process_leaves()  # type: ignore[attr-defined]
    assert n2 not in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 2
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Rejoin n2
    n2_rejoin = build_node("n2", model, tflops=312.0, mem_gb=138.0, x=1, y=0)
    sched.enqueue_join(n2_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.node_manager.nodes)
    assert n2_rejoin.start_layer is not None and n2_rejoin.end_layer is not None
    assert sched.node_manager.num_nodes == 3
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Step 6: n3 leaves and rejoins
    n3_id = n3.node_id
    sched.enqueue_leave(n3_id)
    sched._process_leaves()
    assert n3 not in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 2
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Rejoin n3
    n3_rejoin = build_node("n3", model, tflops=312.0, mem_gb=138.0, x=2, y=0)
    sched.enqueue_join(n3_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.node_manager.nodes)
    assert n3_rejoin.start_layer is not None and n3_rejoin.end_layer is not None
    assert sched.node_manager.num_nodes == 3
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Final verification: all nodes should have layer assignments
    allocations = sched.node_manager.list_node_allocations(model.num_layers)
    assert len(allocations) == 3, "All 3 nodes should have layer assignments"
    # Verify full pipeline coverage
    total_covered = sum(e - s for _, s, e in allocations)
    assert total_covered >= model.num_layers, "All layers should be covered"


def test_rr_expand_pipelines_from_newly_joined_standby_nodes():
    """In RR mode, joining new STANDBY nodes after bootstrap should expand pipelines."""
    model = build_model_info(12)

    # Make single nodes capable of hosting full [0, L) so each can be its own pipeline.
    # Use large memory to ensure capacity >= num_layers.
    n1 = build_node("p0", model, tflops=312.0, mem_gb=400.0, x=0, y=0)
    n2 = build_node("p1", model, tflops=312.0, mem_gb=400.0, x=1, y=0)
    set_rtt_from_coords([n1, n2])

    sched = Scheduler(
        model, [n1], strategy="greedy", routing_strategy="rr", min_nodes_bootstrapping=1
    )
    ok = sched.bootstrap()
    assert ok

    registered = sched.node_manager.get_registered_pipeline_node_ids()
    assert len(registered) == 1

    # Join another node; scheduler should keep it STANDBY, then expand RR pipelines
    # by allocating from STANDBY and extending registered pipelines.
    sched.enqueue_join(n2)
    sched._process_joins()  # type: ignore[attr-defined]

    sched.request_router.expand_pipelines()
    registered2 = sched.node_manager.get_registered_pipeline_node_ids()
    assert len(registered2) == 2
    # Both nodes should now be ACTIVE
    assert sched.node_manager.num_active_nodes == 2


def test_complicated_rr():
    """In RR mode, joining new STANDBY nodes after bootstrap should expand pipelines."""
    model = build_model_info(44)

    # Make single nodes capable of hosting full [0, L) so each can be its own pipeline.
    # Use large memory to ensure capacity >= num_layers.
    n1 = build_node("p0", model, tflops=312.0, mem_gb=138.0, x=0, y=0)
    n2 = build_node("p1", model, tflops=312.0, mem_gb=138.0, x=1, y=0)
    n3 = build_node("p2", model, tflops=312.0, mem_gb=138.0, x=2, y=0)
    n4 = build_node("p3", model, tflops=312.0, mem_gb=138.0, x=1, y=0)
    set_rtt_from_coords([n1, n2, n3, n4])

    sched = Scheduler(
        model, [n1, n2], strategy="greedy", routing_strategy="rr", min_nodes_bootstrapping=2
    )
    ok = sched.bootstrap()
    assert ok

    registered = sched.node_manager.get_registered_pipeline_node_ids()
    assert len(registered) == 1
    print(sched.node_manager.list_node_allocations(model.num_layers))

    # Join another node; scheduler should keep it STANDBY, then expand RR pipelines
    # by allocating from STANDBY and extending registered pipelines.
    sched.enqueue_join(n3)
    sched._process_joins()  # type: ignore[attr-defined]
    registered = sched.node_manager.get_registered_pipeline_node_ids()
    assert len(registered) == 1
    print(sched.node_manager.list_node_allocations(model.num_layers))
    assert sched.node_manager.num_active_nodes == 2
    assert sched.node_manager.num_standby_nodes == 1

    sched.enqueue_join(n4)
    sched._process_joins()  # type: ignore[attr-defined]
    sched.request_router.expand_pipelines()

    registered2 = sched.node_manager.get_registered_pipeline_node_ids()
    print(sched.node_manager.list_node_allocations(model.num_layers))
    assert len(registered2) == 2
    assert sched.node_manager.num_active_nodes == 4
    assert sched.node_manager.num_standby_nodes == 0

    sched.enqueue_leave(n3.node_id)
    sched._process_leaves()  # type: ignore[attr-defined]
    assert n3 not in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 3
    assert sched.node_manager.num_active_nodes == 2
    assert sched.node_manager.num_standby_nodes == 1
    # Leaving any member should invalidate its entire registered pipeline.
    registered_after_leave = sched.node_manager.get_registered_pipeline_node_ids()
    assert len(registered_after_leave) == 1
    assert all(n3.node_id not in p and n4.node_id not in p for p in registered_after_leave.values())

    sched.enqueue_join(n3)
    sched._process_joins()  # type: ignore[attr-defined]
    assert n3 in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 4
    assert sched.node_manager.num_active_nodes == 4
    assert sched.node_manager.num_standby_nodes == 0

    sched.enqueue_leave(n1.node_id)
    sched.enqueue_leave(n4.node_id)
    sched._process_leaves()  # type: ignore[attr-defined]
    assert n1 not in sched.node_manager.nodes
    assert n4 not in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 2
    print(sched.node_manager.list_node_allocations(model.num_layers))
    # REBOOT
    assert sched.node_manager.num_active_nodes == 2
    assert sched.node_manager.num_standby_nodes == 0
