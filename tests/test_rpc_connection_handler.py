from backend.server.contribution_gate import account_hash
from backend.server.rpc_connection_handler import RPCConnectionHandler

from tests.scheduler_tests.test_utils import build_model_info, build_node


class RecordingScheduler:
    def __init__(self):
        self.model_info = build_model_info(12)
        self.node = build_node("worker", self.model_info, mem_gb=16.0)
        self.update = None
        self.joined = None
        self.full_pipeline = True

    def get_node(self, node_id):
        return self.node if node_id == self.node.node_id else None

    def has_full_pipeline(self):
        return self.full_pipeline

    def enqueue_join(self, node):
        self.joined = node

    def enqueue_node_update(self, node_id, **update):
        self.update = (node_id, update)

    def list_node_allocations(self):
        return []


class AllocationScheduler:
    def __init__(self):
        self.model_info = build_model_info(12)
        self.num_layers = 12
        self.enable_weight_refit = False
        self.weight_refit_mode = "disk"
        self.nodes = {
            node_id: build_node(node_id, self.model_info)
            for node_id in ("head-a", "head-b", "tail-a", "tail-b")
        }
        for node_id in ("head-a", "head-b"):
            self.nodes[node_id].set_layer_allocation(0, 6)
        for node_id in ("tail-a", "tail-b"):
            self.nodes[node_id].set_layer_allocation(6, 12)

    def list_node_allocations(self):
        return [(node.node_id, node.start_layer, node.end_layer) for node in self.nodes.values()]

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def negotiated_chunked_prefill_size(self):
        return 0


def test_node_update_forwards_raw_latency_while_worker_is_at_capacity():
    scheduler = RecordingScheduler()
    handler = RPCConnectionHandler.__new__(RPCConnectionHandler)
    handler.scheduler = scheduler

    response = handler.node_update(
        {
            "node_id": "worker",
            "hardware": {
                "node_id": "worker",
                "num_gpus": 1,
                "tflops_fp16": 8.52,
                "gpu_name": "Apple M4",
                "memory_gb": 16.0,
                "memory_bandwidth_gbps": 100.0,
                "device": "mlx",
            },
            "kvcache_mem_ratio": 0.25,
            "param_mem_ratio": 0.65,
            "max_concurrent_requests": 1,
            "max_sequence_length": 65536,
            "kv_cache_token_capacity": 131072,
            "kv_cache_block_size": 32,
            "current_requests": 1,
            "layer_latency_ms": 1.5,
            "is_active": True,
            "start_layer": 0,
            "end_layer": 12,
            "direct_peer_ids": ["downstream-worker"],
            "account_token": "ab" * 32,
        }
    )

    assert response == ({}, {})
    assert scheduler.update is not None
    node_id, update = scheduler.update
    assert node_id == "worker"
    assert update["current_requests"] == 1
    assert update["layer_latency_ms"] == 1.5
    assert update["kv_cache_token_capacity"] == 131072
    assert update["kv_cache_block_size"] == 32
    assert update["max_concurrent_requests"] == 1
    assert update["direct_peer_ids"] == ["downstream-worker"]
    assert update["account_hash"] == account_hash("ab" * 32)


def test_node_update_refreshes_registration_when_bootstrap_is_incomplete():
    scheduler = RecordingScheduler()
    scheduler.full_pipeline = False
    handler = RPCConnectionHandler.__new__(RPCConnectionHandler)
    handler.scheduler = scheduler

    response = handler.node_update(
        {
            "node_id": "worker",
            "hardware": {
                "node_id": "worker",
                "num_gpus": 1,
                "tflops_fp16": 8.52,
                "gpu_name": "Apple M4",
                "memory_gb": 16.0,
                "memory_bandwidth_gbps": 100.0,
                "device": "mlx",
                "usable_memory_bytes": 8 * 1024**3,
            },
            "kvcache_mem_ratio": 0.25,
            "param_mem_ratio": 0.65,
            "max_concurrent_requests": 1,
            "max_sequence_length": 65536,
            "supports_frontend": True,
            "account_token": "ab" * 32,
        }
    )

    assert response == ({}, {})
    assert scheduler.update is None
    assert scheduler.joined is not None
    assert scheduler.joined.node_id == "worker"
    assert scheduler.joined.supports_frontend is True
    assert scheduler.joined.hardware.usable_memory_bytes == 8 * 1024**3


def test_layer_allocation_returns_all_possible_cyclic_outbound_peers():
    handler = RPCConnectionHandler.__new__(RPCConnectionHandler)
    handler.scheduler = AllocationScheduler()

    head = handler.get_layer_allocation("head-a")
    tail = handler.get_layer_allocation("tail-a")

    assert head["outbound_peer_ids"] == ["tail-a", "tail-b"]
    assert tail["outbound_peer_ids"] == ["head-a", "head-b"]


def test_build_node_preserves_frontend_capability():
    scheduler = RecordingScheduler()
    handler = RPCConnectionHandler.__new__(RPCConnectionHandler)
    handler.scheduler = scheduler

    node = handler.build_node(
        {
            "node_id": "windows-worker",
            "hardware": {
                "node_id": "windows-worker",
                "num_gpus": 1,
                "tflops_fp16": 50.0,
                "gpu_name": "RTX 4080 SUPER",
                "memory_gb": 16.0,
                "memory_bandwidth_gbps": 600.0,
                "device": "cuda",
            },
            "kvcache_mem_ratio": 0.25,
            "param_mem_ratio": 0.65,
            "max_concurrent_requests": 1,
            "max_sequence_length": 4096,
            "supports_frontend": False,
        }
    )

    assert node.supports_frontend is False


def test_build_node_preserves_measured_kv_geometry():
    scheduler = RecordingScheduler()
    handler = RPCConnectionHandler.__new__(RPCConnectionHandler)
    handler.scheduler = scheduler

    node = handler.build_node(
        {
            "node_id": "measured-worker",
            "hardware": {
                "node_id": "measured-worker",
                "num_gpus": 1,
                "tflops_fp16": 50.0,
                "gpu_name": "RTX",
                "memory_gb": 16.0,
                "memory_bandwidth_gbps": 600.0,
                "device": "cuda",
            },
            "kvcache_mem_ratio": 0.25,
            "param_mem_ratio": 0.65,
            "max_concurrent_requests": 4,
            "max_sequence_length": 32768,
            "kv_cache_token_capacity": 98304,
            "kv_cache_block_size": 64,
        }
    )

    assert node.kv_cache_token_capacity == 98304
    assert node.kv_cache_block_size == 64
