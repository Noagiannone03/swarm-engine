from backend.server.rpc_connection_handler import RPCConnectionHandler

from tests.scheduler_tests.test_utils import build_model_info, build_node


class RecordingScheduler:
    def __init__(self):
        self.model_info = build_model_info(12)
        self.node = build_node("worker", self.model_info, mem_gb=16.0)
        self.update = None

    def get_node(self, node_id):
        return self.node if node_id == self.node.node_id else None

    def enqueue_node_update(self, node_id, **update):
        self.update = (node_id, update)

    def list_node_allocations(self):
        return []


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
            "current_requests": 1,
            "layer_latency_ms": 1.5,
            "is_active": True,
            "start_layer": 0,
            "end_layer": 12,
        }
    )

    assert response == ({}, {})
    assert scheduler.update is not None
    node_id, update = scheduler.update
    assert node_id == "worker"
    assert update["current_requests"] == 1
    assert update["layer_latency_ms"] == 1.5
