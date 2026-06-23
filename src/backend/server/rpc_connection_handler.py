import os
import time

from lattica import ConnectionHandler, Lattica, rpc_method, rpc_stream, rpc_stream_iter

from parallax_utils.logging_config import get_logger
from scheduling.node import Node, NodeHardwareInfo
from scheduling.scheduler import Scheduler

logger = get_logger(__name__)

import json

import httpx


def _node_join_allocation_wait_seconds() -> float:
    """Short wait for a synchronous layer assignment during node_join.

    A node may legitimately join as standby when the current pipeline is already
    covered by larger workers. In that case node_join must return promptly so
    the worker can start its node_update heartbeat and keep its contribution
    lease alive. Longer waits belong in the worker process, not in the scheduler
    RPC handler.
    """
    raw = os.environ.get("PARALLAX_NODE_JOIN_ALLOCATION_WAIT_S", "5").strip()
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring PARALLAX_NODE_JOIN_ALLOCATION_WAIT_S=%r", raw)
        return 5.0
    return min(60.0, max(0.0, value))


class RPCConnectionHandler(ConnectionHandler):
    """
    Handles RPC requests from clients, forwarding them to the appropriate TransformerBackend.
    Inherits from hivemind's ConnectionHandler.
    """

    def __init__(
        self,
        lattica: Lattica,
        scheduler: Scheduler,
        http_port: int,
    ):
        # Initialize the base class
        super().__init__(lattica)
        self.scheduler = scheduler
        self.http_port = http_port

    @rpc_stream
    def node_join(self, message):
        # node = {
        #     "node_id": "lattica peer id",
        #     "hardware": {
        #         "node_id": "lattica peer id",
        #         "tflops_fp16": 100,
        #         "memory_gb": 100,
        #         "memory_bandwidth_gbps": 100,
        #     },
        #     "kvcache_mem_ratio": 0.3,
        #     "param_mem_ratio": 0.5,
        #     "max_concurrent_requests": 16,
        #     "max_sequence_length": 1024,
        # }
        logger.info(f"receive node_join request: {message}")
        try:
            node = self.build_node(message)
            self._refresh_contribution_lease(message, node, source="node_join")
            self.scheduler.enqueue_join(node)

            response = self.wait_layer_allocation(
                node.node_id, wait_seconds=_node_join_allocation_wait_seconds()
            )
            if not response:
                response = self.standby_join_response(node)
            logger.debug(f"node_join response: {response}")
            return response
        except Exception as e:
            logger.exception(f"node_join error: {e}")
            return {}

    @rpc_method
    def node_leave(self, message):
        logger.debug(f"receive node_leave request: {message}")
        try:
            node = self.build_node(message)
            self.scheduler.enqueue_leave(node.node_id)
            return {}
        except Exception as e:
            logger.exception(f"node_leave error: {e}")
            return {}

    @rpc_method
    def node_update(self, message):
        """
        Returns a Tuple[Dict, Dict] where
        first dict contains layer allocation result and
        second dict records weight refit information.
        """
        logger.debug(f"receive node_update request: {message}")
        try:
            node = self.build_node(message)
            # Contribution gate: this node is heartbeating into the swarm over the
            # scheduler's own RPC channel → refresh its account lease. Only the
            # scheduler ever writes leases (clients never self-declare), so there
            # is nothing for a consumer to forge. No-op when FABI_GATE=off.
            self._refresh_contribution_lease(message, node, source="node_update")
            # Check if node exists in scheduler
            if self.scheduler.get_node(node.node_id) is None:
                # Node not found, automatically join it (e.g., after model switch)
                logger.info(
                    f"Node {node.node_id} not found in scheduler, auto-joining via node_update"
                )
                self.scheduler.enqueue_join(node)
                # Wait a bit for join to be processed
                time.sleep(0.1)
                # Return layer allocation after join
                layer_allocation = self.wait_layer_allocation(node.node_id, wait_seconds=5)
                return layer_allocation or self.standby_join_response(node), {}

            # Node exists, update its info
            self.scheduler.enqueue_node_update(
                node.node_id,
                current_requests=node.current_requests,
                layer_latency_ms=node.layer_latency_ms,
                new_rtt_to_nodes=node.rtt_to_nodes,
                is_active=node.is_active,
                last_refit_time=node.last_refit_time,
                loading_phase=node.loading_phase,
                kv_free_tokens=node.reported_kv_free_tokens,
            )
            # Return current layer allocation to node
            layer_allocation = self.get_layer_allocation(node.node_id)
            refit_request = {}
            if self.scheduler.refit_request:
                if node.node_id not in self.scheduler.refit_set and node.is_active:
                    refit_request = self.scheduler.refit_request
                    self.scheduler.refit_set.add(node.node_id)
            return layer_allocation, refit_request
        except Exception as e:
            logger.exception(f"node_update error: {e}")
            return {}, {}

    def _model_name_for_node(self, node: Node):
        model_info = self.scheduler.model_info if self.scheduler is not None else node.model_info
        if model_info is None:
            return None
        if getattr(node.hardware, "device", None) == "mlx":
            return getattr(model_info, "mlx_model_name", None) or getattr(
                model_info, "model_name", None
            )
        return getattr(model_info, "model_name", None) or getattr(
            model_info, "mlx_model_name", None
        )

    def _refresh_contribution_lease(self, message, node: Node, *, source: str) -> None:
        try:
            from backend.server.contribution_gate import get_gate

            get_gate().refresh(
                message.get("account_token"),
                node.node_id,
                self._model_name_for_node(node),
            )
        except Exception:
            logger.warning("contribution gate refresh skipped during %s", source, exc_info=True)

    def standby_join_response(self, node: Node) -> dict:
        """ACK a valid join even when no layers are currently assigned.

        Standby nodes are real contributors: they are connected, visible to the
        scheduler, and may be promoted later when the allocation changes. Returning
        an explicit standby ACK lets the worker start heartbeats immediately
        instead of timing out before node_update can refresh its contribution
        lease.
        """
        return {
            "node_id": node.node_id,
            "model_name": self._model_name_for_node(node),
            "status": "standby",
            "standby": True,
            "enable_weight_refit": self.scheduler.enable_weight_refit,
            "weight_refit_mode": self.scheduler.weight_refit_mode,
        }

    @rpc_stream_iter
    def chat_completion(
        self,
        request,
    ):
        """Handle chat completion request"""
        logger.debug(f"Chat completion request: {request}, type: {type(request)}")
        try:
            with httpx.Client(timeout=10 * 60, proxy=None, trust_env=False) as client:
                if request.get("stream", False):
                    with client.stream(
                        "POST",
                        f"http://localhost:{self.http_port}/v1/chat/completions",
                        json=request,
                    ) as response:
                        for chunk in response.iter_bytes():
                            if chunk:
                                yield chunk
                else:
                    response = client.post(
                        f"http://localhost:{self.http_port}/v1/chat/completions", json=request
                    ).json()
                    yield json.dumps(response).encode()
        except Exception as e:
            logger.exception(f"Error in chat completion: {e}")
            yield b"internal server error"

    @rpc_stream_iter
    def cluster_status(self):
        try:
            with httpx.Client(timeout=10 * 60, proxy=None, trust_env=False) as client:
                with client.stream(
                    "GET", f"http://localhost:{self.http_port}/cluster/status"
                ) as response:
                    for chunk in response.iter_bytes():
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.exception(f"Error in cluster status: {e}")
            yield json.dumps({"error": "internal server error"}).encode()

    def wait_layer_allocation(self, current_node_id, wait_seconds):
        start_time = time.time()
        while True:
            layer_allocation = self.get_layer_allocation(current_node_id)
            if layer_allocation:
                return layer_allocation
            if time.time() - start_time > wait_seconds:
                return {}
            time.sleep(0.5)

    def get_layer_allocation(self, current_node_id):
        list_node_allocations = self.scheduler.list_node_allocations()
        for node_id, start_layer, end_layer in list_node_allocations:
            if current_node_id == node_id:
                node = self.scheduler.get_node(node_id)
                if node:
                    return {
                        "node_id": node_id,
                        "model_name": (
                            node.model_info.model_name
                            if node.hardware.device != "mlx"
                            else node.model_info.mlx_model_name
                        ),
                        "start_layer": start_layer,
                        "end_layer": end_layer,
                        "tp_size": node.hardware.num_gpus,
                        "enable_weight_refit": self.scheduler.enable_weight_refit,
                        "weight_refit_mode": self.scheduler.weight_refit_mode,
                    }
        return {}

    def build_node(self, node_json: dict):
        node = Node(
            node_id=node_json.get("node_id"),
            hardware=self.build_hardware(node_json.get("hardware")),
            model_info=self.scheduler.model_info,
            kvcache_mem_ratio=node_json.get("kvcache_mem_ratio"),
            param_mem_ratio=node_json.get("param_mem_ratio"),
            max_concurrent_requests=node_json.get("max_concurrent_requests"),
            max_sequence_length=node_json.get("max_sequence_length"),
            is_active=node_json.get("is_active", True),
            # Worker sends its ServerState as "status" — store it in
            # `loading_phase` so the UI can show "downloading" vs "ready" vs
            # "joining" instead of just the binary is_active flag.
            loading_phase=node_json.get("status", "joining"),
            manual_layer_assignment=node_json.get("manual_layer_assignment", False),
            last_refit_time=node_json.get("last_refit_time", 0.0),
        )
        if node_json.get("start_layer", None) is not None:
            node.start_layer = node_json.get("start_layer")
        if node_json.get("end_layer", None) is not None:
            node.end_layer = node_json.get("end_layer")
        if node_json.get("current_requests", None) is not None:
            node.current_requests = node_json.get("current_requests")
        if node_json.get("layer_latency_ms", None) is not None:
            node.avg_layer_latency_ms = node_json.get("layer_latency_ms")
        if node_json.get("rtt_to_nodes", None) is not None:
            node.rtt_to_nodes = node_json.get("rtt_to_nodes")
        if node_json.get("kv_free_tokens", None) is not None:
            node.reported_kv_free_tokens = node_json.get("kv_free_tokens")
        return node

    def build_hardware(self, hardware_json):
        node_id = hardware_json.get("node_id")
        num_gpus = hardware_json.get("num_gpus")
        tflops_fp16 = hardware_json.get("tflops_fp16")
        gpu_name = hardware_json.get("gpu_name")
        memory_gb = hardware_json.get("memory_gb")
        memory_bandwidth_gbps = hardware_json.get("memory_bandwidth_gbps")
        device = hardware_json.get("device")
        return NodeHardwareInfo(
            node_id=node_id,
            num_gpus=num_gpus,
            tflops_fp16=tflops_fp16,
            gpu_name=gpu_name,
            memory_gb=memory_gb,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            device=device,
            # Worker-measured, worker-enforced budget for weights+KV. Absent on
            # legacy workers → capacity helper falls back to memory_gb.
            usable_memory_bytes=hardware_json.get("usable_memory_bytes"),
            total_memory_bytes=hardware_json.get("total_memory_bytes"),
        )
