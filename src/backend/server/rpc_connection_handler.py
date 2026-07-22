import json
import time

import httpx
from lattica import ConnectionHandler, Lattica, rpc_method, rpc_stream, rpc_stream_iter

from backend.server.contribution_gate import account_hash
from parallax_utils.logging_config import get_logger
from scheduling.node import Node, NodeHardwareInfo
from scheduling.scheduler import Scheduler

logger = get_logger(__name__)


def node_log_summary(message: object) -> dict:
    """Return useful node telemetry without credentials or oversized payloads."""

    if not isinstance(message, dict):
        return {"message_type": type(message).__name__}
    hardware = message.get("hardware")
    safe_hardware = {}
    if isinstance(hardware, dict):
        for key in (
            "gpu_name",
            "device",
            "memory_gb",
            "usable_memory_bytes",
            "system_available_memory_bytes",
            "system_reserve_bytes",
            "device_available_memory_bytes",
            "device_reserve_bytes",
        ):
            if key in hardware:
                safe_hardware[key] = hardware[key]
    return {
        key: value
        for key, value in {
            "node_id": message.get("node_id"),
            "status": message.get("status"),
            "start_layer": message.get("start_layer"),
            "end_layer": message.get("end_layer"),
            "current_requests": message.get("current_requests"),
            "memory_pressure": message.get("memory_pressure"),
            "hardware": safe_hardware,
        }.items()
        if value is not None and value != {}
    }


class RPCConnectionHandler(ConnectionHandler):
    """
    Handles RPC requests from clients, forwarding them to the appropriate TransformerBackend.
    Inherits from hivemind's ConnectionHandler.
    """

    def __init__(
        self,
        lattica: Lattica | None,
        scheduler: Scheduler,
        http_port: int,
    ):
        if lattica is not None:
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
        logger.info("receive node_join request: %s", node_log_summary(message))
        try:
            node = self.build_node(message)
            self.scheduler.enqueue_join(node)

            # A full DP pipeline may require several workers. A blocking join
            # prevented the worker from starting its announcer until allocation,
            # while the scheduler expired that waiting node after 30 seconds.
            # Acknowledge registration as soon as the event loop has accepted it;
            # periodic node_update calls then keep the lease alive and deliver the
            # eventual layer allocation.
            response = self.wait_join_registration(node.node_id, wait_seconds=5)
            logger.debug(f"node_join response: {response}")
            return response
        except Exception as e:
            logger.exception(f"node_join error: {e}")
            return {}

    @rpc_method
    def node_leave(self, message):
        logger.debug("receive node_leave request: %s", node_log_summary(message))
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
        logger.debug("receive node_update request: %s", node_log_summary(message))
        try:
            node = self.build_node(message)
            # Check if node exists in scheduler
            if self.scheduler.get_node(node.node_id) is None:
                # Node not found, automatically join it (e.g., after model switch)
                logger.info(
                    f"Node {node.node_id} not found in scheduler, auto-joining via node_update"
                )
                self.scheduler.enqueue_join(node)
                # A scheduler restart must also acknowledge the registration
                # before a full pipeline exists, so this already-running worker
                # keeps sending heartbeats while its peers reconnect.
                response = self.wait_join_registration(node.node_id, wait_seconds=5)
                return response, {}

            if not self.scheduler.has_full_pipeline():
                # A scheduler restart can receive lightweight heartbeats before a
                # full node_join.  That auto-registers a STANDBY node, then later
                # node_update carries the complete worker-owned capabilities
                # (frontend support, live hardware envelope, account binding).
                # Re-route through join so the existing node is refreshed and the
                # scheduler retries bootstrap once the cluster has become eligible.
                logger.info(
                    f"Node {node.node_id} update arrived before bootstrap completed; "
                    "refreshing registration via join"
                )
                self.scheduler.enqueue_join(node)
                return self.pending_join_response(node.node_id), {}

            # Node exists, update its info
            self.scheduler.enqueue_node_update(
                node.node_id,
                current_requests=node.current_requests,
                # The worker reports an already measured base latency. Forward it
                # unchanged; Node.layer_latency_ms also applies load/overload and
                # would persist a transient infinity in the scheduler.
                layer_latency_ms=node.avg_layer_latency_ms,
                new_rtt_to_nodes=node.rtt_to_nodes,
                is_active=node.is_active,
                last_refit_time=node.last_refit_time,
                supports_chunked_prefill=node.supports_chunked_prefill,
                preferred_chunked_prefill_size=node.preferred_chunked_prefill_size,
                chunked_prefill_size=node.chunked_prefill_size,
                kv_cache_token_capacity=node.kv_cache_token_capacity,
                kv_cache_block_size=node.kv_cache_block_size,
                max_concurrent_requests=node.max_concurrent_requests,
                direct_peer_ids=(
                    sorted(node.direct_peer_ids) if node.direct_peer_ids is not None else None
                ),
                reachable_peer_ids=(
                    sorted(node.reachable_peer_ids) if node.reachable_peer_ids is not None else None
                ),
                relayed_peer_ids=(
                    sorted(node.relayed_peer_ids) if node.relayed_peer_ids is not None else None
                ),
                account_hash=node.account_hash,
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

    def wait_join_registration(self, current_node_id, wait_seconds):
        """Return an allocation or a non-empty pending registration response."""

        start_time = time.monotonic()
        while time.monotonic() - start_time <= wait_seconds:
            layer_allocation = self.get_layer_allocation(current_node_id)
            if layer_allocation:
                return layer_allocation
            if self.scheduler.get_node(current_node_id) is not None:
                return self.pending_join_response(current_node_id)
            time.sleep(0.05)
        return {}

    def pending_join_response(self, current_node_id):
        """Acknowledge a live worker whose DP layer range is not ready yet."""

        if self.scheduler.get_node(current_node_id) is None:
            return {}
        return {
            "node_id": current_node_id,
            "status": "waiting",
            "start_layer": None,
            "end_layer": None,
            "chunked_prefill_size": 0,
            "outbound_peer_ids": [],
        }

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
                        "model_max_sequence_length": getattr(
                            self.scheduler.model_info, "max_context_length", None
                        ),
                        "chunked_prefill_size": self.scheduler.chunked_prefill_size_for_node(
                            node_id
                        ),
                        "outbound_peer_ids": self._outbound_peer_ids(
                            current_node_id,
                            end_layer,
                            list_node_allocations,
                        ),
                    }
        return {}

    def _outbound_peer_ids(
        self,
        current_node_id: str,
        end_layer: int,
        allocations,
    ):
        """Return every peer that can follow this shard in a cyclic pipeline."""
        next_start = 0 if end_layer == self.scheduler.num_layers else end_layer
        return sorted(
            node_id
            for node_id, candidate_start, candidate_end in allocations
            if node_id != current_node_id
            and candidate_start == next_start
            and candidate_end > candidate_start
        )

    def build_node(self, node_json: dict):
        node = Node(
            node_id=node_json.get("node_id"),
            hardware=self.build_hardware(node_json.get("hardware")),
            model_info=self.scheduler.model_info,
            kvcache_mem_ratio=node_json.get("kvcache_mem_ratio"),
            param_mem_ratio=node_json.get("param_mem_ratio"),
            max_concurrent_requests=node_json.get("max_concurrent_requests"),
            max_sequence_length=node_json.get("max_sequence_length"),
            kv_cache_token_capacity=node_json.get("kv_cache_token_capacity"),
            kv_cache_block_size=node_json.get("kv_cache_block_size"),
            supports_frontend=node_json.get("supports_frontend", True),
            supports_chunked_prefill=node_json.get("supports_chunked_prefill", False),
            preferred_chunked_prefill_size=node_json.get("preferred_chunked_prefill_size"),
            chunked_prefill_size=node_json.get("chunked_prefill_size"),
            is_active=node_json.get("is_active", True),
            manual_layer_assignment=node_json.get("manual_layer_assignment", False),
            last_refit_time=node_json.get("last_refit_time", 0.0),
            direct_peer_ids=(
                set(node_json["direct_peer_ids"])
                if node_json.get("direct_peer_ids") is not None
                else None
            ),
            reachable_peer_ids=(
                set(node_json["reachable_peer_ids"])
                if node_json.get("reachable_peer_ids") is not None
                else None
            ),
            relayed_peer_ids=(
                set(node_json["relayed_peer_ids"])
                if node_json.get("relayed_peer_ids") is not None
                else None
            ),
            account_hash=account_hash(node_json.get("account_token")),
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
        return node

    def build_hardware(self, hardware_json):
        node_id = hardware_json.get("node_id")
        num_gpus = hardware_json.get("num_gpus")
        tflops_fp16 = hardware_json.get("tflops_fp16")
        gpu_name = hardware_json.get("gpu_name")
        memory_gb = hardware_json.get("memory_gb")
        memory_bandwidth_gbps = hardware_json.get("memory_bandwidth_gbps")
        device = hardware_json.get("device")
        usable_memory_bytes = hardware_json.get("usable_memory_bytes")
        return NodeHardwareInfo(
            node_id=node_id,
            num_gpus=num_gpus,
            tflops_fp16=tflops_fp16,
            gpu_name=gpu_name,
            memory_gb=memory_gb,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            device=device,
            usable_memory_bytes=usable_memory_bytes,
        )
