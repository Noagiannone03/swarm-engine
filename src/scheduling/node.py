"""
Scheduling primitives for distributed LLM inference.

- `NodeHardwareInfo`: static hardware properties
- `RequestSignal`: minimal request envelope (id, received timestamp)
- `RooflinePerformanceModel`: compute/IO roofline estimator with configurable
  sequence/batch shape
- `Node`: worker serving state; manages layer allocation, capacity helpers,
  latency tracking, and RTT cache for network-aware request routing
"""

import time
from dataclasses import dataclass, field
from math import floor, isfinite
from typing import Dict, List, Optional, Set

from parallax_utils.logging_config import get_logger
from scheduling.model_info import ModelInfo

logger = get_logger(__name__)


@dataclass
class NodeHardwareInfo:
    """
    Hardware-only description of a node.

    Contains static properties that do not depend on a specific model, and
    optionally cached RTTs to other nodes for network-aware decisions.
    """

    node_id: str
    num_gpus: int
    tflops_fp16: float
    gpu_name: str
    memory_gb: float
    memory_bandwidth_gbps: float
    device: str


@dataclass
class RequestSignal:
    """
    Minimal request signal container for scheduling.

    - request_id: Unique identifier (hash) for the request
    - received_ts: UNIX timestamp (seconds) when the request was received
    - routing_table: Set by the scheduler when a path is assigned. Semantics:
        None -> not assigned yet; [] -> all pipelines full at the moment; [..] -> route
    - required_context_tokens: Rendered prompt plus the maximum requested output.
    """

    request_id: str
    received_ts: float = field(default_factory=time.time)
    routing_table: Optional[List[str]] = None
    required_context_tokens: int = 0
    # Set when the HTTP waiter gives up before dispatch. The scheduler may keep
    # the object in its FIFO until a pipeline recovers, so cancellation travels
    # with this exact queue entry instead of relying on a reusable request id.
    cancelled: bool = False


class RooflinePerformanceModel:
    """
    Lightweight roofline-based performance estimator.

    Encapsulates compute- and IO-bound latency estimations for a given
    `(hardware, model_info)` pair. Sequence/batch shape can be updated to
    reflect current request context.
    """

    def __init__(
        self,
        hardware: NodeHardwareInfo,
        model_info: ModelInfo,
        quantization_speedup: float = 1.0,
        *,
        batch_size: int = 1,
        target_seq_len: int = 1,
        source_seq_len: int = 256,
        using_mlx: bool = False,
    ) -> None:
        self.tflops = hardware.tflops_fp16
        self.io_bandwidth = hardware.memory_bandwidth_gbps
        self.model_info = model_info
        self.quantization_speedup = quantization_speedup
        self.batch_size = batch_size
        self.target_seq_len = target_seq_len
        self.source_seq_len = source_seq_len
        self.using_mlx = using_mlx

    def get_compute_roofline_latency_ms(self, flops: int) -> float:
        """Compute-bound latency in milliseconds for the given floating-point ops."""
        return flops / (self.quantization_speedup * self.tflops * 1e9)

    def get_io_roofline_latency_ms(self, io_bytes: int) -> float:
        """Memory/IO-bound latency in milliseconds for the given data transfer size."""
        return io_bytes / (self.io_bandwidth * 1e6)

    def set_sequence_shape(
        self,
        *,
        batch_size: Optional[int] = None,
        target_seq_len: Optional[int] = None,
        source_seq_len: Optional[int] = None,
    ) -> None:
        """Convenience setter to update any of batch/target/source sequence sizes."""
        if batch_size is not None:
            self.batch_size = batch_size
        if target_seq_len is not None:
            self.target_seq_len = target_seq_len
        if source_seq_len is not None:
            self.source_seq_len = source_seq_len

    def roofline_layer_latency_ms(
        self,
        include_input_embed: bool = False,
        include_lm_head: bool = False,
        num_current_layers: int = 1,
    ) -> float:
        """Estimate latency to execute the specified layer set on this node.

        Args:
            include_input_embed: Whether to include input embedding I/O
            include_lm_head: Whether to include LM head compute and I/O
            num_current_layers: Number of decoder layers included

        Returns:
            Total latency (ms) combining decoder layers and optional endpoints.
        """
        if num_current_layers <= 0:
            return float("inf")

        decoder_layer_compute_latency = self.get_compute_roofline_latency_ms(
            self.model_info.decoder_layer_flops(
                batch_size=self.batch_size,
                target_seq_len=self.target_seq_len,
                source_seq_len=self.source_seq_len,
            )
        )
        model_btyes = self.model_info.decoder_layer_io_bytes(
            roofline=True,
            batch_size=self.batch_size,
            target_seq_len=self.target_seq_len,
            source_seq_len=self.source_seq_len,
        )
        if self.using_mlx:
            model_btyes *= self.model_info.mlx_bit_factor
        decoder_layer_io_latency = self.get_io_roofline_latency_ms(model_btyes)

        # For first / last layers
        flops, io_bytes = 0, 0
        if include_input_embed:
            # Embedding lookup is I/O-dominant
            io_bytes += self.model_info.embedding_io_bytes

        if include_lm_head:
            flops += self.model_info.lm_head_flops(self.target_seq_len)
            io_bytes += self.model_info.embedding_io_bytes

        compute_time_ms = self.get_compute_roofline_latency_ms(flops)
        io_time_ms = self.get_io_roofline_latency_ms(io_bytes)
        return (
            num_current_layers * max(decoder_layer_compute_latency, decoder_layer_io_latency)
            + max(compute_time_ms, io_time_ms)
        ) / num_current_layers


@dataclass
class Node:
    """
    Dynamic worker node's serving state and network-aware routing hooks.

    - Tracks layer allocation and request load;
    - Capacity helpers for layer allocation;
    - Latency tracking and estimation if not available from node broadcasting;
    - Networking: optional RTT cache and getter for on-demand RTT measurement.

    """

    node_id: str
    hardware: NodeHardwareInfo
    model_info: ModelInfo

    kvcache_mem_ratio: float = 0.3
    param_mem_ratio: float = 0.5

    max_concurrent_requests: Optional[int] = 16
    max_sequence_length: int = 4096
    # Exact cache geometry published by the initialized executor. Hardware
    # estimates remain diagnostic only and are never used to admit traffic.
    kv_cache_token_capacity: Optional[int] = None
    kv_cache_block_size: Optional[int] = None
    supports_frontend: bool = True

    # Chunked prefill is a pipeline-wide wire contract: every shard must process
    # the same activation span for a request. ``preferred`` is the worker's local
    # setting, while ``chunked_prefill_size`` is the scheduler-negotiated value
    # the running executor currently uses. ``None`` means an old/unknown worker,
    # not "disabled"; disabled is represented explicitly as 0 on the wire.
    supports_chunked_prefill: bool = False
    preferred_chunked_prefill_size: Optional[int] = None
    chunked_prefill_size: Optional[int] = None

    manual_layer_assignment: bool = False
    start_layer: Optional[int] = None  # inclusive
    end_layer: Optional[int] = None  # exclusive
    current_requests: int = 0
    reserved_requests: int = field(default=0, init=False)
    reserved_context_tokens: int = field(default=0, init=False)
    _uses_scheduler_reservations: bool = field(default=False, init=False, repr=False)

    # Runtime weight refit for RL
    last_refit_time: float = 0.0

    # todo upload is_active
    is_active: bool = True
    last_heartbeat: float = 0.0
    # Will be updated by node broadcasting
    # otherwise, use roofline performance model to estimate
    avg_layer_latency_ms: Optional[float] = None
    load_compensator: float = 0.05

    rtt_to_nodes: Optional[Dict[str, float]] = None
    # ``None`` preserves compatibility with workers that predate direct-link
    # qualification.  New workers publish an explicit set (possibly empty),
    # allowing routing to fail closed when Lattica only has a relayed path.
    direct_peer_ids: Optional[Set[str]] = None

    _force_max_concurrent_requests: bool = False

    def __post_init__(self):
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()
        if self.rtt_to_nodes is None:
            self.rtt_to_nodes = {}

    @property
    def max_requests(self) -> int:
        """Executor request-count limit.

        KV memory is a token budget, not a fixed batch-size limit: two 2k
        requests do not consume the same cache as two 32k requests. Keeping the
        count and token constraints separate also avoids the upstream bug where
        a derived KV clamp was immediately undone with ``max(requested, derived)``.
        """
        value = 16 if self.max_concurrent_requests is None else int(self.max_concurrent_requests)
        if value <= 0:
            raise ValueError(f"Node {self.node_id} has invalid max concurrent requests: {value}")
        return value

    @property
    def effective_kv_cache_token_capacity(self) -> Optional[int]:
        """Return measured executor capacity; estimates never admit traffic."""
        if self.kv_cache_token_capacity is not None:
            capacity = int(self.kv_cache_token_capacity)
            return capacity if capacity > 0 else None
        return None

    def reservation_tokens(self, required_context_tokens: int) -> int:
        """Round a request to the physical allocation granularity of the backend."""
        required = max(0, int(required_context_tokens))
        if required == 0:
            return 0
        block_size = 1
        if self.kv_cache_block_size is not None and int(self.kv_cache_block_size) > 0:
            block_size = int(self.kv_cache_block_size)
        return ((required + block_size - 1) // block_size) * block_size

    @property
    def static_context_capacity(self) -> int:
        """Largest single request this shard can ever admit."""
        kv_capacity = self.effective_kv_cache_token_capacity
        if kv_capacity is None:
            return 0
        return max(0, min(int(self.max_sequence_length), kv_capacity))

    @property
    def remaining_context_tokens(self) -> Optional[int]:
        """Unreserved scheduler-side KV budget, if it is known."""
        capacity = self.effective_kv_cache_token_capacity
        if capacity is None:
            return None
        return max(0, capacity - self.reserved_context_tokens)

    def can_accept_request(self, required_context_tokens: int = 0) -> bool:
        """Check request-count, context-window, and aggregate KV constraints."""
        required = max(0, int(required_context_tokens))
        reservation = self.reservation_tokens(required)
        if self.routing_load >= self.max_requests:
            return False
        if required == 0:
            return True
        if required > self.static_context_capacity:
            return False
        remaining = self.remaining_context_tokens
        # A context-bearing request requires executor telemetry. This prevents
        # scheduler estimates from becoming product admission decisions.
        return remaining is not None and reservation <= remaining

    @property
    def num_current_layers(self) -> int:
        """Number of currently allocated layers."""
        if self.start_layer is None or self.end_layer is None:
            return 0
        return self.end_layer - self.start_layer

    @property
    def has_embedding(self) -> bool:
        """Check if this node hosts the embedding layer (layer 0)."""
        if self.start_layer is None:
            return False
        return self.start_layer == 0

    @property
    def has_lm_head(self) -> bool:
        """Check if this node hosts the LM head layer (last layer)."""
        if self.end_layer is None:
            return False
        return self.end_layer == self.model_info.num_layers

    @property
    def is_overloaded(self) -> bool:
        """Check if node is at capacity for requests."""
        return self.routing_load >= self.max_requests

    @property
    def routing_load(self) -> int:
        """Return the scheduler-authoritative in-flight load used for routing."""
        if self._uses_scheduler_reservations:
            return self.reserved_requests
        return self.current_requests

    def get_decoder_layer_capacity(
        self, include_input_embed: bool = False, include_lm_head: bool = False
    ) -> int:
        """Return how many decoder layers this node can store for parameters.

        Capacity is measured using the parameter memory budget on the device.
        """
        available_memory_bytes = floor(
            self.hardware.num_gpus
            * self.hardware.memory_gb
            * 1024
            * 1024
            * 1024
            * self.param_mem_ratio
        )
        if include_input_embed:
            available_memory_bytes -= self.model_info.embedding_io_bytes
        if include_lm_head:
            if not (include_input_embed and self.model_info.tie_embedding):
                available_memory_bytes -= self.model_info.embedding_io_bytes

        if self.hardware.device == "mlx":
            # For mlx, consider mlx bit factor
            return floor(
                available_memory_bytes
                / (
                    self.model_info.decoder_layer_io_bytes(roofline=False)
                    * self.model_info.mlx_bit_factor
                )
            )
        else:
            return floor(
                available_memory_bytes / self.model_info.decoder_layer_io_bytes(roofline=False)
            )

    @property
    def per_decoder_layer_kv_cache_memory(self) -> Optional[int]:
        """Return the available memory for kv cache per layer."""
        if self.num_current_layers == 0:
            return None
        return floor(
            (
                self.hardware.num_gpus
                * self.hardware.memory_gb
                * 1024
                * 1024
                * 1024
                * self.kvcache_mem_ratio
            )
            / self.num_current_layers
        )

    def set_layer_allocation(self, start_layer: int, end_layer: int) -> None:
        """Set the layer range allocated to this node."""
        had_allocation = self.start_layer is not None and self.end_layer is not None
        allocation_changed = (self.start_layer, self.end_layer) != (start_layer, end_layer)
        if had_allocation and allocation_changed:
            # Cache geometry depends on the number and type of hosted layers.
            # The reloaded executor must measure and publish it again.
            self.kv_cache_token_capacity = None
            self.kv_cache_block_size = None
            if self.direct_peer_ids is not None:
                self.direct_peer_ids = set()
        self.start_layer = start_layer
        self.end_layer = end_layer

    def clear_layer_allocation(self) -> None:
        """Clear the layer allocation for this node."""
        self.start_layer = None
        self.end_layer = None
        self.kv_cache_token_capacity = None
        self.kv_cache_block_size = None
        if self.direct_peer_ids is not None:
            self.direct_peer_ids = set()

    def clear_serving_state(self) -> None:
        """Clear serving/runtime state for this node.

        - Clear layer allocation
        - Reset in-flight request counter
        - Clear measured avg layer latency (will be re-learned / re-broadcast)

        TODO: Verify the worker side / p2p server side state is kept in sync with this reset
        (e.g. any runtime KV cache, in-flight request bookkeeping, and broadcasted metrics).
        """
        self.clear_layer_allocation()
        self.current_requests = 0
        self.reserved_requests = 0
        self.reserved_context_tokens = 0
        self._uses_scheduler_reservations = False
        self.avg_layer_latency_ms = None

    def set_layer_latency_ms(self, latency_ms: float) -> None:
        """Update the layer latency for this node."""
        self.avg_layer_latency_ms = latency_ms

    def roofline_layer_latency_ms(self) -> float:
        """Get the roofline layer latency for this node."""
        # Compute an effective compute speedup due to quantization.
        bytes_per_elem = float(self.model_info.param_bytes_per_element)
        # bf16/fp16 baseline ~2 bytes
        base = 1.0 if bytes_per_elem <= 0 else 2.0 / bytes_per_elem
        # Empirical efficiency factor: int8 often achieves ~80% of theoretical 2x
        efficiency = 0.8 if bytes_per_elem < 2.0 else 1.0
        quantization_speedup = max(0.1, base * efficiency)
        perf_model = RooflinePerformanceModel(
            hardware=self.hardware,
            model_info=self.model_info,
            quantization_speedup=quantization_speedup,
            batch_size=self.routing_load,
            target_seq_len=1,
            source_seq_len=self.max_sequence_length,
            using_mlx=self.hardware.device == "mlx",
        )
        return perf_model.roofline_layer_latency_ms(
            include_input_embed=self.has_embedding,
            include_lm_head=self.has_lm_head,
            num_current_layers=self.num_current_layers,
        )

    @property
    def layer_latency_ms(self) -> float:
        """Get effective layer latency considering both roofline and load."""
        if self.is_overloaded:
            return float("inf")
        if self.avg_layer_latency_ms is None:
            return self.roofline_layer_latency_ms()
        return self.avg_layer_latency_ms + self.load_compensator * (
            1.0 * self.routing_load / self.max_requests
        )

    def update_rtt(self, target_node_id: str, rtt_ms: float):
        """Update RTT measurement to another node."""
        self.rtt_to_nodes[target_node_id] = rtt_ms

    def can_forward_to(self, other: "Node") -> bool:
        """Return whether this worker has qualified a direct RPC path to ``other``."""
        if self.node_id == other.node_id:
            return True
        if self.direct_peer_ids is None:
            return True
        return other.node_id in self.direct_peer_ids

    def get_rtt_to(self, other: "Node") -> float:
        """Get RTT to another node from cached RTTs.

        A freshly joined pair of workers may not have dialled each other yet,
        even though both are reachable through the same Lattica peer.  In that
        cold-start case, use the shortest sum of their measured RTTs to a
        common peer as a conservative routing estimate.  Once either worker
        reports a direct RTT, the direct measurement always takes precedence.

        Returns:
            RTT in milliseconds, or float("inf") if neither a direct/reverse
            measurement nor a finite common-peer estimate exists.
        """
        if self == other:
            return 0.0
        if self.rtt_to_nodes is None:
            return float("inf")
        if other.node_id not in self.rtt_to_nodes:
            # Best-effort fallback: in real deployments RTT may be reported only from one side.
            # Treat RTT as symmetric for routing/selection purposes if reverse RTT exists.
            if other.rtt_to_nodes is not None and self.node_id in other.rtt_to_nodes:
                return other.rtt_to_nodes[self.node_id]

            # Workers publish measurements to every connected Lattica peer.
            # A common peer provides real reachability evidence before the two
            # workers have established their first direct/relayed stream.
            if other.rtt_to_nodes:
                estimates = []
                for peer_id in self.rtt_to_nodes.keys() & other.rtt_to_nodes.keys():
                    try:
                        self_rtt = float(self.rtt_to_nodes[peer_id])
                        other_rtt = float(other.rtt_to_nodes[peer_id])
                    except (TypeError, ValueError):
                        continue
                    if (
                        self_rtt >= 0
                        and other_rtt >= 0
                        and isfinite(self_rtt)
                        and isfinite(other_rtt)
                    ):
                        estimates.append(self_rtt + other_rtt)
                if estimates:
                    estimated_rtt = min(estimates)
                    logger.debug(
                        "Estimated RTT from node %s to node %s via a common peer: %.3f ms",
                        self.node_id,
                        other.node_id,
                        estimated_rtt,
                    )
                    return estimated_rtt
            logger.warning("Cannot find RTT from node %s to node %s", self.node_id, other.node_id)
            return float("inf")
        return self.rtt_to_nodes[other.node_id]

    def hosts_layer(self, layer_id: int) -> bool:
        """Return True if this node hosts the given layer id.

        Interprets `current_layers` as a half-open interval [start, end).
        """
        if self.start_layer is None or self.end_layer is None:
            return False
        return self.start_layer <= layer_id < self.end_layer

    def add_request(self, required_context_tokens: int = 0):
        """Atomically reserve count and KV-token capacity for one request."""
        required = max(0, int(required_context_tokens))
        reservation = self.reservation_tokens(required)
        if not self.can_accept_request(required):
            raise ValueError(
                f"Node {self.node_id} has insufficient request/KV capacity "
                f"(required_tokens={required}, load={self.routing_load}/{self.max_requests}, "
                f"remaining_kv_tokens={self.remaining_context_tokens})"
            )
        self._uses_scheduler_reservations = True
        self.reserved_requests += 1
        self.reserved_context_tokens += reservation

    def remove_request(self, required_context_tokens: int = 0):
        """Release one scheduler-owned count and KV-token reservation."""
        required = self.reservation_tokens(required_context_tokens)
        self._uses_scheduler_reservations = True
        self.reserved_requests = max(0, self.reserved_requests - 1)
        self.reserved_context_tokens = max(0, self.reserved_context_tokens - required)
