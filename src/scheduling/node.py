"""
Scheduling primitives for distributed LLM inference.

- `NodeHardwareInfo`: static hardware properties
- `RequestSignal`: minimal request envelope (id, received timestamp)
- `RooflinePerformanceModel`: compute/IO roofline estimator with configurable
  sequence/batch shape
- `Node`: worker serving state; manages layer allocation, capacity helpers,
  latency tracking, and RTT cache for network-aware request routing
"""

import os
import time
from dataclasses import dataclass, field
from math import floor
from typing import Dict, List, Optional

from parallax_utils.logging_config import get_logger
from parallax_utils.utils import (
    bytes_per_element,
    compute_max_batch_size,
    compute_max_tokens_in_cache,
)
from scheduling.model_info import ModelInfo

logger = get_logger(__name__)


def _env_float(key: str, default: float, *, minimum: Optional[float] = None) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r (not a number)", key, raw)
        return default
    if minimum is not None and value < minimum:
        return minimum
    return value


# --- Peer reliability backoff -------------------------------------------------
# Faithful port of hivemind.dht.node.Blacklist (used by Petals' routing): a peer
# that fails/errors is temporarily excluded from routing, and each successive ban
# episode lasts longer (base * rate**streak). The scheduler otherwise only learns
# a node is unhealthy via the heartbeat timeout (tens of seconds), so without this
# a reachable-but-failing peer keeps getting routed to and stalls requests.
#
# Tunable per-deployment via env (dedicated inference clusters may want longer
# bans; flaky home swarms shorter ones).
PEER_BAN_BASE_SEC = _env_float("PARALLAX_PEER_BAN_BASE_SEC", 5.0, minimum=0.0)
PEER_BAN_BACKOFF_RATE = _env_float("PARALLAX_PEER_BAN_BACKOFF_RATE", 2.0, minimum=1.0)
PEER_BAN_MAX_SEC = _env_float("PARALLAX_PEER_BAN_MAX_SEC", 300.0, minimum=0.0)


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

    # Measured, worker-ENFORCED memory budget for the model (weights + KV), in
    # bytes, summed across this node's GPUs. This is the SAME ceiling the worker
    # installs at load time (torch.cuda.set_per_process_memory_fraction on CUDA,
    # the MLX wired limit on Apple). The scheduler sizes layer counts against
    # exactly that ceiling, so its estimate can never exceed what the worker's
    # own allocator will admit — which is what removes the over-allocation OOM.
    # None for legacy workers that don't report it → the capacity helper falls
    # back to the advertised ``memory_gb`` parameter budget.
    usable_memory_bytes: Optional[float] = None
    total_memory_bytes: Optional[float] = None


@dataclass
class RequestSignal:
    """
    Minimal request signal container for scheduling.

    - request_id: Unique identifier (hash) for the request
    - received_ts: UNIX timestamp (seconds) when the request was received
    - routing_table: Set by the scheduler when a path is assigned. Semantics:
        None -> not assigned yet; [] -> all pipelines full at the moment; [..] -> route
    - context_tokens: Total sequence budget for this request (prompt tokens +
        expected generation). Used for context-aware routing: a path is only
        eligible if every node on it can hold this many tokens. None = unknown
        (caller couldn't tokenize) -> routing falls back to the load/latency-only
        behaviour, so we never block a request just because counting failed.
    """

    request_id: str
    received_ts: float = field(default_factory=time.time)
    routing_table: Optional[List[str]] = None
    context_tokens: Optional[int] = None


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
        # A node can heartbeat (node_update) while still in STANDBY with no layers
        # assigned yet (num_current_layers == 0) — a state our "keep standby
        # contributors alive" change introduced. Upstream never reaches it: node_join
        # blocks until layers are allocated, so this method was only ever called with
        # num_current_layers >= 1, and the division below was safe by invariant. Now
        # that the invariant no longer holds, guard it: a layer count below 1 is
        # meaningless for a *per-layer* latency, so clamp the divisor to 1. For a
        # standby node (no embedding/lm_head) this yields exactly the single
        # decoder-layer roofline latency — a sound estimate of the node's per-layer
        # speed, which is what the scheduler wants for allocation. No-op for >= 1.
        effective_layers = max(1, num_current_layers)
        return (
            effective_layers * max(decoder_layer_compute_latency, decoder_layer_io_latency)
            + max(compute_time_ms, io_time_ms)
        ) / effective_layers


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

    max_concurrent_requests: int = 16
    max_sequence_length: int = 4096

    manual_layer_assignment: bool = False
    start_layer: Optional[int] = None  # inclusive
    end_layer: Optional[int] = None  # exclusive
    current_requests: int = 0

    # Runtime weight refit for RL
    last_refit_time: float = 0.0

    # todo upload is_active
    is_active: bool = True
    # Worker-reported lifecycle state ("joining" | "initializing" | "ready" |
    # "offline" | "error"). Mirrors `parallax.p2p.server.ServerState`. Until
    # the worker reports otherwise we assume "joining" (Lattica handshake in
    # progress). The UI uses this to distinguish "node connected but still
    # downloading model" from "node is in standby for redundancy".
    loading_phase: str = "joining"
    last_heartbeat: float = 0.0
    # Will be updated by node broadcasting
    # otherwise, use roofline performance model to estimate
    avg_layer_latency_ms: Optional[float] = None
    load_compensator: float = 0.05

    rtt_to_nodes: Optional[Dict[str, float]] = None

    # --- Peer reliability (Blacklist-style backoff) ---
    # Number of *successive* ban episodes so far (hivemind's ban_counter): the
    # exponent for the next ban duration. Reset to 0 on a success.
    failure_streak: int = 0
    # Wall-clock epoch until which this node is excluded from routing. 0.0 = not
    # banned. Routing treats `is_banned()` like `is_overloaded` (unavailable),
    # but — unlike overload — it is deliberately kept OUT of `layer_latency_ms`
    # so bans only affect routing, never (re)allocation/placement.
    banned_until: float = 0.0

    _force_max_concurrent_requests: bool = False

    def __post_init__(self):
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()
        if self.rtt_to_nodes is None:
            self.rtt_to_nodes = {}

    @property
    def max_requests(self) -> int:
        """Max concurrent requests bounded by KV budget using sequence length."""
        if self._force_max_concurrent_requests:
            return self.max_concurrent_requests

        if self.start_layer is None or self.end_layer is None:
            return self.max_concurrent_requests
        try:
            elem_bytes = bytes_per_element(
                getattr(self.model_info, "cache_bytes_per_element", None)
            )
        except Exception:
            elem_bytes = 2
        derived_max = compute_max_batch_size(
            requested_max_batch_size=self.max_concurrent_requests,
            max_sequence_len=self.max_sequence_length,
            device=None,
            kv_cache_memory_fraction=self.kvcache_mem_ratio,
            num_shard_layers=self.num_current_layers,
            num_key_value_heads=self.model_info.num_kv_heads,
            head_dim=self.model_info.head_size,
            elem_bytes=elem_bytes,
            memory_gb=self.hardware.memory_gb,
            head_dim_k=self.model_info.head_size_k,
            head_dim_v=self.model_info.head_size_v,
        )
        if derived_max <= 0:
            raise ValueError(
                f"Node {self.node_id} has invalid max concurrent requests: {derived_max}"
            )
        if self.max_concurrent_requests is None:
            return derived_max
        return min(self.max_concurrent_requests, derived_max)

    @property
    def num_current_layers(self) -> int:
        """Number of currently allocated layers."""
        if self.start_layer is None or self.end_layer is None:
            return 0
        return self.end_layer - self.start_layer

    @property
    def max_context_tokens(self) -> int:
        """Max total sequence length (prompt + generation) this node can serve for
        a SINGLE request — the unit used by context-aware routing.

        Two ceilings, we take the min:
          1. ``max_sequence_length`` — the worker's configured cap; its executor
             truncates anything longer (this is what bites today at 16384).
          2. The KV budget for the node's currently-assigned layers, via the SAME
             helper the batch-size logic uses (``compute_max_tokens_in_cache``) and
             the SAME memory accounting as ``max_requests`` — so routing/admission
             never disagree. Fewer layers => more KV per layer => bigger context.

        Before any layer is assigned (num_current_layers == 0) only the configured
        cap is known, so we return that.
        """
        cap = self.max_sequence_length or 0
        if self.start_layer is None or self.end_layer is None or self.num_current_layers <= 0:
            return cap
        try:
            elem_bytes = bytes_per_element(
                getattr(self.model_info, "cache_bytes_per_element", None)
            )
        except Exception:
            elem_bytes = 2
        try:
            kv_tokens = compute_max_tokens_in_cache(
                device="",
                kv_cache_memory_fraction=self.kvcache_mem_ratio,
                num_shard_layers=self.num_current_layers,
                num_key_value_heads=self.model_info.num_kv_heads,
                head_dim_k=self.model_info.head_size_k,
                head_dim_v=self.model_info.head_size_v,
                elem_bytes=elem_bytes,
                # Same accounting as compute_max_batch_size (single-GPU budget):
                # memory_gb * 1GiB * kv_fraction. Keeps max_context_tokens and
                # max_requests strictly consistent.
                available_cache_bytes=int(
                    self.hardware.memory_gb * 1024**3 * self.kvcache_mem_ratio
                ),
            )
        except Exception:
            return cap
        if kv_tokens <= 0:
            return cap
        return min(cap, kv_tokens) if cap > 0 else kv_tokens

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
        return self.current_requests >= self.max_requests

    def is_banned(self, now: Optional[float] = None) -> bool:
        """True while this peer is temporarily excluded from routing after failures."""
        if self.banned_until <= 0.0:
            return False
        return (now if now is not None else time.time()) < self.banned_until

    def record_request_failure(
        self,
        *,
        now: Optional[float] = None,
        base_sec: float = PEER_BAN_BASE_SEC,
        backoff_rate: float = PEER_BAN_BACKOFF_RATE,
        max_sec: float = PEER_BAN_MAX_SEC,
    ) -> None:
        """Temporarily ban this peer from routing, with exponential backoff.

        Faithful port of ``hivemind.dht.node.Blacklist.register_failure``: an
        already-banned peer is left untouched (no extension), and ``failure_streak``
        only advances per *new* ban episode, so each successive ban lasts
        ``base_sec * backoff_rate ** failure_streak`` — a chronically flaky peer is
        shed for longer while a one-off blip recovers quickly. Capped at ``max_sec``.
        """
        if base_sec <= 0.0:
            return
        t = now if now is not None else time.time()
        if self.is_banned(t):
            return  # don't extend an active ban (matches hivemind)
        ban_duration = base_sec * (backoff_rate**self.failure_streak)
        if max_sec > 0.0:
            ban_duration = min(ban_duration, max_sec)
        self.banned_until = t + ban_duration
        self.failure_streak += 1
        logger.info(
            "Peer %s banned from routing for %.1fs (episode #%d)",
            self.node_id,
            ban_duration,
            self.failure_streak,
        )

    def record_request_success(self) -> None:
        """Clear any ban and reset the backoff (matches ``Blacklist.register_success``)."""
        if self.banned_until > 0.0 or self.failure_streak > 0:
            logger.debug("Peer %s reliability reset (success)", self.node_id)
        self.banned_until = 0.0
        self.failure_streak = 0

    def get_decoder_layer_capacity(
        self, include_input_embed: bool = False, include_lm_head: bool = False
    ) -> int:
        """Return how many decoder layers this node can host for the current model.

        The budget is the node's **measured, worker-enforced** memory ceiling
        (``hardware.usable_memory_bytes`` — the same per-process allocator cap
        the worker installs at load time). Sizing against the exact limit the
        worker enforces is what removes the estimate-vs-reality gap that used to
        OOM small cards on load: the scheduler can no longer hand a node more
        layers than the worker's own allocator will admit.

        This mirrors how every mature local-inference stack assigns work — the
        process that loads the model owns the memory budget (vLLM's
        ``determine_available_memory``, Ollama's per-device budget, llama.cpp
        ``--fit``, Petals' ``_choose_num_blocks``). The scheduler only divides
        that measured budget by the per-layer cost it already knows from
        ``ModelInfo``.

        Per hosted layer we charge the decoder-layer WEIGHTS plus a KV slice big
        enough for ONE request at ``max_sequence_length`` (so a node is never
        handed more layers than it can serve a single full-context request on).
        Concurrency beyond one request is elastic — the worker sizes the KV pool
        from whatever memory is left after weights load — so it is not reserved
        here. A fixed runtime workspace (attention/JIT global buffer +
        CUDA-graph scratch, e.g. flashinfer) is reserved up front, and endpoint
        nodes additionally pay for the input-embedding / lm-head weights.

        Legacy fallback: a node that does not report ``usable_memory_bytes``
        (older worker / generic host) is sized from the parameter-memory budget
        derived from the advertised ``memory_gb`` — the original heuristic, kept
        only for backward compatibility.
        """
        usable = getattr(self.hardware, "usable_memory_bytes", None)
        if usable and usable > 0:
            budget = float(usable) - self._runtime_workspace_bytes()
            if include_input_embed:
                budget -= self.model_info.embedding_io_bytes
            if include_lm_head and not (
                include_input_embed and self.model_info.tie_embedding
            ):
                budget -= self.model_info.embedding_io_bytes
            per_layer = self.model_info.decoder_layer_io_bytes(roofline=False)
            if self.hardware.device == "mlx":
                per_layer *= self.model_info.mlx_bit_factor
            per_layer += self._per_layer_kv_floor_bytes()
            if per_layer <= 0 or budget <= 0:
                return 1
            return max(1, floor(budget / per_layer))

        # --- Legacy fallback: advertised parameter-memory budget -------------
        available_memory_bytes = floor(
            self.hardware.num_gpus * self.hardware.memory_gb * 1024**3 * self.param_mem_ratio
        )
        if include_input_embed:
            available_memory_bytes -= self.model_info.embedding_io_bytes
        if include_lm_head:
            if not (include_input_embed and self.model_info.tie_embedding):
                available_memory_bytes -= self.model_info.embedding_io_bytes
        if self.hardware.device == "mlx":
            return floor(
                available_memory_bytes
                / (
                    self.model_info.decoder_layer_io_bytes(roofline=False)
                    * self.model_info.mlx_bit_factor
                )
            )
        return floor(
            available_memory_bytes / self.model_info.decoder_layer_io_bytes(roofline=False)
        )

    def _runtime_workspace_bytes(self) -> float:
        """Non-weight, non-KV runtime GPU buffers to reserve before counting
        layers: the attention/JIT global workspace (flashinfer allocates a fixed
        ``global_workspace_buffer``; vLLM defaults it to ~0.4 GiB) plus the
        CUDA-graph capture scratch. Small on MLX (unified memory, no flashinfer).
        Tunable via ``PARALLAX_GPU_RUNTIME_WORKSPACE_GB``.
        """
        env = os.environ.get("PARALLAX_GPU_RUNTIME_WORKSPACE_GB", "").strip()
        if env:
            try:
                return max(0.0, float(env)) * 1024**3
            except ValueError:
                pass
        default_gb = 0.25 if self.hardware.device == "mlx" else 0.75
        return default_gb * 1024**3

    def _per_layer_kv_floor_bytes(self) -> float:
        """KV-cache bytes to reserve PER LAYER so the node can actually serve its
        committed load — ``kv_reserve_requests`` concurrent requests at
        ``max_sequence_length`` — for every layer it hosts.

        This is the lever that makes the scheduler SPREAD rather than cram. By
        charging the *real* KV each layer will need, a node is handed only as
        many layers as it can truly serve at full context + concurrency. So a
        bigger context (or more concurrency) ⇒ fewer layers per node ⇒ more nodes
        in the pipeline ⇒ each node keeps the headroom for huge contexts — which
        is exactly the goal. It also makes the runtime KV pool fit by
        construction: a node can never be assigned so many layers that serving
        its batch at max length OOMs (the failure mode we hit at 32k × batch 8).

        Reserve count defaults to the node's own ``max_concurrent_requests`` (the
        concurrency it advertised it will serve); override with
        ``PARALLAX_KV_RESERVE_REQUESTS`` (e.g. 1 to pack more layers/throughput,
        higher to spread thinner for context). Returns 0 when not computable.
        """
        seq = self.max_sequence_length or 0
        if seq <= 0:
            return 0.0
        try:
            elem_bytes = bytes_per_element(
                getattr(self.model_info, "cache_bytes_per_element", None)
            )
        except Exception:
            elem_bytes = 2
        per_token_per_layer = (
            self.model_info.num_kv_heads
            * (self.model_info.head_size_k + self.model_info.head_size_v)
            * elem_bytes
        )
        if per_token_per_layer <= 0:
            return 0.0
        reserve_requests = self.max_concurrent_requests or 1
        env = os.environ.get("PARALLAX_KV_RESERVE_REQUESTS", "").strip()
        if env:
            try:
                reserve_requests = int(env)
            except ValueError:
                pass
        reserve_requests = max(1, reserve_requests)
        return float(per_token_per_layer * seq * reserve_requests)

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
        self.start_layer = start_layer
        self.end_layer = end_layer

    def clear_layer_allocation(self) -> None:
        """Clear the layer allocation for this node."""
        self.start_layer = None
        self.end_layer = None

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
            batch_size=self.current_requests,
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
            1.0 * self.current_requests / self.max_requests
        )

    def update_rtt(self, target_node_id: str, rtt_ms: float):
        """Update RTT measurement to another node."""
        self.rtt_to_nodes[target_node_id] = rtt_ms

    def get_rtt_to(self, other: "Node") -> float:
        """Get RTT to another node from cached RTTs.

        Returns:
            RTT in milliseconds, or float("inf") if no cached RTT exists.
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

    def add_request(self):
        """Add a request to this node."""
        self.current_requests += 1

    def remove_request(self):
        """Remove a request from this node."""
        self.current_requests -= 1

    def clear_serving_state(self) -> None:
        """Clear serving/runtime state for this node.

        TODO: Verify the worker side / p2p server side state is kept in sync with this reset
        (e.g. any runtime KV cache, in-flight request bookkeeping, and broadcasted metrics).
        """
        self.clear_layer_allocation()
        self.current_requests = 0
        self.avg_layer_latency_ms = None
