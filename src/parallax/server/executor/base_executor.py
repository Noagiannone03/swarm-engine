"""
High-level executor for managing model shards, scheduler, and cache pool on each Peer.

Executor handles
1. Loading model shards from the repository;
2. Instantiate scheduler, kv cache manager;
3. Handles tokenization / detokenization if needed;
4. Keep listening to RPC to get requests, feed these to scheduler's request pool;
5. Get batched requests from the scheduler,
    - prepare the MLX tensor input
    - rebuild KV cache
    - feed to model runner;
    For now we process prefill and decode requests separately.
    Later when we have Ragged Paged Flash Attention kernel, we can process both in one batch.
6. Run model forward, our model will returned updated caches,
    kv cache manager will handle updating caches per layer;
7. Get the hidden-states from the model execution.
"""

import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import zmq

from parallax.p2p.message_util import (
    abort_request_to_proto,
    proto_to_abort_request,
    proto_to_request,
    request_to_proto,
)
from parallax.p2p.proto import forward_pb2
from parallax.p2p.server import ServerState
from parallax.server.backend_capabilities import is_torch_device
from parallax.server.engine_core_protocol import (
    ENGINE_IDENTITY,
    EngineCoreFinishReason,
    UnsupportedEngineCoreField,
    decode_engine_core_frame,
    encode_engine_core_outputs,
    engine_core_ready_payload,
    engine_core_request_to_initial_request,
    make_engine_core_output,
)
from parallax.server.request import (
    InitialRequest,
    IntermediateRequest,
    Request,
    RequestStatus,
)
from parallax.server.scheduler import Scheduler
from parallax.utils.shared_state import SharedState
from parallax.utils.utils import get_current_device, get_device_dtype, get_zmq_socket
from parallax_utils.fabi_events import emit as emit_fabi_event
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class ExecutorBatchCancelled(RuntimeError):
    """Cooperative backend cancellation observed during a model call."""

    def __init__(self, request_ids: List[str]):
        self.request_ids = tuple(str(request_id) for request_id in request_ids)
        super().__init__(f"cancelled executor requests: {', '.join(self.request_ids)}")


class BaseExecutor:
    """High-level executor for managing model shards, scheduler, and cache pool on each Peer."""

    def __init__(
        self,
        # Model Configs
        start_layer: int,
        end_layer: int,
        dtype: str = "float16",
        # Device override
        device: Optional[str] = None,
        resolved_dtype: Optional[Any] = None,
        # Scheduler Configs
        max_batch_size: Optional[int] = 8,
        max_sequence_length: Optional[int] = None,
        # Controlling perfill / decode ratio
        max_num_tokens_per_batch: int = 16384,
        prefill_priority: int = 0,
        micro_batch_ratio: int = 2,
        scheduler_wait_ms: int = 500,
        request_timeout_s: Optional[int] = 600,
        # Metrics Configs
        layer_latency_update_every: int = 4096,
        # Communication Configs
        # P2P Communication Configs
        send_to_peer_addr: Optional[str] = None,
        recv_from_peer_addr: Optional[str] = None,
        executor_control_addr: Optional[str] = None,
        # IPC Communication Configs
        executor_input_ipc_addr: Optional[str] = None,
        executor_output_ipc_addr: Optional[str] = None,
        # Tensor Parallel Configs
        tp_rank: Optional[int] = 0,
        tp_size: Optional[int] = 1,
        dp_rank: Optional[int] = 0,
        dp_size: Optional[int] = 1,
        # Optional shared state for layer reallocation detection (when running in subprocess)
        shared_state: Optional[dict] = None,
        # Weight Refit
        enable_weight_refit: Optional[bool] = False,
        weight_refit_mode: Optional[str] = "disk",
        chunked_prefill_size: Optional[int] = None,
        kv_block_size: int = 16,
        # Pipe communication
        conn: Optional[List[Any]] = [],
    ):
        # Backend
        if device is not None:
            self.device = device
        else:
            self.device = get_current_device()
        logger.debug(f"Executor initializing on device: {self.device}")

        # for window attention need to calculate causal mask size
        self.finished_batch = []
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.max_num_tokens_per_batch = max_num_tokens_per_batch
        self._should_stop = False  # Flag to gracefully stop the executor
        # Reference to shared state for layer reallocation detection (when in subprocess mode)
        if shared_state is not None:
            self.shared_state = SharedState(shared_state)  # Auto-converts dict to SharedState
        else:
            self.shared_state = None

        # Pipe communication
        self.conn = conn

        self.is_first_peer = start_layer == 0
        self.is_last_peer = end_layer == self.config.get("num_hidden_layers")
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dp_size = dp_size
        self.dp_rank = dp_rank

        # Runtime weight refit for RL
        self.enable_weight_refit = enable_weight_refit
        self.weight_version = 0
        self.weight_refit_mode = weight_refit_mode
        if self.enable_weight_refit and self.tp_size > 1 and self.weight_refit_mode == "cpu":
            self.weight_refit_mode = "disk"
            logger.warning("Force weight update from disk for TP > 1")

        # Metrics throttling for per-layer latency updates
        self.layer_latency_update_every = int(max(1, layer_latency_update_every))
        self._decode_steps_since_metric = self.layer_latency_update_every

        # TODO: Duplicate code to MLXExecutor.
        self.num_shard_layers = end_layer - start_layer
        self.dtype = (
            resolved_dtype if resolved_dtype is not None else get_device_dtype(dtype, self.device)
        )
        logger.debug(
            f"Executor dtype set to {dtype} (resolved={self.dtype}); shard_layers={self.num_shard_layers}"
        )

        self.eos_token_id = self.config.get("eos_token_id", None)
        if self.eos_token_id is None:
            self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        if self.tokenizer.pad_token_id is None:
            if isinstance(self.eos_token_id, list):
                self.pad_token_id = self.eos_token_id[0] if self.eos_token_id else None
            else:
                self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id

        # Scheduler: derive final max_batch_size with KV constraints
        # Remove this for now as it's not working on gpu devices
        # max_batch_size = compute_max_batch_size(
        #     requested_max_batch_size=max_batch_size,
        #     max_sequence_len=max_sequence_length,
        #     device=self.device,
        #     kv_cache_memory_fraction=kv_cache_memory_fraction,
        #     num_shard_layers=self.num_shard_layers,
        #     num_key_value_heads=self.num_key_value_heads,
        #     head_dim=self.head_dim,
        #     dtype=self.dtype,
        # )

        self.scheduler = Scheduler(
            max_batch_size=max_batch_size,
            max_num_tokens_per_batch=max_num_tokens_per_batch,
            prefill_priority=prefill_priority,
            scheduler_wait_ms=scheduler_wait_ms,
            micro_batch_ratio=micro_batch_ratio,
            is_first_peer=self.is_first_peer,
            tokenizer=self.tokenizer,
            eos_token_id=self.eos_token_id,
            cache_manager=self.cache_manager if self.device == "mlx" else None,
            request_timeout_s=request_timeout_s,
            shared_state=self.shared_state,
            chunked_prefill_size=chunked_prefill_size,
        )
        logger.debug(
            f"Scheduler initialized (max_batch_size={max_batch_size}, max_tokens={max_num_tokens_per_batch}, wait_ms={scheduler_wait_ms})"
        )

        # Store max sequence length before engine-core registration, since the
        # Rust frontend reads this value from the registration payload.
        self.max_sequence_length = max_sequence_length
        self.kv_block_size = int(kv_block_size)
        self._publish_runtime_capacity()
        self.model_path = None

        # Communication Related
        if self.tp_rank == 0:
            self.zmq_context = zmq.Context()
            if recv_from_peer_addr:
                self.recv_from_peer_socket = get_zmq_socket(
                    self.zmq_context, zmq.PULL, recv_from_peer_addr, bind=False
                )
            if send_to_peer_addr:
                self.send_to_peer_socket = get_zmq_socket(
                    self.zmq_context, zmq.PUSH, send_to_peer_addr, bind=False
                )
            if executor_control_addr:
                self.executor_control_socket = get_zmq_socket(
                    self.zmq_context, zmq.REP, executor_control_addr, bind=True
                )
            if self.is_first_peer and executor_input_ipc_addr:
                self.recv_from_ipc_socket = self._connect_engine_core_input_socket(
                    executor_input_ipc_addr
                )
            if self.is_first_peer and executor_output_ipc_addr:
                self.send_to_ipc_socket = get_zmq_socket(
                    self.zmq_context, zmq.PUSH, executor_output_ipc_addr, bind=False
                )
            if self.is_first_peer and executor_input_ipc_addr:
                self._send_engine_core_ready_response(dtype)
        if self.shared_state is not None:
            self.shared_state.set_status(ServerState.READY.value)

        if self.tp_rank == 0:
            kv_capacity_tokens, kv_block_size = self._runtime_kv_cache_geometry()
            event_fields = {
                "start_layer": self.start_layer,
                "end_layer": self.end_layer,
                "context_tokens": kv_capacity_tokens,
                "kv_block_size": kv_block_size,
            }
            files_total = getattr(self, "_fabi_weights_files_total", 0)
            if files_total > 0:
                event_fields.update(
                    files_done=getattr(self, "_fabi_weights_files_done", files_total),
                    files_total=files_total,
                )
            emit_fabi_event("weights_load_done", **event_fields)

        # Log executor ready status
        logger.info(
            f"Executor loaded successfully and ready to serve requests "
            f"(layers [{self.start_layer}, {self.end_layer}), "
            f"tp_rank={self.tp_rank}/{self.tp_size}, "
            f"device={self.device}, "
            f"num_shard_layers={self.num_shard_layers})"
        )

    def _runtime_kv_cache_geometry(self) -> Tuple[int, int]:
        """Read the token capacity and block size from the initialized backend.

        This deliberately avoids RAM/VRAM formulas: each backend has already
        accounted for loaded weights, allocator metadata, tensor parallelism,
        cache dtype, hybrid attention, and its physical page granularity.
        """
        cache_manager = getattr(self, "cache_manager", None)
        num_gpu_blocks = getattr(cache_manager, "num_gpu_blocks", None)
        cache_block_size = getattr(cache_manager, "block_size", None)
        if num_gpu_blocks is not None and cache_block_size is not None:
            return int(num_gpu_blocks) * int(cache_block_size), int(cache_block_size)

        model_runner = getattr(self, "model_runner", None)
        token_allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
        allocator_size = getattr(token_allocator, "size", None)
        runner_page_size = getattr(model_runner, "page_size", None)
        if allocator_size is not None and runner_page_size is not None:
            # SGLang's allocator size and available_size() are expressed in tokens.
            return int(allocator_size), int(runner_page_size)

        kv_cache_config = getattr(model_runner, "kv_cache_config", None)
        num_blocks = getattr(kv_cache_config, "num_blocks", None)
        runner_cache_config = getattr(model_runner, "cache_config", None)
        runner_block_size = getattr(runner_cache_config, "block_size", None)
        if num_blocks is not None and runner_block_size is not None:
            # vLLM's shared block pool contains num_blocks physical pages. KV
            # groups share that pool; dividing by the group count is incorrect.
            return int(num_blocks) * int(runner_block_size), int(runner_block_size)

        raise RuntimeError(
            "Initialized executor did not expose KV cache capacity; refusing to advertise "
            "an estimated product capacity"
        )

    def _publish_runtime_capacity(self) -> None:
        """Publish executor-measured admission limits to the P2P heartbeat."""
        capacity, block_size = BaseExecutor._runtime_kv_cache_geometry(self)
        if capacity <= 0 or block_size <= 0:
            raise RuntimeError(
                f"Invalid runtime KV cache geometry: capacity={capacity}, block_size={block_size}"
            )
        max_requests = int(self.scheduler.max_batch_size)
        if max_requests <= 0:
            raise RuntimeError(f"Invalid runtime request capacity: {max_requests}")
        if self.shared_state is not None:
            self.shared_state.update(
                kv_cache_token_capacity=capacity,
                kv_cache_block_size=block_size,
                max_concurrent_requests=max_requests,
            )
        logger.info(
            "Runtime admission capacity: %d KV tokens (%d-token blocks), %d requests",
            capacity,
            block_size,
            max_requests,
        )

    @abstractmethod
    def handle_input_requests(self, requests: List[Request]):
        """Update requests states and status in scheduler and cache manager."""

    @abstractmethod
    def process_batch(self, prepared_inputs: Dict[str, Any], return_decoded_tokens: bool = True):
        """
        Process a batch of requests.

        Args:
            prepared_inputs: A dictionary containing the prepared inputs for the ShardedModel.
            return_decoded_tokens: Whether to return decoded tokens.

        Returns:
            A tensor of shape (B, L, D) containing the hidden states for the next peer.
            or (B,) containing the decoded tokens.
        """

    @abstractmethod
    def _prepare_prefill_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """Prepares inputs for ShardedModel from a batch of prefill requests."""

    @abstractmethod
    def _prepare_decode_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """Prepares inputs for ShardedModel from a batch of decode requests."""

    @abstractmethod
    def _gen_token_id_from_hidden(self, hidden_states) -> Tuple[int, Any]:
        """
        Inplace modifies hidden_states.
        Returns token_id, hidden_states
        """

    @abstractmethod
    def check_and_refit_weight(self, refit_weight_path: str):
        """Run weight if triggered"""

    @abstractmethod
    def _release_request(self, rid: str):
        """Release request in backend frameworks"""

    def _connect_engine_core_input_socket(self, endpoint: str):
        """Connect the engine DEALER socket with vLLM-compatible identity 0."""
        socket = self.zmq_context.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, ENGINE_IDENTITY)
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.connect(endpoint)
        return socket

    def _resolve_engine_core_max_model_len(self) -> int:
        if self.max_sequence_length is not None:
            return int(self.max_sequence_length)

        for key in (
            "max_position_embeddings",
            "model_max_length",
            "max_sequence_length",
            "seq_length",
        ):
            value = self.config.get(key)
            if value is not None:
                return int(value)

        return 4096

    def _send_engine_core_ready_response(self, dtype: str):
        """Register this Parallax engine with the Rust frontend."""
        payload = engine_core_ready_payload(
            max_model_len=self._resolve_engine_core_max_model_len(),
            block_size=self.kv_block_size,
            dtype=dtype,
            num_gpu_blocks=0,
            dp_stats_address=None,
        )
        self.recv_from_ipc_socket.send(payload)
        logger.debug("Sent vLLM engine-core ready registration")

    def recv_requests_from_http(self) -> List[Request]:
        """Receives requests from the vLLM Rust frontend."""
        if self.tp_rank != 0:
            return []

        recv_reqs = []
        while True:
            decoded = None
            try:
                frames = self.recv_from_ipc_socket.recv_multipart(zmq.NOBLOCK)
                if len(frames) != 2:
                    raise ValueError(f"Expected 2 engine-core frames, got {len(frames)}")

                frame_type, payload = frames
                message_type, decoded = decode_engine_core_frame(frame_type, payload)

                if message_type == "abort":
                    self._abort_engine_core_requests(decoded)
                else:
                    req = engine_core_request_to_initial_request(
                        decoded,
                        max_sequence_length=self.max_sequence_length,
                    )
                    recv_reqs.append(req)
            except zmq.ZMQError:
                break
            except Exception as e:
                logger.exception(f"Error receiving engine-core request: {e}")
                request_id = None
                try:
                    if isinstance(decoded, dict):
                        request_id = decoded.get("request_id")
                except Exception:
                    request_id = None
                if request_id is not None:
                    self._send_engine_core_error(str(request_id), e)

        if len(recv_reqs) > 0:
            logger.debug(f"Received {len(recv_reqs)} engine-core requests")
        return recv_reqs

    def _abort_engine_core_requests(self, request_ids: List[str]):
        """Abort requests from vLLM frontend and emit terminal outputs."""
        for request_id in request_ids:
            request = self.scheduler.get_running_request(request_id)
            removed_from_wait_queue = False
            if request is None:
                for queued in list(self.scheduler._wait_queue):
                    if queued.request_id == request_id:
                        self.scheduler._wait_queue.remove(queued)
                        request = queued
                        removed_from_wait_queue = True
                        break

            if request is not None:
                request.abort = True
                request.update_status(RequestStatus.FINISHED_ABORT)
                self.release_and_evict_request(request_id)
                if self.is_first_peer and not self.is_last_peer:
                    self.finished_batch.append(request)
                logger.debug(
                    "Aborted engine-core request %s (from_wait_queue=%s)",
                    request_id,
                    removed_from_wait_queue,
                )
            else:
                logger.debug("Received abort for inactive engine-core request %s", request_id)

            self._send_engine_core_terminal_output(
                request_id=request_id,
                finish_reason=EngineCoreFinishReason.ABORT,
            )

    def _send_engine_core_error(self, request_id: str, error: Exception):
        if isinstance(error, UnsupportedEngineCoreField):
            logger.warning("Rejecting unsupported engine-core request %s: %s", request_id, error)
        self._send_engine_core_terminal_output(
            request_id=request_id,
            finish_reason=EngineCoreFinishReason.ERROR,
        )

    def _send_engine_core_terminal_output(
        self,
        *,
        request_id: str,
        finish_reason: EngineCoreFinishReason,
        new_token_id: Optional[int] = None,
        stop_reason: Optional[int | str] = None,
    ):
        output = make_engine_core_output(
            request_id=request_id,
            new_token_ids=[] if new_token_id is None else [new_token_id],
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )
        self._send_engine_core_outputs([output], finished_requests=[request_id])

    def _send_engine_core_token_output(
        self,
        *,
        request_id: str,
        token_id: Optional[int],
        finish_reason: Optional[EngineCoreFinishReason] = None,
        stop_reason: Optional[int | str] = None,
    ):
        new_token_ids = [] if token_id is None or token_id < 0 else [int(token_id)]
        output = make_engine_core_output(
            request_id=request_id,
            new_token_ids=new_token_ids,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )
        finished_requests = [request_id] if finish_reason is not None else None
        self._send_engine_core_outputs([output], finished_requests=finished_requests)

    def _send_engine_core_outputs(
        self,
        outputs: List[List[Any]],
        *,
        finished_requests: Optional[List[str]] = None,
    ):
        if not hasattr(self, "send_to_ipc_socket") or self.send_to_ipc_socket is None:
            return
        payload = encode_engine_core_outputs(
            outputs,
            engine_index=0,
            finished_requests=finished_requests,
        )
        self.send_to_ipc_socket.send(payload)

    def _finish_reason_for_request(
        self, request: Request
    ) -> tuple[Optional[EngineCoreFinishReason], Optional[int | str]]:
        if request.status == RequestStatus.FINISHED_EOS:
            token_id = None
            if getattr(request, "output_ids", None):
                token_id = request.output_ids[-1]
            return EngineCoreFinishReason.STOP, token_id
        if request.status == RequestStatus.FINISHED_MAX_LENGTH:
            return EngineCoreFinishReason.LENGTH, None
        if request.status in (RequestStatus.FINISHED_ABORT, RequestStatus.CANCELLED):
            return EngineCoreFinishReason.ABORT, None
        if request.status == RequestStatus.ERROR:
            return EngineCoreFinishReason.ERROR, None
        return None, None

    def send_engine_core_request_output(
        self,
        *,
        request: Request,
        token_id: Optional[int],
    ):
        finish_reason, stop_reason = self._finish_reason_for_request(request)
        self._send_engine_core_token_output(
            request_id=request.request_id,
            token_id=token_id,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )

    def recv_requests_from_peer(self) -> Tuple[List[Request], str]:
        """Receives requests from the RPC server."""
        refit_weight_path = ""
        if self.tp_rank == 0:
            recv_reqs = []
            while True:
                try:
                    recv_req = self.recv_from_peer_socket.recv_multipart(zmq.NOBLOCK)
                    if recv_req[0] == b"forward":
                        # Create a new ForwardRequest instance and parse from bytes
                        forward_request = forward_pb2.ForwardRequest()
                        forward_request.ParseFromString(recv_req[1])
                        recv_req = proto_to_request(forward_request, self.device)

                        # Convert hidden_states dtype if necessary
                        if recv_req is not None and len(recv_req) > 0:
                            for req in recv_req:
                                if req.hidden_states is not None:
                                    if req.hidden_states.dtype != self.dtype:
                                        logger.debug(
                                            f"Converting hidden_states dtype from {req.hidden_states.dtype} to {self.dtype} for request {req.request_id}"
                                        )
                                        if is_torch_device(self.device):
                                            req.hidden_states = req.hidden_states.to(self.dtype)
                                        elif self.device == "mlx":
                                            req.hidden_states = req.hidden_states.astype(self.dtype)
                                        else:
                                            raise ValueError(
                                                f"Unsupported device type: {self.device}"
                                            )

                        # Move current position for first peer
                        if self.is_first_peer:
                            for req in recv_req:
                                req.current_position += 1
                        recv_reqs.extend(recv_req)
                    elif recv_req[0] == b"abort":
                        abort_request = forward_pb2.AbortRequest()
                        abort_request.ParseFromString(recv_req[1])
                        recv_req = proto_to_abort_request(abort_request)
                        recv_reqs.extend(recv_req)

                    elif recv_req[0] == b"refit":
                        refit_weight_path = recv_req[1].decode("ascii")
                        self.weight_version = int(recv_req[2].decode("ascii"))
                    else:
                        raise ValueError(f"Unknown request type: {recv_req[0]}")

                except zmq.ZMQError:
                    break
                except Exception as e:
                    logger.exception(f"Error receiving or deserializing request: {e}")
        else:
            recv_reqs = []

        return recv_reqs, refit_weight_path

    def process_executor_control_requests(self) -> None:
        """Serve local, bounded control messages between inference steps."""

        socket = getattr(self, "executor_control_socket", None)
        if self.tp_rank != 0 or socket is None:
            return
        while True:
            try:
                frames = socket.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            binary_response: bytes | None = None
            try:
                if len(frames) not in {1, 2}:
                    raise ValueError(
                        "executor control request must contain metadata and optional binary"
                    )
                request = msgpack.unpackb(frames[0], raw=False, strict_map_key=True)
                if not isinstance(request, dict):
                    raise TypeError("executor control request must be an object")
                binary_request = frames[1] if len(frames) == 2 else None
                result, binary_response = self.handle_executor_control(request, binary_request)
                metadata = {"ok": True, "result": result}
            except Exception as exc:
                logger.warning("Executor control request failed: %s", exc)
                metadata = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:512],
                }
                binary_response = None
            response = [msgpack.packb(metadata, use_bin_type=True)]
            if binary_response is not None:
                response.append(binary_response)
            socket.send_multipart(response)

    def handle_executor_control(
        self,
        request: dict[str, Any],
        binary_request: bytes | None = None,
    ) -> tuple[dict[str, Any], bytes | None]:
        del request, binary_request
        raise RuntimeError("executor does not support exact recovery checkpoints")

    def prepare_batch_inputs(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """Prepares inputs for ShardedModel from a batch of requests.
        Args:
            batched_requests: A list of requests to prepare inputs for.

        Returns:
            A dictionary containing the prepared inputs for the ShardedModel.
            The dictionary contains "prefill_batch" and "decode_batch",
            with the prepared inputs for the corresponding request type.

            For now we process prefill and decode requests separately.
            Later when we have Ragged Paged Flash Attention kernel,
            we can process both in one batch.
        """
        if len(batched_requests) == 0:
            return None

        prefill_reqs: List[Request] = []
        decode_reqs: List[Request] = []
        for req in batched_requests:
            if req.is_prefill:
                prefill_reqs.append(req)
            elif req.is_decoding:
                decode_reqs.append(req)
        prefill_batch = self._prepare_prefill_batch(prefill_reqs)
        decode_batch = self._prepare_decode_batch(decode_reqs)
        if prefill_batch is None and decode_batch is None:
            return None
        if prefill_batch is not None:
            logger.debug(f"Prepared prefill batch with {len(prefill_batch['requests'])} requests.")
        if decode_batch is not None:
            logger.debug(f"Prepared decode batch with {len(decode_batch['requests'])} requests.")
        return {
            "prefill_batch": prefill_batch,
            "decode_batch": decode_batch,
        }

    @staticmethod
    def _sum_token_counts(value: Any) -> int:
        """Convert backend tensor/list token counts without importing another tensor runtime."""

        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return sum(BaseExecutor._sum_token_counts(item) for item in value)
        return int(value)

    @staticmethod
    def _processed_token_count(batch_type: str, prepared_inputs: Dict[str, Any]) -> Optional[int]:
        """Return backend-scheduled tokens, or ``None`` when no exact counter is exposed."""

        requests = prepared_inputs.get("requests") or []
        if batch_type == "decode_batch":
            # Every maintained backend schedules exactly one decode token per request here.
            return len(requests) or None

        actual_lengths = prepared_inputs.get("actual_processed_lengths")
        if actual_lengths is not None:
            return BaseExecutor._sum_token_counts(actual_lengths)
        scheduler_output = prepared_inputs.get("scheduler_output")
        scheduled = getattr(scheduler_output, "total_num_scheduled_tokens", None)
        if scheduled is not None:
            return int(scheduled)
        if (
            "forward_batch" in prepared_inputs
            and prepared_inputs.get("context_lengths") is not None
        ):
            # SGLang's prefill adapter stores the scheduled extend lengths here (not total
            # sequence lengths as the vLLM adapter does).
            return BaseExecutor._sum_token_counts(prepared_inputs["context_lengths"])
        return None

    def prepare_next_batch_requests(
        self, requests: List[Request], batch_output: Any, context_lengths: Any
    ) -> List[Request]:
        """Prepares a batch of requests for the next stage of the pipeline.

        Args:
            requests: List of requests in the batch
            batch_output: Output from process_batch. Always a dict with:
                - 'hidden_states': token IDs (last peer) or hidden states tensor (intermediate peer)
                - 'probs': list of probabilities (last peer) or None (intermediate peer)
            context_lengths: Context lengths for each request
        """
        # Extract hidden_states and probs from output (always a dict now)
        assert isinstance(batch_output, dict), (
            f"Expected dict from process_batch, got {type(batch_output)}"
        )
        hidden_states = batch_output["hidden_states"]
        token_probs = batch_output["probs"]

        batched_requests = []
        pre_length = 0
        for i, src_request in enumerate(requests):
            if self.is_last_peer:
                # Last peer gets a 1D array of token IDs
                hidden_state_for_req = hidden_states[i : i + 1]
            else:
                # Other peers get a 3D array of hidden states
                if src_request.is_prefill:
                    true_length = int(context_lengths[i])
                    if hidden_states.ndim == 3:
                        hidden_state_for_req = hidden_states[i, :true_length, :]
                    else:
                        hidden_state_for_req = hidden_states[
                            pre_length : pre_length + true_length, :
                        ]
                    pre_length += true_length
                else:
                    if hidden_states.ndim == 3:
                        hidden_state_for_req = hidden_states[i, :, :]
                    else:
                        hidden_state_for_req = hidden_states[pre_length : pre_length + 1, :]
                    pre_length += 1

            # Get prob for this request if available
            token_prob = (
                token_probs[i]
                if (self.is_last_peer and token_probs and i < len(token_probs))
                else None
            )

            next_req = self._prepare_next_single_request(
                src_request, hidden_state_for_req, token_prob
            )
            batched_requests.append(next_req)

        return batched_requests

    def release_and_evict_request(self, rid: str):
        """Release per-request resources and evict from scheduler. Best-effort, never raises."""
        # Release resources
        try:
            self._release_request(rid)
        finally:
            # Cooperative abort markers are generation-scoped. Clear them
            # for every backend so an inactive request cannot leave unbounded
            # manager keys or poison a later replay identity.
            shared_state = getattr(self, "shared_state", None)
            if shared_state is not None:
                shared_state.clear_request_abort(rid)

        # Evict from scheduler
        try:
            self.scheduler.evict_request(rid)
        except Exception:
            pass

    @staticmethod
    def apply_peer_terminal_status(original_request: Request, peer_request: Request) -> None:
        """Apply a terminal status received from another pipeline shard."""
        if getattr(peer_request, "terminal_error", False):
            original_request.terminal_error = True
            original_request.update_status(RequestStatus.ERROR)
        elif peer_request.abort:
            original_request.abort = True

    def fail_batch(self, requests: List[Request]) -> None:
        """Release a failed batch and propagate ERROR across the whole pipeline."""
        for req in requests:
            req.terminal_error = True
            req.update_status(RequestStatus.ERROR)
            self.release_and_evict_request(req.request_id)
            if self.tp_rank != 0:
                continue
            if not (self.is_first_peer and self.is_last_peer):
                self.finished_batch.append(req)
            if self.is_first_peer:
                self._send_engine_core_terminal_output(
                    request_id=req.request_id,
                    finish_reason=EngineCoreFinishReason.ERROR,
                )

    def abort_batch(self, requests: List[Request]) -> None:
        """Commit cooperative cancellation as ABORT, never as backend failure."""

        for req in requests:
            req.abort = True
            req.update_status(RequestStatus.FINISHED_ABORT)
            self.release_and_evict_request(req.request_id)
            if self.tp_rank != 0:
                continue
            if not (self.is_first_peer and self.is_last_peer):
                self.finished_batch.append(req)
            if self.is_first_peer:
                self._send_engine_core_terminal_output(
                    request_id=req.request_id,
                    finish_reason=EngineCoreFinishReason.ABORT,
                )

    def run_loop(self):
        """The main loop of the executor."""
        logger.debug(
            f"Executor for layers [{self.start_layer}, {self.end_layer}) starting run loop..."
        )
        self._should_stop = False
        while not self._should_stop:
            received_requests = []

            # Checkpoint requests are serviced only between model calls. This
            # gives the native page a coherent token boundary without racing
            # the backend's KV mutation.
            self.process_executor_control_requests()

            # Receive requests from the Rust frontend.
            if self.is_first_peer:
                received_requests = self.recv_requests_from_http()

            # Receive requests from peer
            incoming_requests, refit_weight_path = self.recv_requests_from_peer()
            received_requests.extend(incoming_requests)
            if self.enable_weight_refit:
                self.check_and_refit_weight(refit_weight_path)

            self.handle_input_requests(received_requests)
            # Send abort signals to P2P server to broadcast to all nodes
            if len(self.finished_batch) > 0 and self.tp_rank == 0:
                self.send_to_peer_socket.send_multipart(
                    [b"abort", abort_request_to_proto(self.finished_batch).SerializeToString()]
                )
                self.finished_batch = []

            # Check for layer reallocation signal (before batch processing)
            layer_changed = False
            if self.shared_state is not None:
                layer_changed = self.shared_state.get_layer_allocation_changed()

            if layer_changed:
                logger.info(
                    "Layer reallocation detected. Stopping executor to reload with new layers."
                )
                self._should_stop = True
                break

            # 5. Admit requests into running set up to capacity, then form batch
            self.scheduler.admit_requests()
            unschedulable_prefills = self.scheduler.unschedulable_prefills()
            if unschedulable_prefills:
                logger.error(
                    "Rejecting prefills larger than the backend's unchunked token "
                    "budget: request_ids=%s max_num_tokens_per_batch=%d",
                    [request.request_id for request in unschedulable_prefills],
                    self.scheduler.max_num_tokens_per_batch,
                )
                self.fail_batch(unschedulable_prefills)
            # 5.1 Check for request timeouts and abort timed out requests
            try:
                timed_out_reqs = self.scheduler.get_timed_out_requests()
                if timed_out_reqs:
                    for req in timed_out_reqs:
                        rid = req.request_id
                        logger.warning(
                            f"Request {rid} exceeded timeout ({req.timeout_s}s). Aborting and releasing resources."
                        )
                        self.release_and_evict_request(rid)

                        # Notify downstream peers to abort if this peer is the first peer in a pipeline
                        if self.is_first_peer and not self.is_last_peer:
                            self.finished_batch.append(req)
                        if self.is_first_peer and self.tp_rank == 0:
                            self._send_engine_core_terminal_output(
                                request_id=rid,
                                finish_reason=EngineCoreFinishReason.ABORT,
                            )
            except Exception:
                # Non-fatal; continue serving
                pass
            batch_to_process = self.scheduler.form_batch()
            if not batch_to_process:
                continue
            logger.debug(f"Formed batch with {len(batch_to_process)} requests.")

            # 6. Process the batch
            try:
                prepared_inputs_dict = self.prepare_batch_inputs(batch_to_process)

                # We will process prefill and decode batches separately for now
                for batch_type in ["prefill_batch", "decode_batch"]:
                    if prepared_inputs_dict and prepared_inputs_dict.get(batch_type):
                        prepared_inputs = prepared_inputs_dict[batch_type]

                        start_time = time.perf_counter()
                        output = self.process_batch(
                            prepared_inputs, return_decoded_tokens=self.is_last_peer
                        )
                        elapsed_seconds = max(time.perf_counter() - start_time, 1e-9)
                        processed_tokens = self._processed_token_count(batch_type, prepared_inputs)
                        if (
                            batch_type == "prefill_batch"
                            and processed_tokens is not None
                            and processed_tokens > 0
                            and self.tp_rank == 0
                            and self.shared_state is not None
                        ):
                            self.shared_state.update_metrics(
                                prefill_tokens_per_second_sample=(
                                    processed_tokens / elapsed_seconds
                                )
                            )
                        # Update metrics with per-layer latency sample (throttled by decode steps)
                        if batch_type == "decode_batch":
                            try:
                                self._decode_steps_since_metric += len(prepared_inputs["requests"])
                                if (
                                    self._decode_steps_since_metric
                                    >= self.layer_latency_update_every
                                ):
                                    elapsed_ms = elapsed_seconds * 1000.0
                                    assert self.num_shard_layers > 0
                                    per_layer_ms = elapsed_ms / float(self.num_shard_layers)
                                    if self.shared_state is not None and self.tp_rank == 0:
                                        self.shared_state.update_metrics(
                                            layer_latency_ms_sample=per_layer_ms,
                                            decode_tokens_per_second_sample=(
                                                processed_tokens / elapsed_seconds
                                                if processed_tokens is not None
                                                and processed_tokens > 0
                                                else None
                                            ),
                                        )
                                    self._decode_steps_since_metric = 0
                            except Exception:
                                pass
                        # 7. Prepare requests for the next stage in the pipeline
                        next_batch = self.prepare_next_batch_requests(
                            requests=prepared_inputs["requests"],
                            batch_output=output,
                            context_lengths=prepared_inputs.get("context_lengths"),
                        )

                        # 8. Dispatch to the appropriate destination
                        if self.is_last_peer and self.is_first_peer:
                            # Single node: handle locally
                            if next_batch:
                                self.handle_input_requests(next_batch)
                        elif self.tp_rank == 0:
                            if not next_batch:
                                continue
                            # Send output to next peer
                            self.send_to_peer_socket.send_multipart(
                                [
                                    b"forward",
                                    request_to_proto(next_batch, self.device).SerializeToString(),
                                ]
                            )
                            logger.debug(
                                f"Processed batch of type {batch_type} with {len(next_batch)} requests "
                                f"in {(time.time() - start_time) * 1000:.3f} ms"
                            )

            except ExecutorBatchCancelled as cancelled:
                request_ids = set(cancelled.request_ids)
                cancelled_requests = [
                    request for request in batch_to_process if request.request_id in request_ids
                ]
                logger.info(
                    "Native execution cooperatively cancelled requests %s",
                    sorted(request_ids),
                )
                self.abort_batch(cancelled_requests)
            except Exception as e:
                logger.exception(f"Error processing batch: {e}")
                self.fail_batch(batch_to_process)

    def run_loop_in_background(self):
        """Run the executor loop in the background."""

    def shutdown(self):
        """Shuts down the executor."""
        logger.debug("Executor shutting down...")
        self._should_stop = True
        import time

        time.sleep(0.1)  # Give run_loop a moment to exit gracefully

        try:
            all_requests = [req for _, _, _, req in self.scheduler._request_queue] + list(
                self.scheduler._running_requests.values()
            )
            for req in all_requests:
                try:
                    self.scheduler.evict_request(req.request_id, RequestStatus.CANCELLED)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if self.tp_rank == 0:
                for socket_name in (
                    "recv_from_peer_socket",
                    "send_to_peer_socket",
                    "recv_from_ipc_socket",
                    "send_to_ipc_socket",
                    "executor_control_socket",
                ):
                    socket = getattr(self, socket_name, None)
                    if socket is not None:
                        socket.close()
                self.zmq_context.term()
        except Exception as e:
            logger.debug(f"Error closing sockets (may already be closed): {e}")

        logger.debug("Executor shutdown complete.")

    def _prepare_next_single_request(
        self, request: Request, hidden_states: Any, token_prob: Optional[float] = None
    ) -> Request:
        """Handle request state changes both inter and intra peers.

        This function prepares the request object to be sent to the *next* peer in the
        pipeline, or back to the first peer if this is the last peer.

        Args:
            request: The request that was just processed by this peer.
            hidden_states: The output hidden_states/output_ids from the model for this request.
            token_prob: The probability value for the sampled token (optional).

        Returns:
            A new Request object ready to be sent to the next destination.
        """
        # This peer is the last peer or a single node.
        if self.is_last_peer and self.is_first_peer:
            assert isinstance(request, (InitialRequest, IntermediateRequest)), (
                "Invalid request type for decoding."
            )

            next_token_id, hidden_states = self._gen_token_id_from_hidden(hidden_states)
            return IntermediateRequest(
                request_id=request.request_id,
                status=RequestStatus.DECODING,
                current_position=request.total_length + 1,
                input_ids=request.origin_input_ids,
                hidden_states=hidden_states,
                next_token_id=next_token_id,
                routing_table=request.routing_table,
                route_id=request.route_id,
                route_epoch=request.route_epoch,
                authority_request_id=request.authority_request_id,
                lora_path=request.lora_path,
                token_prob=token_prob,
            )
        if self.is_last_peer:
            # Last peer decodes a token and sends it back to the first peer.
            # The token is wrapped in an IntermediateRequest.
            assert isinstance(request, IntermediateRequest), (
                "Last peer must receive an IntermediateRequest."
            )

            next_token_id, hidden_states = self._gen_token_id_from_hidden(hidden_states)
            return IntermediateRequest(
                request_id=request.request_id,
                status=RequestStatus.DECODING,  # Last peer always changes status to DECODING
                current_position=request.total_length,
                input_ids=request.origin_input_ids,
                hidden_states=hidden_states,
                next_token_id=next_token_id,
                routing_table=request.routing_table,
                route_id=request.route_id,
                route_epoch=request.route_epoch,
                authority_request_id=request.authority_request_id,
                lora_path=request.lora_path,
                token_prob=token_prob,
            )
        # This peer is the first or an intermediate peer.
        if self.is_first_peer:
            assert isinstance(request, InitialRequest), "First peer must process an InitialRequest."
            if request.is_finished:
                hidden_states = None
            return IntermediateRequest.from_initial_request(
                request, hidden_states=hidden_states, lora_path=request.lora_path
            )
        assert isinstance(request, IntermediateRequest), (
            "Intermediate peer must process an IntermediateRequest."
        )
        return IntermediateRequest.from_intermediate_request(
            request, hidden_states, lora_path=request.lora_path
        )
