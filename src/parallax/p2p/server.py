"""
P2P server for Parallax.

This module contains the P2P server for Parallax.

It is used to handle the communication between the peers, and communicate with the executor by zmq.

"""

import copy
import dataclasses
import enum
import multiprocessing
import os
import random
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

import dijkstar
import httpx
import zmq
from lattica import ConnectionHandler, Lattica, rpc_method, rpc_stream, rpc_stream_iter

from backend.server.openai_compat import encode_http_response_envelope
from backend.server.rpc_connection_handler import RPCConnectionHandler
from fabi_network.rpc import authenticated_rpc_peer_id
from fabi_network.transport import IrohTransport, using_iroh
from parallax.p2p.liveness import (
    WORKER_HEARTBEAT_INTERVAL_SECONDS,
    WORKER_HEARTBEAT_RPC_TIMEOUT_SECONDS,
)
from parallax.p2p.proto import forward_pb2
from parallax.p2p.utils import (
    AsyncWorker,
    log_nat_traversal_preflight,
    mdns_enabled_for_topology,
)
from parallax.server.server_info import detect_node_hardware
from parallax.server.vllm_rust_frontend import vllm_rust_frontend_available
from parallax.utils.shared_state import SharedState
from parallax.utils.utils import get_zmq_socket
from parallax.utils.weight_refit_utils import (
    calculate_cid_manual,
    concat_weight_partition,
    filer_weight_cid_list,
    parse_safetensors_from_memory,
    release_disk_storage,
)
from parallax_utils.logging_config import get_logger, set_log_level
from swarm_protocol.contracts import (
    BackendKind,
    LayerSpan,
    LinkMetric,
    ModelMemberAdvertisement,
    PathKind,
    ReservationState,
)
from swarm_protocol.execution import WorkerExecutionAdmission
from swarm_protocol.execution_rpc import WorkerExecutionControlService
from swarm_protocol.worker_integration import (
    WorkerProtocolV3Reporter,
    WorkerServingSnapshot,
)
from swarm_protocol.worker_placement import AutonomousWorkerPlacement

logger = get_logger(__name__)

# Global HTTP client for reuse
_http_client = None

_DEFAULT_LINK_PROBE_BYTES = 4 * 1024 * 1024
_MIN_LINK_PROBE_BYTES = 64 * 1024
_MAX_LINK_PROBE_BYTES = 8 * 1024 * 1024
_LINK_PROBE_INTERVAL_SECONDS = 60.0
_LINK_PROBE_MIN_RECEIVE_INTERVAL_SECONDS = 30.0
_LINK_METRIC_TTL_MS = 120_000
_LINK_HEALTH_RPC_TIMEOUT_SECONDS = 5.0
_LINK_HEALTH_PROBE_INTERVAL_SECONDS = 5.0
_LINK_HEALTH_FAILURE_THRESHOLD = 3
# A structural observation must outlive the worst-case failure-detector cycle
# plus one worker heartbeat and one probe interval for publication.  This is a
# soft discovery lease only: request RPC failures still fail immediately.
_LINK_REACHABILITY_TTL_MS = int(
    (
        _LINK_HEALTH_FAILURE_THRESHOLD
        * (_LINK_HEALTH_RPC_TIMEOUT_SECONDS + _LINK_HEALTH_PROBE_INTERVAL_SECONDS)
        + WORKER_HEARTBEAT_INTERVAL_SECONDS
        + _LINK_HEALTH_PROBE_INTERVAL_SECONDS
    )
    * 1_000
)


def _configured_link_probe_bytes() -> int:
    """Return the bounded application payload used for cold-link calibration."""

    raw = os.environ.get("FABI_LINK_PROBE_BYTES", str(_DEFAULT_LINK_PROBE_BYTES)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("FABI_LINK_PROBE_BYTES must be an integer") from exc
    if not _MIN_LINK_PROBE_BYTES <= value <= _MAX_LINK_PROBE_BYTES:
        raise ValueError(
            "FABI_LINK_PROBE_BYTES must be between "
            f"{_MIN_LINK_PROBE_BYTES} and {_MAX_LINK_PROBE_BYTES}"
        )
    return value


def _resolve_worker_key_path() -> str:
    """Return the persistent private directory used for the worker peer key."""
    configured = os.environ.get("PARALLAX_KEY_PATH", "").strip()
    key_path = os.path.abspath(os.path.expanduser(configured or "~/.parallax"))
    os.makedirs(key_path, mode=0o700, exist_ok=True)
    try:
        os.chmod(key_path, 0o700)
    except OSError:
        logger.debug("Could not tighten permissions on %s", key_path, exc_info=True)
    return key_path


def _transfer_metrics(payload_bytes: int, elapsed_ns: int) -> tuple[float, float, float]:
    """Return size, duration, and throughput without a zero-duration division."""
    elapsed_ns = max(elapsed_ns, 1)
    size_mb = payload_bytes / (1024 * 1024)
    elapsed_ms = elapsed_ns / 1_000_000
    speed_mb_s = size_mb / (elapsed_ns / 1_000_000_000)
    return size_mb, elapsed_ms, speed_mb_s


async def get_http_client():
    """Get or create a shared HTTP client"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(3),  # 3 second timeout
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=100),
        )
    return _http_client


class ServerState(enum.Enum):
    """Server state enum."""

    JOINING = "joining"
    INITIALIZING = "initializing"
    READY = "ready"
    OFFLINE = "offline"
    ERROR = "error"


@dataclasses.dataclass
class ServerInfo:
    """Server info data class."""

    state: ServerState
    throughput: Optional[float] = None
    max_batch_size: Optional[int] = None
    max_sequence_len: Optional[int] = None
    error_message: Optional[str] = None


def send_notify(notify_url, block_start_index, block_end_index, request, status):
    # Notifications are optional observability. In scheduler mode a worker can
    # start in standby without a layer span, so do not evaluate span-derived
    # fields unless a notification sink is actually configured.
    if notify_url is None:
        return
    if block_start_index is None or block_end_index is None:
        logger.warning(
            "Skipping %s notification because the serving span is not assigned",
            status,
        )
        return

    payload = [
        {
            "session_id": req.rid,
            "step_id": req.output_length + (block_start_index == 0 and status == "started"),
            "block_idx": block_start_index,
            "total_blocks": block_end_index - block_start_index,
            "status": status,
        }
        for req in request.reqs
    ]

    logger.info(f"Send {status} notification, batch size: {len(payload)}")

    async def send_async(notify_url, payload):
        try:
            client = await get_http_client()
            await client.post(notify_url, json=payload)
        except Exception as e:
            logger.exception(f"Error in send_async: {e}")

    if not hasattr(send_notify, "async_worker"):
        send_notify.async_worker = AsyncWorker()
    send_notify.async_worker.run_coroutine(send_async(notify_url, payload), return_future=True)


class TransformerConnectionHandler(ConnectionHandler):
    """
    Handles RPC requests from clients, forwarding them to the appropriate TransformerBackend.
    Inherits from hivemind's ConnectionHandler.
    """

    def __init__(
        self,
        lattica: Optional[Lattica],
        recv_from_peer_addr: str,
        send_to_peer_addr: str,
        block_start_index: int,
        block_end_index: int,
        http_port: Optional[int] = None,
        notify_url: Optional[str] = None,
        iroh_transport: Optional[IrohTransport] = None,
        execution_admission: Optional[WorkerExecutionAdmission] = None,
        link_probe_authorizer: Optional[Callable[[str], bool]] = None,
    ):
        if lattica is not None:
            super().__init__(lattica)
        if lattica is None and iroh_transport is None:
            raise ValueError("either lattica or iroh_transport must be provided")
        self.iroh_transport = iroh_transport
        self.recv_from_peer_addr = recv_from_peer_addr
        self.send_to_peer_addr = send_to_peer_addr
        self.block_start_index = block_start_index
        self.block_end_index = block_end_index
        self.http_port = http_port
        self.notify_url = notify_url
        self.execution_admission = execution_admission
        self.link_probe_authorizer = link_probe_authorizer
        self._link_probe_lock = threading.Lock()
        self._link_probe_last_received: dict[str, float] = {}
        self._recv_from_peer = None
        self._recv_from_peer_lock = threading.Lock()

    def get_stub(self, peer_id: str):
        if getattr(self, "iroh_transport", None) is not None:
            return self.iroh_transport.stub(peer_id, type(self))
        return super().get_stub(peer_id)

    @property
    def recv_from_peer(self):
        if self._recv_from_peer is None:
            self._recv_from_peer = get_zmq_socket(
                zmq.Context(2), zmq.PUSH, self.recv_from_peer_addr, True
            )
        return self._recv_from_peer

    def update_serving_span(self, block_start_index: int, block_end_index: int) -> None:
        """Atomically refresh metadata copied into the long-lived RPC handler."""
        with self._recv_from_peer_lock:
            self.block_start_index = block_start_index
            self.block_end_index = block_end_index

    @rpc_stream
    def rpc_pp_forward(
        self,
        request: forward_pb2.ForwardRequest,
    ) -> forward_pb2.ForwardResponse:
        """Handle forward pass request with explicit proxy tensors support"""
        if getattr(self, "execution_admission", None) is not None:
            caller = authenticated_rpc_peer_id()
            for req in request.reqs:
                self.execution_admission.authorize_forward(
                    request_id=req.authority_request_id or req.rid,
                    route_id=req.route_id,
                    epoch=req.route_epoch,
                    routing_table=tuple(req.routing_table),
                    caller_endpoint_id=caller,
                )

        # The local enqueue is the data-plane operation. Do it before optional
        # telemetry and let failures propagate to the remote RPC caller instead
        # of returning a false successful response.
        with self._recv_from_peer_lock:
            self.recv_from_peer.send_multipart([b"forward", request.SerializeToString()])
            block_start_index = self.block_start_index
            block_end_index = self.block_end_index

        try:
            send_notify(self.notify_url, block_start_index, block_end_index, request, "started")
        except Exception as e:
            logger.warning("Failed to emit forward notification: %s", e, exc_info=True)
        return forward_pb2.ForwardResponse()

    @rpc_method
    def rpc_health(self, request):
        """Return this peer's authenticated transport identity."""
        del request
        if getattr(self, "iroh_transport", None) is not None:
            return {"peer_id": self.iroh_transport.peer_id()}
        return {"peer_id": self.lattica_instance.peer_id()}

    @rpc_method
    def rpc_link_probe(self, request):
        """Accept one bounded upload from an assigned, authenticated peer.

        The caller measures end-to-end application goodput. The body is never
        echoed, hashed or retained, so calibration cannot amplify traffic or
        compete with model memory after this method returns.
        """

        if getattr(self, "iroh_transport", None) is None:
            raise RuntimeError("link calibration requires the authenticated Iroh transport")
        caller = authenticated_rpc_peer_id()
        authorizer = getattr(self, "link_probe_authorizer", None)
        if authorizer is None or not authorizer(caller):
            raise PermissionError("link calibration caller is not an assigned peer")
        if not isinstance(request, bytes):
            raise TypeError("link calibration payload must be bytes")
        if not _MIN_LINK_PROBE_BYTES <= len(request) <= _MAX_LINK_PROBE_BYTES:
            raise ValueError("link calibration payload size is outside the permitted bounds")

        now = time.monotonic()
        with self._link_probe_lock:
            previous = self._link_probe_last_received.get(caller)
            if previous is not None and now - previous < _LINK_PROBE_MIN_RECEIVE_INTERVAL_SECONDS:
                raise RuntimeError("link calibration is rate limited")
            self._link_probe_last_received[caller] = now
        return {
            "peer_id": self.iroh_transport.peer_id(),
            "received_bytes": len(request),
        }

    @rpc_method
    def rpc_abort(
        self,
        request: forward_pb2.AbortRequest,
    ) -> forward_pb2.AbortResponse:
        if getattr(self, "execution_admission", None) is not None:
            caller = authenticated_rpc_peer_id()
            for req in request.reqs:
                self.execution_admission.authorize_route_peer(
                    request_id=req.authority_request_id or req.rid,
                    route_id=req.route_id,
                    epoch=req.route_epoch,
                    routing_table=tuple(req.routing_table),
                    caller_endpoint_id=caller,
                )
        with self._recv_from_peer_lock:
            self.recv_from_peer.send_multipart([b"abort", request.SerializeToString()])
        return forward_pb2.AbortResponse()

    @rpc_method
    def abort_completion(self, request):
        """Abort a frontend request through vLLM's maintained engine API.

        Stream cancellation remains useful for transport cleanup, but it is
        not a sufficient engine control signal when the producer is stalled
        between chunks. This RPC is deliberately route-fenced and available
        only to the authenticated coordinator while the reservation is still
        committed.
        """

        if self.execution_admission is None:
            raise PermissionError("explicit completion abort requires active protocol v3")
        if not isinstance(request, dict):
            raise TypeError("completion abort request must be an object")
        request_id = request.get("request_id")
        xargs = request.get("vllm_xargs")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
            raise ValueError("completion abort request_id is invalid")
        if not isinstance(xargs, dict):
            raise PermissionError("completion abort is missing route authority")

        self.execution_admission.authorize_frontend(
            request_id=request_id,
            route_id=str(xargs.get("fabi_route_id", "")),
            epoch=int(xargs.get("fabi_route_epoch", 0)),
            routing_table=tuple(xargs.get("parallax_routing_table", ())),
            caller_endpoint_id=authenticated_rpc_peer_id(),
        )
        if self.http_port is None:
            raise RuntimeError("route head has no local HTTP frontend")

        vllm_request_id = f"chatcmpl-{request_id}"
        with httpx.Client(
            timeout=httpx.Timeout(5.0),
            proxy=None,
            trust_env=False,
        ) as client:
            response = client.post(
                f"http://localhost:{self.http_port}/abort_requests",
                json={"request_ids": [vllm_request_id]},
            )
            response.raise_for_status()
        logger.info("Explicitly aborted frontend request %s", request_id)
        return {"aborted": True, "request_id": request_id}

    def ipc_weight_refit(
        self,
        refit_weight_path: str,
        weight_version: int,
    ):
        encoded_weight_version = str(weight_version).encode("ascii")
        encoded_refit_weight_path = refit_weight_path.encode("ascii")
        try:
            with self._recv_from_peer_lock:
                self.recv_from_peer.send_multipart(
                    [b"refit", encoded_refit_weight_path, encoded_weight_version]
                )
        except Exception as e:
            logger.exception(f"Error in ipc_weight_refit: {e}")

    @rpc_stream_iter
    def chat_completion(
        self,
        request,
    ):
        """Handle chat completion request"""
        logger.debug(f"Chat completion request: {request}, type: {type(request)}")
        try:
            if getattr(self, "execution_admission", None) is not None:
                xargs = request.get("vllm_xargs")
                if not isinstance(xargs, dict):
                    raise PermissionError("active v3 request is missing route authority")
                self.execution_admission.authorize_frontend(
                    request_id=str(request.get("request_id", "")),
                    route_id=str(xargs.get("fabi_route_id", "")),
                    epoch=int(xargs.get("fabi_route_epoch", 0)),
                    routing_table=tuple(xargs.get("parallax_routing_table", ())),
                    caller_endpoint_id=authenticated_rpc_peer_id(),
                )
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
                    )
                    yield encode_http_response_envelope(
                        status_code=response.status_code,
                        content_type=response.headers.get("content-type"),
                        body=response.content,
                    )
        except Exception as e:
            logger.exception(f"Error in chat completion: {e}")
            yield encode_http_response_envelope(
                status_code=502,
                content_type="application/json",
                body=(
                    b'{"error":{"message":"Internal server error",'
                    b'"type":"upstream_error","param":null,"code":"upstream_error"}}'
                ),
            )


def check_and_run_weight_refit(gradient_server, message):
    """
    Check and trigger weight refit process.
    Received message is a Dict which at least contains:
        time_stamp: float,      indicating weight refit trigger time.
        cid:        List[str],  cid list.
        index_map:  Dict[str],  key(weight_name): value(cid)
    """

    def _download_weight_thread(cid):
        raw_data = None
        time_out = 20 * 60  # 20 minutes timeout
        time_begin_get_block = time.time()
        time_end_get_block = None
        peer_id = None
        while True:
            try:
                cur_time = time.time()
                if cur_time - time_begin_get_block > time_out:
                    logger.warning(f"Failed to get_block after 10 minutes! cid={cid}")
                    return False, {}
                peer_id, raw_data = gradient_server.lattica.get_block(cid, timeout_secs=30)
                cid_manual = calculate_cid_manual(raw_data)
                if cid_manual != cid:
                    logger.warning(f"Checksum failed. Retry get_block for cid={cid}")
                    continue
                else:
                    time_end_get_block = time.time()
                    break
            except Exception:
                logger.warning(f"Failed to get block: {cid}. Retry in 1 second.")
                time.sleep(1)
        if raw_data is None:
            raise RuntimeError(f"Failed to get block cid={cid}")
        interval_get_block = time_end_get_block - time_begin_get_block
        logger.info(
            f"Finish download cid={cid}, get_block={interval_get_block}s, peer_id={peer_id}"
        )
        # convert raw data to dict
        tensors = parse_safetensors_from_memory(raw_data)
        return True, tensors

    # step0. Release lattica disk storage
    release_disk_storage()

    # step1. Check weight refit trigger message
    time_stamp = message.get("time_stamp", None)
    index_map = message.get("index_map", None)
    weight_version = message.get("version", 0)
    if time_stamp is None or index_map is None:
        return
    if gradient_server.last_refit_time >= float(time_stamp):
        # Weight already updated
        return

    cid_list = filer_weight_cid_list(
        gradient_server.block_start_index,
        gradient_server.block_end_index,
        gradient_server.block_end_index,
        index_map,
    )
    random.seed(time.time())
    random.shuffle(cid_list)

    # add sleep 10s for direct connection first
    logger.debug(f"Received weight refit message: {message}.")
    logger.info(f"Start dealing weight refit version: {weight_version}.")

    # step2. download weight
    weight_dir = os.path.join("/tmp", str(time_stamp))
    folder = os.path.exists(weight_dir)
    if not folder:
        os.makedirs(weight_dir)
        download_res = True
        tensors = {}
        while True:
            if len(cid_list) == 0:
                break
            else:
                cid = cid_list.pop()
                logger.info(f"Start downloading refit weight {cid}")
                res, tensors_loaded = _download_weight_thread(cid)
                if res:
                    tensors.update(tensors_loaded)
                else:
                    download_res = False
                    break

        if not download_res:
            gradient_server.last_refit_time = float(time_stamp)
            logger.info("Error in updating weight. Still holds the previous version of weight.")

        # step3. concat weight
        # workaround: create sub-process to avoid GIL issues for lattica
        logger.info(f"Start sub-process to concat weight partitions in {weight_dir}")
        if gradient_server.weight_refit_mode == "cpu":
            new_tensors = concat_weight_partition(tensors)
            gradient_server.conn.send(new_tensors)
        elif gradient_server.weight_refit_mode == "disk":
            concat_weight_partition(tensors, weight_dir)
        else:
            logger.warning(f"Unrecognized weight refit mode: {gradient_server.weight_refit_mode}")

        # step4. send ipc message to update weight
        gradient_server.connection_handler.ipc_weight_refit(weight_dir, weight_version)
        last_refit_time = float(time_stamp)
        gradient_server.last_refit_time = last_refit_time
        gradient_server.refit_timestamp_history.append(last_refit_time)
        gradient_server.check_and_release_disk_weight()
        logger.info(
            f"Finish download weight_version={weight_version}, last_refit_time={gradient_server.last_refit_time}"
        )
    else:
        logger.warning(f"Already satisfies weight_version={weight_version}")
    gradient_server.refit_finish = True


class GradientServer:
    """
    Main server class for Parallax.

    This class handles communication between peers and communicates with the executor by zmq.
    """

    def __init__(
        self,
        recv_from_peer_addr: str,
        send_to_peer_addr: str,
        initial_peers: List[str] = [],
        scheduler_addr: Optional[str] = None,
        relay_servers: List[str] = [],
        block_start_index: int = 0,
        block_end_index: int = 1,
        hidden_layers: int = 128,
        tp_size: int = 1,
        dp_size: int = 1,
        dht_prefix: str = "gradient",
        host_maddrs: List[str] = [],
        http_port: Optional[int] = None,
        announce_maddrs: List[str] = [],
        notify_url: str = None,
        model_name: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        param_mem_ratio: float = 0.65,
        kvcache_mem_ratio: float = 0.25,
        gpu_backend: str = "sglang",
        chunked_prefill_size: Optional[int] = None,
        conn: Any = None,
    ):
        self.recv_from_peer_addr = recv_from_peer_addr
        self.send_to_peer_addr = send_to_peer_addr
        self.initial_peers = initial_peers
        self.scheduler_addr = scheduler_addr
        self.relay_servers = relay_servers
        self.block_start_index = block_start_index
        self.block_end_index = block_end_index
        self.hidden_layers = hidden_layers
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.dht_prefix = dht_prefix
        self.host_maddrs = host_maddrs
        self.announce_maddrs = announce_maddrs
        self.http_port = http_port
        self.notify_url = notify_url
        self.model_name = model_name
        self.model_revision = None
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.model_max_sequence_length = None
        self.planned_context_tokens = None
        self.allocation_epoch = None
        self.supports_frontend = vllm_rust_frontend_available()
        self.param_mem_ratio = param_mem_ratio
        self.kvcache_mem_ratio = kvcache_mem_ratio
        self.gpu_backend = gpu_backend
        self.preferred_chunked_prefill_size = (
            0 if chunked_prefill_size is None else int(chunked_prefill_size)
        )
        self.chunked_prefill_size = self.preferred_chunked_prefill_size
        self.supports_chunked_prefill = False
        self.enable_weight_refit = False
        self.weight_refit_mode = "disk"
        self.last_refit_time = 0.0
        self.refit_finish = True
        self.refit_timestamp_history = []
        self.prefix_id = f"{dht_prefix}_announce"
        self.lattica = None
        self.iroh_transport = None
        self.routing_table = None
        self.routing_table_update_interval = 10
        self.server_info = ServerInfo(state=ServerState.JOINING)
        self.stubs = {}
        self.rtts = {}
        self.rtt_last_update = 0
        self.rtt_update_interval = 60
        self.link_throughputs = {}
        self.link_throughputs_lock = threading.Lock()
        self.link_probe_bytes = _configured_link_probe_bytes()
        self.link_probe_payload = bytes(self.link_probe_bytes)
        self.link_probe_last_attempt: dict[str, float] = {}
        self.status = ServerState.JOINING
        self.manual_layer_assignment = block_end_index is not None and block_start_index is not None
        self.conn = conn
        # Account credential is transported only over the encrypted scheduler
        # RPC.  The scheduler immediately hashes it and never logs/stores it.
        self.account_token = os.environ.get("FABI_ACCOUNT_TOKEN") or None
        self.swarm_v3_reporter = None
        self.swarm_v3_execution_admission = None
        self.swarm_v3_placement_controller = None
        self.swarm_v3_placement_mode = (
            os.environ.get(
                "FABI_SWARM_V3_PLACEMENT",
                "legacy",
            )
            .strip()
            .lower()
        )
        if self.swarm_v3_placement_mode not in {"legacy", "autonomous"}:
            raise ValueError("FABI_SWARM_V3_PLACEMENT supports only legacy or autonomous")
        self.swarm_v3_init_error = None
        try:
            self.swarm_v3_reporter = WorkerProtocolV3Reporter.from_environment()
        except Exception as exc:
            # Shadow telemetry must never take the qualified v2 heartbeat or serving path down.
            self.swarm_v3_init_error = {
                "code": type(exc).__name__,
                "detail": str(exc)[:256],
            }
            logger.error("Protocol-v3 shadow reporter is disabled: %s", exc)

        self.scheduler_stub = None
        self.scheduler_peer_id = None
        self.routing_table_updater = None
        self.announcer = None
        self.direct_peer_prober = None
        self.connection_handler = None
        self.outbound_peer_ids = []
        self.authorized_link_peer_ids = []
        self.direct_peer_ids = []
        self.reachable_peer_ids = []
        self.relayed_peer_ids = []
        self.link_path_observed_at_ms = {}
        self.link_health_failures = {}
        self.link_topology_lock = threading.RLock()
        self.stop_event = threading.Event()
        logger.debug(f"manual_layer_assignment: {self.manual_layer_assignment}")
        self._layer_allocation_changed = False
        self._shared_state = None  # Will be set if running in subprocess mode
        # Capacity is a contract for this worker process generation. Re-reading
        # free RAM/VRAM after model materialization would count the loaded
        # weights twice during scheduler recovery. Live pressure is reported
        # independently through SharedState and may pause admission or drain.
        self._capacity_hardware_snapshot = None
        self._capacity_hardware_lock = threading.Lock()

    def _sync_to_shared_state(self):
        """Sync current layer allocation and status to shared state if available"""
        if hasattr(self, "_shared_state") and self._shared_state is not None:
            self._shared_state.update(
                block_start_index=self.block_start_index,
                block_end_index=self.block_end_index,
                model_name=self.model_name,
                model_revision=self.model_revision,
                tp_size=self.tp_size,
                enable_weight_refit=self.enable_weight_refit,
                weight_refit_mode=self.weight_refit_mode,
                model_max_sequence_length=self.model_max_sequence_length,
                planned_context_tokens=self.planned_context_tokens,
                allocation_epoch=self.allocation_epoch,
                chunked_prefill_size=self.chunked_prefill_size,
                status=self.status.value,
                _layer_allocation_changed=self._layer_allocation_changed,
            )

    def _apply_v3_span_reload(self, span: LayerSpan, generation: int) -> None:
        """Fence ingress and hand one autonomous span generation to launch.py."""

        if self._shared_state is None:
            raise RuntimeError("autonomous placement requires shared executor state")
        if self.swarm_v3_execution_admission is None:
            raise RuntimeError("autonomous placement requires worker-local v3 admission")
        previous_start = self.block_start_index
        previous_end = self.block_end_index
        placement_phase = self._shared_state.get("swarm_v3_placement_phase", "legacy")
        self.block_start_index = span.start
        self.block_end_index = span.end
        if self.connection_handler is not None:
            self.connection_handler.update_serving_span(span.start, span.end)
        self._layer_allocation_changed = True
        self.status = ServerState.INITIALIZING
        self._sync_to_shared_state()
        self._shared_state.update(
            swarm_v3_placement_generation=generation,
            swarm_v3_placement_phase="building",
            swarm_v3_placement_error=None,
            swarm_v3_previous_start_layer=(
                previous_start
                if placement_phase != "building"
                else self._shared_state.get("swarm_v3_previous_start_layer")
            ),
            swarm_v3_previous_end_layer=(
                previous_end
                if placement_phase != "building"
                else self._shared_state.get("swarm_v3_previous_end_layer")
            ),
            frontend_alive=False,
        )
        logger.warning(
            "Protocol-v3 placement generation %d is reloading layers [%d, %d)",
            generation,
            span.start,
            span.end,
        )

    def check_and_release_disk_weight(self):
        """Only save 3 history versions of weight"""
        while len(self.refit_timestamp_history) > 3:
            time_stamp = self.refit_timestamp_history.pop(0)
            weight_dir = os.path.join("/tmp", str(int(time_stamp)))
            if os.path.isdir(weight_dir):
                try:
                    shutil.rmtree(weight_dir)
                    logger.info(f"Folder '{weight_dir}' and all its contents have been removed.")
                except OSError as e:
                    logger.exception(f"Error: {weight_dir} : {e.strerror}")
            else:
                logger.warning(f"Folder '{weight_dir}' does not exist.")

    def build_lattica(self):
        if using_iroh():
            return self._build_iroh()

        self.lattica = (
            Lattica.builder()
            .with_listen_addrs(self.host_maddrs)
            .with_key_path(_resolve_worker_key_path())
        )
        mdns_enabled = mdns_enabled_for_topology(
            initial_peers=self.initial_peers,
            relay_servers=self.relay_servers,
        )
        logger.info("mDNS discovery enabled: %s", mdns_enabled)
        if not mdns_enabled:
            self.lattica.with_mdns(False)

        if self.scheduler_addr is not None and self.scheduler_addr != "auto":
            if self.scheduler_addr.startswith("/"):
                logger.info(f"Using scheduler addr: {self.scheduler_addr}")
                self.lattica.with_bootstraps([self.scheduler_addr])
            self.scheduler_peer_id = self.scheduler_addr.split("/")[-1]

        if len(self.relay_servers) > 0:
            logger.info(f"Using relay servers: {self.relay_servers}")
            self.lattica.with_relay_servers(self.relay_servers).with_dcutr(True)
            if self.scheduler_peer_id is not None:
                logger.info(f"Using protocol: /{self.scheduler_peer_id}")
                self.lattica.with_protocol("/" + self.scheduler_peer_id)

        if len(self.announce_maddrs) > 0:
            logger.info(f"Using announce maddrs: {self.announce_maddrs}")
            self.lattica.with_external_addrs(self.announce_maddrs)

        if len(self.initial_peers) > 0:
            logger.info(f"Using initial peers: {self.initial_peers}")
            self.lattica.with_bootstraps(self.initial_peers)

        self.lattica.build()

        if len(self.relay_servers) > 0:
            log_nat_traversal_preflight(self.lattica, logger)

        if self.scheduler_addr == "auto":
            self.scheduler_peer_id = None
            for _ in range(20):
                try:
                    time.sleep(3)
                    self.scheduler_peer_id = self.lattica.get("scheduler_peer_id")
                    if self.scheduler_peer_id is not None:
                        self.scheduler_peer_id = self.scheduler_peer_id.value
                        logger.info(f"Found scheduler peer id: {self.scheduler_peer_id}")
                        break
                    logger.info(
                        f"Discovering scheduler peer id, {_ + 1} times, you can specify scheduler peer id by -s"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get scheduler addr: {e}, waiting for 3 seconds.")
            if self.scheduler_peer_id is None:
                logger.error("Failed to get scheduler peer id")
                return False

        return True

    def _build_iroh(self):
        """Build the worker endpoint for central scheduler mode."""

        if self.scheduler_addr in {None, "auto"}:
            raise ValueError("Iroh workers require an explicit scheduler endpoint ID")
        if str(self.scheduler_addr).startswith("/"):
            raise ValueError("Iroh workers require an endpoint ID, not a Lattica multiaddress")
        self.iroh_transport = IrohTransport.from_environment("worker")
        self.lattica = self.iroh_transport
        self.scheduler_peer_id = str(self.scheduler_addr)
        if (
            getattr(self, "swarm_v3_reporter", None) is not None
            and getattr(self.iroh_transport, "catalog_discovery", None) is not None
        ):
            self.swarm_v3_reporter.attach_catalog(self.iroh_transport.catalog_discovery)
        if (
            getattr(self, "swarm_v3_reporter", None) is not None
            and self.swarm_v3_reporter.mode == "active"
        ):
            from swarm_protocol.epochs import SqliteRequestEpochFence

            fence_path = os.environ.get("FABI_SWARM_V3_FENCE_DB")
            if not fence_path:
                state_dir = Path(
                    os.environ.get(
                        "FABI_SWARM_V3_STATE_DIR",
                        str(Path.home() / ".fabi" / "swarm-v3" / "registry"),
                    )
                )
                fence_path = str(state_dir / "request-fences.sqlite3")
            self.swarm_v3_execution_admission = WorkerExecutionAdmission(
                worker_id=self.iroh_transport.peer_id(),
                endpoint_id=self.iroh_transport.peer_id(),
                coordinator_endpoint_id=self.scheduler_peer_id,
                crypto=self.iroh_transport,
                request_epoch_fence=SqliteRequestEpochFence(fence_path),
            )
        if getattr(self, "swarm_v3_placement_mode", "legacy") == "autonomous" and (
            getattr(self, "swarm_v3_execution_admission", None) is None
            or getattr(self.iroh_transport, "catalog_discovery", None) is None
        ):
            raise RuntimeError(
                "autonomous v3 placement requires active mode and the native catalogue DHT"
            )
        logger.info(
            "Iroh worker endpoint ready: %s (scheduler %s)",
            self.iroh_transport.peer_id(),
            self.scheduler_peer_id,
        )
        return True

    def _qualify_iroh_scheduler(self) -> None:
        """Open and authenticate the scheduler connection before reading telemetry."""

        if self.iroh_transport is None:
            return
        response = self.scheduler_stub.rpc_health({}).result(timeout=30)
        if not isinstance(response, dict) or response.get("peer_id") != self.scheduler_peer_id:
            raise RuntimeError("Iroh scheduler health response has the wrong endpoint identity")
        path = self.iroh_transport.selected_path(self.scheduler_peer_id)
        logger.info("Qualified Iroh scheduler connection: path=%s", path)

    def run(self):
        if self.build_lattica():
            logger.info(
                "%s transport built successfully",
                "Iroh" if self.iroh_transport is not None else "Lattica",
            )
        else:
            logger.error("Failed to build network transport")
            exit(1)

        if self.scheduler_addr is not None:  # central scheduler mode
            try:
                if self.iroh_transport is not None:
                    self.scheduler_stub = self.iroh_transport.stub(
                        self.scheduler_peer_id, RPCConnectionHandler
                    )
                else:
                    self.scheduler_stub = RPCConnectionHandler(self.lattica, None, None).get_stub(
                        self.scheduler_peer_id
                    )
                self._qualify_iroh_scheduler()
                node_info = self.get_node_info()
                if node_info == {}:
                    logger.error("Failed to get node info, try again after 10 seconds")
                    self.lattica.close()
                    self.lattica = None
                    time.sleep(10)
                    return self.run()

                if self.manual_layer_assignment:
                    node_info["manual_layer_assignment"] = True

                response = self.scheduler_stub.node_join(node_info)
                response = response.result(timeout=300)
                if response == {}:
                    logger.error("Failed to join scheduler")
                    exit(1)

                logger.info(f"Join scheduler response: {response}")

                if not self.manual_layer_assignment:
                    self.block_start_index = response.get("start_layer")
                    self.block_end_index = response.get("end_layer")
                self.model_name = response.get("model_name")
                self.model_revision = response.get("model_revision")
                self.tp_size = response.get("tp_size")
                self.enable_weight_refit = response.get("enable_weight_refit")
                if self.iroh_transport is not None and self.enable_weight_refit:
                    raise RuntimeError(
                        "weight refit needs a qualified Iroh content plane and cannot use RPC fallback"
                    )
                self.weight_refit_mode = response.get("weight_refit_mode")
                self.model_max_sequence_length = response.get("model_max_sequence_length")
                self.planned_context_tokens = response.get("planned_context_tokens")
                self.allocation_epoch = response.get("allocation_epoch")
                self.chunked_prefill_size = int(response.get("chunked_prefill_size", 0))
                self._update_outbound_peers(response)

                # Sync to shared state if available
                self._sync_to_shared_state()

            except Exception as e:
                logger.exception(f"Error in join scheduler: {e}")
                exit(1)
        else:  # no scheduler mode
            self.start_routing_table_updater()  # thread

        self.connection_handler = TransformerConnectionHandler(
            lattica=None if self.iroh_transport is not None else self.lattica,
            recv_from_peer_addr=self.recv_from_peer_addr,
            send_to_peer_addr=self.send_to_peer_addr,
            block_start_index=self.block_start_index,
            block_end_index=self.block_end_index,
            http_port=self.http_port,
            notify_url=self.notify_url,
            iroh_transport=self.iroh_transport,
            execution_admission=self.swarm_v3_execution_admission,
            link_probe_authorizer=lambda peer_id: (
                peer_id in self.authorized_link_peer_ids and self._link_probe_idle()
            ),
        )  # thread
        if self.iroh_transport is not None:
            self.iroh_transport.register(self.connection_handler)
            if self.swarm_v3_execution_admission is not None:
                self.iroh_transport.register(
                    WorkerExecutionControlService(self.swarm_v3_execution_admission)
                )

        if self.scheduler_addr is not None:
            self.start_direct_peer_prober()
        self.start_node_announcer()  # thread
        self.start_node_sender()  # main loop

    def find_servers(self):
        """Find available servers in the DHT network"""
        # Find all announced blocks
        server_blocks = []
        block_servers = self.lattica.get(self.prefix_id)
        if block_servers is None:
            return []
        for peer_id, value in block_servers.value.items():
            server_blocks.append(
                {
                    "peer_id": peer_id,
                    "block_start_index": value.value["block_start_index"],
                    "block_end_index": value.value["block_end_index"],
                }
            )

        return server_blocks

    def get_stub(self, peer_id):
        if peer_id not in self.stubs:
            self.stubs[peer_id] = self.connection_handler.get_stub(peer_id)
        return self.stubs[peer_id]

    def _update_outbound_peers(self, allocation):
        """Store scheduler-selected candidates for this shard's next stage."""
        peers = allocation.get("outbound_peer_ids") if isinstance(allocation, dict) else None
        if peers is not None:
            outbound_peer_ids = sorted(
                {str(peer_id) for peer_id in peers if str(peer_id) != self.lattica.peer_id()}
            )
            with self.link_topology_lock:
                self.outbound_peer_ids = outbound_peer_ids
                retained = set(outbound_peer_ids)
                self.direct_peer_ids = sorted(retained.intersection(self.direct_peer_ids))
                self.reachable_peer_ids = sorted(retained.intersection(self.reachable_peer_ids))
                self.relayed_peer_ids = sorted(retained.intersection(self.relayed_peer_ids))
                self.link_path_observed_at_ms = {
                    peer_id: measured_at_ms
                    for peer_id, measured_at_ms in self.link_path_observed_at_ms.items()
                    if peer_id in retained
                }
                self.link_health_failures = {
                    peer_id: failures
                    for peer_id, failures in self.link_health_failures.items()
                    if peer_id in retained
                }
        authorized = (
            allocation.get("authorized_link_peer_ids") if isinstance(allocation, dict) else None
        )
        if authorized is not None:
            self.authorized_link_peer_ids = sorted(
                {str(peer_id) for peer_id in authorized if str(peer_id) != self.lattica.peer_id()}
            )

    def _probe_outbound_peers(self):
        """Refresh authenticated reachability through a temporal failure detector.

        A single congested relay probe must not erase a previously qualified
        edge.  Successful RPCs refresh the structural lease and reset the
        consecutive-failure counter.  Failed probes retain the last known path
        until either the failure threshold or the observation TTL is reached.
        """
        if self.connection_handler is None:
            return None
        if self.stop_event.is_set():
            with self.link_topology_lock:
                return list(self.reachable_peer_ids)

        pending = {}
        with self.link_topology_lock:
            candidates = list(self.outbound_peer_ids)
        for peer_id in candidates:
            try:
                pending[peer_id] = self.get_stub(peer_id).rpc_health({})
            except Exception:
                logger.debug("Could not start direct-path probe to %s", peer_id, exc_info=True)

        successful_paths = {}
        for peer_id, future in pending.items():
            try:
                response = (
                    future.result(timeout=_LINK_HEALTH_RPC_TIMEOUT_SECONDS)
                    if hasattr(future, "result")
                    else future
                )
                if isinstance(response, dict) and response.get("peer_id") == peer_id:
                    if self.iroh_transport is None:
                        successful_paths[peer_id] = "direct"
                    else:
                        path = self.iroh_transport.selected_path(peer_id)
                        if path is not None and path.get("kind") == "direct":
                            successful_paths[peer_id] = "direct"
                        elif path is not None and path.get("kind") == "relay":
                            successful_paths[peer_id] = "relay"
                        else:
                            successful_paths[peer_id] = None
            except Exception:
                logger.debug("Transport probe to %s failed", peer_id, exc_info=True)

        observed_at_ms = time.time_ns() // 1_000_000
        retained_after_failure = {}
        with self.link_topology_lock:
            # A heartbeat may replace the allocation while network futures are
            # in flight. Never resurrect a peer removed by that newer contract.
            candidates = [peer_id for peer_id in candidates if peer_id in self.outbound_peer_ids]
            previous_direct = set(self.direct_peer_ids)
            previous_reachable = set(self.reachable_peer_ids)
            previous_relayed = set(self.relayed_peer_ids)
            direct = set()
            reachable = set()
            relayed = set()

            for peer_id in candidates:
                if peer_id in successful_paths:
                    reachable.add(peer_id)
                    self.link_health_failures[peer_id] = 0
                    self.link_path_observed_at_ms[peer_id] = observed_at_ms
                    path_kind = successful_paths[peer_id]
                    if path_kind == "direct":
                        direct.add(peer_id)
                    elif path_kind == "relay":
                        relayed.add(peer_id)
                    elif peer_id in previous_direct:
                        direct.add(peer_id)
                    elif peer_id in previous_relayed:
                        relayed.add(peer_id)
                    continue

                failures = self.link_health_failures.get(peer_id, 0) + 1
                self.link_health_failures[peer_id] = failures
                last_success_ms = self.link_path_observed_at_ms.get(peer_id)
                observation_is_fresh = (
                    last_success_ms is not None
                    and observed_at_ms - last_success_ms < _LINK_REACHABILITY_TTL_MS
                )
                if (
                    peer_id in previous_reachable
                    and failures < _LINK_HEALTH_FAILURE_THRESHOLD
                    and observation_is_fresh
                ):
                    reachable.add(peer_id)
                    retained_after_failure[peer_id] = failures
                    if peer_id in previous_direct:
                        direct.add(peer_id)
                    elif peer_id in previous_relayed:
                        relayed.add(peer_id)

            retained = set(candidates)
            self.link_path_observed_at_ms = {
                peer_id: measured_at_ms
                for peer_id, measured_at_ms in self.link_path_observed_at_ms.items()
                if peer_id in retained
            }
            self.link_health_failures = {
                peer_id: failures
                for peer_id, failures in self.link_health_failures.items()
                if peer_id in retained
            }
            direct_snapshot = sorted(direct)
            reachable_snapshot = sorted(reachable)
            relayed_snapshot = sorted(relayed)
            changed = (
                direct_snapshot != self.direct_peer_ids
                or reachable_snapshot != self.reachable_peer_ids
                or relayed_snapshot != self.relayed_peer_ids
            )
            self.direct_peer_ids = direct_snapshot
            self.reachable_peer_ids = reachable_snapshot
            self.relayed_peer_ids = relayed_snapshot

        if retained_after_failure:
            logger.debug(
                "Retaining qualified outbound peers after transient probe failures: %s",
                retained_after_failure,
            )
        if changed:
            logger.info(
                "Qualified outbound peers: reachable=%s direct=%s relay=%s expected=%s",
                reachable_snapshot,
                direct_snapshot,
                relayed_snapshot,
                candidates,
            )
        if self.iroh_transport is not None:
            for peer_id in successful_paths:
                self._probe_peer_goodput(peer_id)
        return reachable_snapshot

    def _record_link_goodput(
        self,
        peer_id: str,
        bytes_per_second: float,
        *,
        measured_at_ms: int | None = None,
        application_limited: bool = False,
    ) -> bool:
        """Store a qualified delivery-rate sample.

        A request/response transfer can be application-limited: its measured
        bytes/second then reflects serialization and one RPC round trip rather
        than the path's bulk delivery capacity. As in BBR's delivery-rate
        estimator, such a sample may raise an existing lower bound but must
        never lower it. Periodic bulk probes remain able to move the EWMA in
        either direction.
        """

        if bytes_per_second <= 0:
            return False
        measured_at_ms = measured_at_ms or time.time_ns() // 1_000_000
        with self.link_throughputs_lock:
            previous = self.link_throughputs.get(peer_id)
            if application_limited and (
                previous is None or bytes_per_second <= float(previous["bytes_per_second"])
            ):
                return False
            smoothed = float(bytes_per_second)
            if previous is not None:
                smoothed = 0.8 * float(previous["bytes_per_second"]) + 0.2 * smoothed
            self.link_throughputs[peer_id] = {
                "bytes_per_second": smoothed,
                "measured_at_ms": measured_at_ms,
            }
        return True

    def _link_probe_idle(self) -> bool:
        """Return whether bulk calibration cannot contend with inference."""

        admission = getattr(self, "swarm_v3_execution_admission", None)
        if admission is None:
            return True
        try:
            return not any(
                lease.state in {ReservationState.PREPARED, ReservationState.COMMITTED}
                for lease in admission.snapshot()
            )
        except Exception:
            logger.warning("Unable to inspect execution reservations before link probe", exc_info=True)
            return False

    def _probe_peer_goodput(self, peer_id: str) -> bool:
        """Measure cold-link upload goodput without touching the heartbeat path."""

        if not self._link_probe_idle():
            logger.debug("Skipping bulk link calibration to %s during active inference", peer_id)
            return False
        now = time.monotonic()
        previous_attempt = self.link_probe_last_attempt.get(peer_id)
        if previous_attempt is not None and now - previous_attempt < _LINK_PROBE_INTERVAL_SECONDS:
            return False
        self.link_probe_last_attempt[peer_id] = now

        started_ns = time.perf_counter_ns()
        try:
            response_future = self.get_stub(peer_id).rpc_link_probe(self.link_probe_payload)
            response = (
                response_future.result(timeout=15)
                if hasattr(response_future, "result")
                else response_future
            )
            if not isinstance(response, dict) or response.get("peer_id") != peer_id:
                raise RuntimeError("link calibration returned the wrong peer identity")
            if response.get("received_bytes") != self.link_probe_bytes:
                raise RuntimeError("link calibration returned the wrong payload length")
        except Exception:
            logger.debug("Goodput calibration to %s failed", peer_id, exc_info=True)
            return False

        elapsed_ns = max(time.perf_counter_ns() - started_ns, 1)
        bytes_per_second = self.link_probe_bytes / (elapsed_ns / 1_000_000_000)
        self._record_link_goodput(peer_id, bytes_per_second)
        size_mb, elapsed_ms, speed_mb_s = _transfer_metrics(self.link_probe_bytes, elapsed_ns)
        logger.info(
            "Calibrated application goodput to %s: %.3f MB in %.3f ms (%.3f MB/s)",
            peer_id,
            size_mb,
            elapsed_ms,
            speed_mb_s,
        )
        return True

    def start_direct_peer_prober(self):
        """Refresh route topology without ever delaying liveness heartbeats.

        Direct-path qualification can wait several seconds per unavailable
        candidate. It therefore runs independently from ``node_update``: a
        worker doing useful inference must not be evicted just because topology
        telemetry is slow. The initial empty snapshot remains fail-closed.
        """

        def _prober_thread():
            while not self.stop_event.is_set():
                try:
                    self._probe_outbound_peers()
                    self._refresh_peer_rtts()
                except Exception:
                    logger.warning("Network telemetry probe loop failed", exc_info=True)
                self.stop_event.wait(_LINK_HEALTH_PROBE_INTERVAL_SECONDS)

        self.direct_peer_prober = threading.Thread(
            target=_prober_thread,
            name="DirectPeerProber",
            daemon=True,
        )
        self.direct_peer_prober.start()

    def _refresh_peer_rtts(self, peer_attempts: int = 1, rtt_attempts: int = 1) -> bool:
        """Refresh latency telemetry outside the liveness-critical path.

        The initial scheduler join may retry discovery because no heartbeat
        exists yet. Periodic topology refreshes use the one-shot defaults and
        must never hold up node liveness.
        """

        if time.time() - self.rtt_last_update <= self.rtt_update_interval:
            return True

        peers = None
        for attempt in range(max(1, peer_attempts)):
            peers = self.lattica.get_all_peers()
            if peers and self.scheduler_peer_id in peers:
                break
            if attempt + 1 < peer_attempts:
                time.sleep(1)
        if not peers or self.scheduler_peer_id not in peers:
            logger.warning("No peers found or scheduler peer id not found; keeping old RTTs")
            return False

        refreshed = {}
        for peer_id in peers:
            rtt = None
            for attempt in range(max(1, rtt_attempts)):
                try:
                    rtt = self.lattica.get_peer_rtt(peer_id) * 1000
                except Exception:
                    logger.warning("Failed to get RTT to %s", peer_id, exc_info=True)
                if rtt is not None or attempt + 1 >= rtt_attempts:
                    break
                time.sleep(1)
            refreshed[peer_id] = rtt if rtt is not None else 100
        self.rtts = refreshed
        self.rtt_last_update = time.time()
        return True

    def _v3_outgoing_link_metrics(self) -> tuple[LinkMetric, ...]:
        """Return qualified paths, enriched by exact goodput when available.

        The registered RPC health check is the reachability authority.  A
        bandwidth sample only improves route ranking and may expire
        independently, as in Petals' separation of online spans from pings.
        """

        now_ms = time.time_ns() // 1_000_000
        with self.link_throughputs_lock:
            samples = dict(self.link_throughputs)
        with self.link_topology_lock:
            reachable_peer_ids = tuple(self.reachable_peer_ids)
            direct_peer_ids = frozenset(self.direct_peer_ids)
            relayed_peer_ids = frozenset(self.relayed_peer_ids)
            path_observations = dict(self.link_path_observed_at_ms)
        metrics = []
        for peer_id in sorted(reachable_peer_ids):
            measured_at_ms = int(path_observations.get(peer_id, now_ms))
            if now_ms - measured_at_ms >= _LINK_REACHABILITY_TTL_MS:
                continue
            if peer_id in direct_peer_ids:
                path_kind = PathKind.DIRECT
            elif peer_id in relayed_peer_ids:
                path_kind = PathKind.RELAY
            else:
                continue
            rtt_ms = self.rtts.get(peer_id)
            if rtt_ms is None:
                continue
            sample = samples.get(peer_id)
            throughput = None
            throughput_measured_at_ms = None
            if sample is not None:
                sample_time = int(sample["measured_at_ms"])
                if now_ms - sample_time < _LINK_METRIC_TTL_MS:
                    throughput = float(sample["bytes_per_second"])
                    throughput_measured_at_ms = sample_time
            metrics.append(
                LinkMetric(
                    from_worker_id=self.lattica.peer_id(),
                    to_worker_id=peer_id,
                    path_kind=path_kind,
                    rtt_ms=max(0.0, float(rtt_ms)),
                    throughput_bytes_per_second=throughput,
                    throughput_measured_at_ms=throughput_measured_at_ms,
                    measured_at_ms=measured_at_ms,
                    expires_at_ms=measured_at_ms + _LINK_REACHABILITY_TTL_MS,
                )
            )
        return tuple(metrics)

    def start_routing_table_updater(self):
        def _updater_thread():
            while True and not self.stop_event.is_set():
                try:
                    graph = dijkstar.Graph()
                    servers = self.find_servers()
                    for server in servers:
                        start_index = server["block_start_index"]
                        end_index = server["block_end_index"]
                        peer_id = server["peer_id"]
                        graph.add_edge(start_index, end_index, (1, peer_id))
                    try:
                        path = dijkstar.find_path(
                            graph,
                            self.block_end_index,
                            self.hidden_layers,
                            cost_func=lambda u, v, e, prev_path: e[0],
                        )
                        routing_table = [self.lattica.peer_id()] + [edge[1] for edge in path.edges]
                        if self.routing_table != routing_table:
                            self.routing_table = routing_table
                            logger.info(f"Set routing table: {routing_table}")
                    except dijkstar.NoPathError:
                        self.routing_table = None
                        logger.warning(
                            f"No path found from 0 to {self.hidden_layers}, find servers {servers}"
                        )
                except Exception as e:
                    logger.exception(f"Error in routing table updater: {e}")

                time.sleep(self.routing_table_update_interval)

        if self.block_start_index == 0:
            self.routing_table_updater = threading.Thread(target=_updater_thread, daemon=True)
            self.routing_table_updater.start()

    def start_node_sender(self):
        send_to_peer = get_zmq_socket(zmq.Context(2), zmq.PULL, self.send_to_peer_addr, True)

        def group_requests_by_next_peer(requests: List[forward_pb2.Req]):
            grouped_requests = {}
            for req in requests:
                assert len(req.routing_table) > 0, "Request routing table is not set"
                try:
                    self_index = list(req.routing_table).index(self.lattica.peer_id())
                except ValueError as exc:
                    raise RuntimeError("Can not find self in the routing table") from exc

                next_peer_id = req.routing_table[(self_index + 1) % len(req.routing_table)]
                if next_peer_id not in grouped_requests:
                    grouped_requests[next_peer_id] = []
                grouped_requests[next_peer_id].append(req)
            if len(grouped_requests) > 1:
                logger.warning(
                    f"Grouped requests by next peer: {len(grouped_requests)}, {grouped_requests.keys()}"
                )
            return grouped_requests

        while True and not self.stop_event.is_set():
            try:
                if (
                    self.scheduler_addr is None
                    and self.block_start_index == 0
                    and self.routing_table is None
                ):
                    logger.info("Routing table is not ready in head rank, waiting for it to be set")
                    time.sleep(self.routing_table_update_interval)
                    continue

                message_type, message_body = send_to_peer.recv_multipart()[:2]

                if message_type == b"forward":
                    forward_request = forward_pb2.ForwardRequest()
                    forward_request.ParseFromString(message_body)
                    if len(forward_request.reqs) == 0:
                        raise RuntimeError("No requests in the forward request")

                    requests = []
                    for req in forward_request.reqs:
                        # set routing table if not scheduler mode
                        if len(req.routing_table) == 0 and self.scheduler_addr is None:
                            assert self.block_start_index == 0, (
                                "Request routing table is not set for non-head rank"
                            )

                            req.routing_table.extend(self.routing_table)
                            logger.info(
                                f"Set routing table {self.routing_table} for request {req.rid}"
                            )

                        if len(req.routing_table) > 0:
                            requests.append(req)
                        else:
                            logger.error(f"Request {req.rid} has no routing table, drop it")

                    grouped_requests = group_requests_by_next_peer(requests)

                    for next_peer_id, requests in grouped_requests.items():
                        stub = self.get_stub(next_peer_id)
                        start_ns = time.perf_counter_ns()
                        logger.info(f"Start forwarding data to {next_peer_id}")
                        new_forward_request = forward_pb2.ForwardRequest()
                        new_forward_request.forward_mode = forward_request.forward_mode
                        new_forward_request.reqs.extend(requests)
                        response = stub.rpc_pp_forward(new_forward_request)
                        response.result()
                        send_notify(
                            self.notify_url,
                            self.block_start_index,
                            self.block_end_index,
                            new_forward_request,
                            "completed",
                        )

                        size_mb, elapsed_ms, speed_mb_s = _transfer_metrics(
                            new_forward_request.ByteSize(), time.perf_counter_ns() - start_ns
                        )
                        self._record_link_goodput(
                            next_peer_id,
                            speed_mb_s * 1024 * 1024,
                            application_limited=True,
                        )
                        logger.info(
                            f"Forwarding data to {next_peer_id}, "
                            f"total size: {size_mb:.3f} MB, "
                            f"cost time: {elapsed_ms:.3f} ms, "
                            f"speed: {speed_mb_s:.3f} MB/s"
                        )

                elif message_type == b"abort":
                    abort_request = forward_pb2.AbortRequest()
                    abort_request.ParseFromString(message_body)
                    if len(abort_request.reqs) == 0:
                        raise RuntimeError("No requests in the abort request")

                    grouped_requests = {}
                    for req in abort_request.reqs:
                        # set routing table if not scheduler mode
                        if len(req.routing_table) == 0 and self.scheduler_addr is None:
                            assert self.block_start_index == 0, (
                                "Request routing table is not set for non-head rank"
                            )

                            req.routing_table.extend(self.routing_table)
                            logger.info(
                                f"Set routing table {self.routing_table} for request {req.rid}"
                            )

                        if len(req.routing_table) > 0:
                            # broadcast to all other nodes
                            for peer_id in req.routing_table:
                                if peer_id not in grouped_requests:
                                    grouped_requests[peer_id] = []
                                grouped_requests[peer_id].append(req)
                        else:
                            logger.error(f"Abort Request {req.rid} has no routing table, drop it")

                    for peer_id, requests in grouped_requests.items():
                        if peer_id != self.lattica.peer_id():
                            stub = self.get_stub(peer_id)
                            logger.info(
                                f"Send abort request: {[r.rid for r in requests]} to: {peer_id}"
                            )
                            new_abort_request = forward_pb2.AbortRequest()
                            new_abort_request.reqs.extend(requests)
                            stub.rpc_abort(new_abort_request)
                else:
                    logger.error(f"Unknown message type: {message_type}")

            except Exception as e:
                logger.exception(f"Error in handle_request: {e}")
                time.sleep(1)

    def start_node_announcer(self):
        """Start a thread that regularly announces this module's presence on DHT"""

        def _announcer_thread():
            try:
                while not self.stop_event.is_set():
                    # Announce the range ID
                    try:
                        if self.scheduler_peer_id is not None:
                            response_future = self.scheduler_stub.node_update(
                                self.get_node_info(is_update=True)
                            )
                            # Get the response result
                            response, refit_message = (
                                response_future.result(timeout=WORKER_HEARTBEAT_RPC_TIMEOUT_SECONDS)
                                if hasattr(response_future, "result")
                                else response_future
                            )

                            # Print layer allocation information
                            if response and isinstance(response, dict):
                                self._update_outbound_peers(response)
                                start_layer = response.get("start_layer")
                                end_layer = response.get("end_layer")
                                model_name = response.get("model_name")
                                model_revision = response.get("model_revision")
                                model_max_sequence_length = response.get(
                                    "model_max_sequence_length"
                                )
                                planned_context_tokens = response.get("planned_context_tokens")
                                allocation_epoch = response.get("allocation_epoch")
                                has_model_context = "model_max_sequence_length" in response
                                negotiated_chunk_size = response.get("chunked_prefill_size")
                                if self.swarm_v3_placement_controller is not None:
                                    # The qualified scheduler still supplies model and wire
                                    # contracts during migration, but layer ownership belongs to
                                    # the worker-side v3 transaction once it has started.
                                    start_layer = self.block_start_index
                                    end_layer = self.block_end_index
                                if start_layer is not None and end_layer is not None:
                                    logger.debug(
                                        f"Heartbeat: Node {self.lattica.peer_id()}... "
                                        f"Model: {model_name}, Layers: [{start_layer}, {end_layer})"
                                    )
                                    # Check if layer allocation changed
                                    allocation_changed = (
                                        start_layer != self.block_start_index
                                        or end_layer != self.block_end_index
                                        or model_name != self.model_name
                                        or model_revision != self.model_revision
                                    )
                                    prefill_contract_changed = (
                                        negotiated_chunk_size is not None
                                        and int(negotiated_chunk_size) != self.chunked_prefill_size
                                    )
                                    model_context_changed = (
                                        has_model_context
                                        and model_max_sequence_length
                                        != self.model_max_sequence_length
                                    )
                                    planned_context_changed = (
                                        planned_context_tokens is not None
                                        and planned_context_tokens != self.planned_context_tokens
                                    )
                                    allocation_epoch_changed = (
                                        allocation_epoch is not None
                                        and allocation_epoch != self.allocation_epoch
                                    )
                                    if (
                                        allocation_changed
                                        or prefill_contract_changed
                                        or model_context_changed
                                        or planned_context_changed
                                        or allocation_epoch_changed
                                    ):
                                        logger.warning(
                                            f"Worker serving contract changed! "
                                            f"Current: [{self.block_start_index}, {self.block_end_index}) -> "
                                            f"New: [{start_layer}, {end_layer}) "
                                            f"Model: {self.model_name} -> {model_name}; "
                                            f"chunked prefill: {self.chunked_prefill_size} -> "
                                            f"{negotiated_chunk_size}"
                                        )
                                        # Update layer allocation
                                        self.block_start_index = start_layer
                                        self.block_end_index = end_layer
                                        if self.connection_handler is not None:
                                            self.connection_handler.update_serving_span(
                                                start_layer,
                                                end_layer,
                                            )
                                        if model_name:
                                            self.model_name = model_name
                                        self.model_revision = model_revision
                                        if has_model_context:
                                            self.model_max_sequence_length = (
                                                None
                                                if model_max_sequence_length is None
                                                else int(model_max_sequence_length)
                                            )
                                        if planned_context_tokens is not None:
                                            self.planned_context_tokens = int(
                                                planned_context_tokens
                                            )
                                        if allocation_epoch is not None:
                                            self.allocation_epoch = int(allocation_epoch)
                                        if negotiated_chunk_size is not None:
                                            self.chunked_prefill_size = int(negotiated_chunk_size)
                                        # Set flag to trigger executor reload
                                        self._layer_allocation_changed = True
                                        # Set status to INITIALIZING to prevent scheduler from sending requests
                                        # during rebalancing
                                        self.status = ServerState.INITIALIZING

                                        # Sync to shared state if available
                                        self._sync_to_shared_state()

                                        logger.info(
                                            "Layer allocation updated. Executor will reload on next check. "
                                            "Status set to INITIALIZING to prevent new requests."
                                        )
                                else:
                                    logger.debug(
                                        f"Heartbeat: Missing layer info - start_layer={start_layer}, "
                                        f"end_layer={end_layer}, response={response}"
                                    )
                            else:
                                logger.warning(
                                    f"Heartbeat: No layer allocation received yet, response: {response}"
                                )
                                self.status = ServerState.JOINING
                                self.model_name = None
                                if self._shared_state is not None:
                                    self._shared_state.set_status(self.status.value)
                                    self._shared_state.update_metrics(current_requests=0)
                                    self._shared_state.set("model_name", None)
                                logger.debug(
                                    "Status set to JOINING and model_name to None because no valid layer allocation received yet."
                                )
                            if refit_message and isinstance(refit_message, dict):
                                if self.enable_weight_refit:
                                    logger.info("Server begin weight refit process.")
                                    if self.refit_finish:
                                        self.refit_finish = False
                                        t = threading.Thread(
                                            target=check_and_run_weight_refit,
                                            args=(self, refit_message),
                                            daemon=True,
                                        )
                                        t.start()
                                else:
                                    logger.warning(
                                        f"Received weight refit request but enable_weight_refit is set to {self.enable_weight_refit}."
                                    )
                        else:
                            self.lattica.store(
                                key=self.prefix_id,
                                subkey=self.lattica.peer_id(),
                                value={
                                    "block_start_index": self.block_start_index,
                                    "block_end_index": self.block_end_index,
                                },
                                expiration_time=time.time() + 60,  # Valid for 60 seconds
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to announce {self.prefix_id}_{self.lattica.peer_id()}: {e}",
                            exc_info=True,
                        )

                    self.stop_event.wait(WORKER_HEARTBEAT_INTERVAL_SECONDS)
            except Exception as e:
                logger.exception(f"Module announcer thread error: {e}")

        # Start announcer thread
        self.announcer = threading.Thread(target=_announcer_thread, daemon=True)
        self.announcer.start()
        logger.info(
            f"Started node announcer thread (daemon={self.announcer.daemon}, alive={self.announcer.is_alive()})"
        )

    def _get_status(self) -> str:
        """Get current status, checking shared_state if available (subprocess mode)"""
        # When running in subprocess mode, check shared_state status
        if hasattr(self, "_shared_state") and self._shared_state is not None:
            shared_status = self._shared_state.get_status()
            if shared_status is not None:
                if (
                    shared_status == ServerState.READY.value
                    and self._shared_state.get("frontend_required", False)
                    and not self._shared_state.get("frontend_alive", False)
                ):
                    return ServerState.INITIALIZING.value
                return shared_status
        # When running in same process, use local status
        return self.status.value

    def get_node_info(self, is_update: bool = False):
        # A dedicated topology thread owns network probes. Heartbeats only read
        # its last fail-closed result and therefore cannot be starved by probes.
        if not is_update and not self._refresh_peer_rtts(peer_attempts=10, rtt_attempts=30):
            return {}
        with self.link_topology_lock:
            direct_peer_ids = list(self.direct_peer_ids)
            reachable_peer_ids = list(self.reachable_peer_ids)
            relayed_peer_ids = list(self.relayed_peer_ids)
        hardware = self._stable_capacity_hardware()
        runtime_backend = "mlx" if hardware.get("device") == "mlx" else self.gpu_backend
        self.supports_chunked_prefill = runtime_backend in {"mlx", "sglang"}
        if not self.supports_chunked_prefill:
            # The official Parallax vLLM adapter does not implement chunk
            # progression. Advertising 0 here prevents an upstream MLX/SGLang
            # shard from sending partial activations to it.
            self.chunked_prefill_size = 0

        runtime_max_requests = self.max_batch_size
        runtime_kv_capacity = None
        runtime_kv_block_size = None
        memory_contract_failure = None
        if hasattr(self, "_shared_state") and self._shared_state is not None:
            measured_max_requests = self._shared_state.get("max_concurrent_requests")
            if measured_max_requests is not None:
                runtime_max_requests = measured_max_requests
            runtime_kv_capacity = self._shared_state.get("kv_cache_token_capacity")
            runtime_kv_block_size = self._shared_state.get("kv_cache_block_size")
            memory_contract_failure = self._shared_state.get("memory_contract_failure")
            placement_error = self._shared_state.get("swarm_v3_placement_error")
            if placement_error is not None and self.swarm_v3_placement_controller is not None:
                try:
                    self.swarm_v3_placement_controller.mark_failed(
                        generation=int(placement_error["generation"]),
                        error=RuntimeError(str(placement_error["detail"])),
                    )
                except RuntimeError:
                    logger.debug(
                        "Ignoring an already fenced placement failure: %s",
                        placement_error,
                    )

        info = {
            "node_id": self.lattica.peer_id(),
            "hardware": hardware,
            "kvcache_mem_ratio": self.kvcache_mem_ratio,
            "param_mem_ratio": self.param_mem_ratio,
            "max_concurrent_requests": runtime_max_requests,
            "max_sequence_length": (
                1024 if self.max_sequence_length is None else self.max_sequence_length
            ),
            "supports_frontend": self.supports_frontend,
            "supports_chunked_prefill": self.supports_chunked_prefill,
            "preferred_chunked_prefill_size": self.preferred_chunked_prefill_size,
            "chunked_prefill_size": self.chunked_prefill_size,
            "rtt_to_nodes": self.rtts,
            "status": self._get_status(),
            "is_active": self._get_status() == ServerState.READY.value,
            "manual_layer_assignment": self.manual_layer_assignment,
            "last_refit_time": self.last_refit_time,
        }
        if self.account_token:
            info["account_token"] = self.account_token
        if runtime_kv_capacity is not None and runtime_kv_block_size is not None:
            info["kv_cache_token_capacity"] = int(runtime_kv_capacity)
            info["kv_cache_block_size"] = int(runtime_kv_block_size)
        if memory_contract_failure is not None:
            info["memory_contract_failure"] = dict(memory_contract_failure)
        if direct_peer_ids is not None:
            info["direct_peer_ids"] = direct_peer_ids
        if reachable_peer_ids is not None:
            info["reachable_peer_ids"] = reachable_peer_ids
        if relayed_peer_ids is not None:
            info["relayed_peer_ids"] = relayed_peer_ids

        # For manual layer assignment, always include start_layer and end_layer
        if self.manual_layer_assignment:
            info["start_layer"] = self.block_start_index
            info["end_layer"] = self.block_end_index
            logger.info(
                f"Manual assignment: sending start_layer={self.block_start_index}, "
                f"end_layer={self.block_end_index}"
            )

        if is_update:
            metrics = {}
            if hasattr(self, "_shared_state") and self._shared_state is not None:
                metrics = self._shared_state.get_metrics()

            info["current_requests"] = metrics.get("current_requests", 0)
            if metrics.get("layer_latency_ms") is not None:
                info["layer_latency_ms"] = metrics.get("layer_latency_ms")
            if hasattr(self, "_shared_state") and self._shared_state is not None:
                info["memory_pressure"] = self._shared_state.get("memory_pressure", "normal")
                info["memory_pressure_resources"] = self._shared_state.get(
                    "memory_pressure_resources", {}
                )
            # In update mode, always include current allocation
            if not self.manual_layer_assignment:
                info["start_layer"] = self.block_start_index
                info["end_layer"] = self.block_end_index

            if self.swarm_v3_init_error is not None:
                info["swarm_v3"] = {
                    "mode": "active" if self.swarm_v3_execution_admission is not None else "shadow",
                    "state": "rejected",
                    "error": self.swarm_v3_init_error,
                }
            elif self.swarm_v3_reporter is not None:
                required_contract = (
                    self.model_name,
                    self.model_revision,
                    self.block_start_index,
                    self.block_end_index,
                    runtime_kv_capacity,
                    runtime_kv_block_size,
                    runtime_max_requests,
                    hardware.get("usable_memory_bytes"),
                    (
                        self.planned_context_tokens
                        if self.swarm_v3_placement_mode == "autonomous"
                        else 1
                    ),
                )
                if all(value is not None for value in required_contract):
                    backend = (
                        BackendKind.MLX
                        if runtime_backend == "mlx"
                        else (BackendKind.VLLM if runtime_backend == "vllm" else BackendKind.SGLANG)
                    )
                    report = self.swarm_v3_reporter.snapshot(
                        WorkerServingSnapshot(
                            worker_id=self.lattica.peer_id(),
                            endpoint_id=self.lattica.peer_id(),
                            model_id=str(self.model_name),
                            immutable_revision=str(self.model_revision),
                            span=LayerSpan(
                                start=int(self.block_start_index),
                                end=int(self.block_end_index),
                            ),
                            backend=backend,
                            stable_memory_envelope_bytes=int(hardware["usable_memory_bytes"]),
                            kv_cache_token_capacity=int(runtime_kv_capacity),
                            kv_cache_block_size=int(runtime_kv_block_size),
                            max_sessions=int(runtime_max_requests),
                            is_ready=self._get_status() == ServerState.READY.value,
                            current_requests=int(metrics.get("current_requests", 0)),
                            supports_frontend=self.supports_frontend,
                            outgoing_links=self._v3_outgoing_link_metrics(),
                            measured_prefill_tokens_per_second=metrics.get(
                                "prefill_tokens_per_second"
                            ),
                            measured_decode_tokens_per_second=metrics.get(
                                "decode_tokens_per_second"
                            ),
                        )
                    )
                    if self.swarm_v3_execution_admission is not None and report.get("state") in {
                        "ready",
                        "warming",
                    }:
                        try:
                            advertisement = ModelMemberAdvertisement.model_validate(
                                report["advertisement"]
                            )
                            if self.swarm_v3_placement_mode == "autonomous":
                                manifest = self.swarm_v3_reporter.trusted_manifest(
                                    advertisement.lease.model_swarm_id
                                )
                                catalog = self.iroh_transport.catalog_discovery
                                if manifest is None or catalog is None:
                                    raise RuntimeError(
                                        "autonomous placement trust or catalogue is not ready"
                                    )
                                if self.swarm_v3_placement_controller is None:
                                    self.swarm_v3_placement_controller = AutonomousWorkerPlacement(
                                        catalog=catalog,
                                        admission=self.swarm_v3_execution_admission,
                                        state_publisher=self.swarm_v3_reporter,
                                        reload_target=self._apply_v3_span_reload,
                                        current_span=advertisement.lease.hosted_span,
                                    )
                                placement = self.swarm_v3_placement_controller.observe(
                                    advertisement=advertisement,
                                    manifest=manifest,
                                    context_tokens=int(self.planned_context_tokens),
                                )
                                report["placement"] = placement
                                if self._shared_state is not None:
                                    placement_state = {
                                        "swarm_v3_placement_generation": placement["generation"],
                                        "swarm_v3_placement_phase": placement["phase"],
                                    }
                                    if placement["phase"] == "ready":
                                        placement_state.update(
                                            swarm_v3_placement_error=None,
                                            swarm_v3_previous_start_layer=None,
                                            swarm_v3_previous_end_layer=None,
                                        )
                                    self._shared_state.update(**placement_state)
                            else:
                                self.swarm_v3_execution_admission.configure(advertisement)
                        except Exception as exc:
                            report = {
                                "mode": "active",
                                "state": "rejected",
                                "error": {
                                    "code": type(exc).__name__,
                                    "detail": str(exc)[:256],
                                },
                            }
                    info["swarm_v3"] = report
                else:
                    info["swarm_v3"] = {
                        "mode": self.swarm_v3_reporter.mode,
                        "state": "waiting_contract",
                    }

        return info

    def _stable_capacity_hardware(self) -> dict[str, Any]:
        """Return the immutable capacity envelope for this worker generation."""

        with self._capacity_hardware_lock:
            if self._capacity_hardware_snapshot is None:
                detected = detect_node_hardware(self.lattica.peer_id())
                self._capacity_hardware_snapshot = copy.deepcopy(detected)
            return copy.deepcopy(self._capacity_hardware_snapshot)

    def shutdown(self):
        self.stop_event.set()

        self.status = ServerState.OFFLINE
        try:
            self._sync_to_shared_state()
        except Exception:
            # The multiprocessing manager can disappear before this child. A
            # stale manager must never prevent the scheduler leave RPC.
            logger.debug("Failed to sync final P2P state", exc_info=True)
        # Shutdown uses the server's local state from here on. Reading a manager
        # proxy again can fail after the launch process has started tearing down.
        self._shared_state = None

        try:
            if self.scheduler_addr is not None and self.scheduler_stub is not None:
                peer_id = self.lattica.peer_id() if self.lattica is not None else "unknown"
                logger.info(f"Leave scheduler: {peer_id}")
                response = self.scheduler_stub.node_leave(self.get_node_info(is_update=True))
                if hasattr(response, "result"):
                    response.result(timeout=5)
        except Exception:
            logger.warning("Failed to notify scheduler that the worker is leaving", exc_info=True)

        try:
            if self.announcer is not None:
                self.announcer.join(timeout=1)
            if self.direct_peer_prober is not None:
                self.direct_peer_prober.join(timeout=1)
            if self.routing_table_updater is not None:
                self.routing_table_updater.join(timeout=1)
        except Exception:
            logger.debug("Failed to join P2P background threads", exc_info=True)
        finally:
            if self.lattica is not None:
                try:
                    self.lattica.close()
                except Exception:
                    logger.debug("Failed to close Lattica", exc_info=True)


def _run_p2p_server_process(
    initial_peers: List[str],
    scheduler_addr: Optional[str],
    relay_servers: List[str],
    pp_start_layer: int,
    pp_end_layer: int,
    hidden_layers: int,
    tp_size: int,
    dp_size: int,
    tcp_port: int,
    udp_port: int,
    dht_prefix: str,
    announce_maddrs: List[str],
    http_port: Optional[int],
    notify_url: str,
    recv_from_peer_addr: str,
    send_to_peer_addr: str,
    model_name: Optional[str],
    max_batch_size: Optional[int] = None,
    max_sequence_length: Optional[int] = None,
    param_mem_ratio: float = 0.65,
    kvcache_mem_ratio: float = 0.25,
    gpu_backend: str = "sglang",
    chunked_prefill_size: Optional[int] = None,
    shared_state: Optional[dict] = None,
    log_level: str = "INFO",
    conn: Any = None,
):
    """Run P2P server in subprocess"""
    # Set log level in subprocess (spawn mode doesn't inherit log configuration)
    set_log_level(log_level)
    server = None
    # SIGTERM (arret pilote par l IDE / .terminate() / docker stop) doit
    # declencher le meme arret gracieux que Ctrl-C : sinon le finally
    # server.shutdown() (qui envoie node_leave au scheduler) est saute ->
    # noeud fantome cote scheduler.
    import signal as _signal

    def _sigterm_to_kbi(signum, frame):
        raise KeyboardInterrupt()

    try:
        _signal.signal(_signal.SIGTERM, _sigterm_to_kbi)
    except Exception:
        pass
    try:
        server = GradientServer(
            recv_from_peer_addr=recv_from_peer_addr,
            send_to_peer_addr=send_to_peer_addr,
            initial_peers=initial_peers,
            scheduler_addr=scheduler_addr,
            relay_servers=relay_servers,
            block_start_index=pp_start_layer,
            block_end_index=pp_end_layer,
            hidden_layers=hidden_layers,
            tp_size=tp_size,
            dp_size=dp_size,
            dht_prefix=dht_prefix,
            host_maddrs=[
                f"/ip4/0.0.0.0/tcp/{tcp_port}",
                f"/ip4/0.0.0.0/udp/{udp_port}/quic-v1",
            ],
            announce_maddrs=announce_maddrs,
            http_port=http_port,
            notify_url=notify_url,
            model_name=model_name,
            max_batch_size=max_batch_size,
            max_sequence_length=max_sequence_length,
            param_mem_ratio=param_mem_ratio,
            kvcache_mem_ratio=kvcache_mem_ratio,
            gpu_backend=gpu_backend,
            chunked_prefill_size=chunked_prefill_size,
            conn=conn,
        )
        # Attach shared state to server for syncing layer allocation
        if shared_state is not None:
            shared_state = SharedState(shared_state)  # Auto-converts dict to SharedState
            server._shared_state = shared_state
            # Initialize shared state with current values
            shared_state.update(
                block_start_index=server.block_start_index,
                block_end_index=server.block_end_index,
                model_name=server.model_name,
                tp_size=server.tp_size,
                enable_weight_refit=False,
                weight_refit_mode="disk",
                chunked_prefill_size=server.chunked_prefill_size,
                status=server.status.value,
            )

        server.run()
    except KeyboardInterrupt:
        logger.debug("P2P server received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception(f"P2P server error: {e}")
    finally:
        if server is not None:
            server.shutdown()


def launch_p2p_server_process(
    initial_peers: List[str],
    scheduler_addr: Optional[str],
    relay_servers: List[str],
    pp_start_layer: int,
    pp_end_layer: int,
    hidden_layers: int,
    tp_size: int,
    dp_size: int,
    tcp_port: int,
    udp_port: int,
    dht_prefix: str,
    announce_maddrs: List[str],
    http_port: Optional[int],
    notify_url: str,
    recv_from_peer_addr: str,
    send_to_peer_addr: str,
    model_name: Optional[str],
    max_batch_size: Optional[int] = None,
    max_sequence_length: Optional[int] = None,
    param_mem_ratio: float = 0.65,
    kvcache_mem_ratio: float = 0.25,
    gpu_backend: str = "sglang",
    chunked_prefill_size: Optional[int] = None,
    shared_state: Optional[dict] = None,
    log_level: str = "INFO",
    conn: Optional[Any] = None,
) -> multiprocessing.Process:
    """Launch P2P server as a subprocess and return the process object

    Args:
        shared_state: Optional shared dictionary for inter-process communication.
                     If provided, layer allocation info will be synced to this dict.
        log_level: Log level for the subprocess (default: INFO).
    """
    process = multiprocessing.Process(
        target=_run_p2p_server_process,
        args=(
            initial_peers,
            scheduler_addr,
            relay_servers,
            pp_start_layer,
            pp_end_layer,
            hidden_layers,
            tp_size,
            dp_size,
            tcp_port,
            udp_port,
            dht_prefix,
            announce_maddrs,
            http_port,
            notify_url,
            recv_from_peer_addr,
            send_to_peer_addr,
            model_name,
            max_batch_size,
            max_sequence_length,
            param_mem_ratio,
            kvcache_mem_ratio,
            gpu_backend,
            chunked_prefill_size,
            shared_state,
            log_level,
            conn,
        ),
    )
    process.start()
    return process


def stop_p2p_server(p2p_server_process: Optional[multiprocessing.Process]):
    """Stop P2P server subprocess"""
    if p2p_server_process is not None and p2p_server_process.is_alive():
        logger.debug("Terminating P2P server subprocess...")
        try:
            p2p_server_process.terminate()
            p2p_server_process.join(timeout=5)
            if p2p_server_process.is_alive():
                logger.warning("P2P server process did not terminate gracefully, killing...")
                p2p_server_process.kill()
                p2p_server_process.join()
        except Exception as e:
            logger.error(f"Failed to terminate P2P server subprocess: {e}")
