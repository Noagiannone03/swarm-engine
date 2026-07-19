import threading
import time
from typing import List, Literal

from lattica import Lattica

from backend.server.constants import NODE_STATUS_AVAILABLE, NODE_STATUS_WAITING
from backend.server.context_admission import ContextBudget, build_context_budget
from backend.server.rpc_connection_handler import RPCConnectionHandler
from backend.server.static_config import get_model_info, get_node_join_command
from parallax.cli import PUBLIC_INITIAL_PEERS, PUBLIC_RELAY_SERVERS
from parallax.p2p.server import TransformerConnectionHandler
from parallax_utils.logging_config import get_logger
from scheduling.node import RequestSignal
from scheduling.scheduler import Scheduler

logger = get_logger(__name__)


class SchedulerManage:
    """
    Coordinates the in-process scheduler and the P2P RPC layer.

    This manager owns the `Scheduler` instance and the Lattica P2P node,
    wiring RPC calls from workers to scheduler events.
    """

    def __init__(
        self,
        initial_peers: List[str] = [],
        relay_servers: List[str] = [],
        dht_prefix: str = "gradient",
        host_maddrs: List[str] = [],
        announce_maddrs: List[str] = [],
        http_port: int = 3001,
        use_hfcache: bool = False,
        enable_weight_refit: bool = False,
        weight_refit_mode: str = "disk",
        allocation_strategy: Literal["greedy", "dp"] = "dp",
        routing_strategy: Literal["rr", "dp"] = "dp",
    ):
        """Initialize the manager with networking bootstrap parameters."""
        self.initial_peers = initial_peers
        self.relay_servers = relay_servers
        self.dht_prefix = dht_prefix
        self.host_maddrs = host_maddrs
        self.announce_maddrs = announce_maddrs
        self.http_port = http_port
        self.use_hfcache = use_hfcache
        self.enable_weight_refit = enable_weight_refit
        self.weight_refit_mode = weight_refit_mode
        self.allocation_strategy = allocation_strategy
        self.routing_strategy = routing_strategy
        self.model_name = None
        self.init_nodes_num = None
        self.scheduler = None
        self.node_id = f"{dht_prefix}_announce"
        self.lattica = None
        self.stubs = {}
        self.is_local_network = False
        self._context_tokenizer = None
        self._context_tokenizer_model = None
        self._context_tokenizer_lock = threading.Lock()

    def run(self, model_name, init_nodes_num, is_local_network=True):
        """
        Start the scheduler and the P2P service for RPC handling.
        If Lattica is already running, it will be reused.
        Nodes will automatically rejoin via their heartbeat (node_update) mechanism.
        """
        logger.debug(
            "SchedulerManage starting: model_name=%s, init_nodes_num=%s, "
            "allocation_strategy=%s, routing_strategy=%s",
            model_name,
            init_nodes_num,
            self.allocation_strategy,
            self.routing_strategy,
        )
        self.is_local_network = is_local_network
        if not is_local_network and not self.initial_peers and not self.relay_servers:
            logger.debug("Using public relay servers")
            self.initial_peers = PUBLIC_INITIAL_PEERS
            self.relay_servers = PUBLIC_RELAY_SERVERS

        self._start_scheduler(model_name, init_nodes_num)
        self._start_lattica()
        self.completion_handler = TransformerConnectionHandler(
            lattica=self.lattica,
            recv_from_peer_addr="",
            send_to_peer_addr="",
            block_start_index=0,
            block_end_index=1,
        )

    def is_running(self):
        """
        Returns True if the scheduler is running, False otherwise.
        """
        return self.scheduler is not None

    def stop(self):
        """
        Stop the scheduler only. Lattica will remain running.
        """
        logger.info("Stopping scheduler...")

        # Stop scheduler if running
        if self.scheduler is not None:
            logger.debug("Stopping scheduler...")
            self.scheduler._stop_event.set()
            # Wait a bit for threads to finish
            time.sleep(0.1)
            self.scheduler = None
            logger.debug("Scheduler stopped")

        # Note: We don't close Lattica here to allow model switching without restarting P2P

        logger.info("Scheduler stopped")

    def get_model_name(self):
        return self.model_name

    def get_init_nodes_num(self):
        return self.init_nodes_num

    def get_is_local_network(self):
        return self.is_local_network

    def get_peer_id(self):
        if self.lattica is None:
            return None
        return self.lattica.peer_id()

    def weight_refit(self, request_data):
        """
        Trigger weight refit on every nodes.
        """
        if self.scheduler is None:
            return False
        self.scheduler.refit_request = request_data
        self.scheduler.refit_set = set()
        return True

    def get_last_refit_time(self):
        return self.scheduler.update_last_refit_time()

    def need_more_nodes(self):
        return self.scheduler.need_more_nodes() if self.scheduler else False

    def get_cluster_status(self):
        return {
            "type": "cluster_status",
            "data": {
                "status": self.get_schedule_status(),
                "model_name": self.model_name,
                "init_nodes_num": self.init_nodes_num,
                "allocation_strategy": self.allocation_strategy,
                "routing_strategy": self.routing_strategy,
                "chunked_prefill_size": (
                    self.scheduler.negotiated_chunked_prefill_size() if self.scheduler else 0
                ),
                "prefill_contract_ready": (
                    self.scheduler.prefill_contract_ready() if self.scheduler else False
                ),
                "max_supported_context_tokens": self.max_supported_context_tokens(),
                "node_join_command": get_node_join_command(
                    self.get_peer_id(), self.is_local_network
                ),
                "node_list": self.get_node_list(),
                "need_more_nodes": self.need_more_nodes(),
                "max_running_request": (
                    self.scheduler.report_pipeline_capacity()[1] if self.scheduler else 0
                ),
            },
        }

    def get_node_list(self):
        if self.scheduler is None:
            return []

        return [self.build_node_info(node) for node in self.scheduler.node_manager.nodes]

    def build_node_info(self, node):
        return {
            "node_id": node.node_id,
            "status": NODE_STATUS_AVAILABLE if node.is_active else NODE_STATUS_WAITING,
            "gpu_num": node.hardware.num_gpus,
            "gpu_name": node.hardware.gpu_name,
            "gpu_memory": node.hardware.memory_gb,
            "supports_frontend": node.supports_frontend,
            "max_sequence_length": getattr(node, "max_sequence_length", None),
            "max_concurrent_requests": getattr(node, "max_requests", None),
            "kv_cache_telemetry_ready": (
                getattr(node, "kv_cache_token_capacity", None) is not None
            ),
            "kv_cache_token_capacity": getattr(node, "kv_cache_token_capacity", None),
            "kv_cache_block_size": getattr(node, "kv_cache_block_size", None),
            "reserved_context_tokens": getattr(node, "reserved_context_tokens", 0),
            "remaining_context_tokens": getattr(node, "remaining_context_tokens", None),
            "direct_link_telemetry_ready": getattr(node, "direct_peer_ids", None) is not None,
            "direct_peer_ids": (
                sorted(node.direct_peer_ids)
                if getattr(node, "direct_peer_ids", None) is not None
                else None
            ),
        }

    def _start_scheduler(self, model_name, init_nodes_num):
        """
        Create the scheduler and start its background run loop.
        If scheduler already exists, it will be stopped and recreated.
        Nodes will automatically rejoin via their heartbeat (node_update) mechanism.
        """
        # Stop existing scheduler if running
        if self.scheduler is not None:
            logger.info("Scheduler already running, stopping it first for re-initialization")
            self.stop()

        self.model_name = model_name
        self.init_nodes_num = init_nodes_num
        with self._context_tokenizer_lock:
            self._context_tokenizer = None
            self._context_tokenizer_model = None

        model_info = get_model_info(model_name, self.use_hfcache)
        self.scheduler = Scheduler(
            model_info,
            [],
            min_nodes_bootstrapping=init_nodes_num,
            enable_weight_refit=self.enable_weight_refit,
            weight_refit_mode=self.weight_refit_mode,
            strategy=self.allocation_strategy,
            routing_strategy=self.routing_strategy,
        )

        # Run the scheduler's event/dispatch loops in background so the process
        # can continue to serve RPCs and HTTP traffic.
        threading.Thread(
            target=self.scheduler.run,
            kwargs={"poll_interval": 0.05},
            name="SchedulerMain",
            daemon=True,
        ).start()
        logger.debug("Scheduler background thread started (poll_interval=0.05)")
        logger.info("Nodes will automatically rejoin via heartbeat (node_update) mechanism")

    def _start_lattica(self):
        """
        Initialize and start the Lattica P2P node used for RPCs.
        If Lattica already exists, it will be reused (no restart), but connection_handler will be updated.
        """
        # Reuse existing Lattica if running
        if self.lattica is not None:
            logger.debug("Lattica already running, reusing existing instance")
            # Update connection handler with new scheduler if it exists
            if hasattr(self, "connection_handler") and self.connection_handler is not None:
                self.connection_handler.scheduler = self.scheduler
                logger.debug("Updated connection handler with new scheduler")
            else:
                # Create connection handler if it doesn't exist
                self.connection_handler = RPCConnectionHandler(
                    lattica=self.lattica,
                    scheduler=self.scheduler,
                    http_port=self.http_port,
                )
                logger.debug("Created connection handler with existing Lattica")
            return

        logger.debug(
            f"Starting Lattica with host_maddrs={self.host_maddrs}, mdns=False, dht_prefix={self.dht_prefix}"
        )
        self.lattica = Lattica.builder().with_listen_addrs(self.host_maddrs).with_key_path(".")

        if len(self.relay_servers) > 0:
            logger.info(f"Using relay servers: {self.relay_servers}")
            self.lattica.with_relay_servers(self.relay_servers).with_dcutr(True).with_protocol("")

        if len(self.announce_maddrs) > 0:
            logger.info(f"Using announce maddrs: {self.announce_maddrs}")
            self.lattica.with_external_addrs(self.announce_maddrs)

        if len(self.initial_peers) > 0:
            logger.info(f"Using initial peers: {self.initial_peers}")
            self.lattica.with_bootstraps(self.initial_peers)

        self.lattica.build()
        logger.debug("Lattica node built")

        if len(self.relay_servers) > 0:
            try:
                is_symmetric_nat = self.lattica.is_symmetric_nat()
                if is_symmetric_nat is None:
                    logger.warning("Failed to get is symmetric NAT, skip")
                elif is_symmetric_nat:
                    logger.error(
                        "Your network NAT type is symmetric, relay does not work on this type of NAT, see https://en.wikipedia.org/wiki/Network_address_translation"
                    )
                    exit(1)
            except Exception as e:
                logger.exception(f"Error in is symmetric NAT: {e}")

        store_success = False
        for _ in range(10):
            try:
                if self.lattica.store(
                    "scheduler_peer_id",
                    self.lattica.peer_id(),
                    expiration_time=time.time() + 365 * 24 * 60 * 60,
                ):
                    logger.info(f"Stored scheduler peer id: {self.lattica.peer_id()}")
                    store_success = True
                    break
                logger.warning("Failed to store scheduler peer id, waiting for 10 seconds")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Failed to store scheduler peer id: {e}, waiting for 10 seconds")
                time.sleep(10)

        if not store_success:
            logger.error("Failed to store scheduler peer id, after 10 times")
            exit(1)

        self.connection_handler = RPCConnectionHandler(
            lattica=self.lattica,
            scheduler=self.scheduler,
            http_port=self.http_port,
        )
        logger.debug("RPCConnectionHandler initialized")

    def _get_context_tokenizer(self):
        """Lazily load the canonical tokenizer used by the scheduler's model."""
        model_name = self.model_name
        if model_name is None:
            raise RuntimeError("scheduler model is not configured")
        with self._context_tokenizer_lock:
            if self._context_tokenizer is None or self._context_tokenizer_model != model_name:
                from transformers import AutoTokenizer

                self._context_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=self.use_hfcache,
                )
                self._context_tokenizer_model = model_name
            return self._context_tokenizer

    def build_context_budget(self, request_data) -> ContextBudget:
        """Tokenize the fully rendered chat and reserve its requested output."""
        return build_context_budget(self._get_context_tokenizer(), request_data)

    def max_supported_context_tokens(self) -> int:
        if self.scheduler is None:
            return 0
        return self.scheduler.max_supported_context_tokens()

    def get_routing_table(self, request_id, received_ts, required_context_tokens: int = 0):
        """Block briefly until the scheduler assigns a routing path for the request.

        Distinguish three states via `RequestSignal.routing_table`:
        - None: not yet decided, keep waiting up to timeout
        - []: decided but no capacity (pipelines full), return immediately
        - [..]: valid routing path, return immediately
        """
        logger.debug(f"Routing table requested for request_id={request_id}")
        request = RequestSignal(
            request_id,
            received_ts,
            required_context_tokens=required_context_tokens,
        )
        self.scheduler.receive_request(request)

        # Wait up to 5 seconds, but return immediately if the routing table is set (including an empty list)
        start_time = time.time()
        while request.routing_table is None and (time.time() - start_time) < 5.0:
            time.sleep(0.05)

        # Return the routing_table
        if request.routing_table is None:
            # The queue entry can outlive this HTTP waiter when the cluster is
            # temporarily incomplete. Cancel it atomically with dispatch so it
            # cannot reserve a recovered pipeline after the caller receives 503.
            self.scheduler.cancel_request_signal(request)
            logger.debug(
                f"Routing table not ready after {(time.time() - start_time):.2f}s for request_id={request_id}"
            )
        else:
            logger.debug(
                f"Routing table resolved for request_id={request_id}: {request.routing_table}"
            )
        return request.routing_table

    def release_routing_table(self, request_id: str) -> bool:
        """Release scheduler capacity as soon as the forwarded HTTP request ends."""
        if self.scheduler is None:
            return False
        return self.scheduler.release_request(str(request_id))

    def is_routing_table_active(self, request_id: str) -> bool:
        """Return whether a dispatched request still owns a live worker route."""
        if self.scheduler is None:
            return False
        return self.scheduler.is_request_route_active(str(request_id))

    def wait_for_routing_capacity(self, timeout: float) -> bool:
        """Block until a route can be admitted or the bounded wait expires."""
        if self.scheduler is None:
            return False
        return self.scheduler.wait_for_routing_capacity(timeout)

    def get_schedule_status(self):
        """
        Return whether a full pipeline has been allocated across joined nodes.
        """
        if self.scheduler is None:
            logger.debug("SchedulerManage status queried: waiting (scheduler not initialized)")
            return NODE_STATUS_WAITING

        # todo rebalance status
        status = NODE_STATUS_AVAILABLE if self.scheduler.serving_ready() else NODE_STATUS_WAITING
        logger.debug(f"SchedulerManage status queried: {status}")
        return status

    def get_call_url_by_node_id(self, node_id):
        """
        Lookup the HTTP endpoint for a given node id managed by the RPC layer.
        """
        url = self.connection_handler.get_call_url_by_node_id(node_id)
        logger.debug(f"Lookup call_url for node_id={node_id} -> {url}")
        return url
