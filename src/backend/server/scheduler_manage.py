import os
import threading
import time
from typing import List

from lattica import Lattica

from backend.server.constants import NODE_STATUS_AVAILABLE, NODE_STATUS_WAITING
from backend.server.rpc_connection_handler import RPCConnectionHandler
from backend.server.static_config import get_model_info, get_node_join_command
from parallax.cli import get_public_initial_peers, get_public_relay_servers
from parallax.p2p.server import TransformerConnectionHandler
from parallax_utils.logging_config import get_logger
from scheduling.node import RequestSignal
from scheduling.scheduler import Scheduler

logger = get_logger(__name__)

# How often we re-`lattica.store("scheduler_peer_id", ...)` to keep the entry
# alive in the DHT. Kademlia keys live "until expiration" *only as long as
# some peer still caches them* — when DHT peers churn (which happens within
# hours on a small swarm) the key vanishes, and new workers calling
# `lattica.get("scheduler_peer_id")` get nothing back. Upstream stores the
# key once at boot with a 1-year expiration and never refreshes, so after
# ~1–2h of uptime the scheduler becomes invisible to fresh workers even
# though the process is still running. Petals/Hivemind solve this by
# re-announcing every 30–60s; we do it every 5min, which is conservative
# enough to never spam the DHT yet keeps the entry warm well below the
# typical Kademlia replication window.
SCHEDULER_PEER_ID_REANNOUNCE_SEC = 5 * 60


def _scheduler_runtime_overrides() -> dict:
    """Read Scheduler() kwargs that should be tunable in deployment.

    These are exposed as environment variables so an operator running the
    scheduler in a container or systemd unit can change behavior without
    forking the source. Bad/empty values fall back to the Scheduler default.

    Recognized vars:
        PARALLAX_STRATEGY                "greedy" | "dp"
        PARALLAX_HEARTBEAT_TIMEOUT       float seconds, e.g. 20
    """
    overrides: dict = {}

    strategy = os.environ.get("PARALLAX_STRATEGY", "").strip().lower()
    if strategy in ("greedy", "dp"):
        overrides["strategy"] = strategy
    elif strategy:
        logger.warning(
            "Ignoring PARALLAX_STRATEGY=%r (expected 'greedy' or 'dp')", strategy
        )

    raw_timeout = os.environ.get("PARALLAX_HEARTBEAT_TIMEOUT", "").strip()
    if raw_timeout:
        try:
            timeout = float(raw_timeout)
            if timeout <= 0:
                raise ValueError("must be > 0")
            overrides["heartbeat_timeout"] = timeout
        except ValueError as exc:
            logger.warning(
                "Ignoring PARALLAX_HEARTBEAT_TIMEOUT=%r (%s)", raw_timeout, exc
            )

    return overrides


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
        self.model_name = None
        self.init_nodes_num = None
        self.scheduler = None
        self.node_id = f"{dht_prefix}_announce"
        self.lattica = None
        self.stubs = {}
        self.is_local_network = False

    def run(self, model_name, init_nodes_num, is_local_network=True):
        """
        Start the scheduler and the P2P service for RPC handling.
        If Lattica is already running, it will be reused.
        Nodes will automatically rejoin via their heartbeat (node_update) mechanism.
        """
        logger.debug(
            f"SchedulerManage starting: model_name={model_name}, init_nodes_num={init_nodes_num}"
        )
        is_local_network = bool(is_local_network)
        if self.initial_peers or self.relay_servers:
            # If the operator has explicitly configured initial peers or
            # relay servers, the swarm is by definition not a local-only
            # cluster. Tolerate is_local_network=True being passed by an
            # older API caller and force the consistent state.
            is_local_network = False
        self.is_local_network = is_local_network
        if not is_local_network and not self.initial_peers and not self.relay_servers:
            logger.debug("Using public relay servers")
            self.initial_peers = get_public_initial_peers()
            self.relay_servers = get_public_relay_servers()

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
        # Bootstrap result/timestamp are exposed verbatim so clients can detect
        # "failed_capacity" (allocation tried, can't fit) vs "pending" (still
        # running) without polling intervals. Falls back to None when the
        # scheduler hasn't been initialized yet (model not set).
        last_bootstrap_result = (
            self.scheduler.last_bootstrap_result if self.scheduler else None
        )
        last_bootstrap_attempt_ts = (
            self.scheduler.last_bootstrap_attempt_ts if self.scheduler else 0.0
        )
        return {
            "type": "cluster_status",
            "data": {
                "status": self.get_schedule_status(),
                "model_name": self.model_name,
                "init_nodes_num": self.init_nodes_num,
                "node_join_command": get_node_join_command(
                    self.get_peer_id(), self.is_local_network
                ),
                "node_list": self.get_node_list(),
                "need_more_nodes": self.need_more_nodes(),
                "last_bootstrap_result": last_bootstrap_result,
                "last_bootstrap_attempt_ts": last_bootstrap_attempt_ts,
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
        # Per-node state for richer UI feedback. The scheduler's `state_of()`
        # tells us if the node is in an active pipeline (ACTIVE) or held in
        # reserve (STANDBY); `loading_phase` is the worker-reported lifecycle
        # ("joining" / "initializing" / "ready" / ...). Together they let the
        # CLI distinguish "downloading model" from "standby for redundancy"
        # from "ready to serve" without parsing logs.
        node_state = None
        if self.scheduler is not None:
            state_obj = self.scheduler.node_manager.state_of(node.node_id)
            node_state = state_obj.value if state_obj is not None else None
        return {
            "node_id": node.node_id,
            "status": NODE_STATUS_AVAILABLE if node.is_active else NODE_STATUS_WAITING,
            "node_state": node_state,
            "loading_phase": node.loading_phase,
            "start_layer": node.start_layer,
            "end_layer": node.end_layer,
            "gpu_num": node.hardware.num_gpus,
            "gpu_name": node.hardware.gpu_name,
            "gpu_memory": node.hardware.memory_gb,
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

        model_info = get_model_info(model_name, self.use_hfcache)
        scheduler_kwargs = _scheduler_runtime_overrides()
        self.scheduler = Scheduler(
            model_info,
            [],
            min_nodes_bootstrapping=init_nodes_num,
            enable_weight_refit=self.enable_weight_refit,
            weight_refit_mode=self.weight_refit_mode,
            **scheduler_kwargs,
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

        # The historical debug log claimed `mdns=False` but no `with_mdns`
        # call was made; Lattica defaults to mDNS=True. For a public swarm
        # (relays + bootstraps gradient.network) mDNS hurts: it lets a
        # peer on the operator's LAN advertise itself ahead of the public
        # bootstrap and locks the swarm in a private mini-DHT. Turn it off
        # explicitly; opt back in via `PARALLAX_ENABLE_MDNS=1` for
        # genuinely LAN-only deployments.
        mdns_enabled = os.environ.get("PARALLAX_ENABLE_MDNS", "").strip() == "1"
        logger.debug(
            f"Starting Lattica with host_maddrs={self.host_maddrs}, mdns={mdns_enabled}, dht_prefix={self.dht_prefix}"
        )
        self.lattica = (
            Lattica.builder()
            .with_listen_addrs(self.host_maddrs)
            .with_key_path(".")
        )
        if not mdns_enabled:
            self.lattica.with_mdns(False)

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

        # Keep the DHT entry warm. Without this, the key drops out of the
        # Kademlia cache after ~1-2h and workers can no longer discover the
        # scheduler — see comment on SCHEDULER_PEER_ID_REANNOUNCE_SEC.
        self._start_peer_id_reannouncer()

        self.connection_handler = RPCConnectionHandler(
            lattica=self.lattica,
            scheduler=self.scheduler,
            http_port=self.http_port,
        )
        logger.debug("RPCConnectionHandler initialized")

    def _start_peer_id_reannouncer(self):
        """Background daemon that re-stores the scheduler peer ID in the DHT.

        Idempotent: if a thread is already running, do nothing. The thread
        exits silently when `self.lattica` is set to None (e.g. on shutdown).
        """
        if getattr(self, "_peer_id_reannouncer_started", False):
            return
        self._peer_id_reannouncer_started = True

        def _loop():
            while True:
                try:
                    time.sleep(SCHEDULER_PEER_ID_REANNOUNCE_SEC)
                except Exception:
                    return
                lattica = self.lattica
                if lattica is None:
                    logger.debug("Lattica is gone, stopping peer-id reannouncer")
                    return
                try:
                    if lattica.store(
                        "scheduler_peer_id",
                        lattica.peer_id(),
                        expiration_time=time.time() + 365 * 24 * 60 * 60,
                    ):
                        logger.debug(
                            "Re-stored scheduler peer id in DHT: %s",
                            lattica.peer_id(),
                        )
                    else:
                        logger.warning(
                            "Re-store of scheduler peer id returned False; will retry in %ds",
                            SCHEDULER_PEER_ID_REANNOUNCE_SEC,
                        )
                except Exception as exc:
                    logger.warning(
                        "Re-store of scheduler peer id raised: %s; will retry in %ds",
                        exc,
                        SCHEDULER_PEER_ID_REANNOUNCE_SEC,
                    )

        threading.Thread(
            target=_loop,
            name="SchedulerPeerIdReannouncer",
            daemon=True,
        ).start()
        logger.info(
            "Scheduler peer-id DHT re-announcer started (every %ds)",
            SCHEDULER_PEER_ID_REANNOUNCE_SEC,
        )

    def get_routing_table(self, request_id, received_ts):
        """Block briefly until the scheduler assigns a routing path for the request.

        Distinguish three states via `RequestSignal.routing_table`:
        - None: not yet decided, keep waiting up to timeout
        - []: decided but no capacity (pipelines full), return immediately
        - [..]: valid routing path, return immediately
        """
        logger.debug(f"Routing table requested for request_id={request_id}")
        request = RequestSignal(request_id, received_ts)
        self.scheduler.receive_request(request)

        # Wait up to 5 seconds, but return immediately if the routing table is set (including an empty list)
        start_time = time.time()
        while request.routing_table is None and (time.time() - start_time) < 5.0:
            time.sleep(0.05)

        # Return the routing_table
        if request.routing_table is None:
            logger.debug(
                f"Routing table not ready after {(time.time() - start_time):.2f}s for request_id={request_id}"
            )
        else:
            logger.debug(
                f"Routing table resolved for request_id={request_id}: {request.routing_table}"
            )
        return request.routing_table

    def get_schedule_status(self):
        """
        Return whether a full pipeline has been allocated across joined nodes.
        """
        if self.scheduler is None:
            logger.debug("SchedulerManage status queried: waiting (scheduler not initialized)")
            return NODE_STATUS_WAITING

        # todo rebalance status
        status = (
            NODE_STATUS_AVAILABLE if self.scheduler.has_full_pipeline() else NODE_STATUS_WAITING
        )
        logger.debug(f"SchedulerManage status queried: {status}")
        return status

    def get_call_url_by_node_id(self, node_id):
        """
        Lookup the HTTP endpoint for a given node id managed by the RPC layer.
        """
        url = self.connection_handler.get_call_url_by_node_id(node_id)
        logger.debug(f"Lookup call_url for node_id={node_id} -> {url}")
        return url
