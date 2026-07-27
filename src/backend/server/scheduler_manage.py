import os
import threading
import time
from typing import List, Literal

from lattica import Lattica

from backend.server.constants import NODE_STATUS_AVAILABLE, NODE_STATUS_WAITING
from backend.server.context_admission import ContextBudget, build_context_budget
from backend.server.rpc_connection_handler import RPCConnectionHandler
from backend.server.static_config import get_model_info, get_node_join_command
from fabi_network.transport import IrohTransport, using_iroh
from parallax.cli import PUBLIC_INITIAL_PEERS, PUBLIC_RELAY_SERVERS
from parallax.p2p.liveness import (
    DEFAULT_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS,
    validate_scheduler_heartbeat_timeout,
)
from parallax.p2p.server import TransformerConnectionHandler
from parallax.p2p.utils import log_nat_traversal_preflight, mdns_enabled_for_topology
from parallax_utils.logging_config import get_logger
from scheduling.node import RequestSignal, node_is_routable
from scheduling.scheduler import Scheduler
from swarm_protocol.active import ActiveRouteRuntime
from swarm_protocol.coordinator import RouteReservationError
from swarm_protocol.contracts import RecoveryLevel
from swarm_protocol.epochs import InMemoryEpochAllocator, SqliteEpochAllocator
from swarm_protocol.recovery import (
    InMemoryRecoveryJournal,
    RecoveryConflict,
    RecoveryState,
    RequestRecoverySnapshot,
    RequestRecoverySpec,
    sampling_replay_contract,
)
from swarm_protocol.routing import RoutePlanningError

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
        heartbeat_timeout: float = DEFAULT_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS,
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
        self.heartbeat_timeout = validate_scheduler_heartbeat_timeout(heartbeat_timeout)
        self.swarm_v3_mode = os.environ.get("FABI_SWARM_V3_MODE", "off").strip().lower()
        if self.swarm_v3_mode in {"", "disabled"}:
            self.swarm_v3_mode = "off"
        if self.swarm_v3_mode not in {"off", "shadow", "active"}:
            raise ValueError("FABI_SWARM_V3_MODE supports only off, shadow, or active")
        self.swarm_v3_recovery = os.environ.get("FABI_SWARM_V3_RECOVERY", "prefer").strip().lower()
        if self.swarm_v3_recovery not in {"off", "prefer", "require"}:
            raise ValueError("FABI_SWARM_V3_RECOVERY supports only off, prefer, or require")
        self.model_name = None
        self.init_nodes_num = None
        self.scheduler = None
        self.node_id = f"{dht_prefix}_announce"
        self.lattica = None
        self.iroh_transport = None
        self.active_v3_routes = None
        self.stubs = {}
        self.is_local_network = False
        self._context_tokenizer = None
        self._context_tokenizer_model = None
        self._context_tokenizer_lock = threading.Lock()
        self.recovery_journal = InMemoryRecoveryJournal()
        epoch_db = os.environ.get("FABI_SWARM_V3_EPOCH_DB")
        self.epoch_allocator = (
            SqliteEpochAllocator(epoch_db, namespace="scheduler-control-plane")
            if epoch_db
            else InMemoryEpochAllocator()
        )

    @staticmethod
    def _positive_context_env(name: str, default: int) -> int:
        raw = os.environ.get(name, str(default))
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

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
        if (
            not using_iroh()
            and not is_local_network
            and not self.initial_peers
            and not self.relay_servers
        ):
            logger.debug("Using public relay servers")
            self.initial_peers = PUBLIC_INITIAL_PEERS
            self.relay_servers = PUBLIC_RELAY_SERVERS

        self._start_scheduler(model_name, init_nodes_num)
        self._start_lattica()
        self._attach_v3_catalog()
        self._start_active_v3_routes()
        self.completion_handler = TransformerConnectionHandler(
            lattica=None if self.iroh_transport is not None else self.lattica,
            recv_from_peer_addr="",
            send_to_peer_addr="",
            block_start_index=0,
            block_end_index=1,
            iroh_transport=self.iroh_transport,
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

        if self.active_v3_routes is not None:
            self.active_v3_routes.close()
            self.active_v3_routes = None

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
                "heartbeat_timeout_seconds": self.heartbeat_timeout,
                "chunked_prefill_size": (
                    self.scheduler.negotiated_chunked_prefill_size() if self.scheduler else 0
                ),
                "prefill_contract_ready": (
                    self.scheduler.prefill_contract_ready() if self.scheduler else False
                ),
                "planned_context_tokens": (
                    self.scheduler.layer_allocator.selected_context_tokens if self.scheduler else 0
                ),
                "allocation_epoch": (
                    int(self.scheduler.allocation_epoch) if self.scheduler else None
                ),
                "runtime_memory_contract_ready": (
                    self.scheduler.runtime_memory_contract_ready() if self.scheduler else False
                ),
                "max_supported_context_tokens": self.max_supported_context_tokens(),
                "node_join_command": get_node_join_command(
                    self.get_peer_id(), self.is_local_network
                ),
                "node_list": self.get_node_list(),
                "need_more_nodes": self.need_more_nodes(),
                "swarm_v3_shadow": (
                    self.scheduler.swarm_v3_shadow_snapshot if self.scheduler else None
                ),
                "swarm_v3_execution": (
                    self.active_v3_routes.snapshot()
                    if self.active_v3_routes is not None
                    else {"mode": "off", "active_routes": []}
                ),
                "swarm_v3_recovery": {
                    "policy": self.swarm_v3_recovery,
                    **self.recovery_journal.status(),
                },
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
        swarm_v3 = getattr(node, "swarm_v3", None)
        swarm_v3_error = swarm_v3.get("error") if isinstance(swarm_v3, dict) else None
        if isinstance(swarm_v3_error, dict):
            swarm_v3_error = {
                key: str(swarm_v3_error[key])[:256]
                for key in ("code", "detail")
                if swarm_v3_error.get(key) is not None
            }
        else:
            swarm_v3_error = None
        return {
            "node_id": node.node_id,
            "status": (NODE_STATUS_AVAILABLE if node_is_routable(node) else NODE_STATUS_WAITING),
            "gpu_num": node.hardware.num_gpus,
            "gpu_name": node.hardware.gpu_name,
            "gpu_memory": node.hardware.memory_gb,
            "supports_frontend": node.supports_frontend,
            "max_sequence_length": getattr(node, "max_sequence_length", None),
            "max_concurrent_requests": getattr(node, "max_requests", None),
            "kv_cache_telemetry_ready": (
                getattr(node, "kv_cache_token_capacity", None) is not None
            ),
            "swarm_v3_state": (swarm_v3.get("state") if isinstance(swarm_v3, dict) else None),
            "swarm_v3_error": swarm_v3_error,
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
            "reachable_link_telemetry_ready": (
                getattr(node, "reachable_peer_ids", None) is not None
            ),
            "reachable_peer_ids": (
                sorted(node.reachable_peer_ids)
                if getattr(node, "reachable_peer_ids", None) is not None
                else None
            ),
            "relayed_peer_ids": (
                sorted(node.relayed_peer_ids)
                if getattr(node, "relayed_peer_ids", None) is not None
                else None
            ),
            "rtt_to_nodes_ms": dict(getattr(node, "rtt_to_nodes", {}) or {}),
            "liveness": {
                "state": getattr(node, "liveness_state", "healthy"),
                "phi": round(float(getattr(node, "liveness_phi", 0.0)), 6),
                "heartbeat_age_seconds": round(
                    float(getattr(node, "heartbeat_age_seconds", 0.0)), 3
                ),
                "mean_interval_seconds": round(
                    float(getattr(node, "heartbeat_mean_interval_seconds", 0.0)), 3
                ),
                "std_deviation_seconds": round(
                    float(getattr(node, "heartbeat_std_deviation_seconds", 0.0)),
                    3,
                ),
                "samples": int(getattr(node, "heartbeat_samples", 0)),
                "local_health_multiplier": int(getattr(node, "local_health_multiplier", 1)),
            },
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
        if self.swarm_v3_mode == "active" and not os.environ.get("FABI_SWARM_V3_EPOCH_DB"):
            raise RuntimeError(
                "active protocol-v3 requires FABI_SWARM_V3_EPOCH_DB on persistent storage"
            )

        self.model_name = model_name
        self.init_nodes_num = init_nodes_num
        with self._context_tokenizer_lock:
            self._context_tokenizer = None
            self._context_tokenizer_model = None

        model_info = get_model_info(
            model_name,
            self.use_hfcache,
            load_weight_metadata=True,
        )
        self.scheduler = Scheduler(
            model_info,
            [],
            min_nodes_bootstrapping=init_nodes_num,
            enable_weight_refit=self.enable_weight_refit,
            weight_refit_mode=self.weight_refit_mode,
            strategy=self.allocation_strategy,
            routing_strategy=self.routing_strategy,
            heartbeat_timeout=self.heartbeat_timeout,
            planning_context_tokens=self._positive_context_env(
                "PARALLAX_PLANNING_CONTEXT_TOKENS", 16_384
            ),
            preferred_context_tokens=self._positive_context_env(
                "PARALLAX_PREFERRED_CONTEXT_TOKENS", 32_768
            ),
            require_exact_weight_metadata=True,
            epoch_allocator=self.epoch_allocator,
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
        if using_iroh():
            self._start_iroh()
            return

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

        mdns_enabled = mdns_enabled_for_topology(
            initial_peers=self.initial_peers,
            relay_servers=self.relay_servers,
        )
        logger.debug(
            f"Starting Lattica with host_maddrs={self.host_maddrs}, mdns={mdns_enabled}, dht_prefix={self.dht_prefix}"
        )
        self.lattica = Lattica.builder().with_listen_addrs(self.host_maddrs).with_key_path(".")
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
            log_nat_traversal_preflight(self.lattica, logger)

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

    def _start_iroh(self):
        """Start or reuse the centrally scheduled Iroh RPC endpoint."""

        if self.iroh_transport is not None:
            self.connection_handler.scheduler = self.scheduler
            logger.debug("Updated Iroh scheduler RPC handler")
            return

        transport = IrohTransport.from_environment("scheduler")
        handler = RPCConnectionHandler(
            lattica=None,
            scheduler=self.scheduler,
            http_port=self.http_port,
        )
        handler.iroh_transport = transport
        transport.register(handler)
        self.iroh_transport = transport
        # Keep the legacy attribute during the staged migration; callers only
        # depend on peer_id/close in central scheduler mode.
        self.lattica = transport
        self.connection_handler = handler
        logger.info("Iroh scheduler endpoint ready: %s", transport.peer_id())

    def _start_active_v3_routes(self) -> None:
        """Activate v3 traffic only when the verified planner and Iroh are ready."""

        planner = self.scheduler.swarm_v3_shadow if self.scheduler is not None else None
        if self.swarm_v3_mode != "active":
            return
        if planner is None or planner.mode != "active":
            raise RuntimeError("protocol-v3 active planner failed to initialize")
        if self.iroh_transport is None:
            raise RuntimeError("protocol-v3 active mode requires the authenticated Iroh transport")
        if self.active_v3_routes is not None:
            self.active_v3_routes.close()
        self.active_v3_routes = ActiveRouteRuntime(
            planner=planner,
            transport=self.iroh_transport,
            nodes_provider=lambda: list(self.scheduler.node_manager.nodes),
            epoch_allocator=self.epoch_allocator,
        )
        self.scheduler.external_routes_active = self.active_v3_routes.has_active_routes
        logger.info("Protocol-v3 active route admission is ready")

    def _attach_v3_catalog(self) -> None:
        planner = self.scheduler.swarm_v3_shadow if self.scheduler is not None else None
        catalog = self.iroh_transport.catalog_discovery if self.iroh_transport is not None else None
        if planner is not None and catalog is not None:
            planner.attach_catalog(catalog)
            logger.info(
                "Protocol-v3 catalogue attached: peer=%s",
                self.iroh_transport.catalog_peer_id,
            )

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

    def get_routing_table(
        self,
        request_id,
        received_ts,
        required_context_tokens: int = 0,
        *,
        prompt_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        recovery_level: RecoveryLevel = RecoveryLevel.RESTARTABLE,
    ):
        """Block briefly until the scheduler assigns a routing path for the request.

        Distinguish three states via `RequestSignal.routing_table`:
        - None: not yet decided, keep waiting up to timeout
        - []: decided but no capacity (pipelines full), return immediately
        - [..]: valid routing path, return immediately
        """
        logger.debug(f"Routing table requested for request_id={request_id}")
        if self.active_v3_routes is not None:
            planner = getattr(self.scheduler, "swarm_v3_shadow", None)
            dht_workers = planner.live_worker_ids() if planner is not None else None
            if self.scheduler._admission_paused or (
                dht_workers is None and not self.scheduler.serving_ready()
            ):
                return []
            if prompt_tokens is None or reserved_output_tokens is None:
                raise ValueError("active v3 routing requires exact prompt and output token budgets")
            if prompt_tokens + reserved_output_tokens != required_context_tokens:
                raise ValueError("active v3 routing token budget is internally inconsistent")
            try:
                return list(
                    self.active_v3_routes.reserve(
                        request_id=str(request_id),
                        prompt_tokens=prompt_tokens,
                        reserved_output_tokens=reserved_output_tokens,
                        recovery_level=recovery_level,
                    )
                )
            except (RoutePlanningError, RouteReservationError) as exc:
                if (
                    recovery_level == RecoveryLevel.RECOVERABLE
                    and self.swarm_v3_recovery == "prefer"
                ):
                    logger.info(
                        "No fully reserved recovery route for %s; falling back to an "
                        "explicitly restartable route: %s",
                        request_id,
                        exc,
                    )
                    try:
                        return list(
                            self.active_v3_routes.reserve(
                                request_id=str(request_id),
                                prompt_tokens=prompt_tokens,
                                reserved_output_tokens=reserved_output_tokens,
                                recovery_level=RecoveryLevel.RESTARTABLE,
                            )
                        )
                    except (RoutePlanningError, RouteReservationError) as fallback_exc:
                        exc = fallback_exc
                logger.info("Protocol-v3 route not currently admissible: %s", exc)
                return []

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
        if self.active_v3_routes is not None:
            return self.active_v3_routes.release(str(request_id))
        return self.scheduler.release_request(str(request_id))

    def is_routing_table_active(self, request_id: str) -> bool:
        """Return whether a dispatched request still owns a live worker route."""
        if self.scheduler is None:
            return False
        if self.active_v3_routes is not None:
            return self.active_v3_routes.is_active(str(request_id))
        return self.scheduler.is_request_route_active(str(request_id))

    def get_route_authority(self, request_id: str) -> dict[str, object] | None:
        """Return v3 data-plane fencing metadata for the active route."""

        if self.active_v3_routes is None:
            return None
        return self.active_v3_routes.authority(str(request_id))

    def preferred_recovery_level(self, request_data) -> RecoveryLevel:
        """Choose only guarantees the current data plane can reproduce exactly."""

        if (
            self.active_v3_routes is None
            or self.swarm_v3_recovery == "off"
            or not request_data.get("stream", False)
            or sampling_replay_contract(request_data) is None
        ):
            return RecoveryLevel.RESTARTABLE
        return RecoveryLevel.RECOVERABLE

    def should_capture_generation_tokens(self, request_id: str, request_data) -> bool:
        """Return whether this admitted request has an exact token journal contract."""

        return (
            self.active_v3_routes is not None
            and self.active_v3_routes.execution_context(str(request_id)) is not None
            and sampling_replay_contract(request_data) is not None
        )

    def begin_generation_journal(
        self,
        request_id: str,
        *,
        engine_prompt_token_ids: tuple[int, ...],
        expected_prompt_token_ids: tuple[int, ...],
        request_data,
    ) -> RequestRecoverySnapshot:
        """Bind official engine token IDs to the admitted route before decode."""

        if self.active_v3_routes is None:
            raise RecoveryConflict("active v3 routing is not enabled")
        context = self.active_v3_routes.execution_context(str(request_id))
        if context is None:
            raise RecoveryConflict("request route is no longer active")
        if engine_prompt_token_ids != expected_prompt_token_ids:
            raise RecoveryConflict(
                "engine prompt token IDs differ from scheduler context admission"
            )
        plan = context.primary_plan
        if len(engine_prompt_token_ids) != plan.prompt_tokens:
            raise RecoveryConflict("engine prompt token count differs from the reserved route")
        sampling = sampling_replay_contract(request_data)
        if sampling is None:
            raise RecoveryConflict("request sampling is not exactly replayable")
        manifest = context.manifest
        if manifest.model_swarm_id != plan.model_swarm_id:
            raise RecoveryConflict("active route and trusted manifest identify different swarms")
        if context.effective_recovery_level == RecoveryLevel.RECOVERABLE:
            recovery_plan = context.recovery_plan
            if (
                recovery_plan is None
                or recovery_plan.model_swarm_id != plan.model_swarm_id
                or recovery_plan.epoch != plan.epoch
            ):
                raise RecoveryConflict("recoverable route has no compatible reserved backup")
        return self.recovery_journal.begin(
            RequestRecoverySpec(
                request_id=str(request_id),
                model_swarm_id=manifest.model_swarm_id,
                immutable_revision=manifest.immutable_revision,
                tokenizer_hash=manifest.tokenizer_hash,
                dtype=manifest.dtype,
                prefill_contract_hash=manifest.prefill_contract_hash,
                attention_kv_contract_hash=manifest.attention_kv_contract_hash,
                prompt_token_ids=engine_prompt_token_ids,
                sampling=sampling,
                recovery_level=context.effective_recovery_level,
                primary_route_id=plan.route_id,
                epoch=plan.epoch,
                reserved_context_tokens=plan.required_context_tokens,
            )
        )

    def commit_generation_prefill(self, request_id: str, *, epoch: int) -> None:
        snapshot = self._reconcile_generation_recovery(str(request_id), epoch=epoch)
        if snapshot is None:
            raise RecoveryConflict("request is not present in the recovery journal")
        self.recovery_journal.commit_prefill(
            str(request_id),
            epoch=epoch,
            prompt_checksum=snapshot.prompt_checksum,
        )

    def commit_generation_tokens(
        self,
        request_id: str,
        *,
        epoch: int,
        token_ids: tuple[int, ...],
    ) -> None:
        self._reconcile_generation_recovery(str(request_id), epoch=epoch)
        for token_id in token_ids:
            snapshot = self.recovery_journal.get(str(request_id))
            if snapshot is None:
                raise RecoveryConflict("request is not present in the recovery journal")
            self.recovery_journal.commit_token(
                str(request_id),
                epoch=epoch,
                position=snapshot.committed_position,
                token_id=token_id,
            )

    def finish_generation_journal(
        self,
        request_id: str,
        *,
        epoch: int,
        state: RecoveryState,
        failure: str | None = None,
    ) -> None:
        if self._reconcile_generation_recovery(str(request_id), epoch=epoch) is None:
            return
        self.recovery_journal.finish(
            str(request_id),
            epoch=epoch,
            state=state,
            failure=failure,
        )

    def _reconcile_generation_recovery(
        self,
        request_id: str,
        *,
        epoch: int,
    ) -> RequestRecoverySnapshot | None:
        """Keep the journal's live guarantee aligned with reserved topology."""

        snapshot = self.recovery_journal.get(str(request_id))
        if (
            snapshot is None
            or snapshot.effective_recovery_level != RecoveryLevel.RECOVERABLE
            or self.active_v3_routes is None
        ):
            return snapshot
        context = self.active_v3_routes.execution_context(str(request_id))
        if context is not None and context.effective_recovery_level == RecoveryLevel.RESTARTABLE:
            return self.recovery_journal.downgrade_to_restartable(
                str(request_id),
                epoch=epoch,
                reason="reserved backup route is no longer available",
            )
        return snapshot

    def wait_for_routing_capacity(self, timeout: float) -> bool:
        """Block until a route can be admitted or the bounded wait expires."""
        if self.scheduler is None:
            return False
        if self.active_v3_routes is not None:
            return self.active_v3_routes.wait_for_capacity(timeout)
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
