"""
Scheduler for Layer Allocation and Request Routing.
"""

from __future__ import annotations

import copy
import os
import queue
import threading
import time
from collections import deque
from math import isfinite
from typing import Callable, Deque, Dict, List, Literal, Optional, Tuple, TypeAlias

from parallax_utils.logging_config import get_logger
from parallax.p2p.liveness import (
    LivenessSnapshot,
    LocalHealthAwareness,
    PhiAccrualFailureDetector,
)
from scheduling.layer_allocation import (
    DynamicProgrammingLayerAllocator,
    GreedyLayerAllocator,
)
from scheduling.model_info import ModelInfo
from scheduling.node import Node, RequestSignal
from scheduling.node_management import NodeManager
from scheduling.request_routing import (
    DynamicProgrammingRouting,
    RoundRobinOverFixedPipelinesRouting,
)
from swarm_protocol.epochs import EpochAllocator, InMemoryEpochAllocator
from swarm_protocol.shadow import SchedulerProtocolV3Shadow

logger = get_logger(__name__)

_CONTEXT_REPLAN_INITIAL_BACKOFF_SECONDS = 1.0
_CONTEXT_REPLAN_MAX_BACKOFF_SECONDS = 30.0

NodeUpdate: TypeAlias = Tuple[
    str,
    Optional[int],
    Optional[float],
    Optional[Dict[str, float]],
    Optional[bool],
    Optional[float],
    Optional[bool],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[List[str]],
    Optional[List[str]],
    Optional[List[str]],
    Optional[str],
    Optional[Dict[str, object]],
    Optional[Dict[str, object]],
]


class Scheduler:
    """Coordinates allocation, node materialization, and request routing."""

    def __init__(
        self,
        model_info: ModelInfo,
        nodes: List[Node],
        min_nodes_bootstrapping: int = 1,
        enable_weight_refit: bool = False,
        weight_refit_mode: str = "disk",
        strategy: Literal["greedy", "dp"] = "dp",
        routing_strategy: Literal["rr", "dp"] = "rr",
        *,
        request_arrival_horizon_sec: float = 600.0,
        rebalance_threshold: float = float("inf"),
        water_filling_max_iterations: int = 40,
        heartbeat_timeout: float = 30.0,
        trim_layers_on_turning_points: bool = False,
        planning_context_tokens: int = 16_384,
        preferred_context_tokens: int = 32_768,
        require_exact_weight_metadata: bool = False,
        epoch_allocator: EpochAllocator | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            model_info: Model architecture information used by allocators and routers.
            nodes: Initial list of candidate nodes.
            min_nodes_bootstrapping: Minimum nodes required to attempt initial allocation.
            strategy: Layer allocation strategy ("dp" or "greedy").
            routing_strategy: Request routing strategy:
                - "dp": dynamic-programming routing over current allocations (minimum latency).
                - "rr": round-robin selection over fixed, small set of pipelines.
            request_arrival_horizon_sec: Sliding window horizon for arrival-rate tracking.
            rebalance_threshold: Threshold for triggering rebalancing in allocation.
            water_filling_max_iterations: Max iterations for water-filling allocation.
            heartbeat_timeout: Time in seconds to consider node heartbeat stale.
            trim_layers_on_turning_points: Whether to trim layers on turning points.
        """
        if strategy not in ("greedy", "dp"):
            raise ValueError(f"Unsupported layer allocation strategy: {strategy}")
        if routing_strategy not in ("rr", "dp"):
            raise ValueError(f"Unsupported request routing strategy: {routing_strategy}")

        self.model_info = model_info
        self.num_layers = model_info.num_layers
        self.routing_strategy: Literal["rr", "dp"] = routing_strategy
        self.enable_weight_refit = enable_weight_refit
        self.weight_refit_mode = weight_refit_mode
        self.refit_request = {}
        self.node_manager = NodeManager(initial_nodes=nodes)

        allocator_class = (
            GreedyLayerAllocator if strategy == "greedy" else DynamicProgrammingLayerAllocator
        )
        self.dynamic_pipelines_router = routing_strategy == "dp"
        # TODO: expose DP's alpha
        self.layer_allocator = allocator_class(
            model_info=model_info,
            node_management=self.node_manager,
            dynamic_pipelines_router=self.dynamic_pipelines_router,
            rebalance_threshold=rebalance_threshold,
            water_filling_max_iterations=water_filling_max_iterations,
            trim_layers_on_turning_points=trim_layers_on_turning_points,
            planning_context_tokens=planning_context_tokens,
            preferred_context_tokens=preferred_context_tokens,
            require_exact_weight_metadata=require_exact_weight_metadata,
        )
        # Ensure Scheduler and allocator share the same node list to avoid divergence.
        self.min_nodes_bootstrapping = min_nodes_bootstrapping

        self.request_router = (
            DynamicProgrammingRouting(self.node_manager, self.num_layers)
            if routing_strategy == "dp"
            else RoundRobinOverFixedPipelinesRouting(
                self.node_manager, self.num_layers, self.layer_allocator
            )
        )

        self._request_queue: "queue.Queue[RequestSignal]" = queue.Queue()
        # request id -> (ordered node path, per-node requested context tokens)
        self._inflight_routes: Dict[str, Tuple[List[str], int]] = {}
        self._inflight_routes_lock = threading.RLock()
        self._capacity_cv = threading.Condition()
        self.request_arrival_horizon_sec = request_arrival_horizon_sec
        self.heartbeat_timeout = heartbeat_timeout
        self._liveness_lock = threading.RLock()
        self._liveness_detectors: Dict[str, PhiAccrualFailureDetector] = {}
        self._local_health = LocalHealthAwareness()
        self._arrival_ts: Deque[float] = deque()

        # Event queues for main loop orchestration (thread-safe)
        self._pending_joins: "queue.Queue[Node]" = queue.Queue()
        self._pending_leaves: "queue.Queue[str]" = queue.Queue()
        self._pending_node_updates: "queue.Queue[NodeUpdate]" = queue.Queue()
        # A late DP node whose fixed shard cannot join an exact-boundary route
        # requests one drained global allocation.  Membership changes set this
        # flag; memory-pressure samples deliberately do not, preventing reload
        # oscillations on desktop contributors.
        self._pending_rebalance_node_ids: set[str] = set()
        self._rebalance_after_leave_pending = False
        self._next_rebalance_attempt_at: float = 0.0
        self._pending_context_replan: Optional[Dict[str, object]] = None
        self.epoch_allocator = epoch_allocator or InMemoryEpochAllocator()
        self.allocation_epoch = self.epoch_allocator.current()

        # Concurrency controls
        self._stop_event: threading.Event = threading.Event()
        self._wake_event: threading.Event = threading.Event()
        self._bootstrapped_event: threading.Event = threading.Event()
        self._admission_paused: bool = False
        self.external_routes_active: Callable[[], bool] | None = None
        # Engine construction parameters are immutable in upstream Parallax,
        # vLLM and SGLang.  Keep the negotiated wire contract monotonic within
        # one allocation generation: compatibility may force it down, but a
        # transient node departure must not upgrade it and reload healthy
        # executors.  A deliberate global rebootstrap starts a new generation.
        self._prefill_contract_lock = threading.Lock()
        self._prefill_contract: Optional[int] = None
        self._node_count_cv: threading.Condition = threading.Condition()
        self._event_thread: Optional[threading.Thread] = None
        self._dispatch_thread: Optional[threading.Thread] = None
        self._alloc_log_thread: Optional[threading.Thread] = None
        # Latest formatted allocation snapshot (string) for status/inspection.
        # This is updated by `emit_alloc_log_snapshot()`.
        self.alloc_log_snapshot: str = ""
        # Avoid spamming: only emit the "all nodes active" INFO log on transitions.
        self._all_nodes_active_logged: bool = False
        self.swarm_v3_shadow = None
        self.swarm_v3_shadow_snapshot: Dict[str, object] = {
            "mode": "off",
            "state": "disabled",
        }
        try:
            self.swarm_v3_shadow = SchedulerProtocolV3Shadow.from_environment()
            if self.swarm_v3_shadow is not None:
                self.swarm_v3_shadow_snapshot = {
                    "mode": self.swarm_v3_shadow.mode,
                    "state": "waiting_workers",
                }
        except Exception as exc:
            configured_v3_mode = os.environ.get("FABI_SWARM_V3_MODE", "shadow").strip().lower()
            self.swarm_v3_shadow_snapshot = {
                "mode": configured_v3_mode,
                "state": "rejected",
                "error": {"code": type(exc).__name__, "detail": str(exc)[:256]},
            }
            logger.error("Protocol-v3 scheduler planner is disabled: %s", exc)
        logger.info(
            f"Scheduler initialized, min_nodes_bootstrapping {self.min_nodes_bootstrapping}, "
            f"Layer allocations trategy {strategy}, Request routing strategy {routing_strategy}."
        )

        # Weight refit
        self.refit_request = {}
        self.refit_set = set()
        self.last_refit_time = 0.0
        for node in self.node_manager.nodes:
            self._record_node_heartbeat(node, sample_interval=False)

    def list_node_allocations(
        self, total_layers: Optional[int] = None
    ) -> List[Tuple[str, int, int]]:
        """Return current (node_id, start_layer, end_layer) allocations.

        This is a small convenience wrapper around `NodeManager.list_node_allocations` and is
        relied upon by some callers (e.g. backend RPC handler, docs).
        """
        return self.node_manager.list_node_allocations(total_layers or self.num_layers)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Fetch a node by id (compat helper for callers that shouldn't reach into NodeManager)."""
        return self.node_manager.get(node_id)

    def has_full_pipeline(self) -> bool:
        """Check if there is a full pipeline among ACTIVE nodes."""
        return self.node_manager.has_full_pipeline(self.num_layers)

    def negotiated_chunked_prefill_size(self) -> int:
        """Return the cluster-wide chunked-prefill contract.

        Only nodes belonging to at least one complete structural pipeline can
        carry activations, so orphan dynamic ranges do not influence the live
        wire contract.  Within an allocation generation the contract can only
        become more conservative.  This mirrors the immutable executor startup
        configuration in upstream runtimes and prevents a node departure from
        causing an opportunistic cluster-wide reload.
        """

        participant_ids = self.node_manager.full_pipeline_node_ids(self.num_layers)
        participants = [
            node for node in self.node_manager.active_nodes if node.node_id in participant_ids
        ]
        if not participants:
            with self._prefill_contract_lock:
                return 0 if self._prefill_contract is None else self._prefill_contract

        preferred_sizes: List[int] = []
        desired = 0
        for node in participants:
            preferred = node.preferred_chunked_prefill_size
            if not node.supports_chunked_prefill or preferred is None or preferred <= 0:
                break
            preferred_sizes.append(int(preferred))
        else:
            desired = min(preferred_sizes)

        with self._prefill_contract_lock:
            if self._prefill_contract is None or desired < self._prefill_contract:
                self._prefill_contract = desired
            return self._prefill_contract

    def chunked_prefill_size_for_node(self, node_id: str) -> int:
        """Return the contract an allocated node should materialize.

        Nodes outside every complete path are not part of the activation wire.
        They keep a backend-safe local setting until a later allocation makes
        them routeable, at which point normal negotiation may request one
        intentional reload.
        """

        participant_ids = self.node_manager.full_pipeline_node_ids(self.num_layers)
        if node_id in participant_ids:
            return self.negotiated_chunked_prefill_size()

        node = self.get_node(node_id)
        if node is None or not node.supports_chunked_prefill:
            return 0
        preferred = node.preferred_chunked_prefill_size
        return int(preferred) if preferred is not None and preferred > 0 else 0

    def _reset_prefill_contract(self) -> None:
        """Start a fresh immutable contract generation after a full reboot."""

        with self._prefill_contract_lock:
            self._prefill_contract = None

    def prefill_contract_ready(self) -> bool:
        """Whether every allocated worker runs the negotiated prefill contract."""
        participant_ids = self.node_manager.full_pipeline_node_ids(self.num_layers)
        participants = [
            node for node in self.node_manager.active_nodes if node.node_id in participant_ids
        ]
        if not participants:
            return False
        negotiated = self.negotiated_chunked_prefill_size()
        return all(node.chunked_prefill_size == negotiated for node in participants)

    def serving_ready(self) -> bool:
        """Whether a live full pipeline is safe to receive requests."""
        return (
            not self._admission_paused
            and self.node_manager.has_full_pipeline(self.num_layers, ready_only=True)
            and self.prefill_contract_ready()
            and self.request_router.routing_ready()
            and self.runtime_memory_contract_ready()
        )

    def runtime_memory_contract_ready(self) -> bool:
        """Require executor-measured KV geometry to satisfy the selected plan."""

        if not self.layer_allocator.require_exact_weight_metadata:
            return True
        selected = int(self.layer_allocator.selected_context_tokens)
        return selected > 0 and self.max_supported_context_tokens() >= selected

    def max_supported_context_tokens(self) -> int:
        """Largest request context that at least one complete route can hold."""
        route_limit = self.request_router.max_supported_context_tokens()
        model_limit = getattr(self.model_info, "max_context_length", None)
        if model_limit is None:
            return route_limit
        return min(route_limit, int(model_limit))

    def _record_memory_contract_failure(
        self,
        node: Node,
        failure: Dict[str, object],
    ) -> None:
        """Accept one measured, epoch-fenced downward replan request."""

        if not self.layer_allocator.require_exact_weight_metadata:
            return
        try:
            kind = str(failure["kind"])
            failed_epoch = int(failure["allocation_epoch"])
            requested_tokens = int(failure["requested_tokens"])
            supported_tokens = int(failure["supported_tokens"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Ignoring malformed memory contract report from %s", node.node_id)
            return
        selected_tokens = int(self.layer_allocator.selected_context_tokens)
        if kind != "kv_materialization":
            return
        if failed_epoch != self.allocation_epoch:
            logger.info(
                "Ignoring stale memory contract report from %s for epoch %d (current=%d)",
                node.node_id,
                failed_epoch,
                self.allocation_epoch,
            )
            return
        if requested_tokens != selected_tokens or supported_tokens >= requested_tokens:
            logger.warning(
                "Ignoring inconsistent memory contract report from %s: requested=%d, "
                "supported=%d, selected=%d",
                node.node_id,
                requested_tokens,
                supported_tokens,
                selected_tokens,
            )
            return
        if self._pending_context_replan is not None:
            return
        downgraded_tokens = self.layer_allocator.fence_one_runtime_context_downgrade(
            requested_tokens
        )
        if downgraded_tokens is None:
            logger.error(
                "Runtime memory contract failed on %s at %d tokens and no further bounded "
                "downgrade is permitted",
                node.node_id,
                requested_tokens,
            )
            return
        self._pending_context_replan = {
            "failed_epoch": failed_epoch,
            "node_id": node.node_id,
            "from_tokens": requested_tokens,
            "to_tokens": downgraded_tokens,
            "supported_tokens": supported_tokens,
            "attempts": 0,
            "next_attempt_at": 0.0,
        }
        self._admission_paused = True
        logger.warning(
            "Queued one fenced runtime context downgrade after %s measured %d-token "
            "capacity: %d -> %d tokens (epoch=%d)",
            node.node_id,
            supported_tokens,
            requested_tokens,
            downgraded_tokens,
            failed_epoch,
        )

    def _process_pending_context_replan(self) -> bool:
        """Reconcile one fenced downgrade without hot-looping on infeasibility.

        The event loop owns a single pending item. Failed reconciliation uses a
        bounded exponential delay, while a new worker registration explicitly
        makes it eligible immediately. The context fence itself is never
        applied twice.
        """

        pending = self._pending_context_replan
        if pending is None:
            return False
        now = time.monotonic()
        if now < float(pending.get("next_attempt_at", 0.0)):
            return False
        with self._inflight_routes_lock:
            if (
                self._inflight_routes
                or any(
                    node.routing_load > 0 or node.current_requests > 0
                    for node in self.node_manager.active_nodes
                )
                or (self.external_routes_active is not None and self.external_routes_active())
            ):
                return False
            active_ids = [node.node_id for node in self.node_manager.active_nodes]
            if active_ids:
                self.node_manager.standby(active_ids)
            attempts = int(pending.get("attempts", 0)) + 1
            pending["attempts"] = attempts
            logger.warning(
                "Reconciling allocation epoch %d at the fenced %d-token ceiling (attempt=%d)",
                int(pending["failed_epoch"]),
                int(pending["to_tokens"]),
                attempts,
            )
            success = self.bootstrap(reboot=True)
            if not success:
                delay_seconds = min(
                    _CONTEXT_REPLAN_INITIAL_BACKOFF_SECONDS * (2 ** min(attempts - 1, 5)),
                    _CONTEXT_REPLAN_MAX_BACKOFF_SECONDS,
                )
                pending["next_attempt_at"] = time.monotonic() + delay_seconds
                logger.error(
                    "Fenced runtime downgrade could not form a complete %d-token route; "
                    "retrying after %.1fs or immediately on a worker registration",
                    int(pending["to_tokens"]),
                    delay_seconds,
                )
                return False
            self._pending_context_replan = None
            self._admission_paused = False
            return True

    def _wake_pending_context_replan(self) -> None:
        """Make a failed reconciliation immediately eligible after new capacity."""

        pending = self._pending_context_replan
        if pending is None:
            return
        pending["attempts"] = 0
        pending["next_attempt_at"] = 0.0

    def report_pipeline_capacity(
        self,
        ready_only: bool = True,
    ) -> Tuple[Optional[Dict[int, Tuple[int, int]]], int, int]:
        """Backward-compatible capacity report delegated to ``NodeManager``."""
        return self.node_manager.report_pipeline_capacity(ready_only=ready_only)

    def bootstrap(self, reboot: bool = False) -> bool:
        """Initial Node Allocation Assignment."""
        logger.info("[Scheduler] Starting Bootstrap")
        overide_min_node_check = False
        if reboot:
            # Clear any fixed pipeline registrations; they are no longer valid.
            # This also detaches member nodes and clears their layer allocations.
            logger.info("[Scheduler] Rebooting, moving every node to standby")
            self._reset_prefill_contract()
            self.node_manager.clear_registered_pipelines()
            self._bootstrapped_event.clear()
            overide_min_node_check = True
        else:
            # If we already bootstrapped, return True
            if self._bootstrapped_event.is_set():
                logger.info("[Scheduler] Already bootstrapped, returning Success")
                return True
        # Check if we have enough nodes for bootstraping
        if (
            self.node_manager.num_nodes < self.min_nodes_bootstrapping
            and not overide_min_node_check
        ):
            logger.info(
                f"[Scheduler] Bootstrap deferred: have {self.node_manager.num_nodes} nodes; need >= {self.min_nodes_bootstrapping}"
            )
            return False

        # Perform global allocation
        try:
            success = self.layer_allocator.allocate_from_standby()
        except Exception:
            # A malformed or unexpectedly infeasible allocation must not strand
            # a worker's synchronous join RPC.  Roll back any partial bootstrap
            # so a later capacity update or node join can retry cleanly.
            logger.exception("Global allocation raised; rolling back partial bootstrap")
            for node in list(self.node_manager.active_nodes):
                try:
                    self.layer_allocator.deallocate(node)
                except Exception:
                    logger.exception("Failed to roll back allocation for node %s", node.node_id)
            self._bootstrapped_event.clear()
            return False
        if not success:
            logger.warning("Global allocation failed to produce a full pipeline")
            # Stay un-bootstrapped so future joins can retry bootstrap.
            self._bootstrapped_event.clear()
            return False

        assignments = self.node_manager.list_node_allocations(self.num_layers)
        logger.info(f"[Scheduler] Post Bootstrap Layer Assignments: {assignments}")

        self.request_router.bootstrap()
        # Joining workers wait for bootstrap before they can start their heartbeat loop.
        # Grant every newly active node a full lease once its allocation is ready.
        lease_started_at = time.time()
        for node in self.node_manager.active_nodes:
            node.last_heartbeat = lease_started_at
            self._record_node_heartbeat(node, sample_interval=False)
        self.allocation_epoch = self.epoch_allocator.next_epoch()
        self._bootstrapped_event.set()
        self._queue_bootstrap_standby_rebalances()
        # Snapshot at INFO after bootstrap since allocations/pipelines may have materially changed.
        self.emit_alloc_log_snapshot(reason="Post Bootstrap")
        return True

    def update_last_refit_time(self):
        min_refit_time = None
        for node in self.node_manager.nodes:
            cur_node_refit_time = node.last_refit_time
            if cur_node_refit_time < self.last_refit_time:
                continue
            if min_refit_time is None:
                min_refit_time = cur_node_refit_time
            else:
                min_refit_time = min(min_refit_time, cur_node_refit_time)
        if min_refit_time is not None:
            self.last_refit_time = min_refit_time
        return self.last_refit_time

    def update_node_info(
        self,
        node: Node,
        *,
        current_requests: Optional[int] = None,
        layer_latency_ms: Optional[float] = None,
        new_rtt_to_nodes: Optional[Dict[str, float]] = None,
        is_active: Optional[bool] = None,
        last_refit_time: Optional[float] = 0.0,
        supports_chunked_prefill: Optional[bool] = None,
        preferred_chunked_prefill_size: Optional[int] = None,
        chunked_prefill_size: Optional[int] = None,
        kv_cache_token_capacity: Optional[int] = None,
        kv_cache_block_size: Optional[int] = None,
        max_concurrent_requests: Optional[int] = None,
        direct_peer_ids: Optional[List[str]] = None,
        reachable_peer_ids: Optional[List[str]] = None,
        relayed_peer_ids: Optional[List[str]] = None,
        account_hash: Optional[str] = None,
        memory_contract_failure: Optional[Dict[str, object]] = None,
        swarm_v3: Optional[Dict[str, object]] = None,
    ) -> None:
        """Update the info of a node."""
        if current_requests is not None:
            node.current_requests = current_requests
        if layer_latency_ms is not None:
            node.set_layer_latency_ms(layer_latency_ms)
        if new_rtt_to_nodes is not None:
            node.rtt_to_nodes = new_rtt_to_nodes
        if is_active is not None:
            node.is_active = is_active
        if last_refit_time > 0.0:
            node.last_refit_time = last_refit_time
        if supports_chunked_prefill is not None:
            node.supports_chunked_prefill = supports_chunked_prefill
        if preferred_chunked_prefill_size is not None:
            node.preferred_chunked_prefill_size = preferred_chunked_prefill_size
        if chunked_prefill_size is not None:
            node.chunked_prefill_size = chunked_prefill_size
        if kv_cache_token_capacity is not None:
            node.kv_cache_token_capacity = int(kv_cache_token_capacity)
        if kv_cache_block_size is not None:
            node.kv_cache_block_size = int(kv_cache_block_size)
        if max_concurrent_requests is not None:
            node.max_concurrent_requests = int(max_concurrent_requests)
        if direct_peer_ids is not None:
            node.direct_peer_ids = set(direct_peer_ids)
        if reachable_peer_ids is not None:
            node.reachable_peer_ids = set(reachable_peer_ids)
        if relayed_peer_ids is not None:
            node.relayed_peer_ids = set(relayed_peer_ids)
        if account_hash is not None:
            node.account_hash = account_hash
        node.memory_contract_failure = memory_contract_failure
        if swarm_v3 is not None:
            node.swarm_v3 = swarm_v3
        if memory_contract_failure is not None:
            self._record_memory_contract_failure(node, memory_contract_failure)
        node.last_heartbeat = time.time()
        self._record_node_heartbeat(node)

    # Async-style event enqueuers for main loop
    def enqueue_join(self, node: Node) -> None:
        """Enqueue a join event."""
        logger.debug(f"Enqueueing join event for node {node.node_id}")
        self._pending_joins.put(node)
        self._wake_event.set()

    def enqueue_leave(self, node_id: str) -> None:
        """Enqueue a leave event."""
        self._pending_leaves.put(node_id)
        self._wake_event.set()

    def enqueue_node_update(
        self,
        node_id: str,
        *,
        current_requests: Optional[int] = None,
        layer_latency_ms: Optional[float] = None,
        new_rtt_to_nodes: Optional[Dict[str, float]] = None,
        is_active: Optional[bool] = None,
        last_refit_time: Optional[float] = 0.0,
        supports_chunked_prefill: Optional[bool] = None,
        preferred_chunked_prefill_size: Optional[int] = None,
        chunked_prefill_size: Optional[int] = None,
        kv_cache_token_capacity: Optional[int] = None,
        kv_cache_block_size: Optional[int] = None,
        max_concurrent_requests: Optional[int] = None,
        direct_peer_ids: Optional[List[str]] = None,
        reachable_peer_ids: Optional[List[str]] = None,
        relayed_peer_ids: Optional[List[str]] = None,
        account_hash: Optional[str] = None,
        memory_contract_failure: Optional[Dict[str, object]] = None,
        swarm_v3: Optional[Dict[str, object]] = None,
    ) -> None:
        """Enqueue a node update event."""
        self._pending_node_updates.put(
            (
                node_id,
                current_requests,
                layer_latency_ms,
                new_rtt_to_nodes,
                is_active,
                last_refit_time,
                supports_chunked_prefill,
                preferred_chunked_prefill_size,
                chunked_prefill_size,
                kv_cache_token_capacity,
                kv_cache_block_size,
                max_concurrent_requests,
                direct_peer_ids,
                reachable_peer_ids,
                relayed_peer_ids,
                account_hash,
                memory_contract_failure,
                swarm_v3,
            )
        )
        self._wake_event.set()

    def _record_node_heartbeat(self, node: Node, *, sample_interval: bool = True) -> None:
        """Refresh adaptive liveness after one successfully handled heartbeat."""

        now = time.monotonic()
        multiplier = self._local_health.multiplier
        with self._liveness_lock:
            detector = self._liveness_detectors.get(node.node_id)
            if detector is None:
                detector = PhiAccrualFailureDetector()
                self._liveness_detectors[node.node_id] = detector
            detector.heartbeat(
                now=now,
                local_health_multiplier=multiplier,
                sample_interval=sample_interval,
            )
            snapshot = detector.snapshot(
                now=now,
                local_health_multiplier=multiplier,
                hard_timeout_seconds=self.heartbeat_timeout,
            )
        self._apply_liveness_snapshot(node, snapshot)

    @staticmethod
    def _apply_liveness_snapshot(node: Node, snapshot: LivenessSnapshot) -> None:
        """Publish one detector snapshot for routing and observability."""

        node.liveness_state = snapshot.state
        node.liveness_phi = snapshot.phi
        node.heartbeat_age_seconds = snapshot.heartbeat_age_seconds or 0.0
        node.heartbeat_mean_interval_seconds = snapshot.mean_interval_seconds
        node.heartbeat_std_deviation_seconds = snapshot.std_deviation_seconds
        node.heartbeat_samples = snapshot.samples
        node.local_health_multiplier = snapshot.local_health_multiplier

    def checking_node_heartbeat(self) -> None:
        """Update reversible suspicion and expire only the hard heartbeat lease."""

        now = time.monotonic()
        multiplier = self._local_health.multiplier
        for node in self.node_manager.nodes:
            with self._liveness_lock:
                detector = self._liveness_detectors.get(node.node_id)
                if detector is None:
                    detector = PhiAccrualFailureDetector()
                    self._liveness_detectors[node.node_id] = detector
                    detector.heartbeat(now=now, local_health_multiplier=multiplier)
                snapshot = detector.snapshot(
                    now=now,
                    local_health_multiplier=multiplier,
                    hard_timeout_seconds=self.heartbeat_timeout,
                )

            previous_state = node.liveness_state
            self._apply_liveness_snapshot(node, snapshot)
            if snapshot.state != previous_state:
                log = logger.warning if snapshot.state != "healthy" else logger.info
                log(
                    "Node %s liveness %s -> %s (phi=%.3f age=%.3fs local_multiplier=%d)",
                    node.node_id,
                    previous_state,
                    snapshot.state,
                    snapshot.phi,
                    snapshot.heartbeat_age_seconds or 0.0,
                    snapshot.local_health_multiplier,
                )
                with self._capacity_cv:
                    self._capacity_cv.notify_all()

            if snapshot.state == "expired":
                logger.warning(
                    "Node %s hard heartbeat lease expired after %.3fs",
                    node.node_id,
                    snapshot.heartbeat_age_seconds or 0.0,
                )
                # Route leave through the event loop so fencing/rebalance is serialized.
                self.enqueue_leave(node.node_id)

    # Dynamic node management
    def join(self, node: Node) -> None:
        """Add a node to allocation and refresh plan and materialized nodes."""
        bootstrapped = self._bootstrapped_event.is_set()
        # ``bootstrapped`` is only a cache of the allocation invariant.  A
        # scheduler that lost its last complete route must never treat the next
        # worker as a lightweight dynamic join: doing so can hand a decoder-only
        # node an orphan range such as [1, L).  Keep this guard at the mutation
        # boundary as well as in ``leave`` so direct/test callers cannot observe
        # a stale event either.
        if bootstrapped and not self.has_full_pipeline():
            logger.warning(
                "Clearing stale bootstrap state before joining %s: no full pipeline remains",
                node.node_id,
            )
            self._bootstrapped_event.clear()
            bootstrapped = False
        logger.info(
            "Joining node %s (kv_ratio=%.2f, param_ratio=%.2f, manual_assignment=%s, bootstrapped=%s)",
            node.node_id,
            node.kvcache_mem_ratio,
            node.param_mem_ratio,
            node.manual_layer_assignment,
            bootstrapped,
        )
        existing = self.node_manager.get(node.node_id)
        if existing is not None:
            # ``node_join`` is a streaming RPC and can be retried when the
            # response channel closes after the scheduler has already handled
            # the request.  Replacing the registered Node here used to discard
            # its scheduler-owned layer range, KV telemetry, and reservations.
            # Treat a repeated join as an idempotent reconnect: preserve the
            # scheduler-owned serving state while refreshing worker-owned live
            # capacity and protocol capabilities. This is also required when a
            # heartbeat auto-registers a worker just before its full join after
            # a scheduler restart.
            existing.refresh_registration(node)
            logger.info(
                "Node %s is already registered; refreshed capabilities and preserving "
                "layers [%s, %s) on repeated join",
                node.node_id,
                existing.start_layer,
                existing.end_layer,
            )
        else:
            # Automatic workers can reconnect with the last assignment they received.
            # The scheduler owns that state and must allocate the fresh STANDBY node
            # from scratch; manual assignments are the only client-owned ranges.
            if not node.manual_layer_assignment:
                node.clear_serving_state()
            self.node_manager.upsert(node)
            if bootstrapped:
                if self.dynamic_pipelines_router:
                    # for dynamic pipelines router, join the node to the lightest layer
                    candidate = self.layer_allocator.dynamic_join_candidate(node)
                    try:
                        joined = self.layer_allocator.dynamic_join(node)
                    except ValueError:
                        logger.exception(
                            "Dynamic join rejected node %s; keeping it in standby",
                            node.node_id,
                        )
                        self.node_manager.standby([node.node_id])
                        node.clear_serving_state()
                        joined = False
                    if not joined:
                        logger.info(
                            "Node %s remains standby after dynamic join rejection",
                            node.node_id,
                        )
                        if candidate is not None:
                            self._pending_rebalance_node_ids.add(node.node_id)
                            logger.info(
                                "Queued one drained global rebalance for late node %s",
                                node.node_id,
                            )
                else:
                    joined = True
                if joined:
                    try:
                        self.request_router.expand_pipelines()
                    except NotImplementedError:
                        pass

        # Manual layer assignment bypasses bootstrap waiting
        if node.manual_layer_assignment:
            # Manual layer assignment: use the layers specified by the node
            if node.start_layer is None or node.end_layer is None:
                raise ValueError(
                    f"Node {node.node_id} has manual_layer_assignment=True "
                    f"but start_layer ({node.start_layer}) or end_layer ({node.end_layer}) is None"
                )
            logger.info(
                f"Manual layer assignment for node {node.node_id}: "
                f"layers [{node.start_layer}, {node.end_layer})"
            )
            # Directly allocate the specified layers without automatic assignment
            self.layer_allocator.allocate(node, node.start_layer, node.end_layer)

            # Check if manual allocations now cover the full pipeline
            if self.has_full_pipeline():
                if not self._bootstrapped_event.is_set():
                    logger.info(
                        "[Scheduler] Manual layer assignments have established a full pipeline; "
                        "marking scheduler as bootstrapped"
                    )
                    self._bootstrapped_event.set()

        # Notify waiters that node count changed
        # Snapshot at INFO after join since allocations/pipelines may have changed.
        self.emit_alloc_log_snapshot(reason=f"after join {node.node_id}")
        registered = self.node_manager.get(node.node_id)
        if registered is not None:
            # JOIN is a lifecycle retry, not a periodic heartbeat sample.  It
            # refreshes liveness without teaching the detector an artificial
            # near-zero interval when a worker retries registration quickly.
            self._record_node_heartbeat(registered, sample_interval=False)
        with self._node_count_cv:
            self._node_count_cv.notify_all()

    def leave(self, node_id: str) -> None:
        """Remove a node from the node manager.

        If using fixed pipeliens:
        - Nullify the pipeline the node is in;
        - Move all remaining nodes in the pipeline to STANDBY;

        Note: Global rebalance/reboot is handled by the event loop (`_process_leaves`) to
        ensure we don't concurrently reboot when multiple leave events arrive.
        """
        node = self.node_manager.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in nodes")
        logger.info("Leaving node %s (start=%s, end=%s)", node_id, node.start_layer, node.end_layer)
        invalidated = self._invalidate_routes_for_node(node_id)
        if invalidated:
            logger.warning(
                "Invalidated %d in-flight route(s) after node %s left: %s",
                len(invalidated),
                node_id,
                invalidated,
            )
        self.node_manager.remove(node_id)
        with self._liveness_lock:
            self._liveness_detectors.pop(node_id, None)

        if not self.node_manager.list_node_allocations(self.num_layers):
            self._reset_prefill_contract()

        # Bootstrap state means that at least one complete [0, L) route exists,
        # not merely that bootstrap succeeded at some point in the past.  Clear
        # it synchronously when the last complete route disappears so a join
        # queued immediately after this leave goes through global bootstrap.
        if self._bootstrapped_event.is_set() and not self.has_full_pipeline():
            logger.info(
                "Full pipeline coverage lost after node %s left; clearing bootstrap state",
                node_id,
            )
            self._bootstrapped_event.clear()

        # Snapshot at INFO after leave since allocations/pipelines may have changed.
        self.emit_alloc_log_snapshot(reason=f"after leave {node_id}")

        with self._node_count_cv:
            self._node_count_cv.notify_all()

    def receive_request(self, request: RequestSignal) -> None:
        """Add a request to the wait pool."""
        self._request_queue.put(request)
        self._wake_event.set()
        now = time.time()
        self._arrival_ts.append(now)
        logger.debug(
            "Received request %s (queue_size=%d)", request.request_id, self._request_queue.qsize()
        )
        # Trim old timestamps to keep arrival-rate window bounded
        horizon = self.request_arrival_horizon_sec
        while self._arrival_ts and now - self._arrival_ts[0] > horizon:
            self._arrival_ts.popleft()

    def dispatch_next_request(
        self, *, timeout: Optional[float] = None
    ) -> Optional[Tuple[str, List[str], float]]:
        """Route the next request in the wait pool; returns (request_id, path, latency).

        If `timeout` is provided, blocks up to `timeout` seconds waiting for a request.
        """
        # Don't dequeue requests until routing is actually possible.
        # Admission, dequeue, route selection and reservations share the same
        # transaction as a topology rebalance.  This prevents a request from
        # entering the old allocation after the reconfiguration drain passed.
        with self._inflight_routes_lock:
            if not self.serving_ready():
                return None
            try:
                req = (
                    self._request_queue.get(timeout=timeout)
                    if timeout is not None
                    else self._request_queue.get_nowait()
                )
            except queue.Empty:
                return None
            request_key = str(req.request_id)
            if req.cancelled:
                logger.debug("Discarded cancelled request %s before dispatch", request_key)
                req.routing_table = []
                return req.request_id, [], float("inf")
            if req.required_context_tokens > self.max_supported_context_tokens():
                path, latency = [], float("inf")
            else:
                if request_key in self._inflight_routes:
                    logger.warning(
                        "Request %s was already reserved; replacing its route", req.request_id
                    )
                    self._release_request_locked(request_key)
                path, latency = [], float("inf")
                # A heartbeat or leave can change capacity between the router's
                # snapshot and NodeManager's guarded mutation. Retry once from a
                # fresh snapshot instead of killing the dispatch loop.
                for attempt in range(2):
                    candidate, candidate_latency = self.request_router.find_optimal_path(
                        last_refit_time=self.last_refit_time,
                        required_context_tokens=req.required_context_tokens,
                    )
                    if not candidate:
                        break
                    reserved: List[str] = []
                    try:
                        for node_id in candidate:
                            self.node_manager.add_request(node_id, req.required_context_tokens)
                            reserved.append(node_id)
                    except (ValueError, KeyError) as exc:
                        for node_id in reserved:
                            self.node_manager.remove_request(node_id, req.required_context_tokens)
                        logger.info(
                            "Route reservation changed during dispatch for request %s "
                            "(attempt %d): %s",
                            req.request_id,
                            attempt + 1,
                            exc,
                        )
                        continue
                    path, latency = list(candidate), float(candidate_latency)
                    self._inflight_routes[request_key] = (
                        path,
                        int(req.required_context_tokens),
                    )
                    break
            req.routing_table = path
        logger.debug(
            "Dispatched request %s via path %s (est_lat=%.2fms)", req.request_id, path, latency
        )
        return req.request_id, path, latency

    def _release_request_locked(self, request_id: str) -> bool:
        reservation = self._inflight_routes.pop(request_id, None)
        if reservation is None:
            return False
        path, required_context_tokens = reservation
        for node_id in path:
            self.node_manager.remove_request(node_id, required_context_tokens)
        return True

    def _invalidate_routes_for_node(self, node_id: str) -> List[str]:
        """Release every route containing a worker before removing that worker.

        The HTTP handler monitors this registry while blocked on the Lattica
        response. Removing a route therefore fences the failed pipeline and
        gives the handler a deterministic signal to cancel its RPC stream.
        """
        with self._inflight_routes_lock:
            affected = [
                request_id
                for request_id, (path, _) in self._inflight_routes.items()
                if node_id in path
            ]
            for request_id in affected:
                self._release_request_locked(request_id)
        if affected:
            with self._capacity_cv:
                self._capacity_cv.notify_all()
        return affected

    def is_request_route_active(self, request_id: str) -> bool:
        """Return whether a request still owns its originally dispatched route."""
        with self._inflight_routes_lock:
            return str(request_id) in self._inflight_routes

    def release_request(self, request_id: str) -> bool:
        """Release a dispatched route exactly once when its HTTP request ends."""
        request_key = str(request_id)
        with self._inflight_routes_lock:
            released = self._release_request_locked(request_key)
        if released:
            logger.debug("Released scheduler reservation for request %s", request_key)
            with self._capacity_cv:
                self._capacity_cv.notify_all()
        return released

    def cancel_request_signal(self, request: RequestSignal) -> bool:
        """Atomically cancel one queued signal and release it if dispatch won the race.

        A worker loss can make ``get_routing_table`` time out while the signal is
        still in the scheduler FIFO. Merely returning HTTP 503 leaves that signal
        eligible for dispatch after the cluster recovers, creating a route with no
        client and a permanent KV reservation. The signal flag and route release
        share the dispatch lock so every timing resolves to either discard-before-
        reserve or reserve-then-release.
        """
        request_key = str(request.request_id)
        with self._inflight_routes_lock:
            request.cancelled = True
            released = self._release_request_locked(request_key)
        if released:
            with self._capacity_cv:
                self._capacity_cv.notify_all()
        return released

    def has_routing_capacity(self) -> bool:
        """Return whether the current router can admit at least one request."""
        if not self.serving_ready():
            return False
        if self.routing_strategy == "rr":
            _, _, remaining = self.node_manager.report_pipeline_capacity(ready_only=True)
            return remaining > 0
        return any(
            node.is_routable and not node.is_overloaded for node in self.node_manager.active_nodes
        )

    def wait_for_routing_capacity(self, timeout: float) -> bool:
        """Wait until capacity is available, waking immediately after a release."""
        with self._capacity_cv:
            return self._capacity_cv.wait_for(self.has_routing_capacity, timeout=max(0.0, timeout))

    def emit_alloc_log_snapshot(self, *, reason: Optional[str] = None) -> str:
        """Update `self.alloc_log_snapshot` and emit it.

        - Periodic/heartbeat snapshots (no reason) are logged at DEBUG.
        - Mutating events (join/leave/bootstrap) provide a reason and are logged at INFO.
        """
        try:
            snapshot = self.request_router.scheduler_format_snapshot()
        except Exception as exc:
            snapshot = f"(failed to build allocation snapshot: {exc})"
            logger.warning("Allocation snapshot build error: %s", exc)

        self.alloc_log_snapshot = snapshot

        if reason:
            logger.info("Allocation snapshot (%s)\n%s", reason, snapshot)
        else:
            logger.debug("Allocation snapshot\n%s", snapshot)
        return snapshot

    def run(self, *, poll_interval: float = 0.05, allocation_log_interval: float = 5.0) -> None:
        """Run the scheduler concurrently until `stop()` is called.

        Starts background threads for event processing (joins/leaves/updates/heartbeats)
        and request dispatching. At startup, waits until at least
        `min_nodes_bootstrapping` nodes are present, then runs `bootstrap()`.
        """
        logger.debug("Running scheduler")
        self._stop_event.clear()

        # Start event thread first so joins can be processed while we wait to bootstrap
        self._event_thread = threading.Thread(
            target=self._event_loop, args=(poll_interval,), name="SchedulerEventLoop", daemon=True
        )
        self._event_thread.start()

        # Bootstrap gating
        if not self._wait_for_bootstrap(poll_interval):
            return

        # Start dispatcher only after successful bootstrap
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            args=(poll_interval,),
            name="SchedulerDispatcher",
            daemon=True,
        )
        self._dispatch_thread.start()

        # Start periodic allocation logger thread
        def _alloc_log_loop() -> None:
            """Periodically log current layer allocations."""
            while not self._stop_event.is_set():
                try:
                    self.emit_alloc_log_snapshot()
                except Exception as exc:
                    logger.warning(f"Allocation logger error: {exc}")

                # After bootstrap, periodically check if *all* nodes report active and log once.
                if self._bootstrapped_event.is_set():
                    nodes = self.node_manager.nodes
                    if nodes:
                        all_active = all(n.is_active for n in nodes)
                        if all_active and not self._all_nodes_active_logged:
                            logger.info("All %d nodes are active", len(nodes))
                            # Emit snapshot at INFO once when all nodes become active.
                            self.emit_alloc_log_snapshot(reason="All nodes are active")
                            self._all_nodes_active_logged = True
                        elif not all_active:
                            self._all_nodes_active_logged = False
                time.sleep(max(1.0, allocation_log_interval))

        self._alloc_log_thread = threading.Thread(
            target=_alloc_log_loop, name="SchedulerAllocLogger", daemon=True
        )
        self._alloc_log_thread.start()

        # Block until stop is requested
        try:
            while not self._stop_event.is_set():
                time.sleep(max(0.5, poll_interval))
        finally:
            if self._event_thread is not None:
                self._event_thread.join(timeout=2.0)
            if self._dispatch_thread is not None:
                self._dispatch_thread.join(timeout=2.0)
            if self._alloc_log_thread is not None:
                self._alloc_log_thread.join(timeout=2.0)

    # === Modularized worker loops ===
    def _event_loop(self, poll_interval: float) -> None:
        """Process joins/leaves/updates and perform heartbeat checks."""
        last_hb_check: float | None = None
        last_v3_shadow_check = 0.0
        while not self._stop_event.is_set():
            self._process_node_updates()
            self._process_pending_context_replan()
            self._process_joins()
            self._process_leaves()
            self._process_pending_rebalance()
            now = time.time()
            if self.swarm_v3_shadow is not None and now - last_v3_shadow_check >= 2.0:
                selected_context = int(
                    getattr(self.layer_allocator, "selected_context_tokens", 0) or 0
                )
                try:
                    self.swarm_v3_shadow_snapshot = self.swarm_v3_shadow.observe(
                        list(self.node_manager.nodes),
                        model_num_layers=self.num_layers,
                        planning_context_tokens=max(
                            1,
                            selected_context
                            or int(
                                getattr(
                                    self.layer_allocator,
                                    "planning_context_tokens",
                                    16_384,
                                )
                            ),
                        ),
                        epoch=self.allocation_epoch,
                    )
                except Exception:
                    logger.warning("Protocol-v3 shadow comparison failed", exc_info=True)
                last_v3_shadow_check = now
            monotonic_now = time.monotonic()
            heartbeat_check_interval = max(0.5, poll_interval)
            if last_hb_check is None or monotonic_now - last_hb_check >= heartbeat_check_interval:
                if last_hb_check is not None:
                    self._local_health.observe_loop_delay(
                        monotonic_now - last_hb_check,
                        heartbeat_check_interval,
                    )
                self.checking_node_heartbeat()
                last_hb_check = monotonic_now
            self._wake_event.wait(timeout=poll_interval)
            self._wake_event.clear()

    def _dispatch_loop(self, poll_interval: float) -> None:
        """Continuously dispatch incoming requests while running."""
        while not self._stop_event.is_set():
            if not self.request_router.routing_ready():
                time.sleep(max(0.0, poll_interval))
                continue

            # Block (briefly) waiting for the next request, then dispatch it.
            _ = self.dispatch_next_request(timeout=poll_interval)

    def _wait_for_bootstrap(self, poll_interval: float) -> bool:
        """Wait until enough nodes then run bootstrap. Returns False if stopped."""
        logger.debug("Waiting for bootstrap")
        while not self._stop_event.is_set() and not self._bootstrapped_event.is_set():
            with self._node_count_cv:
                self._node_count_cv.wait(timeout=max(0.5, poll_interval))
        return not self._stop_event.is_set()

    def _process_node_updates(self) -> None:
        """Apply pending node stats updates from the queue."""
        while True:
            try:
                (
                    node_id,
                    cur,
                    lat,
                    rtts,
                    is_active,
                    last_refit_time,
                    supports_chunked_prefill,
                    preferred_chunked_prefill_size,
                    chunked_prefill_size,
                    kv_cache_token_capacity,
                    kv_cache_block_size,
                    max_concurrent_requests,
                    direct_peer_ids,
                    reachable_peer_ids,
                    relayed_peer_ids,
                    account_hash,
                    memory_contract_failure,
                    swarm_v3,
                ) = self._pending_node_updates.get_nowait()
            except queue.Empty:
                break
            node = self.node_manager.get(node_id)
            if node is None:
                logger.warning(f"Node {node_id} not found in node manager, ignore the update")
                continue
            self.update_node_info(
                node,
                current_requests=cur,
                layer_latency_ms=lat,
                new_rtt_to_nodes=rtts,
                is_active=is_active,
                last_refit_time=last_refit_time,
                supports_chunked_prefill=supports_chunked_prefill,
                preferred_chunked_prefill_size=preferred_chunked_prefill_size,
                chunked_prefill_size=chunked_prefill_size,
                kv_cache_token_capacity=kv_cache_token_capacity,
                kv_cache_block_size=kv_cache_block_size,
                max_concurrent_requests=max_concurrent_requests,
                direct_peer_ids=direct_peer_ids,
                reachable_peer_ids=reachable_peer_ids,
                relayed_peer_ids=relayed_peer_ids,
                account_hash=account_hash,
                memory_contract_failure=memory_contract_failure,
                swarm_v3=swarm_v3,
            )

        # Manual allocations can complete before their executors finish loading.
        # Register the fixed RR route once every stage reports ready; otherwise
        # the scheduler is bootstrapped but keeps returning zero route capacity.
        if (
            self.routing_strategy == "rr"
            and self._bootstrapped_event.is_set()
            and not self.request_router.routing_ready()
            and self.node_manager.has_full_pipeline(self.num_layers, ready_only=True)
        ):
            try:
                self.request_router.bootstrap()
                if self.request_router.routing_ready():
                    logger.info(
                        "[Scheduler] Routing pipelines registered after nodes became active"
                    )
            except Exception:
                logger.warning(
                    "Failed to register routing pipelines after node update", exc_info=True
                )

    def _process_joins(self) -> None:
        """Handle pending join events, honoring bootstrap state for assignment."""
        joined_any = False
        had_manual_assignment = False
        while True:
            try:
                node = self._pending_joins.get_nowait()
            except queue.Empty:
                break
            # During bootstrap (no full pipeline yet), only declare nodes; no dynamic assignment.
            # After bootstrap, allow dynamic light-weight joins.
            # Exception: manual layer assignments are processed immediately regardless of bootstrap state.
            self.join(node)
            joined_any = True
            if node.manual_layer_assignment:
                had_manual_assignment = True
        if joined_any:
            self._wake_pending_context_replan()

        # If we are not bootstrapped (e.g., after a leave-triggered rebalance) and
        # new nodes just joined, attempt a greedy bootstrap immediately when we have
        # enough nodes. If it doesn't produce a full pipeline, we'll try again on
        # subsequent joins.
        # Skip bootstrap if manual assignments were used (they handle bootstrapping internally).
        if (
            joined_any
            and self._pending_context_replan is None
            and not self._bootstrapped_event.is_set()
            and not had_manual_assignment
        ):
            if self.node_manager.num_standby_nodes >= self.min_nodes_bootstrapping:
                try:
                    ok = self.bootstrap()
                    if not ok:
                        logger.debug(
                            "Bootstrap attempt after join did not produce a full pipeline; will retry on future joins"
                        )
                except Exception as exc:
                    logger.debug(
                        f"Bootstrap attempt after join failed: {exc}; will retry on future joins"
                    )
            else:
                logger.debug(
                    "Deferring bootstrap: have %d nodes; need >= %d",
                    self.node_manager.num_standby_nodes,
                    self.min_nodes_bootstrapping,
                )

        self._process_pending_rebalance()

    def _queue_bootstrap_standby_rebalances(self) -> None:
        """Reconsider useful workers skipped by the initial DP solution.

        The upstream DP objective minimizes stage count before it has runtime
        measurements.  A frontend node capable of hosting the whole model can
        therefore close a one-node pipeline while a much faster decoder-only
        worker, which may even have joined first, is left in STANDBY.  The
        allocator's lightweight dynamic join cannot activate that worker when
        its proposed overlap is not already an exact-boundary route.

        Treat that leftover exactly like a route-dead late join: keep the live
        bootstrap intact, then let the existing drained, RTT-aware planner
        decide once whether the worker improves the route.  Capacity-invalid
        workers are not queued, and telemetry updates never call this method,
        so host-memory fluctuations cannot create reload oscillations.
        """

        if not self.dynamic_pipelines_router:
            return
        allocations = self.list_node_allocations()
        queued: List[str] = []
        for node in self.node_manager.standby_nodes:
            if node.manual_layer_assignment:
                continue
            candidate = self.layer_allocator.dynamic_join_candidate(node)
            if candidate is None:
                continue
            proposed = [*allocations, (node.node_id, *candidate)]
            participants = self.node_manager.full_pipeline_segment_ids(proposed, self.num_layers)
            if node.node_id in participants:
                # ``allocate_standby_nodes`` should already have admitted an
                # exact-route shard. Avoid scheduling a reload if a custom
                # allocator left one behind for another reason.
                continue
            self._pending_rebalance_node_ids.add(node.node_id)
            queued.append(node.node_id)
        if queued:
            logger.info(
                "Queued one drained global rebalance for bootstrap-skipped node(s): %s",
                sorted(queued),
            )

    def _plan_global_rebalance(self) -> Dict[str, Tuple[int, int]]:
        """Trim proposed late overlaps into exact boundaries on detached copies.

        Parallax already contains a layer-level turning-point optimizer for its
        warm-up phase.  Reuse it here after adding the late shard hypothetically:
        if the new worker improves the path, the optimizer trims the old tail
        and new prefix into a contiguous route; otherwise the live allocation
        remains untouched.
        """

        planned_nodes = [copy.deepcopy(node) for node in self.node_manager.nodes]
        for node in planned_nodes:
            node.clear_serving_state()
            node.is_active = False
        planned_manager = NodeManager(initial_nodes=planned_nodes)
        planned_by_id = {node.node_id: node for node in planned_nodes}
        allocator_type = type(self.layer_allocator)
        planned_allocator = allocator_type(
            model_info=self.model_info,
            node_management=planned_manager,
            dynamic_pipelines_router=self.dynamic_pipelines_router,
            rebalance_threshold=self.layer_allocator.rebalance_threshold,
            water_filling_max_iterations=self.layer_allocator.water_filling_max_iterations,
            trim_layers_on_turning_points=self.layer_allocator.trim_layers_on_turning_points,
            planning_context_tokens=self.layer_allocator.planning_context_tokens,
            preferred_context_tokens=self.layer_allocator.preferred_context_tokens,
            require_exact_weight_metadata=self.layer_allocator.require_exact_weight_metadata,
            context_ceiling_tokens=self.layer_allocator.context_ceiling_tokens,
            runtime_context_downgrade_used=(self.layer_allocator.runtime_context_downgrade_used),
        )
        for node_id, start_layer, end_layer in self.list_node_allocations():
            planned_allocator.allocate(planned_by_id[node_id], start_layer, end_layer)
        for node_id in sorted(self._pending_rebalance_node_ids):
            node = planned_by_id.get(node_id)
            if node is None:
                continue
            candidate = planned_allocator.dynamic_join_candidate(node)
            if candidate is not None:
                planned_allocator.allocate(node, *candidate)
        planned_allocator.adjust_for_turning_points(self.num_layers)
        allocations = planned_manager.list_node_allocations(self.num_layers)
        participants = planned_manager.full_pipeline_segment_ids(allocations, self.num_layers)
        return {
            node_id: (start_layer, end_layer)
            for node_id, start_layer, end_layer in allocations
            if node_id in participants
        }

    def _apply_rebalance_plan(self, plan: Dict[str, Tuple[int, int]]) -> bool:
        """Publish a precomputed allocation, restoring the old shape on error."""

        previous = {
            node_id: (start_layer, end_layer)
            for node_id, start_layer, end_layer in self.list_node_allocations()
        }

        def restore_previous() -> None:
            for node in list(self.node_manager.active_nodes):
                try:
                    self.layer_allocator.deallocate(node)
                except Exception:
                    logger.exception(
                        "Failed to clear partial rebalance allocation on %s", node.node_id
                    )
            for node_id, (start_layer, end_layer) in previous.items():
                node = self.node_manager.get(node_id)
                if node is None:
                    continue
                try:
                    self.layer_allocator.allocate(node, start_layer, end_layer)
                except Exception:
                    logger.exception("Failed to restore previous allocation on %s", node_id)
            if self.has_full_pipeline():
                self._bootstrapped_event.set()

        self._bootstrapped_event.clear()
        self._reset_prefill_contract()
        try:
            for node in self.node_manager.nodes:
                node.is_active = False
            for node in list(self.node_manager.active_nodes):
                self.layer_allocator.deallocate(node)
            for node_id, (start_layer, end_layer) in plan.items():
                node = self.node_manager.get(node_id)
                if node is None:
                    raise ValueError(f"Planned node {node_id} disappeared before commit")
                self.layer_allocator.allocate(node, start_layer, end_layer)
            if not self.has_full_pipeline():
                raise RuntimeError("Committed rebalance has no complete pipeline")
            self.request_router.bootstrap()
        except Exception:
            logger.exception("Drained global rebalance failed; restoring previous allocation")
            restore_previous()
            return False

        lease_started_at = time.time()
        for node in self.node_manager.active_nodes:
            node.last_heartbeat = lease_started_at
            self._record_node_heartbeat(node, sample_interval=False)
        self.allocation_epoch = self.epoch_allocator.next_epoch()
        self._bootstrapped_event.set()
        self.emit_alloc_log_snapshot(reason="after drained global rebalance")
        return True

    def _process_pending_rebalance(self, *, force: bool = False) -> bool:
        """Repartition once after a useful late join, only at an idle boundary."""

        if not self._pending_rebalance_node_ids or not self.dynamic_pipelines_router:
            return False
        now = time.monotonic()
        if not force and now < self._next_rebalance_attempt_at:
            return False
        if any(node.manual_layer_assignment for node in self.node_manager.nodes):
            logger.warning("Skipping automatic rebalance for mixed/manual assignments")
            self._pending_rebalance_node_ids.clear()
            return False

        with self._inflight_routes_lock:
            if (
                self._inflight_routes
                or any(
                    node.routing_load > 0 or node.current_requests > 0
                    for node in self.node_manager.active_nodes
                )
                or (self.external_routes_active is not None and self.external_routes_active())
            ):
                return False
            active_nodes = self.node_manager.active_nodes
            pending_nodes = [
                node
                for node_id in self._pending_rebalance_node_ids
                if (node := self.node_manager.get(node_id)) is not None
            ]
            if any(
                not any(isfinite(node.get_rtt_to(active)) for active in active_nodes)
                for node in pending_nodes
            ):
                # A join is acknowledged before its heartbeat RTT probes settle.
                # Planning with infinite edges would permanently reject an
                # otherwise useful worker. Keep it standby and retry at a
                # bounded cadence instead of spinning the event loop.
                self._next_rebalance_attempt_at = now + 1.0
                logger.debug(
                    "Deferring late-node rebalance until RTT telemetry is available: %s",
                    sorted(self._pending_rebalance_node_ids),
                )
                return False
            self._admission_paused = True
            try:
                plan = self._plan_global_rebalance()
                useful_pending = self._pending_rebalance_node_ids.intersection(plan)
                if not plan or not useful_pending:
                    logger.info(
                        "Keeping late node(s) in standby because the global plan "
                        "does not place them on a complete route: %s",
                        sorted(self._pending_rebalance_node_ids),
                    )
                    self._pending_rebalance_node_ids.clear()
                    self._next_rebalance_attempt_at = 0.0
                    return False
                if not self._apply_rebalance_plan(plan):
                    return False
                logger.info(
                    "Drained global rebalance admitted late node(s): %s",
                    sorted(useful_pending),
                )
                self._pending_rebalance_node_ids.clear()
                self._next_rebalance_attempt_at = 0.0
                return True
            finally:
                self._admission_paused = False

    def _process_leaves(self) -> None:
        """Handle pending leave events safely.

        Important: This is the only place we trigger global rebalance/reboot so leave events
        are serialized by the single event-loop thread.
        """
        removed_any = False
        while True:
            try:
                node_id = self._pending_leaves.get_nowait()
            except queue.Empty:
                break
            try:
                self.leave(node_id)
                removed_any = True
            except Exception as exc:
                logger.warning(f"Leave failed for {node_id}: {exc}")

        if removed_any:
            self._rebalance_after_leave_pending = True

        # After draining all leaves, decide whether to do a single global
        # rebalance. Keep the intent pending while an unrelated v3 request owns
        # a route; consuming the leave queue must not lose this transition.
        if not self._rebalance_after_leave_pending:
            return
        if self.external_routes_active is not None and self.external_routes_active():
            return

        if not self.layer_allocator.should_global_rebalance():
            self._rebalance_after_leave_pending = False
            return

        nodes = self.node_manager.nodes
        logger.warning("Global rebalance triggered due to node leave")

        # Count manual vs automatic nodes
        manual_count = sum(1 for n in nodes if n.manual_layer_assignment)
        total_count = len(nodes)
        logger.debug(f"Node count: {manual_count} manual, {total_count - manual_count} automatic")
        if total_count == 0:
            logger.debug("No nodes left after leave(s); skipping global rebalance")
            self._rebalance_after_leave_pending = False
            return
        if manual_count == total_count:
            logger.debug("All nodes are manual assignment, skipping global rebalance")
            self._rebalance_after_leave_pending = False
            return
        if manual_count > 0:
            logger.error(
                f"Mixed assignment detected ({manual_count} manual, {total_count - manual_count} automatic); skipping rebalance"
            )
            self._rebalance_after_leave_pending = False
            return

        # Move active nodes to standby and re-bootstrap (reboot) once.
        self.node_manager.standby([n.node_id for n in self.node_manager.active_nodes])
        assert self.node_manager.num_standby_nodes == self.node_manager.num_nodes, (
            "All active nodes should be moved to standby"
        )
        assert self.node_manager.num_active_nodes == 0, "No active nodes before re-bootstrap"
        logger.warning("Re-bootstrapping for global rebalance")
        try:
            self.bootstrap(reboot=True)
        finally:
            self._rebalance_after_leave_pending = False
            # Ensure snapshot reflects post-rebalance state even if bootstrap fails.
            self.emit_alloc_log_snapshot(reason="after global rebalance")

    def stop(self) -> None:
        """Signal background threads to stop and wake any waiters."""
        self._stop_event.set()
        self._wake_event.set()
        with self._node_count_cv:
            self._node_count_cv.notify_all()

    def need_more_nodes(self):
        return (
            not self._bootstrapped_event.is_set()
            and self.node_manager.num_standby_nodes >= self.min_nodes_bootstrapping
        )
