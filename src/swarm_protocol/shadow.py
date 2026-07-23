"""Read-only v3 route comparison beside the qualified Parallax scheduler."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from swarm_protocol.contracts import (
    ModelMemberAdvertisement,
    RecoveryLevel,
    RequestContract,
)
from swarm_protocol.registry import ModelRegistryBundle, TrustedModelRegistry
from swarm_protocol.routing import ExactRoutePlanner, NoFeasibleRoute

logger = logging.getLogger(__name__)


class SchedulerProtocolV3Shadow:
    """Validate worker reports independently and compare v3 routing without serving traffic."""

    def __init__(self, registry: TrustedModelRegistry) -> None:
        self.registry = registry
        self.planner = ExactRoutePlanner()
        self._lock = threading.RLock()
        self._bundles: dict[str, ModelRegistryBundle] = {}
        self._pending: set[str] = set()
        self._bundle_errors: dict[str, dict[str, str]] = {}
        self._bundle_retry_after: dict[str, float] = {}
        self._latest: dict[str, object] = {"mode": "shadow", "state": "waiting_workers"}
        self._last_log_key = ""

    @classmethod
    def from_environment(cls) -> "SchedulerProtocolV3Shadow | None":
        mode = os.environ.get("FABI_SWARM_V3_MODE", "off").strip().lower()
        if mode in {"", "off", "disabled"}:
            return None
        if mode != "shadow":
            raise ValueError("FABI_SWARM_V3_MODE currently supports only 'off' or 'shadow'")
        metadata_url = os.environ.get("FABI_MODEL_REGISTRY_METADATA_URL")
        targets_url = os.environ.get("FABI_MODEL_REGISTRY_TARGETS_URL")
        root_path = os.environ.get("FABI_MODEL_REGISTRY_ROOT")
        if not metadata_url or not targets_url or not root_path:
            raise ValueError("scheduler shadow mode requires registry URLs and a pinned root path")
        state_dir = Path(
            os.environ.get(
                "FABI_SWARM_V3_SCHEDULER_STATE_DIR",
                str(Path.home() / ".fabi" / "swarm-v3" / "scheduler-registry"),
            )
        )
        return cls(
            TrustedModelRegistry(
                state_dir,
                metadata_base_url=metadata_url,
                target_base_url=targets_url,
                bootstrap_root=Path(root_path).read_bytes(),
            )
        )

    @property
    def latest(self) -> dict[str, object]:
        with self._lock:
            return dict(self._latest)

    def observe(
        self,
        nodes: list[Any],
        *,
        model_num_layers: int,
        planning_context_tokens: int,
        epoch: int,
    ) -> dict[str, object]:
        """Perform a CPU-only comparison and schedule registry I/O off-thread."""

        now_ms = time.time_ns() // 1_000_000
        advertisements = []
        rejected_workers: dict[str, str] = {}
        for node in nodes:
            report = getattr(node, "swarm_v3", None)
            if not isinstance(report, dict) or report.get("state") not in {"ready", "warming"}:
                if isinstance(report, dict):
                    rejected_workers[str(node.node_id)] = str(report.get("state", "invalid"))
                continue
            raw_advertisement = report.get("advertisement")
            try:
                advertisement = ModelMemberAdvertisement.model_validate(raw_advertisement)
            except Exception as exc:  # noqa: BLE001 - untrusted RPC boundary
                rejected_workers[str(node.node_id)] = f"invalid:{type(exc).__name__}"
                continue
            if advertisement.offer.worker_id != str(node.node_id):
                rejected_workers[str(node.node_id)] = "worker_identity_mismatch"
                continue
            advertisements.append(advertisement)

        if not advertisements:
            return self._store(
                {
                    "mode": "shadow",
                    "state": "waiting_workers",
                    "accepted_workers": 0,
                    "rejected_workers": rejected_workers,
                }
            )

        swarm_counts: dict[str, int] = {}
        for advertisement in advertisements:
            swarm_id = advertisement.lease.model_swarm_id
            swarm_counts[swarm_id] = swarm_counts.get(swarm_id, 0) + 1
        model_swarm_id = min(
            swarm_counts,
            key=lambda swarm_id: (-swarm_counts[swarm_id], swarm_id),
        )
        advertisements = [
            advertisement
            for advertisement in advertisements
            if advertisement.lease.model_swarm_id == model_swarm_id
        ]

        with self._lock:
            bundle = self._bundles.get(model_swarm_id)
            bundle_error = self._bundle_errors.get(model_swarm_id)
            retry_allowed = time.monotonic() >= self._bundle_retry_after.get(model_swarm_id, 0)
            if bundle is None and model_swarm_id not in self._pending and retry_allowed:
                self._pending.add(model_swarm_id)
                threading.Thread(
                    target=self._fetch_bundle,
                    args=(model_swarm_id,),
                    name="SwarmV3ShadowRegistry",
                    daemon=True,
                ).start()
        if bundle is None:
            return self._store(
                {
                    "mode": "shadow",
                    "state": "registry_rejected" if bundle_error else "verifying_registry",
                    "model_swarm_id": model_swarm_id,
                    "accepted_workers": len(advertisements),
                    "registry_error": bundle_error,
                    "rejected_workers": rejected_workers,
                }
            )

        prompt_tokens = max(1, planning_context_tokens - min(4096, planning_context_tokens - 1))
        output_tokens = planning_context_tokens - prompt_tokens
        request = RequestContract(
            request_id=f"shadow-{now_ms}",
            model_swarm_id=model_swarm_id,
            prompt_tokens=prompt_tokens,
            reserved_output_tokens=output_tokens,
            recovery_level=RecoveryLevel.RESTARTABLE,
        )
        offers = tuple(advertisement.offer for advertisement in advertisements)
        leases = tuple(advertisement.lease for advertisement in advertisements)
        links = tuple(
            link for advertisement in advertisements for link in advertisement.outgoing_links
        )
        blockers = []
        if any(
            lease.measured_prefill_tokens_per_second is None
            or lease.measured_decode_tokens_per_second is None
            for lease in leases
        ):
            blockers.append("missing_executor_throughput")
        if len(leases) > 1 and not links:
            blockers.append("missing_link_goodput")
        try:
            planned = self.planner.plan(
                manifest=bundle.manifest,
                request=request,
                offers=offers,
                leases=leases,
                links=links,
                snapshot_time_ms=now_ms,
                coordinator_id="qualified-scheduler-shadow",
                reservation_deadline_ms=now_ms + 5_000,
                plan_expires_at_ms=now_ms + 10_000,
                epoch=epoch,
            )
        except NoFeasibleRoute as exc:
            return self._store(
                {
                    "mode": "shadow",
                    "state": "no_feasible_route",
                    "model_swarm_id": model_swarm_id,
                    "required_context_tokens": planning_context_tokens,
                    "accepted_workers": len(advertisements),
                    "blockers": blockers or [str(exc)],
                    "rejected_workers": rejected_workers,
                }
            )

        v3_route = tuple(stage.worker_id for stage in planned.plan.stages)
        legacy_routes = self._legacy_complete_routes(nodes, model_num_layers)
        agrees = v3_route in legacy_routes
        return self._store(
            {
                "mode": "shadow",
                "state": "agreement" if agrees else "divergence",
                "model_swarm_id": model_swarm_id,
                "required_context_tokens": planning_context_tokens,
                "v3_route": v3_route,
                "legacy_routes": legacy_routes,
                "projected_ttft_ms": planned.estimate.ttft_ms,
                "projected_inter_token_ms": planned.estimate.inter_token_ms,
                "rejected_workers": rejected_workers,
            }
        )

    def _fetch_bundle(self, model_swarm_id: str) -> None:
        try:
            bundle = self.registry.fetch(model_swarm_id)
        except Exception as exc:  # noqa: BLE001 - converted to shadow status
            with self._lock:
                self._bundle_errors[model_swarm_id] = {
                    "code": type(exc).__name__,
                    "detail": str(exc)[:256],
                }
                self._bundle_retry_after[model_swarm_id] = time.monotonic() + 30.0
                self._pending.discard(model_swarm_id)
            return
        with self._lock:
            self._bundles[model_swarm_id] = bundle
            self._bundle_errors.pop(model_swarm_id, None)
            self._bundle_retry_after.pop(model_swarm_id, None)
            self._pending.discard(model_swarm_id)

    @staticmethod
    def _legacy_complete_routes(nodes: list[Any], num_layers: int) -> tuple[tuple[str, ...], ...]:
        by_start: dict[int, list[Any]] = {}
        for node in nodes:
            start = getattr(node, "start_layer", None)
            end = getattr(node, "end_layer", None)
            if not getattr(node, "is_active", False) or start is None or end is None:
                continue
            by_start.setdefault(int(start), []).append(node)

        routes: list[tuple[str, ...]] = []

        def walk(position: int, path: tuple[str, ...]) -> None:
            if position == num_layers:
                routes.append(path)
                return
            for node in sorted(by_start.get(position, []), key=lambda item: str(item.node_id)):
                end = int(node.end_layer)
                if end > position and str(node.node_id) not in path:
                    walk(end, (*path, str(node.node_id)))

        walk(0, ())
        return tuple(sorted(routes))

    def _store(self, value: dict[str, object]) -> dict[str, object]:
        log_key = repr(value)
        with self._lock:
            self._latest = value
            if log_key != self._last_log_key:
                logger.info("Protocol-v3 shadow comparison: %s", value)
                self._last_log_key = log_key
            return dict(value)
