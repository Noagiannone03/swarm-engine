"""Admission control for Fabi's ``contribute while you consume`` contract.

This is deliberately not a credit ledger.  The scheduler derives a short-lived
capability from authenticated live membership on every admission: an account
may start inference only while it owns at least one initialized worker.  In
active v3 mode, that means both a verified READY lease and a live DHT offer;
scheduler-allocated workers are never a product fallback.

Only a SHA-256 account identifier is retained on ``Node`` objects.  The account
credential itself is accepted from the encrypted worker RPC and the HTTPS
OpenAI request, validated, hashed, and then discarded.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from backend.server.route_permits import AuthorizedContributionPermit, RoutePermitConflict
from fabi_network.capability import RouteRecoveryPolicy
from parallax_utils.logging_config import get_logger
from scheduling.node import node_is_routable
from swarm_protocol.contracts import ModelMemberAdvertisement, SpanState

logger = get_logger(__name__)

_ACCOUNT_CREDENTIAL = re.compile(r"^[0-9a-fA-F]{64}$")
_ENABLED = {"1", "true", "yes", "on", "strict"}
_MAX_ROUTE_PERMIT_TTL_MS = 5 * 60 * 1_000
_PUBLIC_RECOVERY_POLICIES = frozenset(
    {
        RouteRecoveryPolicy.BEST_EFFORT,
        RouteRecoveryPolicy.REPLAN_COLD,
    }
)


class ContributionRoutePermitLedger(Protocol):
    def get_active(self, permit_id: str) -> AuthorizedContributionPermit: ...

    def active_count(self, account_id: str) -> int: ...

    def find_active_by_idempotency(
        self,
        *,
        account_id: str,
        idempotency_key: str,
        coordinator_endpoint_id: str,
    ) -> AuthorizedContributionPermit | None: ...

    def issue(
        self,
        *,
        account_id: str,
        request_id: str,
        idempotency_key: str,
        coordinator_endpoint_id: str,
        model_swarm_id: str,
        max_context_tokens: int,
        recovery_policies: frozenset[RouteRecoveryPolicy],
        ttl_ms: int,
        max_active_per_account: int,
        permit_id: str | None = None,
    ) -> AuthorizedContributionPermit: ...

    def keepalive_owned(
        self,
        permit_id: str,
        account_id: str,
        *,
        ttl_ms: int,
        idempotency_key: str,
    ) -> AuthorizedContributionPermit: ...


class ContributionPermitDenied(PermissionError):
    def __init__(self, status: "ContributionStatus") -> None:
        super().__init__(f"contribution permit denied: {status.reason}")
        self.status = status


def account_hash(credential: object) -> Optional[str]:
    """Validate and hash the 32-byte hexadecimal account credential."""

    if not isinstance(credential, str):
        return None
    normalized = credential.strip()
    if not _ACCOUNT_CREDENTIAL.fullmatch(normalized):
        return None
    return hashlib.sha256(normalized.lower().encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ContributionStatus:
    allowed: bool
    reason: str
    eligible_workers: int = 0
    active_requests: int = 0
    max_concurrent_requests: int = 0
    account_id: Optional[str] = None
    worker_state: Optional[str] = None
    worker_usable_memory_bytes: Optional[int] = None
    worker_storage_missing_bytes: Optional[int] = None

    def public_payload(self, *, enabled: bool) -> dict:
        payload = {
            "enabled": enabled,
            "allowed": self.allowed,
            "reason": self.reason,
            "eligible_workers": self.eligible_workers,
            "active_requests": self.active_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
        }
        if self.worker_state is not None:
            payload["worker_state"] = self.worker_state
        if self.worker_usable_memory_bytes is not None:
            payload["worker_usable_memory_bytes"] = self.worker_usable_memory_bytes
        if self.worker_storage_missing_bytes is not None:
            payload["worker_storage_missing_bytes"] = self.worker_storage_missing_bytes
        return payload


@dataclass(frozen=True)
class ContributionAdmission:
    status: ContributionStatus

    @property
    def allowed(self) -> bool:
        return self.status.allowed

    @property
    def account_id(self) -> Optional[str]:
        return self.status.account_id


class ContributionGate:
    """Binary contribution admission bound to account and live worker identity.

    One eligible worker grants one concurrent request by default.  This keeps
    the product free of balances and currencies while preventing a single tiny
    contribution credential from being shared to create unbounded traffic.
    Admission is checked when a request starts. V3 Request Agents then keep the
    permit alive with explicit authority acknowledgements; every acknowledgement
    revalidates that the same account still contributes.  Existing route leases,
    rather than fresh-admission capacity, decide whether the in-flight data plane
    is still serving. A slow generation is therefore not mistaken for a failed
    one merely because it consumes the swarm's last free session.
    """

    def __init__(self) -> None:
        mode = os.environ.get("FABI_GATE", "off").strip().lower()
        self.enabled = mode in _ENABLED
        raw_limit = os.environ.get("FABI_GATE_REQUESTS_PER_WORKER", "1").strip()
        try:
            self.requests_per_worker = min(16, max(1, int(raw_limit)))
        except ValueError:
            self.requests_per_worker = 1
            logger.warning(
                "Ignoring invalid FABI_GATE_REQUESTS_PER_WORKER=%r; using 1",
                raw_limit,
            )
        self._lock = threading.RLock()
        self._active_requests: dict[str, int] = {}
        self._route_permit_ledger: ContributionRoutePermitLedger | None = None
        logger.info(
            "Contribution gate %s (requests_per_worker=%d)",
            "enabled" if self.enabled else "disabled",
            self.requests_per_worker,
        )

    def bind_route_permit_ledger(self, ledger: ContributionRoutePermitLedger) -> None:
        """Share one capacity counter between gateway and Request Agent traffic."""

        with self._lock:
            if self._route_permit_ledger is not None and self._route_permit_ledger is not ledger:
                raise RuntimeError("contribution gate already has a route permit ledger")
            self._route_permit_ledger = ledger

    @staticmethod
    def _external_ready_worker_ids(scheduler) -> frozenset[str] | None:
        provider = getattr(scheduler, "external_ready_worker_ids", None)
        if not callable(provider):
            return None
        try:
            workers = provider()
        except Exception:
            logger.warning("Unable to read authoritative v3 membership for contribution")
            return frozenset()
        return None if workers is None else frozenset(str(worker) for worker in workers)

    @classmethod
    def _eligible_workers(cls, scheduler, identity: str, now: float) -> int:
        if scheduler is None:
            return 0
        timeout = max(0.0, float(getattr(scheduler, "heartbeat_timeout", 0.0)))
        external_workers = cls._external_ready_worker_ids(scheduler)
        candidates = (
            scheduler.node_manager.nodes
            if external_workers is not None
            else scheduler.node_manager.active_nodes
        )
        workers = 0
        for node in candidates:
            if getattr(node, "account_hash", None) != identity:
                continue
            if not node_is_routable(node):
                continue
            report = getattr(node, "swarm_v3", None)
            autonomous = bool(
                getattr(node, "uses_autonomous_placement", False)
                or (isinstance(report, dict) and report.get("placement_mode") == "autonomous")
            )
            if external_workers is not None and not autonomous:
                # The presence of an authoritative DHT membership provider
                # means the product is running v3-only.  A stale scheduler
                # allocation must not unlock consumption.
                continue
            if autonomous:
                if external_workers is None or str(node.node_id) not in external_workers:
                    continue
                if not isinstance(report, dict) or report.get("state") != "ready":
                    continue
                try:
                    advertisement = ModelMemberAdvertisement.model_validate(
                        report.get("advertisement")
                    )
                except Exception:
                    continue
                if (
                    advertisement.offer.worker_id != str(node.node_id)
                    or advertisement.lease.worker_id != str(node.node_id)
                    or advertisement.lease.state is not SpanState.READY
                ):
                    continue
            else:
                start = getattr(node, "start_layer", None)
                end = getattr(node, "end_layer", None)
                if start is None or end is None or int(end) <= int(start):
                    continue
            # Current schedulers decide from a monotonic adaptive detector.
            # Keep the wall-clock timeout only for legacy projections that do
            # not expose liveness, avoiding false gate closure after an OS clock
            # correction.
            if not hasattr(node, "liveness_state"):
                heartbeat = float(getattr(node, "last_heartbeat", 0.0))
                if timeout > 0 and now - heartbeat > timeout:
                    continue
            # Measured executor KV telemetry is part of the product serving
            # contract.  A worker that merely claims READY without an initialized
            # cache is not yet a contributor eligible to unlock consumption.
            if getattr(node, "effective_kv_cache_token_capacity", None) is None:
                continue
            workers += 1
        return workers

    @staticmethod
    def _account_worker_diagnostic(
        scheduler, identity: str
    ) -> tuple[str | None, int | None, int | None]:
        """Return an account-scoped lifecycle summary without exposing peer IDs."""

        if scheduler is None:
            return None, None
        manager = getattr(scheduler, "node_manager", None)
        nodes = getattr(manager, "nodes", getattr(manager, "active_nodes", ()))
        states: list[str] = []
        usable: list[int] = []
        storage_missing: list[int] = []
        for node in nodes:
            if getattr(node, "account_hash", None) != identity:
                continue
            report = getattr(node, "swarm_v3", None)
            if not isinstance(report, dict):
                states.append("connected")
                continue
            capacity = report.get("capacity")
            if isinstance(capacity, dict):
                value = capacity.get("usable_memory_bytes")
                if isinstance(value, int) and value >= 0:
                    usable.append(value)
            storage = report.get("storage")
            if isinstance(storage, dict):
                missing = storage.get("minimum_missing_bytes", storage.get("missing_bytes"))
                if isinstance(missing, int) and missing >= 0:
                    storage_missing.append(missing)
            if report.get("state") == "ready":
                states.append("ready")
                continue
            placement = report.get("placement")
            phase = placement.get("phase") if isinstance(placement, dict) else None
            decision = placement.get("decision") if isinstance(placement, dict) else None
            if phase == "building":
                states.append("building")
            elif decision in {
                "insufficient_live_memory",
                "no_exact_span_fits_the_stable_memory_envelope",
            }:
                states.append("insufficient_memory")
            elif decision == "no_exact_span_fits_local_artifact_storage":
                states.append("insufficient_storage")
            elif decision == "waiting_catalog":
                states.append("waiting_catalog")
            else:
                states.append("standby")
        for preferred in (
            "ready",
            "building",
            "waiting_catalog",
            "standby",
            "insufficient_storage",
            "insufficient_memory",
            "connected",
        ):
            if preferred in states:
                return (
                    preferred,
                    max(usable) if usable else None,
                    min(storage_missing) if storage_missing else None,
                )
        return None, None, None

    def status(self, credential: object, scheduler) -> ContributionStatus:
        if not self.enabled:
            return ContributionStatus(True, "gate_disabled")

        identity = account_hash(credential)
        if identity is None:
            reason = "missing_credential" if not credential else "invalid_credential"
            return ContributionStatus(False, reason)

        product_ready = getattr(scheduler, "product_serving_ready", None)
        serving_ready = bool(
            scheduler is not None
            and (product_ready() if callable(product_ready) else scheduler.serving_ready())
        )
        eligible = self._eligible_workers(scheduler, identity, time.time())
        with self._lock:
            gateway_active = self._active_requests.get(identity, 0)
            try:
                permit_active = (
                    0
                    if self._route_permit_ledger is None
                    else self._route_permit_ledger.active_count(identity)
                )
            except Exception:
                logger.exception("Unable to read the route permit capacity ledger")
                return ContributionStatus(
                    False,
                    "admission_unavailable",
                    eligible_workers=eligible,
                    active_requests=gateway_active,
                    max_concurrent_requests=eligible * self.requests_per_worker,
                    account_id=identity,
                )
            active = gateway_active + permit_active
        maximum = eligible * self.requests_per_worker

        if eligible == 0:
            worker_state, worker_usable_memory, worker_storage_missing = (
                self._account_worker_diagnostic(
                scheduler, identity
                )
            )
            return ContributionStatus(
                False,
                "no_eligible_worker",
                active_requests=active,
                account_id=identity,
                worker_state=worker_state,
                worker_usable_memory_bytes=worker_usable_memory,
                worker_storage_missing_bytes=worker_storage_missing,
            )
        if active >= maximum:
            return ContributionStatus(
                False,
                "capacity_reached",
                eligible_workers=eligible,
                active_requests=active,
                max_concurrent_requests=maximum,
                account_id=identity,
            )
        if not serving_ready:
            return ContributionStatus(
                False,
                "swarm_not_ready",
                eligible_workers=eligible,
                active_requests=active,
                max_concurrent_requests=maximum,
                account_id=identity,
            )
        return ContributionStatus(
            True,
            "eligible",
            eligible_workers=eligible,
            active_requests=active,
            max_concurrent_requests=maximum,
            account_id=identity,
        )

    def admit(self, credential: object, scheduler) -> ContributionAdmission:
        status = self.status(credential, scheduler)
        if not status.allowed or not self.enabled or status.account_id is None:
            return ContributionAdmission(status)
        with self._lock:
            gateway_active = self._active_requests.get(status.account_id, 0)
            try:
                permit_active = (
                    0
                    if self._route_permit_ledger is None
                    else self._route_permit_ledger.active_count(status.account_id)
                )
            except Exception:
                logger.exception("Unable to read the route permit capacity ledger")
                return ContributionAdmission(
                    ContributionStatus(
                        False,
                        "admission_unavailable",
                        eligible_workers=status.eligible_workers,
                        active_requests=gateway_active,
                        max_concurrent_requests=status.max_concurrent_requests,
                        account_id=status.account_id,
                    )
                )
            active = gateway_active + permit_active
            if active >= status.max_concurrent_requests:
                return ContributionAdmission(
                    ContributionStatus(
                        False,
                        "capacity_reached",
                        eligible_workers=status.eligible_workers,
                        active_requests=active,
                        max_concurrent_requests=status.max_concurrent_requests,
                        account_id=status.account_id,
                    )
                )
            self._active_requests[status.account_id] = gateway_active + 1
            return ContributionAdmission(
                ContributionStatus(
                    True,
                    "eligible",
                    eligible_workers=status.eligible_workers,
                    active_requests=active + 1,
                    max_concurrent_requests=status.max_concurrent_requests,
                    account_id=status.account_id,
                )
            )

    def issue_route_permit(
        self,
        credential: object,
        scheduler,
        *,
        request_id: str,
        idempotency_key: str,
        coordinator_endpoint_id: str,
        model_swarm_id: str,
        max_context_tokens: int,
        recovery_policies: frozenset[RouteRecoveryPolicy],
        ttl_ms: int,
    ) -> AuthorizedContributionPermit:
        """Atomically exchange one live contribution slot for a route permit."""

        if not self.enabled:
            raise RuntimeError("route permit authority requires FABI_GATE=on")
        if ttl_ms <= 0 or ttl_ms > _MAX_ROUTE_PERMIT_TTL_MS:
            raise ValueError("route permit TTL must be between 1 ms and 5 minutes")
        if not recovery_policies or not recovery_policies <= _PUBLIC_RECOVERY_POLICIES:
            raise ValueError("route permit requests an unsupported public recovery policy")
        with self._lock:
            ledger = self._route_permit_ledger
            if ledger is None:
                raise RuntimeError("route permit ledger is not configured")
            identity = account_hash(credential)
            if identity is None:
                reason = "missing_credential" if not credential else "invalid_credential"
                raise ContributionPermitDenied(ContributionStatus(False, reason))
            existing = ledger.find_active_by_idempotency(
                account_id=identity,
                idempotency_key=idempotency_key,
                coordinator_endpoint_id=coordinator_endpoint_id,
            )
            if existing is not None:
                if (
                    existing.request_id != request_id
                    or existing.model_swarm_id != model_swarm_id
                    or existing.max_context_tokens != max_context_tokens
                    or existing.recovery_policies != recovery_policies
                    or existing.initial_ttl_ms != ttl_ms
                ):
                    raise RoutePermitConflict(
                        "request idempotency key was reused with a different permit contract"
                    )
                return existing
            status = self.status(credential, scheduler)
            if not status.allowed or status.account_id is None:
                raise ContributionPermitDenied(status)
            gateway_active = self._active_requests.get(status.account_id, 0)
            permit_capacity = status.max_concurrent_requests - gateway_active
            if permit_capacity <= 0:
                raise ContributionPermitDenied(
                    ContributionStatus(
                        False,
                        "capacity_reached",
                        eligible_workers=status.eligible_workers,
                        active_requests=status.active_requests,
                        max_concurrent_requests=status.max_concurrent_requests,
                        account_id=status.account_id,
                    )
                )
            return ledger.issue(
                account_id=status.account_id,
                request_id=request_id,
                idempotency_key=idempotency_key,
                coordinator_endpoint_id=coordinator_endpoint_id,
                model_swarm_id=model_swarm_id,
                max_context_tokens=max_context_tokens,
                recovery_policies=recovery_policies,
                ttl_ms=ttl_ms,
                max_active_per_account=permit_capacity,
            )

    def keepalive_route_permit(
        self,
        credential: object,
        scheduler,
        *,
        permit_id: str,
        ttl_ms: int,
        idempotency_key: str,
    ) -> AuthorizedContributionPermit:
        """Continue one in-flight slot only while its worker still contributes."""

        if not self.enabled:
            raise RuntimeError("route permit authority requires FABI_GATE=on")
        if ttl_ms <= 0 or ttl_ms > _MAX_ROUTE_PERMIT_TTL_MS:
            raise ValueError("route permit TTL must be between 1 ms and 5 minutes")
        identity = account_hash(credential)
        if identity is None:
            reason = "missing_credential" if not credential else "invalid_credential"
            raise ContributionPermitDenied(ContributionStatus(False, reason))
        with self._lock:
            ledger = self._route_permit_ledger
            if ledger is None:
                raise RuntimeError("route permit ledger is not configured")
            permit = ledger.get_active(permit_id)
            if permit.account_id != identity:
                raise PermissionError("route permit belongs to a different account")
            eligible = self._eligible_workers(scheduler, identity, time.time())
            if eligible == 0:
                raise ContributionPermitDenied(
                    ContributionStatus(
                        False,
                        "no_eligible_worker",
                        active_requests=1,
                        account_id=identity,
                    )
                )
            return ledger.keepalive_owned(
                permit_id,
                identity,
                ttl_ms=ttl_ms,
                idempotency_key=idempotency_key,
            )

    def release(self, admission: ContributionAdmission) -> None:
        identity = admission.account_id
        if not self.enabled or identity is None or not admission.allowed:
            return
        with self._lock:
            active = self._active_requests.get(identity, 0)
            if active <= 1:
                self._active_requests.pop(identity, None)
            else:
                self._active_requests[identity] = active - 1

    @staticmethod
    def denial_payload(status: ContributionStatus) -> dict:
        if status.reason == "capacity_reached":
            message = "Ta contribution sert déjà une génération. Réessaie quand elle est terminée."
            code = "contribution_capacity_reached"
        elif status.reason == "swarm_not_ready":
            message = "Ton worker contribue, mais aucun pipeline complet n'est encore prêt."
            code = "swarm_not_ready"
        else:
            message = "Connecte et charge ton worker sur ce modèle pour utiliser Fabi."
            code = "contribution_required"
        return {
            "error": {
                "message": message,
                "type": "contribution_required",
                "code": code,
            }
        }


_singleton: Optional[ContributionGate] = None
_singleton_lock = threading.Lock()


def get_gate() -> ContributionGate:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ContributionGate()
    return _singleton
