"""Admission control for Fabi's ``contribute while you consume`` contract.

This is deliberately not a credit ledger.  The scheduler derives a short-lived
capability from its own live node registry on every admission: an account may
start inference only while it owns at least one ready worker with a real layer
allocation in this model's serving pipeline.

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
from typing import Optional

from parallax_utils.logging_config import get_logger
from scheduling.node import node_is_routable

logger = get_logger(__name__)

_ACCOUNT_CREDENTIAL = re.compile(r"^[0-9a-fA-F]{64}$")
_ENABLED = {"1", "true", "yes", "on", "strict"}


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

    def public_payload(self, *, enabled: bool) -> dict:
        return {
            "enabled": enabled,
            "allowed": self.allowed,
            "reason": self.reason,
            "eligible_workers": self.eligible_workers,
            "active_requests": self.active_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
        }


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
    """Binary, scheduler-authoritative contribution admission gate.

    One eligible worker grants one concurrent request by default.  This keeps
    the product free of balances and currencies while preventing a single tiny
    contribution credential from being shared to create unbounded traffic.
    In-flight generations are never interrupted; eligibility is checked only
    when a new request is admitted.
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
        self._lock = threading.Lock()
        self._active_requests: dict[str, int] = {}
        logger.info(
            "Contribution gate %s (requests_per_worker=%d)",
            "enabled" if self.enabled else "disabled",
            self.requests_per_worker,
        )

    @staticmethod
    def _eligible_workers(scheduler, identity: str, now: float) -> int:
        if scheduler is None:
            return 0
        timeout = max(0.0, float(getattr(scheduler, "heartbeat_timeout", 0.0)))
        workers = 0
        for node in scheduler.node_manager.active_nodes:
            if getattr(node, "account_hash", None) != identity:
                continue
            if not node_is_routable(node):
                continue
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

    def status(self, credential: object, scheduler) -> ContributionStatus:
        if not self.enabled:
            return ContributionStatus(True, "gate_disabled")

        identity = account_hash(credential)
        if identity is None:
            reason = "missing_credential" if not credential else "invalid_credential"
            return ContributionStatus(False, reason)

        serving_ready = bool(scheduler is not None and scheduler.serving_ready())
        eligible = self._eligible_workers(scheduler, identity, time.time())
        with self._lock:
            active = self._active_requests.get(identity, 0)
        maximum = eligible * self.requests_per_worker

        if eligible == 0:
            return ContributionStatus(
                False,
                "no_eligible_worker",
                active_requests=active,
                account_id=identity,
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
            active = self._active_requests.get(status.account_id, 0)
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
            self._active_requests[status.account_id] = active + 1
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
