"""Contribution gate — « tu contribues = tu consommes ».

Porte binaire (pas de comptabilité de points) : un compte dont un worker est
actif dans le swarm peut consommer l'API sans limite ; sans worker actif, l'API
de complétion répond 402.

Conception (cf. docs/fabi-contribution-gate-plan.md) :

- Le scheduler est juge ET témoin : il assigne les couches, reçoit les
  heartbeats et route les requêtes. Le client ne déclare jamais rien → rien à
  falsifier. Le bail (« lease ») n'est rafraîchi QUE par le scheduler, depuis sa
  propre table de nœuds actifs (hook dans node_update), jamais sur demande d'un
  client.
- Le bail est une clé à TTL (300 s par défaut). Un worker évincé (timeout
  heartbeat, blacklist backoff) cesse d'être rafraîchi → le bail expire seul →
  la porte se referme. Aucun code de révocation.
- Store : Redis si disponible (partagé entre les schedulers par-modèle → un
  worker sur un swarm débloque tous les modèles) ; sinon fallback dict en
  mémoire (porte par-process — dégradation documentée).
- Le token de compte n'est jamais stocké en clair : on n'indexe que son SHA-256.

Activation : `FABI_GATE=on` (défaut `off` → comportement historique inchangé,
upstream-friendly). Surcharges : `FABI_GATE_LEASE_S`, `FABI_GATE_REDIS_URL`,
`FABI_GATE_ALLOWLIST` (tokens admin séparés par virgule).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Dict, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_LEASE_PREFIX = "fabi:lease:"


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _parse_allowlist(raw: str) -> set:
    return {_hash_token(t.strip()) for t in raw.split(",") if t.strip()}


class ContributionGate:
    """Singleton-friendly gate. Thread-safe (heartbeat thread refreshes,
    HTTP handler reads)."""

    def __init__(self) -> None:
        self.mode = os.environ.get("FABI_GATE", "off").strip().lower()
        self.enabled = self.mode in {"on", "strict"}
        try:
            self.lease_s = max(10, int(float(os.environ.get("FABI_GATE_LEASE_S", "300"))))
        except (TypeError, ValueError):
            self.lease_s = 300
        self._allow_hashes = _parse_allowlist(os.environ.get("FABI_GATE_ALLOWLIST", ""))

        self._lock = threading.Lock()
        self._mem: Dict[str, float] = {}  # token_hash -> expiry (time.time())
        self._redis = None
        self._redis_url = os.environ.get("FABI_GATE_REDIS_URL", "redis://127.0.0.1:6379/0")

        if self.enabled:
            self._init_redis()
            logger.info(
                "ContributionGate ENABLED (mode=%s, lease=%ss, store=%s, allowlist=%d)",
                self.mode,
                self.lease_s,
                "redis" if self._redis is not None else "memory",
                len(self._allow_hashes),
            )
        else:
            logger.info("ContributionGate disabled (FABI_GATE=off) — open access")

    def _init_redis(self) -> None:
        try:
            import redis  # optional dep, present in the scheduler image only

            client = redis.Redis.from_url(
                self._redis_url, socket_timeout=1.0, socket_connect_timeout=1.0
            )
            client.ping()
            self._redis = client
        except Exception as exc:  # pragma: no cover - redis absent/unreachable
            logger.warning(
                "ContributionGate: Redis unavailable (%s) — falling back to in-memory "
                "lease store (per-process; leases not shared across schedulers).",
                exc,
            )
            self._redis = None

    # ----- write side (scheduler only) -------------------------------------

    def refresh(self, account_token: Optional[str], node_id: str, model: Optional[str]) -> None:
        """Refresh the contribution lease for an account. Called by the
        scheduler from node_update, ONLY for a node it considers active."""
        if not self.enabled or not account_token:
            return
        h = _hash_token(account_token)
        payload = json.dumps({"node_id": node_id, "model": model, "ts": int(time.time())})
        if self._redis is not None:
            try:
                self._redis.setex(_LEASE_PREFIX + h, self.lease_s, payload)
                return
            except Exception as exc:  # pragma: no cover - transient redis error
                logger.warning("ContributionGate: redis setex failed (%s); using memory", exc)
        with self._lock:
            self._mem[h] = time.time() + self.lease_s

    # ----- read side (HTTP handler) ----------------------------------------

    def is_allowed(self, account_token: Optional[str]) -> bool:
        if not self.enabled:
            return True
        if not account_token:
            return False
        h = _hash_token(account_token)
        if h in self._allow_hashes:
            return True
        if self._redis is not None:
            try:
                return bool(self._redis.exists(_LEASE_PREFIX + h))
            except Exception as exc:  # pragma: no cover - transient redis error
                logger.warning("ContributionGate: redis exists failed (%s); using memory", exc)
        with self._lock:
            expiry = self._mem.get(h)
            if expiry is None:
                return False
            if expiry <= time.time():
                self._mem.pop(h, None)
                return False
            return True

    # ----- denial payload (HTTP 402) ---------------------------------------

    @staticmethod
    def denial_payload(scheduler_peer: Optional[str] = None) -> dict:
        peer = scheduler_peer or "<scheduler-peer-id>"
        return {
            "error": {
                "code": "contribution_required",
                "message": (
                    "Fabi est un swarm : connecte un worker pour contribuer, et tu "
                    "pourras consommer sans limite. (tu utilises = tu contribues)"
                ),
                "type": "contribution_required",
                "join_command": f"parallax join -s {peer} --account-token <ton-token>",
            }
        }


_gate_singleton: Optional[ContributionGate] = None
_gate_lock = threading.Lock()


def get_gate() -> ContributionGate:
    """Process-wide singleton (FastAPI route + RPC handler share it)."""
    global _gate_singleton
    if _gate_singleton is None:
        with _gate_lock:
            if _gate_singleton is None:
                _gate_singleton = ContributionGate()
    return _gate_singleton
