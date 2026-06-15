"""Contribution gate — « tu contribues = tu consommes ».

Porte binaire (pas de comptabilité de points) : un compte dont un worker est
connecté au swarm (actif OU standby) peut consommer l'API sans limite ; sans
worker connecté, l'API de complétion répond 402.

Conception (cf. docs/fabi-contribution-gate-plan.md) :

- Le scheduler est juge ET témoin : il accepte les joins, assigne les couches,
  reçoit les heartbeats et route les requêtes. Le client ne déclare jamais rien
  → rien à falsifier. Le bail (« lease ») n'est rafraîchi QUE par le scheduler,
  lors d'un node_join accepté puis par les node_update, jamais sur demande d'un
  client.
- Le bail est une clé à TTL (300 s par défaut). Un worker évincé (timeout
  heartbeat, blacklist backoff) cesse d'être rafraîchi → le bail expire seul →
  la porte se referme. Aucun code de révocation.
- Store : par défaut, dict EN MÉMOIRE par scheduler (= par modèle). C'est le bon
  modèle : un utilisateur contribue toujours au swarm du modèle qu'il consomme
  (changer de modèle = quitter un swarm et rejoindre l'autre), donc le bail
  par-modèle suffit. Redis est OPTIONNEL (FABI_GATE_REDIS_URL) et ne sert qu'à un
  éventuel déblocage cross-modèle (contribuer à X, consommer Y) — non requis.
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
        # Redis est OPTIONNEL et opt-in. Par défaut (URL non définie) → store
        # mémoire PAR SCHEDULER. C'est le bon choix : un utilisateur contribue
        # toujours au swarm du modèle qu'il consomme (changer de modèle = changer
        # de swarm), donc le bail par-modèle suffit — pas besoin de partager les
        # baux entre les 5 schedulers. Définir FABI_GATE_REDIS_URL UNIQUEMENT si
        # l'on veut un déblocage cross-modèle (contribuer à X, consommer Y).
        self._redis_url = os.environ.get("FABI_GATE_REDIS_URL") or None

        if self.enabled:
            if self._redis_url:
                self._init_redis()
            logger.info(
                "ContributionGate ENABLED (mode=%s, lease=%ss, store=%s, allowlist=%d)",
                self.mode,
                self.lease_s,
                "redis(cross-model)" if self._redis is not None else "memory(per-model)",
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
        """Refresh the contribution lease for an account.

        Called by the scheduler for accepted node_join and node_update messages.
        Standby nodes refresh leases too: they are connected contributors waiting
        to be promoted if the active pipeline loses capacity.
        """
        if not self.enabled or not account_token:
            return
        h = _hash_token(account_token)
        payload = json.dumps({"node_id": node_id, "model": model, "ts": int(time.time())})
        if self._redis is not None:
            try:
                self._redis.setex(_LEASE_PREFIX + h, self.lease_s, payload)
                logger.info("[gate] lease refreshed (redis) token=%s… model=%s", h[:8], model)
                return
            except Exception as exc:  # pragma: no cover - transient redis error
                logger.warning("ContributionGate: redis setex failed (%s); using memory", exc)
        with self._lock:
            self._mem[h] = time.time() + self.lease_s
            n = len(self._mem)
        # Diagnostic : gate=id permet de détecter un éventuel double-singleton
        # (refresh et is_allowed doivent loguer le MÊME gate=...).
        logger.info(
            "[gate] lease refreshed (mem) token=%s… model=%s mem=%d gate=%x",
            h[:8],
            model,
            n,
            id(self),
        )

    # ----- read side (HTTP handler) ----------------------------------------

    def is_allowed(self, account_token: Optional[str]) -> bool:
        if not self.enabled:
            return True
        if not account_token:
            return False
        h = _hash_token(account_token)
        if h in self._allow_hashes:
            logger.info("[gate] is_allowed token=%s… -> allowlist", h[:8])
            return True
        if self._redis is not None:
            try:
                ok = bool(self._redis.exists(_LEASE_PREFIX + h))
                logger.info("[gate] is_allowed token=%s… -> redis lease=%s", h[:8], ok)
                return ok
            except Exception as exc:  # pragma: no cover - transient redis error
                logger.warning("ContributionGate: redis exists failed (%s); using memory", exc)
        with self._lock:
            expiry = self._mem.get(h)
            n = len(self._mem)
            alive = expiry is not None and expiry > time.time()
            if expiry is not None and not alive:
                self._mem.pop(h, None)
        logger.info("[gate] is_allowed token=%s… -> mem lease=%s mem=%d gate=%x", h[:8], alive, n, id(self))
        return alive

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
