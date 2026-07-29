"""Transactional permits for client-coordinated Fabi route capabilities."""

from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import stat
import threading
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from fabi_network.capability import RouteRecoveryPolicy

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ENDPOINT_RE = _HASH_RE
_DIGEST_RE = _HASH_RE
_REVOCATION_RE = re.compile(r"^[0-9a-f]{128}$")
_MAX_CAPABILITY_TOKEN_BYTES = 1024 * 1024


class RoutePermitError(RuntimeError):
    pass


class RoutePermitConflict(RoutePermitError):
    pass


class RoutePermitCapacityReached(RoutePermitError):
    pass


class RoutePermitExpired(RoutePermitError):
    pass


class StalePermitEpoch(RoutePermitError):
    pass


class RoutePermitState(str, Enum):
    ACTIVE = "active"
    RELEASED = "released"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass(frozen=True)
class AuthorizedContributionPermit:
    """One account/request slot allowed to mint bounded route capabilities."""

    permit_id: str
    account_id: str
    request_id: str
    coordinator_endpoint_id: str
    model_swarm_id: str
    max_context_tokens: int
    recovery_policies: frozenset[RouteRecoveryPolicy]
    issued_at_ms: int
    expires_at_ms: int


@dataclass(frozen=True)
class ClaimedRoutePlan:
    permit: AuthorizedContributionPermit
    epoch: int
    route_plan_digest: str


@dataclass(frozen=True)
class RouteCapabilityIssuance:
    permit_id: str
    epoch: int
    route_plan_digest: str
    authority_key_id: str
    capability_token: str
    root_revocation_id: str
    recovery_policy: RouteRecoveryPolicy
    expires_at_ms: int


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _validate_hash(value: str, name: str) -> None:
    if not _HASH_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 32-byte hexadecimal value")


def _validate_request_id(value: str) -> None:
    if (
        not value
        or len(value) > 512
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError("request ID is empty, too long, or contains control characters")


class SqliteRoutePermitLedger:
    """Single-service durable ledger with atomic quota and epoch CAS.

    SQLite WAL keeps readers non-blocking. ``BEGIN IMMEDIATE`` serializes the
    short quota/epoch mutations and fails before any partial state is visible.
    The API is storage-agnostic so a PostgreSQL implementation can provide the
    same contract for horizontally scaled issuers.
    """

    def __init__(
        self,
        path: Path,
        *,
        clock_ms=_system_clock_ms,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        elif os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise PermissionError(f"route permit ledger must be owner-only: {path}")
        self._clock_ms = clock_ms
        self._connection = sqlite3.connect(
            path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS route_permits (
                permit_id TEXT PRIMARY KEY NOT NULL,
                account_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                coordinator_endpoint_id TEXT NOT NULL,
                model_swarm_id TEXT NOT NULL,
                max_context_tokens INTEGER NOT NULL,
                recovery_policies_json TEXT NOT NULL,
                issued_at_ms INTEGER NOT NULL,
                expires_at_ms INTEGER NOT NULL,
                state TEXT NOT NULL,
                current_epoch INTEGER NOT NULL DEFAULT 0,
                current_plan_digest TEXT,
                UNIQUE(account_id, request_id, coordinator_endpoint_id)
            );
            CREATE INDEX IF NOT EXISTS route_permits_account_state
                ON route_permits(account_id, state, expires_at_ms);
            CREATE TABLE IF NOT EXISTS route_capability_issuances (
                permit_id TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                route_plan_digest TEXT NOT NULL,
                authority_key_id TEXT NOT NULL,
                capability_token TEXT NOT NULL,
                root_revocation_id TEXT NOT NULL,
                recovery_policy TEXT NOT NULL,
                expires_at_ms INTEGER NOT NULL,
                PRIMARY KEY(permit_id, epoch),
                FOREIGN KEY(permit_id) REFERENCES route_permits(permit_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS route_capability_revocations
                ON route_capability_issuances(root_revocation_id);
            """)
        self._lock = threading.RLock()

    def _now_ms(self) -> int:
        now = int(self._clock_ms())
        if now < 0:
            raise RuntimeError("route permit ledger clock returned a negative timestamp")
        return now

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def _expire(self, now_ms: int) -> None:
        self._connection.execute(
            """
            UPDATE route_permits
            SET state = ?
            WHERE state = ? AND expires_at_ms <= ?
            """,
            (
                RoutePermitState.EXPIRED.value,
                RoutePermitState.ACTIVE.value,
                now_ms,
            ),
        )

    @staticmethod
    def _permit(row: sqlite3.Row) -> AuthorizedContributionPermit:
        policies = frozenset(
            RouteRecoveryPolicy(value) for value in json.loads(str(row["recovery_policies_json"]))
        )
        return AuthorizedContributionPermit(
            permit_id=str(row["permit_id"]),
            account_id=str(row["account_id"]),
            request_id=str(row["request_id"]),
            coordinator_endpoint_id=str(row["coordinator_endpoint_id"]),
            model_swarm_id=str(row["model_swarm_id"]),
            max_context_tokens=int(row["max_context_tokens"]),
            recovery_policies=policies,
            issued_at_ms=int(row["issued_at_ms"]),
            expires_at_ms=int(row["expires_at_ms"]),
        )

    @staticmethod
    def _issuance(row: sqlite3.Row) -> RouteCapabilityIssuance:
        return RouteCapabilityIssuance(
            permit_id=str(row["permit_id"]),
            epoch=int(row["epoch"]),
            route_plan_digest=str(row["route_plan_digest"]),
            authority_key_id=str(row["authority_key_id"]),
            capability_token=str(row["capability_token"]),
            root_revocation_id=str(row["root_revocation_id"]),
            recovery_policy=RouteRecoveryPolicy(str(row["recovery_policy"])),
            expires_at_ms=int(row["expires_at_ms"]),
        )

    def get_active(self, permit_id: str) -> AuthorizedContributionPermit:
        _validate_hash(permit_id, "permit ID")
        now_ms = self._now_ms()
        with self._transaction():
            self._expire(now_ms)
            row = self._connection.execute(
                "SELECT * FROM route_permits WHERE permit_id = ?",
                (permit_id,),
            ).fetchone()
            if row is None:
                raise RoutePermitError("unknown route permit")
            if row["state"] != RoutePermitState.ACTIVE.value:
                raise RoutePermitExpired(f"route permit is {row['state']}")
            return self._permit(row)

    def active_count(self, account_id: str) -> int:
        _validate_hash(account_id, "account ID")
        now_ms = self._now_ms()
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM route_permits
                WHERE account_id = ? AND state = ? AND expires_at_ms > ?
                """,
                (account_id, RoutePermitState.ACTIVE.value, now_ms),
            ).fetchone()
            assert row is not None
            return int(row["count"])

    def find_active(
        self,
        *,
        account_id: str,
        request_id: str,
        coordinator_endpoint_id: str,
    ) -> AuthorizedContributionPermit | None:
        _validate_hash(account_id, "account ID")
        _validate_request_id(request_id)
        _validate_hash(coordinator_endpoint_id, "coordinator EndpointId")
        now_ms = self._now_ms()
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM route_permits
                WHERE account_id = ? AND request_id = ?
                  AND coordinator_endpoint_id = ? AND state = ?
                  AND expires_at_ms > ?
                """,
                (
                    account_id,
                    request_id,
                    coordinator_endpoint_id,
                    RoutePermitState.ACTIVE.value,
                    now_ms,
                ),
            ).fetchone()
            return None if row is None else self._permit(row)

    def issue(
        self,
        *,
        account_id: str,
        request_id: str,
        coordinator_endpoint_id: str,
        model_swarm_id: str,
        max_context_tokens: int,
        recovery_policies: frozenset[RouteRecoveryPolicy],
        ttl_ms: int,
        max_active_per_account: int,
        permit_id: str | None = None,
    ) -> AuthorizedContributionPermit:
        for value, name in (
            (account_id, "account ID"),
            (coordinator_endpoint_id, "coordinator EndpointId"),
            (model_swarm_id, "model swarm ID"),
        ):
            _validate_hash(value, name)
        _validate_request_id(request_id)
        if max_context_tokens <= 0 or ttl_ms <= 0 or max_active_per_account <= 0:
            raise ValueError("context, TTL, and account capacity must be positive")
        if not recovery_policies:
            raise ValueError("at least one recovery policy is required")
        identifier = permit_id or secrets.token_hex(32)
        _validate_hash(identifier, "permit ID")
        now_ms = self._now_ms()
        expires_at_ms = now_ms + ttl_ms
        policies_json = json.dumps(
            sorted(policy.value for policy in recovery_policies),
            separators=(",", ":"),
        )

        with self._transaction():
            self._expire(now_ms)
            existing = self._connection.execute(
                """
                SELECT * FROM route_permits
                WHERE account_id = ? AND request_id = ?
                  AND coordinator_endpoint_id = ?
                """,
                (account_id, request_id, coordinator_endpoint_id),
            ).fetchone()
            if existing is not None:
                permit = self._permit(existing)
                if (
                    existing["state"] == RoutePermitState.ACTIVE.value
                    and permit.model_swarm_id == model_swarm_id
                    and permit.max_context_tokens == max_context_tokens
                    and permit.recovery_policies == recovery_policies
                ):
                    return permit
                raise RoutePermitConflict(
                    "request idempotency key was reused with a different permit contract"
                )

            active = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM route_permits
                WHERE account_id = ? AND state = ? AND expires_at_ms > ?
                """,
                (account_id, RoutePermitState.ACTIVE.value, now_ms),
            ).fetchone()
            assert active is not None
            if int(active["count"]) >= max_active_per_account:
                raise RoutePermitCapacityReached("account route permit capacity reached")

            self._connection.execute(
                """
                INSERT INTO route_permits(
                    permit_id, account_id, request_id, coordinator_endpoint_id,
                    model_swarm_id, max_context_tokens, recovery_policies_json,
                    issued_at_ms, expires_at_ms, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    account_id,
                    request_id,
                    coordinator_endpoint_id,
                    model_swarm_id,
                    max_context_tokens,
                    policies_json,
                    now_ms,
                    expires_at_ms,
                    RoutePermitState.ACTIVE.value,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM route_permits WHERE permit_id = ?",
                (identifier,),
            ).fetchone()
            assert row is not None
            return self._permit(row)

    def claim_plan(
        self,
        *,
        permit_id: str,
        account_id: str,
        request_id: str,
        coordinator_endpoint_id: str,
        model_swarm_id: str,
        epoch: int,
        route_plan_digest: str,
        required_context_tokens: int,
        recovery_policy: RouteRecoveryPolicy,
    ) -> ClaimedRoutePlan:
        _validate_hash(permit_id, "permit ID")
        _validate_hash(account_id, "account ID")
        _validate_request_id(request_id)
        _validate_hash(coordinator_endpoint_id, "coordinator EndpointId")
        _validate_hash(model_swarm_id, "model swarm ID")
        if not _DIGEST_RE.fullmatch(route_plan_digest):
            raise ValueError("route plan digest must be lowercase SHA-256")
        if epoch <= 0 or required_context_tokens <= 0:
            raise ValueError("epoch and required context must be positive")
        now_ms = self._now_ms()

        with self._transaction():
            self._expire(now_ms)
            row = self._connection.execute(
                "SELECT * FROM route_permits WHERE permit_id = ?",
                (permit_id,),
            ).fetchone()
            if row is None:
                raise RoutePermitError("unknown route permit")
            if row["state"] != RoutePermitState.ACTIVE.value:
                raise RoutePermitExpired(f"route permit is {row['state']}")
            permit = self._permit(row)
            if (
                permit.account_id != account_id
                or permit.request_id != request_id
                or permit.coordinator_endpoint_id != coordinator_endpoint_id
                or permit.model_swarm_id != model_swarm_id
            ):
                raise PermissionError("route plan does not match its contribution permit")
            if required_context_tokens > permit.max_context_tokens:
                raise PermissionError("route context exceeds its contribution permit")
            if recovery_policy not in permit.recovery_policies:
                raise PermissionError("route recovery policy exceeds its contribution permit")

            current_epoch = int(row["current_epoch"])
            current_digest = row["current_plan_digest"]
            if epoch < current_epoch:
                raise StalePermitEpoch(f"route epoch {epoch} is below permit epoch {current_epoch}")
            if epoch == current_epoch and current_epoch > 0:
                if current_digest != route_plan_digest:
                    raise RoutePermitConflict(
                        "one permit epoch cannot authorize two different route plans"
                    )
            else:
                self._connection.execute(
                    """
                    UPDATE route_permits
                    SET current_epoch = ?, current_plan_digest = ?
                    WHERE permit_id = ?
                    """,
                    (epoch, route_plan_digest, permit_id),
                )
            return ClaimedRoutePlan(
                permit=permit,
                epoch=epoch,
                route_plan_digest=route_plan_digest,
            )

    def record_issuance(
        self,
        claim: ClaimedRoutePlan,
        *,
        authority_key_id: str,
        capability_token: str,
        root_revocation_id: str,
        recovery_policy: RouteRecoveryPolicy,
        expires_at_ms: int,
    ) -> RouteCapabilityIssuance:
        _validate_hash(authority_key_id, "route authority key ID")
        if (
            not capability_token
            or len(capability_token.encode("utf-8")) > _MAX_CAPABILITY_TOKEN_BYTES
        ):
            raise ValueError("Biscuit capability token is empty or too large")
        if not _REVOCATION_RE.fullmatch(root_revocation_id):
            raise ValueError("Biscuit revocation identifier must be 64-byte lowercase hex")
        if recovery_policy not in claim.permit.recovery_policies:
            raise PermissionError("capability recovery policy exceeds its contribution permit")
        if expires_at_ms <= 0 or expires_at_ms > claim.permit.expires_at_ms:
            raise PermissionError("capability lifetime exceeds its contribution permit")
        now_ms = self._now_ms()
        if expires_at_ms <= now_ms:
            raise RoutePermitExpired("capability already expired before persistence")
        with self._transaction():
            self._expire(now_ms)
            row = self._connection.execute(
                """
                SELECT current_epoch, current_plan_digest, state
                FROM route_permits WHERE permit_id = ?
                """,
                (claim.permit.permit_id,),
            ).fetchone()
            if (
                row is None
                or row["state"] != RoutePermitState.ACTIVE.value
                or int(row["current_epoch"]) != claim.epoch
                or row["current_plan_digest"] != claim.route_plan_digest
            ):
                raise RoutePermitConflict("route permit moved before capability persistence")
            self._connection.execute(
                """
                INSERT INTO route_capability_issuances(
                    permit_id, epoch, route_plan_digest,
                    authority_key_id, capability_token, root_revocation_id,
                    recovery_policy, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(permit_id, epoch) DO NOTHING
                """,
                (
                    claim.permit.permit_id,
                    claim.epoch,
                    claim.route_plan_digest,
                    authority_key_id,
                    capability_token,
                    root_revocation_id,
                    recovery_policy.value,
                    expires_at_ms,
                ),
            )
            persisted_row = self._connection.execute(
                """
                SELECT * FROM route_capability_issuances
                WHERE permit_id = ? AND epoch = ?
                """,
                (claim.permit.permit_id, claim.epoch),
            ).fetchone()
            assert persisted_row is not None
            persisted = self._issuance(persisted_row)
            if (
                persisted.route_plan_digest != claim.route_plan_digest
                or persisted.recovery_policy != recovery_policy
                or persisted.expires_at_ms != expires_at_ms
            ):
                raise RoutePermitConflict(
                    "permit epoch already contains a different capability contract"
                )
            return persisted

    def get_issuance(
        self,
        *,
        permit_id: str,
        epoch: int,
    ) -> RouteCapabilityIssuance | None:
        _validate_hash(permit_id, "permit ID")
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM route_capability_issuances
                WHERE permit_id = ? AND epoch = ?
                """,
                (permit_id, epoch),
            ).fetchone()
        return None if row is None else self._issuance(row)

    def release(self, permit_id: str) -> bool:
        _validate_hash(permit_id, "permit ID")
        with self._transaction():
            changed = self._connection.execute(
                """
                UPDATE route_permits SET state = ?
                WHERE permit_id = ? AND state = ?
                """,
                (
                    RoutePermitState.RELEASED.value,
                    permit_id,
                    RoutePermitState.ACTIVE.value,
                ),
            ).rowcount
            return changed > 0

    def release_owned(self, permit_id: str, account_id: str) -> bool:
        _validate_hash(permit_id, "permit ID")
        _validate_hash(account_id, "account ID")
        with self._transaction():
            changed = self._connection.execute(
                """
                UPDATE route_permits SET state = ?
                WHERE permit_id = ? AND account_id = ? AND state = ?
                """,
                (
                    RoutePermitState.RELEASED.value,
                    permit_id,
                    account_id,
                    RoutePermitState.ACTIVE.value,
                ),
            ).rowcount
            return changed > 0

    def revoke(self, permit_id: str) -> tuple[str, ...]:
        _validate_hash(permit_id, "permit ID")
        with self._transaction():
            self._connection.execute(
                """
                UPDATE route_permits SET state = ?
                WHERE permit_id = ? AND state != ?
                """,
                (
                    RoutePermitState.REVOKED.value,
                    permit_id,
                    RoutePermitState.EXPIRED.value,
                ),
            )
            return tuple(
                str(row["root_revocation_id"])
                for row in self._connection.execute(
                    """
                    SELECT root_revocation_id
                    FROM route_capability_issuances
                    WHERE permit_id = ? AND expires_at_ms > ?
                    ORDER BY epoch
                    """,
                    (permit_id, self._now_ms()),
                )
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()
