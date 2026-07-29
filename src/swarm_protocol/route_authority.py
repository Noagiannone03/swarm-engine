"""Route-scoped admission authorities for protocol-v3 workers.

The data plane authenticates every Iroh peer independently.  This module only
decides whether that authenticated peer may coordinate one exact signed route.
It deliberately does not plan routes, reserve KV, or proxy inference traffic.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from fabi_network.capability import (
    RouteCapabilityContext,
    RouteRecoveryPolicy,
    verify_route_capability,
)
from swarm_protocol.contracts import RoutePlan
from swarm_protocol.control import (
    ControlCrypto,
    ControlMessageKind,
    SignedControlMessage,
    verify_control_contract,
)

if TYPE_CHECKING:
    from swarm_protocol.registry import TrustedModelRegistry

_MAX_CAPABILITY_BYTES = 32 * 1024
_DEFAULT_TRUST_REFRESH_MS = 60_000


class RouteAdmissionEnvelope(BaseModel):
    """A signed plan plus the authority grant that makes its signer eligible."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signed_plan: SignedControlMessage
    authority_key_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_token: str = Field(min_length=1, max_length=_MAX_CAPABILITY_BYTES)
    permit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    account_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    recovery_policy: RouteRecoveryPolicy


@dataclass(frozen=True)
class AuthorizedRoutePlan:
    plan: RoutePlan
    signed_plan: SignedControlMessage
    coordinator_endpoint_id: str
    route_plan_digest: str
    permit_id: str | None = None
    account_id: str | None = None
    recovery_policy: RouteRecoveryPolicy | None = None
    root_revocation_id: str | None = None


class RoutePlanAuthority(Protocol):
    def authorize_route(
        self,
        admission: SignedControlMessage | RouteAdmissionEnvelope | dict[str, object],
        *,
        caller_endpoint_id: str,
        crypto: ControlCrypto,
        now_ms: int,
    ) -> AuthorizedRoutePlan: ...

    def authorize_unbound_control(self, caller_endpoint_id: str) -> bool: ...


def route_plan_digest(signed_plan: SignedControlMessage) -> str:
    """Digest the exact authenticated payload, never a reserialized contract."""

    return hashlib.sha256(signed_plan.payload).hexdigest()


class FixedCoordinatorRouteAuthority:
    """Migration adapter for the current gateway-coordinated V3 runtime."""

    def __init__(self, coordinator_endpoint_id: str) -> None:
        if not coordinator_endpoint_id:
            raise ValueError("coordinator endpoint identity is required")
        self.coordinator_endpoint_id = coordinator_endpoint_id

    def authorize_route(
        self,
        admission: SignedControlMessage | RouteAdmissionEnvelope | dict[str, object],
        *,
        caller_endpoint_id: str,
        crypto: ControlCrypto,
        now_ms: int,
    ) -> AuthorizedRoutePlan:
        del now_ms
        if isinstance(admission, RouteAdmissionEnvelope) or (
            isinstance(admission, dict) and "signed_plan" in admission
        ):
            raise ValueError("fixed coordinator admission does not accept a capability envelope")
        if caller_endpoint_id != self.coordinator_endpoint_id:
            raise PermissionError("only the configured route coordinator may admit routes")
        signed_plan = (
            admission
            if isinstance(admission, SignedControlMessage)
            else SignedControlMessage.model_validate(admission)
        )
        plan = verify_control_contract(
            signed_plan,
            expected_kind=ControlMessageKind.ROUTE_PLAN,
            expected_signer_endpoint_id=self.coordinator_endpoint_id,
            contract_type=RoutePlan,
            crypto=crypto,
        )
        if plan.coordinator_id != self.coordinator_endpoint_id:
            raise PermissionError("route plan coordinator does not match its authenticated signer")
        return AuthorizedRoutePlan(
            plan=plan,
            signed_plan=signed_plan,
            coordinator_endpoint_id=self.coordinator_endpoint_id,
            route_plan_digest=route_plan_digest(signed_plan),
        )

    def authorize_unbound_control(self, caller_endpoint_id: str) -> bool:
        return caller_endpoint_id == self.coordinator_endpoint_id


CapabilityVerifier = Callable[
    [str, str, RouteCapabilityContext, tuple[str, ...]],
    str,
]
RevocationSource = Callable[[], tuple[str, ...]]


@dataclass(frozen=True)
class RouteAuthorityTrustSnapshot:
    public_keys: Mapping[str, str]
    revoked_identifiers: tuple[str, ...]
    generation: int
    expires_at_ms: int


TrustSnapshotSource = Callable[[int], RouteAuthorityTrustSnapshot]


class RouteAuthorityTrustStore:
    """Non-blocking cache over the existing TUF registry updater.

    The first keyset is fetched synchronously during worker startup. Subsequent
    refreshes happen on a daemon thread so PREPARE and heartbeats never wait on
    registry I/O. The last authenticated keyset remains usable only until its
    own signed expiry.
    """

    def __init__(
        self,
        registry: "TrustedModelRegistry",
        *,
        refresh_interval_ms: int = _DEFAULT_TRUST_REFRESH_MS,
        clock_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    ) -> None:
        if refresh_interval_ms <= 0:
            raise ValueError("route authority refresh interval must be positive")
        self._registry = registry
        self._refresh_interval_ms = refresh_interval_ms
        self._clock_ms = clock_ms
        self._keyset = registry.route_authorities()
        self._next_refresh_ms = int(clock_ms()) + refresh_interval_ms
        self._refresh_active = False
        self._last_refresh_error: str | None = None
        self._lock = threading.RLock()

    def _refresh(self) -> None:
        try:
            candidate = self._registry.route_authorities()
            with self._lock:
                if candidate.generation < self._keyset.generation:
                    raise ValueError("route authority generation rollback")
                self._keyset = candidate
                self._last_refresh_error = None
        except Exception as error:
            with self._lock:
                self._last_refresh_error = f"{type(error).__name__}: {error}"
        finally:
            with self._lock:
                self._next_refresh_ms = int(self._clock_ms()) + self._refresh_interval_ms
                self._refresh_active = False

    def snapshot(self, now_ms: int) -> RouteAuthorityTrustSnapshot:
        with self._lock:
            keyset = self._keyset
            if now_ms >= self._next_refresh_ms and not self._refresh_active:
                self._refresh_active = True
                threading.Thread(
                    target=self._refresh,
                    name="RouteAuthorityTrustRefresh",
                    daemon=True,
                ).start()
            public_keys = keyset.active_public_keys(now_ms)
            if not public_keys:
                raise PermissionError("route authority keyset has no currently active key")
            return RouteAuthorityTrustSnapshot(
                public_keys=public_keys,
                revoked_identifiers=keyset.revoked_identifiers,
                generation=keyset.generation,
                expires_at_ms=keyset.expires_at_ms,
            )

    @property
    def last_refresh_error(self) -> str | None:
        with self._lock:
            return self._last_refresh_error


class CapabilityRouteAuthority:
    """Authorize dynamic Fabi request agents with short-lived Biscuit grants."""

    def __init__(
        self,
        *,
        authority_public_keys: Mapping[str, str],
        revoked_identifiers: RevocationSource = tuple,
        verifier: CapabilityVerifier = verify_route_capability,
        trust_snapshots: TrustSnapshotSource | None = None,
    ) -> None:
        if not authority_public_keys and trust_snapshots is None:
            raise ValueError("at least one capability authority key is required")
        if authority_public_keys and trust_snapshots is not None:
            raise ValueError("configure static keys or a dynamic trust snapshot source")
        self._authority_public_keys = dict(authority_public_keys)
        self._revoked_identifiers = revoked_identifiers
        self._verifier = verifier
        self._trust_snapshots = trust_snapshots
        self._trust_store: RouteAuthorityTrustStore | None = None

    @classmethod
    def from_trusted_registry(
        cls,
        registry: "TrustedModelRegistry",
        *,
        refresh_interval_ms: int = _DEFAULT_TRUST_REFRESH_MS,
        clock_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        verifier: CapabilityVerifier = verify_route_capability,
    ) -> "CapabilityRouteAuthority":
        store = RouteAuthorityTrustStore(
            registry,
            refresh_interval_ms=refresh_interval_ms,
            clock_ms=clock_ms,
        )
        authority = cls(
            authority_public_keys={},
            verifier=verifier,
            trust_snapshots=store.snapshot,
        )
        authority._trust_store = store
        return authority

    def authorize_route(
        self,
        admission: SignedControlMessage | RouteAdmissionEnvelope | dict[str, object],
        *,
        caller_endpoint_id: str,
        crypto: ControlCrypto,
        now_ms: int,
    ) -> AuthorizedRoutePlan:
        if isinstance(admission, SignedControlMessage):
            raise PermissionError("dynamic route admission requires an authority capability")
        envelope = (
            admission
            if isinstance(admission, RouteAdmissionEnvelope)
            else RouteAdmissionEnvelope.model_validate(admission)
        )
        signed_plan = envelope.signed_plan
        if caller_endpoint_id != signed_plan.signer_endpoint_id:
            raise PermissionError("route caller does not match the authenticated plan signer")
        plan = verify_control_contract(
            signed_plan,
            expected_kind=ControlMessageKind.ROUTE_PLAN,
            expected_signer_endpoint_id=caller_endpoint_id,
            contract_type=RoutePlan,
            crypto=crypto,
        )
        if plan.coordinator_id != caller_endpoint_id:
            raise PermissionError("route plan coordinator does not match its authenticated signer")
        if self._trust_snapshots is None:
            public_keys = self._authority_public_keys
            revoked_identifiers = self._revoked_identifiers()
        else:
            trust = self._trust_snapshots(now_ms)
            public_keys = trust.public_keys
            revoked_identifiers = trust.revoked_identifiers
        public_key = public_keys.get(envelope.authority_key_id)
        if public_key is None:
            raise PermissionError("route capability authority key is not trusted")
        digest = route_plan_digest(signed_plan)
        context = RouteCapabilityContext(
            permit_id=envelope.permit_id,
            account_id=envelope.account_id,
            request_id=plan.request_id,
            model_swarm_id=plan.model_swarm_id,
            coordinator_endpoint_id=caller_endpoint_id,
            route_plan_digest=digest,
            epoch=plan.epoch,
            required_context_tokens=plan.required_context_tokens,
            recovery_policy=envelope.recovery_policy,
            now_ms=now_ms,
        )
        root_revocation_id = self._verifier(
            public_key,
            envelope.capability_token,
            context,
            revoked_identifiers,
        )
        return AuthorizedRoutePlan(
            plan=plan,
            signed_plan=signed_plan,
            coordinator_endpoint_id=caller_endpoint_id,
            route_plan_digest=digest,
            permit_id=envelope.permit_id,
            account_id=envelope.account_id,
            recovery_policy=envelope.recovery_policy,
            root_revocation_id=root_revocation_id,
        )

    def authorize_unbound_control(self, caller_endpoint_id: str) -> bool:
        del caller_endpoint_id
        return False
