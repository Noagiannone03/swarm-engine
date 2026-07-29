"""Bounded route-capability issuance for client-coordinated Fabi requests."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from fabi_network.capability import (
    RouteCapabilityClaims,
    RouteRecoveryPolicy,
    capability_public_key,
    issue_route_capability,
)
from swarm_protocol.contracts import RoutePlan
from swarm_protocol.control import (
    ControlCrypto,
    ControlMessageKind,
    SignedControlMessage,
    verify_control_contract,
)
from swarm_protocol.registry import RouteAuthorityKeyset
from swarm_protocol.route_authority import RouteAdmissionEnvelope, route_plan_digest

_MAX_PRIVATE_KEY_FILE_BYTES = 256


@dataclass(frozen=True)
class AuthorizedContributionPermit:
    """Atomic contribution/quota decision made before capability issuance."""

    permit_id: str
    account_id: str
    model_swarm_id: str
    max_context_tokens: int
    recovery_policies: frozenset[RouteRecoveryPolicy]
    expires_at_ms: int


class IssuedRouteCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admission: RouteAdmissionEnvelope
    expires_at_ms: int


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


def load_route_capability_private_key(path: Path) -> str:
    """Load an owner-only Ed25519 seed without ever logging its contents."""

    if os.name != "nt":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise PermissionError(f"route capability private key must be owner-only: {path}")
    with path.open("rb") as source:
        payload = source.read(_MAX_PRIVATE_KEY_FILE_BYTES + 1)
    if len(payload) > _MAX_PRIVATE_KEY_FILE_BYTES:
        raise ValueError("route capability private key file is too large")
    private_key = payload.rstrip(b"\r\n").decode("ascii")
    if (
        len(private_key) != 64
        or private_key.lower() != private_key
        or any(character not in "0123456789abcdef" for character in private_key)
    ):
        raise ValueError("route capability private key must be a 32-byte lowercase hex seed")
    return private_key


class RouteCapabilityIssuer:
    """Issue a Biscuit only after all ambient service checks have passed."""

    def __init__(
        self,
        *,
        private_key_hex: str,
        crypto: ControlCrypto,
        trusted_keyset: Callable[[], RouteAuthorityKeyset],
        clock_ms: Callable[[], int] = _system_clock_ms,
    ) -> None:
        self._private_key_hex = private_key_hex
        self._public_key_hex = capability_public_key(private_key_hex)
        self.authority_key_id = hashlib.sha256(bytes.fromhex(self._public_key_hex)).hexdigest()
        self._crypto = crypto
        self._trusted_keyset = trusted_keyset
        self._clock_ms = clock_ms

    @classmethod
    def from_private_key_file(
        cls,
        path: Path,
        *,
        crypto: ControlCrypto,
        trusted_keyset: Callable[[], RouteAuthorityKeyset],
        clock_ms: Callable[[], int] = _system_clock_ms,
    ) -> "RouteCapabilityIssuer":
        return cls(
            private_key_hex=load_route_capability_private_key(path),
            crypto=crypto,
            trusted_keyset=trusted_keyset,
            clock_ms=clock_ms,
        )

    def issue(
        self,
        signed_plan: SignedControlMessage | dict[str, object],
        *,
        caller_endpoint_id: str,
        permit: AuthorizedContributionPermit,
        recovery_policy: RouteRecoveryPolicy,
    ) -> IssuedRouteCapability:
        now_ms = int(self._clock_ms())
        if now_ms < 0:
            raise RuntimeError("route capability issuer clock returned a negative timestamp")
        envelope = (
            signed_plan
            if isinstance(signed_plan, SignedControlMessage)
            else SignedControlMessage.model_validate(signed_plan)
        )
        if envelope.signer_endpoint_id != caller_endpoint_id:
            raise PermissionError("plan signer does not match the authenticated request agent")
        plan = verify_control_contract(
            envelope,
            expected_kind=ControlMessageKind.ROUTE_PLAN,
            expected_signer_endpoint_id=caller_endpoint_id,
            contract_type=RoutePlan,
            crypto=self._crypto,
        )
        if plan.coordinator_id != caller_endpoint_id:
            raise PermissionError("route coordinator does not match its plan signer")
        if plan.model_swarm_id != permit.model_swarm_id:
            raise PermissionError("contribution permit targets a different model swarm")
        if plan.required_context_tokens > permit.max_context_tokens:
            raise PermissionError("route context exceeds the contribution permit")
        if recovery_policy not in permit.recovery_policies:
            raise PermissionError("route recovery policy exceeds the contribution permit")
        if plan.reservation_deadline_ms <= now_ms or plan.plan_expires_at_ms > permit.expires_at_ms:
            raise PermissionError("route lifetime exceeds the contribution permit")

        trusted = self._trusted_keyset()
        active_keys = trusted.active_public_keys(now_ms)
        if active_keys.get(self.authority_key_id) != self._public_key_hex:
            raise PermissionError("route capability signing key is not active in the TUF registry")

        claims = RouteCapabilityClaims(
            permit_id=permit.permit_id,
            account_id=permit.account_id,
            request_id=plan.request_id,
            model_swarm_id=plan.model_swarm_id,
            coordinator_endpoint_id=caller_endpoint_id,
            route_plan_digest=route_plan_digest(envelope),
            epoch=plan.epoch,
            max_context_tokens=permit.max_context_tokens,
            recovery_policy=recovery_policy,
            issued_at_ms=now_ms,
            expires_at_ms=plan.plan_expires_at_ms,
        )
        token = issue_route_capability(self._private_key_hex, claims)
        return IssuedRouteCapability(
            admission=RouteAdmissionEnvelope(
                signed_plan=envelope,
                authority_key_id=self.authority_key_id,
                capability_token=token,
                permit_id=permit.permit_id,
                account_id=permit.account_id,
                recovery_policy=recovery_policy,
            ),
            expires_at_ms=claims.expires_at_ms,
        )
