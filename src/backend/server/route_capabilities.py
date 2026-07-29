"""Bounded route-capability issuance for client-coordinated Fabi requests."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from pydantic import BaseModel, ConfigDict

from backend.server.route_permits import (
    AuthorizedContributionPermit,
    ClaimedRoutePlan,
    RouteCapabilityIssuance,
    RoutePermitConflict,
)
from fabi_network.capability import (
    RouteCapabilityClaims,
    RouteRecoveryPolicy,
    capability_public_key,
    issue_route_capability,
    route_capability_root_revocation_id,
)
from swarm_protocol.contracts import RoutePlan
from swarm_protocol.control import (
    ControlCrypto,
    ControlMessageKind,
    SignedControlMessage,
    verify_control_contract,
)
from swarm_protocol.route_authority import RouteAdmissionEnvelope, route_plan_digest

_MAX_PRIVATE_KEY_FILE_BYTES = 256


class IssuedRouteCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admission: RouteAdmissionEnvelope
    expires_at_ms: int
    root_revocation_id: str


@dataclass(frozen=True)
class _PreparedRouteCapability:
    envelope: SignedControlMessage
    plan: RoutePlan
    permit: AuthorizedContributionPermit
    recovery_policy: RouteRecoveryPolicy
    claims: RouteCapabilityClaims


class RoutePermitLedger(Protocol):
    """Storage contract shared by SQLite now and a future PostgreSQL authority."""

    def get_active(self, permit_id: str) -> AuthorizedContributionPermit: ...

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
    ) -> ClaimedRoutePlan: ...

    def get_issuance(
        self,
        *,
        permit_id: str,
        epoch: int,
    ) -> RouteCapabilityIssuance | None: ...

    def record_issuance(
        self,
        claim: ClaimedRoutePlan,
        *,
        authority_key_id: str,
        capability_token: str,
        root_revocation_id: str,
        recovery_policy: RouteRecoveryPolicy,
        expires_at_ms: int,
    ) -> RouteCapabilityIssuance: ...


class ActiveRouteAuthorityKeys(Protocol):
    def active_public_keys(self, now_ms: int) -> Mapping[str, str]: ...


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
        trusted_keyset: Callable[[], ActiveRouteAuthorityKeys],
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
        trusted_keyset: Callable[[], ActiveRouteAuthorityKeys],
        clock_ms: Callable[[], int] = _system_clock_ms,
    ) -> "RouteCapabilityIssuer":
        return cls(
            private_key_hex=load_route_capability_private_key(path),
            crypto=crypto,
            trusted_keyset=trusted_keyset,
            clock_ms=clock_ms,
        )

    def prepare(
        self,
        signed_plan: SignedControlMessage | dict[str, object],
        *,
        caller_endpoint_id: str,
        permit: AuthorizedContributionPermit,
        recovery_policy: RouteRecoveryPolicy,
    ) -> _PreparedRouteCapability:
        """Validate the complete ambient contract before mutating the permit ledger."""

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
        if plan.request_id != permit.request_id:
            raise PermissionError("contribution permit targets a different request")
        if caller_endpoint_id != permit.coordinator_endpoint_id:
            raise PermissionError("contribution permit targets a different request agent")
        if plan.model_swarm_id != permit.model_swarm_id:
            raise PermissionError("contribution permit targets a different model swarm")
        if plan.required_context_tokens > permit.max_context_tokens:
            raise PermissionError("route context exceeds the contribution permit")
        if recovery_policy not in permit.recovery_policies:
            raise PermissionError("route recovery policy exceeds the contribution permit")
        if (
            permit.authorization_generation == 0 and plan.reservation_deadline_ms <= now_ms
        ) or plan.plan_expires_at_ms > permit.expires_at_ms:
            raise PermissionError("route lifetime exceeds the contribution permit")

        self.validate_active_key(now_ms=now_ms)

        claims = RouteCapabilityClaims(
            permit_id=permit.permit_id,
            account_id=permit.account_id,
            request_id=plan.request_id,
            model_swarm_id=plan.model_swarm_id,
            coordinator_endpoint_id=caller_endpoint_id,
            route_plan_digest=route_plan_digest(envelope),
            epoch=plan.epoch,
            authorization_generation=permit.authorization_generation,
            max_context_tokens=permit.max_context_tokens,
            recovery_policy=recovery_policy,
            issued_at_ms=now_ms,
            expires_at_ms=permit.expires_at_ms,
        )
        return _PreparedRouteCapability(
            envelope=envelope,
            plan=plan,
            permit=permit,
            recovery_policy=recovery_policy,
            claims=claims,
        )

    def validate_active_key(self, *, now_ms: int | None = None) -> None:
        current_ms = int(self._clock_ms()) if now_ms is None else int(now_ms)
        if current_ms < 0:
            raise RuntimeError("route capability issuer clock returned a negative timestamp")
        trusted = self._trusted_keyset()
        active_keys = trusted.active_public_keys(current_ms)
        if active_keys.get(self.authority_key_id) != self._public_key_hex:
            raise PermissionError("route capability signing key is not active in the TUF registry")

    def _mint(self, prepared: _PreparedRouteCapability) -> IssuedRouteCapability:
        """Mint one validated capability; callers must persist it before returning."""

        token = issue_route_capability(self._private_key_hex, prepared.claims)
        return IssuedRouteCapability(
            admission=RouteAdmissionEnvelope(
                signed_plan=prepared.envelope,
                authority_key_id=self.authority_key_id,
                capability_token=token,
                permit_id=prepared.permit.permit_id,
                account_id=prepared.permit.account_id,
                authorization_generation=prepared.permit.authorization_generation,
                expires_at_ms=prepared.claims.expires_at_ms,
                recovery_policy=prepared.recovery_policy,
            ),
            expires_at_ms=prepared.claims.expires_at_ms,
            root_revocation_id=route_capability_root_revocation_id(
                self._public_key_hex,
                token,
            ),
        )

    def issue(
        self,
        signed_plan: SignedControlMessage | dict[str, object],
        *,
        caller_endpoint_id: str,
        permit: AuthorizedContributionPermit,
        recovery_policy: RouteRecoveryPolicy,
    ) -> IssuedRouteCapability:
        """Compatibility entry point for callers that own persistence themselves."""

        return self._mint(
            self.prepare(
                signed_plan,
                caller_endpoint_id=caller_endpoint_id,
                permit=permit,
                recovery_policy=recovery_policy,
            )
        )


class RouteCapabilityService:
    """Atomic permit claim plus retry-safe Biscuit issuance."""

    def __init__(
        self,
        *,
        ledger: RoutePermitLedger,
        issuer: RouteCapabilityIssuer,
    ) -> None:
        self._ledger = ledger
        self._issuer = issuer

    @staticmethod
    def _response(
        prepared: _PreparedRouteCapability,
        issuance: RouteCapabilityIssuance,
    ) -> IssuedRouteCapability:
        if (
            issuance.route_plan_digest != prepared.claims.route_plan_digest
            or issuance.recovery_policy != prepared.recovery_policy
            or issuance.expires_at_ms != prepared.claims.expires_at_ms
            or issuance.authorization_generation != prepared.permit.authorization_generation
        ):
            raise RoutePermitConflict(
                "permit epoch already contains a different capability contract"
            )
        return IssuedRouteCapability(
            admission=RouteAdmissionEnvelope(
                signed_plan=prepared.envelope,
                authority_key_id=issuance.authority_key_id,
                capability_token=issuance.capability_token,
                permit_id=prepared.permit.permit_id,
                account_id=prepared.permit.account_id,
                authorization_generation=issuance.authorization_generation,
                expires_at_ms=issuance.expires_at_ms,
                recovery_policy=issuance.recovery_policy,
            ),
            expires_at_ms=issuance.expires_at_ms,
            root_revocation_id=issuance.root_revocation_id,
        )

    def issue(
        self,
        signed_plan: SignedControlMessage | dict[str, object],
        *,
        permit_id: str,
        account_id: str,
        caller_endpoint_id: str,
        recovery_policy: RouteRecoveryPolicy,
    ) -> IssuedRouteCapability:
        permit = self._ledger.get_active(permit_id)
        if permit.account_id != account_id:
            raise PermissionError("route permit belongs to a different account")
        prepared = self._issuer.prepare(
            signed_plan,
            caller_endpoint_id=caller_endpoint_id,
            permit=permit,
            recovery_policy=recovery_policy,
        )
        claim = self._ledger.claim_plan(
            permit_id=permit.permit_id,
            account_id=permit.account_id,
            request_id=prepared.claims.request_id,
            coordinator_endpoint_id=prepared.claims.coordinator_endpoint_id,
            model_swarm_id=prepared.claims.model_swarm_id,
            epoch=prepared.claims.epoch,
            route_plan_digest=prepared.claims.route_plan_digest,
            required_context_tokens=prepared.plan.required_context_tokens,
            recovery_policy=prepared.recovery_policy,
        )
        existing = self._ledger.get_issuance(
            permit_id=permit.permit_id,
            epoch=prepared.claims.epoch,
            authorization_generation=permit.authorization_generation,
        )
        if existing is not None:
            return self._response(prepared, existing)

        minted = self._issuer._mint(prepared)
        persisted = self._ledger.record_issuance(
            claim,
            authority_key_id=minted.admission.authority_key_id,
            capability_token=minted.admission.capability_token,
            root_revocation_id=minted.root_revocation_id,
            recovery_policy=prepared.recovery_policy,
            expires_at_ms=minted.expires_at_ms,
        )
        return self._response(prepared, persisted)
