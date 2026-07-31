"""Authenticated HTTP authority for local Fabi Request Agents."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Mapping

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.server.contribution_gate import (
    ContributionGate,
    ContributionPermitDenied,
    ContributionStatus,
    account_hash,
)
from backend.server.route_capabilities import (
    IssuedRouteCapability,
    RouteCapabilityIssuer,
    RouteCapabilityService,
)
from backend.server.route_permits import (
    AuthorizedContributionPermit,
    RoutePermitCapacityReached,
    RoutePermitConflict,
    RoutePermitError,
    RoutePermitExpired,
    SqliteRoutePermitLedger,
    StalePermitEpoch,
)
from fabi_network.capability import RouteRecoveryPolicy
from swarm_protocol.control import SignedControlMessage
from swarm_protocol.route_authority import RouteAuthorityTrustStore

router = APIRouter(prefix="/v1/swarm", tags=["Fabi Request Agent"])


class RoutePermitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1, max_length=512)
    coordinator_endpoint_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_swarm_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_context_tokens: int = Field(gt=0)
    recovery_policies: tuple[RouteRecoveryPolicy, ...] = (RouteRecoveryPolicy.REPLAN_COLD,)
    ttl_ms: int = Field(default=120_000, gt=0, le=300_000)

    @model_validator(mode="after")
    def validate_recovery_policies(self) -> "RoutePermitRequest":
        if not self.recovery_policies:
            raise ValueError("at least one recovery policy is required")
        if len(set(self.recovery_policies)) != len(self.recovery_policies):
            raise ValueError("recovery policies must be unique")
        return self


class RoutePermitResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    permit_id: str
    request_id: str
    coordinator_endpoint_id: str
    model_swarm_id: str
    max_context_tokens: int
    recovery_policies: tuple[RouteRecoveryPolicy, ...]
    issued_at_ms: int
    expires_at_ms: int
    authorization_generation: int = Field(ge=0)

    @classmethod
    def from_permit(cls, permit: AuthorizedContributionPermit) -> "RoutePermitResponse":
        return cls(
            permit_id=permit.permit_id,
            request_id=permit.request_id,
            coordinator_endpoint_id=permit.coordinator_endpoint_id,
            model_swarm_id=permit.model_swarm_id,
            max_context_tokens=permit.max_context_tokens,
            recovery_policies=tuple(sorted(permit.recovery_policies, key=lambda item: item.value)),
            issued_at_ms=permit.issued_at_ms,
            expires_at_ms=permit.expires_at_ms,
            authorization_generation=permit.authorization_generation,
        )


class RoutePermitKeepaliveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ttl_ms: int = Field(default=120_000, gt=0, le=300_000)


class RouteCapabilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    permit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    signed_plan: SignedControlMessage
    recovery_policy: RouteRecoveryPolicy


class _IssuerTrustView:
    def __init__(self, trust_store: RouteAuthorityTrustStore) -> None:
        self._trust_store = trust_store

    def active_public_keys(self, now_ms: int) -> Mapping[str, str]:
        return self._trust_store.snapshot(now_ms).public_keys


def _bearer_credential(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    scheme, separator, credential = authorization.partition(" ")
    if separator and scheme.lower() == "bearer":
        return credential.strip() or None
    return None


def _error(
    status_code: int,
    code: str,
    message: str,
    *,
    retry_after: int | None = None,
) -> JSONResponse:
    headers = None if retry_after is None else {"Retry-After": str(retry_after)}
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "code": code,
                "message": message,
                "type": "request_agent_authority_error",
            }
        },
    )


class RequestAgentAuthority:
    """Account-scoped façade over contribution permits and Biscuit issuance."""

    def __init__(
        self,
        *,
        gate: ContributionGate,
        ledger: SqliteRoutePermitLedger,
        capabilities: RouteCapabilityService,
        scheduler_provider: Callable[[], object | None],
        model_swarm_id_provider: Callable[[], str],
        max_context_tokens_provider: Callable[[], int],
    ) -> None:
        if not gate.enabled:
            raise RuntimeError("Request Agent authority requires FABI_GATE=on")
        self.gate = gate
        self.ledger = ledger
        self.capabilities = capabilities
        self._scheduler_provider = scheduler_provider
        self._model_swarm_id_provider = model_swarm_id_provider
        self._max_context_tokens_provider = max_context_tokens_provider
        gate.bind_route_permit_ledger(ledger)

    def issue_permit(
        self,
        *,
        credential: object,
        idempotency_key: str,
        contract: RoutePermitRequest,
    ) -> RoutePermitResponse:
        identity = account_hash(credential)
        if identity is None:
            reason = "missing_credential" if not credential else "invalid_credential"
            raise ContributionPermitDenied(ContributionStatus(False, reason))
        existing = self.ledger.find_active_by_idempotency(
            account_id=identity,
            idempotency_key=idempotency_key,
            coordinator_endpoint_id=contract.coordinator_endpoint_id,
        )
        if existing is not None:
            if (
                existing.request_id != contract.request_id
                or existing.model_swarm_id != contract.model_swarm_id
                or existing.max_context_tokens != contract.max_context_tokens
                or existing.recovery_policies != frozenset(contract.recovery_policies)
                or existing.initial_ttl_ms != contract.ttl_ms
            ):
                raise RoutePermitConflict(
                    "request idempotency key was reused with a different permit contract"
                )
            return RoutePermitResponse.from_permit(existing)
        current_model = self._model_swarm_id_provider()
        if contract.model_swarm_id != current_model:
            raise PermissionError("requested model does not match this swarm authority")
        supported_context = int(self._max_context_tokens_provider())
        if supported_context <= 0:
            raise RuntimeError("swarm context capacity is not ready")
        if contract.max_context_tokens > supported_context:
            raise ValueError(
                f"requested context {contract.max_context_tokens} exceeds "
                f"the live route limit {supported_context}"
            )
        permit = self.gate.issue_route_permit(
            credential,
            self._scheduler_provider(),
            request_id=contract.request_id,
            idempotency_key=idempotency_key,
            coordinator_endpoint_id=contract.coordinator_endpoint_id,
            model_swarm_id=contract.model_swarm_id,
            max_context_tokens=contract.max_context_tokens,
            recovery_policies=frozenset(contract.recovery_policies),
            ttl_ms=contract.ttl_ms,
        )
        return RoutePermitResponse.from_permit(permit)

    def issue_capability(
        self,
        *,
        credential: object,
        contract: RouteCapabilityRequest,
    ) -> IssuedRouteCapability:
        identity = account_hash(credential)
        if identity is None:
            reason = "missing" if not credential else "invalid"
            raise PermissionError(f"{reason} account credential")
        return self.capabilities.issue(
            contract.signed_plan,
            permit_id=contract.permit_id,
            account_id=identity,
            caller_endpoint_id=contract.signed_plan.signer_endpoint_id,
            recovery_policy=contract.recovery_policy,
        )

    def keepalive(
        self,
        *,
        credential: object,
        permit_id: str,
        idempotency_key: str,
        contract: RoutePermitKeepaliveRequest,
    ) -> RoutePermitResponse:
        permit = self.gate.keepalive_route_permit(
            credential,
            self._scheduler_provider(),
            permit_id=permit_id,
            ttl_ms=contract.ttl_ms,
            idempotency_key=idempotency_key,
        )
        return RoutePermitResponse.from_permit(permit)

    def release(self, *, credential: object, permit_id: str) -> bool:
        identity = account_hash(credential)
        if identity is None:
            reason = "missing" if not credential else "invalid"
            raise PermissionError(f"{reason} account credential")
        return self.ledger.release_owned(permit_id, identity)


_authority: RequestAgentAuthority | None = None
_authority_lock = threading.RLock()


def set_request_agent_authority(authority: RequestAgentAuthority | None) -> None:
    """Install the process authority; primarily useful for application startup and tests."""

    global _authority
    with _authority_lock:
        _authority = authority


def get_request_agent_authority() -> RequestAgentAuthority | None:
    with _authority_lock:
        return _authority


def configure_request_agent_authority(
    scheduler_manage,
    gate: ContributionGate,
) -> RequestAgentAuthority | None:
    """Fail-closed environment provisioning for the public V3 authority."""

    private_key_path = os.environ.get("FABI_ROUTE_CAPABILITY_PRIVATE_KEY")
    ledger_path = os.environ.get("FABI_ROUTE_PERMIT_DB")
    if not private_key_path and not ledger_path:
        return None
    if not private_key_path or not ledger_path:
        raise RuntimeError(
            "configure both FABI_ROUTE_CAPABILITY_PRIVATE_KEY and FABI_ROUTE_PERMIT_DB"
        )
    if getattr(scheduler_manage, "swarm_v3_mode", None) != "active":
        raise RuntimeError("Request Agent authority requires FABI_SWARM_V3_MODE=active")
    scheduler = getattr(scheduler_manage, "scheduler", None)
    planner = getattr(scheduler, "swarm_v3_shadow", None)
    transport = getattr(scheduler_manage, "iroh_transport", None)
    if scheduler is None or planner is None or transport is None:
        raise RuntimeError("Request Agent authority requires the active V3 planner and Iroh")

    with _authority_lock:
        if _authority is not None:
            return _authority
        trust_store = RouteAuthorityTrustStore(planner.registry)
        issuer = RouteCapabilityIssuer.from_private_key_file(
            Path(private_key_path),
            crypto=transport,
            trusted_keyset=lambda: _IssuerTrustView(trust_store),
        )
        issuer.validate_active_key()
        ledger = SqliteRoutePermitLedger(Path(ledger_path))

        def current_model_swarm_id() -> str:
            current_scheduler = getattr(scheduler_manage, "scheduler", None)
            current_planner = getattr(current_scheduler, "swarm_v3_shadow", None)
            if current_scheduler is None or current_planner is None:
                raise RuntimeError("active V3 planner is not ready")
            return current_planner.ready_model_swarm_id(list(current_scheduler.node_manager.nodes))

        authority = RequestAgentAuthority(
            gate=gate,
            ledger=ledger,
            capabilities=RouteCapabilityService(ledger=ledger, issuer=issuer),
            scheduler_provider=lambda: getattr(scheduler_manage, "scheduler", None),
            model_swarm_id_provider=current_model_swarm_id,
            max_context_tokens_provider=scheduler_manage.max_supported_context_tokens,
        )
        set_request_agent_authority(authority)
        return authority


def _required_authority() -> RequestAgentAuthority | JSONResponse:
    authority = get_request_agent_authority()
    if authority is None:
        return _error(
            503,
            "request_agent_authority_unavailable",
            "The V3 Request Agent authority is not configured.",
            retry_after=1,
        )
    return authority


@router.post("/route-permits")
def create_route_permit(
    contract: RoutePermitRequest,
    raw_request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    authority = _required_authority()
    if isinstance(authority, JSONResponse):
        return authority
    if idempotency_key is None:
        return _error(
            400,
            "idempotency_key_required",
            "Idempotency-Key is required for safe permit retries.",
        )
    try:
        return authority.issue_permit(
            credential=_bearer_credential(raw_request),
            idempotency_key=idempotency_key,
            contract=contract,
        )
    except ContributionPermitDenied as error:
        reason = error.status.reason
        if reason == "capacity_reached":
            return _error(429, "contribution_capacity_reached", str(error), retry_after=1)
        if reason in {"swarm_not_ready", "admission_unavailable"}:
            return _error(503, reason, str(error), retry_after=1)
        if reason in {"missing_credential", "invalid_credential"}:
            return _error(401, reason, str(error))
        return _error(403, "contribution_required", str(error))
    except RoutePermitConflict as error:
        return _error(422, "idempotency_key_reused", str(error))
    except RoutePermitCapacityReached as error:
        return _error(429, "contribution_capacity_reached", str(error), retry_after=1)
    except PermissionError as error:
        return _error(403, "model_not_authorized", str(error))
    except (ValueError, StalePermitEpoch) as error:
        return _error(422, "invalid_route_permit", str(error))
    except RuntimeError as error:
        return _error(503, "route_authority_unavailable", str(error), retry_after=1)


@router.post("/route-capabilities")
def create_route_capability(contract: RouteCapabilityRequest, raw_request: Request):
    authority = _required_authority()
    if isinstance(authority, JSONResponse):
        return authority
    try:
        return authority.issue_capability(
            credential=_bearer_credential(raw_request),
            contract=contract,
        )
    except PermissionError as error:
        message = str(error)
        status = 401 if "credential" in message else 403
        return _error(status, "route_capability_denied", message)
    except RoutePermitExpired as error:
        return _error(410, "route_permit_expired", str(error))
    except (RoutePermitConflict, StalePermitEpoch) as error:
        return _error(409, "route_epoch_conflict", str(error))
    except (RoutePermitError, ValueError) as error:
        return _error(422, "invalid_route_capability", str(error))
    except RuntimeError as error:
        return _error(503, "route_authority_unavailable", str(error), retry_after=1)


@router.post("/route-permits/{permit_id}/keepalive")
def keepalive_route_permit(
    permit_id: str,
    contract: RoutePermitKeepaliveRequest,
    raw_request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    authority = _required_authority()
    if isinstance(authority, JSONResponse):
        return authority
    if idempotency_key is None:
        return _error(
            400,
            "idempotency_key_required",
            "Idempotency-Key is required for a route permit keepalive.",
        )
    try:
        return authority.keepalive(
            credential=_bearer_credential(raw_request),
            permit_id=permit_id,
            idempotency_key=idempotency_key,
            contract=contract,
        )
    except ContributionPermitDenied as error:
        reason = error.status.reason
        if reason in {"swarm_not_ready", "admission_unavailable"}:
            return _error(503, reason, str(error), retry_after=1)
        if reason in {"missing_credential", "invalid_credential"}:
            return _error(401, reason, str(error))
        return _error(403, "contribution_required", str(error))
    except PermissionError as error:
        return _error(404, "route_permit_not_found", str(error))
    except RoutePermitExpired as error:
        return _error(410, "route_permit_expired", str(error))
    except RoutePermitConflict as error:
        return _error(409, "route_permit_keepalive_conflict", str(error))
    except RoutePermitError as error:
        return _error(404, "route_permit_not_found", str(error))
    except ValueError as error:
        return _error(422, "invalid_route_permit_keepalive", str(error))
    except RuntimeError as error:
        return _error(503, "route_authority_unavailable", str(error), retry_after=1)


@router.delete("/route-permits/{permit_id}")
def release_route_permit(permit_id: str, raw_request: Request):
    authority = _required_authority()
    if isinstance(authority, JSONResponse):
        return authority
    try:
        released = authority.release(
            credential=_bearer_credential(raw_request),
            permit_id=permit_id,
        )
    except PermissionError as error:
        return _error(401, "invalid_credential", str(error))
    except ValueError as error:
        return _error(422, "invalid_route_permit", str(error))
    if not released:
        return _error(404, "route_permit_not_found", "No active route permit was found.")
    return JSONResponse(status_code=200, content={"released": True})
