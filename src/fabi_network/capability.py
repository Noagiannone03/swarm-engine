"""Python contracts for short-lived, route-bound Biscuit capabilities.

Cryptographic parsing and authorization stay in the Rust extension.  This
module gives the rest of the Python runtime typed, testable boundaries without
silently falling back to a Python security implementation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

HashHex = str


class RouteRecoveryPolicy(str, Enum):
    BEST_EFFORT = "best_effort"
    REPLAN_COLD = "replan_cold"
    ACTIVATION_REPLAY = "activation_replay"
    RESERVED_ROUTE = "reserved_route"
    HOT_REPLICA = "hot_replica"


class RouteCapabilityClaims(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    permit_id: HashHex = Field(pattern=r"^[0-9a-f]{64}$")
    account_id: HashHex = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str = Field(min_length=1, max_length=512)
    model_swarm_id: HashHex = Field(pattern=r"^[0-9a-f]{64}$")
    coordinator_endpoint_id: HashHex = Field(pattern=r"^[0-9a-f]{64}$")
    route_plan_digest: HashHex = Field(pattern=r"^[0-9a-f]{64}$")
    epoch: int = Field(gt=0)
    authorization_generation: int = Field(default=0, ge=0)
    max_context_tokens: int = Field(gt=0)
    recovery_policy: RouteRecoveryPolicy
    issued_at_ms: int = Field(ge=0)
    expires_at_ms: int = Field(gt=0)


class RouteCapabilityContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    permit_id: HashHex = Field(pattern=r"^[0-9a-f]{64}$")
    account_id: HashHex = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str = Field(min_length=1, max_length=512)
    model_swarm_id: HashHex = Field(pattern=r"^[0-9a-f]{64}$")
    coordinator_endpoint_id: HashHex = Field(pattern=r"^[0-9a-f]{64}$")
    route_plan_digest: HashHex = Field(pattern=r"^[0-9a-f]{64}$")
    epoch: int = Field(gt=0)
    authorization_generation: int = Field(default=0, ge=0)
    capability_expires_at_ms: int = Field(gt=0)
    required_context_tokens: int = Field(gt=0)
    recovery_policy: RouteRecoveryPolicy
    now_ms: int = Field(ge=0)


def _native_module():
    try:
        import fabi_network_native
    except ImportError as error:  # pragma: no cover - release packaging
        raise RuntimeError(
            "fabi-network-native with Biscuit support is required for route capabilities"
        ) from error
    return fabi_network_native


def capability_public_key(private_key_hex: str) -> str:
    return str(_native_module().capability_public_key(private_key_hex))


def issue_route_capability(
    private_key_hex: str,
    claims: RouteCapabilityClaims,
) -> str:
    return str(
        _native_module().issue_route_capability(
            private_key_hex,
            claims.model_dump_json(),
        )
    )


def verify_route_capability(
    public_key_hex: str,
    token: str,
    context: RouteCapabilityContext,
    revoked_identifiers_hex: Iterable[str] = (),
) -> str:
    """Return the root revocation identifier after fail-closed verification."""

    return str(
        _native_module().verify_route_capability(
            public_key_hex,
            token,
            context.model_dump_json(),
            list(revoked_identifiers_hex),
        )
    )


def route_capability_root_revocation_id(public_key_hex: str, token: str) -> str:
    return str(_native_module().route_capability_root_revocation_id(public_key_hex, token))


def claims_json(claims: RouteCapabilityClaims) -> str:
    """Canonical JSON helper for non-Python issuers and diagnostic tooling."""

    return json.dumps(
        claims.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
