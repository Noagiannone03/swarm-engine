from __future__ import annotations

import secrets
import time

import pytest

pytest.importorskip("fabi_network_native")

from fabi_network.capability import (  # noqa: E402
    RouteCapabilityClaims,
    RouteCapabilityContext,
    RouteRecoveryPolicy,
    capability_public_key,
    issue_route_capability,
    verify_route_capability,
)


def test_native_biscuit_capability_roundtrip_binding_expiry_and_revocation():
    now_ms = time.time_ns() // 1_000_000
    private_key = secrets.token_hex(32)
    public_key = capability_public_key(private_key)
    claims = RouteCapabilityClaims(
        permit_id=secrets.token_hex(32),
        account_id=secrets.token_hex(32),
        request_id="request-native-wheel",
        model_swarm_id=secrets.token_hex(32),
        coordinator_endpoint_id=secrets.token_hex(32),
        route_plan_digest=secrets.token_hex(32),
        epoch=7,
        max_context_tokens=32_768,
        recovery_policy=RouteRecoveryPolicy.ACTIVATION_REPLAY,
        issued_at_ms=now_ms,
        expires_at_ms=now_ms + 60_000,
    )
    token = issue_route_capability(private_key, claims)
    context = RouteCapabilityContext(
        permit_id=claims.permit_id,
        account_id=claims.account_id,
        request_id=claims.request_id,
        model_swarm_id=claims.model_swarm_id,
        coordinator_endpoint_id=claims.coordinator_endpoint_id,
        route_plan_digest=claims.route_plan_digest,
        epoch=claims.epoch,
        authorization_generation=claims.authorization_generation,
        capability_expires_at_ms=claims.expires_at_ms,
        required_context_tokens=12_220 + 4_096,
        recovery_policy=claims.recovery_policy,
        now_ms=now_ms,
    )

    root_revocation_id = verify_route_capability(public_key, token, context)
    assert len(root_revocation_id) == 128

    with pytest.raises(RuntimeError, match="does not authorize"):
        verify_route_capability(
            public_key,
            token,
            context.model_copy(update={"route_plan_digest": secrets.token_hex(32)}),
        )
    with pytest.raises(RuntimeError, match="revoked"):
        verify_route_capability(
            public_key,
            token,
            context,
            (root_revocation_id,),
        )
    with pytest.raises(RuntimeError, match="expired"):
        verify_route_capability(
            public_key,
            token,
            context.model_copy(update={"now_ms": claims.expires_at_ms}),
        )
