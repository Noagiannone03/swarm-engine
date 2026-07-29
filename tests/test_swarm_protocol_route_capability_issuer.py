from __future__ import annotations

import hashlib
import hmac
import secrets

import pytest

pytest.importorskip("fabi_network_native")

from backend.server.route_capabilities import (  # noqa: E402
    AuthorizedContributionPermit,
    RouteCapabilityIssuer,
    RouteCapabilityService,
)
from backend.server.route_permits import SqliteRoutePermitLedger  # noqa: E402
from fabi_network.capability import (  # noqa: E402
    RouteCapabilityContext,
    RouteRecoveryPolicy,
    verify_route_capability,
)
from swarm_protocol.contracts import (  # noqa: E402
    PathKind,
    RecoveryLevel,
    RoutePlan,
    RouteStage,
    LayerSpan,
)
from swarm_protocol.control import (  # noqa: E402
    ControlMessageKind,
    sign_control_contract,
)
from swarm_protocol.registry import RouteAuthorityKey, RouteAuthorityKeyset  # noqa: E402
from swarm_protocol.route_authority import route_plan_digest  # noqa: E402

WORKER = "11" * 32
COORDINATOR = "22" * 32
MODEL = "33" * 32
ACCOUNT = "44" * 32
PERMIT = "55" * 32


class Crypto:
    def peer_id(self):
        return COORDINATOR

    def sign_control_payload(self, payload):
        return hmac.new(b"coordinator", payload, hashlib.sha512).digest()

    def verify_control_payload(self, signer_endpoint_id, payload, signature):
        if signer_endpoint_id != COORDINATOR:
            raise ValueError("unknown signer")
        expected = hmac.new(b"coordinator", payload, hashlib.sha512).digest()
        if not hmac.compare_digest(expected, signature):
            raise ValueError("invalid signature")


def route(now_ms: int, *, context_tokens: int = 1_000) -> RoutePlan:
    return RoutePlan(
        request_id="request",
        route_id="route",
        epoch=1,
        model_swarm_id=MODEL,
        model_num_layers=4,
        prompt_tokens=context_tokens - 100,
        reserved_output_tokens=100,
        stages=(
            RouteStage(
                worker_id="worker",
                endpoint_id=WORKER,
                hosted_span=LayerSpan(start=0, end=4),
                effective_span=LayerSpan(start=0, end=4),
                path_to_next=PathKind.DIRECT,
                rounded_context_tokens=context_tokens,
                exact_kv_bytes=context_tokens * 16,
            ),
        ),
        recovery_level=RecoveryLevel.RESTARTABLE,
        coordinator_id=COORDINATOR,
        reservation_deadline_ms=now_ms + 5_000,
        plan_expires_at_ms=now_ms + 10_000,
    )


def test_issuer_binds_contribution_plan_and_tuf_key_before_native_verification():
    now = [1_000]
    private_key = secrets.token_hex(32)
    from fabi_network.capability import capability_public_key

    public_key = capability_public_key(private_key)
    key_id = hashlib.sha256(bytes.fromhex(public_key)).hexdigest()
    keyset = RouteAuthorityKeyset(
        generation=1,
        issued_at_ms=0,
        expires_at_ms=100_000,
        keys=(
            RouteAuthorityKey(
                key_id=key_id,
                public_key=public_key,
                not_before_ms=0,
                not_after_ms=100_000,
            ),
        ),
    )
    issuer = RouteCapabilityIssuer(
        private_key_hex=private_key,
        crypto=Crypto(),
        trusted_keyset=lambda: keyset,
        clock_ms=lambda: now[0],
    )
    plan = route(now[0])
    signed = sign_control_contract(
        plan,
        kind=ControlMessageKind.ROUTE_PLAN,
        crypto=Crypto(),
    )
    permit = AuthorizedContributionPermit(
        permit_id=PERMIT,
        account_id=ACCOUNT,
        request_id=plan.request_id,
        coordinator_endpoint_id=COORDINATOR,
        model_swarm_id=MODEL,
        max_context_tokens=2_000,
        recovery_policies=frozenset({RouteRecoveryPolicy.REPLAN_COLD}),
        issued_at_ms=now[0],
        expires_at_ms=20_000,
    )

    issued = issuer.issue(
        signed,
        caller_endpoint_id=COORDINATOR,
        permit=permit,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
    )
    context = RouteCapabilityContext(
        permit_id=PERMIT,
        account_id=ACCOUNT,
        request_id=plan.request_id,
        model_swarm_id=MODEL,
        coordinator_endpoint_id=COORDINATOR,
        route_plan_digest=route_plan_digest(signed),
        epoch=plan.epoch,
        authorization_generation=issued.admission.authorization_generation,
        capability_expires_at_ms=issued.expires_at_ms,
        required_context_tokens=plan.required_context_tokens,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        now_ms=now[0],
    )
    assert (
        len(
            verify_route_capability(
                public_key,
                issued.admission.capability_token,
                context,
            )
        )
        == 128
    )
    assert len(issued.root_revocation_id) == 128

    with pytest.raises(PermissionError, match="context exceeds"):
        issuer.issue(
            sign_control_contract(
                route(now[0], context_tokens=3_000),
                kind=ControlMessageKind.ROUTE_PLAN,
                crypto=Crypto(),
            ),
            caller_endpoint_id=COORDINATOR,
            permit=permit,
            recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        )
    with pytest.raises(PermissionError, match="recovery policy"):
        issuer.issue(
            signed,
            caller_endpoint_id=COORDINATOR,
            permit=permit,
            recovery_policy=RouteRecoveryPolicy.HOT_REPLICA,
        )


def test_service_persists_one_biscuit_for_an_exact_retry(tmp_path):
    now = [1_000]
    private_key = secrets.token_hex(32)
    from fabi_network.capability import capability_public_key

    public_key = capability_public_key(private_key)
    key_id = hashlib.sha256(bytes.fromhex(public_key)).hexdigest()
    issuer = RouteCapabilityIssuer(
        private_key_hex=private_key,
        crypto=Crypto(),
        trusted_keyset=lambda: RouteAuthorityKeyset(
            generation=1,
            issued_at_ms=0,
            expires_at_ms=100_000,
            keys=(
                RouteAuthorityKey(
                    key_id=key_id,
                    public_key=public_key,
                    not_before_ms=0,
                    not_after_ms=100_000,
                ),
            ),
        ),
        clock_ms=lambda: now[0],
    )
    ledger = SqliteRoutePermitLedger(
        tmp_path / "route-permits.sqlite3",
        clock_ms=lambda: now[0],
    )
    ledger.issue(
        account_id=ACCOUNT,
        request_id="request",
        coordinator_endpoint_id=COORDINATOR,
        model_swarm_id=MODEL,
        max_context_tokens=2_000,
        recovery_policies=frozenset({RouteRecoveryPolicy.REPLAN_COLD}),
        ttl_ms=20_000,
        max_active_per_account=1,
        permit_id=PERMIT,
    )
    signed = sign_control_contract(
        route(now[0]),
        kind=ControlMessageKind.ROUTE_PLAN,
        crypto=Crypto(),
    )
    service = RouteCapabilityService(ledger=ledger, issuer=issuer)

    first = service.issue(
        signed,
        permit_id=PERMIT,
        account_id=ACCOUNT,
        caller_endpoint_id=COORDINATOR,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
    )
    retry = service.issue(
        signed,
        permit_id=PERMIT,
        account_id=ACCOUNT,
        caller_endpoint_id=COORDINATOR,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
    )

    assert retry == first
    now[0] = 12_000
    refreshed_permit = ledger.keepalive_owned(
        PERMIT,
        ACCOUNT,
        ttl_ms=20_000,
        idempotency_key="long-generation-1",
    )
    refreshed = service.issue(
        signed,
        permit_id=PERMIT,
        account_id=ACCOUNT,
        caller_endpoint_id=COORDINATOR,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
    )
    assert refreshed.admission.authorization_generation == 1
    assert refreshed.expires_at_ms == refreshed_permit.expires_at_ms
    assert refreshed.admission.capability_token != first.admission.capability_token
    assert ledger.revoke(PERMIT) == (
        first.root_revocation_id,
        refreshed.root_revocation_id,
    )
    context = RouteCapabilityContext(
        permit_id=PERMIT,
        account_id=ACCOUNT,
        request_id="request",
        model_swarm_id=MODEL,
        coordinator_endpoint_id=COORDINATOR,
        route_plan_digest=route_plan_digest(signed),
        epoch=1,
        authorization_generation=refreshed.admission.authorization_generation,
        capability_expires_at_ms=refreshed.expires_at_ms,
        required_context_tokens=1_000,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        now_ms=now[0],
    )
    with pytest.raises(RuntimeError, match="revoked"):
        verify_route_capability(
            public_key,
            refreshed.admission.capability_token,
            context,
            ledger.revoke(PERMIT),
        )
