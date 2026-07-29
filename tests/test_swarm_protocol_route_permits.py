from __future__ import annotations

import os
import stat
import threading

import pytest

from backend.server.route_permits import (
    RoutePermitCapacityReached,
    RoutePermitConflict,
    RoutePermitError,
    RoutePermitExpired,
    SqliteRoutePermitLedger,
    StalePermitEpoch,
)
from fabi_network.capability import RouteRecoveryPolicy

ACCOUNT = "11" * 32
COORDINATOR = "22" * 32
MODEL = "33" * 32
PERMIT = "44" * 32
DIGEST = "55" * 32


def issue(ledger, *, request_id="request", permit_id=PERMIT, capacity=1):
    return ledger.issue(
        account_id=ACCOUNT,
        request_id=request_id,
        coordinator_endpoint_id=COORDINATOR,
        model_swarm_id=MODEL,
        max_context_tokens=16_384,
        recovery_policies=frozenset(
            {
                RouteRecoveryPolicy.REPLAN_COLD,
                RouteRecoveryPolicy.ACTIVATION_REPLAY,
            }
        ),
        ttl_ms=60_000,
        max_active_per_account=capacity,
        permit_id=permit_id,
    )


def claim(ledger, permit, *, epoch=1, digest=DIGEST):
    return ledger.claim_plan(
        permit_id=permit.permit_id,
        account_id=permit.account_id,
        request_id=permit.request_id,
        coordinator_endpoint_id=permit.coordinator_endpoint_id,
        model_swarm_id=permit.model_swarm_id,
        epoch=epoch,
        route_plan_digest=digest,
        required_context_tokens=12_000,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
    )


def test_permit_issue_is_idempotent_and_quota_is_atomic(tmp_path):
    ledger = SqliteRoutePermitLedger(tmp_path / "permits.sqlite3", clock_ms=lambda: 1_000)
    first = issue(ledger)
    assert issue(ledger) == first

    with pytest.raises(RoutePermitCapacityReached):
        issue(ledger, request_id="other", permit_id="66" * 32)

    ledger.release(first.permit_id)
    assert issue(ledger, request_id="other", permit_id="66" * 32).request_id == "other"


def test_epoch_cas_is_idempotent_but_never_forks_or_moves_backwards(tmp_path):
    ledger = SqliteRoutePermitLedger(tmp_path / "permits.sqlite3", clock_ms=lambda: 1_000)
    permit = issue(ledger)
    first = claim(ledger, permit)
    assert claim(ledger, permit) == first

    with pytest.raises(RoutePermitConflict, match="two different"):
        claim(ledger, permit, digest="77" * 32)
    second = claim(ledger, permit, epoch=2, digest="88" * 32)
    assert second.epoch == 2
    with pytest.raises(StalePermitEpoch):
        claim(ledger, permit, epoch=1)


def test_revocation_returns_every_still_live_capability(tmp_path):
    ledger = SqliteRoutePermitLedger(tmp_path / "permits.sqlite3", clock_ms=lambda: 1_000)
    permit = issue(ledger)
    first = claim(ledger, permit)
    ledger.record_issuance(
        first,
        authority_key_id="dd" * 32,
        capability_token="first-token",
        root_revocation_id="aa" * 64,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        expires_at_ms=20_000,
    )
    second = claim(ledger, permit, epoch=2, digest="99" * 32)
    ledger.record_issuance(
        second,
        authority_key_id="dd" * 32,
        capability_token="second-token",
        root_revocation_id="bb" * 64,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        expires_at_ms=30_000,
    )
    assert ledger.revoke(permit.permit_id) == ("aa" * 64, "bb" * 64)
    with pytest.raises(RoutePermitExpired):
        claim(ledger, permit, epoch=3, digest="cc" * 32)


def test_two_connections_cannot_both_consume_the_last_account_slot(tmp_path):
    path = tmp_path / "permits.sqlite3"
    first = SqliteRoutePermitLedger(path, clock_ms=lambda: 1_000)
    second = SqliteRoutePermitLedger(path, clock_ms=lambda: 1_000)
    barrier = threading.Barrier(2)
    outcomes = []

    def attempt(ledger, request_id, permit_id):
        barrier.wait()
        try:
            ledger.issue(
                account_id=ACCOUNT,
                request_id=request_id,
                coordinator_endpoint_id=COORDINATOR,
                model_swarm_id=MODEL,
                max_context_tokens=16_384,
                recovery_policies=frozenset({RouteRecoveryPolicy.REPLAN_COLD}),
                ttl_ms=60_000,
                max_active_per_account=1,
                permit_id=permit_id,
            )
            outcomes.append("issued")
        except RoutePermitCapacityReached:
            outcomes.append("full")

    threads = (
        threading.Thread(target=attempt, args=(first, "one", "aa" * 32)),
        threading.Thread(target=attempt, args=(second, "two", "bb" * 32)),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert sorted(outcomes) == ["full", "issued"]


def test_capability_retry_returns_the_first_persisted_token(tmp_path):
    ledger = SqliteRoutePermitLedger(tmp_path / "permits.sqlite3", clock_ms=lambda: 1_000)
    permit = issue(ledger)
    route_claim = claim(ledger, permit)
    first = ledger.record_issuance(
        route_claim,
        authority_key_id="dd" * 32,
        capability_token="first-token",
        root_revocation_id="aa" * 64,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        expires_at_ms=20_000,
    )
    retry = ledger.record_issuance(
        route_claim,
        authority_key_id="dd" * 32,
        capability_token="discarded-retry-token",
        root_revocation_id="bb" * 64,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        expires_at_ms=20_000,
    )

    assert retry == first
    assert retry.capability_token == "first-token"
    assert ledger.get_issuance(permit_id=permit.permit_id, epoch=1) == first


def test_expiration_releases_capacity_and_ledger_is_private(tmp_path):
    now = [1_000]
    path = tmp_path / "permits.sqlite3"
    ledger = SqliteRoutePermitLedger(path, clock_ms=lambda: now[0])
    issue(ledger)

    now[0] = 61_001
    replacement = issue(
        ledger,
        request_id="replacement",
        permit_id="ee" * 32,
    )

    assert replacement.request_id == "replacement"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_account_scoped_release_cannot_mutate_another_accounts_permit(tmp_path):
    ledger = SqliteRoutePermitLedger(tmp_path / "permits.sqlite3", clock_ms=lambda: 1_000)
    permit = issue(ledger)

    assert ledger.active_count(ACCOUNT) == 1
    assert ledger.release_owned(permit.permit_id, "ff" * 32) is False
    assert ledger.active_count(ACCOUNT) == 1
    assert ledger.release_owned(permit.permit_id, ACCOUNT) is True
    assert ledger.active_count(ACCOUNT) == 0


def test_permit_keepalive_is_owned_monotone_and_idempotent(tmp_path):
    now = [1_000]
    ledger = SqliteRoutePermitLedger(
        tmp_path / "permits.sqlite3",
        clock_ms=lambda: now[0],
    )
    permit = issue(ledger)

    now[0] = 20_000
    first = ledger.keepalive_owned(
        permit.permit_id,
        ACCOUNT,
        ttl_ms=60_000,
        idempotency_key="keepalive-0",
    )
    assert first.authorization_generation == 1
    assert first.expires_at_ms == 80_000
    assert (
        ledger.keepalive_owned(
            permit.permit_id,
            ACCOUNT,
            ttl_ms=60_000,
            idempotency_key="keepalive-0",
        )
        == first
    )
    with pytest.raises(RoutePermitConflict, match="different TTL"):
        ledger.keepalive_owned(
            permit.permit_id,
            ACCOUNT,
            ttl_ms=30_000,
            idempotency_key="keepalive-0",
        )
    with pytest.raises(RoutePermitError):
        ledger.keepalive_owned(
            permit.permit_id,
            "ff" * 32,
            ttl_ms=60_000,
            idempotency_key="foreign",
        )

    now[0] = 30_000
    second = ledger.keepalive_owned(
        permit.permit_id,
        ACCOUNT,
        ttl_ms=60_000,
        idempotency_key="keepalive-1",
    )
    assert second.authorization_generation == 2
    assert second.expires_at_ms == 90_000


def test_capability_refresh_generation_preserves_initial_issuance(tmp_path):
    now = [1_000]
    ledger = SqliteRoutePermitLedger(
        tmp_path / "permits.sqlite3",
        clock_ms=lambda: now[0],
    )
    permit = issue(ledger)
    initial_claim = claim(ledger, permit)
    initial = ledger.record_issuance(
        initial_claim,
        authority_key_id="dd" * 32,
        capability_token="initial-token",
        root_revocation_id="aa" * 64,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        expires_at_ms=20_000,
    )
    refreshed_permit = ledger.keepalive_owned(
        permit.permit_id,
        ACCOUNT,
        ttl_ms=60_000,
        idempotency_key="refresh",
    )
    refreshed_claim = claim(ledger, refreshed_permit)
    refreshed = ledger.record_issuance(
        refreshed_claim,
        authority_key_id="dd" * 32,
        capability_token="refreshed-token",
        root_revocation_id="bb" * 64,
        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
        expires_at_ms=refreshed_permit.expires_at_ms,
    )

    assert initial.authorization_generation == 0
    assert refreshed.authorization_generation == 1
    assert ledger.get_issuance(permit_id=permit.permit_id, epoch=1) == initial
    assert (
        ledger.get_issuance(
            permit_id=permit.permit_id,
            epoch=1,
            authorization_generation=1,
        )
        == refreshed
    )
    assert ledger.revoke(permit.permit_id) == ("aa" * 64, "bb" * 64)


def test_two_connections_serialize_permit_keepalive_generations(tmp_path):
    path = tmp_path / "permits.sqlite3"
    first = SqliteRoutePermitLedger(path, clock_ms=lambda: 1_000)
    permit = issue(first)
    second = SqliteRoutePermitLedger(path, clock_ms=lambda: 1_000)
    barrier = threading.Barrier(2)
    generations = []

    def renew(ledger, key):
        barrier.wait()
        generations.append(
            ledger.keepalive_owned(
                permit.permit_id,
                ACCOUNT,
                ttl_ms=60_000,
                idempotency_key=key,
            ).authorization_generation
        )

    threads = (
        threading.Thread(target=renew, args=(first, "parallel-1")),
        threading.Thread(target=renew, args=(second, "parallel-2")),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert sorted(generations) == [1, 2]
    assert first.get_active(permit.permit_id).authorization_generation == 2
