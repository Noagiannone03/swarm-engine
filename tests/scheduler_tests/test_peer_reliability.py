"""
Unit tests for the peer-reliability backoff (Blacklist-style) layer.

Covers:
- Node ban arithmetic: exponential backoff, no-extension of an active ban,
  success reset, max cap, and the disable switch (base_sec=0).
- Routing enforcement: estimate_pipeline_latency excludes banned peers, the
  `ignore_banned` escape valve, and ban expiry.
- Round-robin serving: a pipeline with a banned node is skipped in favour of a
  healthy one, and the "all viable pipelines are banned → dispatch anyway" valve.
"""

import pytest

from scheduling.node import Node
from scheduling.request_routing import (
    RoundRobinOverFixedPipelinesRouting,
    estimate_pipeline_latency,
)

from .test_utils import build_model_info as build_model
from .test_utils import build_node, build_node_management, set_rtt_from_coords


# --------------------------------------------------------------------------- #
# Node ban arithmetic                                                         #
# --------------------------------------------------------------------------- #
def _fresh_node() -> Node:
    model = build_model(10)
    return build_node("n", model)


def test_fresh_node_is_not_banned():
    assert _fresh_node().is_banned(now=1000.0) is False


def test_failure_bans_for_base_then_expires():
    n = _fresh_node()
    n.record_request_failure(now=1000.0, base_sec=5.0, backoff_rate=2.0, max_sec=300.0)
    assert n.is_banned(now=1000.0) is True
    assert n.is_banned(now=1004.9) is True
    assert n.is_banned(now=1005.0) is False  # expiry is exclusive lower bound
    assert n.failure_streak == 1


def test_successive_episodes_escalate():
    n = _fresh_node()
    # Episode 1: base * rate**0 = 5s
    n.record_request_failure(now=1000.0, base_sec=5.0, backoff_rate=2.0, max_sec=300.0)
    assert n.banned_until == pytest.approx(1005.0)
    # Episode 2 (after expiry): base * rate**1 = 10s
    n.record_request_failure(now=1005.0, base_sec=5.0, backoff_rate=2.0, max_sec=300.0)
    assert n.banned_until == pytest.approx(1015.0)
    # Episode 3 (after expiry): base * rate**2 = 20s
    n.record_request_failure(now=1015.0, base_sec=5.0, backoff_rate=2.0, max_sec=300.0)
    assert n.banned_until == pytest.approx(1035.0)
    assert n.failure_streak == 3


def test_active_ban_is_not_extended():
    n = _fresh_node()
    n.record_request_failure(now=1000.0, base_sec=5.0, backoff_rate=2.0, max_sec=300.0)
    # A second failure while still banned must be a no-op (matches hivemind).
    n.record_request_failure(now=1002.0, base_sec=5.0, backoff_rate=2.0, max_sec=300.0)
    assert n.banned_until == pytest.approx(1005.0)
    assert n.failure_streak == 1


def test_success_clears_ban_and_resets_streak():
    n = _fresh_node()
    n.record_request_failure(now=1000.0, base_sec=5.0, backoff_rate=2.0, max_sec=300.0)
    n.record_request_failure(now=1005.0, base_sec=5.0, backoff_rate=2.0, max_sec=300.0)
    n.record_request_success()
    assert n.banned_until == 0.0
    assert n.failure_streak == 0
    assert n.is_banned(now=1005.0) is False
    # After a reset the next ban starts from base again.
    n.record_request_failure(now=2000.0, base_sec=5.0, backoff_rate=2.0, max_sec=300.0)
    assert n.banned_until == pytest.approx(2005.0)


def test_ban_duration_capped_at_max():
    n = _fresh_node()
    n.failure_streak = 10  # 5 * 2**10 = 5120s, far above the cap
    n.record_request_failure(now=1000.0, base_sec=5.0, backoff_rate=2.0, max_sec=300.0)
    assert n.banned_until == pytest.approx(1300.0)


def test_base_zero_disables_banning():
    n = _fresh_node()
    n.record_request_failure(now=1000.0, base_sec=0.0, backoff_rate=2.0, max_sec=300.0)
    assert n.is_banned(now=1000.0) is False
    assert n.failure_streak == 0


# --------------------------------------------------------------------------- #
# estimate_pipeline_latency enforcement                                       #
# --------------------------------------------------------------------------- #
def test_estimate_excludes_banned_peer():
    model = build_model(10)
    a = build_node("a", model, x=0.0, y=0.0)
    b = build_node("b", model, x=1.0, y=0.0)
    a.set_layer_allocation(0, 5)
    b.set_layer_allocation(5, 10)
    set_rtt_from_coords([a, b])
    id_to_node = {n.node_id: n for n in (a, b)}

    healthy = estimate_pipeline_latency(["a", "b"], id_to_node=id_to_node)
    assert healthy != float("inf")

    a.record_request_failure(now=1000.0, base_sec=5.0, backoff_rate=2.0, max_sec=300.0)
    # Banned -> excluded.
    assert estimate_pipeline_latency(["a", "b"], id_to_node=id_to_node, now=1000.0) == float("inf")
    # Escape valve -> finite again.
    assert estimate_pipeline_latency(
        ["a", "b"], id_to_node=id_to_node, ignore_banned=True, now=1000.0
    ) == pytest.approx(healthy)
    # After expiry -> finite again without the valve.
    assert estimate_pipeline_latency(["a", "b"], id_to_node=id_to_node, now=1006.0) == pytest.approx(
        healthy
    )


# --------------------------------------------------------------------------- #
# Round-robin serving                                                          #
# --------------------------------------------------------------------------- #
def _two_pipeline_router(num_layers=10):
    model = build_model(num_layers)
    p1a = build_node("p1a", model, x=0.0, y=0.0)
    p1b = build_node("p1b", model, x=1.0, y=0.0)
    p2a = build_node("p2a", model, x=0.0, y=1.0)
    p2b = build_node("p2b", model, x=1.0, y=1.0)
    p1a.set_layer_allocation(0, 4)
    p1b.set_layer_allocation(4, num_layers)
    p2a.set_layer_allocation(0, 3)
    p2b.set_layer_allocation(3, num_layers)
    nodes = [p1a, p1b, p2a, p2b]
    set_rtt_from_coords(nodes)
    nm = build_node_management(nodes)
    nm.activate([n.node_id for n in nodes])
    rr = RoundRobinOverFixedPipelinesRouting(nm, total_layers=num_layers)
    rr.bootstrap()
    return rr, {n.node_id: n for n in nodes}


def test_round_robin_skips_banned_pipeline():
    rr, by_id = _two_pipeline_router()
    # Ban a node in pipeline 1 -> every dispatch must pick pipeline 2.
    by_id["p1b"].record_request_failure(base_sec=60.0, backoff_rate=2.0, max_sec=300.0)
    for _ in range(3):
        node_ids, latency = rr.find_optimal_path()
        assert node_ids == ["p2a", "p2b"]
        assert latency != float("inf")


def test_round_robin_escape_valve_when_all_banned():
    rr, by_id = _two_pipeline_router()
    # Ban a node in BOTH pipelines -> no fully-healthy pipeline remains, but the
    # request must still be routed (fallback) rather than failing.
    by_id["p1b"].record_request_failure(base_sec=60.0, backoff_rate=2.0, max_sec=300.0)
    by_id["p2b"].record_request_failure(base_sec=60.0, backoff_rate=2.0, max_sec=300.0)
    node_ids, latency = rr.find_optimal_path()
    assert node_ids in (["p1a", "p1b"], ["p2a", "p2b"])
    assert latency != float("inf")


def test_round_robin_overload_still_fails_even_with_escape_valve():
    """The valve only forgives bans — a genuinely overloaded pipeline stays out."""
    rr, by_id = _two_pipeline_router()
    for nid in ("p1b", "p2b"):
        by_id[nid].current_requests = by_id[nid].max_requests  # overload, not ban
    node_ids, latency = rr.find_optimal_path()
    assert node_ids == []
    assert latency == float("inf")
