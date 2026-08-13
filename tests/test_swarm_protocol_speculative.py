import pytest
from pydantic import ValidationError

from swarm_protocol import (
    SpeculativeSampling,
    SpeculativeStageMetrics,
    SpeculativeStrategy,
    SpeculativeVerifyResponse,
    SpeculativeVerifyWindow,
    SpeculativeWindowFence,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def window(*, window_id=1, epoch=7, route_id="route-a", digest=DIGEST_A, **updates):
    values = {
        "request_id": "request-a",
        "route_id": route_id,
        "epoch": epoch,
        "route_plan_digest": digest,
        "window_id": window_id,
        "base_committed_position": 100,
        "input_position": 100,
        "starts_epoch": True,
        "prior_boundary_prediction": None,
        "input_tokens": (41, 42, 43, 44),
        "proposal_tokens": (42, 43, 44),
        "strategy": SpeculativeStrategy.NGRAM_SUFFIX,
        "proposer_id": "mesh-longest-suffix",
        "proposer_version": "0.75.1",
        "sampling": SpeculativeSampling(seed=9, temperature=0),
        "reserved_context_tokens": 32_768,
    }
    values.update(updates)
    return SpeculativeVerifyWindow(**values)


def response(candidate: SpeculativeVerifyWindow, **updates):
    values = {
        "request_id": candidate.request_id,
        "route_id": candidate.route_id,
        "epoch": candidate.epoch,
        "route_plan_digest": candidate.route_plan_digest,
        "window_id": candidate.window_id,
        "input_position": candidate.input_position,
        "input_token_count": len(candidate.input_tokens),
        "verified_position": candidate.input_position + len(candidate.input_tokens),
        "predicted_tokens": (42, 43, 44, 45),
        "metrics": (
            SpeculativeStageMetrics(
                stage_index=0,
                compute_us=500,
                wait_us=2_000,
                activation_bytes=4096,
            ),
            SpeculativeStageMetrics(
                stage_index=1,
                compute_us=700,
                wait_us=1_000,
                activation_bytes=0,
            ),
        ),
    }
    values.update(updates)
    return SpeculativeVerifyResponse(**values)


def test_window_binds_exact_route_sampling_layout_and_context() -> None:
    candidate = window()
    candidate.verify_response(response(candidate))

    with pytest.raises(ValidationError, match="layout"):
        window(input_tokens=(42, 43, 44))
    with pytest.raises(ValidationError, match="reserved context"):
        window(input_position=32_766)
    with pytest.raises(ValidationError, match="target-only"):
        window(strategy=SpeculativeStrategy.TARGET_ONLY)
    with pytest.raises(ValidationError, match="cannot carry"):
        window(prior_boundary_prediction=42)
    with pytest.raises(ValidationError, match="requires"):
        window(
            starts_epoch=False,
            input_tokens=(42, 43, 44),
            prior_boundary_prediction=None,
        )


def test_response_requires_complete_predictions_and_exact_native_position() -> None:
    candidate = window()
    with pytest.raises(ValidationError, match="cover"):
        response(candidate, predicted_tokens=(42, 43, 44))
    with pytest.raises(ValidationError, match="position"):
        response(candidate, verified_position=999)
    with pytest.raises(ValidationError, match="sorted and unique"):
        response(
            candidate,
            metrics=(
                SpeculativeStageMetrics(stage_index=1, compute_us=1, wait_us=1, activation_bytes=1),
                SpeculativeStageMetrics(stage_index=0, compute_us=1, wait_us=1, activation_bytes=1),
            ),
        )


def test_window_rejects_substituted_route_epoch_digest_and_wire_overflow() -> None:
    candidate = window()
    for changed in (
        {"route_id": "route-b"},
        {"epoch": 8},
        {"route_plan_digest": DIGEST_B},
        {"window_id": 2},
    ):
        with pytest.raises(ValueError, match="fenced window"):
            candidate.verify_response(response(candidate, **changed))

    tiny_budget = window(max_response_bytes=1)
    with pytest.raises(ValueError, match="wire budget"):
        tiny_budget.verify_response(response(tiny_budget))


def test_fence_accepts_each_current_response_once_and_drops_old_epoch() -> None:
    fence = SpeculativeWindowFence()
    first = window(window_id=1)
    fence.admit(first)
    assert fence.pending_count(first.request_id) == 1
    assert fence.accept_response(response(first)) == first
    assert fence.pending_count(first.request_id) == 0
    with pytest.raises(ValueError, match="duplicate"):
        fence.accept_response(response(first))

    old = window(window_id=2)
    replacement = window(window_id=3, epoch=8, route_id="route-b", digest=DIGEST_B)
    fence.admit(old)
    fence.admit(replacement)
    assert fence.pending_count(old.request_id) == 1
    with pytest.raises(ValueError, match="stale"):
        fence.accept_response(response(old))
    assert fence.accept_response(response(replacement)) == replacement


def test_fence_rejects_window_id_replay_and_same_epoch_route_change() -> None:
    fence = SpeculativeWindowFence()
    fence.admit(window(window_id=5))
    with pytest.raises(ValueError, match="strictly increasing"):
        fence.admit(window(window_id=5))
    with pytest.raises(ValueError, match="without a new epoch"):
        fence.admit(window(window_id=6, route_id="route-b", digest=DIGEST_B))

    fence.fence("request-a", newer_epoch=8)
    with pytest.raises(ValueError, match="stale"):
        fence.accept_response(response(window(window_id=5)))

    replacement = window(window_id=6, epoch=8, route_id="route-b", digest=DIGEST_B)
    fence.admit(replacement)
    assert fence.accept_response(response(replacement)) == replacement


def test_lifecycle_fence_is_idempotent_and_retirement_rejects_late_work() -> None:
    fence = SpeculativeWindowFence()
    candidate = window(window_id=1)
    fence.admit(candidate)

    assert fence.fence_at_least(candidate.request_id, newer_epoch=8)
    assert not fence.fence_at_least(candidate.request_id, newer_epoch=8)
    assert fence.pending_count(candidate.request_id) == 0
    with pytest.raises(ValueError, match="stale"):
        fence.accept_response(response(candidate))

    replacement = window(
        window_id=2,
        epoch=8,
        route_id="route-b",
        digest=DIGEST_B,
    )
    fence.admit(replacement)
    fence.retire(candidate.request_id)
    assert fence.pending_count(candidate.request_id) == 0
    with pytest.raises(ValueError, match="retired"):
        fence.admit(
            window(
                window_id=3,
                epoch=9,
                route_id="route-c",
                digest="c" * 64,
            )
        )


def test_settlement_commits_exact_target_prefix_and_mismatch_correction() -> None:
    candidate = window()
    full = candidate.settlement_plan(response(candidate), max_commit_tokens=8)
    assert full.accepted_proposal_tokens == 3
    assert full.committed_tokens == (42, 43, 44, 45)
    assert not full.rejected
    assert full.next_boundary_prediction is None

    mismatch = candidate.settlement_plan(
        response(candidate, predicted_tokens=(42, 99, 100, 101)),
        max_commit_tokens=8,
    )
    assert mismatch.accepted_proposal_tokens == 1
    assert mismatch.committed_tokens == (42, 99)
    assert mismatch.rejected


def test_settlement_chains_continuation_boundary_without_duplicate_bonus() -> None:
    first = window()
    deferred = first.settlement_plan(
        response(first),
        max_commit_tokens=8,
        defer_full_accept_bonus=True,
    )
    assert deferred.committed_tokens == (42, 43, 44)
    assert deferred.next_boundary_prediction == 45

    continuation = window(
        window_id=2,
        base_committed_position=103,
        input_position=104,
        starts_epoch=False,
        prior_boundary_prediction=deferred.next_boundary_prediction,
        input_tokens=(46, 47),
        proposal_tokens=(45, 46),
    )
    continuation_response = response(
        continuation,
        predicted_tokens=(46, 47),
    )
    plan = continuation.settlement_plan(continuation_response, max_commit_tokens=8)
    assert continuation.target_predictions(continuation_response) == (45, 46, 47)
    assert plan.committed_tokens == (45, 46, 47)
    assert plan.accepted_proposal_tokens == 2


def test_settlement_honors_stop_and_output_limit_before_publication() -> None:
    stop_window = window(
        sampling=SpeculativeSampling(seed=9, temperature=0, stop_token_ids=(43,)),
    )
    stopped = stop_window.settlement_plan(response(stop_window), max_commit_tokens=8)
    assert stopped.committed_tokens == (42, 43)
    assert stopped.accepted_proposal_tokens == 2
    assert stopped.reached_stop

    mismatch_stop_window = window(
        sampling=SpeculativeSampling(seed=9, temperature=0, stop_token_ids=(99,)),
    )
    mismatch_stop = mismatch_stop_window.settlement_plan(
        response(mismatch_stop_window, predicted_tokens=(42, 99, 100, 101)),
        max_commit_tokens=8,
    )
    assert mismatch_stop.committed_tokens == (42, 99)
    assert mismatch_stop.rejected
    assert mismatch_stop.reached_stop

    bonus_stop_window = window(
        sampling=SpeculativeSampling(seed=9, temperature=0, stop_token_ids=(45,)),
    )
    bonus_stop = bonus_stop_window.settlement_plan(
        response(bonus_stop_window),
        max_commit_tokens=8,
        defer_full_accept_bonus=True,
    )
    assert bonus_stop.committed_tokens == (42, 43, 44, 45)
    assert bonus_stop.reached_stop
    assert bonus_stop.next_boundary_prediction is None

    limited = window().settlement_plan(response(window()), max_commit_tokens=2)
    assert limited.committed_tokens == (42, 43)
    assert limited.accepted_proposal_tokens == 2
    assert limited.reached_output_limit
    with pytest.raises(ValueError, match="budget"):
        window().settlement_plan(response(window()), max_commit_tokens=0)


def test_fence_consumes_only_after_successful_durable_settlement() -> None:
    fence = SpeculativeWindowFence()
    candidate = window()
    reply = response(candidate)
    fence.admit(candidate)
    attempts = []

    def failing_settlement(admitted, verified):
        attempts.append((admitted.window_id, verified.window_id))
        raise RuntimeError("durable store unavailable")

    with pytest.raises(RuntimeError, match="durable store"):
        fence.settle_response(reply, failing_settlement)
    assert fence.pending_count(candidate.request_id) == 1

    plan = fence.settle_response(
        reply,
        lambda admitted, verified: admitted.settlement_plan(
            verified,
            max_commit_tokens=8,
        ),
    )
    assert plan.committed_tokens == (42, 43, 44, 45)
    assert fence.pending_count(candidate.request_id) == 0


def test_durable_bridge_commits_before_return_and_checks_exact_boundary() -> None:
    fence = SpeculativeWindowFence()
    candidate = window(base_committed_position=0, input_position=100)
    reply = response(candidate)
    fence.admit(candidate)
    durable = {candidate.request_id: []}
    observations = []

    def committed_position(request_id):
        return len(durable[request_id]) if request_id in durable else None

    def commit_tokens(plan):
        observations.append(("commit", plan.committed_tokens))
        durable[plan.request_id].extend(plan.committed_tokens)

    plan = fence.settle_response_durably(
        reply,
        max_commit_tokens=8,
        committed_position=committed_position,
        commit_tokens=commit_tokens,
    )
    observations.append(("publish", plan.committed_tokens))
    assert observations == [
        ("commit", (42, 43, 44, 45)),
        ("publish", (42, 43, 44, 45)),
    ]
    assert durable[candidate.request_id] == [42, 43, 44, 45]


def test_durable_bridge_fails_closed_on_boundary_rng_and_partial_commit() -> None:
    scenarios = (
        (
            window(base_committed_position=1, input_position=100),
            lambda _request_id: 0,
            lambda _plan: None,
            "durable boundary",
        ),
        (
            window(
                base_committed_position=0,
                input_position=100,
                sampling=SpeculativeSampling(seed=9, temperature=0.5),
            ),
            lambda _request_id: 0,
            lambda _plan: None,
            "greedy sampling",
        ),
        (
            window(base_committed_position=0, input_position=100),
            lambda _request_id: 0,
            lambda _plan: None,
            "advance exactly",
        ),
    )
    for candidate, position, commit, message in scenarios:
        fence = SpeculativeWindowFence()
        fence.admit(candidate)
        with pytest.raises(ValueError, match=message):
            fence.settle_response_durably(
                response(candidate),
                max_commit_tokens=8,
                committed_position=position,
                commit_tokens=commit,
            )
        assert fence.pending_count(candidate.request_id) == 1
