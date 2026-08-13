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
