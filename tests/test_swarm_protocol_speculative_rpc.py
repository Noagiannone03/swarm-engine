import pytest
from pydantic import ValidationError

import swarm_protocol.execution_rpc as execution_rpc
from swarm_protocol import (
    SpeculativeSampling,
    SpeculativeStrategy,
    SpeculativeVerifyResponse,
    SpeculativeVerifyWindow,
)
from swarm_protocol.execution_rpc import WorkerSpeculativeVerifyService


def window(**updates):
    values = {
        "request_id": "request-a",
        "route_id": "route-a",
        "epoch": 3,
        "route_plan_digest": "a" * 64,
        "window_id": 1,
        "base_committed_position": 0,
        "input_position": 10,
        "starts_epoch": True,
        "input_tokens": (7, 8, 9),
        "proposal_tokens": (8, 9),
        "strategy": SpeculativeStrategy.NGRAM_SUFFIX,
        "proposer_id": "mesh-longest-suffix",
        "proposer_version": "0.75.1",
        "sampling": SpeculativeSampling(seed=0, temperature=0),
        "reserved_context_tokens": 100,
    }
    values.update(updates)
    return SpeculativeVerifyWindow(**values)


def response(candidate, **updates):
    values = {
        "request_id": candidate.request_id,
        "route_id": candidate.route_id,
        "epoch": candidate.epoch,
        "route_plan_digest": candidate.route_plan_digest,
        "window_id": candidate.window_id,
        "input_position": candidate.input_position,
        "input_token_count": len(candidate.input_tokens),
        "verified_position": candidate.input_position + len(candidate.input_tokens),
        "predicted_tokens": (8, 9, 10),
    }
    values.update(updates)
    return SpeculativeVerifyResponse(**values)


class Admission:
    def __init__(self):
        self.calls = []

    def authorize_speculative_verify(self, **values):
        self.calls.append(values)


def test_rpc_authorizes_exact_digest_and_returns_only_verified_predictions(monkeypatch):
    monkeypatch.setattr(execution_rpc, "authenticated_rpc_peer_id", lambda: "coordinator")
    admission = Admission()
    candidate = window()
    service = WorkerSpeculativeVerifyService(admission, response)

    wire = service.verify(candidate.model_dump(mode="json"))

    assert SpeculativeVerifyResponse.model_validate(wire) == response(candidate)
    assert admission.calls == [
        {
            "request_id": "request-a",
            "route_id": "route-a",
            "epoch": 3,
            "route_plan_digest": "a" * 64,
            "caller_endpoint_id": "coordinator",
        }
    ]
    assert "proposal_tokens" not in wire


def test_rpc_fails_closed_on_substituted_or_unbounded_payload(monkeypatch):
    monkeypatch.setattr(execution_rpc, "authenticated_rpc_peer_id", lambda: "coordinator")
    admission = Admission()
    candidate = window()
    substituted = WorkerSpeculativeVerifyService(
        admission,
        lambda verified: response(verified, route_plan_digest="b" * 64),
    )
    with pytest.raises(ValueError, match="fenced window"):
        substituted.verify(candidate.model_dump(mode="json"))

    oversized = candidate.model_dump(mode="json")
    oversized["proposal_tokens"] = list(range(65))
    oversized["input_tokens"] = list(range(66))
    with pytest.raises(ValidationError):
        substituted.verify(oversized)
