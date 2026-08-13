"""Bounded, route-fenced protocol-v3 speculative verification contracts.

These contracts expose target-model verification only. They deliberately do
not publish tokens, choose an accepted prefix, or activate speculation. Durable
settlement remains the Request Agent's commit-before-SSE responsibility.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Annotated, Callable, Self, TypeVar

from pydantic import Field, model_validator

from swarm_protocol.contracts import (
    PROTOCOL_VERSION,
    ContractModel,
    HashHex,
    NonEmpty,
    NonNegativeInt,
    PositiveInt,
)

MAX_SPECULATIVE_PROPOSAL_TOKENS = 64
MAX_SPECULATIVE_INPUT_TOKENS = MAX_SPECULATIVE_PROPOSAL_TOKENS + 1
MAX_SPECULATIVE_RESPONSE_BYTES = 16 * 1024 * 1024

TokenId = Annotated[int, Field(ge=0, le=2**31 - 1)]
BoundedId = Annotated[str, Field(min_length=1, max_length=255)]
SettlementValue = TypeVar("SettlementValue")


class SpeculativeStrategy(str, Enum):
    TARGET_ONLY = "target-only"
    NGRAM_SUFFIX = "ngram-suffix"
    NATIVE_MTP = "native-mtp"
    NATIVE_MTP_SUFFIX = "native-mtp+suffix"
    DRAFT_MODEL = "draft-model"


class SpeculativeSampling(ContractModel):
    """Exact target sampler inputs bound to one verification window."""

    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    temperature: Annotated[float, Field(ge=0, le=10)]
    top_p: Annotated[float, Field(gt=0, le=1)] = 1.0
    top_k: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    min_p: Annotated[float, Field(ge=0, le=1)] = 0.0
    presence_penalty: Annotated[float, Field(ge=-2, le=2)] = 0.0
    frequency_penalty: Annotated[float, Field(ge=-2, le=2)] = 0.0
    repeat_penalty: Annotated[float, Field(gt=0, le=10)] = 1.0
    penalty_last_n: Annotated[int, Field(ge=-1, le=1_000_000)] = -1
    stop_token_ids: Annotated[tuple[TokenId, ...], Field(max_length=256)] = ()

    @model_validator(mode="after")
    def validate_stop_tokens(self) -> Self:
        if len(self.stop_token_ids) != len(set(self.stop_token_ids)):
            raise ValueError("speculative stop token ids must be unique")
        return self


class SpeculativeVerifyWindow(ContractModel):
    """One unpublished candidate traversal on an exact reserved route."""

    protocol_version: int = PROTOCOL_VERSION
    request_id: NonEmpty
    route_id: NonEmpty
    epoch: NonNegativeInt
    route_plan_digest: HashHex
    window_id: PositiveInt
    base_committed_position: NonNegativeInt
    input_position: NonNegativeInt
    starts_epoch: bool
    prior_boundary_prediction: TokenId | None = None
    input_tokens: Annotated[
        tuple[TokenId, ...], Field(min_length=1, max_length=MAX_SPECULATIVE_INPUT_TOKENS)
    ]
    proposal_tokens: Annotated[
        tuple[TokenId, ...], Field(min_length=1, max_length=MAX_SPECULATIVE_PROPOSAL_TOKENS)
    ]
    strategy: SpeculativeStrategy
    proposer_id: BoundedId
    proposer_version: BoundedId
    sampling: SpeculativeSampling
    reserved_context_tokens: PositiveInt
    max_response_bytes: Annotated[int, Field(gt=0, le=MAX_SPECULATIVE_RESPONSE_BYTES)] = (
        MAX_SPECULATIVE_RESPONSE_BYTES
    )

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if self.strategy is SpeculativeStrategy.TARGET_ONLY:
            raise ValueError("target-only decode does not use speculative verify windows")
        expected_inputs = len(self.proposal_tokens) + int(self.starts_epoch)
        if len(self.input_tokens) != expected_inputs:
            raise ValueError("verification inputs do not match the proposal window layout")
        if self.starts_epoch and self.prior_boundary_prediction is not None:
            raise ValueError("epoch-start verification cannot carry a prior boundary prediction")
        if not self.starts_epoch and self.prior_boundary_prediction is None:
            raise ValueError("continuation verification requires a prior boundary prediction")
        if self.input_position < self.base_committed_position:
            raise ValueError("verification input precedes the durable committed position")
        if self.input_position + len(self.input_tokens) > self.reserved_context_tokens:
            raise ValueError("verification window exceeds its reserved context")
        return self

    def verify_response(self, response: SpeculativeVerifyResponse) -> None:
        """Fail closed on stale, substituted, oversized, or malformed replies."""

        expected_identity = (
            self.request_id,
            self.route_id,
            self.epoch,
            self.route_plan_digest,
            self.window_id,
            self.input_position,
            len(self.input_tokens),
        )
        actual_identity = (
            response.request_id,
            response.route_id,
            response.epoch,
            response.route_plan_digest,
            response.window_id,
            response.input_position,
            response.input_token_count,
        )
        if actual_identity != expected_identity:
            raise ValueError("speculative response does not match its fenced window")
        response_bytes = len(response.model_dump_json().encode("utf-8"))
        if response_bytes > self.max_response_bytes:
            raise ValueError("speculative response exceeds the reserved wire budget")

    def target_predictions(
        self,
        response: SpeculativeVerifyResponse,
    ) -> tuple[int, ...]:
        """Compose proposal-aligned target predictions exactly like Mesh 0.75.1.

        An epoch-start traversal predicts every proposal and one free boundary
        token. A continuation traversal predicts proposals 2..K and the free
        token; its proposal-1 prediction is the prior window's fenced boundary.
        """

        self.verify_response(response)
        if self.starts_epoch:
            return response.predicted_tokens
        boundary = self.prior_boundary_prediction
        if boundary is None:  # pragma: no cover - guarded by model validation
            raise ValueError("continuation verification has no prior boundary prediction")
        return (boundary, *response.predicted_tokens)

    def settlement_plan(
        self,
        response: SpeculativeVerifyResponse,
        *,
        max_commit_tokens: int,
        defer_full_accept_bonus: bool = False,
    ) -> SpeculativeSettlementPlan:
        """Build an unpublished exact-token settlement plan.

        The caller must durably commit ``committed_tokens`` before exposing
        their corresponding SSE bytes. Keeping the full-accept bonus deferred
        allows a following pipelined window to use it as its fenced boundary.
        """

        if max_commit_tokens <= 0 or max_commit_tokens > MAX_SPECULATIVE_INPUT_TOKENS:
            raise ValueError("speculative settlement commit budget is out of bounds")
        predictions = self.target_predictions(response)
        if len(predictions) != len(self.proposal_tokens) + 1:
            raise ValueError("target predictions do not cover proposal settlement")

        committed: list[int] = []
        accepted = 0
        rejected = False
        reached_stop = False
        reached_output_limit = False
        stop_tokens = set(self.sampling.stop_token_ids)
        for proposal, predicted in zip(
            self.proposal_tokens,
            predictions[:-1],
            strict=True,
        ):
            if len(committed) >= max_commit_tokens:
                reached_output_limit = True
                break
            committed.append(predicted)
            reached_stop = predicted in stop_tokens
            if predicted != proposal:
                rejected = True
                break
            accepted += 1
            if reached_stop:
                break

        next_boundary_prediction = None
        fully_accepted = accepted == len(self.proposal_tokens)
        if fully_accepted and not reached_stop:
            if len(committed) >= max_commit_tokens:
                reached_output_limit = True
            elif defer_full_accept_bonus and predictions[-1] not in stop_tokens:
                next_boundary_prediction = predictions[-1]
            else:
                bonus = predictions[-1]
                committed.append(bonus)
                reached_stop = bonus in stop_tokens

        if not committed:
            raise ValueError("speculative settlement cannot commit an empty token span")
        return SpeculativeSettlementPlan(
            request_id=self.request_id,
            route_id=self.route_id,
            epoch=self.epoch,
            route_plan_digest=self.route_plan_digest,
            window_id=self.window_id,
            base_committed_position=self.base_committed_position,
            verified_position=response.verified_position,
            accepted_proposal_tokens=accepted,
            committed_tokens=tuple(committed),
            rejected=rejected,
            reached_stop=reached_stop,
            reached_output_limit=reached_output_limit,
            next_boundary_prediction=next_boundary_prediction,
        )


class SpeculativeStageMetrics(ContractModel):
    """Content-free measurements for one route span and verify window."""

    stage_index: NonNegativeInt
    compute_us: NonNegativeInt
    wait_us: NonNegativeInt
    activation_bytes: NonNegativeInt
    stale_execution_us: NonNegativeInt = 0


class SpeculativeVerifyResponse(ContractModel):
    """Raw target predictions; still unpublished and not durably settled."""

    protocol_version: int = PROTOCOL_VERSION
    request_id: NonEmpty
    route_id: NonEmpty
    epoch: NonNegativeInt
    route_plan_digest: HashHex
    window_id: PositiveInt
    input_position: NonNegativeInt
    input_token_count: Annotated[int, Field(gt=0, le=MAX_SPECULATIVE_INPUT_TOKENS)]
    verified_position: PositiveInt
    predicted_tokens: Annotated[
        tuple[TokenId, ...], Field(min_length=1, max_length=MAX_SPECULATIVE_INPUT_TOKENS)
    ]
    metrics: Annotated[tuple[SpeculativeStageMetrics, ...], Field(max_length=256)] = ()

    @model_validator(mode="after")
    def validate_response(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if len(self.predicted_tokens) != self.input_token_count:
            raise ValueError("target predictions do not cover the verification inputs")
        if self.verified_position != self.input_position + self.input_token_count:
            raise ValueError("verified position does not match the verification input range")
        stage_indices = tuple(metric.stage_index for metric in self.metrics)
        if stage_indices != tuple(sorted(set(stage_indices))):
            raise ValueError("speculative stage metrics must be sorted and unique")
        return self


class SpeculativeSettlementPlan(ContractModel):
    """Unpublished target tokens ready for one durable Request Agent commit."""

    protocol_version: int = PROTOCOL_VERSION
    request_id: NonEmpty
    route_id: NonEmpty
    epoch: NonNegativeInt
    route_plan_digest: HashHex
    window_id: PositiveInt
    base_committed_position: NonNegativeInt
    verified_position: PositiveInt
    accepted_proposal_tokens: Annotated[int, Field(ge=0, le=MAX_SPECULATIVE_PROPOSAL_TOKENS)]
    committed_tokens: Annotated[
        tuple[TokenId, ...], Field(min_length=1, max_length=MAX_SPECULATIVE_INPUT_TOKENS)
    ]
    rejected: bool
    reached_stop: bool
    reached_output_limit: bool
    next_boundary_prediction: TokenId | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        terminal = self.rejected or self.reached_stop or self.reached_output_limit
        if terminal and self.next_boundary_prediction is not None:
            raise ValueError("terminal speculative settlement cannot retain a boundary")
        return self


class SpeculativeWindowFence:
    """Request-local replay guard for uncommitted verification responses.

    It is intentionally not a settlement journal. It only ensures that a
    response belongs to a currently admitted route/epoch/window and can be
    consumed once. A replan to a higher epoch drops all older in-flight windows.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._request_locks: dict[str, threading.RLock] = {}
        self._bindings: dict[str, tuple[str, int, str]] = {}
        self._minimum_epoch: dict[str, int] = {}
        self._last_window_id: dict[str, int] = {}
        self._pending: dict[tuple[str, int], SpeculativeVerifyWindow] = {}
        self._retired: set[str] = set()

    def admit(self, window: SpeculativeVerifyWindow) -> None:
        with self._request_lock(window.request_id):
            with self._lock:
                if window.request_id in self._retired:
                    raise ValueError("speculative request is retired")
                if window.epoch < self._minimum_epoch.get(window.request_id, 0):
                    raise ValueError("speculative window epoch is stale")
                binding = self._bindings.get(window.request_id)
                if binding is not None:
                    route_id, epoch, digest = binding
                    if window.epoch < epoch:
                        raise ValueError("speculative window epoch is stale")
                    if window.epoch == epoch and (
                        window.route_id != route_id or window.route_plan_digest != digest
                    ):
                        raise ValueError("speculative window changed route without a new epoch")
                    if window.epoch > epoch:
                        self._drop_pending(window.request_id)
                last_window_id = self._last_window_id.get(window.request_id, 0)
                if window.window_id <= last_window_id:
                    raise ValueError("speculative window id is not strictly increasing")
                self._bindings[window.request_id] = (
                    window.route_id,
                    window.epoch,
                    window.route_plan_digest,
                )
                self._last_window_id[window.request_id] = window.window_id
                self._pending[(window.request_id, window.window_id)] = window

    def accept_response(self, response: SpeculativeVerifyResponse) -> SpeculativeVerifyWindow:
        return self.settle_response(response, lambda window, _response: window)

    def settle_response(
        self,
        response: SpeculativeVerifyResponse,
        settle: Callable[[SpeculativeVerifyWindow, SpeculativeVerifyResponse], SettlementValue],
    ) -> SettlementValue:
        """Consume a response only after its caller-provided durable settlement.

        The callback runs under the request-local fence, without blocking
        settlement for unrelated requests. If it raises, the
        response remains pending and may be retried after the durable store has
        established whether anything committed. No other window can replan or
        consume the same response during that transaction.
        """

        with self._request_lock(response.request_id):
            with self._lock:
                key = (response.request_id, response.window_id)
                window = self._pending.get(key)
                if window is None:
                    raise ValueError("speculative response is stale, duplicate, or unknown")
                binding = self._bindings.get(response.request_id)
                if binding != (response.route_id, response.epoch, response.route_plan_digest):
                    raise ValueError("speculative response was fenced by a route change")
                window.verify_response(response)
            settled = settle(window, response)
            with self._lock:
                if self._pending.get(key) is not window:
                    raise ValueError("speculative response changed during durable settlement")
                del self._pending[key]
            return settled

    def settle_response_durably(
        self,
        response: SpeculativeVerifyResponse,
        *,
        max_commit_tokens: int,
        committed_position: Callable[[str], int | None],
        commit_tokens: Callable[[SpeculativeSettlementPlan], None],
        defer_full_accept_bonus: bool = False,
    ) -> SpeculativeSettlementPlan:
        """Atomically bridge one fenced greedy response to a durable journal.

        This returns unpublished target tokens. The future SSE encoder may use
        them only after this method returns. Seeded/non-greedy settlement stays
        fail-closed until the protocol carries exact per-token RNG positions.
        """

        def commit(
            window: SpeculativeVerifyWindow,
            verified: SpeculativeVerifyResponse,
        ) -> SpeculativeSettlementPlan:
            if window.sampling.temperature != 0:
                raise ValueError(
                    "durable speculative settlement requires greedy sampling until RNG fencing"
                )
            before = committed_position(window.request_id)
            if before is None:
                raise ValueError("speculative request has no durable journal")
            if before != window.base_committed_position:
                raise ValueError("speculative window does not start at the durable boundary")
            plan = window.settlement_plan(
                verified,
                max_commit_tokens=max_commit_tokens,
                defer_full_accept_bonus=defer_full_accept_bonus,
            )
            commit_tokens(plan)
            after = committed_position(window.request_id)
            expected = before + len(plan.committed_tokens)
            if after != expected:
                raise ValueError("durable speculative commit did not advance exactly")
            return plan

        return self.settle_response(response, commit)

    def fence(self, request_id: str, *, newer_epoch: int) -> None:
        if not self.fence_at_least(request_id, newer_epoch=newer_epoch):
            raise ValueError("speculative fence epoch must move forward")

    def fence_at_least(self, request_id: str, *, newer_epoch: int) -> bool:
        """Fence every older window, idempotently, before route recovery.

        Request lifecycle code can observe the same failure through more than
        one path (lease invalidation, explicit cold replan, then cleanup).  The
        first observer advances the request fence; later observers are safe
        no-ops instead of reopening or weakening that fence.
        """

        request_id = str(request_id)
        if newer_epoch < 0:
            raise ValueError("speculative fence epoch must be non-negative")
        with self._request_lock(request_id):
            with self._lock:
                if request_id in self._retired:
                    return False
                binding = self._bindings.get(request_id)
                current_epoch = max(
                    self._minimum_epoch.get(request_id, 0),
                    -1 if binding is None else binding[1],
                )
                if newer_epoch <= current_epoch:
                    return False
                self._drop_pending(request_id)
                self._bindings.pop(request_id, None)
                self._minimum_epoch[request_id] = newer_epoch
                return True

    def retire(self, request_id: str) -> None:
        """Permanently reject late work after release, abort, or completion."""

        request_id = str(request_id)
        with self._request_lock(request_id):
            with self._lock:
                self._drop_pending(request_id)
                self._bindings.pop(request_id, None)
                self._minimum_epoch.pop(request_id, None)
                self._last_window_id.pop(request_id, None)
                self._retired.add(request_id)

    def pending_count(self, request_id: str) -> int:
        with self._lock:
            return sum(key[0] == request_id for key in self._pending)

    def _drop_pending(self, request_id: str) -> None:
        for key in tuple(self._pending):
            if key[0] == request_id:
                del self._pending[key]

    def _request_lock(self, request_id: str) -> threading.RLock:
        with self._lock:
            return self._request_locks.setdefault(str(request_id), threading.RLock())
