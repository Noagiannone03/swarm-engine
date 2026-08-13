"""Bounded, route-fenced protocol-v3 speculative verification contracts.

These contracts expose target-model verification only. They deliberately do
not publish tokens, choose an accepted prefix, or activate speculation. Durable
settlement remains the Request Agent's commit-before-SSE responsibility.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Annotated, Self

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


class SpeculativeWindowFence:
    """Request-local replay guard for uncommitted verification responses.

    It is intentionally not a settlement journal. It only ensures that a
    response belongs to a currently admitted route/epoch/window and can be
    consumed once. A replan to a higher epoch drops all older in-flight windows.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bindings: dict[str, tuple[str, int, str]] = {}
        self._minimum_epoch: dict[str, int] = {}
        self._last_window_id: dict[str, int] = {}
        self._pending: dict[tuple[str, int], SpeculativeVerifyWindow] = {}

    def admit(self, window: SpeculativeVerifyWindow) -> None:
        with self._lock:
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
        with self._lock:
            key = (response.request_id, response.window_id)
            window = self._pending.get(key)
            if window is None:
                raise ValueError("speculative response is stale, duplicate, or unknown")
            binding = self._bindings.get(response.request_id)
            if binding != (response.route_id, response.epoch, response.route_plan_digest):
                raise ValueError("speculative response was fenced by a route change")
            window.verify_response(response)
            del self._pending[key]
            return window

    def fence(self, request_id: str, *, newer_epoch: int) -> None:
        with self._lock:
            binding = self._bindings.get(request_id)
            current_epoch = max(
                self._minimum_epoch.get(request_id, 0),
                -1 if binding is None else binding[1],
            )
            if newer_epoch <= current_epoch:
                raise ValueError("speculative fence epoch must move forward")
            self._drop_pending(request_id)
            self._bindings.pop(request_id, None)
            self._minimum_epoch[request_id] = newer_epoch

    def pending_count(self, request_id: str) -> int:
        with self._lock:
            return sum(key[0] == request_id for key in self._pending)

    def _drop_pending(self, request_id: str) -> None:
        for key in tuple(self._pending):
            if key[0] == request_id:
                del self._pending[key]
