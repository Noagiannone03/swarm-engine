"""Dormant, bounded controller contracts for adaptive speculative decoding.

The controller never sees prompts, token ids, or generated text. It compares
committed-token cost on one exact request/route/sampling binding and fails back
to target-only until a speculative variant is conservatively better.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
from swarm_protocol.speculative import (
    MAX_SPECULATIVE_PROPOSAL_TOKENS,
    SpeculativeStrategy,
)

Microseconds = Annotated[int, Field(ge=0, le=2**63 - 1)]


class SpeculativeControllerReason(str, Enum):
    BASELINE_REQUIRED = "baseline-required"
    VARIANT_REQUIRED = "variant-required"
    NO_PROVEN_GAIN = "no-proven-gain"
    COOLDOWN = "cooldown"
    PROVEN_GAIN = "proven-gain"


class SpeculativeCostObservation(ContractModel):
    """One token-free cost observation at a durable commit boundary."""

    protocol_version: int = PROTOCOL_VERSION
    request_id: NonEmpty
    route_id: NonEmpty
    epoch: NonNegativeInt
    model_swarm_id: HashHex
    sampling_fingerprint: HashHex
    strategy: SpeculativeStrategy
    proposal_window_tokens: Annotated[int, Field(ge=0, le=MAX_SPECULATIVE_PROPOSAL_TOKENS)]
    proposed_tokens: Annotated[int, Field(ge=0, le=MAX_SPECULATIVE_PROPOSAL_TOKENS)]
    accepted_tokens: Annotated[int, Field(ge=0, le=MAX_SPECULATIVE_PROPOSAL_TOKENS)]
    committed_tokens: PositiveInt
    draft_us: Microseconds = 0
    verification_us: Microseconds = 0
    network_wait_us: Microseconds = 0
    commit_us: Microseconds = 0
    stale_execution_us: Microseconds = 0

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if self.accepted_tokens > self.proposed_tokens:
            raise ValueError("accepted tokens cannot exceed proposed tokens")
        if self.strategy is SpeculativeStrategy.TARGET_ONLY:
            if self.proposal_window_tokens or self.proposed_tokens or self.accepted_tokens:
                raise ValueError("target-only observations cannot contain proposals")
        elif self.proposal_window_tokens == 0:
            raise ValueError("speculative observations require a positive proposal window")
        if self.total_cost_us <= 0:
            raise ValueError("controller observation cost must be positive")
        return self

    @property
    def total_cost_us(self) -> int:
        return (
            self.draft_us
            + self.verification_us
            + self.network_wait_us
            + self.commit_us
            + self.stale_execution_us
        )

    @property
    def cost_per_committed_token_us(self) -> float:
        return self.total_cost_us / self.committed_tokens


class SpeculativeControllerDecision(ContractModel):
    """Bounded unpublished policy decision for the current request binding."""

    protocol_version: int = PROTOCOL_VERSION
    strategy: SpeculativeStrategy
    max_proposal_tokens: Annotated[int, Field(ge=0, le=MAX_SPECULATIVE_PROPOSAL_TOKENS)]
    reason: SpeculativeControllerReason
    baseline_committed_tokens: NonNegativeInt
    variant_committed_tokens: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.strategy is SpeculativeStrategy.TARGET_ONLY:
            if self.max_proposal_tokens != 0:
                raise ValueError("target-only decisions cannot carry a proposal window")
        elif self.max_proposal_tokens == 0:
            raise ValueError("speculative decisions require a proposal window")
        return self


@dataclass(frozen=True)
class _ControllerBinding:
    request_id: str
    route_id: str
    epoch: int
    model_swarm_id: str
    sampling_fingerprint: str


@dataclass
class _OnlineCost:
    windows: int = 0
    committed_tokens: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def observe(self, value: float, committed_tokens: int) -> None:
        self.windows += 1
        self.committed_tokens += committed_tokens
        delta = value - self.mean
        self.mean += delta / self.windows
        self.m2 += delta * (value - self.mean)

    def standard_error(self) -> float:
        if self.windows < 2:
            return math.inf
        variance = self.m2 / (self.windows - 1)
        return math.sqrt(max(0.0, variance) / self.windows)


class AdaptiveSpeculativeController:
    """Request-local conservative selector; dormant until explicitly wired."""

    def __init__(
        self,
        *,
        signed_max_proposal_tokens: int,
        min_committed_tokens: int = 32,
        cooldown_commits: int = 16,
        confidence_z: float = 1.96,
        max_variants: int = 64,
    ) -> None:
        if not 1 <= signed_max_proposal_tokens <= MAX_SPECULATIVE_PROPOSAL_TOKENS:
            raise ValueError("signed proposal bound is outside the protocol limit")
        if min_committed_tokens <= 0:
            raise ValueError("minimum committed token count must be positive")
        if cooldown_commits < 0:
            raise ValueError("controller cooldown must be non-negative")
        if not math.isfinite(confidence_z) or confidence_z <= 0:
            raise ValueError("controller confidence multiplier must be positive and finite")
        if max_variants <= 0:
            raise ValueError("controller variant bound must be positive")
        self.signed_max_proposal_tokens = int(signed_max_proposal_tokens)
        self.min_committed_tokens = int(min_committed_tokens)
        self.cooldown_commits = int(cooldown_commits)
        self.confidence_z = float(confidence_z)
        self.max_variants = int(max_variants)
        self._binding: _ControllerBinding | None = None
        self._baseline = _OnlineCost()
        self._variants: dict[tuple[SpeculativeStrategy, int], _OnlineCost] = {}
        self._decision = self._target_only(SpeculativeControllerReason.BASELINE_REQUIRED)
        self._commits_since_change = 0

    def observe(self, observation: SpeculativeCostObservation) -> SpeculativeControllerDecision:
        binding = _ControllerBinding(
            request_id=observation.request_id,
            route_id=observation.route_id,
            epoch=observation.epoch,
            model_swarm_id=observation.model_swarm_id,
            sampling_fingerprint=observation.sampling_fingerprint,
        )
        if binding != self._binding:
            self._reset(binding)
        self._commits_since_change += observation.committed_tokens
        if observation.strategy is SpeculativeStrategy.TARGET_ONLY:
            self._baseline.observe(
                observation.cost_per_committed_token_us,
                observation.committed_tokens,
            )
        else:
            if observation.proposal_window_tokens > self.signed_max_proposal_tokens:
                raise ValueError("observation exceeds the signed proposal bound")
            key = (observation.strategy, observation.proposal_window_tokens)
            stats = self._variants.get(key)
            if stats is None:
                if len(self._variants) >= self.max_variants:
                    raise ValueError("controller variant bound is full")
                stats = self._variants[key] = _OnlineCost()
            stats.observe(
                observation.cost_per_committed_token_us,
                observation.committed_tokens,
            )
        return self.decision()

    def decision(self) -> SpeculativeControllerDecision:
        if self._baseline.committed_tokens < self.min_committed_tokens:
            return self._change_or_hold(
                self._target_only(SpeculativeControllerReason.BASELINE_REQUIRED)
            )
        baseline_lower = self._baseline.mean - self.confidence_z * self._baseline.standard_error()
        eligible: list[tuple[float, SpeculativeStrategy, int, _OnlineCost]] = []
        for (strategy, window), stats in self._variants.items():
            if stats.committed_tokens < self.min_committed_tokens:
                continue
            upper = stats.mean + self.confidence_z * stats.standard_error()
            if upper < baseline_lower:
                eligible.append((upper, strategy, window, stats))
        if not eligible:
            reason = (
                SpeculativeControllerReason.VARIANT_REQUIRED
                if not self._variants
                else SpeculativeControllerReason.NO_PROVEN_GAIN
            )
            return self._change_or_hold(self._target_only(reason))
        _, strategy, window, stats = min(
            eligible,
            key=lambda item: (item[0], item[2], item[1].value),
        )
        candidate = SpeculativeControllerDecision(
            strategy=strategy,
            max_proposal_tokens=window,
            reason=SpeculativeControllerReason.PROVEN_GAIN,
            baseline_committed_tokens=self._baseline.committed_tokens,
            variant_committed_tokens=stats.committed_tokens,
        )
        return self._change_or_hold(candidate)

    def state_size(self) -> int:
        """Return bounded aggregate count for deterministic memory tests."""

        return len(self._variants) + 1

    def _change_or_hold(
        self,
        candidate: SpeculativeControllerDecision,
    ) -> SpeculativeControllerDecision:
        same_policy = (
            candidate.strategy == self._decision.strategy
            and candidate.max_proposal_tokens == self._decision.max_proposal_tokens
        )
        if same_policy:
            self._decision = candidate
            return candidate
        if (
            candidate.strategy is SpeculativeStrategy.TARGET_ONLY
            and self._decision.strategy is not SpeculativeStrategy.TARGET_ONLY
        ):
            self._decision = candidate
            self._commits_since_change = 0
            return candidate
        if self._commits_since_change < self.cooldown_commits:
            return self._decision.model_copy(
                update={"reason": SpeculativeControllerReason.COOLDOWN}
            )
        self._decision = candidate
        self._commits_since_change = 0
        return candidate

    def _target_only(self, reason: SpeculativeControllerReason) -> SpeculativeControllerDecision:
        return SpeculativeControllerDecision(
            strategy=SpeculativeStrategy.TARGET_ONLY,
            max_proposal_tokens=0,
            reason=reason,
            baseline_committed_tokens=self._baseline.committed_tokens,
        )

    def _reset(self, binding: _ControllerBinding) -> None:
        self._binding = binding
        self._baseline = _OnlineCost()
        self._variants.clear()
        self._decision = self._target_only(SpeculativeControllerReason.BASELINE_REQUIRED)
        self._commits_since_change = 0
