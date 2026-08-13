import pytest

from swarm_protocol.speculative import SpeculativeStrategy
from swarm_protocol.speculative_control import (
    AdaptiveSpeculativeController,
    SpeculativeControllerReason,
    SpeculativeCostObservation,
)

MODEL = "a" * 64
SAMPLING = "b" * 64


def observation(
    strategy=SpeculativeStrategy.TARGET_ONLY,
    *,
    route_id="route-a",
    epoch=1,
    window=0,
    proposed=0,
    accepted=0,
    committed=16,
    total_us=1_600,
):
    return SpeculativeCostObservation(
        request_id="request",
        route_id=route_id,
        epoch=epoch,
        model_swarm_id=MODEL,
        sampling_fingerprint=SAMPLING,
        strategy=strategy,
        proposal_window_tokens=window,
        proposed_tokens=proposed,
        accepted_tokens=accepted,
        committed_tokens=committed,
        verification_us=total_us,
    )


def test_controller_requires_conservative_baseline_and_variant_evidence() -> None:
    controller = AdaptiveSpeculativeController(
        signed_max_proposal_tokens=8,
        min_committed_tokens=32,
        cooldown_commits=0,
    )
    controller.observe(observation())
    assert controller.decision().reason is SpeculativeControllerReason.BASELINE_REQUIRED
    controller.observe(observation())
    assert controller.decision().reason is SpeculativeControllerReason.VARIANT_REQUIRED

    for _ in range(2):
        decision = controller.observe(
            observation(
                SpeculativeStrategy.NGRAM_SUFFIX,
                window=4,
                proposed=4,
                accepted=3,
                total_us=800,
            )
        )

    assert decision.strategy is SpeculativeStrategy.NGRAM_SUFFIX
    assert decision.max_proposal_tokens == 4
    assert decision.reason is SpeculativeControllerReason.PROVEN_GAIN


def test_high_acceptance_does_not_override_worse_end_to_end_cost() -> None:
    controller = AdaptiveSpeculativeController(
        signed_max_proposal_tokens=8,
        min_committed_tokens=32,
        cooldown_commits=0,
    )
    for _ in range(2):
        controller.observe(observation(total_us=800))
        decision = controller.observe(
            observation(
                SpeculativeStrategy.NGRAM_SUFFIX,
                window=4,
                proposed=4,
                accepted=4,
                total_us=1_600,
            )
        )

    assert decision.strategy is SpeculativeStrategy.TARGET_ONLY
    assert decision.reason is SpeculativeControllerReason.NO_PROVEN_GAIN


def test_route_epoch_change_resets_all_previous_evidence() -> None:
    controller = AdaptiveSpeculativeController(
        signed_max_proposal_tokens=8,
        min_committed_tokens=32,
        cooldown_commits=0,
    )
    for _ in range(2):
        controller.observe(observation(committed=16, total_us=1_600))
        decision = controller.observe(
            observation(
                SpeculativeStrategy.NGRAM_SUFFIX,
                window=4,
                proposed=4,
                accepted=4,
                committed=16,
                total_us=800,
            )
        )
    assert decision.strategy is SpeculativeStrategy.NGRAM_SUFFIX

    reset = controller.observe(observation(route_id="route-b", epoch=2))

    assert reset.strategy is SpeculativeStrategy.TARGET_ONLY
    assert reset.reason is SpeculativeControllerReason.BASELINE_REQUIRED
    assert controller.state_size() == 1


def test_controller_state_and_signed_window_are_bounded() -> None:
    controller = AdaptiveSpeculativeController(
        signed_max_proposal_tokens=2,
        min_committed_tokens=1,
        cooldown_commits=0,
        max_variants=1,
    )
    controller.observe(observation())
    controller.observe(
        observation(
            SpeculativeStrategy.NGRAM_SUFFIX,
            window=1,
            proposed=1,
            accepted=1,
        )
    )
    assert controller.state_size() == 2
    with pytest.raises(ValueError, match="variant bound"):
        controller.observe(
            observation(
                SpeculativeStrategy.NGRAM_SUFFIX,
                window=2,
                proposed=2,
                accepted=1,
            )
        )
    with pytest.raises(ValueError, match="signed proposal bound"):
        controller.observe(
            observation(
                SpeculativeStrategy.NGRAM_SUFFIX,
                window=3,
                proposed=3,
                accepted=1,
            )
        )


def test_proven_regression_falls_back_without_waiting_for_cooldown() -> None:
    controller = AdaptiveSpeculativeController(
        signed_max_proposal_tokens=8,
        min_committed_tokens=16,
        cooldown_commits=100,
    )
    for _ in range(4):
        controller.observe(observation(total_us=1_600))
        decision = controller.observe(
            observation(
                SpeculativeStrategy.NGRAM_SUFFIX,
                window=4,
                proposed=4,
                accepted=4,
                total_us=800,
            )
        )
    assert decision.strategy is SpeculativeStrategy.NGRAM_SUFFIX

    fallback = controller.observe(
        observation(
            SpeculativeStrategy.NGRAM_SUFFIX,
            window=4,
            proposed=4,
            accepted=4,
            total_us=16_000,
        )
    )

    assert fallback.strategy is SpeculativeStrategy.TARGET_ONLY
    assert fallback.reason is SpeculativeControllerReason.NO_PROVEN_GAIN
