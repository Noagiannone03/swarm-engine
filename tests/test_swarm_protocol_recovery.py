from __future__ import annotations

import hashlib

import pytest

from swarm_protocol.contracts import RecoveryLevel
from swarm_protocol.recovery import (
    InMemoryRecoveryJournal,
    RecoveryConflict,
    RecoveryJournalCapacityError,
    RecoveryState,
    RequestRecoverySpec,
    SamplingReplayContract,
    SamplingReplayMode,
    StaleRecoveryEpoch,
    exact_replay_sampling_params,
    replay_sequence_checksum,
    sampling_replay_contract,
    token_sequence_checksum,
)


def digest(character: str) -> str:
    return character * 64


def spec(
    request_id: str = "request",
    *,
    epoch: int = 7,
    recovery_level: RecoveryLevel = RecoveryLevel.RECOVERABLE,
    prompt_token_ids: tuple[int, ...] = (10, 20, 30),
    sampling: SamplingReplayContract | None = None,
) -> RequestRecoverySpec:
    return RequestRecoverySpec(
        request_id=request_id,
        model_swarm_id=digest("a"),
        immutable_revision="model-commit",
        tokenizer_hash=digest("b"),
        dtype="bfloat16",
        prefill_contract_hash=digest("c"),
        attention_kv_contract_hash=digest("d"),
        prompt_token_ids=prompt_token_ids,
        sampling=sampling
        or SamplingReplayContract(
            params_json="{}",
            params_hash=hashlib.sha256(b"{}").hexdigest(),
            mode=SamplingReplayMode.GREEDY,
        ),
        recovery_level=recovery_level,
        primary_route_id="route-primary",
        epoch=epoch,
        reserved_context_tokens=128,
    )


def test_exact_cold_replay_state_machine_fences_the_old_epoch() -> None:
    journal = InMemoryRecoveryJournal()
    started = journal.begin(spec())
    assert started.state == RecoveryState.PREFILLING
    assert started.prompt_checksum == token_sequence_checksum((10, 20, 30))

    decoding = journal.commit_prefill(
        "request",
        epoch=7,
        prompt_checksum=started.prompt_checksum,
    )
    assert decoding.state == RecoveryState.DECODING
    first = journal.commit_token(
        "request",
        epoch=7,
        position=0,
        token_id=40,
    )
    assert first.state == RecoveryState.DECODING
    assert first.replay_token_ids == (10, 20, 30, 40)

    recovering = journal.begin_recovery(
        "request",
        failed_epoch=7,
        new_epoch=8,
        replacement_route_id="route-replacement",
    )
    assert recovering.state == RecoveryState.RECOVERING
    assert recovering.route_ids == ("route-primary", "route-replacement")
    assert recovering.effective_recovery_level == RecoveryLevel.RESTARTABLE

    with pytest.raises(StaleRecoveryEpoch):
        journal.commit_token(
            "request",
            epoch=7,
            position=1,
            token_id=99,
        )
    with pytest.raises(RecoveryConflict, match="checksum"):
        journal.complete_replay(
            "request",
            epoch=8,
            sequence_checksum=digest("e"),
            rng_position=0,
        )

    resumed = journal.complete_replay(
        "request",
        epoch=8,
        sequence_checksum=recovering.sequence_checksum,
        rng_position=0,
    )
    assert resumed.state == RecoveryState.DECODING
    second = journal.commit_token(
        "request",
        epoch=8,
        position=1,
        token_id=50,
    )
    completed = journal.finish(
        "request",
        epoch=8,
        state=RecoveryState.COMPLETED,
    )
    assert second.committed_output_token_ids == (40, 50)
    assert completed.state == RecoveryState.COMPLETED


def test_token_commit_is_contiguous_and_idempotent_only_for_the_same_token() -> None:
    journal = InMemoryRecoveryJournal()
    started = journal.begin(spec())
    journal.commit_prefill("request", epoch=7, prompt_checksum=started.prompt_checksum)
    committed = journal.commit_token(
        "request",
        epoch=7,
        position=0,
        token_id=40,
    )

    assert (
        journal.commit_token(
            "request",
            epoch=7,
            position=0,
            token_id=40,
        )
        == committed
    )
    with pytest.raises(RecoveryConflict, match="different token"):
        journal.commit_token(
            "request",
            epoch=7,
            position=0,
            token_id=41,
        )
    with pytest.raises(RecoveryConflict, match="contiguous"):
        journal.commit_token(
            "request",
            epoch=7,
            position=2,
            token_id=42,
        )


def test_seeded_sampling_requires_monotonic_rng_position_across_replay() -> None:
    journal = InMemoryRecoveryJournal()
    started = journal.begin(
        spec(
            sampling=SamplingReplayContract(
                params_json='{"seed":1234}',
                params_hash=hashlib.sha256(b'{"seed":1234}').hexdigest(),
                mode=SamplingReplayMode.SEEDED,
                seed=1234,
            )
        )
    )
    journal.commit_prefill("request", epoch=7, prompt_checksum=started.prompt_checksum)
    first = journal.commit_token(
        "request",
        epoch=7,
        position=0,
        token_id=40,
        rng_position=3,
    )
    with pytest.raises(RecoveryConflict, match="RNG position"):
        journal.commit_token(
            "request",
            epoch=7,
            position=1,
            token_id=50,
            rng_position=3,
        )

    recovery = journal.begin_recovery(
        "request",
        failed_epoch=7,
        new_epoch=8,
        replacement_route_id="route-replacement",
    )
    with pytest.raises(RecoveryConflict, match="RNG position"):
        journal.complete_replay(
            "request",
            epoch=8,
            sequence_checksum=recovery.sequence_checksum,
            rng_position=2,
        )
    resumed = journal.complete_replay(
        "request",
        epoch=8,
        sequence_checksum=recovery.sequence_checksum,
        rng_position=3,
    )
    assert resumed.rng_position == first.rng_position


def test_sampling_contract_is_canonical_and_only_admits_deterministic_requests() -> None:
    contract = sampling_replay_contract(
        {
            "temperature": 0,
            "top_p": 0.9,
            "max_completion_tokens": 128,
            "stream": True,
            "messages": [{"role": "user", "content": "private"}],
        }
    )

    assert contract is not None
    assert contract.mode == SamplingReplayMode.GREEDY
    assert contract.params_json == ('{"max_completion_tokens":128,"temperature":0,"top_p":0.9}')
    assert "private" not in contract.params_json
    assert sampling_replay_contract({"temperature": 0.7}) is None
    assert sampling_replay_contract({"top_k": 1, "n": 2}) is None
    assert sampling_replay_contract({"temperature": 0, "stop": ["END"]}) is None
    assert sampling_replay_contract({"temperature": 0, "guided_regex": "[a-z]+"}) is None
    assert sampling_replay_contract({"temperature": 0, "frequency_penalty": 0.5}) is None
    assert sampling_replay_contract({"temperature": 0, "presence_penalty": 0.5}) is None
    assert sampling_replay_contract({"temperature": 0, "repetition_penalty": 1.1}) is None
    assert sampling_replay_contract({"temperature": 0, "thinking_token_budget": 128}) is None
    assert sampling_replay_contract({"temperature": 0, "logprobs": True}) is None
    assert sampling_replay_contract({"temperature": 0, "prompt_logprobs": 1}) is None
    assert (
        sampling_replay_contract(
            {
                "temperature": 0,
                "frequency_penalty": 0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
            }
        )
        is not None
    )


def test_exact_replay_sampling_subtracts_committed_minimum_and_output_budget() -> None:
    contract = sampling_replay_contract(
        {
            "temperature": 0,
            "top_p": 0.9,
            "min_tokens": 8,
            "max_tokens": 32,
            "tools": [{"type": "function", "function": {"name": "read"}}],
        }
    )
    assert contract is not None

    assert exact_replay_sampling_params(
        contract,
        committed_output_tokens=3,
        remaining_output_tokens=29,
    ) == {
        "temperature": 0,
        "top_p": 0.9,
        "min_tokens": 5,
        "max_tokens": 29,
    }


def test_restartable_request_cannot_be_promoted_to_recoverable_after_failure() -> None:
    journal = InMemoryRecoveryJournal()
    started = journal.begin(spec(recovery_level=RecoveryLevel.RESTARTABLE))
    journal.commit_prefill("request", epoch=7, prompt_checksum=started.prompt_checksum)

    with pytest.raises(RecoveryConflict, match="not admitted as recoverable"):
        journal.begin_recovery(
            "request",
            failed_epoch=7,
            new_epoch=8,
            replacement_route_id="route-replacement",
        )


def test_lost_backup_revokes_recovery_without_stopping_primary_decode() -> None:
    journal = InMemoryRecoveryJournal()
    started = journal.begin(spec())
    journal.commit_prefill("request", epoch=7, prompt_checksum=started.prompt_checksum)

    downgraded = journal.downgrade_to_restartable(
        "request",
        epoch=7,
        reason="reserved backup left the swarm",
    )

    assert downgraded.spec.recovery_level == RecoveryLevel.RECOVERABLE
    assert downgraded.effective_recovery_level == RecoveryLevel.RESTARTABLE
    assert downgraded.recovery_degradation_reason == "reserved backup left the swarm"
    continued = journal.commit_token(
        "request",
        epoch=7,
        position=0,
        token_id=40,
    )
    assert continued.state == RecoveryState.DECODING
    with pytest.raises(RecoveryConflict, match="not admitted as recoverable"):
        journal.begin_recovery(
            "request",
            failed_epoch=7,
            new_epoch=8,
            replacement_route_id="route-replacement",
        )


def test_recovery_downgrade_is_epoch_fenced_and_rejects_terminal_requests() -> None:
    journal = InMemoryRecoveryJournal()
    journal.begin(spec())

    with pytest.raises(StaleRecoveryEpoch):
        journal.downgrade_to_restartable("request", epoch=6, reason="stale observer")

    journal.finish("request", epoch=7, state=RecoveryState.ABORTED)
    with pytest.raises(RecoveryConflict, match="while aborted"):
        journal.downgrade_to_restartable("request", epoch=7, reason="too late")


def test_journal_never_evicts_active_requests_to_admit_more_tokens() -> None:
    journal = InMemoryRecoveryJournal(
        max_entries=2,
        max_terminal_entries=1,
        max_prompt_tokens_per_request=4,
        max_output_tokens_per_request=2,
        max_resident_token_ids=7,
    )
    first = journal.begin(spec("one"))
    journal.commit_prefill("one", epoch=7, prompt_checksum=first.prompt_checksum)
    journal.commit_token("one", epoch=7, position=0, token_id=40)
    journal.begin(spec("two"))

    with pytest.raises(RecoveryJournalCapacityError):
        journal.begin(spec("three", prompt_token_ids=(1,)))
    with pytest.raises(RecoveryJournalCapacityError):
        journal.commit_token("one", epoch=7, position=1, token_id=50)

    journal.finish("two", epoch=7, state=RecoveryState.ABORTED)
    admitted = journal.begin(spec("three", prompt_token_ids=(1,)))
    assert admitted.spec.request_id == "three"
    assert journal.get("one") is not None


def test_request_identity_and_terminal_outcome_are_conflict_checked() -> None:
    journal = InMemoryRecoveryJournal()
    started = journal.begin(spec())
    assert journal.begin(spec()) == started
    with pytest.raises(RecoveryConflict, match="different recovery spec"):
        journal.begin(spec(epoch=8))

    failed = journal.finish(
        "request",
        epoch=7,
        state=RecoveryState.FAILED,
        failure="worker lost",
    )
    assert failed.failure == "worker lost"
    assert (
        journal.finish(
            "request",
            epoch=7,
            state=RecoveryState.FAILED,
            failure="worker lost",
        )
        == failed
    )
    with pytest.raises(RecoveryConflict, match="different terminal"):
        journal.finish(
            "request",
            epoch=7,
            state=RecoveryState.ABORTED,
        )


def test_checksum_framing_distinguishes_order_and_token_boundaries() -> None:
    assert token_sequence_checksum((1, 23)) != token_sequence_checksum((12, 3))
    assert token_sequence_checksum((1, 2)) != token_sequence_checksum((2, 1))
    assert token_sequence_checksum((1, 2)) == token_sequence_checksum((1, 2))
    assert replay_sequence_checksum((1, 2), (3, 4)) != replay_sequence_checksum(
        (1, 2),
        (4, 3),
    )
