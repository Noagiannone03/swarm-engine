from __future__ import annotations

import hashlib
import os
import sqlite3

import pytest

from swarm_protocol.contracts import RecoveryLevel
from swarm_protocol.recovery import (
    RecoveryConflict,
    RecoveryJournalCapacityError,
    RecoveryState,
    RequestRecoverySpec,
    SamplingReplayContract,
    SamplingReplayMode,
    StaleRecoveryEpoch,
)
from swarm_protocol.recovery_sqlite import SqliteRecoveryJournal


def _digest(character: str) -> str:
    return character * 64


def _spec(
    request_id: str = "request",
    *,
    epoch: int = 7,
    prompt_token_ids: tuple[int, ...] = (10, 20, 30),
) -> RequestRecoverySpec:
    return RequestRecoverySpec(
        request_id=request_id,
        model_swarm_id=_digest("a"),
        immutable_revision="model-commit",
        tokenizer_hash=_digest("b"),
        dtype="bfloat16",
        prefill_contract_hash=_digest("c"),
        attention_kv_contract_hash=_digest("d"),
        prompt_token_ids=prompt_token_ids,
        sampling=SamplingReplayContract(
            params_json="{}",
            params_hash=hashlib.sha256(b"{}").hexdigest(),
            mode=SamplingReplayMode.GREEDY,
        ),
        recovery_level=RecoveryLevel.RECOVERABLE,
        primary_route_id="route-primary",
        epoch=epoch,
        reserved_context_tokens=128,
    )


def test_sqlite_journal_persists_exact_commits_across_reopen(tmp_path) -> None:
    path = tmp_path / "recovery.sqlite3"
    journal = SqliteRecoveryJournal(path)
    started = journal.begin(_spec())
    journal.commit_prefill("request", epoch=7, prompt_checksum=started.prompt_checksum)
    journal.commit_tokens(
        "request",
        epoch=7,
        position=0,
        token_ids=(40, 50, 60),
    )
    before_restart = journal.get("request")
    journal.close()

    reopened = SqliteRecoveryJournal(path)
    after_restart = reopened.get("request")

    assert after_restart == before_restart
    assert after_restart is not None
    assert after_restart.committed_output_token_ids == (40, 50, 60)
    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0
    reopened.close()


def test_sqlite_token_batch_is_atomic_contiguous_and_epoch_fenced(tmp_path) -> None:
    journal = SqliteRecoveryJournal(tmp_path / "recovery.sqlite3")
    started = journal.begin(_spec())
    journal.commit_prefill("request", epoch=7, prompt_checksum=started.prompt_checksum)
    journal.commit_tokens(
        "request",
        epoch=7,
        position=0,
        token_ids=(40, 50),
    )

    journal.commit_tokens(
        "request",
        epoch=7,
        position=0,
        token_ids=(40, 50),
    )
    with pytest.raises(RecoveryConflict, match="different token"):
        journal.commit_tokens(
            "request",
            epoch=7,
            position=0,
            token_ids=(40, 51),
        )
    with pytest.raises(RecoveryConflict, match="contiguous"):
        journal.commit_tokens(
            "request",
            epoch=7,
            position=3,
            token_ids=(60,),
        )
    with pytest.raises(StaleRecoveryEpoch):
        journal.commit_tokens(
            "request",
            epoch=6,
            position=2,
            token_ids=(60,),
        )

    assert journal.get("request").committed_output_token_ids == (40, 50)
    journal.close()


def test_sqlite_replan_can_recover_more_than_once_without_reserved_backup(tmp_path) -> None:
    journal = SqliteRecoveryJournal(tmp_path / "recovery.sqlite3")
    started = journal.begin(_spec())
    journal.commit_prefill("request", epoch=7, prompt_checksum=started.prompt_checksum)
    journal.commit_token("request", epoch=7, position=0, token_id=40)

    first = journal.begin_recovery(
        "request",
        failed_epoch=7,
        new_epoch=8,
        replacement_route_id="route-two",
        retain_recovery_level=True,
    )
    assert first.effective_recovery_level == RecoveryLevel.RECOVERABLE
    journal.complete_replay(
        "request",
        epoch=8,
        sequence_checksum=first.sequence_checksum,
        rng_position=0,
    )
    journal.commit_token("request", epoch=8, position=1, token_id=50)
    second = journal.begin_recovery(
        "request",
        failed_epoch=8,
        new_epoch=9,
        replacement_route_id="route-three",
        retain_recovery_level=True,
    )

    assert second.route_ids == ("route-primary", "route-two", "route-three")
    assert second.committed_output_token_ids == (40, 50)
    assert second.effective_recovery_level == RecoveryLevel.RECOVERABLE
    journal.close()


def test_sqlite_journal_aborts_streams_interrupted_by_process_restart(tmp_path) -> None:
    journal = SqliteRecoveryJournal(tmp_path / "recovery.sqlite3")
    started = journal.begin(_spec())
    journal.commit_prefill("request", epoch=7, prompt_checksum=started.prompt_checksum)

    assert journal.abort_unfinished("request agent restarted") == 1
    snapshot = journal.get("request")
    assert snapshot is not None
    assert snapshot.state == RecoveryState.ABORTED
    assert snapshot.failure == "request agent restarted"
    assert journal.abort_unfinished("request agent restarted again") == 0
    journal.close()


def test_sqlite_journal_never_evicts_active_requests(tmp_path) -> None:
    journal = SqliteRecoveryJournal(
        tmp_path / "recovery.sqlite3",
        max_entries=2,
        max_terminal_entries=1,
        max_prompt_tokens_per_request=4,
        max_output_tokens_per_request=2,
        max_resident_token_ids=7,
    )
    first = journal.begin(_spec("one"))
    journal.commit_prefill("one", epoch=7, prompt_checksum=first.prompt_checksum)
    journal.commit_token("one", epoch=7, position=0, token_id=40)
    journal.begin(_spec("two"))

    with pytest.raises(RecoveryJournalCapacityError):
        journal.begin(_spec("three", prompt_token_ids=(1,)))
    with pytest.raises(RecoveryJournalCapacityError):
        journal.commit_token("one", epoch=7, position=1, token_id=50)

    journal.finish("two", epoch=7, state=RecoveryState.ABORTED)
    assert journal.begin(_spec("three", prompt_token_ids=(1,))).spec.request_id == "three"
    assert journal.get("one") is not None
    journal.close()


def test_sqlite_journal_detects_persisted_token_corruption(tmp_path) -> None:
    path = tmp_path / "recovery.sqlite3"
    journal = SqliteRecoveryJournal(path)
    started = journal.begin(_spec())
    journal.commit_prefill("request", epoch=7, prompt_checksum=started.prompt_checksum)
    journal.commit_token("request", epoch=7, position=0, token_id=40)
    journal.close()

    connection = sqlite3.connect(path)
    connection.execute(
        """
        UPDATE recovery_output_tokens SET token_id = 41
        WHERE request_id = 'request' AND position = 0
        """
    )
    connection.commit()
    connection.close()

    reopened = SqliteRecoveryJournal(path)
    with pytest.raises(RecoveryConflict, match="checksum"):
        reopened.get("request")
    reopened.close()
