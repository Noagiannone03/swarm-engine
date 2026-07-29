"""Crash-durable SQLite journal for exact generation recovery.

The database is local to one Request Agent.  Each output-token commit is an
explicit SQLite transaction and therefore completes before the matching SSE
event may be published.  WAL plus ``synchronous=FULL`` preserves committed
transactions across an OS crash or power loss; unfinished HTTP streams are
marked aborted when a new Request Agent process takes ownership.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Iterator

from swarm_protocol.contracts import RecoveryLevel
from swarm_protocol.recovery import (
    RecoveryConflict,
    RecoveryJournalCapacityError,
    RecoveryState,
    RequestRecoverySnapshot,
    RequestRecoverySpec,
    SamplingReplayContract,
    SamplingReplayMode,
    StaleRecoveryEpoch,
    extend_token_sequence_checksum,
    replay_sequence_checksum,
    token_sequence_checksum,
)

_TERMINAL_STATES = (
    RecoveryState.COMPLETED,
    RecoveryState.FAILED,
    RecoveryState.ABORTED,
)
_ACTIVE_STATES = (
    RecoveryState.PREFILLING,
    RecoveryState.DECODING,
    RecoveryState.RECOVERING,
)
_SCHEMA_VERSION = 1


class SqliteRecoveryJournal:
    """Bounded, transactionally durable recovery state for a local frontend."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_entries: int = 1_024,
        max_terminal_entries: int = 64,
        max_prompt_tokens_per_request: int = 262_144,
        max_output_tokens_per_request: int = 65_536,
        max_resident_token_ids: int = 2_000_000,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        for name, value in (
            ("max_entries", max_entries),
            ("max_prompt_tokens_per_request", max_prompt_tokens_per_request),
            ("max_output_tokens_per_request", max_output_tokens_per_request),
            ("max_resident_token_ids", max_resident_token_ids),
            ("busy_timeout_ms", busy_timeout_ms),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if max_terminal_entries < 0 or max_terminal_entries > max_entries:
            raise ValueError("max_terminal_entries must be between zero and max_entries")

        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = int(max_entries)
        self.max_terminal_entries = int(max_terminal_entries)
        self.max_prompt_tokens_per_request = int(max_prompt_tokens_per_request)
        self.max_output_tokens_per_request = int(max_output_tokens_per_request)
        self.max_resident_token_ids = int(max_resident_token_ids)
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            self.path,
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            self._connection.execute("PRAGMA foreign_keys=ON")
            journal_mode = self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(journal_mode).lower() != "wal":
                raise RuntimeError("recovery journal requires SQLite WAL mode")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._initialize_schema()
            if os.name != "nt":
                self.path.chmod(0o600)
        except BaseException:
            self._connection.close()
            raise

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        self._require_open()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def _initialize_schema(self) -> None:
        with self._lock, self._write_transaction():
            for statement in (
                """
                CREATE TABLE IF NOT EXISTS recovery_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS recovery_requests (
                    request_id TEXT PRIMARY KEY,
                    spec_json TEXT NOT NULL,
                    effective_recovery_level TEXT NOT NULL,
                    state TEXT NOT NULL,
                    epoch INTEGER NOT NULL CHECK (epoch >= 0),
                    route_ids_json TEXT NOT NULL,
                    prompt_checksum TEXT NOT NULL,
                    sequence_checksum TEXT NOT NULL,
                    rng_position INTEGER NOT NULL CHECK (rng_position >= 0),
                    recovery_degradation_reason TEXT,
                    failure TEXT,
                    prompt_token_count INTEGER NOT NULL CHECK (prompt_token_count > 0),
                    output_token_count INTEGER NOT NULL CHECK (output_token_count >= 0),
                    updated_at_ns INTEGER NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS recovery_output_tokens (
                    request_id TEXT NOT NULL
                        REFERENCES recovery_requests(request_id) ON DELETE CASCADE,
                    position INTEGER NOT NULL CHECK (position >= 0),
                    token_id INTEGER NOT NULL CHECK (token_id >= 0 AND token_id <= 4294967295),
                    PRIMARY KEY (request_id, position)
                ) WITHOUT ROWID
                """,
                """
                CREATE INDEX IF NOT EXISTS recovery_requests_state_updated
                    ON recovery_requests(state, updated_at_ns)
                """,
            ):
                self._connection.execute(statement)
            row = self._connection.execute(
                "SELECT schema_version FROM recovery_meta WHERE singleton = 1"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO recovery_meta(singleton, schema_version) VALUES (1, ?)",
                    (_SCHEMA_VERSION,),
                )
            elif int(row["schema_version"]) != _SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported recovery journal schema version {row['schema_version']}"
                )

    def begin(self, spec: RequestRecoverySpec) -> RequestRecoverySnapshot:
        """Create a durable PREFILLING entry, idempotently for the exact spec."""

        if len(spec.prompt_token_ids) > self.max_prompt_tokens_per_request:
            raise RecoveryJournalCapacityError("prompt exceeds the per-request journal bound")
        with self._lock, self._write_transaction():
            existing = self._request_row(spec.request_id)
            if existing is not None:
                snapshot = self._snapshot_from_row(existing)
                if snapshot.spec == spec:
                    return snapshot
                raise RecoveryConflict("request id already belongs to a different recovery spec")
            self._evict_until_capacity(len(spec.prompt_token_ids))
            counts = self._capacity_counts()
            if counts["entries"] >= self.max_entries:
                raise RecoveryJournalCapacityError("recovery journal request bound is full")
            if (
                counts["resident_token_ids"] + len(spec.prompt_token_ids)
                > self.max_resident_token_ids
            ):
                raise RecoveryJournalCapacityError("recovery journal token bound is full")

            prompt_checksum = token_sequence_checksum(spec.prompt_token_ids)
            self._connection.execute(
                """
                INSERT INTO recovery_requests(
                    request_id, spec_json, effective_recovery_level, state, epoch,
                    route_ids_json, prompt_checksum, sequence_checksum, rng_position,
                    recovery_degradation_reason, failure, prompt_token_count,
                    output_token_count, updated_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, 0, ?)
                """,
                (
                    spec.request_id,
                    _serialize_spec(spec),
                    spec.recovery_level.value,
                    RecoveryState.PREFILLING.value,
                    spec.epoch,
                    _canonical_json([spec.primary_route_id]),
                    prompt_checksum,
                    prompt_checksum,
                    len(spec.prompt_token_ids),
                    time.time_ns(),
                ),
            )
            return self._require_snapshot(spec.request_id)

    def get(self, request_id: str) -> RequestRecoverySnapshot | None:
        with self._lock:
            self._require_open()
            row = self._request_row(str(request_id))
            return None if row is None else self._snapshot_from_row(row)

    def committed_position(self, request_id: str) -> int | None:
        """Read only the append position without materializing token history."""

        with self._lock:
            self._require_open()
            row = self._connection.execute(
                """
                SELECT output_token_count FROM recovery_requests
                WHERE request_id = ?
                """,
                (str(request_id),),
            ).fetchone()
            return None if row is None else int(row["output_token_count"])

    def commit_prefill(
        self,
        request_id: str,
        *,
        epoch: int,
        prompt_checksum: str,
    ) -> RequestRecoverySnapshot:
        with self._lock, self._write_transaction():
            current = self._require_snapshot(str(request_id))
            _require_epoch(current, epoch)
            if prompt_checksum != current.prompt_checksum:
                raise RecoveryConflict("prefill prompt checksum does not match the journal")
            if current.state == RecoveryState.DECODING:
                return current
            if current.state != RecoveryState.PREFILLING:
                raise RecoveryConflict(f"cannot commit prefill while {current.state.value}")
            self._update_request(
                current.spec.request_id,
                state=RecoveryState.DECODING.value,
            )
            return self._require_snapshot(current.spec.request_id)

    def commit_token(
        self,
        request_id: str,
        *,
        epoch: int,
        position: int,
        token_id: int,
        rng_position: int = 0,
    ) -> RequestRecoverySnapshot:
        """Commit one output token and return the complete durable snapshot."""

        self.commit_tokens(
            request_id,
            epoch=epoch,
            position=position,
            token_ids=(token_id,),
            rng_position=rng_position,
        )
        snapshot = self.get(str(request_id))
        if snapshot is None:  # pragma: no cover - guarded by the transaction above
            raise RecoveryConflict("request is not present in the recovery journal")
        return snapshot

    def commit_tokens(
        self,
        request_id: str,
        *,
        epoch: int,
        position: int,
        token_ids: tuple[int, ...],
        rng_position: int = 0,
    ) -> None:
        """Atomically commit a contiguous SSE token batch before publication."""

        if position < 0 or rng_position < 0:
            raise ValueError("token and RNG positions must be non-negative")
        _validate_token_ids(token_ids)
        if not token_ids:
            return
        request_id = str(request_id)
        with self._lock, self._write_transaction():
            row = self._require_row(request_id)
            row_epoch = int(row["epoch"])
            if epoch != row_epoch:
                raise StaleRecoveryEpoch(
                    f"request epoch {epoch} does not match journal epoch {row_epoch}"
                )
            committed_position = int(row["output_token_count"])
            if position < committed_position:
                end = position + len(token_ids)
                if end > committed_position:
                    raise RecoveryConflict("token retry overlaps the committed boundary")
                existing = tuple(
                    int(item["token_id"])
                    for item in self._connection.execute(
                        """
                        SELECT token_id FROM recovery_output_tokens
                        WHERE request_id = ? AND position >= ? AND position < ?
                        ORDER BY position
                        """,
                        (request_id, position, end),
                    )
                )
                if existing == token_ids:
                    return
                raise RecoveryConflict("token position already contains a different token")
            if RecoveryState(str(row["state"])) != RecoveryState.DECODING:
                raise RecoveryConflict(f"cannot commit token while {row['state']}")
            if position != committed_position:
                raise RecoveryConflict("token commits must be contiguous")
            if committed_position + len(token_ids) > self.max_output_tokens_per_request:
                raise RecoveryJournalCapacityError("output exceeds the per-request journal bound")
            counts = self._capacity_counts()
            if counts["resident_token_ids"] + len(token_ids) > self.max_resident_token_ids:
                raise RecoveryJournalCapacityError("recovery journal token bound is full")

            spec = _deserialize_spec(str(row["spec_json"]))
            previous_rng_position = int(row["rng_position"])
            if spec.sampling.mode == SamplingReplayMode.GREEDY:
                if rng_position != 0:
                    raise RecoveryConflict("greedy generation cannot advance RNG state")
            elif rng_position <= previous_rng_position:
                raise RecoveryConflict("seeded sampling RNG position must advance per token")

            checksum = str(row["sequence_checksum"])
            for offset, token_id in enumerate(token_ids):
                token_position = position + offset
                checksum = extend_token_sequence_checksum(
                    checksum,
                    position=token_position,
                    token_id=token_id,
                )
                self._connection.execute(
                    """
                    INSERT INTO recovery_output_tokens(request_id, position, token_id)
                    VALUES (?, ?, ?)
                    """,
                    (request_id, token_position, token_id),
                )
            self._update_request(
                request_id,
                sequence_checksum=checksum,
                rng_position=rng_position,
                output_token_count=committed_position + len(token_ids),
            )

    def downgrade_to_restartable(
        self,
        request_id: str,
        *,
        epoch: int,
        reason: str,
    ) -> RequestRecoverySnapshot:
        bounded_reason = str(reason).strip()[:256]
        if not bounded_reason:
            raise ValueError("recovery downgrade reason must not be empty")
        with self._lock, self._write_transaction():
            current = self._require_snapshot(str(request_id))
            _require_epoch(current, epoch)
            if current.state not in {RecoveryState.PREFILLING, RecoveryState.DECODING}:
                raise RecoveryConflict(f"cannot downgrade recovery while {current.state.value}")
            if current.effective_recovery_level == RecoveryLevel.RESTARTABLE:
                return current
            self._update_request(
                current.spec.request_id,
                effective_recovery_level=RecoveryLevel.RESTARTABLE.value,
                recovery_degradation_reason=bounded_reason,
            )
            return self._require_snapshot(current.spec.request_id)

    def begin_recovery(
        self,
        request_id: str,
        *,
        failed_epoch: int,
        new_epoch: int,
        replacement_route_id: str,
        retain_recovery_level: bool = False,
    ) -> RequestRecoverySnapshot:
        """Fence the failed route and bind replay to a newer route epoch."""

        if not replacement_route_id:
            raise ValueError("replacement_route_id must not be empty")
        if new_epoch <= failed_epoch:
            raise ValueError("replacement epoch must be newer than the failed epoch")
        with self._lock, self._write_transaction():
            current = self._require_snapshot(str(request_id))
            if current.epoch > failed_epoch:
                if (
                    current.state == RecoveryState.RECOVERING
                    and current.epoch == new_epoch
                    and current.route_ids[-1] == replacement_route_id
                ):
                    return current
                raise StaleRecoveryEpoch(
                    f"failed epoch {failed_epoch} is older than journal epoch {current.epoch}"
                )
            _require_epoch(current, failed_epoch)
            if current.effective_recovery_level != RecoveryLevel.RECOVERABLE:
                raise RecoveryConflict("request was not admitted as recoverable")
            recoverable_states = {RecoveryState.PREFILLING, RecoveryState.DECODING}
            if retain_recovery_level:
                recoverable_states.add(RecoveryState.RECOVERING)
            if current.state not in recoverable_states:
                raise RecoveryConflict(f"cannot begin recovery while {current.state.value}")
            self._update_request(
                current.spec.request_id,
                state=RecoveryState.RECOVERING.value,
                epoch=new_epoch,
                route_ids_json=_canonical_json([*current.route_ids, replacement_route_id]),
                effective_recovery_level=(
                    RecoveryLevel.RECOVERABLE.value
                    if retain_recovery_level
                    else RecoveryLevel.RESTARTABLE.value
                ),
            )
            return self._require_snapshot(current.spec.request_id)

    def complete_replay(
        self,
        request_id: str,
        *,
        epoch: int,
        sequence_checksum: str,
        rng_position: int,
    ) -> RequestRecoverySnapshot:
        with self._lock, self._write_transaction():
            current = self._require_snapshot(str(request_id))
            _require_epoch(current, epoch)
            if current.state != RecoveryState.RECOVERING:
                raise RecoveryConflict(f"cannot complete replay while {current.state.value}")
            if sequence_checksum != current.sequence_checksum:
                raise RecoveryConflict("replacement sequence checksum does not match the journal")
            if rng_position != current.rng_position:
                raise RecoveryConflict("replacement RNG position does not match the journal")
            self._update_request(
                current.spec.request_id,
                state=RecoveryState.DECODING.value,
            )
            return self._require_snapshot(current.spec.request_id)

    def finish(
        self,
        request_id: str,
        *,
        epoch: int,
        state: RecoveryState,
        failure: str | None = None,
    ) -> RequestRecoverySnapshot:
        if state not in _TERMINAL_STATES:
            raise ValueError("finish requires a terminal state")
        with self._lock, self._write_transaction():
            current = self._require_snapshot(str(request_id))
            _require_epoch(current, epoch)
            if current.state in _TERMINAL_STATES:
                if current.state == state and current.failure == failure:
                    return current
                raise RecoveryConflict("request already has a different terminal outcome")
            if state == RecoveryState.COMPLETED and current.state != RecoveryState.DECODING:
                raise RecoveryConflict("only a decoding request can complete successfully")
            bounded_failure = None if failure is None else str(failure)[:256]
            self._update_request(
                current.spec.request_id,
                state=state.value,
                failure=bounded_failure,
            )
            snapshot = self._require_snapshot(current.spec.request_id)
            self._trim_terminal_history()
            return snapshot

    def abort_unfinished(self, reason: str) -> int:
        """Mark streams left active by a previous process as explicitly aborted."""

        bounded_reason = str(reason).strip()[:256]
        if not bounded_reason:
            raise ValueError("abort reason must not be empty")
        with self._lock, self._write_transaction():
            placeholders = ",".join("?" for _state in _ACTIVE_STATES)
            cursor = self._connection.execute(
                f"""
                UPDATE recovery_requests
                SET state = ?, failure = ?, updated_at_ns = ?
                WHERE state IN ({placeholders})
                """,
                (
                    RecoveryState.ABORTED.value,
                    bounded_reason,
                    time.time_ns(),
                    *(state.value for state in _ACTIVE_STATES),
                ),
            )
            self._trim_terminal_history()
            return max(0, int(cursor.rowcount))

    def status(self) -> dict[str, int]:
        with self._lock:
            self._require_open()
            counts = {state.value: 0 for state in RecoveryState}
            for row in self._connection.execute(
                "SELECT state, COUNT(*) AS count FROM recovery_requests GROUP BY state"
            ):
                counts[RecoveryState(str(row["state"])).value] = int(row["count"])
            capacity = self._capacity_counts()
            return {
                **counts,
                **capacity,
                "max_entries": self.max_entries,
                "max_resident_token_ids": self.max_resident_token_ids,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._connection.close()
            self._closed = True

    def _request_row(self, request_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT rowid, * FROM recovery_requests WHERE request_id = ?",
            (str(request_id),),
        ).fetchone()

    def _require_row(self, request_id: str) -> sqlite3.Row:
        row = self._request_row(request_id)
        if row is None:
            raise RecoveryConflict("request is not present in the recovery journal")
        return row

    def _require_snapshot(self, request_id: str) -> RequestRecoverySnapshot:
        return self._snapshot_from_row(self._require_row(request_id))

    def _snapshot_from_row(self, row: sqlite3.Row) -> RequestRecoverySnapshot:
        spec = _deserialize_spec(str(row["spec_json"]))
        token_rows = tuple(
            self._connection.execute(
                """
                SELECT position, token_id FROM recovery_output_tokens
                WHERE request_id = ? ORDER BY position
                """,
                (spec.request_id,),
            )
        )
        output_count = int(row["output_token_count"])
        if len(token_rows) != output_count or any(
            int(item["position"]) != position for position, item in enumerate(token_rows)
        ):
            raise RecoveryConflict("durable recovery token sequence is not contiguous")
        output_token_ids = tuple(int(item["token_id"]) for item in token_rows)
        prompt_checksum = token_sequence_checksum(spec.prompt_token_ids)
        sequence_checksum = replay_sequence_checksum(spec.prompt_token_ids, output_token_ids)
        if prompt_checksum != str(row["prompt_checksum"]) or sequence_checksum != str(
            row["sequence_checksum"]
        ):
            raise RecoveryConflict("durable recovery checksum verification failed")
        try:
            route_ids_value = json.loads(str(row["route_ids_json"]))
        except json.JSONDecodeError as error:
            raise RecoveryConflict("durable recovery route history is invalid") from error
        if (
            not isinstance(route_ids_value, list)
            or not route_ids_value
            or any(not isinstance(route_id, str) or not route_id for route_id in route_ids_value)
        ):
            raise RecoveryConflict("durable recovery route history is invalid")
        return RequestRecoverySnapshot(
            spec=spec,
            effective_recovery_level=RecoveryLevel(str(row["effective_recovery_level"])),
            state=RecoveryState(str(row["state"])),
            epoch=int(row["epoch"]),
            route_ids=tuple(route_ids_value),
            prompt_checksum=prompt_checksum,
            sequence_checksum=sequence_checksum,
            committed_output_token_ids=output_token_ids,
            rng_position=int(row["rng_position"]),
            recovery_degradation_reason=row["recovery_degradation_reason"],
            failure=row["failure"],
        )

    def _update_request(self, request_id: str, **values: object) -> None:
        if not values:
            return
        allowed = {
            "effective_recovery_level",
            "state",
            "epoch",
            "route_ids_json",
            "sequence_checksum",
            "rng_position",
            "recovery_degradation_reason",
            "failure",
            "output_token_count",
        }
        if not set(values).issubset(allowed):
            raise RuntimeError("attempted to update an unsupported recovery journal column")
        values["updated_at_ns"] = time.time_ns()
        assignments = ", ".join(f"{name} = ?" for name in values)
        parameters = (*values.values(), request_id)
        cursor = self._connection.execute(
            f"UPDATE recovery_requests SET {assignments} WHERE request_id = ?",
            parameters,
        )
        if cursor.rowcount != 1:
            raise RecoveryConflict("request is not present in the recovery journal")

    def _capacity_counts(self) -> dict[str, int]:
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS entries,
                   COALESCE(SUM(prompt_token_count + output_token_count), 0)
                       AS resident_token_ids
            FROM recovery_requests
            """
        ).fetchone()
        return {
            "entries": int(row["entries"]),
            "resident_token_ids": int(row["resident_token_ids"]),
        }

    def _evict_until_capacity(self, required_token_ids: int) -> None:
        terminal_values = tuple(state.value for state in _TERMINAL_STATES)
        placeholders = ",".join("?" for _state in terminal_values)
        while True:
            counts = self._capacity_counts()
            if (
                counts["entries"] < self.max_entries
                and counts["resident_token_ids"] + required_token_ids <= self.max_resident_token_ids
            ):
                return
            row = self._connection.execute(
                f"""
                SELECT request_id FROM recovery_requests
                WHERE state IN ({placeholders})
                ORDER BY updated_at_ns, rowid LIMIT 1
                """,
                terminal_values,
            ).fetchone()
            if row is None:
                return
            self._connection.execute(
                "DELETE FROM recovery_requests WHERE request_id = ?",
                (str(row["request_id"]),),
            )

    def _trim_terminal_history(self) -> None:
        terminal_values = tuple(state.value for state in _TERMINAL_STATES)
        placeholders = ",".join("?" for _state in terminal_values)
        rows = tuple(
            self._connection.execute(
                f"""
                SELECT request_id FROM recovery_requests
                WHERE state IN ({placeholders})
                ORDER BY updated_at_ns DESC, rowid DESC
                LIMIT -1 OFFSET ?
                """,
                (*terminal_values, self.max_terminal_entries),
            )
        )
        for row in rows:
            self._connection.execute(
                "DELETE FROM recovery_requests WHERE request_id = ?",
                (str(row["request_id"]),),
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("recovery journal is closed")


def _serialize_spec(spec: RequestRecoverySpec) -> str:
    return _canonical_json(
        {
            "attention_kv_contract_hash": spec.attention_kv_contract_hash,
            "dtype": spec.dtype,
            "epoch": spec.epoch,
            "immutable_revision": spec.immutable_revision,
            "model_swarm_id": spec.model_swarm_id,
            "prefill_contract_hash": spec.prefill_contract_hash,
            "primary_route_id": spec.primary_route_id,
            "prompt_token_ids": list(spec.prompt_token_ids),
            "recovery_level": spec.recovery_level.value,
            "request_id": spec.request_id,
            "reserved_context_tokens": spec.reserved_context_tokens,
            "sampling": {
                "mode": spec.sampling.mode.value,
                "params_hash": spec.sampling.params_hash,
                "params_json": spec.sampling.params_json,
                "seed": spec.sampling.seed,
            },
            "tokenizer_hash": spec.tokenizer_hash,
        }
    )


def _deserialize_spec(value: str) -> RequestRecoverySpec:
    try:
        payload = json.loads(value)
        sampling_payload = payload.pop("sampling")
        prompt_token_ids = payload.pop("prompt_token_ids")
        payload["sampling"] = SamplingReplayContract(
            params_json=sampling_payload["params_json"],
            params_hash=sampling_payload["params_hash"],
            mode=SamplingReplayMode(sampling_payload["mode"]),
            seed=sampling_payload["seed"],
        )
        payload["prompt_token_ids"] = tuple(prompt_token_ids)
        payload["recovery_level"] = RecoveryLevel(payload["recovery_level"])
        return RequestRecoverySpec(**payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RecoveryConflict("durable recovery specification is invalid") from error


def _require_epoch(current: RequestRecoverySnapshot, epoch: int) -> None:
    if epoch != current.epoch:
        raise StaleRecoveryEpoch(
            f"request epoch {epoch} does not match journal epoch {current.epoch}"
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validate_token_ids(token_ids: tuple[int, ...]) -> None:
    for token_id in token_ids:
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id > 2**32 - 1
        ):
            raise ValueError("token ids must be unsigned 32-bit integers")
