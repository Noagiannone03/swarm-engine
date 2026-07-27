"""Epoch-fenced request journal for exact generation recovery.

The journal is the control-plane source of truth for replay.  It stores token
ids, never reconstructed SSE text, and advances through a strict state machine.
Data-plane integration will commit each token here before exposing it to the
client; a replacement route can then rebuild KV from the exact committed
sequence.

This first implementation is intentionally process-local.  It is bounded and
does not persist prompt contents to disk.  Crash recovery of the routing server
requires a separately encrypted/durable journal and is not implied here.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import threading
from typing import Any, Mapping

from swarm_protocol.contracts import RecoveryLevel

_CHECKSUM_DOMAIN = b"fabi-swarm-v3-token-sequence\0"
_OUTPUT_CHECKSUM_DOMAIN = b"fabi-swarm-v3-output-token\0"
_TOKEN_ID_LIMIT = 2**32 - 1


class RecoveryJournalError(RuntimeError):
    """Base error for invalid or unsafe recovery transitions."""


class RecoveryConflict(RecoveryJournalError):
    """The operation conflicts with already committed request state."""


class StaleRecoveryEpoch(RecoveryJournalError):
    """A superseded route attempted to mutate the request journal."""


class RecoveryJournalCapacityError(RecoveryJournalError):
    """The bounded in-memory journal cannot safely admit another request."""


class RecoveryState(str, Enum):
    PREFILLING = "prefilling"
    DECODING = "decoding"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class SamplingReplayMode(str, Enum):
    GREEDY = "greedy"
    SEEDED = "seeded"


@dataclass(frozen=True)
class SamplingReplayContract:
    """Minimal sampling identity required to reproduce a committed sequence."""

    params_json: str
    params_hash: str
    mode: SamplingReplayMode
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.params_json, str) or len(self.params_json.encode("utf-8")) > 65_536:
            raise ValueError("params_json must be a bounded canonical JSON string")
        try:
            params = json.loads(self.params_json)
        except json.JSONDecodeError as exc:
            raise ValueError("params_json must be valid JSON") from exc
        if not isinstance(params, dict) or _canonical_json(params) != self.params_json:
            raise ValueError("params_json must be a canonical JSON object")
        _validate_sha256("params_hash", self.params_hash)
        if hashlib.sha256(self.params_json.encode("utf-8")).hexdigest() != self.params_hash:
            raise ValueError("params_hash does not match params_json")
        if not isinstance(self.mode, SamplingReplayMode):
            raise ValueError("mode must be a SamplingReplayMode")
        if self.mode == SamplingReplayMode.SEEDED:
            if self.seed is None or self.seed < 0 or self.seed > 2**64 - 1:
                raise ValueError("seeded sampling requires an unsigned 64-bit seed")
        elif self.seed is not None:
            raise ValueError("greedy sampling must not carry RNG seed state")


@dataclass(frozen=True)
class RequestRecoverySpec:
    """Immutable replay identity captured before prefill starts."""

    request_id: str
    model_swarm_id: str
    immutable_revision: str
    tokenizer_hash: str
    dtype: str
    prefill_contract_hash: str
    attention_kv_contract_hash: str
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingReplayContract
    recovery_level: RecoveryLevel
    primary_route_id: str
    epoch: int
    reserved_context_tokens: int

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("immutable_revision", self.immutable_revision),
            ("dtype", self.dtype),
            ("primary_route_id", self.primary_route_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must not be empty")
        _validate_sha256("model_swarm_id", self.model_swarm_id)
        _validate_sha256("tokenizer_hash", self.tokenizer_hash)
        _validate_sha256("prefill_contract_hash", self.prefill_contract_hash)
        _validate_sha256("attention_kv_contract_hash", self.attention_kv_contract_hash)
        if not isinstance(self.prompt_token_ids, tuple):
            raise ValueError("prompt_token_ids must be an immutable tuple")
        if not isinstance(self.recovery_level, RecoveryLevel):
            raise ValueError("recovery_level must be a RecoveryLevel")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        _validate_token_ids(self.prompt_token_ids)
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        if self.reserved_context_tokens < len(self.prompt_token_ids):
            raise ValueError("reserved context must contain the complete prompt")


@dataclass(frozen=True)
class RequestRecoverySnapshot:
    """Immutable internal replay view of one request."""

    spec: RequestRecoverySpec
    effective_recovery_level: RecoveryLevel
    state: RecoveryState
    epoch: int
    route_ids: tuple[str, ...]
    prompt_checksum: str
    sequence_checksum: str
    committed_output_token_ids: tuple[int, ...]
    rng_position: int
    recovery_degradation_reason: str | None = None
    failure: str | None = None

    @property
    def committed_position(self) -> int:
        return len(self.committed_output_token_ids)

    @property
    def replay_token_ids(self) -> tuple[int, ...]:
        return self.spec.prompt_token_ids + self.committed_output_token_ids


def token_sequence_checksum(token_ids: tuple[int, ...]) -> str:
    """Hash an exact prompt token sequence with explicit framing."""

    _validate_token_ids(token_ids)
    digest = hashlib.sha256()
    digest.update(_CHECKSUM_DOMAIN)
    digest.update(len(token_ids).to_bytes(8, "big"))
    for token_id in token_ids:
        digest.update(token_id.to_bytes(4, "big"))
    return digest.hexdigest()


def extend_token_sequence_checksum(
    previous_checksum: str,
    *,
    position: int,
    token_id: int,
) -> str:
    """Append one committed output token to the checksum chain in O(1)."""

    _validate_sha256("previous_checksum", previous_checksum)
    _validate_token_ids((token_id,))
    if position < 0:
        raise ValueError("token position must be non-negative")
    digest = hashlib.sha256()
    digest.update(_OUTPUT_CHECKSUM_DOMAIN)
    digest.update(bytes.fromhex(previous_checksum))
    digest.update(position.to_bytes(8, "big"))
    digest.update(token_id.to_bytes(4, "big"))
    return digest.hexdigest()


def replay_sequence_checksum(
    prompt_token_ids: tuple[int, ...],
    output_token_ids: tuple[int, ...],
) -> str:
    """Recompute the journal checksum from a replayed prompt and output."""

    checksum = token_sequence_checksum(prompt_token_ids)
    for position, token_id in enumerate(output_token_ids):
        checksum = extend_token_sequence_checksum(
            checksum,
            position=position,
            token_id=token_id,
        )
    return checksum


_SAMPLING_FIELDS = (
    "allowed_token_ids",
    "bad_words",
    "best_of",
    "early_stopping",
    "frequency_penalty",
    "guided_choice",
    "guided_decoding_backend",
    "guided_grammar",
    "guided_json",
    "guided_regex",
    "ignore_eos",
    "include_stop_str_in_output",
    "length_penalty",
    "logit_bias",
    "max_completion_tokens",
    "max_tokens",
    "min_p",
    "min_tokens",
    "n",
    "presence_penalty",
    "repetition_detection",
    "repetition_penalty",
    "response_format",
    "seed",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "stop",
    "stop_token_ids",
    "structured_outputs",
    "temperature",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
    "use_beam_search",
)


def sampling_replay_contract(
    request_data: Mapping[str, Any],
) -> SamplingReplayContract | None:
    """Build an exact replay contract only for deterministic generation.

    Parallax does not yet propagate a portable RNG state through MLX, vLLM and
    SGLang. Until that wire contract exists, sampled requests remain
    ``restartable`` while temperature-zero or top-k-one requests can use exact
    cold replay.
    """

    n = request_data.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n != 1:
        return None
    temperature = request_data.get("temperature")
    top_k = request_data.get("top_k")
    temperature_is_zero = (
        not isinstance(temperature, bool)
        and isinstance(temperature, (int, float))
        and float(temperature) == 0.0
    )
    top_k_is_one = not isinstance(top_k, bool) and isinstance(top_k, int) and top_k == 1
    if not (temperature_is_zero or top_k_is_one):
        return None

    params = {
        field: request_data[field]
        for field in _SAMPLING_FIELDS
        if field in request_data and request_data[field] is not None
    }
    try:
        params_json = _canonical_json(params)
        return SamplingReplayContract(
            params_json=params_json,
            params_hash=hashlib.sha256(params_json.encode("utf-8")).hexdigest(),
            mode=SamplingReplayMode.GREEDY,
        )
    except (TypeError, ValueError):
        return None


class InMemoryRecoveryJournal:
    """Thread-safe bounded recovery state for active generations.

    Active entries are never evicted.  When either the request-count or token
    budget is exhausted, admission fails closed.  Only terminal entries are
    removed automatically, oldest first.
    """

    def __init__(
        self,
        *,
        max_entries: int = 1_024,
        max_terminal_entries: int = 64,
        max_prompt_tokens_per_request: int = 262_144,
        max_output_tokens_per_request: int = 65_536,
        max_resident_token_ids: int = 2_000_000,
    ) -> None:
        for name, value in (
            ("max_entries", max_entries),
            ("max_prompt_tokens_per_request", max_prompt_tokens_per_request),
            ("max_output_tokens_per_request", max_output_tokens_per_request),
            ("max_resident_token_ids", max_resident_token_ids),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if max_terminal_entries < 0 or max_terminal_entries > max_entries:
            raise ValueError("max_terminal_entries must be between zero and max_entries")
        self.max_entries = int(max_entries)
        self.max_terminal_entries = int(max_terminal_entries)
        self.max_prompt_tokens_per_request = int(max_prompt_tokens_per_request)
        self.max_output_tokens_per_request = int(max_output_tokens_per_request)
        self.max_resident_token_ids = int(max_resident_token_ids)
        self._entries: OrderedDict[str, RequestRecoverySnapshot] = OrderedDict()
        self._resident_token_ids = 0
        self._lock = threading.RLock()

    def begin(self, spec: RequestRecoverySpec) -> RequestRecoverySnapshot:
        """Create a PREFILLING journal entry, idempotently for the exact spec."""

        if len(spec.prompt_token_ids) > self.max_prompt_tokens_per_request:
            raise RecoveryJournalCapacityError("prompt exceeds the per-request journal bound")
        with self._lock:
            existing = self._entries.get(spec.request_id)
            if existing is not None:
                if existing.spec == spec:
                    return existing
                raise RecoveryConflict("request id already belongs to a different recovery spec")
            self._evict_terminal_locked(required_token_ids=len(spec.prompt_token_ids))
            if len(self._entries) >= self.max_entries:
                raise RecoveryJournalCapacityError("recovery journal request bound is full")
            if self._resident_token_ids + len(spec.prompt_token_ids) > self.max_resident_token_ids:
                raise RecoveryJournalCapacityError("recovery journal token bound is full")

            prompt_checksum = token_sequence_checksum(spec.prompt_token_ids)
            snapshot = RequestRecoverySnapshot(
                spec=spec,
                effective_recovery_level=spec.recovery_level,
                state=RecoveryState.PREFILLING,
                epoch=spec.epoch,
                route_ids=(spec.primary_route_id,),
                prompt_checksum=prompt_checksum,
                sequence_checksum=prompt_checksum,
                committed_output_token_ids=(),
                rng_position=0,
            )
            self._entries[spec.request_id] = snapshot
            self._resident_token_ids += len(spec.prompt_token_ids)
            return snapshot

    def get(self, request_id: str) -> RequestRecoverySnapshot | None:
        with self._lock:
            return self._entries.get(str(request_id))

    def commit_prefill(
        self,
        request_id: str,
        *,
        epoch: int,
        prompt_checksum: str,
    ) -> RequestRecoverySnapshot:
        """Confirm that the serving route consumed the exact journalled prompt."""

        with self._lock:
            current = self._require_locked(request_id)
            self._require_epoch(current, epoch)
            if prompt_checksum != current.prompt_checksum:
                raise RecoveryConflict("prefill prompt checksum does not match the journal")
            if current.state == RecoveryState.DECODING:
                return current
            if current.state != RecoveryState.PREFILLING:
                raise RecoveryConflict(f"cannot commit prefill while {current.state.value}")
            return self._store_locked(replace(current, state=RecoveryState.DECODING))

    def commit_token(
        self,
        request_id: str,
        *,
        epoch: int,
        position: int,
        token_id: int,
        rng_position: int = 0,
    ) -> RequestRecoverySnapshot:
        """Commit one token id before the corresponding SSE bytes are emitted."""

        _validate_token_ids((token_id,))
        if position < 0 or rng_position < 0:
            raise ValueError("token and RNG positions must be non-negative")
        with self._lock:
            current = self._require_locked(request_id)
            self._require_epoch(current, epoch)
            if position < current.committed_position:
                committed = current.committed_output_token_ids[position]
                if committed == token_id:
                    return current
                raise RecoveryConflict("token position already contains a different token")
            if current.state != RecoveryState.DECODING:
                raise RecoveryConflict(f"cannot commit token while {current.state.value}")
            if position != current.committed_position:
                raise RecoveryConflict("token commits must be contiguous")
            if current.committed_position >= self.max_output_tokens_per_request:
                raise RecoveryJournalCapacityError("output exceeds the per-request journal bound")
            if self._resident_token_ids >= self.max_resident_token_ids:
                raise RecoveryJournalCapacityError("recovery journal token bound is full")
            if current.spec.sampling.mode == SamplingReplayMode.GREEDY:
                if rng_position != 0:
                    raise RecoveryConflict("greedy generation cannot advance RNG state")
            elif rng_position <= current.rng_position:
                raise RecoveryConflict("seeded sampling RNG position must advance per token")

            output = (*current.committed_output_token_ids, token_id)
            sequence_checksum = extend_token_sequence_checksum(
                current.sequence_checksum,
                position=position,
                token_id=token_id,
            )
            updated = replace(
                current,
                committed_output_token_ids=output,
                sequence_checksum=sequence_checksum,
                rng_position=rng_position,
            )
            self._resident_token_ids += 1
            return self._store_locked(updated)

    def downgrade_to_restartable(
        self,
        request_id: str,
        *,
        epoch: int,
        reason: str,
    ) -> RequestRecoverySnapshot:
        """Atomically revoke live replay when the reserved backup disappears.

        The immutable spec records what admission originally guaranteed.  The
        effective level records what the currently reserved topology can still
        provide.  A downgrade never interrupts the healthy primary route, but
        it prevents a later recovery attempt from using a guarantee that no
        longer exists.
        """

        bounded_reason = str(reason).strip()[:256]
        if not bounded_reason:
            raise ValueError("recovery downgrade reason must not be empty")
        with self._lock:
            current = self._require_locked(request_id)
            self._require_epoch(current, epoch)
            if current.state not in {
                RecoveryState.PREFILLING,
                RecoveryState.DECODING,
            }:
                raise RecoveryConflict(f"cannot downgrade recovery while {current.state.value}")
            if current.effective_recovery_level == RecoveryLevel.RESTARTABLE:
                return current
            return self._store_locked(
                replace(
                    current,
                    effective_recovery_level=RecoveryLevel.RESTARTABLE,
                    recovery_degradation_reason=bounded_reason,
                )
            )

    def begin_recovery(
        self,
        request_id: str,
        *,
        failed_epoch: int,
        new_epoch: int,
        replacement_route_id: str,
    ) -> RequestRecoverySnapshot:
        """Fence the failed route and bind replay to a newer route epoch."""

        if not replacement_route_id:
            raise ValueError("replacement_route_id must not be empty")
        if new_epoch <= failed_epoch:
            raise ValueError("replacement epoch must be newer than the failed epoch")
        with self._lock:
            current = self._require_locked(request_id)
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
            self._require_epoch(current, failed_epoch)
            if current.effective_recovery_level != RecoveryLevel.RECOVERABLE:
                raise RecoveryConflict("request was not admitted as recoverable")
            if current.state not in {RecoveryState.PREFILLING, RecoveryState.DECODING}:
                raise RecoveryConflict(f"cannot begin recovery while {current.state.value}")
            return self._store_locked(
                replace(
                    current,
                    state=RecoveryState.RECOVERING,
                    epoch=new_epoch,
                    route_ids=(*current.route_ids, replacement_route_id),
                )
            )

    def complete_replay(
        self,
        request_id: str,
        *,
        epoch: int,
        sequence_checksum: str,
        rng_position: int,
    ) -> RequestRecoverySnapshot:
        """Accept replacement KV only after exact sequence and RNG verification."""

        with self._lock:
            current = self._require_locked(request_id)
            self._require_epoch(current, epoch)
            if current.state != RecoveryState.RECOVERING:
                raise RecoveryConflict(f"cannot complete replay while {current.state.value}")
            if sequence_checksum != current.sequence_checksum:
                raise RecoveryConflict("replacement sequence checksum does not match the journal")
            if rng_position != current.rng_position:
                raise RecoveryConflict("replacement RNG position does not match the journal")
            return self._store_locked(replace(current, state=RecoveryState.DECODING))

    def finish(
        self,
        request_id: str,
        *,
        epoch: int,
        state: RecoveryState,
        failure: str | None = None,
    ) -> RequestRecoverySnapshot:
        """Move a request to one explicit terminal state."""

        if state not in {
            RecoveryState.COMPLETED,
            RecoveryState.FAILED,
            RecoveryState.ABORTED,
        }:
            raise ValueError("finish requires a terminal state")
        with self._lock:
            current = self._require_locked(request_id)
            self._require_epoch(current, epoch)
            if current.state in {
                RecoveryState.COMPLETED,
                RecoveryState.FAILED,
                RecoveryState.ABORTED,
            }:
                if current.state == state and current.failure == failure:
                    return current
                raise RecoveryConflict("request already has a different terminal outcome")
            if state == RecoveryState.COMPLETED and current.state != RecoveryState.DECODING:
                raise RecoveryConflict("only a decoding request can complete successfully")
            bounded_failure = None if failure is None else str(failure)[:256]
            snapshot = self._store_locked(replace(current, state=state, failure=bounded_failure))
            self._trim_terminal_history_locked()
            return snapshot

    def status(self) -> dict[str, int]:
        """Return aggregate diagnostics without exposing prompt or output tokens."""

        with self._lock:
            counts = {state.value: 0 for state in RecoveryState}
            for snapshot in self._entries.values():
                counts[snapshot.state.value] += 1
            return {
                **counts,
                "entries": len(self._entries),
                "resident_token_ids": self._resident_token_ids,
                "max_entries": self.max_entries,
                "max_resident_token_ids": self.max_resident_token_ids,
            }

    def _require_locked(self, request_id: str) -> RequestRecoverySnapshot:
        current = self._entries.get(str(request_id))
        if current is None:
            raise RecoveryConflict("request is not present in the recovery journal")
        return current

    @staticmethod
    def _require_epoch(current: RequestRecoverySnapshot, epoch: int) -> None:
        if epoch != current.epoch:
            raise StaleRecoveryEpoch(
                f"request epoch {epoch} does not match journal epoch {current.epoch}"
            )

    def _store_locked(self, snapshot: RequestRecoverySnapshot) -> RequestRecoverySnapshot:
        self._entries[snapshot.spec.request_id] = snapshot
        self._entries.move_to_end(snapshot.spec.request_id)
        return snapshot

    def _evict_terminal_locked(self, *, required_token_ids: int) -> None:
        terminal = {
            RecoveryState.COMPLETED,
            RecoveryState.FAILED,
            RecoveryState.ABORTED,
        }
        for request_id, snapshot in tuple(self._entries.items()):
            if (
                len(self._entries) < self.max_entries
                and self._resident_token_ids + required_token_ids <= self.max_resident_token_ids
            ):
                break
            if snapshot.state in terminal:
                self._drop_locked(request_id, snapshot)

    def _trim_terminal_history_locked(self) -> None:
        terminal_states = {
            RecoveryState.COMPLETED,
            RecoveryState.FAILED,
            RecoveryState.ABORTED,
        }
        terminal_items = [
            (request_id, snapshot)
            for request_id, snapshot in self._entries.items()
            if snapshot.state in terminal_states
        ]
        excess = len(terminal_items) - self.max_terminal_entries
        for request_id, snapshot in terminal_items[: max(0, excess)]:
            self._drop_locked(request_id, snapshot)

    def _drop_locked(
        self,
        request_id: str,
        snapshot: RequestRecoverySnapshot,
    ) -> None:
        self._entries.pop(request_id, None)
        self._resident_token_ids -= len(snapshot.replay_token_ids)
        if self._resident_token_ids < 0:
            raise RuntimeError("recovery journal token accounting underflow")


def _validate_sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


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
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ValueError("token ids must be integers")
        if token_id < 0 or token_id > _TOKEN_ID_LIMIT:
            raise ValueError("token ids must be unsigned 32-bit integers")
