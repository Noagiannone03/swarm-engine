"""Fail-closed contracts for portable Skippy recovery checkpoints.

KV movement is an optimisation, never the source of truth.  A checkpoint is
usable only when its complete model/runtime/stage identity matches the target
and its token-prefix checksum matches the durable recovery journal.  Any
mismatch sends the request through the existing exact cold-replay path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum

from swarm_protocol.recovery import token_sequence_checksum

_IDENTITY_DOMAIN = b"fabi-swarm-v3-kv-snapshot-identity\0"
_MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024 * 1024


class KvStateKind(str, Enum):
    """Exact continuation state carried by a checkpoint."""

    DENSE_ATTENTION_KV = "dense_attention_kv"
    KV_RECURRENT = "kv_recurrent"


@dataclass(frozen=True)
class KvSnapshotCompatibility:
    """Everything that must match before native state import is attempted."""

    model_swarm_id: str
    immutable_revision: str
    tokenizer_hash: str
    dtype: str
    prefill_contract_hash: str
    attention_kv_contract_hash: str
    execution_plan_id: str
    package_source_sha256: str
    runtime_release: str
    runtime_abi_version: str
    layer_start: int
    layer_end: int
    state_kind: KvStateKind
    page_version: int
    k_type: int
    v_type: int
    k_row_bytes: int
    v_row_bytes: int
    v_element_bytes: int
    flags: int = 0

    def __post_init__(self) -> None:
        for name in (
            "model_swarm_id",
            "tokenizer_hash",
            "prefill_contract_hash",
            "attention_kv_contract_hash",
            "package_source_sha256",
        ):
            _validate_sha256(name, getattr(self, name))
        for name in (
            "immutable_revision",
            "dtype",
            "execution_plan_id",
            "runtime_release",
            "runtime_abi_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must not be empty")
        if self.layer_start < 0 or self.layer_end <= self.layer_start:
            raise ValueError("snapshot compatibility has an invalid layer range")
        if not isinstance(self.state_kind, KvStateKind):
            raise TypeError("state_kind must be a KvStateKind")
        if self.page_version <= 0:
            raise ValueError("snapshot page version must be positive")
        for name in ("k_type", "v_type", "k_row_bytes", "v_row_bytes", "v_element_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.k_row_bytes == 0 or self.v_row_bytes == 0 or self.v_element_bytes == 0:
            raise ValueError("snapshot row and element widths must be positive")
        if self.flags < 0:
            raise ValueError("snapshot flags must be non-negative")

    @property
    def layer_count(self) -> int:
        return self.layer_end - self.layer_start

    @property
    def identity_hash(self) -> str:
        """Canonical identity used as authenticated-data by the snapshot store."""

        payload = asdict(self)
        payload["state_kind"] = self.state_kind.value
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(_IDENTITY_DOMAIN + canonical).hexdigest()
        return digest

    def require_compatible(self, target: KvSnapshotCompatibility) -> None:
        """Reject a target unless every native continuation invariant matches."""

        if not isinstance(target, KvSnapshotCompatibility) or self != target:
            raise KvSnapshotIncompatible("replacement worker has a different KV identity")


@dataclass(frozen=True)
class KvSnapshotEnvelope:
    """Bounded metadata for one exact checkpoint payload."""

    compatibility: KvSnapshotCompatibility
    request_id: str
    source_worker_id: str
    source_route_id: str
    source_epoch: int
    checkpoint_index: int
    token_start: int
    token_count: int
    token_prefix_checksum: str
    payload_sha256: str
    payload_bytes: int

    def __post_init__(self) -> None:
        for name in ("request_id", "source_worker_id", "source_route_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must not be empty")
        if self.source_epoch < 0 or self.checkpoint_index < 0:
            raise ValueError("snapshot epoch and checkpoint index must be non-negative")
        if self.token_start != 0:
            raise ValueError("recovery checkpoints must contain the complete KV prefix")
        if self.token_count <= 0:
            raise ValueError("snapshot token count must be positive")
        _validate_sha256("token_prefix_checksum", self.token_prefix_checksum)
        _validate_sha256("payload_sha256", self.payload_sha256)
        if self.payload_bytes <= 0 or self.payload_bytes > _MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot payload size is outside the bounded wire contract")

    @classmethod
    def create(
        cls,
        *,
        compatibility: KvSnapshotCompatibility,
        request_id: str,
        source_worker_id: str,
        source_route_id: str,
        source_epoch: int,
        checkpoint_index: int,
        replay_token_ids: tuple[int, ...],
        payload: bytes,
    ) -> KvSnapshotEnvelope:
        if not replay_token_ids:
            raise ValueError("snapshot token prefix must not be empty")
        if not payload:
            raise ValueError("snapshot payload must not be empty")
        return cls(
            compatibility=compatibility,
            request_id=request_id,
            source_worker_id=source_worker_id,
            source_route_id=source_route_id,
            source_epoch=source_epoch,
            checkpoint_index=checkpoint_index,
            token_start=0,
            token_count=len(replay_token_ids),
            token_prefix_checksum=token_sequence_checksum(replay_token_ids),
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            payload_bytes=len(payload),
        )

    def validate_payload(self, payload: bytes) -> None:
        if len(payload) != self.payload_bytes:
            raise KvSnapshotIncompatible("snapshot payload length changed")
        if hashlib.sha256(payload).hexdigest() != self.payload_sha256:
            raise KvSnapshotIncompatible("snapshot payload digest changed")

    def resume_suffix(self, replay_token_ids: tuple[int, ...]) -> tuple[int, ...]:
        """Return only tokens not represented by the imported native state."""

        if self.token_count > len(replay_token_ids):
            raise KvSnapshotIncompatible("snapshot is ahead of the committed journal")
        prefix = replay_token_ids[: self.token_count]
        if token_sequence_checksum(prefix) != self.token_prefix_checksum:
            raise KvSnapshotIncompatible("snapshot token prefix differs from the journal")
        return replay_token_ids[self.token_count :]


class KvSnapshotIncompatible(ValueError):
    """A warm checkpoint cannot safely restore this target or token journal."""


def _validate_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from exc
    if len(decoded) != 32 or value.lower() != value:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
