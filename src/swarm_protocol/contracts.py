"""Strict domain contracts for Fabi Swarm Protocol v3.

These models deliberately contain no transport, DHT, scheduler, or backend code.  They are the
validated boundary shared by those components.  Cryptographic envelopes will sign the canonical
protobuf representation introduced with the wire protocol; ``model_swarm_id`` is already stable
because its input contains strings and integers only and is encoded as sorted compact JSON.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL_VERSION = 3

NonEmpty = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
HashHex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ContractModel(BaseModel):
    """Base model with forward-compatible but strict local semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class BackendKind(str, Enum):
    MLX = "mlx"
    VLLM = "vllm"
    SGLANG = "sglang"


class WorkerRole(str, Enum):
    EXECUTOR = "executor"
    FRONTEND = "frontend"
    WEIGHT_SEEDER = "weight_seeder"
    RELAY = "relay"
    WARM_REPLICA = "warm_replica"
    VERIFIER = "verifier"


class SpanState(str, Enum):
    BUILDING = "building"
    WARMING = "warming"
    READY = "ready"
    DRAINING = "draining"


class EffectiveSpanMode(str, Enum):
    FIXED = "fixed"
    SUBSPAN = "subspan"


class PathKind(str, Enum):
    DIRECT = "direct"
    RELAY = "relay"


class LinkMetric(ContractModel):
    from_worker_id: NonEmpty
    to_worker_id: NonEmpty
    path_kind: PathKind
    rtt_ms: Annotated[float, Field(ge=0)]
    throughput_bytes_per_second: Annotated[float, Field(gt=0)]
    loss_rate: Annotated[float, Field(ge=0, lt=1)] = 0
    measured_at_ms: NonNegativeInt
    expires_at_ms: PositiveInt

    @model_validator(mode="after")
    def validate_link(self) -> Self:
        if self.from_worker_id == self.to_worker_id:
            raise ValueError("network link must connect two different workers")
        if self.expires_at_ms <= self.measured_at_ms:
            raise ValueError("link metric must expire after it was measured")
        return self


class RecoveryLevel(str, Enum):
    NONE = "none"
    RESTARTABLE = "restartable"
    RECOVERABLE = "recoverable"


class ReservationState(str, Enum):
    PREPARED = "prepared"
    COMMITTED = "committed"
    RELEASED = "released"
    EXPIRED = "expired"


class LayerSpan(ContractModel):
    start: NonNegativeInt
    end: PositiveInt

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.end <= self.start:
            raise ValueError("layer span end must be greater than start")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, other: LayerSpan) -> bool:
        return self.start <= other.start and other.end <= self.end


class KvGeometry(ContractModel):
    block_size_tokens: PositiveInt
    bytes_per_token_per_layer: PositiveInt
    allocatable_bytes: NonNegativeInt

    def rounded_tokens(self, requested_tokens: int) -> int:
        if requested_tokens <= 0:
            raise ValueError("requested_tokens must be positive")
        blocks = (requested_tokens + self.block_size_tokens - 1) // self.block_size_tokens
        return blocks * self.block_size_tokens

    def required_bytes(self, span: LayerSpan, requested_tokens: int) -> int:
        return self.rounded_tokens(requested_tokens) * self.bytes_per_token_per_layer * span.length


class ModelManifest(ContractModel):
    protocol_version: int = PROTOCOL_VERSION
    model_id: NonEmpty
    immutable_revision: NonEmpty
    architecture_graph_hash: HashHex
    tokenizer_hash: HashHex
    weight_collection_hash: HashHex
    weight_format: NonEmpty
    quantization: NonEmpty
    dtype: NonEmpty
    num_layers: PositiveInt
    activation_bytes_per_token: PositiveInt
    rope_context_contract_hash: HashHex
    attention_kv_contract_hash: HashHex
    prefill_contract_hash: HashHex
    wire_protocol_version: PositiveInt

    @model_validator(mode="after")
    def require_protocol_version(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        return self

    @property
    def model_swarm_id(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class WorkerOffer(ContractModel):
    protocol_version: int = PROTOCOL_VERSION
    worker_id: NonEmpty
    endpoint_id: NonEmpty
    account_attestation: NonEmpty | None = None
    runtime_version: NonEmpty
    platform: NonEmpty
    backend: BackendKind
    stable_memory_envelope_bytes: PositiveInt
    execution_granularity_layers: PositiveInt = 1
    supported_roles: frozenset[WorkerRole]
    offer_seq: NonNegativeInt
    issued_at_ms: NonNegativeInt
    expires_at_ms: PositiveInt

    @model_validator(mode="after")
    def validate_offer(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if not self.supported_roles:
            raise ValueError("worker must advertise at least one contribution role")
        if self.expires_at_ms <= self.issued_at_ms:
            raise ValueError("offer must expire after it is issued")
        return self


class SpanLease(ContractModel):
    protocol_version: int = PROTOCOL_VERSION
    model_swarm_id: HashHex
    worker_id: NonEmpty
    hosted_span: LayerSpan
    effective_span_mode: EffectiveSpanMode
    state: SpanState
    weight_hashes: tuple[HashHex, ...]
    measured_prefill_tokens_per_second: Annotated[float, Field(gt=0)] | None = None
    measured_decode_tokens_per_second: Annotated[float, Field(gt=0)] | None = None
    kv_geometry: KvGeometry
    available_kv_bytes_snapshot: NonNegativeInt
    max_sessions: PositiveInt
    lease_seq: NonNegativeInt
    issued_at_ms: NonNegativeInt
    expires_at_ms: PositiveInt

    @model_validator(mode="after")
    def validate_lease(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if not self.weight_hashes:
            raise ValueError("span lease must bind at least one weight hash")
        if self.available_kv_bytes_snapshot > self.kv_geometry.allocatable_bytes:
            raise ValueError("available KV snapshot exceeds the worker's allocatable KV envelope")
        if self.expires_at_ms <= self.issued_at_ms:
            raise ValueError("span lease must expire after it is issued")
        return self


class ModelMemberAdvertisement(ContractModel):
    """Self-contained planning view carried by one signed model-membership subkey.

    This mirrors Petals' per-peer ``ServerInfo`` value: one DHT read yields the worker offer,
    hosted span and its recent outgoing link observations without an N+1 catalogue lookup.
    """

    protocol_version: int = PROTOCOL_VERSION
    offer: WorkerOffer
    lease: SpanLease
    outgoing_links: tuple[LinkMetric, ...] = ()

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if self.offer.worker_id != self.lease.worker_id:
            raise ValueError("member offer and span lease identify different workers")
        for metric in self.outgoing_links:
            if metric.from_worker_id != self.offer.worker_id:
                raise ValueError("member link source must match the advertised worker")
        return self


class RequestContract(ContractModel):
    request_id: NonEmpty
    model_swarm_id: HashHex
    prompt_tokens: PositiveInt
    reserved_output_tokens: PositiveInt
    recovery_level: RecoveryLevel = RecoveryLevel.NONE

    @property
    def required_context_tokens(self) -> int:
        return self.prompt_tokens + self.reserved_output_tokens


class RouteStage(ContractModel):
    worker_id: NonEmpty
    endpoint_id: NonEmpty
    hosted_span: LayerSpan
    effective_span: LayerSpan
    path_to_next: PathKind
    rounded_context_tokens: PositiveInt
    exact_kv_bytes: PositiveInt

    @model_validator(mode="after")
    def validate_effective_span(self) -> Self:
        if not self.hosted_span.contains(self.effective_span):
            raise ValueError("effective span must be contained in hosted span")
        return self


class RoutePlan(ContractModel):
    protocol_version: int = PROTOCOL_VERSION
    request_id: NonEmpty
    route_id: NonEmpty
    epoch: NonNegativeInt
    model_swarm_id: HashHex
    model_num_layers: PositiveInt
    prompt_tokens: PositiveInt
    reserved_output_tokens: PositiveInt
    stages: tuple[RouteStage, ...]
    recovery_level: RecoveryLevel
    coordinator_id: NonEmpty
    reservation_deadline_ms: PositiveInt
    plan_expires_at_ms: PositiveInt

    @property
    def required_context_tokens(self) -> int:
        return self.prompt_tokens + self.reserved_output_tokens

    @model_validator(mode="after")
    def validate_complete_route(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if not self.stages:
            raise ValueError("route must contain at least one stage")
        if self.stages[0].effective_span.start != 0:
            raise ValueError("route must start at layer zero")
        if self.stages[-1].effective_span.end != self.model_num_layers:
            raise ValueError("route must end at model_num_layers")
        for previous, current in zip(self.stages, self.stages[1:]):
            if previous.effective_span.end != current.effective_span.start:
                raise ValueError("route stages must form a contiguous, non-overlapping cover")
        for stage in self.stages:
            if stage.rounded_context_tokens < self.required_context_tokens:
                raise ValueError("stage KV reservation is smaller than the request context")
        if self.plan_expires_at_ms <= self.reservation_deadline_ms:
            raise ValueError("route plan must outlive its reservation deadline")
        return self


class ReservationLease(ContractModel):
    protocol_version: int = PROTOCOL_VERSION
    reservation_id: NonEmpty
    request_id: NonEmpty
    route_id: NonEmpty
    epoch: NonNegativeInt
    worker_id: NonEmpty
    effective_span: LayerSpan
    exact_kv_bytes: PositiveInt
    state: ReservationState
    issued_at_ms: NonNegativeInt
    expires_at_ms: PositiveInt

    @model_validator(mode="after")
    def validate_reservation(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if self.expires_at_ms <= self.issued_at_ms:
            raise ValueError("reservation must expire after it is issued")
        return self


class ContributionReceipt(ContractModel):
    protocol_version: int = PROTOCOL_VERSION
    receipt_id: NonEmpty
    account_id: NonEmpty
    worker_id: NonEmpty
    model_swarm_id: HashHex
    route_id: NonEmpty
    epoch: NonNegativeInt
    role: WorkerRole
    work_units: PositiveInt
    started_at_ms: NonNegativeInt
    completed_at_ms: PositiveInt
    result_digest: HashHex
    peer_acknowledgements: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if self.completed_at_ms < self.started_at_ms:
            raise ValueError("receipt completion precedes its start")
        return self
