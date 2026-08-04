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


class ArtifactRole(str, Enum):
    ARCHITECTURE = "architecture"
    TOKENIZER = "tokenizer"
    WEIGHT = "weight"


class ArtifactDescriptor(ContractModel):
    """Content-addressed model file descriptor, following the OCI descriptor shape."""

    path: Annotated[str, Field(min_length=1, max_length=1024)]
    size: NonNegativeInt
    sha256: HashHex
    media_type: Annotated[str, Field(min_length=1, max_length=255)]
    role: ArtifactRole
    # Hugging Face Xet identifies the complete file with a cryptographic
    # BLAKE3/Merkle hash.  When present in the signed TUF bundle, workers can
    # authenticate byte-range reconstruction without downloading the rest of
    # the file merely to recompute its legacy LFS SHA-256.
    xet_file_hash: HashHex | None = None

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        parts = self.path.split("/")
        if (
            self.path.startswith("/")
            or "\\" in self.path
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("artifact path must be a normalized relative POSIX path")
        return self


class TensorArtifactDescriptor(ContractModel):
    """Signed location and shape of one tensor inside a SafeTensors source file."""

    name: Annotated[str, Field(min_length=1, max_length=4096)]
    source_path: Annotated[str, Field(min_length=1, max_length=1024)]
    offset: NonNegativeInt
    length: NonNegativeInt
    sha256: HashHex
    dtype: Annotated[str, Field(min_length=1, max_length=64)]
    shape: Annotated[tuple[NonNegativeInt, ...], Field(max_length=32)]

    @model_validator(mode="after")
    def validate_tensor(self) -> Self:
        parts = self.source_path.split("/")
        if (
            self.source_path.startswith("/")
            or "\\" in self.source_path
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("tensor source path must be a normalized relative POSIX path")
        return self


class ModelArtifactIndex(ContractModel):
    """Persistent artifact index referenced by the compact DHT model manifest."""

    protocol_version: int = PROTOCOL_VERSION
    model_id: NonEmpty
    immutable_revision: NonEmpty
    artifacts: Annotated[tuple[ArtifactDescriptor, ...], Field(min_length=1, max_length=100_000)]
    tensors: Annotated[tuple[TensorArtifactDescriptor, ...], Field(max_length=1_000_000)] = ()

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if not self.artifacts:
            raise ValueError("model artifact index cannot be empty")
        paths = [artifact.path for artifact in self.artifacts]
        if paths != sorted(paths):
            raise ValueError("model artifacts must be sorted by path")
        if len(paths) != len(set(paths)):
            raise ValueError("model artifact paths must be unique")
        tensor_names = [tensor.name for tensor in self.tensors]
        if tensor_names != sorted(tensor_names):
            raise ValueError("tensor artifacts must be sorted by name")
        if len(tensor_names) != len(set(tensor_names)):
            raise ValueError("tensor artifact names must be unique")
        artifacts = {artifact.path: artifact for artifact in self.artifacts}
        for tensor in self.tensors:
            source = artifacts.get(tensor.source_path)
            if source is None or source.role is not ArtifactRole.WEIGHT:
                raise ValueError("tensor source must reference a signed weight artifact")
            if source.xet_file_hash is None:
                raise ValueError("selective tensor source must have a signed Xet file hash")
            if tensor.offset + tensor.length > source.size:
                raise ValueError("tensor byte range exceeds its signed source artifact")
        return self


class LinkMetric(ContractModel):
    """Authenticated reachability plus optional measured transfer goodput.

    Reachability is a structural routing fact; goodput is an optimization
    sample.  A healthy path must not disappear merely because its last
    bandwidth sample expired.  This follows Petals' routing model, where
    unknown latency affects the score but does not remove an online edge.
    """

    from_worker_id: NonEmpty
    to_worker_id: NonEmpty
    path_kind: PathKind
    rtt_ms: Annotated[float, Field(ge=0)]
    throughput_bytes_per_second: Annotated[float, Field(gt=0)] | None = None
    throughput_measured_at_ms: NonNegativeInt | None = None
    loss_rate: Annotated[float, Field(ge=0, lt=1)] = 0
    measured_at_ms: NonNegativeInt
    expires_at_ms: PositiveInt

    @model_validator(mode="after")
    def validate_link(self) -> Self:
        if self.from_worker_id == self.to_worker_id:
            raise ValueError("network link must connect two different workers")
        if self.expires_at_ms <= self.measured_at_ms:
            raise ValueError("link metric must expire after it was measured")
        if self.throughput_measured_at_ms is not None and self.throughput_bytes_per_second is None:
            raise ValueError("throughput timestamp requires a throughput measurement")
        # Reachability and goodput are independent observations.  A successful
        # RPC health check establishes the edge, then the bounded payload probe
        # normally completes a little later.  Conversely, later health checks
        # may refresh reachability while retaining an older goodput sample.
        # Neither ordering invalidates the other measurement.
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


class ReservationAction(str, Enum):
    COMMIT = "commit"
    RENEW = "renew"
    RELEASE = "release"
    FENCE = "fence"


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
    bytes_per_token_per_layer: PositiveInt | None = None
    bytes_per_token_by_layer: tuple[PositiveInt, ...] = ()
    allocatable_bytes: NonNegativeInt

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        if (self.bytes_per_token_per_layer is None) == (not self.bytes_per_token_by_layer):
            raise ValueError(
                "KV geometry requires exactly one uniform or per-layer byte representation"
            )
        return self

    def rounded_tokens(self, requested_tokens: int) -> int:
        if requested_tokens <= 0:
            raise ValueError("requested_tokens must be positive")
        blocks = (requested_tokens + self.block_size_tokens - 1) // self.block_size_tokens
        return blocks * self.block_size_tokens

    def required_bytes(self, span: LayerSpan, requested_tokens: int) -> int:
        if self.bytes_per_token_by_layer:
            if span.end > len(self.bytes_per_token_by_layer):
                raise ValueError("layer span exceeds per-layer KV geometry")
            bytes_per_token = sum(self.bytes_per_token_by_layer[span.start : span.end])
        else:
            assert self.bytes_per_token_per_layer is not None
            bytes_per_token = self.bytes_per_token_per_layer * span.length
        return self.rounded_tokens(requested_tokens) * bytes_per_token


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
    kv_bytes_per_token_by_layer: tuple[PositiveInt, ...]
    weight_bytes_by_layer: tuple[PositiveInt, ...] = ()
    input_endpoint_weight_bytes: NonNegativeInt = 0
    output_endpoint_weight_bytes: NonNegativeInt = 0
    shared_endpoint_weight_bytes: NonNegativeInt = 0
    rope_context_contract_hash: HashHex
    attention_kv_contract_hash: HashHex
    prefill_contract_hash: HashHex
    wire_protocol_version: PositiveInt

    @model_validator(mode="after")
    def require_protocol_version(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if len(self.kv_bytes_per_token_by_layer) != self.num_layers:
            raise ValueError("KV byte geometry must contain exactly one value per model layer")
        if self.weight_bytes_by_layer and len(self.weight_bytes_by_layer) != self.num_layers:
            raise ValueError("weight byte geometry must contain exactly one value per model layer")
        if self.shared_endpoint_weight_bytes > min(
            self.input_endpoint_weight_bytes,
            self.output_endpoint_weight_bytes,
        ):
            raise ValueError("shared endpoint weights exceed an endpoint")
        return self

    def weight_bytes(self, span: LayerSpan) -> int:
        """Return exact resident checkpoint bytes for one stage."""

        if not self.weight_bytes_by_layer:
            raise ValueError("manifest has no exact per-layer weight geometry")
        if span.end > self.num_layers:
            raise ValueError("weight span exceeds the model")
        total = sum(self.weight_bytes_by_layer[span.start : span.end])
        owns_input = span.start == 0
        owns_output = span.end == self.num_layers
        if owns_input:
            total += self.input_endpoint_weight_bytes
        if owns_output:
            total += self.output_endpoint_weight_bytes
        if owns_input and owns_output:
            total -= self.shared_endpoint_weight_bytes
        return total

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
    # Hard per-session limit enforced by the serving frontend. This must stay
    # separate from aggregate KV capacity: a worker can have room for several
    # sessions while rejecting one sequence that exceeds its engine max length.
    max_context_tokens: PositiveInt
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
        if self.kv_geometry.bytes_per_token_by_layer and self.hosted_span.end > len(
            self.kv_geometry.bytes_per_token_by_layer
        ):
            raise ValueError("hosted span exceeds the advertised per-layer KV geometry")
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
    stages: Annotated[tuple[RouteStage, ...], Field(min_length=1, max_length=256)]
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
        worker_ids = [stage.worker_id for stage in self.stages]
        endpoint_ids = [stage.endpoint_id for stage in self.stages]
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("route may use each worker at most once")
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError("route may use each endpoint at most once")
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


class ReservationCommand(ContractModel):
    """Short-lived coordinator command for an existing local reservation."""

    protocol_version: int = PROTOCOL_VERSION
    action: ReservationAction
    reservation_id: NonEmpty | None = None
    request_id: NonEmpty
    route_id: NonEmpty
    epoch: NonNegativeInt
    ttl_ms: PositiveInt | None = None
    issued_at_ms: NonNegativeInt
    expires_at_ms: PositiveInt

    @model_validator(mode="after")
    def validate_command(self) -> Self:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if self.expires_at_ms <= self.issued_at_ms:
            raise ValueError("reservation command must expire after it was issued")
        if self.action == ReservationAction.FENCE:
            if self.reservation_id is not None or self.ttl_ms is not None:
                raise ValueError("fence command must not carry a reservation id or TTL")
        else:
            if self.reservation_id is None:
                raise ValueError(f"{self.action.value} command requires a reservation id")
            if self.action == ReservationAction.RENEW and self.ttl_ms is None:
                raise ValueError("renew command requires a TTL")
            if self.action != ReservationAction.RENEW and self.ttl_ms is not None:
                raise ValueError(f"{self.action.value} command must not carry a TTL")
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
