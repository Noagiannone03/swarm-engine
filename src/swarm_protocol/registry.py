"""TUF-backed authority and client for immutable Fabi model bundles.

The DHT is an availability and discovery plane, not a trust root.  A model manifest becomes
admissible only after python-tuf has verified the pinned root chain, role thresholds, metadata
freshness and target hash.  Private keys are injected into the offline publisher API and are never
stored by the runtime.
"""

from __future__ import annotations

import json
import os
import tempfile
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator
from securesystemslib.signer import Signer
from tuf.api.exceptions import UnsignedMetadataError
from tuf.api.metadata import (
    Metadata,
    MetaFile,
    Root,
    Snapshot,
    TargetFile,
    Targets,
    Timestamp,
)
from tuf.api.serialization.json import JSONSerializer
from tuf.ngclient import Updater
from tuf.ngclient.fetcher import FetcherInterface

from swarm_protocol.contracts import (
    PROTOCOL_VERSION,
    ArtifactRole,
    ContractModel,
    HashHex,
    ModelArtifactIndex,
    ModelManifest,
    NonEmpty,
)
from swarm_protocol.model_manifest import artifact_collection_hash

_SERIALIZER = JSONSerializer(compact=True, validate=True)
_MAX_BUNDLE_BYTES = 32 * 1024 * 1024
_TOP_LEVEL_ROLES = ("root", "targets", "snapshot", "timestamp")
_CATALOG_TARGET_PATH = "catalog.json"
_ROUTE_AUTHORITIES_TARGET_PATH = "route-authorities.json"
RevocationIdentifierHex = Annotated[str, Field(pattern=r"^[0-9a-f]{128}$")]


class ModelCatalogEntry(ContractModel):
    """Human model identity mapped to one immutable execution contract."""

    model_id: NonEmpty
    immutable_revision: NonEmpty
    model_swarm_id: HashHex
    quantization: NonEmpty
    dtype: NonEmpty


class ModelRegistryCatalog(ContractModel):
    """Small signed lookup index; DHT records still use only ``model_swarm_id``."""

    protocol_version: int = PROTOCOL_VERSION
    models: tuple[ModelCatalogEntry, ...]

    @model_validator(mode="after")
    def validate_catalog(self) -> "ModelRegistryCatalog":
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        keys = [
            (entry.model_id, entry.immutable_revision, entry.quantization, entry.dtype)
            for entry in self.models
        ]
        if keys != sorted(keys):
            raise ValueError("model catalog entries must use canonical identity ordering")
        if len(keys) != len(set(keys)):
            raise ValueError("model catalog contains a duplicate execution identity")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


class RouteAuthorityKey(ContractModel):
    """One Biscuit root key accepted for a bounded wall-clock interval."""

    key_id: HashHex
    public_key: HashHex
    not_before_ms: int
    not_after_ms: int

    @model_validator(mode="after")
    def validate_key(self) -> "RouteAuthorityKey":
        expected_key_id = hashlib.sha256(bytes.fromhex(self.public_key)).hexdigest()
        if self.key_id != expected_key_id:
            raise ValueError("route authority key ID does not match its public key")
        if self.not_before_ms < 0 or self.not_after_ms <= self.not_before_ms:
            raise ValueError("route authority key has an invalid validity interval")
        return self


class RouteAuthorityKeyset(ContractModel):
    """TUF-authenticated capability keys and emergency revocations."""

    protocol_version: int = PROTOCOL_VERSION
    generation: int
    issued_at_ms: int
    expires_at_ms: int
    keys: tuple[RouteAuthorityKey, ...]
    revoked_identifiers: tuple[RevocationIdentifierHex, ...] = ()

    @model_validator(mode="after")
    def validate_keyset(self) -> "RouteAuthorityKeyset":
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if self.generation <= 0:
            raise ValueError("route authority generation must be positive")
        if self.issued_at_ms < 0 or self.expires_at_ms <= self.issued_at_ms:
            raise ValueError("route authority keyset has an invalid validity interval")
        key_ids = [key.key_id for key in self.keys]
        if not key_ids:
            raise ValueError("route authority keyset must contain at least one key")
        if key_ids != sorted(key_ids) or len(key_ids) != len(set(key_ids)):
            raise ValueError("route authority keys must be unique and canonically ordered")
        if tuple(sorted(self.revoked_identifiers)) != self.revoked_identifiers:
            raise ValueError("route revocations must use canonical ordering")
        if len(set(self.revoked_identifiers)) != len(self.revoked_identifiers):
            raise ValueError("route authority keyset contains duplicate revocations")
        return self

    def active_public_keys(self, now_ms: int) -> dict[str, str]:
        if now_ms < self.issued_at_ms or now_ms >= self.expires_at_ms:
            raise ValueError("route authority keyset is not currently valid")
        return {
            key.key_id: key.public_key
            for key in self.keys
            if key.not_before_ms <= now_ms < key.not_after_ms
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


class ModelRegistryBundle(ContractModel):
    """Signed TUF target containing the compact manifest and persistent file index."""

    protocol_version: int = PROTOCOL_VERSION
    manifest: ModelManifest
    artifact_index: ModelArtifactIndex

    @model_validator(mode="after")
    def validate_bundle(self) -> "ModelRegistryBundle":
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if self.manifest.model_id != self.artifact_index.model_id:
            raise ValueError("manifest and artifact index identify different models")
        if self.manifest.immutable_revision != self.artifact_index.immutable_revision:
            raise ValueError("manifest and artifact index use different immutable revisions")
        expected = {
            ArtifactRole.ARCHITECTURE: self.manifest.architecture_graph_hash,
            ArtifactRole.TOKENIZER: self.manifest.tokenizer_hash,
            ArtifactRole.WEIGHT: self.manifest.weight_collection_hash,
        }
        for role, digest in expected.items():
            if artifact_collection_hash(self.artifact_index, role) != digest:
                raise ValueError(f"manifest {role.value} collection hash does not match its index")
        from swarm_protocol.model_manifest import execution_plan_hash

        if self.artifact_index.execution_plans:
            if self.manifest.execution_plan_hash is None:
                raise ValueError("portable execution plans are not bound by the model manifest")
            if execution_plan_hash(self.artifact_index) != self.manifest.execution_plan_hash:
                raise ValueError("manifest execution plan hash does not match its index")
            for plan in self.artifact_index.execution_plans:
                self._validate_execution_plan_layers(plan)
        elif self.manifest.execution_plan_hash is not None:
            raise ValueError("manifest binds an execution plan missing from its artifact index")
        return self

    def _validate_execution_plan_layers(self, plan) -> None:
        from swarm_protocol.contracts import ExecutionStageKind

        inputs = [stage for stage in plan.stages if stage.kind is ExecutionStageKind.INPUT]
        outputs = [stage for stage in plan.stages if stage.kind is ExecutionStageKind.OUTPUT]
        decoders = sorted(
            (stage for stage in plan.stages if stage.kind is ExecutionStageKind.DECODER),
            key=lambda stage: (stage.start_layer, stage.end_layer),
        )
        if len(inputs) != 1 or inputs[0].start_layer != 0:
            raise ValueError("execution plan requires exactly one layer-zero input endpoint")
        if len(outputs) != 1 or outputs[0].start_layer != self.manifest.num_layers:
            raise ValueError("execution plan requires exactly one final output endpoint")
        cursor = 0
        for stage in decoders:
            if stage.start_layer != cursor or stage.end_layer > self.manifest.num_layers:
                raise ValueError("decoder execution stages must tile model layers exactly")
            cursor = stage.end_layer
        if cursor != self.manifest.num_layers:
            raise ValueError("decoder execution stages do not cover every model layer")

    @property
    def model_swarm_id(self) -> str:
        return self.manifest.model_swarm_id

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


def model_target_path(model_swarm_id: str) -> str:
    """Return the stable logical TUF target path for a model swarm."""

    # Validate with the same Pydantic pattern used by protocol contracts.
    class _ModelId(ContractModel):
        value: HashHex

    _ModelId(value=model_swarm_id)
    return f"models/{model_swarm_id}.json"


@dataclass(frozen=True)
class RegistryRoleSigners:
    """Operator-provided signers and thresholds; no key material is persisted here."""

    root: tuple[Signer, ...]
    targets: tuple[Signer, ...]
    snapshot: tuple[Signer, ...]
    timestamp: tuple[Signer, ...]
    root_threshold: int = 1
    targets_threshold: int = 1
    snapshot_threshold: int = 1
    timestamp_threshold: int = 1

    def for_role(self, role: str) -> tuple[Signer, ...]:
        if role not in _TOP_LEVEL_ROLES:
            raise ValueError(f"unknown TUF role: {role}")
        return getattr(self, role)

    def threshold_for_role(self, role: str) -> int:
        if role not in _TOP_LEVEL_ROLES:
            raise ValueError(f"unknown TUF role: {role}")
        return getattr(self, f"{role}_threshold")

    def validate(self) -> None:
        for role in _TOP_LEVEL_ROLES:
            signers = self.for_role(role)
            threshold = self.threshold_for_role(role)
            unique_keys = {signer.public_key.keyid for signer in signers}
            if threshold <= 0 or len(unique_keys) < threshold:
                raise ValueError(
                    f"TUF {role} role has {len(unique_keys)} unique keys for threshold {threshold}"
                )


@dataclass(frozen=True)
class RegistryExpiryPolicy:
    """Role expirations following TUF's offline-root/short-lived-timestamp pattern."""

    root: timedelta = timedelta(days=365)
    targets: timedelta = timedelta(days=30)
    snapshot: timedelta = timedelta(days=7)
    timestamp: timedelta = timedelta(days=1)


def _utc_now(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("TUF publication time must be timezone-aware")
    return value.astimezone(timezone.utc).replace(microsecond=0)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Atomically publish one non-secret TUF repository object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        # TUF metadata and targets are public by design.  NamedTemporaryFile defaults to 0600,
        # which prevents an nginx/caddy service account from reading an operator-published tree.
        if os.name != "nt":
            os.fchmod(temporary.fileno(), 0o644)
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _sign(metadata: Metadata, signers: tuple[Signer, ...]) -> bytes:
    metadata.signatures.clear()
    for signer in signers:
        metadata.sign(signer, append=True)
    return metadata.to_bytes(_SERIALIZER)


def _build_root(
    signers: RegistryRoleSigners,
    *,
    version: int,
    expires: datetime,
) -> Metadata[Root]:
    root = Metadata(Root(version=version, expires=expires, consistent_snapshot=True))
    for role in _TOP_LEVEL_ROLES:
        for signer in signers.for_role(role):
            root.signed.add_key(signer.public_key, role)
        root.signed.roles[role].threshold = signers.threshold_for_role(role)
    return root


def _target_payloads(
    bundles: tuple[ModelRegistryBundle, ...],
    route_authorities: RouteAuthorityKeyset | None = None,
) -> dict[str, bytes]:
    entries = tuple(
        sorted(
            (
                ModelCatalogEntry(
                    model_id=bundle.manifest.model_id,
                    immutable_revision=bundle.manifest.immutable_revision,
                    model_swarm_id=bundle.model_swarm_id,
                    quantization=bundle.manifest.quantization,
                    dtype=bundle.manifest.dtype,
                )
                for bundle in bundles
            ),
            key=lambda entry: (
                entry.model_id,
                entry.immutable_revision,
                entry.quantization,
                entry.dtype,
            ),
        )
    )
    catalog = ModelRegistryCatalog(models=entries)
    payloads: dict[str, bytes] = {_CATALOG_TARGET_PATH: catalog.canonical_bytes()}
    if route_authorities is not None:
        payloads[_ROUTE_AUTHORITIES_TARGET_PATH] = route_authorities.canonical_bytes()
    for bundle in bundles:
        target_path = model_target_path(bundle.model_swarm_id)
        if target_path in payloads:
            raise ValueError(f"duplicate model registry target: {target_path}")
        payload = bundle.canonical_bytes()
        if len(payload) > _MAX_BUNDLE_BYTES:
            raise ValueError(f"model registry bundle exceeds {_MAX_BUNDLE_BYTES} bytes")
        payloads[target_path] = payload
    if not payloads:
        raise ValueError("registry snapshot must publish at least one model bundle")
    return payloads


class TufRegistryPublisher:
    """Offline/operator-side publisher for complete, versioned registry snapshots."""

    def __init__(
        self,
        repository_dir: Path,
        signers: RegistryRoleSigners,
        *,
        expiry: RegistryExpiryPolicy | None = None,
    ) -> None:
        signers.validate()
        self.repository_dir = repository_dir
        self.metadata_dir = repository_dir / "metadata"
        self.targets_dir = repository_dir / "targets"
        self.signers = signers
        self.expiry = expiry or RegistryExpiryPolicy()

    def initialize(
        self,
        bundles: tuple[ModelRegistryBundle, ...],
        *,
        route_authorities: RouteAuthorityKeyset | None = None,
        now: datetime | None = None,
    ) -> bytes:
        """Create version one and return root bytes to embed out-of-band in clients."""

        if self.metadata_dir.exists() and any(self.metadata_dir.iterdir()):
            raise FileExistsError("TUF registry metadata already exists")
        current = _utc_now(now)
        root = _build_root(
            self.signers,
            version=1,
            expires=current + self.expiry.root,
        )
        root_bytes = _sign(root, self.signers.root)
        root.signed.verify_delegate("root", root.signed_bytes, root.signatures)
        _atomic_write(self.metadata_dir / "1.root.json", root_bytes)
        self._publish_generation(
            bundles,
            targets_version=1,
            snapshot_version=1,
            timestamp_version=1,
            root=root,
            now=current,
            route_authorities=route_authorities,
        )
        return root_bytes

    def publish(
        self,
        bundles: tuple[ModelRegistryBundle, ...],
        *,
        route_authorities: RouteAuthorityKeyset | None = None,
        now: datetime | None = None,
    ) -> int:
        """Atomically expose a new complete targets/snapshot/timestamp generation."""

        root = self._load_latest_root()
        targets_version = self._latest_role_version("targets") + 1
        snapshot_version = self._latest_role_version("snapshot") + 1
        timestamp_version = self._current_timestamp_version() + 1
        self._publish_generation(
            bundles,
            targets_version=targets_version,
            snapshot_version=snapshot_version,
            timestamp_version=timestamp_version,
            root=root,
            now=_utc_now(now),
            route_authorities=route_authorities,
        )
        return targets_version

    def _publish_generation(
        self,
        bundles: tuple[ModelRegistryBundle, ...],
        *,
        targets_version: int,
        snapshot_version: int,
        timestamp_version: int,
        root: Metadata[Root],
        now: datetime,
        route_authorities: RouteAuthorityKeyset | None,
    ) -> None:
        payloads = _target_payloads(bundles, route_authorities)
        targets = Metadata(
            Targets(
                version=targets_version,
                expires=now + self.expiry.targets,
                targets={
                    path: TargetFile.from_data(path, payload)
                    for path, payload in sorted(payloads.items())
                },
            )
        )
        targets_bytes = _sign(targets, self.signers.targets)
        root.signed.verify_delegate("targets", targets.signed_bytes, targets.signatures)

        snapshot = Metadata(
            Snapshot(
                version=snapshot_version,
                expires=now + self.expiry.snapshot,
                meta={
                    "targets.json": MetaFile.from_data(
                        targets_version,
                        targets_bytes,
                        ["sha256"],
                    ),
                },
            )
        )
        snapshot_bytes = _sign(snapshot, self.signers.snapshot)
        root.signed.verify_delegate("snapshot", snapshot.signed_bytes, snapshot.signatures)

        timestamp = Metadata(
            Timestamp(
                version=timestamp_version,
                expires=now + self.expiry.timestamp,
                snapshot_meta=MetaFile.from_data(
                    snapshot_version,
                    snapshot_bytes,
                    ["sha256"],
                ),
            )
        )
        timestamp_bytes = _sign(timestamp, self.signers.timestamp)
        root.signed.verify_delegate("timestamp", timestamp.signed_bytes, timestamp.signatures)

        # Targets and versioned metadata are immutable. Publishing timestamp last makes the new
        # generation visible only after every object it authenticates is durable.
        for target_path, payload in payloads.items():
            target_hash = targets.signed.targets[target_path].hashes["sha256"]
            logical = Path(target_path)
            consistent_name = logical.with_name(f"{target_hash}.{logical.name}")
            _atomic_write(self.targets_dir / consistent_name, payload)
        _atomic_write(
            self.metadata_dir / f"{targets_version}.targets.json",
            targets_bytes,
        )
        _atomic_write(
            self.metadata_dir / f"{snapshot_version}.snapshot.json",
            snapshot_bytes,
        )
        _atomic_write(self.metadata_dir / "timestamp.json", timestamp_bytes)

    def _current_timestamp_version(self) -> int:
        path = self.metadata_dir / "timestamp.json"
        metadata = Metadata.from_file(str(path))
        if not isinstance(metadata.signed, Timestamp):
            raise ValueError("timestamp.json does not contain TUF timestamp metadata")
        # A root rotation can replace the timestamp key before the next full
        # publication.  The current timestamp is then correctly signed by the
        # immediately preceding root, not by the latest one.  Accept its
        # version only if one root in the retained, versioned trust history
        # authenticates it; never derive a rollback-sensitive version from
        # unverified local JSON.
        for version in range(self._latest_role_version("root"), 0, -1):
            root_path = self.metadata_dir / f"{version}.root.json"
            if not root_path.exists():
                continue
            root = Metadata.from_file(str(root_path))
            if not isinstance(root.signed, Root):
                continue
            try:
                root.signed.verify_delegate(
                    "timestamp",
                    metadata.signed_bytes,
                    metadata.signatures,
                )
            except (UnsignedMetadataError, ValueError):
                continue
            return metadata.signed.version
        raise ValueError("timestamp.json is not authenticated by any retained root")

    def rotate_root(
        self,
        new_signers: RegistryRoleSigners,
        *,
        now: datetime | None = None,
    ) -> bytes:
        """Publish a root signed by both old and new root thresholds.

        Operators must construct a new publisher with ``new_signers`` for subsequent snapshots.
        This mirrors TUF's required continuity rule and supports explicit compromised-key removal.
        """

        new_signers.validate()
        previous = self._load_latest_root()
        replacement = _build_root(
            new_signers,
            version=previous.signed.version + 1,
            expires=_utc_now(now) + self.expiry.root,
        )
        replacement.signatures.clear()
        unique_signers = {
            signer.public_key.keyid: signer for signer in (*self.signers.root, *new_signers.root)
        }
        for signer in unique_signers.values():
            replacement.sign(signer, append=True)
        previous.signed.verify_delegate("root", replacement.signed_bytes, replacement.signatures)
        replacement.signed.verify_delegate("root", replacement.signed_bytes, replacement.signatures)
        payload = replacement.to_bytes(_SERIALIZER)
        _atomic_write(
            self.metadata_dir / f"{replacement.signed.version}.root.json",
            payload,
        )
        return payload

    def _latest_role_version(self, role: str) -> int:
        versions = []
        for path in self.metadata_dir.glob(f"*.{role}.json"):
            try:
                versions.append(int(path.name.split(".", 1)[0]))
            except ValueError:
                continue
        if not versions:
            raise FileNotFoundError(f"TUF registry has no versioned {role} metadata")
        return max(versions)

    def _load_latest_root(self) -> Metadata[Root]:
        version = self._latest_role_version("root")
        metadata = Metadata.from_file(str(self.metadata_dir / f"{version}.root.json"))
        if not isinstance(metadata.signed, Root):
            raise ValueError("latest root metadata does not contain a TUF root role")
        return metadata


class TufTimestampRefresher:
    """Refresh only online timestamp metadata over one existing snapshot.

    TUF intentionally allows the frequently-rotated timestamp signer to stay
    online without exposing root, targets or snapshot signing keys.
    """

    def __init__(
        self,
        repository_dir: Path,
        signers: tuple[Signer, ...],
        *,
        validity: timedelta = RegistryExpiryPolicy().timestamp,
    ) -> None:
        if not signers:
            raise ValueError("timestamp refresh requires at least one signer")
        if validity <= timedelta(0):
            raise ValueError("timestamp validity must be positive")
        self.metadata_dir = repository_dir / "metadata"
        self.signers = signers
        self.validity = validity

    def refresh(self, *, now: datetime | None = None) -> int:
        """Sign and atomically publish a fresh timestamp for the current snapshot."""

        current = _utc_now(now)
        root = self._latest_root()
        previous = Metadata.from_file(str(self.metadata_dir / "timestamp.json"))
        if not isinstance(previous.signed, Timestamp):
            raise ValueError("timestamp.json does not contain TUF timestamp metadata")
        root.signed.verify_delegate(
            "timestamp",
            previous.signed_bytes,
            previous.signatures,
        )

        snapshot_version = previous.signed.snapshot_meta.version
        snapshot_path = self.metadata_dir / f"{snapshot_version}.snapshot.json"
        snapshot = Metadata.from_file(str(snapshot_path))
        if not isinstance(snapshot.signed, Snapshot):
            raise ValueError("referenced metadata does not contain a TUF snapshot role")
        if snapshot.signed.version != snapshot_version:
            raise ValueError("timestamp and snapshot versions disagree")
        root.signed.verify_delegate(
            "snapshot",
            snapshot.signed_bytes,
            snapshot.signatures,
        )
        if snapshot.signed.is_expired(current):
            raise ValueError("refusing to refresh timestamp over an expired snapshot")

        version = previous.signed.version + 1
        replacement = Metadata(
            Timestamp(
                version=version,
                expires=current + self.validity,
                snapshot_meta=MetaFile.from_data(
                    snapshot_version,
                    snapshot.to_bytes(_SERIALIZER),
                    ["sha256"],
                ),
            )
        )
        payload = _sign(replacement, self.signers)
        root.signed.verify_delegate(
            "timestamp",
            replacement.signed_bytes,
            replacement.signatures,
        )
        _atomic_write(self.metadata_dir / "timestamp.json", payload)
        return version

    def _latest_root(self) -> Metadata[Root]:
        versions = []
        for path in self.metadata_dir.glob("*.root.json"):
            try:
                versions.append(int(path.name.split(".", 1)[0]))
            except ValueError:
                continue
        if not versions:
            raise FileNotFoundError("TUF registry has no versioned root metadata")
        metadata = Metadata.from_file(str(self.metadata_dir / f"{max(versions)}.root.json"))
        if not isinstance(metadata.signed, Root):
            raise ValueError("latest root metadata does not contain a TUF root role")
        return metadata


class _PortableTufUpdater(Updater):
    """Project the current trusted root without requiring filesystem symlinks.

    python-tuf 7 keeps every trusted root under ``root_history`` and exposes the
    current one through a ``root.json`` symlink.  Creating that symlink requires
    SeCreateSymbolicLinkPrivilege on a default Windows installation, which a
    normal Fabi worker intentionally does not have.  The non-versioned file is a
    cache projection, not a trust anchor: every Fabi updater is bootstrapped from
    immutable application bytes.  An atomic regular-file projection therefore
    preserves TUF's verified root history and rollback checks while remaining
    usable by an unprivileged process on every supported platform.

    This is deliberately scoped to the private hook in the exactly pinned
    ``tuf==7.0.0`` dependency.  A TUF upgrade must revalidate this adapter.
    """

    def _update_root_symlink(self) -> None:
        version = self._trusted_set.root.version
        versioned_root = Path(self._dir, "root_history", f"{version}.root.json")
        self._persist_file(
            str(Path(self._dir, "root.json")),
            versioned_root.read_bytes(),
        )


class TrustedModelRegistry:
    """Runtime TUF client rooted in immutable bootstrap bytes shipped with Fabi."""

    def __init__(
        self,
        state_dir: Path,
        *,
        metadata_base_url: str,
        target_base_url: str,
        bootstrap_root: bytes,
        fetcher: FetcherInterface | None = None,
    ) -> None:
        if not bootstrap_root:
            raise ValueError("trusted registry requires non-empty bootstrap root bytes")
        self.metadata_dir = state_dir / "metadata"
        self.target_dir = state_dir / "targets"
        self.metadata_base_url = metadata_base_url
        self.target_base_url = target_base_url
        self.bootstrap_root = bootstrap_root
        self.fetcher = fetcher

    def fetch(self, model_swarm_id: str) -> ModelRegistryBundle:
        """Refresh trusted metadata and return one hash-verified registry target."""

        target_path = model_target_path(model_swarm_id)
        payload = self._fetch_target(target_path)
        bundle = ModelRegistryBundle.model_validate_json(payload)
        if bundle.model_swarm_id != model_swarm_id:
            raise ValueError("trusted registry target path and model bundle identity disagree")
        return bundle

    def catalog(self) -> ModelRegistryCatalog:
        """Return the current authenticated human-name to swarm-id catalog."""

        return ModelRegistryCatalog.model_validate_json(self._fetch_target(_CATALOG_TARGET_PATH))

    def route_authorities(self) -> RouteAuthorityKeyset:
        """Return the current TUF-authenticated capability authority keyset."""

        return RouteAuthorityKeyset.model_validate_json(
            self._fetch_target(_ROUTE_AUTHORITIES_TARGET_PATH)
        )

    def resolve(
        self,
        model_id: str,
        *,
        immutable_revision: str | None = None,
        quantization: str | None = None,
        dtype: str | None = None,
    ) -> ModelRegistryBundle:
        """Resolve one unambiguous signed execution identity and fetch its bundle."""

        matches = [
            entry
            for entry in self.catalog().models
            if entry.model_id == model_id
            and (immutable_revision is None or entry.immutable_revision == immutable_revision)
            and (quantization is None or entry.quantization == quantization)
            and (dtype is None or entry.dtype == dtype)
        ]
        if not matches:
            raise LookupError(f"trusted registry has no compatible manifest for {model_id!r}")
        if len(matches) != 1:
            raise LookupError(
                f"trusted registry has {len(matches)} execution variants for {model_id!r}; "
                "revision, quantization or dtype must disambiguate the request"
            )
        return self.fetch(matches[0].model_swarm_id)

    def _fetch_target(self, target_path: str) -> bytes:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        updater = _PortableTufUpdater(
            metadata_dir=str(self.metadata_dir),
            metadata_base_url=self.metadata_base_url,
            target_dir=str(self.target_dir),
            target_base_url=self.target_base_url,
            fetcher=self.fetcher,
            bootstrap=self.bootstrap_root,
        )
        updater.refresh()
        target_info = updater.get_targetinfo(target_path)
        if target_info is None:
            raise LookupError(f"trusted registry does not contain target {target_path!r}")
        if target_info.length > _MAX_BUNDLE_BYTES:
            raise ValueError("trusted registry target exceeds the model bundle size limit")
        local_path = updater.find_cached_target(target_info)
        if local_path is None:
            local_path = updater.download_target(target_info)
        return Path(local_path).read_bytes()
