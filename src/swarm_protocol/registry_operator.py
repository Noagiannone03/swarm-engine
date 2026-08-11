"""Operator tooling for the staging Fabi model registry.

Private TUF keys are deliberately kept outside the published repository.  They are serialized as
encrypted PKCS#8 PEM with pyca/cryptography and are only converted to securesystemslib signers in
memory.  This module is staging-oriented: production root ceremonies should use offline threshold
keys or a KMS/HSM signer behind the same :class:`RegistryRoleSigners` interface.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import math
import os
import secrets
import stat
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    Encoding,
    PrivateFormat,
    load_pem_private_key,
)
from securesystemslib.signer import CryptoSigner

from fabi_network.capability import capability_public_key
from swarm_protocol.contracts import (
    ArtifactDescriptor,
    ArtifactRole,
    BackendKind,
    ExecutionProviderKind,
    ExecutionStageDescriptor,
    ExecutionStageKind,
    ModelArtifactIndex,
    ModelExecutionPlan,
    ModelManifest,
    OnnxExportTarget,
    SkippyExactStateKind,
)
from swarm_protocol.model_manifest import (
    build_hub_model_bundle,
    execution_plan_hash,
)
from swarm_protocol.onnx_stage_builder import portable_build_inventory_hash
from swarm_protocol.registry import (
    ModelRegistryBundle,
    RegistryRoleSigners,
    RouteAuthorityKey,
    RouteAuthorityKeyset,
    TrustedModelRegistry,
    TufRegistryPublisher,
    TufTimestampRefresher,
)
from swarm_protocol.skippy_package_import import (
    attach_skippy_direct_gguf,
    attach_skippy_package,
    certify_skippy_exact_state,
)

_KEY_ROLES = ("root", "targets", "snapshot", "timestamp")
_MAX_SECRET_BYTES = 4096
_MAX_BUNDLE_BYTES = 32 * 1024 * 1024
_MAX_PORTABLE_INVENTORY_BYTES = 32 * 1024 * 1024
_MAX_ROUTE_AUTHORITY_BYTES = 1024 * 1024
_DEFAULT_ROUTE_AUTHORITY_VALIDITY_DAYS = 90
_DEFAULT_ROUTE_AUTHORITY_CLOCK_SKEW_SECONDS = 300


def _assert_private_path(path: Path, *, description: str) -> None:
    """Reject group/world-readable operator secrets on Unix."""

    if os.name == "nt":
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(f"{description} must not be accessible by group or others: {path}")


def read_passphrase_file(path: Path) -> bytes:
    """Read a bounded passphrase from an owner-only file without logging it."""

    _assert_private_path(path, description="registry passphrase file")
    with path.open("rb") as source:
        payload = source.read(_MAX_SECRET_BYTES + 1)
    if len(payload) > _MAX_SECRET_BYTES:
        raise ValueError(f"registry passphrase exceeds {_MAX_SECRET_BYTES} bytes")
    # Supporting a conventional one-line secret file is useful for operator automation.  Remove
    # only its line terminator: spaces and all other bytes remain part of the passphrase.
    passphrase = payload.rstrip(b"\r\n")
    if not passphrase:
        raise ValueError("registry passphrase must not be empty")
    return passphrase


def generate_passphrase_file(path: Path) -> None:
    """Create a high-entropy staging passphrase file without printing its contents."""

    # token_urlsafe(32) contains 256 bits of CSPRNG entropy before encoding and avoids embedded
    # newlines, making the resulting owner-only file suitable for unattended staging publication.
    _atomic_create_private(path, secrets.token_urlsafe(32).encode("ascii"))


def prompt_passphrase(*, confirm: bool) -> bytes:
    """Read a passphrase from the terminal, optionally confirming a new secret."""

    first = getpass.getpass("Registry key passphrase: ").encode("utf-8")
    if not first:
        raise ValueError("registry passphrase must not be empty")
    if confirm:
        second = getpass.getpass("Confirm registry key passphrase: ").encode("utf-8")
        if first != second:
            raise ValueError("registry passphrase confirmation does not match")
    return first


def _atomic_create_private(path: Path, payload: bytes) -> None:
    """Create one private file without ever replacing existing key material."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_create_public(path: Path, payload: bytes) -> None:
    """Create one public operator artifact without replacing an existing generation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def generate_route_authority(
    private_key_output: Path,
    keyset_output: Path,
    *,
    generation: int = 1,
    valid_for_days: int = _DEFAULT_ROUTE_AUTHORITY_VALIDITY_DAYS,
    clock_skew_seconds: int = _DEFAULT_ROUTE_AUTHORITY_CLOCK_SKEW_SECONDS,
    now_ms: int | None = None,
) -> RouteAuthorityKeyset:
    """Generate an owner-only Biscuit seed and its public TUF keyset.

    The private seed is never written to stdout and is deliberately separate
    from the public keyset that the targets role signs.
    """

    if private_key_output == keyset_output:
        raise ValueError("route authority private key and keyset paths must be different")
    if private_key_output.exists():
        raise FileExistsError(f"refusing to overwrite route authority key: {private_key_output}")
    if keyset_output.exists():
        raise FileExistsError(f"refusing to overwrite route authority keyset: {keyset_output}")
    if generation <= 0:
        raise ValueError("route authority generation must be positive")
    if not 1 <= valid_for_days <= 3650:
        raise ValueError("route authority validity must be between 1 and 3650 days")
    if not 0 <= clock_skew_seconds <= 3600:
        raise ValueError("route authority clock skew must be between 0 and 3600 seconds")

    current_ms = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    if current_ms < 0:
        raise ValueError("route authority publication time must not be negative")
    valid_for_ms = valid_for_days * 24 * 60 * 60 * 1000
    skew_ms = clock_skew_seconds * 1000
    issued_at_ms = max(0, current_ms - skew_ms)
    expires_at_ms = current_ms + valid_for_ms
    private_key = secrets.token_hex(32)
    public_key = capability_public_key(private_key)
    authority = RouteAuthorityKey(
        key_id=hashlib.sha256(bytes.fromhex(public_key)).hexdigest(),
        public_key=public_key,
        not_before_ms=issued_at_ms,
        not_after_ms=expires_at_ms,
    )
    keyset = RouteAuthorityKeyset(
        generation=generation,
        issued_at_ms=issued_at_ms,
        expires_at_ms=expires_at_ms,
        keys=(authority,),
    )

    created_private = False
    try:
        _atomic_create_private(private_key_output, private_key.encode("ascii") + b"\n")
        created_private = True
        _atomic_create_public(keyset_output, keyset.canonical_bytes() + b"\n")
    except BaseException:
        if created_private:
            try:
                private_key_output.unlink()
            except FileNotFoundError:
                pass
        raise
    return keyset


def generate_staging_keys(key_dir: Path, passphrase: bytes) -> RegistryRoleSigners:
    """Generate one encrypted Ed25519 key per TUF role and return in-memory signers."""

    if not passphrase:
        raise ValueError("registry passphrase must not be empty")
    expected_paths = tuple(key_dir / f"{role}.pem" for role in _KEY_ROLES)
    existing = [path for path in expected_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite registry key: {existing[0]}")

    signers: dict[str, CryptoSigner] = {}
    created: list[Path] = []
    try:
        for role, path in zip(_KEY_ROLES, expected_paths, strict=True):
            private_key = Ed25519PrivateKey.generate()
            pem = private_key.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=BestAvailableEncryption(passphrase),
            )
            _atomic_create_private(path, pem)
            created.append(path)
            signers[role] = CryptoSigner(private_key)
    except BaseException:
        # A partially-created authority is unsafe and confusing.  Only remove files created by
        # this call; pre-existing paths were rejected before the first write.
        for path in created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    return RegistryRoleSigners(
        root=(signers["root"],),
        targets=(signers["targets"],),
        snapshot=(signers["snapshot"],),
        timestamp=(signers["timestamp"],),
    )


def load_staging_keys(key_dir: Path, passphrase: bytes) -> RegistryRoleSigners:
    """Load the four encrypted staging keys and enforce their type and permissions."""

    if not passphrase:
        raise ValueError("registry passphrase must not be empty")
    signers: dict[str, CryptoSigner] = {}
    for role in _KEY_ROLES:
        path = key_dir / f"{role}.pem"
        _assert_private_path(path, description=f"registry {role} private key")
        private_key = load_pem_private_key(path.read_bytes(), password=passphrase)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError(f"registry {role} key is not Ed25519")
        signers[role] = CryptoSigner(private_key)
    return RegistryRoleSigners(
        root=(signers["root"],),
        targets=(signers["targets"],),
        snapshot=(signers["snapshot"],),
        timestamp=(signers["timestamp"],),
    )


def load_staging_timestamp_signers(
    key_dir: Path,
    passphrase: bytes,
) -> tuple[CryptoSigner, ...]:
    """Load only the encrypted online timestamp signer."""

    if not passphrase:
        raise ValueError("registry passphrase must not be empty")
    path = key_dir / "timestamp.pem"
    _assert_private_path(path, description="registry timestamp private key")
    private_key = load_pem_private_key(path.read_bytes(), password=passphrase)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("registry timestamp key is not Ed25519")
    return (CryptoSigner(private_key),)


def load_bundles(paths: Sequence[Path]) -> tuple[ModelRegistryBundle, ...]:
    """Load complete immutable registry bundles from bounded JSON files."""

    if not paths:
        raise ValueError("at least one model bundle is required")
    bundles = []
    for path in paths:
        if path.stat().st_size > _MAX_BUNDLE_BYTES:
            raise ValueError(f"model bundle exceeds {_MAX_BUNDLE_BYTES} bytes: {path}")
        bundles.append(ModelRegistryBundle.model_validate_json(path.read_bytes()))
    return tuple(bundles)


def load_route_authorities(path: Path | None) -> RouteAuthorityKeyset | None:
    """Load a bounded public keyset; the TUF targets role signs it on publish."""

    if path is None:
        return None
    if path.stat().st_size > _MAX_ROUTE_AUTHORITY_BYTES:
        raise ValueError("route authority keyset exceeds 1 MiB")
    return RouteAuthorityKeyset.model_validate_json(path.read_bytes())


def _atomic_write_public(path: Path, payload: bytes) -> None:
    """Write a non-secret operator artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_hub_bundle_file(
    output: Path,
    *,
    model_id: str,
    revision: str | None,
    quantization: str,
    dtype: str,
    token: bool | str | None = None,
) -> ModelRegistryBundle:
    """Resolve Hub metadata and persist one canonical registry bundle."""

    resolved = build_hub_model_bundle(
        model_id,
        revision=revision,
        quantization=quantization,
        dtype=dtype,
        token=token,
        include_weight_profile=True,
        include_selective_weight_index=True,
    )
    bundle = ModelRegistryBundle(
        manifest=resolved.manifest,
        artifact_index=resolved.artifact_index,
    )
    _atomic_write_public(output, bundle.canonical_bytes() + b"\n")
    return bundle


def build_skippy_package_bundle_file(
    output: Path,
    *,
    model_id: str,
    revision: str | None,
    package_repository_id: str,
    package_revision: str | None,
    plan_id: str,
    quantization: str,
    dtype: str,
    runtime_release: str,
    runtime_abi_version: str,
    providers: tuple[ExecutionProviderKind, ...],
    exact_state_kind: SkippyExactStateKind = SkippyExactStateKind.DISABLED,
    token: bool | str | None = None,
) -> ModelRegistryBundle:
    """Build a signed-model candidate directly around a Skippy layer package.

    Package-backed execution does not consume the source checkpoint tensors.  Resolve and hash
    only the source repository's architecture/tokenizer files plus its immutable weight-file
    descriptors, then let the Skippy package provide the exact executable layer geometry.  This
    deliberately avoids the expensive SafeTensors range index and weight-profile scan used by
    the legacy checkpoint executor.
    """

    resolved = build_hub_model_bundle(
        model_id,
        revision=revision,
        quantization=quantization,
        dtype=dtype,
        token=token,
        include_weight_profile=False,
        include_selective_weight_index=False,
    )
    base_bundle = ModelRegistryBundle(
        manifest=resolved.manifest,
        artifact_index=resolved.artifact_index,
    )
    bundle = attach_skippy_package(
        base_bundle,
        package_repository_id=package_repository_id,
        package_revision=package_revision,
        plan_id=plan_id,
        runtime_release=runtime_release,
        runtime_abi_version=runtime_abi_version,
        expected_quantization=quantization,
        providers=providers,
        exact_state_kind=exact_state_kind,
        token=token,
    )
    _atomic_write_public(output, bundle.canonical_bytes() + b"\n")
    return bundle


def _expect_exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    description: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{description} has an incompatible schema (missing={missing}, unknown={unknown})"
        )


def _mapping(value: object, *, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _sequence(value: object, *, description: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be an array")
    return value


def _verify_local_artifact(root: Path, descriptor: ArtifactDescriptor) -> None:
    """Verify one regular, root-confined publisher artifact from an open handle."""

    root = root.resolve(strict=True)
    candidate = root.joinpath(*descriptor.path.split("/"))
    cursor = root
    for part in descriptor.path.split("/"):
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"portable artifact must not traverse a symlink: {descriptor.path}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"portable artifact escapes its root: {descriptor.path}") from exc
    if not resolved.is_file():
        raise ValueError(f"portable artifact is not a regular file: {descriptor.path}")

    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        before = os.fstat(source.fileno())
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(source.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"portable artifact changed while hashing: {descriptor.path}")
    if after.st_size != descriptor.size:
        raise ValueError(f"portable artifact size mismatch: {descriptor.path}")
    if digest.hexdigest() != descriptor.sha256:
        raise ValueError(f"portable artifact digest mismatch: {descriptor.path}")


def _portable_inventory_descriptors(
    inventory: Mapping[str, object],
) -> tuple[ArtifactDescriptor, ...]:
    descriptors: list[ArtifactDescriptor] = []
    for position, raw in enumerate(
        _sequence(inventory["artifacts"], description="portable inventory artifacts")
    ):
        artifact = _mapping(raw, description=f"portable artifact {position}")
        _expect_exact_fields(
            artifact,
            {"path", "size", "sha256", "media_type", "role"},
            description=f"portable artifact {position}",
        )
        descriptor = ArtifactDescriptor.model_validate(artifact)
        if descriptor.role not in {
            ArtifactRole.EXECUTION_GRAPH,
            ArtifactRole.EXECUTION_DATA,
        }:
            raise ValueError("portable inventory may contain only execution artifacts")
        if descriptor.role is ArtifactRole.EXECUTION_GRAPH and (
            descriptor.media_type != "application/onnx" or not descriptor.path.endswith(".onnx")
        ):
            raise ValueError("portable execution graph must be an application/onnx file")
        descriptors.append(descriptor)
    if tuple(item.path for item in descriptors) != tuple(sorted(item.path for item in descriptors)):
        raise ValueError("portable inventory artifacts must be sorted by path")
    return tuple(descriptors)


def _portable_inventory_stages(
    inventory: Mapping[str, object],
) -> tuple[ExecutionStageDescriptor, ...]:
    stages: list[ExecutionStageDescriptor] = []
    expected = {
        "stage_id",
        "kind",
        "start_layer",
        "end_layer",
        "graph_path",
        "external_data_paths",
        "io_contract_hash",
        "inputs",
        "outputs",
    }
    for position, raw in enumerate(
        _sequence(inventory["stages"], description="portable inventory stages")
    ):
        stage = _mapping(raw, description=f"portable stage {position}")
        _expect_exact_fields(stage, expected, description=f"portable stage {position}")
        for field in ("inputs", "outputs"):
            values = _sequence(stage[field], description=f"portable stage {field}")
            if not values or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"portable stage {field} must contain non-empty names")
        stages.append(
            ExecutionStageDescriptor.model_validate(
                {key: stage[key] for key in expected - {"inputs", "outputs"}}
            )
        )
    return tuple(stages)


def _validate_source_artifacts(
    inventory: Mapping[str, object],
    source_root: Path,
) -> int:
    descriptors: list[ArtifactDescriptor] = []
    for position, raw in enumerate(
        _sequence(
            inventory["source_artifacts"],
            description="portable source artifacts",
        )
    ):
        artifact = _mapping(raw, description=f"portable source artifact {position}")
        _expect_exact_fields(
            artifact,
            {"path", "size", "sha256"},
            description=f"portable source artifact {position}",
        )
        descriptors.append(
            ArtifactDescriptor.model_validate(
                {
                    **artifact,
                    "media_type": "application/octet-stream",
                    "role": ArtifactRole.EXECUTION_DATA,
                }
            )
        )
    if tuple(item.path for item in descriptors) != tuple(sorted(item.path for item in descriptors)):
        raise ValueError("portable source artifacts must be sorted by path")
    if not descriptors:
        raise ValueError("portable build inventory has no source artifacts")
    for descriptor in descriptors:
        _verify_local_artifact(source_root, descriptor)
    return len(descriptors)


def attach_portable_execution(
    base_bundle: ModelRegistryBundle,
    inventory: Mapping[str, object],
    *,
    artifact_root: Path,
    source_root: Path,
    artifact_repository_id: str,
    artifact_revision: str,
    plan_id: str,
    precision: str,
    quantization: str,
    providers: tuple[ExecutionProviderKind, ...] | None = None,
) -> ModelRegistryBundle:
    """Bind a locally verified portable build to one immutable model bundle."""

    expected_inventory_fields = {
        "format_version",
        "builder",
        "source_model_id",
        "source_model_revision",
        "exporter",
        "exporter_revision",
        "target_execution_provider",
        "execution_geometry",
        "provider_assignment_policy",
        "num_layers",
        "shared_initializers",
        "source_artifacts",
        "artifacts",
        "stages",
        "inventory_hash",
    }
    _expect_exact_fields(
        inventory,
        expected_inventory_fields,
        description="portable build inventory",
    )
    if inventory["format_version"] != 1:
        raise ValueError("unsupported portable build inventory format")
    if inventory["builder"] != "fabi/swarm-engine/portable-onnx-stage-builder":
        raise ValueError("portable build inventory has an unknown builder")
    if inventory["inventory_hash"] != portable_build_inventory_hash(inventory):
        raise ValueError("portable build inventory hash does not match its contents")
    if inventory["source_model_id"] != base_bundle.manifest.model_id:
        raise ValueError("portable build and base bundle identify different models")
    if inventory["source_model_revision"] != base_bundle.manifest.immutable_revision:
        raise ValueError("portable build and base bundle use different source revisions")
    if inventory["num_layers"] != base_bundle.manifest.num_layers:
        raise ValueError("portable build and base bundle have different layer counts")

    shared = _sequence(
        inventory["shared_initializers"],
        description="portable shared initializers",
    )
    if any(not isinstance(value, str) or not value for value in shared):
        raise ValueError("portable shared initializer names must be non-empty")
    if shared != sorted(set(shared)):
        raise ValueError("portable shared initializers must be sorted and unique")

    artifacts = _portable_inventory_descriptors(inventory)
    stages = _portable_inventory_stages(inventory)
    referenced_paths = {
        path for stage in stages for path in (stage.graph_path, *stage.external_data_paths)
    }
    if referenced_paths != {artifact.path for artifact in artifacts}:
        raise ValueError("portable inventory contains missing or unreferenced execution artifacts")
    for descriptor in artifacts:
        _verify_local_artifact(artifact_root, descriptor)
    _validate_source_artifacts(inventory, source_root)

    geometry = _mapping(
        inventory["execution_geometry"],
        description="portable execution geometry",
    )
    _expect_exact_fields(
        geometry,
        {
            "activation_dtype",
            "activation_hidden_size",
            "kv_num_heads",
            "kv_head_dim",
        },
        description="portable execution geometry",
    )
    policy = _mapping(
        inventory["provider_assignment_policy"],
        description="portable provider policy",
    )
    _expect_exact_fields(
        policy,
        {
            "allowed_cpu_fallback_nodes",
            "allowed_cpu_only_stages",
            "require_profiled_assignment",
        },
        description="portable provider policy",
    )
    allowed_nodes = tuple(
        _sequence(
            policy["allowed_cpu_fallback_nodes"],
            description="allowed CPU fallback nodes",
        )
    )
    allowed_stages = tuple(
        _sequence(
            policy["allowed_cpu_only_stages"],
            description="allowed CPU-only stages",
        )
    )

    target = inventory["target_execution_provider"]
    if target == "dml":
        export_target = OnnxExportTarget.ORT_GENAI_DML
        default_providers = (ExecutionProviderKind.DIRECTML,)
        if policy["require_profiled_assignment"] is not True:
            raise ValueError("DirectML inventory must require profiled provider assignment")
    elif target == "cpu":
        export_target = OnnxExportTarget.ORT_GENAI_CPU
        default_providers = (ExecutionProviderKind.CPU,)
        if policy["require_profiled_assignment"] is not False or allowed_nodes or allowed_stages:
            raise ValueError("CPU inventory cannot declare provider fallback exceptions")
    else:
        raise ValueError(f"unsupported portable execution target: {target!r}")

    decoder_lengths = [
        stage.end_layer - stage.start_layer
        for stage in stages
        if stage.kind is ExecutionStageKind.DECODER
    ]
    granularity = math.gcd(*decoder_lengths) if decoder_lengths else 1
    plan = ModelExecutionPlan(
        plan_id=plan_id,
        backend=BackendKind.ONNXRUNTIME,
        precision=precision,
        quantization=quantization,
        exporter=inventory["exporter"],
        exporter_revision=inventory["exporter_revision"],
        artifact_repository_id=artifact_repository_id,
        artifact_revision=artifact_revision,
        export_target=export_target,
        activation_dtype=geometry["activation_dtype"],
        activation_hidden_size=geometry["activation_hidden_size"],
        kv_num_heads=geometry["kv_num_heads"],
        kv_head_dim=geometry["kv_head_dim"],
        allowed_cpu_fallback_nodes=allowed_nodes,
        allowed_cpu_only_stages=allowed_stages,
        execution_granularity_layers=granularity,
        providers=providers or default_providers,
        stages=stages,
    )

    index = ModelArtifactIndex(
        model_id=base_bundle.artifact_index.model_id,
        immutable_revision=base_bundle.artifact_index.immutable_revision,
        artifacts=tuple(
            sorted((*base_bundle.artifact_index.artifacts, *artifacts), key=lambda x: x.path)
        ),
        tensors=base_bundle.artifact_index.tensors,
        execution_plans=tuple(
            sorted((*base_bundle.artifact_index.execution_plans, plan), key=lambda x: x.plan_id)
        ),
    )
    manifest = ModelManifest.model_validate(
        {
            **base_bundle.manifest.model_dump(mode="json"),
            "execution_plan_hash": execution_plan_hash(index),
        }
    )
    return ModelRegistryBundle(manifest=manifest, artifact_index=index)


def attach_portable_execution_file(
    output: Path,
    *,
    bundle_path: Path,
    inventory_path: Path,
    artifact_root: Path,
    source_root: Path,
    artifact_repository_id: str,
    artifact_revision: str,
    plan_id: str,
    precision: str,
    quantization: str,
    providers: tuple[ExecutionProviderKind, ...] | None = None,
) -> ModelRegistryBundle:
    """Read, verify and atomically emit a portable execution bundle."""

    resolved_output = output.resolve()
    if resolved_output in {bundle_path.resolve(), inventory_path.resolve()}:
        raise ValueError("portable bundle output must not replace an input")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite portable bundle: {output}")
    if inventory_path.stat().st_size > _MAX_PORTABLE_INVENTORY_BYTES:
        raise ValueError("portable build inventory exceeds 32 MiB")
    raw = json.loads(inventory_path.read_bytes())
    inventory = _mapping(raw, description="portable build inventory")
    bundle = attach_portable_execution(
        load_bundles((bundle_path,))[0],
        inventory,
        artifact_root=artifact_root,
        artifact_repository_id=artifact_repository_id,
        artifact_revision=artifact_revision,
        plan_id=plan_id,
        precision=precision,
        quantization=quantization,
        providers=providers,
        source_root=source_root,
    )
    _atomic_create_public(output, bundle.canonical_bytes() + b"\n")
    return bundle


def attach_skippy_package_file(
    output: Path,
    *,
    bundle_path: Path,
    package_repository_id: str,
    package_revision: str | None,
    plan_id: str,
    runtime_release: str,
    runtime_abi_version: str,
    providers: tuple[ExecutionProviderKind, ...],
    exact_state_kind: SkippyExactStateKind = SkippyExactStateKind.DISABLED,
    token: bool | str | None = None,
) -> ModelRegistryBundle:
    """Resolve a public layer package and atomically emit its Fabi bundle."""

    if output.resolve() == bundle_path.resolve():
        raise ValueError("Skippy bundle output must not replace its input")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Skippy bundle: {output}")
    bundle = attach_skippy_package(
        load_bundles((bundle_path,))[0],
        package_repository_id=package_repository_id,
        package_revision=package_revision,
        plan_id=plan_id,
        runtime_release=runtime_release,
        runtime_abi_version=runtime_abi_version,
        providers=providers,
        exact_state_kind=exact_state_kind,
        token=token,
    )
    _atomic_create_public(output, bundle.canonical_bytes() + b"\n")
    return bundle


def attach_skippy_direct_file(
    output: Path,
    *,
    bundle_path: Path,
    repository_id: str,
    revision: str | None,
    source_paths: tuple[str, ...],
    plan_id: str,
    quantization: str,
    runtime_release: str,
    runtime_abi_version: str,
    runtime_root: Path,
    providers: tuple[ExecutionProviderKind, ...],
    exact_state_kind: SkippyExactStateKind = SkippyExactStateKind.DISABLED,
    token: bool | str | None = None,
) -> ModelRegistryBundle:
    """Inspect an ordinary GGUF with the qualified runtime and bind it atomically."""

    if output.resolve() == bundle_path.resolve():
        raise ValueError("Skippy bundle output must not replace its input")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Skippy bundle: {output}")
    try:
        import fabi_network_native
    except (ImportError, OSError) as exc:
        raise RuntimeError("the qualified Fabi native wheel is required to inspect GGUF") from exc
    raw_runtime = json.loads((runtime_root / "manifest.json").read_bytes())
    runtime = _mapping(raw_runtime.get("runtime"), description="Skippy runtime manifest")
    backend = runtime.get("backend")
    if isinstance(backend, Mapping):
        backend = backend.get("kind")
    mesh_release = runtime_release.removeprefix("mesh-llm/").removeprefix("v")
    fabi_network_native.load_skippy_native_runtime(
        runtime_root,
        mesh_release,
        runtime_abi_version,
        str(backend),
    )
    bundle = attach_skippy_direct_gguf(
        load_bundles((bundle_path,))[0],
        repository_id=repository_id,
        revision=revision,
        source_paths=source_paths,
        plan_id=plan_id,
        quantization=quantization,
        runtime_release=runtime_release,
        runtime_abi_version=runtime_abi_version,
        providers=providers,
        exact_state_kind=exact_state_kind,
        token=token,
    )
    _atomic_create_public(output, bundle.canonical_bytes() + b"\n")
    return bundle


def certify_skippy_exact_state_file(
    output: Path,
    *,
    bundle_path: Path,
    plan_id: str,
    state_kind: SkippyExactStateKind,
) -> ModelRegistryBundle:
    """Certify one existing plan and emit a new bundle without mutating the input."""

    if output.resolve() == bundle_path.resolve():
        raise ValueError("certified bundle output must not replace its input")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite certified bundle: {output}")
    bundle = certify_skippy_exact_state(
        load_bundles((bundle_path,))[0],
        plan_id=plan_id,
        state_kind=state_kind,
    )
    _atomic_create_public(output, bundle.canonical_bytes() + b"\n")
    return bundle


def initialize_staging_registry(
    repository_dir: Path,
    key_dir: Path,
    bundle_paths: Sequence[Path],
    passphrase: bytes,
    *,
    bootstrap_root_output: Path,
    route_authority_path: Path | None = None,
) -> tuple[ModelRegistryBundle, ...]:
    """Create a fresh 1-of-1 staging authority and its first signed snapshot."""

    if repository_dir.exists() and any(repository_dir.iterdir()):
        raise FileExistsError("refusing to initialize a non-empty registry repository")
    if bootstrap_root_output.exists():
        raise FileExistsError(f"refusing to overwrite bootstrap root: {bootstrap_root_output}")
    bundles = load_bundles(bundle_paths)
    route_authorities = load_route_authorities(route_authority_path)
    signers = generate_staging_keys(key_dir, passphrase)
    try:
        root = TufRegistryPublisher(repository_dir, signers).initialize(
            bundles,
            route_authorities=route_authorities,
        )
        _atomic_write_public(bootstrap_root_output, root)
    except BaseException:
        # Key deletion is intentionally not automatic here: once an authority has been created,
        # retaining its private keys is safer than silently replacing it on a later retry.
        raise
    return bundles


def publish_staging_registry(
    repository_dir: Path,
    key_dir: Path,
    bundle_paths: Sequence[Path],
    passphrase: bytes,
    *,
    route_authority_path: Path | None = None,
) -> tuple[int, tuple[ModelRegistryBundle, ...]]:
    """Publish a complete new staging snapshot with already-created keys."""

    bundles = load_bundles(bundle_paths)
    route_authorities = load_route_authorities(route_authority_path)
    version = TufRegistryPublisher(
        repository_dir,
        load_staging_keys(key_dir, passphrase),
    ).publish(bundles, route_authorities=route_authorities)
    return version, bundles


def _passphrase(args: argparse.Namespace, *, confirm: bool) -> bytes:
    if args.passphrase_file is not None:
        return read_passphrase_file(args.passphrase_file)
    if not sys.stdin.isatty():
        raise ValueError("non-interactive use requires --passphrase-file")
    return prompt_passphrase(confirm=confirm)


def _bundle_summary(bundle: ModelRegistryBundle) -> dict[str, object]:
    return {
        "model_id": bundle.manifest.model_id,
        "immutable_revision": bundle.manifest.immutable_revision,
        "model_swarm_id": bundle.model_swarm_id,
        "num_layers": bundle.manifest.num_layers,
        "model_max_context_tokens": bundle.manifest.model_max_context_tokens,
        "context_classes": list(bundle.manifest.context_classes),
        "artifacts": len(bundle.artifact_index.artifacts),
        "tensors": len(bundle.artifact_index.tensors),
        "execution_plans": [plan.plan_id for plan in bundle.artifact_index.execution_plans],
        "signed_tensor_bytes": sum(tensor.length for tensor in bundle.artifact_index.tensors),
        "signed_execution_bytes": sum(
            artifact.size
            for artifact in bundle.artifact_index.artifacts
            if artifact.role
            in {
                ArtifactRole.EXECUTION_DATA,
                ArtifactRole.EXECUTION_GRAPH,
                ArtifactRole.EXECUTION_LAYER,
                ArtifactRole.EXECUTION_PACKAGE_MANIFEST,
                ArtifactRole.EXECUTION_SHARED,
            }
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fabi-swarm-registry",
        description="Build and publish the signed Fabi staging model registry",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    passphrase = commands.add_parser("generate-passphrase")
    passphrase.add_argument("--output", type=Path, required=True)

    route_authority = commands.add_parser("generate-route-authority")
    route_authority.add_argument("--private-key-output", type=Path, required=True)
    route_authority.add_argument("--keyset-output", type=Path, required=True)
    route_authority.add_argument("--generation", type=int, default=1)
    route_authority.add_argument(
        "--valid-for-days",
        type=int,
        default=_DEFAULT_ROUTE_AUTHORITY_VALIDITY_DAYS,
    )
    route_authority.add_argument(
        "--clock-skew-seconds",
        type=int,
        default=_DEFAULT_ROUTE_AUTHORITY_CLOCK_SKEW_SECONDS,
    )

    build = commands.add_parser("build-hub-bundle")
    build.add_argument("--model-id", required=True)
    build.add_argument("--revision")
    build.add_argument("--quantization", required=True)
    build.add_argument("--dtype", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--use-hf-token",
        action="store_true",
        help="use the token from the normal Hugging Face credential store",
    )

    build_skippy = commands.add_parser("build-skippy-package-bundle")
    build_skippy.add_argument("--model-id", required=True)
    build_skippy.add_argument("--revision")
    build_skippy.add_argument("--package-repository-id", required=True)
    build_skippy.add_argument("--package-revision")
    build_skippy.add_argument("--plan-id", required=True)
    build_skippy.add_argument("--quantization", required=True)
    build_skippy.add_argument("--dtype", required=True)
    build_skippy.add_argument("--runtime-release", required=True)
    build_skippy.add_argument("--runtime-abi-version", required=True)
    build_skippy.add_argument(
        "--exact-state-kind",
        choices=[
            SkippyExactStateKind.DISABLED.value,
            SkippyExactStateKind.DENSE_ATTENTION_KV.value,
        ],
        default=SkippyExactStateKind.DISABLED.value,
        help="operator-qualified continuation state; dense KV requires live qualification",
    )
    build_skippy.add_argument(
        "--provider",
        action="append",
        required=True,
        choices=[
            ExecutionProviderKind.CPU.value,
            ExecutionProviderKind.CUDA.value,
            ExecutionProviderKind.METAL.value,
            ExecutionProviderKind.ROCM.value,
            ExecutionProviderKind.VULKAN.value,
        ],
        help="qualified native backend; repeat for every tested backend",
    )
    build_skippy.add_argument("--use-hf-token", action="store_true")
    build_skippy.add_argument("--output", type=Path, required=True)

    attach = commands.add_parser("attach-portable-execution")
    attach.add_argument("--bundle", type=Path, required=True)
    attach.add_argument("--inventory", type=Path, required=True)
    attach.add_argument("--artifact-root", type=Path, required=True)
    attach.add_argument("--source-root", type=Path, required=True)
    attach.add_argument("--artifact-repository-id", required=True)
    attach.add_argument("--artifact-revision", required=True)
    attach.add_argument("--plan-id", required=True)
    attach.add_argument("--precision", required=True)
    attach.add_argument("--quantization", required=True)
    attach.add_argument(
        "--provider",
        action="append",
        choices=[provider.value for provider in ExecutionProviderKind],
        help="qualified provider; repeat only when the same graph was verified on each provider",
    )
    attach.add_argument("--output", type=Path, required=True)

    attach_skippy = commands.add_parser("attach-skippy-execution")
    attach_skippy.add_argument("--bundle", type=Path, required=True)
    attach_skippy.add_argument("--package-repository-id", required=True)
    attach_skippy.add_argument("--package-revision")
    attach_skippy.add_argument("--plan-id", required=True)
    attach_skippy.add_argument("--runtime-release", default="mesh-llm/v0.74.0")
    attach_skippy.add_argument("--runtime-abi-version", default="0.1.32")
    attach_skippy.add_argument(
        "--exact-state-kind",
        choices=[
            SkippyExactStateKind.DISABLED.value,
            SkippyExactStateKind.DENSE_ATTENTION_KV.value,
        ],
        default=SkippyExactStateKind.DISABLED.value,
        help="operator-qualified continuation state; dense KV requires live qualification",
    )
    attach_skippy.add_argument(
        "--provider",
        action="append",
        choices=[
            ExecutionProviderKind.CPU.value,
            ExecutionProviderKind.CUDA.value,
            ExecutionProviderKind.METAL.value,
            ExecutionProviderKind.ROCM.value,
            ExecutionProviderKind.VULKAN.value,
        ],
        help="qualified native backend; repeat to restrict the release matrix",
    )
    attach_skippy.add_argument("--use-hf-token", action="store_true")
    attach_skippy.add_argument("--output", type=Path, required=True)

    attach_skippy_direct = commands.add_parser("attach-skippy-direct")
    attach_skippy_direct.add_argument("--bundle", type=Path, required=True)
    attach_skippy_direct.add_argument("--repository-id", required=True)
    attach_skippy_direct.add_argument("--revision")
    attach_skippy_direct.add_argument(
        "--source-path",
        action="append",
        required=True,
        help="exact GGUF file; repeat for an ordered split-GGUF set",
    )
    attach_skippy_direct.add_argument("--plan-id", required=True)
    attach_skippy_direct.add_argument("--quantization", required=True)
    attach_skippy_direct.add_argument("--runtime-release", default="mesh-llm/v0.74.0")
    attach_skippy_direct.add_argument("--runtime-abi-version", default="0.1.32")
    attach_skippy_direct.add_argument(
        "--exact-state-kind",
        choices=[
            SkippyExactStateKind.DISABLED.value,
            SkippyExactStateKind.DENSE_ATTENTION_KV.value,
        ],
        default=SkippyExactStateKind.DISABLED.value,
        help="operator-qualified continuation state; dense KV requires live qualification",
    )
    attach_skippy_direct.add_argument("--runtime-root", type=Path, required=True)
    attach_skippy_direct.add_argument(
        "--provider",
        action="append",
        choices=[
            ExecutionProviderKind.CPU.value,
            ExecutionProviderKind.CUDA.value,
            ExecutionProviderKind.METAL.value,
            ExecutionProviderKind.ROCM.value,
            ExecutionProviderKind.VULKAN.value,
        ],
        help="qualified native backend; repeat to restrict the release matrix",
    )
    attach_skippy_direct.add_argument("--use-hf-token", action="store_true")
    attach_skippy_direct.add_argument("--output", type=Path, required=True)

    certify_skippy = commands.add_parser("certify-skippy-exact-state")
    certify_skippy.add_argument("--bundle", type=Path, required=True)
    certify_skippy.add_argument("--plan-id", required=True)
    certify_skippy.add_argument(
        "--state-kind",
        choices=[SkippyExactStateKind.DENSE_ATTENTION_KV.value],
        required=True,
        help="operator-qualified model-family continuation state",
    )
    certify_skippy.add_argument("--output", type=Path, required=True)

    initialize = commands.add_parser("init-staging")
    initialize.add_argument("--repository-dir", type=Path, required=True)
    initialize.add_argument("--key-dir", type=Path, required=True)
    initialize.add_argument("--bundle", type=Path, action="append", required=True)
    initialize.add_argument("--bootstrap-root-output", type=Path, required=True)
    initialize.add_argument("--route-authorities", type=Path)
    initialize.add_argument("--passphrase-file", type=Path)

    publish = commands.add_parser("publish")
    publish.add_argument("--repository-dir", type=Path, required=True)
    publish.add_argument("--key-dir", type=Path, required=True)
    publish.add_argument("--bundle", type=Path, action="append", required=True)
    publish.add_argument("--route-authorities", type=Path)
    publish.add_argument("--passphrase-file", type=Path)

    refresh_timestamp = commands.add_parser("refresh-timestamp")
    refresh_timestamp.add_argument("--repository-dir", type=Path, required=True)
    refresh_timestamp.add_argument("--key-dir", type=Path, required=True)
    refresh_timestamp.add_argument("--passphrase-file", type=Path)

    verify = commands.add_parser("verify-remote")
    verify.add_argument("--bootstrap-root", type=Path, required=True)
    verify.add_argument("--metadata-url", required=True)
    verify.add_argument("--targets-url", required=True)
    verify.add_argument("--state-dir", type=Path, required=True)
    identity = verify.add_mutually_exclusive_group(required=True)
    identity.add_argument("--model-swarm-id")
    identity.add_argument("--model-id")
    verify.add_argument("--revision")
    verify.add_argument("--quantization")
    verify.add_argument("--dtype")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the operator CLI and emit only non-secret JSON summaries."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    if args.command == "generate-passphrase":
        generate_passphrase_file(args.output)
        result = {"status": "generated", "output": str(args.output)}
    elif args.command == "generate-route-authority":
        keyset = generate_route_authority(
            args.private_key_output,
            args.keyset_output,
            generation=args.generation,
            valid_for_days=args.valid_for_days,
            clock_skew_seconds=args.clock_skew_seconds,
        )
        result = {
            "status": "generated",
            "generation": keyset.generation,
            "key_id": keyset.keys[0].key_id,
            "expires_at_ms": keyset.expires_at_ms,
            "private_key_output": str(args.private_key_output),
            "keyset_output": str(args.keyset_output),
        }
    elif args.command == "build-hub-bundle":
        bundle = build_hub_bundle_file(
            args.output,
            model_id=args.model_id,
            revision=args.revision,
            quantization=args.quantization,
            dtype=args.dtype,
            token=True if args.use_hf_token else None,
        )
        result = {"status": "built", **_bundle_summary(bundle), "output": str(args.output)}
    elif args.command == "build-skippy-package-bundle":
        bundle = build_skippy_package_bundle_file(
            args.output,
            model_id=args.model_id,
            revision=args.revision,
            package_repository_id=args.package_repository_id,
            package_revision=args.package_revision,
            plan_id=args.plan_id,
            quantization=args.quantization,
            dtype=args.dtype,
            runtime_release=args.runtime_release,
            runtime_abi_version=args.runtime_abi_version,
            providers=tuple(
                sorted(
                    {ExecutionProviderKind(value) for value in args.provider},
                    key=lambda provider: provider.value,
                )
            ),
            exact_state_kind=SkippyExactStateKind(args.exact_state_kind),
            token=True if args.use_hf_token else None,
        )
        result = {
            "status": "skippy_package_bundle_built",
            **_bundle_summary(bundle),
            "output": str(args.output),
        }
    elif args.command == "attach-portable-execution":
        providers = (
            tuple(ExecutionProviderKind(value) for value in args.provider)
            if args.provider
            else None
        )
        bundle = attach_portable_execution_file(
            args.output,
            bundle_path=args.bundle,
            inventory_path=args.inventory,
            artifact_root=args.artifact_root,
            source_root=args.source_root,
            artifact_repository_id=args.artifact_repository_id,
            artifact_revision=args.artifact_revision,
            plan_id=args.plan_id,
            precision=args.precision,
            quantization=args.quantization,
            providers=providers,
        )
        result = {
            "status": "portable_execution_attached",
            **_bundle_summary(bundle),
            "output": str(args.output),
        }
    elif args.command == "attach-skippy-execution":
        provider_values = args.provider or [
            ExecutionProviderKind.CPU.value,
            ExecutionProviderKind.CUDA.value,
            ExecutionProviderKind.METAL.value,
            ExecutionProviderKind.ROCM.value,
            ExecutionProviderKind.VULKAN.value,
        ]
        bundle = attach_skippy_package_file(
            args.output,
            bundle_path=args.bundle,
            package_repository_id=args.package_repository_id,
            package_revision=args.package_revision,
            plan_id=args.plan_id,
            runtime_release=args.runtime_release,
            runtime_abi_version=args.runtime_abi_version,
            providers=tuple(
                sorted(
                    {ExecutionProviderKind(value) for value in provider_values},
                    key=lambda provider: provider.value,
                )
            ),
            exact_state_kind=SkippyExactStateKind(args.exact_state_kind),
            token=True if args.use_hf_token else None,
        )
        result = {
            "status": "skippy_execution_attached",
            **_bundle_summary(bundle),
            "output": str(args.output),
        }
    elif args.command == "attach-skippy-direct":
        provider_values = args.provider or [
            ExecutionProviderKind.CPU.value,
            ExecutionProviderKind.CUDA.value,
            ExecutionProviderKind.METAL.value,
            ExecutionProviderKind.ROCM.value,
            ExecutionProviderKind.VULKAN.value,
        ]
        bundle = attach_skippy_direct_file(
            args.output,
            bundle_path=args.bundle,
            repository_id=args.repository_id,
            revision=args.revision,
            source_paths=tuple(sorted(set(args.source_path))),
            plan_id=args.plan_id,
            quantization=args.quantization,
            runtime_release=args.runtime_release,
            runtime_abi_version=args.runtime_abi_version,
            runtime_root=args.runtime_root,
            providers=tuple(
                sorted(
                    {ExecutionProviderKind(value) for value in provider_values},
                    key=lambda provider: provider.value,
                )
            ),
            exact_state_kind=SkippyExactStateKind(args.exact_state_kind),
            token=True if args.use_hf_token else None,
        )
        result = {
            "status": "skippy_direct_attached",
            **_bundle_summary(bundle),
            "output": str(args.output),
        }
    elif args.command == "certify-skippy-exact-state":
        bundle = certify_skippy_exact_state_file(
            args.output,
            bundle_path=args.bundle,
            plan_id=args.plan_id,
            state_kind=SkippyExactStateKind(args.state_kind),
        )
        result = {
            "status": "skippy_exact_state_certified",
            **_bundle_summary(bundle),
            "output": str(args.output),
        }
    elif args.command == "init-staging":
        bundles = initialize_staging_registry(
            args.repository_dir,
            args.key_dir,
            args.bundle,
            _passphrase(args, confirm=True),
            bootstrap_root_output=args.bootstrap_root_output,
            route_authority_path=args.route_authorities,
        )
        result = {
            "status": "initialized_staging_1_of_1",
            "models": [_bundle_summary(bundle) for bundle in bundles],
            "repository": str(args.repository_dir),
            "bootstrap_root": str(args.bootstrap_root_output),
        }
    elif args.command == "publish":
        version, bundles = publish_staging_registry(
            args.repository_dir,
            args.key_dir,
            args.bundle,
            _passphrase(args, confirm=False),
            route_authority_path=args.route_authorities,
        )
        result = {
            "status": "published",
            "version": version,
            "models": [_bundle_summary(bundle) for bundle in bundles],
        }
    elif args.command == "refresh-timestamp":
        version = TufTimestampRefresher(
            args.repository_dir,
            load_staging_timestamp_signers(
                args.key_dir,
                _passphrase(args, confirm=False),
            ),
        ).refresh()
        result = {
            "status": "timestamp_refreshed",
            "version": version,
            "repository": str(args.repository_dir),
        }
    else:
        registry = TrustedModelRegistry(
            args.state_dir,
            metadata_base_url=args.metadata_url,
            target_base_url=args.targets_url,
            bootstrap_root=args.bootstrap_root.read_bytes(),
        )
        if args.model_swarm_id is not None:
            bundle = registry.fetch(args.model_swarm_id)
        else:
            bundle = registry.resolve(
                args.model_id,
                immutable_revision=args.revision,
                quantization=args.quantization,
                dtype=args.dtype,
            )
        result = {"status": "verified", **_bundle_summary(bundle)}

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
