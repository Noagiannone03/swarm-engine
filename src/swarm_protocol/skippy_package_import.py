"""Import direct GGUF sources or sparse Skippy packages into Fabi's registry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Protocol

from huggingface_hub import HfApi, hf_hub_download

from swarm_protocol.contracts import (
    ArtifactDescriptor,
    ArtifactRole,
    ExecutionProviderKind,
    ModelArtifactIndex,
    ModelManifest,
    SkippyExecutionPlan,
    SkippyRuntimeFeature,
)
from swarm_protocol.model_manifest import execution_plan_hash
from swarm_protocol.registry import ModelRegistryBundle

_MAX_PACKAGE_MANIFEST_BYTES = 16 * 1024 * 1024
_COMMIT_LENGTH = 40
_DEFAULT_PROVIDERS = (
    ExecutionProviderKind.CPU,
    ExecutionProviderKind.CUDA,
    ExecutionProviderKind.METAL,
    ExecutionProviderKind.ROCM,
    ExecutionProviderKind.VULKAN,
)
_REQUIRED_FEATURES = (
    SkippyRuntimeFeature.ACTIVATION_FRAME,
    SkippyRuntimeFeature.BACKEND_DEVICES,
    SkippyRuntimeFeature.GENERATION_SIGNALS,
    SkippyRuntimeFeature.LAYER_PACKAGE,
    SkippyRuntimeFeature.SESSION_RESET,
)
_DIRECT_REQUIRED_FEATURES = (
    SkippyRuntimeFeature.ACTIVATION_FRAME,
    SkippyRuntimeFeature.BACKEND_DEVICES,
    SkippyRuntimeFeature.GENERATION_SIGNALS,
    SkippyRuntimeFeature.RUNTIME_SLICE,
    SkippyRuntimeFeature.SESSION_RESET,
)


class _Geometry(Protocol):
    activation_width: int
    context_length: int
    kv_bytes_per_token: int
    layer_count: int
    static_bytes_by_layer: list[int] | tuple[int, ...]


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Skippy package {label} must be an object")
    return value


def _artifact_entry(value: object, label: str) -> tuple[str, int, str]:
    entry = _mapping(value, label)
    path = str(entry.get("path", ""))
    size = entry.get("artifact_bytes")
    digest = str(entry.get("sha256", "")).lower()
    if not path or not isinstance(size, int) or size <= 0:
        raise ValueError(f"Skippy package {label} has no valid path/size")
    if len(digest) != 64 or not all(character in "0123456789abcdef" for character in digest):
        raise ValueError(f"Skippy package {label} has no valid SHA-256")
    return path, size, digest


def _remote_descriptor(
    *,
    path: str,
    size: int,
    sha256: str,
    role: ArtifactRole,
    siblings: Mapping[str, object],
) -> ArtifactDescriptor:
    sibling = siblings.get(path)
    if sibling is None:
        raise ValueError(f"Skippy package repository is missing {path!r}")
    remote_size = _field(sibling, "size")
    if remote_size != size:
        raise ValueError(f"Skippy package repository size mismatch for {path!r}")
    lfs = _field(sibling, "lfs")
    lfs_sha = str(_field(lfs, "sha256") or "").lower()
    if lfs_sha != sha256:
        raise ValueError(f"Skippy package repository digest mismatch for {path!r}")
    return ArtifactDescriptor(
        path=path,
        size=size,
        sha256=sha256,
        media_type="application/vnd.gguf",
        role=role,
    )


def _read_package_manifest(
    repository_id: str,
    immutable_revision: str,
    *,
    token: bool | str | None,
    downloader: Callable[..., str],
) -> tuple[bytes, Mapping[str, Any]]:
    path = Path(
        downloader(
            repo_id=repository_id,
            filename="model-package.json",
            revision=immutable_revision,
            token=token,
        )
    )
    if path.stat().st_size > _MAX_PACKAGE_MANIFEST_BYTES:
        raise ValueError("Skippy package manifest exceeds 16 MiB")
    payload = path.read_bytes()
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Skippy package manifest is not valid UTF-8 JSON") from exc
    return payload, _mapping(parsed, "manifest")


def _inspect_direct_geometry(
    paths: list[Path], cache_type_k: str, cache_type_v: str
) -> _Geometry:
    try:
        import fabi_network_native
    except (ImportError, OSError) as exc:  # pragma: no cover - product wheel integration
        raise RuntimeError("the qualified Fabi native wheel is required to inspect GGUF") from exc
    return fabi_network_native.inspect_skippy_source_geometry(
        paths,
        cache_type_k,
        cache_type_v,
    )


def _direct_source_hash(descriptors: list[ArtifactDescriptor]) -> str:
    digest = hashlib.sha256(b"fabi/skippy-direct-source/v1\0")
    for descriptor in sorted(descriptors, key=lambda item: item.path):
        digest.update(descriptor.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(descriptor.sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(descriptor.size).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def attach_skippy_direct_gguf(
    base_bundle: ModelRegistryBundle,
    *,
    repository_id: str,
    revision: str | None,
    source_paths: tuple[str, ...],
    plan_id: str,
    quantization: str,
    runtime_release: str,
    runtime_abi_version: str,
    providers: tuple[ExecutionProviderKind, ...] = _DEFAULT_PROVIDERS,
    token: bool | str | None = None,
    api: HfApi | None = None,
    downloader: Callable[..., str] = hf_hub_download,
    geometry_inspector: Callable[[list[Path], str, str], _Geometry] = _inspect_direct_geometry,
) -> ModelRegistryBundle:
    """Bind an ordinary immutable GGUF directly, without producing a layer package."""

    if not source_paths:
        raise ValueError("direct Skippy execution requires at least one GGUF file")
    if tuple(sorted(set(source_paths))) != source_paths:
        raise ValueError("direct GGUF source paths must be sorted and unique")
    client = api or HfApi()
    resolved = client.model_info(repository_id, revision=revision, token=token)
    immutable_revision = str(_field(resolved, "sha") or "").lower()
    if len(immutable_revision) != _COMMIT_LENGTH or not all(
        character in "0123456789abcdef" for character in immutable_revision
    ):
        raise ValueError("Hugging Face did not resolve an immutable GGUF commit")
    info = client.model_info(
        repository_id,
        revision=immutable_revision,
        files_metadata=True,
        token=token,
    )
    if str(_field(info, "sha") or "").lower() != immutable_revision:
        raise ValueError("Hugging Face GGUF metadata changed during resolution")
    siblings = {
        str(_field(sibling, "rfilename")): sibling for sibling in (_field(info, "siblings") or ())
    }
    descriptors: list[ArtifactDescriptor] = []
    local_paths: list[Path] = []
    for source_path in source_paths:
        parts = source_path.split("/")
        if (
            not source_path.lower().endswith(".gguf")
            or source_path.startswith("/")
            or "\\" in source_path
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError(f"unsafe or non-GGUF direct source path: {source_path!r}")
        sibling = siblings.get(source_path)
        if sibling is None:
            raise ValueError(f"GGUF repository is missing {source_path!r}")
        size = _field(sibling, "size")
        lfs = _field(sibling, "lfs")
        sha256 = str(_field(lfs, "sha256") or "").lower()
        lfs_size = _field(lfs, "size")
        if (
            not isinstance(size, int)
            or size <= 0
            or lfs_size != size
            or len(sha256) != 64
            or not all(character in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"GGUF source {source_path!r} has no exact LFS identity")
        descriptors.append(
            ArtifactDescriptor(
                path=source_path,
                size=size,
                sha256=sha256,
                media_type="application/vnd.gguf",
                role=ArtifactRole.EXECUTION_MODEL,
            )
        )
        local_paths.append(
            Path(
                downloader(
                    repo_id=repository_id,
                    filename=source_path,
                    revision=immutable_revision,
                    token=token,
                )
            )
        )
    geometry = geometry_inspector(local_paths, "f16", "f16")
    manifest = base_bundle.manifest
    static_bytes = tuple(int(value) for value in geometry.static_bytes_by_layer)
    if int(geometry.layer_count) != manifest.num_layers or len(static_bytes) != manifest.num_layers:
        raise ValueError("direct GGUF layer geometry differs from the logical model")
    if any(value <= 0 for value in static_bytes):
        raise ValueError("direct GGUF has incomplete per-layer tensor geometry")
    if int(geometry.context_length) < manifest.model_max_context_tokens:
        raise ValueError("direct GGUF context is smaller than the logical model contract")
    if int(geometry.kv_bytes_per_token) != sum(manifest.kv_bytes_per_token_by_layer):
        raise ValueError("direct GGUF KV geometry differs from the logical model contract")
    dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(manifest.dtype)
    if dtype_bytes is None or manifest.activation_bytes_per_token % dtype_bytes:
        raise ValueError("logical model has unsupported activation geometry")
    if int(geometry.activation_width) != manifest.activation_bytes_per_token // dtype_bytes:
        raise ValueError("direct GGUF activation width differs from the logical model contract")
    plan = SkippyExecutionPlan(
        plan_id=plan_id,
        format="gguf-direct",
        quantization=quantization,
        package_repository_id=repository_id,
        package_revision=immutable_revision,
        source_model_paths=source_paths,
        package_schema_version=None,
        package_source_sha256=_direct_source_hash(descriptors),
        runtime_release=runtime_release,
        runtime_abi_version=runtime_abi_version,
        required_runtime_features=_DIRECT_REQUIRED_FEATURES,
        activation_width=int(geometry.activation_width),
        activation_bytes_per_token=int(geometry.activation_width) * 4,
        cache_type_k="f16",
        cache_type_v="f16",
        kv_bytes_per_token_by_layer=manifest.kv_bytes_per_token_by_layer,
        model_max_context_tokens=manifest.model_max_context_tokens,
        providers=providers,
        direct_static_bytes_by_layer=static_bytes,
    )
    existing_paths = {artifact.path for artifact in base_bundle.artifact_index.artifacts}
    collisions = sorted(existing_paths & {artifact.path for artifact in descriptors})
    if collisions:
        raise ValueError(f"direct GGUF paths collide with existing artifacts: {collisions}")
    if any(
        existing.plan_id == plan.plan_id for existing in base_bundle.artifact_index.execution_plans
    ):
        raise ValueError(f"execution plan id already exists: {plan.plan_id}")
    index = ModelArtifactIndex(
        model_id=base_bundle.artifact_index.model_id,
        immutable_revision=base_bundle.artifact_index.immutable_revision,
        artifacts=tuple(
            sorted(
                (*base_bundle.artifact_index.artifacts, *descriptors),
                key=lambda artifact: artifact.path,
            )
        ),
        tensors=base_bundle.artifact_index.tensors,
        execution_plans=tuple(
            sorted(
                (*base_bundle.artifact_index.execution_plans, plan),
                key=lambda execution_plan: execution_plan.plan_id,
            )
        ),
    )
    bound_manifest = ModelManifest.model_validate(
        {
            **manifest.model_dump(mode="json"),
            "execution_plan_hash": execution_plan_hash(index),
        }
    )
    return ModelRegistryBundle(manifest=bound_manifest, artifact_index=index)


def attach_skippy_package(
    base_bundle: ModelRegistryBundle,
    *,
    package_repository_id: str,
    package_revision: str | None,
    plan_id: str,
    runtime_release: str,
    runtime_abi_version: str,
    providers: tuple[ExecutionProviderKind, ...] = _DEFAULT_PROVIDERS,
    token: bool | str | None = None,
    api: HfApi | None = None,
    downloader: Callable[..., str] = hf_hub_download,
) -> ModelRegistryBundle:
    """Resolve, validate, and bind one public Skippy package to a Fabi bundle."""

    client = api or HfApi()
    info = client.model_info(
        package_repository_id,
        revision=package_revision,
        files_metadata=True,
        token=token,
    )
    immutable_revision = str(_field(info, "sha") or "").lower()
    if len(immutable_revision) != _COMMIT_LENGTH or not all(
        character in "0123456789abcdef" for character in immutable_revision
    ):
        raise ValueError("Hugging Face did not resolve an immutable package commit")
    siblings = {
        str(_field(sibling, "rfilename")): sibling for sibling in (_field(info, "siblings") or ())
    }
    package_bytes, package = _read_package_manifest(
        package_repository_id,
        immutable_revision,
        token=token,
        downloader=downloader,
    )
    if package.get("schema_version") != 1 or package.get("format") != "layer-package":
        raise ValueError("unsupported Skippy package format")
    if package.get("layer_count") != base_bundle.manifest.num_layers:
        raise ValueError("Skippy package and Fabi model have different layer counts")
    activation_width = package.get("activation_width")
    if not isinstance(activation_width, int) or activation_width <= 0:
        raise ValueError("Skippy package has no valid activation width")
    package_model_id = str(package.get("model_id", ""))
    package_abi_version = str(package.get("skippy_abi_version", ""))
    source = _mapping(package.get("source_model"), "source_model")
    source_sha256 = str(source.get("sha256", "")).lower()
    if len(source_sha256) != 64:
        raise ValueError("Skippy package source model has no SHA-256")

    manifest_descriptor = ArtifactDescriptor(
        path="model-package.json",
        size=len(package_bytes),
        sha256=hashlib.sha256(package_bytes).hexdigest(),
        media_type="application/vnd.mesh-llm.layer-package+json",
        role=ArtifactRole.EXECUTION_PACKAGE_MANIFEST,
    )
    remote_manifest = siblings.get("model-package.json")
    if remote_manifest is None or _field(remote_manifest, "size") != len(package_bytes):
        raise ValueError("Skippy package manifest metadata does not match downloaded bytes")

    shared = _mapping(package.get("shared"), "shared")
    shared_entries = {
        key: _artifact_entry(shared.get(key), f"shared.{key}")
        for key in ("metadata", "embeddings", "output")
    }
    layers = package.get("layers")
    if not isinstance(layers, list) or len(layers) != base_bundle.manifest.num_layers:
        raise ValueError("Skippy package layers do not cover the Fabi model")
    layer_entries = []
    for expected_index, raw_layer in enumerate(layers):
        layer = _mapping(raw_layer, f"layer {expected_index}")
        if layer.get("layer_index") != expected_index:
            raise ValueError("Skippy package layers must be ordered and contiguous")
        layer_entries.append(_artifact_entry(layer, f"layer {expected_index}"))

    execution_artifacts = [manifest_descriptor]
    for path, size, digest in shared_entries.values():
        execution_artifacts.append(
            _remote_descriptor(
                path=path,
                size=size,
                sha256=digest,
                role=ArtifactRole.EXECUTION_SHARED,
                siblings=siblings,
            )
        )
    for path, size, digest in layer_entries:
        execution_artifacts.append(
            _remote_descriptor(
                path=path,
                size=size,
                sha256=digest,
                role=ArtifactRole.EXECUTION_LAYER,
                siblings=siblings,
            )
        )

    quantization = package_model_id.rpartition(":")[2]
    if not quantization or quantization == package_model_id:
        distribution_id = source.get("distribution_id")
        quantization = str(distribution_id or "").strip()
    if not quantization:
        raise ValueError("Skippy package does not identify its quantization/distribution")
    plan = SkippyExecutionPlan(
        plan_id=plan_id,
        format="gguf-layer-package",
        quantization=quantization,
        package_repository_id=package_repository_id,
        package_revision=immutable_revision,
        package_manifest_path=manifest_descriptor.path,
        package_manifest_sha256=manifest_descriptor.sha256,
        package_model_id=package_model_id,
        package_source_sha256=source_sha256,
        package_abi_version=package_abi_version,
        runtime_release=runtime_release,
        runtime_abi_version=runtime_abi_version,
        required_runtime_features=_REQUIRED_FEATURES,
        activation_width=activation_width,
        activation_bytes_per_token=activation_width * 4,
        cache_type_k="f16",
        cache_type_v="f16",
        kv_bytes_per_token_by_layer=base_bundle.manifest.kv_bytes_per_token_by_layer,
        model_max_context_tokens=base_bundle.manifest.model_max_context_tokens,
        providers=providers,
        shared_metadata_path=shared_entries["metadata"][0],
        embeddings_path=shared_entries["embeddings"][0],
        output_path=shared_entries["output"][0],
        layer_paths=tuple(entry[0] for entry in layer_entries),
    )
    existing_paths = {artifact.path for artifact in base_bundle.artifact_index.artifacts}
    collisions = sorted(existing_paths & {artifact.path for artifact in execution_artifacts})
    if collisions:
        raise ValueError(f"Skippy package paths collide with existing artifacts: {collisions}")
    if any(
        existing.plan_id == plan.plan_id for existing in base_bundle.artifact_index.execution_plans
    ):
        raise ValueError(f"execution plan id already exists: {plan.plan_id}")
    index = ModelArtifactIndex(
        model_id=base_bundle.artifact_index.model_id,
        immutable_revision=base_bundle.artifact_index.immutable_revision,
        artifacts=tuple(
            sorted(
                (*base_bundle.artifact_index.artifacts, *execution_artifacts),
                key=lambda artifact: artifact.path,
            )
        ),
        tensors=base_bundle.artifact_index.tensors,
        execution_plans=tuple(
            sorted(
                (*base_bundle.artifact_index.execution_plans, plan),
                key=lambda execution_plan: execution_plan.plan_id,
            )
        ),
    )
    manifest = ModelManifest.model_validate(
        {
            **base_bundle.manifest.model_dump(mode="json"),
            "execution_plan_hash": execution_plan_hash(index),
        }
    )
    return ModelRegistryBundle(manifest=manifest, artifact_index=index)
