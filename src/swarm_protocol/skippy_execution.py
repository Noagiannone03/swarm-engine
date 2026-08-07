"""Fail-closed materialization of signed Skippy sparse-GGUF layer packages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from parallax.server.backend_capabilities import DeviceKind, device_kind
from swarm_protocol.artifact_verification import verify_artifact
from swarm_protocol.contracts import (
    ArtifactDescriptor,
    ExecutionProviderKind,
    LayerSpan,
    ModelArtifactIndex,
    ModelManifest,
    SkippyExecutionPlan,
)

_MAX_PACKAGE_MANIFEST_BYTES = 16 * 1024 * 1024
_DEVICE_PROVIDERS = {
    DeviceKind.CPU: ExecutionProviderKind.CPU,
    DeviceKind.CUDA: ExecutionProviderKind.CUDA,
    DeviceKind.METAL: ExecutionProviderKind.METAL,
    DeviceKind.ROCM: ExecutionProviderKind.ROCM,
    DeviceKind.VULKAN: ExecutionProviderKind.VULKAN,
}


@dataclass(frozen=True)
class VerifiedSkippySpan:
    """Exact package parts verified for one worker's contiguous layer range."""

    plan: SkippyExecutionPlan
    span: LayerSpan
    package_root: Path
    package_manifest_path: Path
    part_paths: tuple[Path, ...]
    artifact_bytes: int
    artifact_hashes: tuple[str, ...]

    @property
    def weight_hashes(self) -> tuple[str, ...]:
        """Compatibility name used by the signed READY lease."""

        return self.artifact_hashes


def skippy_execution_provider_for_device(device: str) -> ExecutionProviderKind:
    """Map only native-runtime devices supported by a Skippy release bundle."""

    kind = device_kind(device)
    try:
        return _DEVICE_PROVIDERS[kind]
    except KeyError as exc:
        raise ValueError(f"device {device!r} has no Skippy native runtime provider") from exc


def select_skippy_execution_plan(
    artifact_index: ModelArtifactIndex,
    *,
    device: str,
    plan_id: str | None = None,
) -> SkippyExecutionPlan:
    """Select exactly one signed Skippy plan compatible with the concrete device."""

    provider = skippy_execution_provider_for_device(device)
    candidates = tuple(
        plan
        for plan in artifact_index.execution_plans
        if isinstance(plan, SkippyExecutionPlan)
        and provider in plan.providers
        and (plan_id is None or plan.plan_id == plan_id)
    )
    if not candidates:
        requested = f" plan {plan_id!r}" if plan_id is not None else ""
        raise ValueError(f"no signed Skippy{requested} supports {provider.value}")
    if len(candidates) != 1:
        raise ValueError("multiple signed Skippy plans support this device; select a plan id")
    return candidates[0]


def required_skippy_descriptors(
    artifact_index: ModelArtifactIndex,
    plan: SkippyExecutionPlan,
    manifest: ModelManifest,
    span: LayerSpan,
) -> tuple[ArtifactDescriptor, ...]:
    """Return the manifest, shared boundary parts, and exact assigned layers."""

    if span.end > manifest.num_layers or len(plan.layer_paths) != manifest.num_layers:
        raise ValueError("Skippy execution span exceeds the signed model layer count")
    paths = [plan.package_manifest_path, plan.shared_metadata_path]
    if span.start == 0:
        paths.append(plan.embeddings_path)
    paths.extend(plan.layer_paths[span.start : span.end])
    if span.end == manifest.num_layers:
        paths.append(plan.output_path)
    descriptors = {artifact.path: artifact for artifact in artifact_index.artifacts}
    try:
        return tuple(descriptors[path] for path in paths)
    except KeyError as exc:
        raise ValueError(f"Skippy plan references unsigned artifact {exc.args[0]!r}") from exc


def required_skippy_storage_bytes(
    artifact_index: ModelArtifactIndex,
    plan: SkippyExecutionPlan,
    manifest: ModelManifest,
    span: LayerSpan,
) -> int:
    return sum(
        descriptor.size
        for descriptor in required_skippy_descriptors(artifact_index, plan, manifest, span)
    )


def skippy_span_static_bytes(
    artifact_index: ModelArtifactIndex,
    plan: SkippyExecutionPlan,
    manifest: ModelManifest,
    span: LayerSpan,
) -> int | None:
    """Return exact static bytes; every transformer-layer boundary is executable."""

    if span.end > manifest.num_layers:
        return None
    return required_skippy_storage_bytes(artifact_index, plan, manifest, span)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Skippy package {label} must be an object")
    return value


def _manifest_artifact(
    value: object,
    *,
    label: str,
    descriptor: ArtifactDescriptor,
) -> None:
    artifact = _object(value, label)
    if artifact.get("path") != descriptor.path:
        raise ValueError(f"Skippy package {label} path does not match the signed plan")
    if artifact.get("artifact_bytes") != descriptor.size:
        raise ValueError(f"Skippy package {label} size does not match the signed descriptor")
    if str(artifact.get("sha256", "")).lower() != descriptor.sha256:
        raise ValueError(f"Skippy package {label} digest does not match the signed descriptor")


def _verify_package_manifest_contract(
    root: Path,
    artifact_index: ModelArtifactIndex,
    manifest: ModelManifest,
    plan: SkippyExecutionPlan,
) -> Path:
    descriptor_by_path = {artifact.path: artifact for artifact in artifact_index.artifacts}
    package_path = verify_artifact(root, descriptor_by_path[plan.package_manifest_path])
    if package_path.stat().st_size > _MAX_PACKAGE_MANIFEST_BYTES:
        raise ValueError("Skippy package manifest exceeds 16 MiB")
    try:
        payload = json.loads(package_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Skippy package manifest is not valid UTF-8 JSON") from exc
    package = _object(payload, "manifest")
    expected_scalars = {
        "schema_version": plan.package_schema_version,
        "format": "layer-package",
        "model_id": plan.package_model_id,
        "layer_count": manifest.num_layers,
        "activation_width": plan.activation_width,
        "skippy_abi_version": plan.package_abi_version,
    }
    for field, expected in expected_scalars.items():
        if package.get(field) != expected:
            raise ValueError(f"Skippy package field {field!r} does not match the signed plan")
    source = _object(package.get("source_model"), "source_model")
    if str(source.get("sha256", "")).lower() != plan.package_source_sha256:
        raise ValueError("Skippy package source digest does not match the signed plan")
    shared = _object(package.get("shared"), "shared")
    for key, path in (
        ("metadata", plan.shared_metadata_path),
        ("embeddings", plan.embeddings_path),
        ("output", plan.output_path),
    ):
        _manifest_artifact(
            shared.get(key),
            label=f"shared.{key}",
            descriptor=descriptor_by_path[path],
        )
    layers = package.get("layers")
    if not isinstance(layers, list) or len(layers) != manifest.num_layers:
        raise ValueError("Skippy package layers do not cover the signed model")
    for layer_index, (raw_layer, path) in enumerate(zip(layers, plan.layer_paths, strict=True)):
        layer = _object(raw_layer, f"layer {layer_index}")
        if layer.get("layer_index") != layer_index:
            raise ValueError("Skippy package layers must be ordered and contiguous")
        _manifest_artifact(
            layer,
            label=f"layer {layer_index}",
            descriptor=descriptor_by_path[path],
        )
    return package_path


def verify_skippy_execution_span(
    root: Path,
    artifact_index: ModelArtifactIndex,
    manifest: ModelManifest,
    span: LayerSpan,
    *,
    device: str,
    plan_id: str | None = None,
) -> VerifiedSkippySpan:
    """Verify package semantics and every selected byte before READY."""

    plan = select_skippy_execution_plan(artifact_index, device=device, plan_id=plan_id)
    package_manifest_path = _verify_package_manifest_contract(
        root,
        artifact_index,
        manifest,
        plan,
    )
    required = required_skippy_descriptors(artifact_index, plan, manifest, span)
    selected_parts = tuple(
        verify_artifact(root, descriptor)
        for descriptor in required
        if descriptor.path != plan.package_manifest_path
    )
    return VerifiedSkippySpan(
        plan=plan,
        span=span,
        package_root=root,
        package_manifest_path=package_manifest_path,
        part_paths=selected_parts,
        artifact_bytes=sum(descriptor.size for descriptor in required),
        artifact_hashes=tuple(descriptor.sha256 for descriptor in required),
    )


def _download_package_snapshot(**kwargs) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(**kwargs))


def materialize_skippy_execution_span(
    artifact_index: ModelArtifactIndex,
    manifest: ModelManifest,
    span: LayerSpan,
    *,
    device: str,
    plan_id: str | None = None,
    local_files_only: bool = False,
    token: bool | str | None = None,
    max_workers: int = 4,
    snapshot_downloader: Callable[..., Path] = _download_package_snapshot,
) -> VerifiedSkippySpan:
    """Download and verify only the package parts required by one worker span."""

    if max_workers <= 0:
        raise ValueError("Skippy snapshot download workers must be positive")
    plan = select_skippy_execution_plan(artifact_index, device=device, plan_id=plan_id)
    required = required_skippy_descriptors(artifact_index, plan, manifest, span)
    root = Path(
        snapshot_downloader(
            repo_id=plan.package_repository_id,
            revision=plan.package_revision,
            allow_patterns=[descriptor.path for descriptor in required],
            local_files_only=local_files_only,
            token=token,
            max_workers=max_workers,
        )
    )
    return verify_skippy_execution_span(
        root,
        artifact_index,
        manifest,
        span,
        device=device,
        plan_id=plan.plan_id,
    )
