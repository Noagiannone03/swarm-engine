"""Fail-closed materialization of signed Skippy GGUF execution sources."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
import requests

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
_HUB_SNAPSHOT_RETRIES = 8
_HUB_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
_DEVICE_PROVIDERS = {
    DeviceKind.CPU: ExecutionProviderKind.CPU,
    DeviceKind.CUDA: ExecutionProviderKind.CUDA,
    DeviceKind.METAL: ExecutionProviderKind.METAL,
    DeviceKind.ROCM: ExecutionProviderKind.ROCM,
    DeviceKind.VULKAN: ExecutionProviderKind.VULKAN,
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifiedSkippySpan:
    """Exact GGUF source bytes verified for one worker's contiguous range."""

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

    @property
    def geometry_path(self) -> Path:
        """Authenticated GGUF carrying the architecture and KV metadata."""

        if self.plan.format == "gguf-direct":
            return self.part_paths[0]
        assert self.plan.shared_metadata_path is not None
        return self.package_root / self.plan.shared_metadata_path


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
    """Return the complete direct source or exact package parts for a span."""

    if (
        span.end > manifest.num_layers
        or len(plan.kv_bytes_per_token_by_layer) != manifest.num_layers
    ):
        raise ValueError("Skippy execution span exceeds the signed model layer count")
    descriptors = {artifact.path: artifact for artifact in artifact_index.artifacts}
    if plan.format == "gguf-direct":
        try:
            return tuple(descriptors[path] for path in plan.source_model_paths)
        except KeyError as exc:  # pragma: no cover - index contract catches this
            raise ValueError("direct Skippy plan references an unsigned GGUF") from exc
    assert plan.package_manifest_path is not None
    assert plan.shared_metadata_path is not None
    assert plan.embeddings_path is not None
    assert plan.output_path is not None
    if len(plan.layer_paths) != manifest.num_layers:
        raise ValueError("Skippy layer package does not cover the signed model")
    paths = [plan.package_manifest_path, plan.shared_metadata_path]
    # Mesh's maintained LayerPackage loader requires the token embeddings on
    # both boundary roles: the input stage embeds prompt tokens, while the
    # terminal stage embeds each sampled token before the next decode step.
    # A single-stage replica naturally satisfies both roles with one artifact.
    if span.start == 0 or span.end == manifest.num_layers:
        paths.append(plan.embeddings_path)
    paths.extend(plan.layer_paths[span.start : span.end])
    if span.end == manifest.num_layers:
        paths.append(plan.output_path)
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
    """Return exact resident tensor bytes for an executable transformer span."""

    if span.end > manifest.num_layers:
        return None
    if plan.format == "gguf-direct":
        if len(plan.direct_static_bytes_by_layer) != manifest.num_layers:
            return None
        return sum(plan.direct_static_bytes_by_layer[span.start : span.end])
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
    if plan.format != "gguf-layer-package":
        raise ValueError("direct GGUF execution has no layer-package manifest")
    assert plan.package_manifest_path is not None
    assert plan.package_model_id is not None
    assert plan.package_abi_version is not None
    assert plan.shared_metadata_path is not None
    assert plan.embeddings_path is not None
    assert plan.output_path is not None
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
    if plan.format == "gguf-layer-package":
        package_manifest_path = _verify_package_manifest_contract(
            root,
            artifact_index,
            manifest,
            plan,
        )
    else:
        direct_descriptor = next(
            artifact
            for artifact in artifact_index.artifacts
            if artifact.path == plan.source_model_paths[0]
        )
        package_manifest_path = verify_artifact(root, direct_descriptor)
    required = required_skippy_descriptors(artifact_index, plan, manifest, span)
    selected_parts = tuple(
        verify_artifact(root, descriptor)
        for descriptor in required
        if plan.format == "gguf-direct" or descriptor.path != plan.package_manifest_path
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


def _snapshot_progress_tqdm_class(
    progress_callback: Callable[[int, int], None],
) -> type:
    """Build a Hugging Face progress class reporting completed snapshot files.

    ``snapshot_download`` uses the same tqdm class for its byte-level download
    bars and for the outer ``Fetching N files`` bar.  Only the latter represents
    stable package progress, so filename/byte bars are deliberately ignored.
    Telemetry is best-effort and can never fail materialization.
    """

    from huggingface_hub.utils import tqdm as huggingface_tqdm

    progress_lock = threading.Lock()
    reported_done = 0
    reported_total = 0
    has_reported = False

    class SnapshotProgressTqdm(huggingface_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            description = str(kwargs.get("desc") or "")
            total = kwargs.get("total")
            self._fabi_file_counter = description.startswith("Fetching ") and isinstance(total, int)
            self._fabi_files_done = int(kwargs.get("initial") or 0)
            self._fabi_files_total = int(total) if self._fabi_file_counter else 0
            self._fabi_progress_lock = threading.Lock()
            super().__init__(*args, **kwargs)
            if self._fabi_file_counter:
                self._fabi_notify()

        def _fabi_notify(self) -> None:
            nonlocal reported_done, reported_total, has_reported
            try:
                with progress_lock:
                    previous = (reported_done, reported_total)
                    reported_total = max(reported_total, self._fabi_files_total)
                    reported_done = min(
                        reported_total,
                        max(reported_done, self._fabi_files_done),
                    )
                    progress = (reported_done, reported_total)
                    if has_reported and progress == previous:
                        return
                    has_reported = True
                progress_callback(*progress)
            except Exception:
                # UI telemetry must never interrupt an authenticated download.
                pass

        def update(self, n: int | float = 1) -> bool | None:
            result = super().update(n)
            if self._fabi_file_counter:
                with self._fabi_progress_lock:
                    self._fabi_files_done = min(
                        self._fabi_files_total,
                        self._fabi_files_done + max(0, int(n)),
                    )
                    self._fabi_notify()
            return result

    return SnapshotProgressTqdm


def _download_snapshot_with_backoff(
    snapshot_downloader: Callable[..., Path],
    download_kwargs: dict[str, Any],
) -> Path:
    """Resume one immutable Hub snapshot across transient transport outages."""

    for attempt in range(_HUB_SNAPSHOT_RETRIES + 1):
        try:
            return Path(snapshot_downloader(**download_kwargs))
        except _HUB_NETWORK_ERRORS as exc:
            if attempt == _HUB_SNAPSHOT_RETRIES:
                raise
            delay = min(2**attempt, 30)
            logger.warning(
                "Hub snapshot download failed (%s); preserving cache and retrying in %ds "
                "(%d/%d)",
                type(exc).__name__,
                delay,
                attempt + 1,
                _HUB_SNAPSHOT_RETRIES,
            )
            time.sleep(delay)
    raise AssertionError("unreachable Hub snapshot retry state")


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
    progress_callback: Callable[[int, int], None] | None = None,
    snapshot_downloader: Callable[..., Path] = _download_package_snapshot,
) -> VerifiedSkippySpan:
    """Download and verify only the package parts required by one worker span."""

    if max_workers <= 0:
        raise ValueError("Skippy snapshot download workers must be positive")
    plan = select_skippy_execution_plan(artifact_index, device=device, plan_id=plan_id)
    required = required_skippy_descriptors(artifact_index, plan, manifest, span)
    download_kwargs: dict[str, Any] = {
        "repo_id": plan.package_repository_id,
        "revision": plan.package_revision,
        "allow_patterns": [descriptor.path for descriptor in required],
        "local_files_only": local_files_only,
        "token": token,
        "max_workers": max_workers,
    }
    if progress_callback is not None:
        download_kwargs["tqdm_class"] = _snapshot_progress_tqdm_class(progress_callback)
    root = _download_snapshot_with_backoff(snapshot_downloader, download_kwargs)
    return verify_skippy_execution_span(
        root,
        artifact_index,
        manifest,
        span,
        device=device,
        plan_id=plan.plan_id,
    )
