"""Fail-closed selection and verification of portable layer-stage artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from parallax.server.backend_capabilities import DeviceKind, device_kind
from swarm_protocol.artifact_verification import verify_artifact
from swarm_protocol.contracts import (
    ArtifactDescriptor,
    ExecutionProviderKind,
    ExecutionStageDescriptor,
    ExecutionStageKind,
    LayerSpan,
    ModelArtifactIndex,
    ModelExecutionPlan,
    ModelManifest,
)

_DEVICE_PROVIDERS = {
    DeviceKind.WINML: ExecutionProviderKind.WINML,
    DeviceKind.DIRECTML: ExecutionProviderKind.DIRECTML,
    DeviceKind.OPENVINO: ExecutionProviderKind.OPENVINO,
    DeviceKind.QNN: ExecutionProviderKind.QNN,
}


@dataclass(frozen=True)
class VerifiedExecutionStage:
    descriptor: ExecutionStageDescriptor
    graph_path: Path
    external_data_paths: tuple[Path, ...]


@dataclass(frozen=True)
class VerifiedExecutionSpan:
    plan: ModelExecutionPlan
    span: LayerSpan
    stages: tuple[VerifiedExecutionStage, ...]
    artifact_bytes: int
    artifact_hashes: tuple[str, ...] = ()

    @property
    def weight_hashes(self) -> tuple[str, ...]:
        """Compatibility name used by the signed READY lease."""

        return self.artifact_hashes


def execution_provider_for_device(device: str) -> ExecutionProviderKind:
    """Resolve only device families with a portable execution contract."""

    kind = device_kind(device)
    try:
        return _DEVICE_PROVIDERS[kind]
    except KeyError as exc:
        raise ValueError(f"device {device!r} has no portable execution provider") from exc


def select_execution_plan(
    artifact_index: ModelArtifactIndex,
    *,
    device: str,
    plan_id: str | None = None,
) -> ModelExecutionPlan:
    """Select one signed plan compatible with the concrete execution device."""

    provider = execution_provider_for_device(device)
    candidates = tuple(
        plan
        for plan in artifact_index.execution_plans
        if provider in plan.providers and (plan_id is None or plan.plan_id == plan_id)
    )
    if not candidates:
        requested = f" plan {plan_id!r}" if plan_id is not None else ""
        raise ValueError(f"no signed{requested} execution plan supports {provider.value}")
    if len(candidates) != 1:
        raise ValueError(
            "multiple signed execution plans support this device; an explicit plan id is required"
        )
    return candidates[0]


def stages_for_span(
    plan: ModelExecutionPlan,
    manifest: ModelManifest,
    span: LayerSpan,
) -> tuple[ExecutionStageDescriptor, ...]:
    """Return an exact composable stage sequence for a worker layer span."""

    if span.end > manifest.num_layers:
        raise ValueError("portable execution span exceeds model layer count")
    decoders = sorted(
        (stage for stage in plan.stages if stage.kind is ExecutionStageKind.DECODER),
        key=lambda stage: stage.start_layer,
    )
    selected: list[ExecutionStageDescriptor] = []
    if span.start == 0:
        selected.extend(stage for stage in plan.stages if stage.kind is ExecutionStageKind.INPUT)
    cursor = span.start
    for stage in decoders:
        if stage.end_layer <= span.start or stage.start_layer >= span.end:
            continue
        if stage.start_layer != cursor or stage.end_layer > span.end:
            raise ValueError(
                "worker span is not aligned to signed execution-stage boundaries"
            )
        selected.append(stage)
        cursor = stage.end_layer
    if cursor != span.end:
        raise ValueError("signed execution stages do not cover the worker span")
    if span.end == manifest.num_layers:
        selected.extend(stage for stage in plan.stages if stage.kind is ExecutionStageKind.OUTPUT)
    return tuple(selected)


def required_execution_descriptors(
    artifact_index: ModelArtifactIndex,
    stages: tuple[ExecutionStageDescriptor, ...],
) -> tuple[ArtifactDescriptor, ...]:
    """Resolve and de-duplicate signed files used by a stage sequence."""

    artifacts = {artifact.path: artifact for artifact in artifact_index.artifacts}
    required_paths = {
        path
        for stage in stages
        for path in (stage.graph_path, *stage.external_data_paths)
    }
    try:
        return tuple(artifacts[path] for path in sorted(required_paths))
    except KeyError as exc:  # Defensive: ModelArtifactIndex normally rejects this first.
        raise ValueError(f"portable execution stage references unsigned artifact {exc.args[0]!r}") from exc


def required_execution_storage_bytes(
    artifact_index: ModelArtifactIndex,
    plan: ModelExecutionPlan,
    manifest: ModelManifest,
    span: LayerSpan,
) -> int:
    stages = stages_for_span(plan, manifest, span)
    return sum(
        descriptor.size
        for descriptor in required_execution_descriptors(artifact_index, stages)
    )


def verify_execution_span(
    root: Path,
    artifact_index: ModelArtifactIndex,
    manifest: ModelManifest,
    span: LayerSpan,
    *,
    device: str,
    plan_id: str | None = None,
) -> VerifiedExecutionSpan:
    """Verify every graph/data byte before a portable worker can become READY."""

    plan = select_execution_plan(artifact_index, device=device, plan_id=plan_id)
    stages = stages_for_span(plan, manifest, span)
    required = required_execution_descriptors(artifact_index, stages)
    verified_paths = {
        descriptor.path: verify_artifact(root, descriptor) for descriptor in required
    }
    verified: list[VerifiedExecutionStage] = []
    for stage in stages:
        verified.append(
            VerifiedExecutionStage(
                descriptor=stage,
                graph_path=verified_paths[stage.graph_path],
                external_data_paths=tuple(
                    verified_paths[path] for path in stage.external_data_paths
                ),
            )
        )
    artifact_bytes = required_execution_storage_bytes(
        artifact_index, plan, manifest, span
    )
    return VerifiedExecutionSpan(
        plan=plan,
        span=span,
        stages=tuple(verified),
        artifact_bytes=artifact_bytes,
        artifact_hashes=tuple(descriptor.sha256 for descriptor in required),
    )


def _download_execution_snapshot(**kwargs) -> Path:
    # Keep registry/control-plane environments importable without the Hub
    # client until a worker actually materializes a portable executor.
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(**kwargs))


def materialize_execution_span(
    artifact_index: ModelArtifactIndex,
    manifest: ModelManifest,
    span: LayerSpan,
    *,
    device: str,
    plan_id: str | None = None,
    local_files_only: bool = False,
    token: bool | str | None = None,
    max_workers: int = 4,
    snapshot_downloader: Callable[..., Path] = _download_execution_snapshot,
) -> VerifiedExecutionSpan:
    """Download only one signed portable span, then verify every local byte.

    Hugging Face owns cache locking, resumable downloads and immutable snapshot
    addressing. Fabi owns the signed allow-list and hashes, so a cache hit is
    never trusted merely because the Hub client returned a path.
    """

    if max_workers <= 0:
        raise ValueError("portable snapshot download workers must be positive")
    plan = select_execution_plan(artifact_index, device=device, plan_id=plan_id)
    stages = stages_for_span(plan, manifest, span)
    required = required_execution_descriptors(artifact_index, stages)
    root = Path(
        snapshot_downloader(
            repo_id=plan.artifact_repository_id,
            revision=plan.artifact_revision,
            allow_patterns=[descriptor.path for descriptor in required],
            local_files_only=local_files_only,
            token=token,
            max_workers=max_workers,
        )
    )
    verified = verify_execution_span(
        root,
        artifact_index,
        manifest,
        span,
        device=device,
        plan_id=plan.plan_id,
    )
    if verified.plan != plan:  # Defensive against future selector changes.
        raise RuntimeError("verified portable execution plan changed during materialization")
    return verified
