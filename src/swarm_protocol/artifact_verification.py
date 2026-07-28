"""Fail-closed verification of the model bytes used by a worker span.

The signed registry index authenticates every runtime artifact.  This module turns a layer span
into the exact checkpoint files needed by the existing Parallax loaders and hashes those files
before a worker is allowed to advertise a READY lease.  The mapping deliberately reuses the same
weight-key rules as the loader so admission and execution cannot silently disagree.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from parallax.utils.weight_filter_utils import (
    normalize_language_model_weight_key,
    should_include_weight_key,
)
from swarm_protocol.contracts import (
    ArtifactDescriptor,
    ArtifactRole,
    LayerSpan,
    ModelArtifactIndex,
    ModelManifest,
)

_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_WEIGHT_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


@dataclass(frozen=True)
class VerifiedSpanArtifacts:
    """Verified files and weight identities bound into a worker span lease."""

    runtime_paths: tuple[Path, ...]
    weight_paths: tuple[Path, ...]
    weight_hashes: tuple[str, ...]


def _resolved_artifact_path(root: Path, descriptor: ArtifactDescriptor) -> Path:
    resolved_root = root.resolve(strict=True)
    logical_path = resolved_root / descriptor.path
    candidate = logical_path.resolve(strict=True)
    # Hugging Face snapshots intentionally contain symlinks into their sibling
    # content-addressed ``blobs`` directory. The signed descriptor already
    # constrains the logical path, exact length and file digest, so rejecting a
    # resolved path outside the snapshot would reject the official cache layout
    # without adding integrity. A mismatched symlink target still fails below.
    if not candidate.is_file():
        raise ValueError(f"artifact is not a regular file: {descriptor.path!r}")
    return candidate


def verify_artifact(root: Path, descriptor: ArtifactDescriptor) -> Path:
    """Verify one local artifact by exact length and streaming SHA-256."""

    path = _resolved_artifact_path(root, descriptor)
    stat_before = path.stat()
    if stat_before.st_size != descriptor.size:
        raise ValueError(
            f"artifact size mismatch for {descriptor.path!r}: "
            f"expected {descriptor.size}, got {stat_before.st_size}"
        )

    digest = hashlib.sha256()
    bytes_read = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            bytes_read += len(chunk)
            digest.update(chunk)

    stat_after = path.stat()
    if (
        bytes_read != descriptor.size
        or stat_after.st_size != stat_before.st_size
        or stat_after.st_mtime_ns != stat_before.st_mtime_ns
        or getattr(stat_after, "st_ino", None) != getattr(stat_before, "st_ino", None)
    ):
        raise ValueError(f"artifact changed while it was being verified: {descriptor.path!r}")
    actual = digest.hexdigest()
    if actual != descriptor.sha256:
        raise ValueError(
            f"artifact SHA-256 mismatch for {descriptor.path!r}: "
            f"expected {descriptor.sha256}, got {actual}"
        )
    return path


def _load_verified_json(root: Path, descriptor: ArtifactDescriptor) -> dict[str, object]:
    path = verify_artifact(root, descriptor)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"verified artifact is not valid JSON: {descriptor.path!r}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"verified JSON artifact must contain an object: {descriptor.path!r}")
    return value


def required_weight_descriptors(
    root: Path,
    artifact_index: ModelArtifactIndex,
    manifest: ModelManifest,
    span: LayerSpan,
) -> tuple[ArtifactDescriptor, ...]:
    """Resolve the signed checkpoint files required to execute ``span``.

    Sharded checkpoints are resolved from their authenticated weight map.  A mapping that points
    outside the signed index, or one that cannot produce files for a non-empty span, is rejected
    instead of falling back to an unverified whole-repository download.
    """

    if span.end > manifest.num_layers:
        raise ValueError("worker span exceeds model layer count")
    descriptors = {artifact.path: artifact for artifact in artifact_index.artifacts}
    weight_descriptors = {
        path: descriptor
        for path, descriptor in descriptors.items()
        if descriptor.role is ArtifactRole.WEIGHT
    }
    if not weight_descriptors:
        raise ValueError("model artifact index has no weight artifacts")

    index_descriptor = next(
        (weight_descriptors[name] for name in _WEIGHT_INDEX_NAMES if name in weight_descriptors),
        None,
    )
    if index_descriptor is None:
        checkpoint_files = [
            descriptor
            for descriptor in weight_descriptors.values()
            if not descriptor.path.endswith(".index.json")
        ]
        if len(checkpoint_files) != 1:
            raise ValueError(
                "checkpoint without a supported weight index must contain exactly one weight file"
            )
        return tuple(checkpoint_files)

    index_data = _load_verified_json(root, index_descriptor)
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"weight index has no non-empty weight_map: {index_descriptor.path!r}")

    config_descriptor = descriptors.get("config.json")
    if config_descriptor is None or config_descriptor.role is not ArtifactRole.ARCHITECTURE:
        raise ValueError("signed artifact index has no architecture config.json")
    config = _load_verified_json(root, config_descriptor)
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
    required_paths: set[str] = set()

    for raw_key, raw_filename in weight_map.items():
        if not isinstance(raw_key, str) or not isinstance(raw_filename, str):
            raise ValueError("weight_map keys and filenames must be strings")
        filename = raw_filename.replace(os.sep, "/")
        descriptor = weight_descriptors.get(filename)
        if descriptor is None or filename.endswith(".index.json"):
            raise ValueError(f"weight_map references an unsigned checkpoint file: {filename!r}")
        key = normalize_language_model_weight_key(raw_key)
        try:
            include = should_include_weight_key(
                key=key,
                start_layer=span.start,
                end_layer=span.end,
                is_first_shard=span.start == 0,
                is_last_shard=span.end == manifest.num_layers,
                tie_word_embeddings=tie_word_embeddings,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid layer key in weight_map: {raw_key!r}") from exc
        if include:
            required_paths.add(filename)

    if not required_paths:
        raise ValueError(
            f"weight_map contains no checkpoint files for span [{span.start}, {span.end})"
        )
    return tuple([index_descriptor] + [weight_descriptors[path] for path in sorted(required_paths)])


def verify_worker_span(
    root: Path,
    artifact_index: ModelArtifactIndex,
    manifest: ModelManifest,
    span: LayerSpan,
    *,
    include_tokenizer: bool = False,
) -> VerifiedSpanArtifacts:
    """Verify all executable metadata and exact weights needed by a worker span."""

    if artifact_index.model_id != manifest.model_id:
        raise ValueError("artifact index and model manifest identify different models")
    if artifact_index.immutable_revision != manifest.immutable_revision:
        raise ValueError("artifact index and model manifest use different immutable revisions")

    runtime_descriptors = tuple(
        artifact
        for artifact in artifact_index.artifacts
        if artifact.role is ArtifactRole.ARCHITECTURE
        or (include_tokenizer and artifact.role is ArtifactRole.TOKENIZER)
    )
    runtime_paths = tuple(verify_artifact(root, artifact) for artifact in runtime_descriptors)
    if artifact_index.tensors:
        from parallax.utils.selective_safetensors import (
            selective_receipt_path,
            verify_selective_projection,
        )

        if selective_receipt_path(root).is_file():
            config_descriptor = next(
                (
                    artifact
                    for artifact in artifact_index.artifacts
                    if artifact.path == "config.json" and artifact.role is ArtifactRole.ARCHITECTURE
                ),
                None,
            )
            if config_descriptor is None:
                raise ValueError("signed artifact index has no architecture config.json")
            config = _load_verified_json(root, config_descriptor)
            weight_paths, weight_hashes = verify_selective_projection(
                model_root=root,
                artifact_index=artifact_index,
                start_layer=span.start,
                end_layer=span.end,
                num_layers=manifest.num_layers,
                tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            )
            return VerifiedSpanArtifacts(
                runtime_paths=runtime_paths,
                weight_paths=weight_paths,
                weight_hashes=weight_hashes,
            )

    weight_descriptors = required_weight_descriptors(root, artifact_index, manifest, span)
    weight_paths = tuple(verify_artifact(root, artifact) for artifact in weight_descriptors)
    return VerifiedSpanArtifacts(
        runtime_paths=runtime_paths,
        weight_paths=weight_paths,
        weight_hashes=tuple(artifact.sha256 for artifact in weight_descriptors),
    )
