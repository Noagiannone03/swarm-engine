"""Materialize signed SafeTensors layer packs from authenticated Xet ranges.

The Hub publishes large checkpoints in file-sized shards, while Parallax owns a
layer span.  Downloading at file granularity can therefore transfer gigabytes
that the executor will never read.  This module keeps the upstream checkpoint
immutable and builds a deterministic local projection containing only the
tensors needed by a worker.

Network byte ranges are reconstructed by Hugging Face's maintained ``hf-xet``
client.  The Xet file hash and tensor offsets come from the TUF-signed Fabi
artifact index; the Hub response is used only to obtain a short-lived access
route and must agree with those signed identities.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import struct
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

from filelock import FileLock
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers

from parallax.utils.model_artifact_cache import (
    CacheObjectRequirement,
    ModelArtifactCache,
    ModelArtifactCachePool,
)
from parallax.utils.weight_filter_utils import (
    normalize_language_model_weight_key,
    should_include_weight_key,
)
from swarm_protocol.contracts import (
    ArtifactDescriptor,
    ArtifactRole,
    ModelArtifactIndex,
    TensorArtifactDescriptor,
)
from swarm_protocol.xet_transport import (
    get_hf_file_metadata_with_backoff,
    get_xet_session,
    xet_headers_without_auth,
)

_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_RECEIPT_NAME = ".fabi-selective-artifacts.json"
_SOURCE_INDEX_NAME = "model.safetensors.index.json"

logger = logging.getLogger(__name__)


def artifact_index_identity(index: ModelArtifactIndex) -> str:
    payload = json.dumps(
        index.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _verify_descriptor(path: Path, descriptor: ArtifactDescriptor) -> None:
    if not path.is_file() or path.stat().st_size != descriptor.size:
        raise ValueError(f"artifact size mismatch for {descriptor.path!r}")
    actual = _sha256_file(path)
    if actual != descriptor.sha256:
        raise ValueError(
            f"artifact SHA-256 mismatch for {descriptor.path!r}: "
            f"expected {descriptor.sha256}, got {actual}"
        )


def pack_name_for_tensor(tensor_name: str) -> str | None:
    key = normalize_language_model_weight_key(tensor_name)
    parts = key.split(".")
    if "layers" in parts:
        position = parts.index("layers") + 1
        if position >= len(parts) or not parts[position].isdigit():
            raise ValueError(f"cannot identify decoder layer for tensor {tensor_name!r}")
        return f"model-fabi-layer-{int(parts[position]):05d}.safetensors"
    if key.startswith("model.") and "embed_tokens" in key:
        return "model-fabi-embedding.safetensors"
    if "model.norm" in key or "lm_head" in key:
        return "model-fabi-output.safetensors"
    return None


def selected_tensor_artifacts(
    index: ModelArtifactIndex,
    *,
    start_layer: int,
    end_layer: int,
    num_layers: int,
    tie_word_embeddings: bool,
) -> tuple[TensorArtifactDescriptor, ...]:
    selected = []
    for tensor in index.tensors:
        key = normalize_language_model_weight_key(tensor.name)
        if should_include_weight_key(
            key=key,
            start_layer=start_layer,
            end_layer=end_layer,
            is_first_shard=start_layer == 0,
            is_last_shard=end_layer >= num_layers,
            tie_word_embeddings=tie_word_embeddings,
        ):
            if pack_name_for_tensor(tensor.name) is None:
                raise ValueError(f"selected tensor has no deterministic pack: {tensor.name!r}")
            selected.append(tensor)
    if not selected:
        raise ValueError(
            f"signed tensor index contains no weights for [{start_layer}, {end_layer})"
        )
    return tuple(selected)


def _generated_weight_index(index: ModelArtifactIndex) -> bytes:
    weight_map = {
        tensor.name: pack
        for tensor in index.tensors
        if (pack := pack_name_for_tensor(tensor.name)) is not None
    }
    if not weight_map:
        raise ValueError("signed tensor index has no executable model weights")
    payload = {
        "metadata": {
            "total_size": sum(
                tensor.length
                for tensor in index.tensors
                if pack_name_for_tensor(tensor.name) is not None
            )
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _copy_runtime_artifacts(
    metadata_root: Path,
    projection_root: Path,
    index: ModelArtifactIndex,
) -> None:
    for descriptor in index.artifacts:
        if descriptor.role is ArtifactRole.WEIGHT:
            continue
        source = metadata_root / descriptor.path
        _verify_descriptor(source, descriptor)
        destination = projection_root / descriptor.path
        if destination.exists():
            try:
                _verify_descriptor(destination, descriptor)
                continue
            except ValueError:
                # Remove the invalid copy before creating the replacement.
                # This makes the signed file size its exact net disk growth
                # instead of temporarily requiring both copies.
                destination.unlink(missing_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(source, temporary)
            _verify_descriptor(temporary, descriptor)
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _safetensors_header(
    tensors: tuple[TensorArtifactDescriptor, ...],
    *,
    immutable_revision: str,
) -> tuple[bytes, tuple[TensorArtifactDescriptor, ...]]:
    # Source order maximizes adjacent Xet ranges. SafeTensors keys do not need
    # to be stored lexicographically; each header entry points to its new
    # deterministic position in the projected data section.
    ordered = tuple(sorted(tensors, key=lambda item: (item.source_path, item.offset, item.name)))
    cursor = 0
    header: dict[str, object] = {
        "__metadata__": {
            "format": "pt",
            "fabi_projection": "layer-packs/v1",
            "fabi_source_revision": immutable_revision,
        }
    }
    for tensor in ordered:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [cursor, cursor + tensor.length],
        }
        cursor += tensor.length
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded, ordered


def _coalesced_ranges(
    tensors: tuple[TensorArtifactDescriptor, ...],
) -> tuple[tuple[str, int, int], ...]:
    ranges: list[tuple[str, int, int]] = []
    for tensor in tensors:
        if tensor.length == 0:
            continue
        start = tensor.offset
        end = tensor.offset + tensor.length
        if ranges and ranges[-1][0] == tensor.source_path and ranges[-1][2] == start:
            source_path, previous_start, _ = ranges[-1]
            ranges[-1] = (source_path, previous_start, end)
        else:
            ranges.append((tensor.source_path, start, end))
    return tuple(ranges)


class _RangeSources:
    def __init__(
        self,
        *,
        repo_id: str,
        immutable_revision: str,
        metadata_root: Path,
        sources: dict[str, ArtifactDescriptor],
        local_files_only: bool,
        token: bool | str | None,
    ) -> None:
        self.repo_id = repo_id
        self.immutable_revision = immutable_revision
        self.metadata_root = metadata_root
        self.sources = sources
        self.local_files_only = local_files_only
        self.token = token
        self._verified_local: set[str] = set()
        self._xet_groups: dict[str, object] = {}
        self.local_bytes = 0
        self.network_bytes = 0

    def _local_source(self, source_path: str) -> Path | None:
        path = self.metadata_root / source_path
        if not path.is_file():
            return None
        if source_path not in self._verified_local:
            _verify_descriptor(path, self.sources[source_path])
            self._verified_local.add(source_path)
        return path

    def _xet_group(self, source_path: str):
        existing = self._xet_groups.get(source_path)
        if existing is not None:
            return existing
        source = self.sources[source_path]
        if source.xet_file_hash is None:
            raise ValueError(f"signed artifact has no Xet identity: {source_path!r}")
        headers = build_hf_headers(token=self.token)
        remote = get_hf_file_metadata_with_backoff(
            hf_hub_url(self.repo_id, source_path, revision=self.immutable_revision),
            token=self.token,
            headers=headers,
        )
        remote_xet = remote.xet_file_data
        if remote.size != source.size or str(remote.etag or "").strip('"').lower() != source.sha256:
            raise ValueError(f"Hub source metadata disagrees with signed artifact {source_path!r}")
        if remote_xet is None or remote_xet.file_hash.lower() != source.xet_file_hash:
            raise ValueError(f"Hub Xet identity disagrees with signed artifact {source_path!r}")
        group = get_xet_session().new_download_stream_group(
            token_refresh_url=remote_xet.refresh_route,
            token_refresh_headers=headers,
            custom_headers=xet_headers_without_auth(headers),
        )
        self._xet_groups[source_path] = group
        return group

    def copy_range(self, source_path: str, start: int, end: int, destination: BinaryIO) -> int:
        if end < start:
            raise ValueError("range end precedes its start")
        expected = end - start
        local = self._local_source(source_path)
        if local is not None:
            copied = 0
            with local.open("rb") as source:
                source.seek(start)
                remaining = expected
                while remaining:
                    chunk = source.read(min(_HASH_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise OSError(f"source file ended inside tensor range: {source_path!r}")
                    destination.write(chunk)
                    copied += len(chunk)
                    remaining -= len(chunk)
            self.local_bytes += copied
            return copied
        if self.local_files_only:
            raise FileNotFoundError(f"selective tensor pack is not cached for {source_path!r}")

        from hf_xet import XetFileInfo

        source = self.sources[source_path]
        group = self._xet_group(source_path)
        copied = 0
        stream = group.download_stream(
            XetFileInfo(source.xet_file_hash, source.size),
            start=start,
            end=end,
        )
        for chunk in stream:
            destination.write(chunk)
            copied += len(chunk)
        if copied != expected:
            raise OSError(
                f"Xet reconstructed {copied} bytes for {source_path!r}, expected {expected}"
            )
        self.network_bytes += copied
        return copied


class _HashingWriter:
    def __init__(self, destination: BinaryIO) -> None:
        self.destination = destination
        self.digest = hashlib.sha256()
        self.bytes_written = 0

    def write(self, payload: bytes) -> int:
        written = self.destination.write(payload)
        if written != len(payload):
            raise OSError("short write while materializing SafeTensors pack")
        self.digest.update(payload)
        self.bytes_written += written
        return written


def _pack_spec_hash(tensors: Iterable[TensorArtifactDescriptor]) -> str:
    payload = [tensor.model_dump(mode="json", exclude_none=True) for tensor in tensors]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _load_receipt(path: Path, *, expected_identity: str) -> dict[str, object]:
    try:
        receipt = json.loads(path.read_bytes())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"version": 1, "artifact_index_sha256": expected_identity, "packs": {}}
    if (
        not isinstance(receipt, dict)
        or receipt.get("version") != 1
        or receipt.get("artifact_index_sha256") != expected_identity
        or not isinstance(receipt.get("packs"), dict)
    ):
        return {"version": 1, "artifact_index_sha256": expected_identity, "packs": {}}
    return receipt


def _valid_cached_pack(
    path: Path,
    receipt_entry: object,
    *,
    spec_hash: str,
) -> bool:
    if not isinstance(receipt_entry, dict) or receipt_entry.get("spec_sha256") != spec_hash:
        return False
    if not path.is_file() or path.stat().st_size != receipt_entry.get("size"):
        return False
    return _sha256_file(path) == receipt_entry.get("sha256")


def _verify_pack_contents(
    path: Path,
    tensors: tuple[TensorArtifactDescriptor, ...],
) -> None:
    """Verify projected metadata and every tensor against its signed digest."""

    ordered = tuple(sorted(tensors, key=lambda item: (item.source_path, item.offset, item.name)))
    with path.open("rb") as source:
        encoded_length = source.read(8)
        if len(encoded_length) != 8:
            raise ValueError(f"projected SafeTensors header is truncated: {path.name}")
        header_length = struct.unpack("<Q", encoded_length)[0]
        if header_length <= 0 or header_length > 64 * 1024 * 1024:
            raise ValueError(f"projected SafeTensors header length is invalid: {path.name}")
        encoded_header = source.read(header_length)
        if len(encoded_header) != header_length:
            raise ValueError(f"projected SafeTensors header is truncated: {path.name}")
        try:
            header = json.loads(encoded_header)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"projected SafeTensors header is invalid: {path.name}") from exc
        if not isinstance(header, dict):
            raise ValueError(f"projected SafeTensors header is not an object: {path.name}")
        metadata = header.pop("__metadata__", None)
        if not isinstance(metadata, dict) or metadata.get("fabi_projection") != "layer-packs/v1":
            raise ValueError(f"projected SafeTensors provenance is missing: {path.name}")
        if set(header) != {tensor.name for tensor in ordered}:
            raise ValueError(
                f"projected SafeTensors tensor set disagrees with registry: {path.name}"
            )

        cursor = 0
        for tensor in ordered:
            entry = header.get(tensor.name)
            if not isinstance(entry, dict):
                raise ValueError(f"projected tensor metadata is invalid: {tensor.name!r}")
            offsets = entry.get("data_offsets")
            if (
                entry.get("dtype") != tensor.dtype
                or entry.get("shape") != list(tensor.shape)
                or offsets != [cursor, cursor + tensor.length]
            ):
                raise ValueError(
                    f"projected tensor contract disagrees with registry: {tensor.name!r}"
                )
            cursor += tensor.length
        expected_size = 8 + header_length + cursor
        if path.stat().st_size != expected_size:
            raise ValueError(f"projected SafeTensors size is invalid: {path.name}")

        data_offset = 8 + header_length
        cursor = 0
        for tensor in ordered:
            source.seek(data_offset + cursor)
            digest = hashlib.sha256()
            remaining = tensor.length
            while remaining:
                chunk = source.read(min(_HASH_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ValueError(f"projected tensor is truncated: {tensor.name!r}")
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != tensor.sha256:
                raise ValueError(f"projected tensor SHA-256 mismatch: {tensor.name!r}")
            cursor += tensor.length


def _materialize_pack(
    path: Path,
    tensors: tuple[TensorArtifactDescriptor, ...],
    *,
    immutable_revision: str,
    sources: _RangeSources,
) -> dict[str, object]:
    header, ordered = _safetensors_header(tensors, immutable_revision=immutable_revision)
    local_before = sources.local_bytes
    network_before = sources.network_bytes
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw_destination:
            destination = _HashingWriter(raw_destination)
            destination.write(header)
            for source_path, start, end in _coalesced_ranges(ordered):
                sources.copy_range(source_path, start, end, destination)
            raw_destination.flush()
            os.fsync(raw_destination.fileno())
            sha256 = destination.digest.hexdigest()
            size = destination.bytes_written
        expected_size = len(header) + sum(tensor.length for tensor in ordered)
        if size != expected_size:
            raise OSError(f"projected SafeTensors size is {size}, expected {expected_size}")
        _verify_pack_contents(temporary, ordered)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {
        "sha256": sha256,
        "size": size,
        "spec_sha256": _pack_spec_hash(ordered),
        "tensors": [tensor.name for tensor in ordered],
        "source_bytes": sum(tensor.length for tensor in ordered),
        "local_bytes": sources.local_bytes - local_before,
        "network_bytes": sources.network_bytes - network_before,
    }


def _pack_requirement(
    pack_name: str,
    tensors: tuple[TensorArtifactDescriptor, ...],
    *,
    immutable_revision: str,
) -> CacheObjectRequirement:
    header, ordered = _safetensors_header(tensors, immutable_revision=immutable_revision)
    return CacheObjectRequirement(
        relative_path=pack_name,
        size_bytes=len(header) + sum(tensor.length for tensor in ordered),
        is_weight_pack=True,
    )


def materialize_tensor_span(
    *,
    repo_id: str,
    immutable_revision: str,
    metadata_root: Path,
    artifact_index: ModelArtifactIndex,
    start_layer: int,
    end_layer: int,
    local_files_only: bool = False,
    token: bool | str | None = None,
    cache_root: Path | None = None,
) -> Path:
    """Return a reusable model projection containing only ``[start_layer, end_layer)``."""

    if artifact_index.model_id != repo_id:
        raise ValueError("artifact index and requested model identity disagree")
    if artifact_index.immutable_revision != immutable_revision:
        raise ValueError("artifact index and requested immutable revision disagree")
    if not artifact_index.tensors:
        raise ValueError("artifact index has no signed selective tensor metadata")
    if start_layer < 0 or end_layer <= start_layer:
        raise ValueError("invalid layer span")

    config_descriptor = next(
        (
            descriptor
            for descriptor in artifact_index.artifacts
            if descriptor.path == "config.json" and descriptor.role is ArtifactRole.ARCHITECTURE
        ),
        None,
    )
    if config_descriptor is None:
        raise ValueError("signed artifact index has no architecture config.json")
    config_path = metadata_root / "config.json"
    _verify_descriptor(config_path, config_descriptor)
    config = json.loads(config_path.read_bytes())
    num_layers = int(
        config.get("num_hidden_layers") or config.get("num_layers") or config.get("n_layer") or 0
    )
    if num_layers <= 0 or end_layer > num_layers:
        raise ValueError("layer span exceeds model configuration")
    selected = selected_tensor_artifacts(
        artifact_index,
        start_layer=start_layer,
        end_layer=end_layer,
        num_layers=num_layers,
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
    )

    identity = artifact_index_identity(artifact_index)

    source_descriptors = {
        descriptor.path: descriptor
        for descriptor in artifact_index.artifacts
        if descriptor.role is ArtifactRole.WEIGHT and descriptor.xet_file_hash is not None
    }
    pack_tensors: dict[str, list[TensorArtifactDescriptor]] = defaultdict(list)
    for tensor in selected:
        pack_name = pack_name_for_tensor(tensor.name)
        assert pack_name is not None
        if tensor.source_path not in source_descriptors:
            raise ValueError(f"tensor source has no signed Xet descriptor: {tensor.source_path!r}")
        pack_tensors[pack_name].append(tensor)

    generated_index = _generated_weight_index(artifact_index)
    requirements = [
        CacheObjectRequirement(relative_path=descriptor.path, size_bytes=descriptor.size)
        for descriptor in artifact_index.artifacts
        if descriptor.role is not ArtifactRole.WEIGHT
    ]
    requirements.append(
        CacheObjectRequirement(
            relative_path=_SOURCE_INDEX_NAME,
            size_bytes=len(generated_index),
        )
    )
    requirements.extend(
        _pack_requirement(
            pack_name,
            tuple(values),
            immutable_revision=immutable_revision,
        )
        for pack_name, values in sorted(pack_tensors.items())
    )

    storage_pool = (
        ModelArtifactCachePool((ModelArtifactCache(cache_root),))
        if cache_root is not None
        else ModelArtifactCachePool.configured()
    )
    storage, reservation, storage_snapshot = storage_pool.reserve(
        artifact_identity=identity,
        model_id=repo_id,
        immutable_revision=immutable_revision,
        required_objects=requirements,
    )
    projection_root = storage.cache_root / identity
    projection_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Reserved %d cache bytes for %s at %s (%d bytes reclaimed, %d bytes free)",
        storage_snapshot.required_growth_bytes,
        repo_id,
        projection_root,
        storage_snapshot.reclaimed_bytes,
        storage_snapshot.free_bytes,
    )
    try:
        lock = FileLock(str(projection_root.with_suffix(".lock")))
        with lock:
            _copy_runtime_artifacts(metadata_root, projection_root, artifact_index)
            generated_index_path = projection_root / _SOURCE_INDEX_NAME
            try:
                index_is_current = generated_index_path.read_bytes() == generated_index
            except OSError:
                index_is_current = False
            if not index_is_current:
                generated_index_path.unlink(missing_ok=True)
                _atomic_write(generated_index_path, generated_index)
            receipt_path = projection_root / _RECEIPT_NAME
            receipt = _load_receipt(receipt_path, expected_identity=identity)
            receipt.update(
                {
                    "model_id": repo_id,
                    "immutable_revision": immutable_revision,
                }
            )
            packs = receipt["packs"]
            assert isinstance(packs, dict)
            range_sources = _RangeSources(
                repo_id=repo_id,
                immutable_revision=immutable_revision,
                metadata_root=metadata_root,
                sources=source_descriptors,
                local_files_only=local_files_only,
                token=token,
            )
            for pack_name, values in sorted(pack_tensors.items()):
                tensors = tuple(values)
                spec_hash = _pack_spec_hash(
                    sorted(tensors, key=lambda item: (item.source_path, item.offset, item.name))
                )
                pack_path = projection_root / pack_name
                if _valid_cached_pack(pack_path, packs.get(pack_name), spec_hash=spec_hash):
                    try:
                        _verify_pack_contents(pack_path, tensors)
                        continue
                    except ValueError:
                        pass
                if (
                    local_files_only
                    and not pack_path.exists()
                    and not all(
                        (metadata_root / tensor.source_path).is_file() for tensor in tensors
                    )
                ):
                    raise FileNotFoundError(
                        f"selective pack is not available offline: {pack_name}"
                    )
                # A corrupt or truncated pack is not useful cache content.
                # Removing it first preserves the exact net-growth reservation.
                pack_path.unlink(missing_ok=True)
                entry = _materialize_pack(
                    pack_path,
                    tensors,
                    immutable_revision=immutable_revision,
                    sources=range_sources,
                )
                packs[pack_name] = entry
                logger.info(
                    "Materialized %s (%d tensor bytes: %d network, %d verified local)",
                    pack_name,
                    entry["source_bytes"],
                    entry["network_bytes"],
                    entry["local_bytes"],
                )
                _atomic_write(
                    receipt_path,
                    json.dumps(
                        receipt,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8"),
                )
            if not receipt_path.exists():
                _atomic_write(
                    receipt_path,
                    json.dumps(
                        receipt,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8"),
                )
        storage.commit(reservation)
    except Exception:
        storage.abort(reservation)
        raise
    return projection_root


def selective_receipt_path(model_root: Path) -> Path:
    return model_root / _RECEIPT_NAME


def verify_selective_projection(
    *,
    model_root: Path,
    artifact_index: ModelArtifactIndex,
    start_layer: int,
    end_layer: int,
    num_layers: int,
    tie_word_embeddings: bool,
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Fail closed unless every required projected tensor matches the signed registry."""

    identity = artifact_index_identity(artifact_index)
    receipt_path = selective_receipt_path(model_root)
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("selective artifact receipt is missing or invalid") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("version") != 1
        or receipt.get("artifact_index_sha256") != identity
        or receipt.get("model_id") != artifact_index.model_id
        or receipt.get("immutable_revision") != artifact_index.immutable_revision
        or not isinstance(receipt.get("packs"), dict)
    ):
        raise ValueError("selective artifact receipt disagrees with the signed model index")
    generated_index_path = model_root / _SOURCE_INDEX_NAME
    if generated_index_path.read_bytes() != _generated_weight_index(artifact_index):
        raise ValueError("generated SafeTensors weight index is missing or invalid")

    selected = selected_tensor_artifacts(
        artifact_index,
        start_layer=start_layer,
        end_layer=end_layer,
        num_layers=num_layers,
        tie_word_embeddings=tie_word_embeddings,
    )
    grouped: dict[str, list[TensorArtifactDescriptor]] = defaultdict(list)
    for tensor in selected:
        pack_name = pack_name_for_tensor(tensor.name)
        assert pack_name is not None
        grouped[pack_name].append(tensor)

    packs = receipt["packs"]
    assert isinstance(packs, dict)
    paths: list[Path] = [generated_index_path]
    hashes: list[str] = [_sha256_file(generated_index_path)]
    for pack_name, values in sorted(grouped.items()):
        tensors = tuple(values)
        ordered = tuple(
            sorted(tensors, key=lambda item: (item.source_path, item.offset, item.name))
        )
        entry = packs.get(pack_name)
        path = model_root / pack_name
        if not _valid_cached_pack(path, entry, spec_hash=_pack_spec_hash(ordered)):
            raise ValueError(f"selective SafeTensors pack receipt is invalid: {pack_name}")
        _verify_pack_contents(path, ordered)
        assert isinstance(entry, dict) and isinstance(entry.get("sha256"), str)
        paths.append(path)
        hashes.append(entry["sha256"])
    return tuple(paths), tuple(hashes)
