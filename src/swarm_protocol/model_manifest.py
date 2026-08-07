"""Build immutable Fabi model manifests from content-addressed Hub artifacts.

The Hub commit identifies a repository snapshot, while each artifact descriptor identifies the
actual bytes used by a runtime.  Large LFS files use the SHA-256 published by Hugging Face.  Small
Git-tracked runtime files are downloaded at the resolved commit and hashed locally; a Git SHA-1 is
never relabelled as a SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers

from parallax.utils.model_config import get_model_context_limit, normalize_model_config
from swarm_protocol.contracts import (
    ArtifactDescriptor,
    ArtifactRole,
    ModelArtifactIndex,
    ModelManifest,
    TensorArtifactDescriptor,
)
from swarm_protocol.xet_transport import (
    get_hf_file_metadata_with_backoff,
    get_xet_session,
    xet_headers_without_auth,
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_HASHED_GIT_FILE_BYTES = 64 * 1024 * 1024
_HASH_PROGRESS_BYTES = 512 * 1024 * 1024

logger = logging.getLogger(__name__)

ArtifactReader = Callable[[str, str, str, bool | str | None], bytes]
_HUB_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
_HUB_METADATA_RETRIES = 5

_TOKENIZER_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_TOKENIZER_PREFIXES = (
    "chat_template",
    "preprocessor",
    "processor",
    "tokenization_",
    "tokenizer",
    "vocab",
)
_WEIGHT_SUFFIXES = {
    "pytorch_bin": (".bin",),
    "safetensors": (".safetensors",),
}
_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "float32": "float32",
    "fp32": "float32",
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
}
_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}
_QUANTIZATION_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "none": "unquantized",
    "unquantized": "unquantized",
}


@dataclass(frozen=True)
class ResolvedModelBundle:
    """Compact routing manifest plus its persistent content-addressed artifact index."""

    manifest: ModelManifest
    artifact_index: ModelArtifactIndex


def _canonical_hash(domain: str, payload: object) -> str:
    encoded = json.dumps(
        {"domain": domain, "payload": payload},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_collection_hash(index: ModelArtifactIndex, role: ArtifactRole) -> str:
    """Hash one ordered artifact collection with an explicit domain separator."""

    # ``exclude_none`` preserves the v1 identity of bundles published before
    # optional transport identities were added. Existing swarms must not fork
    # merely because a newer parser supplies ``xet_file_hash=None``.
    descriptors = [
        artifact.model_dump(mode="json", exclude_none=True)
        for artifact in index.artifacts
        if artifact.role is role
    ]
    if not descriptors:
        raise ValueError(f"model artifact index has no {role.value} artifacts")
    return _canonical_hash(f"fabi/model-artifacts/{role.value}/v1", descriptors)


def execution_plan_hash(index: ModelArtifactIndex) -> str:
    """Bind portable execution topology and every referenced byte descriptor."""

    if not index.execution_plans:
        raise ValueError("model artifact index has no execution plans")
    portable_roles = {
        ArtifactRole.EXECUTION_GRAPH,
        ArtifactRole.EXECUTION_DATA,
        ArtifactRole.EXECUTION_PACKAGE_MANIFEST,
        ArtifactRole.EXECUTION_LAYER,
        ArtifactRole.EXECUTION_SHARED,
    }
    payload = {
        "plans": [plan.model_dump(mode="json") for plan in index.execution_plans],
        "artifacts": [
            artifact.model_dump(mode="json", exclude_none=True)
            for artifact in index.artifacts
            if artifact.role in portable_roles
        ],
    }
    return _canonical_hash("fabi/model-execution-plans/v1", payload)


def _media_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".safetensors"):
        return "application/vnd.safetensors"
    if lower.endswith(".json"):
        return "application/json"
    if lower.endswith(".py"):
        return "text/x-python"
    if lower.endswith((".model", ".bin")):
        return "application/octet-stream"
    if lower.endswith((".jinja", ".txt")):
        return "text/plain"
    return "application/octet-stream"


def _artifact_role(path: str, weight_format: str) -> ArtifactRole | None:
    lower = path.lower()
    basename = PurePosixPath(lower).name
    suffixes = _WEIGHT_SUFFIXES.get(weight_format)
    if suffixes is None:
        raise ValueError(f"unsupported weight format: {weight_format}")
    if lower.endswith(suffixes) or any(
        lower.endswith(f"{suffix}.index.json") for suffix in suffixes
    ):
        return ArtifactRole.WEIGHT
    if basename in _TOKENIZER_NAMES or basename.startswith(_TOKENIZER_PREFIXES):
        return ArtifactRole.TOKENIZER
    # Parallax downloads non-weight metadata and permits trusted remote model/tokenizer code.
    # Hash every executable or structured runtime file conservatively. Documentation and model
    # cards are deliberately excluded so editorial changes do not fork a model swarm.
    if lower.endswith((".json", ".py", ".jinja")):
        return ArtifactRole.ARCHITECTURE
    return None


def _validate_repo_path(path: str) -> None:
    parts = path.split("/")
    if path.startswith("/") or "\\" in path or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Hub returned an unsafe repository path: {path!r}")


def _field(value: object, name: str) -> object | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _default_reader(
    repo_id: str,
    filename: str,
    revision: str,
    token: bool | str | None,
) -> bytes:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )
    return Path(path).read_bytes()


def _attach_selective_safetensors_metadata(
    *,
    client: HfApi,
    repo_id: str,
    immutable_revision: str,
    token: bool | str | None,
    descriptors: list[ArtifactDescriptor],
) -> tuple[list[ArtifactDescriptor], tuple[TensorArtifactDescriptor, ...]]:
    """Bind official Xet identities and exact tensor ranges into the signed index.

    SafeTensors offsets are relative to the tensor-data section.  The Hub file
    metadata gives the immutable full file size, so the absolute range begins
    at ``file_size - max_tensor_end``.  We deliberately sign the Xet file hash
    as well as the legacy LFS SHA-256: Xet can then authenticate a reconstructed
    sub-range without forcing each worker to download the rest of the file.
    The publishing authority streams each immutable data section once and
    signs every tensor SHA-256, allowing workers to detect local pack damage
    later without another network read.
    """

    metadata = _get_safetensors_metadata_with_backoff(
        client=client,
        repo_id=repo_id,
        immutable_revision=immutable_revision,
        token=token,
    )
    files_metadata = _field(metadata, "files_metadata")
    if not isinstance(files_metadata, Mapping) or not files_metadata:
        raise ValueError("Hub returned no SafeTensors file metadata")

    descriptor_by_path = {descriptor.path: descriptor for descriptor in descriptors}
    tensor_descriptors: list[TensorArtifactDescriptor] = []

    for source_path, file_metadata in sorted(files_metadata.items()):
        source_path = str(source_path)
        source = descriptor_by_path.get(source_path)
        if source is None or source.role is not ArtifactRole.WEIGHT:
            raise ValueError(
                f"SafeTensors metadata references an unsigned weight file: {source_path!r}"
            )
        headers = build_hf_headers(token=token)
        remote = get_hf_file_metadata_with_backoff(
            hf_hub_url(repo_id, source_path, revision=immutable_revision),
            token=token,
            headers=headers,
        )
        if remote.size != source.size:
            raise ValueError(f"Xet metadata size mismatch for {source_path!r}")
        if str(remote.etag or "").strip('"').lower() != source.sha256:
            raise ValueError(f"Xet metadata digest mismatch for {source_path!r}")
        xet = remote.xet_file_data
        xet_file_hash = str(_field(xet, "file_hash") or "").lower()
        if not _SHA256_RE.fullmatch(xet_file_hash):
            raise ValueError(f"Hub returned no valid Xet file hash for {source_path!r}")
        descriptor_by_path[source_path] = source.model_copy(update={"xet_file_hash": xet_file_hash})

        tensors = _field(file_metadata, "tensors")
        if not isinstance(tensors, Mapping) or not tensors:
            raise ValueError(f"SafeTensors file has no tensor metadata: {source_path!r}")
        relative_ranges: list[tuple[str, int, int, object]] = []
        for raw_name, tensor in tensors.items():
            name = str(raw_name)
            offsets = _field(tensor, "data_offsets")
            if offsets is None or len(offsets) != 2:
                raise ValueError(f"tensor has invalid offsets: {name!r}")
            start, end = int(offsets[0]), int(offsets[1])
            if start < 0 or end < start:
                raise ValueError(f"tensor has invalid byte range: {name!r}")
            relative_ranges.append((name, start, end, tensor))

        relative_ranges.sort(key=lambda item: (item[1], item[2], item[0]))
        previous_end = 0
        for name, start, end, _ in relative_ranges:
            if start != previous_end:
                raise ValueError(
                    f"SafeTensors data section is not contiguous before tensor {name!r}"
                )
            previous_end = end
        data_section_offset = source.size - previous_end
        if data_section_offset < 8:
            raise ValueError(f"SafeTensors header is invalid for {source_path!r}")

        tensor_hashes = _hash_xet_tensor_ranges(
            source=source,
            xet_file_hash=xet_file_hash,
            refresh_route=str(_field(xet, "refresh_route") or ""),
            headers=headers,
            data_section_offset=data_section_offset,
            relative_ranges=relative_ranges,
        )

        for name, start, end, tensor in relative_ranges:
            dtype = str(_field(tensor, "dtype") or "")
            shape_value = _field(tensor, "shape")
            if not dtype or shape_value is None:
                raise ValueError(f"tensor has incomplete type metadata: {name!r}")
            shape = tuple(int(dimension) for dimension in shape_value)
            if any(dimension < 0 for dimension in shape):
                raise ValueError(f"tensor has a negative dimension: {name!r}")
            tensor_descriptors.append(
                TensorArtifactDescriptor(
                    name=name,
                    source_path=source_path,
                    offset=data_section_offset + start,
                    length=end - start,
                    sha256=tensor_hashes[name],
                    dtype=dtype,
                    shape=shape,
                )
            )

    updated_descriptors = [descriptor_by_path[descriptor.path] for descriptor in descriptors]
    return updated_descriptors, tuple(
        sorted(tensor_descriptors, key=lambda descriptor: descriptor.name)
    )


def _get_safetensors_metadata_with_backoff(
    *,
    client: HfApi,
    repo_id: str,
    immutable_revision: str,
    token: bool | str | None,
) -> object:
    """Retry the official parallel header parser on transient transport failures.

    Hugging Face's maintained parser fetches a small range from each shard in a
    thread pool, but unlike ``get_hf_file_metadata(retry_on_errors=True)`` it
    currently exposes no retry option.  Keep its parser and exception semantics;
    only retry the same network exception family used by Hub ``http_backoff``.
    """

    for attempt in range(_HUB_METADATA_RETRIES + 1):
        try:
            return client.get_safetensors_metadata(
                repo_id,
                revision=immutable_revision,
                token=token,
            )
        except _HUB_NETWORK_ERRORS as exc:
            if attempt == _HUB_METADATA_RETRIES:
                raise
            delay = min(2**attempt, 8)
            logger.warning(
                "SafeTensors header fetch failed for %s (%s); retrying in %ds",
                repo_id,
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable SafeTensors metadata retry state")


def _hash_xet_tensor_ranges(
    *,
    source: ArtifactDescriptor,
    xet_file_hash: str,
    refresh_route: str,
    headers: dict[str, str],
    data_section_offset: int,
    relative_ranges: list[tuple[str, int, int, object]],
) -> dict[str, str]:
    """Hash one contiguous SafeTensors data section from one verified Xet stream."""

    if not refresh_route:
        raise ValueError(f"Hub returned no Xet refresh route for {source.path!r}")
    from hf_xet import XetFileInfo

    group = get_xet_session().new_download_stream_group(
        token_refresh_url=refresh_route,
        token_refresh_headers=headers,
        custom_headers=xet_headers_without_auth(headers),
    )
    stream = group.download_stream(
        XetFileInfo(xet_file_hash, source.size),
        start=data_section_offset,
        end=source.size,
    )
    digests = [hashlib.sha256() for _ in relative_ranges]
    tensor_index = 0
    tensor_bytes = 0
    total_bytes = 0
    expected_bytes = source.size - data_section_offset
    next_progress = _HASH_PROGRESS_BYTES
    logger.info("Hashing signed tensor ranges from %s (%d bytes)", source.path, expected_bytes)

    for chunk in stream:
        view = memoryview(chunk)
        total_bytes += len(view)
        if total_bytes >= next_progress:
            logger.info(
                "Hashed %d/%d bytes from %s",
                total_bytes,
                expected_bytes,
                source.path,
            )
            next_progress = ((total_bytes // _HASH_PROGRESS_BYTES) + 1) * _HASH_PROGRESS_BYTES
        while view:
            while (
                tensor_index < len(relative_ranges)
                and relative_ranges[tensor_index][2] == relative_ranges[tensor_index][1]
            ):
                tensor_index += 1
                tensor_bytes = 0
            if tensor_index >= len(relative_ranges):
                raise OSError(f"Xet returned excess bytes for {source.path!r}")
            _, start, end, _ = relative_ranges[tensor_index]
            remaining = end - start - tensor_bytes
            consumed = min(len(view), remaining)
            digests[tensor_index].update(view[:consumed])
            view = view[consumed:]
            tensor_bytes += consumed
            if tensor_bytes == end - start:
                tensor_index += 1
                tensor_bytes = 0

    while (
        tensor_index < len(relative_ranges)
        and relative_ranges[tensor_index][2] == relative_ranges[tensor_index][1]
    ):
        tensor_index += 1
    if total_bytes != expected_bytes or tensor_index != len(relative_ranges) or tensor_bytes:
        raise OSError(
            f"Xet reconstructed an incomplete SafeTensors data section for {source.path!r}: "
            f"expected {expected_bytes} bytes, got {total_bytes}"
        )
    logger.info("Hashed all %d tensor bytes from %s", total_bytes, source.path)
    return {relative_ranges[index][0]: digest.hexdigest() for index, digest in enumerate(digests)}


def _runtime_contract(config: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: config[key] for key in keys if key in config and config[key] is not None}


def context_classes_for(
    model_max_context_tokens: int,
    *,
    minimum_class_tokens: int = 4_096,
) -> tuple[int, ...]:
    """Build a compact deterministic context ladder ending at the exact limit.

    Powers of two keep distributed demand summaries interoperable while the
    final non-power-of-two model boundary (for example Qwen3's 40,960 tokens)
    is never rounded up or silently discarded.
    """

    if model_max_context_tokens <= 0 or minimum_class_tokens <= 0:
        raise ValueError("model context limit and minimum class must be positive")
    current = min(model_max_context_tokens, minimum_class_tokens)
    classes: list[int] = []
    while current < model_max_context_tokens:
        classes.append(current)
        current = min(model_max_context_tokens, current * 2)
    classes.append(model_max_context_tokens)
    return tuple(classes)


def _kv_bytes_per_token_by_layer(
    config: Mapping[str, Any],
    *,
    num_layers: int,
    hidden_size: int,
    dtype_bytes: int,
) -> tuple[int, ...]:
    """Derive exact uniform attention-cache geometry from the model contract.

    Current Parallax executors expose a decoder-layer KV cache. Architectures with recurrent or
    state-space layers need a different cache contract and are rejected until the runtime can
    measure and advertise their per-layer state explicitly.
    """

    attention_heads = int(config.get("num_attention_heads") or 0)
    kv_heads = int(config.get("num_key_value_heads") or attention_heads)
    if attention_heads <= 0 or kv_heads <= 0 or hidden_size % attention_heads != 0:
        raise ValueError("model config does not expose a valid attention/KV head geometry")
    default_head_dim = hidden_size // attention_heads
    key_head_dim = int(
        (config.get("qk_nope_head_dim") or 0) + (config.get("qk_rope_head_dim") or 0)
        or config.get("head_dim")
        or default_head_dim
    )
    value_head_dim = int(config.get("v_head_dim") or config.get("head_dim") or default_head_dim)
    if key_head_dim <= 0 or value_head_dim <= 0:
        raise ValueError("model config declares invalid KV head dimensions")

    layer_types = config.get("layer_types")
    if layer_types is not None:
        if not isinstance(layer_types, list) or len(layer_types) != num_layers:
            raise ValueError("model layer_types must contain exactly one entry per layer")
        unsupported = [
            layer_type
            for layer_type in layer_types
            if not isinstance(layer_type, str)
            or not any(token in layer_type.lower() for token in ("attention", "sliding"))
        ]
        if unsupported:
            raise ValueError(
                "state-space or recurrent layer cache geometry is not yet supported by protocol v3"
            )

    per_layer = dtype_bytes * kv_heads * (key_head_dim + value_head_dim)
    return (per_layer,) * num_layers


def _quantization_identity(label: str, config: Mapping[str, Any]) -> str:
    normalized_label = _QUANTIZATION_ALIASES.get(label.strip().lower(), label.strip().lower())
    quantization_config = config.get("quantization_config") or config.get("quantization")
    if quantization_config is None and config.get("quant_method") is not None:
        quantization_config = {"quant_method": config["quant_method"]}
    if quantization_config is None:
        return normalized_label
    if not isinstance(quantization_config, Mapping):
        raise ValueError("model quantization contract must be a JSON object")
    fingerprint = _canonical_hash("fabi/model-contract/quantization/v1", dict(quantization_config))
    return f"{normalized_label}@sha256:{fingerprint}"


def build_hub_model_bundle(
    repo_id: str,
    *,
    revision: str | None = None,
    weight_format: str = "safetensors",
    quantization: str,
    dtype: str,
    wire_protocol_version: int = 1,
    token: bool | str | None = None,
    api: HfApi | None = None,
    artifact_reader: ArtifactReader | None = None,
    include_weight_profile: bool = False,
    include_selective_weight_index: bool = False,
) -> ResolvedModelBundle:
    """Resolve a mutable Hub reference into one reproducible Fabi model bundle.

    The call fails closed if the Hub does not return complete file metadata, if an oversized
    Git-tracked runtime file would need to be trusted without a SHA-256, or if the model lacks one
    of the architecture/tokenizer/weight collections required by a pipeline.
    """

    if not repo_id or not quantization or not dtype:
        raise ValueError("repo_id, quantization and dtype must be non-empty")
    if wire_protocol_version <= 0:
        raise ValueError("wire_protocol_version must be positive")
    dtype_key = _DTYPE_ALIASES.get(dtype.lower())
    if dtype_key is None:
        raise ValueError(f"unsupported activation dtype: {dtype}")

    client = api or HfApi()
    resolved_info = client.model_info(repo_id, revision=revision, token=token)
    immutable_revision = str(_field(resolved_info, "sha") or "").lower()
    if not _COMMIT_RE.fullmatch(immutable_revision):
        raise ValueError(f"Hub did not resolve {repo_id!r} to a full immutable commit")

    # Fetch file metadata against the immutable commit, never against the moving input ref.
    info = client.model_info(
        repo_id,
        revision=immutable_revision,
        files_metadata=True,
        token=token,
    )
    if str(_field(info, "sha") or "").lower() != immutable_revision:
        raise ValueError("Hub file metadata does not match the resolved immutable revision")
    canonical_model_id = str(_field(info, "id") or _field(resolved_info, "id") or repo_id)
    siblings = _field(info, "siblings")
    if not siblings:
        raise ValueError(f"Hub returned no file metadata for {repo_id}")

    reader = artifact_reader or _default_reader
    descriptors: list[ArtifactDescriptor] = []
    config_bytes: bytes | None = None

    for sibling in siblings:
        path = str(_field(sibling, "rfilename") or "")
        _validate_repo_path(path)
        role = _artifact_role(path, weight_format)
        if role is None:
            continue
        size_value = _field(sibling, "size")
        if size_value is None or int(size_value) < 0:
            raise ValueError(f"Hub file metadata has no valid size for {path!r}")
        size = int(size_value)

        lfs = _field(sibling, "lfs")
        lfs_sha256 = str(_field(lfs, "sha256") or "").lower()
        lfs_size = _field(lfs, "size")
        if lfs is not None and (lfs_size is None or int(lfs_size) != size):
            raise ValueError(f"Hub LFS size does not match file metadata for {path!r}")
        must_read = path == "config.json" or not lfs_sha256
        content: bytes | None = None
        if must_read:
            if size > _MAX_HASHED_GIT_FILE_BYTES:
                raise ValueError(
                    f"Git-tracked runtime file {path!r} is too large to hash safely; "
                    "publish it with LFS SHA-256 metadata"
                )
            content = reader(repo_id, path, immutable_revision, token)
            if len(content) != size:
                raise ValueError(
                    f"downloaded size for {path!r} is {len(content)}, Hub declared {size}"
                )
            digest = hashlib.sha256(content).hexdigest()
            if lfs_sha256 and digest != lfs_sha256:
                raise ValueError(f"downloaded bytes for {path!r} do not match Hub LFS SHA-256")
        else:
            if not _SHA256_RE.fullmatch(lfs_sha256):
                raise ValueError(f"Hub returned an invalid LFS SHA-256 for {path!r}")
            digest = lfs_sha256

        if path == "config.json":
            config_bytes = content
        descriptors.append(
            ArtifactDescriptor(
                path=path,
                size=size,
                sha256=digest,
                media_type=_media_type(path),
                role=role,
            )
        )

    if config_bytes is None:
        raise ValueError("model repository has no readable root config.json")
    try:
        raw_config = json.loads(config_bytes)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("model config.json is not valid JSON") from exc
    if not isinstance(raw_config, dict):
        raise ValueError("model config.json must contain an object")
    config = normalize_model_config(raw_config)

    # CUDA executors currently resolve their model dtype from config.json.  Reject a manifest
    # label that those maintained loaders would ignore instead of advertising a false common
    # activation contract. MLX workers may still explicitly choose the same resolved dtype.
    configured_dtype = config.get("torch_dtype") or config.get("dtype")
    if configured_dtype is not None:
        configured_dtype_key = _DTYPE_ALIASES.get(str(configured_dtype).lower())
        if configured_dtype_key is None:
            raise ValueError(f"model config declares unsupported dtype: {configured_dtype}")
        if configured_dtype_key != dtype_key:
            raise ValueError(
                f"requested dtype {dtype_key} does not match model config dtype "
                f"{configured_dtype_key}"
            )

    tensor_descriptors: tuple[TensorArtifactDescriptor, ...] = ()
    if include_selective_weight_index:
        if weight_format != "safetensors":
            raise ValueError("selective tensor artifacts require SafeTensors weights")
        descriptors, tensor_descriptors = _attach_selective_safetensors_metadata(
            client=client,
            repo_id=canonical_model_id,
            immutable_revision=immutable_revision,
            token=token,
            descriptors=descriptors,
        )

    index = ModelArtifactIndex(
        model_id=canonical_model_id,
        immutable_revision=immutable_revision,
        artifacts=tuple(sorted(descriptors, key=lambda descriptor: descriptor.path)),
        tensors=tensor_descriptors,
    )
    architecture_hash = artifact_collection_hash(index, ArtifactRole.ARCHITECTURE)
    tokenizer_hash = artifact_collection_hash(index, ArtifactRole.TOKENIZER)
    weight_hash = artifact_collection_hash(index, ArtifactRole.WEIGHT)

    num_layers = int(
        config.get("num_hidden_layers") or config.get("num_layers") or config.get("n_layer") or 0
    )
    hidden_size = int(config.get("hidden_size") or config.get("d_model") or config.get("dim") or 0)
    if num_layers <= 0 or hidden_size <= 0:
        raise ValueError("model config does not expose positive layer count and hidden size")
    model_max_context_tokens = get_model_context_limit(config)
    if model_max_context_tokens is None:
        raise ValueError("model config does not expose a finite context limit")

    rope_contract = _runtime_contract(
        config,
        (
            "model_type",
            "max_position_embeddings",
            "model_max_length",
            "rope_theta",
            "rope_scaling",
            "partial_rotary_factor",
            "rotary_pct",
            "sliding_window",
            "use_sliding_window",
            "max_window_layers",
            "layer_types",
        ),
    )
    attention_contract = _runtime_contract(
        config,
        (
            "model_type",
            "hidden_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "num_attention_groups",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "kv_lora_rank",
            "layer_types",
        ),
    )
    attention_contract.update({"activation_dtype": dtype_key, "cache_dtype": dtype_key})
    prefill_contract = {
        "activation_dtype": dtype_key,
        "hidden_size": hidden_size,
        "position_contract": "input_length_plus_output_length/v1",
        "chunking_contract": "contiguous_prefix_suffix/v1",
        "tensor_encoding": "safetensors/v1",
        "wire_protocol_version": wire_protocol_version,
    }
    weight_profile = None
    if include_weight_profile:
        from backend.server.model_weight_metadata import (
            build_weight_profile_from_tensors,
            load_hub_weight_profile,
        )

        if tensor_descriptors:
            weight_profile = build_weight_profile_from_tensors(
                {tensor.name: tensor for tensor in tensor_descriptors},
                num_layers=num_layers,
                tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
                source_revision=immutable_revision,
            )
        else:
            weight_profile = load_hub_weight_profile(
                canonical_model_id,
                num_layers=num_layers,
                tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
                revision=immutable_revision,
                token=token,
                api=client,
            )

    manifest = ModelManifest(
        model_id=canonical_model_id,
        immutable_revision=immutable_revision,
        architecture_graph_hash=architecture_hash,
        tokenizer_hash=tokenizer_hash,
        weight_collection_hash=weight_hash,
        weight_format=weight_format,
        quantization=_quantization_identity(quantization, config),
        dtype=dtype_key,
        num_layers=num_layers,
        model_max_context_tokens=model_max_context_tokens,
        context_classes=context_classes_for(model_max_context_tokens),
        activation_bytes_per_token=hidden_size * _DTYPE_BYTES[dtype_key],
        kv_bytes_per_token_by_layer=_kv_bytes_per_token_by_layer(
            config,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dtype_bytes=_DTYPE_BYTES[dtype_key],
        ),
        weight_bytes_by_layer=(() if weight_profile is None else weight_profile.layer_bytes),
        input_endpoint_weight_bytes=(
            0 if weight_profile is None else weight_profile.input_endpoint_bytes
        ),
        output_endpoint_weight_bytes=(
            0 if weight_profile is None else weight_profile.output_endpoint_bytes
        ),
        shared_endpoint_weight_bytes=(
            0 if weight_profile is None else weight_profile.shared_endpoint_bytes
        ),
        rope_context_contract_hash=_canonical_hash(
            "fabi/model-contract/rope-context/v1", rope_contract
        ),
        attention_kv_contract_hash=_canonical_hash(
            "fabi/model-contract/attention-kv/v1", attention_contract
        ),
        prefill_contract_hash=_canonical_hash("fabi/model-contract/prefill/v1", prefill_contract),
        wire_protocol_version=wire_protocol_version,
    )
    return ResolvedModelBundle(manifest=manifest, artifact_index=index)
