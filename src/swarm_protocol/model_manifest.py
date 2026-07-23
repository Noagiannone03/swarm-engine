"""Build immutable Fabi model manifests from content-addressed Hub artifacts.

The Hub commit identifies a repository snapshot, while each artifact descriptor identifies the
actual bytes used by a runtime.  Large LFS files use the SHA-256 published by Hugging Face.  Small
Git-tracked runtime files are downloaded at the resolved commit and hashed locally; a Git SHA-1 is
never relabelled as a SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from parallax.utils.model_config import normalize_model_config
from swarm_protocol.contracts import (
    ArtifactDescriptor,
    ArtifactRole,
    ModelArtifactIndex,
    ModelManifest,
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_HASHED_GIT_FILE_BYTES = 64 * 1024 * 1024

ArtifactReader = Callable[[str, str, str, bool | str | None], bytes]

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

    descriptors = [
        artifact.model_dump(mode="json") for artifact in index.artifacts if artifact.role is role
    ]
    if not descriptors:
        raise ValueError(f"model artifact index has no {role.value} artifacts")
    return _canonical_hash(f"fabi/model-artifacts/{role.value}/v1", descriptors)


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


def _runtime_contract(config: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: config[key] for key in keys if key in config and config[key] is not None}


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

    index = ModelArtifactIndex(
        model_id=canonical_model_id,
        immutable_revision=immutable_revision,
        artifacts=tuple(sorted(descriptors, key=lambda descriptor: descriptor.path)),
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
        from backend.server.model_weight_metadata import load_hub_weight_profile

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
