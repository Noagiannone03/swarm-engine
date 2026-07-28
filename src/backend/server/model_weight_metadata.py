"""Exact model weight accounting from maintained safetensors metadata APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from huggingface_hub import HfApi

from parallax.utils.weight_filter_utils import normalize_language_model_weight_key
from scheduling.model_info import ModelWeightProfile


def resolve_hub_revision(repo_id: str, *, api: HfApi | None = None) -> str:
    """Resolve a moving Hub ref to one immutable commit hash."""

    if Path(repo_id).exists():
        raise ValueError("A local model path has no Hub revision")
    revision = (api or HfApi()).model_info(repo_id).sha
    if not revision:
        raise ValueError(f"Could not resolve an immutable revision for {repo_id}")
    return revision


def _tensor_bytes(tensor_info: object) -> int:
    signed_length = getattr(tensor_info, "length", None)
    if signed_length is not None:
        length = int(signed_length)
        if length < 0:
            raise ValueError(f"Invalid signed tensor length: {signed_length!r}")
        return length
    offsets = getattr(tensor_info, "data_offsets", None)
    if offsets is None or len(offsets) != 2:
        raise ValueError("Safetensors metadata is missing data offsets")
    start, end = (int(offsets[0]), int(offsets[1]))
    if start < 0 or end < start:
        raise ValueError(f"Invalid safetensors data offsets: {offsets!r}")
    return end - start


def build_weight_profile_from_tensors(
    tensors: Mapping[str, object],
    *,
    num_layers: int,
    tie_word_embeddings: bool,
    source_revision: str | None = None,
) -> ModelWeightProfile:
    """Classify exact tensor byte ranges using the runtime shard-loading contract."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    layer_bytes = [0] * num_layers
    input_tensors: dict[str, int] = {}
    output_tensors: dict[str, int] = {}

    for original_key, tensor_info in tensors.items():
        key = normalize_language_model_weight_key(original_key)
        size = _tensor_bytes(tensor_info)

        parts = key.split(".")
        if "layers" in parts:
            index_position = parts.index("layers") + 1
            if index_position >= len(parts) or not parts[index_position].isdigit():
                raise ValueError(f"Cannot identify decoder layer for tensor {original_key!r}")
            layer_index = int(parts[index_position])
            if not 0 <= layer_index < num_layers:
                raise ValueError(
                    f"Tensor {original_key!r} targets layer {layer_index}, "
                    f"outside [0, {num_layers})"
                )
            layer_bytes[layer_index] += size
            continue

        if key.startswith("model.") and "embed_tokens" in key:
            input_tensors[original_key] = size
            if tie_word_embeddings:
                output_tensors[original_key] = size
            continue

        if "model.norm" in key or "lm_head" in key:
            output_tensors[original_key] = size

    missing_layers = [index for index, size in enumerate(layer_bytes) if size <= 0]
    if missing_layers:
        raise ValueError(f"Safetensors metadata has no weights for decoder layers {missing_layers}")
    if not input_tensors:
        raise ValueError("Safetensors metadata has no input embedding weights")
    if not output_tensors:
        raise ValueError("Safetensors metadata has no final norm or language-model head weights")

    shared_names = input_tensors.keys() & output_tensors.keys()
    return ModelWeightProfile(
        layer_bytes=tuple(layer_bytes),
        input_endpoint_bytes=sum(input_tensors.values()),
        output_endpoint_bytes=sum(output_tensors.values()),
        shared_endpoint_bytes=sum(input_tensors[name] for name in shared_names),
        source_revision=source_revision,
    )


def load_hub_weight_profile(
    repo_id: str,
    *,
    num_layers: int,
    tie_word_embeddings: bool,
    revision: str | None = None,
    token: bool | str | None = None,
    api: HfApi | None = None,
) -> ModelWeightProfile:
    """Fetch all tensor headers with HTTP Range requests and build an exact profile."""

    if Path(repo_id).exists():
        raise ValueError("Local safetensors profiles are not supported by the Hub API")
    client = api or HfApi()
    resolved_revision = revision
    if resolved_revision is None:
        resolved_revision = resolve_hub_revision(repo_id, api=client)
    if not resolved_revision:
        raise ValueError(f"Could not resolve an immutable revision for {repo_id}")
    metadata = client.get_safetensors_metadata(
        repo_id,
        revision=resolved_revision,
        token=token,
    )
    tensors = {
        name: tensor
        for file_metadata in metadata.files_metadata.values()
        for name, tensor in file_metadata.tensors.items()
    }
    if not tensors:
        raise ValueError(f"No safetensors metadata found for {repo_id}")
    return build_weight_profile_from_tensors(
        tensors,
        num_layers=num_layers,
        tie_word_embeddings=tie_word_embeddings,
        source_revision=resolved_revision,
    )
