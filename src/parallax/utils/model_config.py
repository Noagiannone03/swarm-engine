"""Pure model-config normalization shared by runtime loading and registry tooling."""

from __future__ import annotations

from typing import Any


def _normalize_quantization_key(key: str) -> str:
    """Map VLM text tower quantization keys to the text-only key layout."""

    prefixes = ("model.language_model.", "language_model.")
    for prefix in prefixes:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix.startswith("model.lm_head."):
            return suffix.replace("model.", "", 1)
        if suffix.startswith("model.") or suffix.startswith("lm_head."):
            return suffix
        return f"model.{suffix}"
    return key


def _normalize_quantization_config(quantization: Any) -> Any:
    if not isinstance(quantization, dict):
        return quantization

    normalized = {}
    for key, value in quantization.items():
        if isinstance(value, dict):
            normalized[_normalize_quantization_key(key)] = _normalize_quantization_config(value)
        elif key == "ignored_layers" and isinstance(value, list):
            normalized[key] = [
                _normalize_quantization_key(layer) if isinstance(layer, str) else layer
                for layer in value
            ]
        else:
            normalized[key] = value
    return normalized


def normalize_model_config(config: dict) -> dict:
    """Expose nested text model fields at the top level for VLM-style configs."""

    text_config = config.get("text_config")
    if config.get("model_type") in {"qwen3_5", "qwen3_5_moe"} and isinstance(text_config, dict):
        normalized = {**config, **text_config}
        normalized["model_type"] = config["model_type"]
        normalized["architectures"] = config.get("architectures", normalized.get("architectures"))
        normalized["tie_word_embeddings"] = text_config.get(
            "tie_word_embeddings", config.get("tie_word_embeddings", False)
        )
        return normalized
    if config.get("model_type") == "minimax_m3_vl" and isinstance(text_config, dict):
        normalized = {**config, **text_config}
        normalized["model_type"] = "minimax_m3"
        normalized["original_model_type"] = config["model_type"]
        normalized["architectures"] = text_config.get("architectures") or [
            "MiniMaxM3SparseForCausalLM"
        ]
        normalized["tie_word_embeddings"] = text_config.get(
            "tie_word_embeddings", config.get("tie_word_embeddings", False)
        )

        sparse_config = normalized.get("sparse_attention_config")
        if isinstance(sparse_config, dict):
            normalized["index_head_dim"] = sparse_config.get(
                "sparse_index_dim", normalized.get("index_head_dim")
            )
            # MiniMax-M3 stores a single sparse index key head; sparse_num_index_heads is the
            # number of query heads used for block selection.
            normalized["index_n_heads"] = normalized.get("index_n_heads", 1)
            normalized["index_block_size"] = sparse_config.get(
                "sparse_block_size", normalized.get("index_block_size")
            )
            normalized["index_topk_blocks"] = sparse_config.get(
                "sparse_topk_blocks", normalized.get("index_topk_blocks")
            )
            normalized["index_local_blocks"] = sparse_config.get(
                "sparse_local_block", normalized.get("index_local_blocks")
            )

        if (
            normalized.get("moe_intermediate_size") is None
            and normalized.get("intermediate_size") is not None
        ):
            normalized["moe_intermediate_size"] = normalized["intermediate_size"]

        for quantization_key in ("quantization", "quantization_config"):
            if quantization_key in normalized:
                normalized[quantization_key] = _normalize_quantization_config(
                    normalized[quantization_key]
                )
        if "quantization" not in normalized and "quantization_config" in normalized:
            normalized["quantization"] = normalized["quantization_config"]
        return normalized
    return config


def get_model_context_limit(config: dict) -> int | None:
    """Return the finite total-sequence limit declared by a model config.

    Hugging Face text models usually expose ``max_position_embeddings`` while
    some tokenizers/models use ``model_max_length``. VLM repositories can put
    the same fields under ``text_config``. Very large tokenizer sentinels mean
    "unknown" and are excluded.
    """

    candidates = [
        config.get("max_position_embeddings"),
        config.get("model_max_length"),
    ]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        candidates.extend(
            [
                text_config.get("max_position_embeddings"),
                text_config.get("model_max_length"),
            ]
        )
    limits = [
        int(value)
        for value in candidates
        if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < 2**63
    ]
    return min(limits) if limits else None
