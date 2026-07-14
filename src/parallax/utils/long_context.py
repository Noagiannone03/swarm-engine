"""Backend-independent long-context configuration.

Requesting a larger KV pool is not enough to extend a model's positional
encoding.  This module applies only model-family overrides that are documented
by the model author.  Unknown architectures fail closed instead of silently
running beyond their trained/configured context.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional


_QWEN3_YARN_MODEL_TYPES = {"qwen3", "qwen3_moe"}
_QWEN3_ORIGINAL_CONTEXT = 32768


def configure_long_context(
    config: Mapping[str, Any], requested_tokens: Optional[int]
) -> dict[str, Any]:
    """Return a model config that safely supports ``requested_tokens``.

    Qwen documents static YaRN for Qwen3 contexts beyond the repository's
    default limit.  Other families must carry their own long-context config in
    the model repository until an explicit, tested policy is added here.
    """

    result = deepcopy(dict(config))
    if requested_tokens is None:
        return result

    requested = int(requested_tokens)
    if requested <= 0:
        raise ValueError("requested context length must be positive")

    configured_limit = int(result.get("max_position_embeddings") or 0)
    if configured_limit <= 0:
        configured_limit = int(result.get("model_max_length") or 0)
    if configured_limit <= 0 or requested <= configured_limit:
        return result

    model_type = str(result.get("model_type") or "").lower()
    if model_type not in _QWEN3_YARN_MODEL_TYPES:
        raise ValueError(
            f"{model_type or 'unknown model'} is configured for {configured_limit} tokens, "
            f"but {requested} were requested and no validated long-context policy exists"
        )

    original = _QWEN3_ORIGINAL_CONTEXT
    existing = result.get("rope_scaling")
    if isinstance(existing, Mapping):
        original = int(existing.get("original_max_position_embeddings") or original)
    if original <= 0:
        raise ValueError("invalid original context length for Qwen3 YaRN")

    result["max_position_embeddings"] = requested
    result["rope_scaling"] = {
        "rope_type": "yarn",
        "factor": float(requested) / float(original),
        "original_max_position_embeddings": original,
    }
    return result


def long_context_overrides(
    original: Mapping[str, Any], configured: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the HF/SGLang fields changed by ``configure_long_context``."""

    overrides: dict[str, Any] = {}
    for key in ("max_position_embeddings", "rope_scaling"):
        if configured.get(key) != original.get(key):
            overrides[key] = deepcopy(configured.get(key))
    return overrides
