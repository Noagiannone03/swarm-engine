from __future__ import annotations

from typing import Any

import torch


def count_uncached_prefill_tokens(prompt_tokens: int, cached_tokens: int) -> int:
    """Return the prompt suffix that vLLM must compute for a cache hit."""
    if prompt_tokens <= 0:
        raise ValueError("A prefill request must contain at least one token")
    if cached_tokens < 0 or cached_tokens >= prompt_tokens:
        raise ValueError(
            "Cached prefill tokens must leave at least one token for logits: "
            f"cached={cached_tokens}, prompt={prompt_tokens}"
        )
    return prompt_tokens - cached_tokens


def select_pipeline_activation_suffix(
    tensor: torch.Tensor, *, scheduled_tokens: int
) -> torch.Tensor:
    """Select the prompt suffix required by a downstream prefix-cache hit."""
    if scheduled_tokens <= 0:
        raise ValueError("scheduled_tokens must be positive")

    received_tokens = tensor.shape[0]
    if received_tokens < scheduled_tokens:
        raise RuntimeError(
            "Distributed prefix cache mismatch: the upstream shard sent "
            f"{received_tokens} token activations, but this shard needs "
            f"{scheduled_tokens}. The pipeline must retry prefill with a "
            "common cache prefix."
        )

    return tensor[-scheduled_tokens:]


def pad_pipeline_activations(
    intermediate_tensors: Any,
    *,
    scheduled_tokens: int,
    padded_tokens: int,
) -> Any:
    """Pad an already aligned activation batch to vLLM's input size."""
    if scheduled_tokens <= 0:
        raise ValueError("scheduled_tokens must be positive")
    if padded_tokens < scheduled_tokens:
        raise ValueError(
            f"padded_tokens ({padded_tokens}) cannot be less than "
            f"scheduled_tokens ({scheduled_tokens})"
        )

    for key, tensor in list(intermediate_tensors.items()):
        received_tokens = tensor.shape[0]
        if received_tokens != scheduled_tokens:
            raise ValueError(
                f"Aligned activation tensor {key!r} contains {received_tokens} tokens; "
                f"expected {scheduled_tokens}"
            )

        if padded_tokens > scheduled_tokens:
            pad_shape = (padded_tokens - scheduled_tokens,) + tensor.shape[1:]
            pad = tensor.new_zeros(pad_shape)
            tensor = torch.cat((tensor, pad), dim=0)

        intermediate_tensors[key] = tensor

    return intermediate_tensors
