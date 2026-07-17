import pytest
import torch

from parallax.vllm.prefix_cache import (
    count_uncached_prefill_tokens,
    pad_pipeline_activations,
    select_pipeline_activation_suffix,
)


def test_counts_only_uncached_prefill_suffix():
    assert count_uncached_prefill_tokens(prompt_tokens=26, cached_tokens=16) == 10


@pytest.mark.parametrize("cached_tokens", [-1, 26, 27])
def test_rejects_invalid_cached_prefill_length(cached_tokens):
    with pytest.raises(ValueError):
        count_uncached_prefill_tokens(prompt_tokens=26, cached_tokens=cached_tokens)


def test_uses_activation_tail_when_downstream_cache_hit_is_longer():
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    selected = select_pipeline_activation_suffix(
        hidden_states,
        scheduled_tokens=2,
    )

    assert torch.equal(selected, hidden_states[-2:])


def test_pads_aligned_multi_request_batch():
    first = select_pipeline_activation_suffix(torch.ones((4, 3)), scheduled_tokens=2)
    second = select_pipeline_activation_suffix(torch.full((5, 3), 2.0), scheduled_tokens=3)
    tensors = {"hidden_states": torch.cat((first, second), dim=0)}

    padded = pad_pipeline_activations(
        tensors,
        scheduled_tokens=5,
        padded_tokens=8,
    )

    assert padded["hidden_states"].shape == (8, 3)
    assert torch.equal(padded["hidden_states"][:2], first)
    assert torch.equal(padded["hidden_states"][2:5], second)
    assert torch.count_nonzero(padded["hidden_states"][5:]) == 0


def test_rejects_upstream_cache_hit_that_downstream_cannot_satisfy():
    tensors = {"hidden_states": torch.zeros((2, 4))}

    with pytest.raises(RuntimeError, match="common cache prefix"):
        select_pipeline_activation_suffix(
            tensors["hidden_states"],
            scheduled_tokens=3,
        )
