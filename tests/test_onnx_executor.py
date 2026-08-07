import numpy as np
import pytest

from parallax.server.executor.onnx_executor import sample_numpy_token
from parallax.server.request import Request
from parallax.server.sampling.sampling_params import SamplingParams


def request_with(**params):
    return Request(
        input_ids=[1],
        sampling_params=SamplingParams(**params),
    )


def test_numpy_sampler_greedy_and_repetition_penalty():
    rng = np.random.default_rng(7)
    greedy = request_with(top_k=1)
    assert sample_numpy_token(greedy, np.asarray([1.0, 3.0]), [], rng) == (1, 1.0)

    penalized = request_with(top_k=1, repetition_penalty=2.0)
    token, _ = sample_numpy_token(
        penalized,
        np.asarray([1.75, 3.0]),
        [1],
        rng,
    )
    assert token == 0


def test_numpy_sampler_top_k_never_selects_filtered_token():
    request = request_with(top_k=2, top_p=1.0, temperature=1.0)
    rng = np.random.default_rng(42)

    selected = {
        sample_numpy_token(request, np.asarray([9.0, 8.0, 7.0, 6.0]), [], rng)[0]
        for _ in range(100)
    }

    assert selected <= {0, 1}
    assert selected == {0, 1}


def test_numpy_sampler_fails_closed_for_unqualified_grammar():
    request = request_with(top_k=1, json_schema='{"type":"object"}')

    with pytest.raises(ValueError, match="grammar"):
        sample_numpy_token(
            request,
            np.asarray([1.0, 2.0]),
            [],
            np.random.default_rng(1),
        )
