from __future__ import annotations

import pytest

from parallax.vllm.request_compat import create_vllm_request


class LegacyRequest:
    def __init__(self, *, sampling_params, eos_token_id, request_id):
        self.sampling_params = sampling_params
        self.eos_token_id = eos_token_id
        self.request_id = request_id


class SamplingParamsEOS:
    def __init__(self):
        self.eos_token_id = None
        self.updated_with = None

    def update_from_generation_config(self, generation_config, eos_token_id):
        self.updated_with = (generation_config, eos_token_id)
        self.eos_token_id = eos_token_id


class CurrentRequest:
    def __init__(self, *, sampling_params, request_id):
        self.sampling_params = sampling_params
        self.request_id = request_id


def test_passes_eos_to_official_vllm_016_request_contract():
    sampling_params = object()

    request = create_vllm_request(
        LegacyRequest,
        request_id="official-016",
        sampling_params=sampling_params,
        eos_token_id=151645,
    )

    assert request.eos_token_id == 151645
    assert request.sampling_params is sampling_params


def test_moves_eos_to_current_sampling_params_contract():
    sampling_params = SamplingParamsEOS()

    request = create_vllm_request(
        CurrentRequest,
        request_id="windows-016",
        sampling_params=sampling_params,
        eos_token_id=151645,
    )

    assert request.sampling_params.eos_token_id == 151645
    assert sampling_params.updated_with == ({}, 151645)


def test_omits_empty_eos_when_current_request_contract_is_used():
    sampling_params = object()

    request = create_vllm_request(
        CurrentRequest,
        request_id="no-eos",
        sampling_params=sampling_params,
        eos_token_id=None,
    )

    assert request.sampling_params is sampling_params


def test_rejects_unknown_request_contract_when_eos_would_be_lost():
    with pytest.raises(RuntimeError, match="Unsupported vLLM request API"):
        create_vllm_request(
            CurrentRequest,
            request_id="unknown",
            sampling_params=object(),
            eos_token_id=151645,
        )


def test_does_not_hide_type_errors_raised_inside_vllm_constructor():
    class BrokenRequest:
        def __init__(self, *, sampling_params, eos_token_id):
            raise TypeError("internal vLLM failure")

    with pytest.raises(TypeError, match="internal vLLM failure"):
        create_vllm_request(
            BrokenRequest,
            sampling_params=object(),
            eos_token_id=None,
        )
