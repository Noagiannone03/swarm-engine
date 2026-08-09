from types import SimpleNamespace

import pytest

from parallax.p2p.message_util import NativeActivationFrame
from parallax.server.executor.skippy_executor import SkippyExecutor


def activation(layer_start: int = 0, layer_end: int = 4) -> NativeActivationFrame:
    return NativeActivationFrame(
        version=1,
        dtype="f32",
        layout="token_major",
        producer_stage_index=layer_start,
        layer_start=layer_start,
        layer_end=layer_end,
        token_count=2,
        sequence_count=1,
        flags=0,
        payload=b"native-activation",
    )


class FakeRunner:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def prefill(self, *args):
        self.calls.append(("prefill", args))
        return self.result

    def decode(self, *args):
        self.calls.append(("decode", args))
        return self.result


def executor(*, first: bool, last: bool, result):
    instance = SkippyExecutor.__new__(SkippyExecutor)
    instance.runner = FakeRunner(result)
    instance.is_first_peer = first
    instance.is_last_peer = last
    return instance


def test_intermediate_stage_prefill_returns_typed_activation_without_sampling():
    input_frame = activation(0, 4)
    output_frame = activation(4, 8)
    instance = executor(
        first=False,
        last=False,
        result=SimpleNamespace(activation=output_frame, predicted_token=None),
    )
    request = SimpleNamespace(
        request_id="request-1",
        input_ids=[10, 11],
        hidden_states=input_frame,
        sampling_params=object(),
        is_prefill=True,
    )

    output = instance.process_batch({"requests": [request]}, return_decoded_tokens=False)

    assert output == {"hidden_states": [output_frame], "probs": None}
    assert instance.runner.calls == [
        ("prefill", ("request-1", [10, 11], input_frame, None))
    ]


def test_final_stage_decode_returns_native_sampled_token():
    input_frame = activation(4, 8)
    instance = executor(
        first=False,
        last=True,
        result=SimpleNamespace(activation=None, predicted_token=42),
    )
    sampling = object()
    request = SimpleNamespace(
        request_id="request-2",
        input_ids=[10, 11],
        hidden_states=input_frame,
        sampling_params=sampling,
        is_prefill=False,
        next_token_id=11,
    )

    output = instance.process_batch({"requests": [request]}, return_decoded_tokens=True)

    assert output == {"hidden_states": [42], "probs": None}
    assert instance.runner.calls == [
        ("decode", ("request-2", 11, input_frame, sampling))
    ]


def test_final_stage_rejects_missing_native_sample():
    instance = executor(
        first=True,
        last=True,
        result=SimpleNamespace(activation=activation(), predicted_token=None),
    )
    request = SimpleNamespace(
        request_id="request-3",
        input_ids=[10],
        hidden_states=None,
        sampling_params=object(),
        is_prefill=True,
    )

    with pytest.raises(RuntimeError, match="did not sample"):
        instance.process_batch({"requests": [request]}, return_decoded_tokens=True)


def test_intermediate_stage_rejects_missing_native_activation():
    instance = executor(
        first=True,
        last=False,
        result=SimpleNamespace(activation=None, predicted_token=None),
    )
    request = SimpleNamespace(
        request_id="request-4",
        input_ids=[10],
        hidden_states=None,
        sampling_params=object(),
        is_prefill=True,
    )

    with pytest.raises(RuntimeError, match="non-final"):
        instance.process_batch({"requests": [request]}, return_decoded_tokens=False)


def test_final_token_contract_rejects_non_integer_values():
    instance = SkippyExecutor.__new__(SkippyExecutor)

    assert instance._gen_token_id_from_hidden(7) == (7, [7])
    with pytest.raises(TypeError, match="exactly one integer"):
        instance._gen_token_id_from_hidden(True)
    with pytest.raises(TypeError, match="exactly one integer"):
        instance._gen_token_id_from_hidden([7])
