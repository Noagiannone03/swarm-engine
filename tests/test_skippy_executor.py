import hashlib
from types import SimpleNamespace

import pytest

from parallax.p2p.message_util import NativeActivationFrame
from parallax.server.executor.base_executor import ExecutorBatchCancelled
from parallax.server.executor.skippy_executor import SkippyExecutor
from swarm_protocol.contracts import SkippyExactStateKind
from swarm_protocol.kv_snapshot import KvSnapshotIncompatible


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

    def prefill(self, *args, **kwargs):
        del kwargs
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


def checkpoint_executor():
    class Registry:
        def __init__(self):
            self.begin_calls = []

        def begin_import(self, *, request_id, descriptor):
            self.begin_calls.append((request_id, descriptor))
            return "import-1"

    instance = SkippyExecutor.__new__(SkippyExecutor)
    instance.execution_plan = SimpleNamespace(
        plan_id="skippy-qwen3-q4",
        package_source_sha256=hashlib.sha256(b"weights").hexdigest(),
        runtime_release="mesh-llm/v0.74.0",
        runtime_abi_version="0.1.32",
        exact_state_kind=SkippyExactStateKind.DENSE_ATTENTION_KV,
    )
    instance.model_manifest = SimpleNamespace(
        model_swarm_id=hashlib.sha256(b"swarm").hexdigest(),
        immutable_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        tokenizer_hash=hashlib.sha256(b"tokenizer").hexdigest(),
        dtype="bfloat16",
        prefill_contract_hash=hashlib.sha256(b"prefill").hexdigest(),
        attention_kv_contract_hash=hashlib.sha256(b"attention").hexdigest(),
    )
    instance.checkpoint_exports = Registry()
    return instance


def checkpoint_descriptor(instance):
    descriptor = {
        "version": 1,
        "layer_start": 0,
        "layer_end": 4,
        "k_type": 1,
        "v_type": 1,
        "k_row_bytes": 128,
        "v_row_bytes": 128,
        "v_element_bytes": 2,
        "flags": 0,
    }
    compatibility = instance._checkpoint_compatibility(descriptor)
    descriptor["compatibility"] = compatibility.to_wire_dict()
    descriptor["compatibility_identity_hash"] = compatibility.identity_hash
    return descriptor


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
    assert instance.runner.calls == [("prefill", ("request-1", [10, 11], input_frame, None))]


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
    assert instance.runner.calls == [("decode", ("request-2", 11, input_frame, sampling))]


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


def test_shared_abort_marker_prevents_entering_native_execution():
    instance = executor(
        first=True,
        last=True,
        result=SimpleNamespace(activation=None, predicted_token=42),
    )
    instance.shared_state = SimpleNamespace(
        request_abort_requested=lambda request_id: request_id == "cancel-me"
    )
    request = SimpleNamespace(
        request_id="cancel-me",
        input_ids=[10],
        hidden_states=None,
        sampling_params=object(),
        is_prefill=True,
    )

    with pytest.raises(ExecutorBatchCancelled) as raised:
        instance.process_batch({"requests": [request]}, return_decoded_tokens=True)

    assert raised.value.request_ids == ("cancel-me",)
    assert instance.runner.calls == []


def test_checkpoint_import_accepts_only_the_exact_model_runtime_and_native_layout():
    instance = checkpoint_executor()
    descriptor = checkpoint_descriptor(instance)

    response, binary = instance.handle_executor_control(
        {
            "command": "checkpoint_import_begin",
            "request_id": "request-5",
            "descriptor": descriptor,
        }
    )

    assert response == {"handle": "import-1"}
    assert binary is None
    assert instance.checkpoint_exports.begin_calls == [("request-5", descriptor)]


def test_checkpoint_import_rejects_incompatible_abi_before_native_allocation():
    instance = checkpoint_executor()
    descriptor = checkpoint_descriptor(instance)
    instance.execution_plan.runtime_abi_version = "0.1.33"

    with pytest.raises(KvSnapshotIncompatible, match="different KV identity"):
        instance.handle_executor_control(
            {
                "command": "checkpoint_import_begin",
                "request_id": "request-6",
                "descriptor": descriptor,
            }
        )

    assert instance.checkpoint_exports.begin_calls == []


def test_checkpoint_import_rejects_a_forged_compatibility_identity():
    instance = checkpoint_executor()
    descriptor = checkpoint_descriptor(instance)
    descriptor["compatibility_identity_hash"] = hashlib.sha256(b"forged").hexdigest()

    with pytest.raises(ValueError, match="compatibility identity is invalid"):
        instance.handle_executor_control(
            {
                "command": "checkpoint_import_begin",
                "request_id": "request-7",
                "descriptor": descriptor,
            }
        )

    assert instance.checkpoint_exports.begin_calls == []
