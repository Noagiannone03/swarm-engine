import hashlib
from types import SimpleNamespace

import pytest

from parallax.p2p.message_util import (
    NativeActivationFrame,
    proto_to_request,
    request_to_proto,
)
from parallax.server.executor.base_executor import ExecutorBatchCancelled
from parallax.server.executor.skippy_executor import (
    SkippyExecutor,
    _local_prefill_chunk_tokens,
    _SkippyResumeMarker,
)
from parallax.server.request import IntermediateRequest, RequestStatus
from swarm_protocol.contracts import SkippyExactStateKind
from swarm_protocol.kv_snapshot import KvSnapshotIncompatible
from swarm_protocol.recovery import token_sequence_checksum


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

    def release(self, request_id):
        self.calls.append(("release", request_id))


def test_complete_replica_exposes_its_exact_native_prefill_quantum_to_scheduler():
    assert (
        _local_prefill_chunk_tokens(
            is_full_model_stage=True,
            max_num_tokens_per_batch=8192,
        )
        == 512
    )
    assert (
        _local_prefill_chunk_tokens(
            is_full_model_stage=True,
            max_num_tokens_per_batch=256,
        )
        == 256
    )


def test_split_stage_does_not_claim_unqualified_distributed_chunking():
    assert (
        _local_prefill_chunk_tokens(
            is_full_model_stage=False,
            max_num_tokens_per_batch=8192,
        )
        is None
    )


def executor(*, first: bool, last: bool, result):
    instance = SkippyExecutor.__new__(SkippyExecutor)
    instance.runner = FakeRunner(result)
    instance.is_first_peer = first
    instance.is_last_peer = last
    instance._checkpoint_import_markers = {}
    instance._checkpoint_import_handle_by_request = {}
    instance._checkpoint_resume_markers = {}
    instance._authority_by_engine_request = {}
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
        runtime_release="mesh-llm/v0.75.1",
        runtime_abi_version="0.1.35",
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
    instance.runner = FakeRunner(SimpleNamespace(activation=None, predicted_token=None))
    instance._checkpoint_import_markers = {}
    instance._checkpoint_import_handle_by_request = {}
    instance._checkpoint_resume_markers = {}
    instance._authority_by_engine_request = {}
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
        "token_count": 2,
        "resume_route_id": "route-8",
        "resume_route_epoch": 8,
        "token_prefix_checksum": token_sequence_checksum((10, 20)),
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

    assert instance._gen_token_id_from_hidden(7) == (7, None)
    with pytest.raises(TypeError, match="exactly one integer"):
        instance._gen_token_id_from_hidden(True)
    with pytest.raises(TypeError, match="exactly one integer"):
        instance._gen_token_id_from_hidden([7])


def test_final_stage_token_zero_crosses_native_wire_without_tensor_serialization():
    instance = executor(
        first=False,
        last=True,
        result=SimpleNamespace(activation=None, predicted_token=0),
    )
    request = IntermediateRequest(
        request_id="request-terminal-zero",
        input_ids=[10, 11],
        current_position=2,
        status=RequestStatus.DECODING,
        hidden_states=activation(4, 8),
        next_token_id=11,
        routing_table=["worker-head"],
    )

    [outbound] = instance.prepare_next_batch_requests(
        [request],
        {"hidden_states": [0], "probs": None},
        context_lengths=[1],
    )
    encoded = request_to_proto([outbound], device="metal")
    [restored] = proto_to_request(encoded, device="metal")

    assert outbound.hidden_states is None
    assert outbound.next_token_id == 0
    assert encoded.reqs[0].HasField("next_token_id")
    assert encoded.reqs[0].hidden_states == b""
    assert restored.status is RequestStatus.DECODING
    assert restored.hidden_states is None
    assert restored.next_token_id == 0


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


def test_warm_resume_verifies_the_full_prefix_and_prefills_only_the_suffix():
    instance = executor(
        first=True,
        last=True,
        result=SimpleNamespace(activation=None, predicted_token=42),
    )
    marker = _SkippyResumeMarker(
        route_id="route-9",
        epoch=9,
        token_count=3,
        token_prefix_checksum=token_sequence_checksum((10, 20, 30)),
    )
    instance._checkpoint_resume_markers["request-8"] = marker
    request = SimpleNamespace(
        request_id="request-8",
        input_ids=[10, 20, 30, 40],
        origin_input_ids=[10, 20, 30, 40],
        hidden_states=None,
        sampling_params=object(),
        is_prefill=True,
        route_id="route-9",
        route_epoch=9,
    )

    output = instance.process_batch({"requests": [request]}, return_decoded_tokens=True)

    assert output == {"hidden_states": [42], "probs": None}
    assert instance.runner.calls == [
        ("prefill", ("request-8", [40], None, request.sampling_params))
    ]
    assert instance._checkpoint_resume_markers == {}


def test_warm_resume_rejects_a_divergent_token_prefix_before_native_execution():
    instance = executor(
        first=True,
        last=True,
        result=SimpleNamespace(activation=None, predicted_token=42),
    )
    instance._checkpoint_resume_markers["request-9"] = _SkippyResumeMarker(
        route_id="route-9",
        epoch=9,
        token_count=3,
        token_prefix_checksum=token_sequence_checksum((10, 20, 99)),
    )
    request = SimpleNamespace(
        request_id="request-9",
        input_ids=[10, 20, 30, 40],
        origin_input_ids=[10, 20, 30, 40],
        hidden_states=None,
        sampling_params=object(),
        is_prefill=True,
        route_id="route-9",
        route_epoch=9,
    )

    with pytest.raises(ValueError, match="prefix differs"):
        instance.process_batch({"requests": [request]}, return_decoded_tokens=True)

    assert instance.runner.calls == []


def test_native_session_uses_the_stable_authority_id_across_engine_request_ids():
    instance = executor(
        first=True,
        last=True,
        result=SimpleNamespace(activation=None, predicted_token=42),
    )
    request = SimpleNamespace(
        request_id="engine-internal-1",
        authority_request_id="request-agent-1",
        input_ids=[10, 20],
        hidden_states=None,
        sampling_params=object(),
        is_prefill=True,
    )

    instance.process_batch({"requests": [request]}, return_decoded_tokens=True)
    instance._release_request("engine-internal-1")

    assert instance.runner.calls == [
        ("prefill", ("request-agent-1", [10, 20], None, request.sampling_params)),
        ("release", "request-agent-1"),
    ]
