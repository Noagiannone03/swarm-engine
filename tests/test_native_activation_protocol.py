import pytest

from parallax.p2p import message_util
from parallax.p2p.message_util import (
    NativeActivationFrame,
    proto_to_request,
    request_to_proto,
)
from parallax.server.request import IntermediateRequest, RequestStatus


def test_skippy_activation_round_trips_without_tensor_runtime_imports():
    activation = NativeActivationFrame(
        version=1,
        dtype="f32",
        layout="token_major",
        producer_stage_index=0,
        layer_start=0,
        layer_end=4,
        token_count=2,
        sequence_count=1,
        flags=0,
        payload=b"opaque-native-activation",
    )
    request = IntermediateRequest(
        request_id="request-1",
        input_ids=[1, 2],
        current_position=2,
        status=RequestStatus.PREFILLING,
        hidden_states=activation,
        routing_table=["worker-b"],
    )

    encoded = request_to_proto([request], device="vulkan:0")
    [restored] = proto_to_request(encoded, device="vulkan:0")

    assert encoded.reqs[0].HasField("native_activation")
    assert encoded.reqs[0].hidden_states == b""
    assert restored.hidden_states == activation


@pytest.mark.parametrize("token_id", [0, 42])
def test_native_terminal_token_round_trips_without_fake_activation(token_id):
    request = IntermediateRequest(
        request_id="request-terminal-token",
        input_ids=[1, 2],
        current_position=3,
        status=RequestStatus.DECODING,
        hidden_states=None,
        next_token_id=token_id,
        routing_table=["worker-head"],
    )

    encoded = request_to_proto([request], device="metal")
    [restored] = proto_to_request(encoded, device="metal")

    assert encoded.reqs[0].HasField("next_token_id")
    assert encoded.reqs[0].next_token_id == token_id
    assert encoded.reqs[0].hidden_states == b""
    assert not encoded.reqs[0].HasField("native_activation")
    assert restored.status is RequestStatus.DECODING
    assert restored.next_token_id == token_id
    assert restored.hidden_states is None


def test_skippy_activation_rejects_payload_over_protocol_limit(monkeypatch):
    monkeypatch.setattr(message_util, "_MAX_NATIVE_ACTIVATION_BYTES", 3)

    with pytest.raises(ValueError, match="512 MiB protocol limit"):
        NativeActivationFrame(
            version=1,
            dtype="f32",
            layout="opaque",
            producer_stage_index=0,
            layer_start=0,
            layer_end=4,
            token_count=1,
            sequence_count=1,
            flags=0,
            payload=b"four",
        )
