"""
Utility functions for message serialization and deserialization.

This module contains utility functions for serializing and deserializing messages
between the P2P server and the executor.
"""

import io
from dataclasses import dataclass
from typing import Any, List, Optional

try:
    import mlx.core as mx
except ImportError:  # MLX is not available in native Windows CUDA runtimes.
    mx = None

from parallax.p2p.proto import forward_pb2
from parallax.server.backend_capabilities import TensorRuntime, tensor_runtime_for_device
from parallax.server.request import IntermediateRequest, Request, RequestStatus
from parallax.server.sampling.sampling_params import SamplingParams


# Keep the same hard ceiling as Mesh-LLM's maintained Skippy protocol. This is
# a protocol safety bound, not a generation timeout or a context-size policy.
_MAX_NATIVE_ACTIVATION_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class NativeActivationFrame:
    """Backend-neutral Python carrier for one verified Skippy activation frame."""

    version: int
    dtype: str
    layout: str
    producer_stage_index: int
    layer_start: int
    layer_end: int
    token_count: int
    sequence_count: int
    flags: int
    payload: bytes

    def __post_init__(self) -> None:
        if self.dtype not in {"f32", "f16", "bf16"}:
            raise ValueError(f"unsupported native activation dtype {self.dtype!r}")
        if self.layout not in {"opaque", "token_major"}:
            raise ValueError(f"unsupported native activation layout {self.layout!r}")
        if self.layer_end < self.layer_start:
            raise ValueError("native activation has an invalid layer range")
        if self.token_count <= 0 or self.sequence_count <= 0:
            raise ValueError("native activation dimensions must be positive")
        if not self.payload:
            raise ValueError("native activation payload must not be empty")
        if len(self.payload) > _MAX_NATIVE_ACTIVATION_BYTES:
            raise ValueError("native activation payload exceeds the 512 MiB protocol limit")


def _native_activation_to_proto(
    frame: NativeActivationFrame,
    target: forward_pb2.NativeActivationFrame,
) -> None:
    target.version = frame.version
    target.dtype = frame.dtype
    target.layout = frame.layout
    target.producer_stage_index = frame.producer_stage_index
    target.layer_start = frame.layer_start
    target.layer_end = frame.layer_end
    target.token_count = frame.token_count
    target.sequence_count = frame.sequence_count
    target.flags = frame.flags
    target.payload = frame.payload


def _native_activation_from_proto(
    frame: forward_pb2.NativeActivationFrame,
) -> NativeActivationFrame:
    return NativeActivationFrame(
        version=frame.version,
        dtype=frame.dtype,
        layout=frame.layout,
        producer_stage_index=frame.producer_stage_index,
        layer_start=frame.layer_start,
        layer_end=frame.layer_end,
        token_count=frame.token_count,
        sequence_count=frame.sequence_count,
        flags=frame.flags,
        payload=bytes(frame.payload),
    )


def _require_mlx():
    if mx is None:
        raise RuntimeError("MLX tensor serialization requires the MLX runtime")
    return mx


def request_to_proto(
    requests: List[IntermediateRequest],
    device: Optional[str] = "mlx",
) -> forward_pb2.ForwardRequest:
    """
    Convert a list of IntermediateRequest objects to a ForwardRequest protobuf message.
    IntermediateRequest contains request_id, current_position, status, and hidden_states.
    """
    forward_request = forward_pb2.ForwardRequest()
    assert len(requests) > 0, "No requests to convert"
    assert all(request.status == requests[0].status for request in requests), (
        "All requests must have the same status"
    )
    if requests[0].status == RequestStatus.PREFILLING:
        forward_request.forward_mode = forward_pb2.ForwardMode.EXTEND
    elif requests[0].status == RequestStatus.DECODING:
        forward_request.forward_mode = forward_pb2.ForwardMode.DECODE
    else:
        raise ValueError(f"Invalid status: {requests[0].status}")

    for request in requests:
        proto_req = forward_pb2.Req()
        proto_req.rid = request.request_id
        proto_req.output_length = request.current_position - len(request.input_ids)
        proto_req.input_ids.extend(request.input_ids)
        proto_req.routing_table.extend(request.routing_table)
        proto_req.sampling_params.CopyFrom(sampling_params_to_proto(request.sampling_params))
        proto_req.lora_path = request.lora_path if request.lora_path is not None else ""
        proto_req.route_id = request.route_id
        proto_req.route_epoch = request.route_epoch
        proto_req.authority_request_id = request.authority_request_id

        if request.hidden_states is not None:
            if isinstance(request.hidden_states, NativeActivationFrame):
                _native_activation_to_proto(
                    request.hidden_states,
                    proto_req.native_activation,
                )
            else:
                proto_req.hidden_states = tensor_to_bytes(request.hidden_states, device=device)

        if request.next_token_id is not None:
            proto_req.next_token_id = request.next_token_id

        # Add token_prob if available
        if hasattr(request, "token_prob") and request.token_prob is not None:
            proto_req.token_prob = request.token_prob

        # Add return_probs flag
        if hasattr(request, "return_probs"):
            proto_req.return_probs = request.return_probs

        forward_request.reqs.append(proto_req)

    return forward_request


def proto_to_request(
    proto_request: forward_pb2.ForwardRequest,
    device: Optional[str] = "mlx",
) -> List[IntermediateRequest]:
    """
    Convert a ForwardRequest protobuf message to a IntermediateRequest object.
    """

    requests = []

    for proto_req in proto_request.reqs:
        current_position = len(proto_req.input_ids) + proto_req.output_length

        next_token_id = proto_req.next_token_id if proto_req.HasField("next_token_id") else None

        hidden_states = None
        if proto_req.HasField("native_activation"):
            hidden_states = _native_activation_from_proto(proto_req.native_activation)
        elif proto_req.hidden_states:
            hidden_states = bytes_to_tensor(proto_req.hidden_states, device)

        status = None
        if hidden_states is None and next_token_id is None:
            status = RequestStatus.FINISHED_EOS
        elif proto_request.forward_mode == forward_pb2.ForwardMode.EXTEND:
            status = RequestStatus.PREFILLING
        elif proto_request.forward_mode == forward_pb2.ForwardMode.DECODE:
            status = RequestStatus.DECODING
        else:
            raise ValueError(f"Invalid forward mode: {proto_request.forward_mode}")

        sampling_params = proto_to_sampling_params(proto_req.sampling_params)

        # Extract token_prob if present
        token_prob = None
        if proto_req.HasField("token_prob"):
            token_prob = proto_req.token_prob

        # Extract return_probs (defaults to False if not present)
        return_probs = proto_req.return_probs if hasattr(proto_req, "return_probs") else False

        request = IntermediateRequest(
            request_id=proto_req.rid,
            current_position=current_position,
            status=status,
            input_ids=list(proto_req.input_ids),
            hidden_states=hidden_states,
            routing_table=list(proto_req.routing_table),
            next_token_id=next_token_id,
            sampling_params=sampling_params,
            lora_path=proto_req.lora_path if proto_req.lora_path != "" else None,
            token_prob=token_prob,
            return_probs=return_probs,
            route_id=proto_req.route_id,
            route_epoch=proto_req.route_epoch,
            authority_request_id=proto_req.authority_request_id or None,
        )

        requests.append(request)

    return requests


def abort_request_to_proto(reqs: List[Request]) -> forward_pb2.AbortRequest:
    """Converts aborted/finished requests to a AbortRequest"""
    proto = forward_pb2.AbortRequest()
    for req in reqs:
        req_proto = forward_pb2.Req()
        req_proto.rid = req.request_id
        if req.routing_table is not None:
            req_proto.routing_table.extend(req.routing_table)
        req_proto.terminal_error = bool(
            getattr(req, "terminal_error", False) or req.status == RequestStatus.ERROR
        )
        req_proto.route_id = req.route_id
        req_proto.route_epoch = req.route_epoch
        req_proto.authority_request_id = req.authority_request_id
        proto.reqs.append(req_proto)
    return proto


def proto_to_abort_request(proto_request: forward_pb2.AbortRequest) -> List[IntermediateRequest]:
    """
    Converts a AbortRequest a list of IntermediateRequest objects.
    Only request_id and routing table are useful information.
    """
    requests = []
    for proto_req in proto_request.reqs:
        terminal_error = bool(proto_req.terminal_error)
        status = RequestStatus.ERROR if terminal_error else RequestStatus.FINISHED_ABORT
        request = IntermediateRequest(
            request_id=proto_req.rid,
            current_position=0,
            status=status,
            routing_table=list(proto_req.routing_table),
            route_id=proto_req.route_id,
            route_epoch=proto_req.route_epoch,
            authority_request_id=proto_req.authority_request_id or None,
        )
        request.abort = not terminal_error
        request.terminal_error = terminal_error

        requests.append(request)

    return requests


def proto_to_sampling_params(proto: forward_pb2.SamplingParams) -> SamplingParams:
    """Convert protobuf message to SamplingParams."""
    if proto is None or not proto.ByteSize():
        return SamplingParams()
    sampling_params = SamplingParams(
        max_new_tokens=proto.max_new_tokens,
        min_new_tokens=proto.min_new_tokens,
        temperature=proto.temperature,
        top_p=proto.top_p,
        min_p=proto.min_p,
        top_k=proto.top_k,
        stop_strs=list(proto.stop_strs),
        stop_token_ids=list(proto.stop_token_ids),
        ignore_eos=proto.ignore_eos,
        repetition_penalty=proto.repetition_penalty,
        presence_penalty=proto.presence_penalty,
        frequency_penalty=proto.frequency_penalty,
        json_schema=proto.json_schema,
    )
    return sampling_params


def sampling_params_to_proto(params: SamplingParams) -> forward_pb2.SamplingParams:
    """Convert SamplingParams to protobuf message."""
    proto = forward_pb2.SamplingParams()

    proto.max_new_tokens = params.max_new_tokens
    proto.min_new_tokens = params.min_new_tokens
    proto.temperature = params.temperature
    proto.top_p = params.top_p
    proto.min_p = params.min_p
    proto.top_k = params.top_k
    if params.stop_strs is not None:
        proto.stop_strs.extend(params.stop_strs)
    if params.stop_token_ids is not None:
        proto.stop_token_ids.extend(params.stop_token_ids)
    proto.ignore_eos = params.ignore_eos
    proto.repetition_penalty = params.repetition_penalty
    proto.presence_penalty = params.presence_penalty
    proto.frequency_penalty = params.frequency_penalty
    if params.json_schema is not None:
        proto.json_schema = params.json_schema
    return proto


def tensor_to_bytes(tensor: Any, device: Optional[str] = "mlx") -> bytes:
    """Convert tensor to protobuf Tensor using safetensor serialization."""
    tensor_runtime = tensor_runtime_for_device(device)
    if tensor_runtime is TensorRuntime.TORCH:
        from safetensors.torch import save

        # Convert tensor to CPU
        if tensor.device.type != "cpu":
            cpu_tensor = tensor.cpu()
        else:
            cpu_tensor = tensor
        # Store buffer using safetensor (dtype and size are automatically preserved)
        serialized_data = save({"tensor": cpu_tensor.contiguous()})
        return serialized_data
    elif tensor_runtime is TensorRuntime.NUMPY:
        import numpy as np
        from safetensors.numpy import save

        array = np.ascontiguousarray(tensor)
        if array.size == 0:
            raise ValueError("Tensor must have size > 0")
        return save({"tensor": array})
    elif tensor_runtime is TensorRuntime.NATIVE:
        raise TypeError("native activation frames must use the typed protobuf field")
    else:
        mlx = _require_mlx()
        assert tensor.size > 0, "Tensor must have size > 0"
        buffer = io.BytesIO()
        mlx.save_safetensors(buffer, {"tensor": tensor})
        return buffer.getvalue()


def bytes_to_tensor(
    tensor: bytes,
    device: Optional[str] = "mlx",
) -> Any:
    """Convert bytes (safetensor format) to tensor."""
    tensor_runtime = tensor_runtime_for_device(device)
    if tensor_runtime is TensorRuntime.TORCH:
        from safetensors.torch import load

        tensor_dict = load(tensor)
        tensor = tensor_dict["tensor"].to(device)
    elif tensor_runtime is TensorRuntime.NUMPY:
        from safetensors.numpy import load

        return load(tensor)["tensor"]
    elif tensor_runtime is TensorRuntime.NATIVE:
        raise TypeError("native activation frames must use the typed protobuf field")
    else:
        mlx = _require_mlx()
        buffer = io.BytesIO(tensor)
        tensors_dict = mlx.load(buffer, format="safetensors")
        tensor = tensors_dict["tensor"]
    return tensor
