#!/usr/bin/env python3
"""Qualify a two-host Skippy runtime-slice pipeline over Fabi's native RPC.

This is an operator tool, not a second data plane. It deliberately reuses the
production ABI3 extension and ``NativeActivationFrame`` protobuf so a live
qualification covers model slicing, KV continuity, activation serialization,
Iroh discovery/relay, and final-stage sampling without requiring a published
catalogue candidate first.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from parallax.p2p.proto import forward_pb2


FORWARD_METHOD = "fabi.skippy.qualification.forward.v1"


def _relay_token(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    prefix = "IROH_RELAY_ACCESS_TOKEN="
    if value.startswith(prefix):
        value = value.removeprefix(prefix).strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("relay token file must contain exactly one credential")
    return value


def _native_frame(native: Any, frame: forward_pb2.NativeActivationFrame) -> Any:
    if not frame.payload:
        raise ValueError("received an empty native activation")
    return native.SkippyActivationFrame(
        frame.payload,
        version=frame.version,
        dtype=frame.dtype,
        layout=frame.layout,
        producer_stage_index=frame.producer_stage_index,
        layer_start=frame.layer_start,
        layer_end=frame.layer_end,
        token_count=frame.token_count,
        sequence_count=frame.sequence_count,
        flags=frame.flags,
    )


def _protobuf_frame(native_frame: Any) -> forward_pb2.NativeActivationFrame:
    return forward_pb2.NativeActivationFrame(
        version=native_frame.version,
        dtype=native_frame.dtype,
        layout=native_frame.layout,
        producer_stage_index=native_frame.producer_stage_index,
        layer_start=native_frame.layer_start,
        layer_end=native_frame.layer_end,
        token_count=native_frame.token_count,
        sequence_count=native_frame.sequence_count,
        flags=native_frame.flags,
        payload=bytes(native_frame.payload()),
    )


def _sampling_kwargs(params: forward_pb2.SamplingParams) -> dict[str, Any]:
    return {
        "sample": True,
        "temperature": float(params.temperature or 1.0),
        "top_p": float(params.top_p or 1.0),
        "top_k": int(params.top_k),
        "min_p": float(params.min_p),
        "presence_penalty": float(params.presence_penalty),
        "frequency_penalty": float(params.frequency_penalty),
        "repeat_penalty": float(params.repetition_penalty or 1.0),
        "penalty_last_n": -1,
    }


def _load_native(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import fabi_network_native as native

    native.load_skippy_native_runtime(
        args.runtime_root,
        args.mesh_release,
        args.runtime_abi,
        args.backend,
    )
    geometry = native.inspect_skippy_source_geometry(
        args.model_path,
        args.cache_type_k,
        args.cache_type_v,
    )
    if not 0 < args.split_layer < geometry.layer_count:
        raise ValueError(
            f"split layer must be between 1 and {geometry.layer_count - 1}"
        )
    return native, geometry, args.model_path


def _stage(
    native: Any,
    args: argparse.Namespace,
    *,
    start: int,
    end: int,
) -> Any:
    return native.SkippyStage(
        args.model_path,
        stage_index=start,
        layer_start=start,
        layer_end=end,
        model_layer_count=args.layer_count,
        context_tokens=args.context_tokens,
        lane_count=1,
        selected_backend_device=args.backend_device,
        cache_type_k=args.cache_type_k,
        cache_type_v=args.cache_type_v,
        use_mmap=True,
        load_mode="runtime_slice",
    )


def _network_node(native: Any, args: argparse.Namespace) -> Any:
    return native.NetworkNode(
        args.identity_path,
        args.relay_url,
        _relay_token(args.relay_token_file),
        force_relay=args.force_relay,
        response_timeout_seconds=args.rpc_timeout_seconds,
    )


def serve_tail(args: argparse.Namespace) -> None:
    native, geometry, _ = _load_native(args)
    args.layer_count = int(geometry.layer_count)
    stage = _stage(
        native,
        args,
        start=args.split_layer,
        end=args.layer_count,
    )
    node = _network_node(native, args)
    endpoint_tmp = args.endpoint_file.with_suffix(args.endpoint_file.suffix + ".tmp")
    endpoint_tmp.parent.mkdir(parents=True, exist_ok=True)
    endpoint_tmp.write_text(node.endpoint_id + "\n", encoding="utf-8")
    os.replace(endpoint_tmp, args.endpoint_file)
    print(f"READY endpoint={node.endpoint_id} span=[{args.split_layer},{args.layer_count})")
    handled = 0
    try:
        while handled < args.max_calls:
            incoming = node.recv(timeout_seconds=args.rpc_timeout_seconds)
            try:
                if incoming.method != FORWARD_METHOD:
                    raise ValueError(f"unsupported qualification RPC {incoming.method!r}")
                request = forward_pb2.ForwardRequest.FromString(bytes(incoming.body))
                if len(request.reqs) != 1:
                    raise ValueError("qualification RPC requires exactly one request")
                item = request.reqs[0]
                activation = _native_frame(native, item.native_activation)
                started = time.perf_counter()
                if request.forward_mode == forward_pb2.EXTEND:
                    output = stage.prefill(
                        item.rid,
                        list(item.input_ids),
                        activation,
                        **_sampling_kwargs(item.sampling_params),
                    )
                elif request.forward_mode == forward_pb2.DECODE:
                    output = stage.decode(
                        item.rid,
                        item.next_token_id,
                        activation,
                        **_sampling_kwargs(item.sampling_params),
                    )
                else:
                    raise ValueError("mixed qualification batches are not supported")
                if output.predicted_token is None:
                    raise RuntimeError("tail stage did not sample a token")
                response = forward_pb2.Req(
                    rid=item.rid,
                    next_token_id=output.predicted_token,
                )
                incoming.respond(response.SerializeToString())
                handled += 1
                print(
                    f"CALL index={handled} mode={request.forward_mode} "
                    f"token={output.predicted_token} seconds={time.perf_counter() - started:.6f}",
                    flush=True,
                )
            except Exception as error:
                incoming.fail(str(error))
                raise
    finally:
        stage.drop_session(args.session_id)
        node.close()


def _forward_request(
    *,
    mode: int,
    session_id: str,
    token_ids: list[int],
    next_token_id: int | None,
    activation: Any,
) -> bytes:
    request = forward_pb2.ForwardRequest(forward_mode=mode)
    item = request.reqs.add(
        rid=session_id,
        input_ids=token_ids,
        next_token_id=next_token_id or 0,
    )
    item.native_activation.CopyFrom(_protobuf_frame(activation))
    item.sampling_params.temperature = 1.0
    item.sampling_params.top_p = 1.0
    item.sampling_params.top_k = 1
    item.sampling_params.repetition_penalty = 1.0
    return request.SerializeToString()


def run_head(args: argparse.Namespace) -> None:
    native, geometry, _ = _load_native(args)
    args.layer_count = int(geometry.layer_count)
    stage = _stage(native, args, start=0, end=args.split_layer)
    node = _network_node(native, args)
    tokens: list[int] = []
    rpc_seconds: list[float] = []
    started = time.perf_counter()
    try:
        head = stage.prefill(args.session_id, args.prompt_token_ids, None, sample=False)
        rpc_started = time.perf_counter()
        wire = node.call(
            args.tail_endpoint,
            FORWARD_METHOD,
            _forward_request(
                mode=forward_pb2.EXTEND,
                session_id=args.session_id,
                token_ids=args.prompt_token_ids,
                next_token_id=None,
                activation=head.activation,
            ),
            timeout_seconds=args.rpc_timeout_seconds,
        )
        rpc_seconds.append(time.perf_counter() - rpc_started)
        response = forward_pb2.Req.FromString(bytes(wire))
        tokens.append(response.next_token_id)
        for _ in range(args.max_calls - 1):
            head = stage.decode(args.session_id, tokens[-1], None, sample=False)
            rpc_started = time.perf_counter()
            wire = node.call(
                args.tail_endpoint,
                FORWARD_METHOD,
                _forward_request(
                    mode=forward_pb2.DECODE,
                    session_id=args.session_id,
                    token_ids=[],
                    next_token_id=tokens[-1],
                    activation=head.activation,
                ),
                timeout_seconds=args.rpc_timeout_seconds,
            )
            rpc_seconds.append(time.perf_counter() - rpc_started)
            response = forward_pb2.Req.FromString(bytes(wire))
            tokens.append(response.next_token_id)
        print("TOKENS=" + json.dumps(tokens))
        print("PATHS=" + node.paths(args.tail_endpoint))
        print(f"TOTAL_SECONDS={time.perf_counter() - started:.6f}")
        print(f"RPC_PREFILL_SECONDS={rpc_seconds[0]:.6f}")
        if len(rpc_seconds) > 1:
            average = sum(rpc_seconds[1:]) / len(rpc_seconds[1:])
            print(f"RPC_DECODE_AVERAGE_SECONDS={average:.6f}")
        print(f"SESSION_TOKENS={stage.session_token_count(args.session_id)}")
    finally:
        stage.drop_session(args.session_id)
        node.close()


def _parse_token_ids(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("prompt token IDs must be non-negative integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("tail", "head"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, action="append", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--backend-device", required=True)
    parser.add_argument("--relay-url", required=True)
    parser.add_argument("--relay-token-file", type=Path, required=True)
    parser.add_argument("--identity-path", type=Path, required=True)
    parser.add_argument("--split-layer", type=int, default=14)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--session-id", default="skippy-live-qualification")
    parser.add_argument("--mesh-release", default="0.74.0")
    parser.add_argument("--runtime-abi", default="0.1.32")
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--rpc-timeout-seconds", type=int, default=600)
    parser.add_argument("--force-relay", action="store_true")
    parser.add_argument("--endpoint-file", type=Path)
    parser.add_argument("--tail-endpoint")
    parser.add_argument(
        "--prompt-token-ids",
        type=_parse_token_ids,
        default=[785, 6722, 315, 9625, 374],
    )
    args = parser.parse_args()
    if args.context_tokens <= 0 or args.max_calls <= 0 or args.rpc_timeout_seconds <= 0:
        parser.error("context, calls, and timeout must be positive")
    if args.role == "tail" and args.endpoint_file is None:
        parser.error("tail requires --endpoint-file")
    if args.role == "head" and not args.tail_endpoint:
        parser.error("head requires --tail-endpoint")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.role == "tail":
        serve_tail(arguments)
    else:
        run_head(arguments)
