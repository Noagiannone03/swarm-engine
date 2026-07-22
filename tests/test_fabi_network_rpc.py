from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from lattica import rpc_method, rpc_stream, rpc_stream_iter

from fabi_network.rpc import (
    RpcServiceStub,
    _decode_value,
    _encode_value,
    _service_methods,
)
from fabi_network.transport import _relay_token, configured_transport
from parallax.p2p.proto import forward_pb2


class ExampleService:
    @rpc_method
    def echo(self, value):
        return value

    @rpc_stream
    def abort(self, request: forward_pb2.AbortRequest) -> forward_pb2.AbortResponse:
        assert request.reqs
        return forward_pb2.AbortResponse()

    @rpc_stream_iter
    def tokens(self, values):
        yield from values


def test_safe_codec_round_trips_supported_values():
    for value in (None, b"bytes", "text", {"nested": [1, True, None]}):
        assert _decode_value(_encode_value(value)) == value

    request = forward_pb2.AbortRequest()
    request.reqs.add().rid = "protobuf-request"
    assert _decode_value(_encode_value(request), forward_pb2.AbortRequest) == request


def test_safe_codec_rejects_implicit_or_unsupported_payloads():
    class Unsupported:
        pass

    with pytest.raises(TypeError, match="unsupported RPC value type"):
        _encode_value(Unsupported())
    with pytest.raises(ValueError, match="codec byte"):
        _decode_value(b"")
    with pytest.raises(ValueError, match="unsupported RPC codec"):
        _decode_value(b"\xffpayload")
    with pytest.raises(TypeError, match="no declared message type"):
        _decode_value(_encode_value(forward_pb2.AbortRequest()))


def test_service_discovery_preserves_unary_and_streaming_contracts():
    methods = _service_methods(ExampleService())
    assert methods["ExampleService.echo"].streaming is False
    assert methods["ExampleService.abort"].streaming is False
    assert methods["ExampleService.abort"].request_protobuf is forward_pb2.AbortRequest
    assert methods["ExampleService.abort"].response_protobuf is forward_pb2.AbortResponse
    assert methods["ExampleService.tokens"].streaming is True


class _FakeNativeStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def cancel(self):
        self.closed = True


class _FakeNode:
    def __init__(self):
        self.calls = []

    def call(self, peer_id, method, body, timeout):
        self.calls.append(("unary", peer_id, method, timeout))
        return body

    def call_stream(self, peer_id, method, body, timeout):
        self.calls.append(("stream", peer_id, method, timeout))
        values = _decode_value(body)
        return _FakeNativeStream([_encode_value(value) for value in values])


def test_service_stub_returns_futures_and_cancellable_iterators():
    node = _FakeNode()
    with ThreadPoolExecutor(max_workers=2) as outbound:
        runtime = SimpleNamespace(_node=node, _outbound=outbound, timeout_seconds=12.5)
        stub = RpcServiceStub(runtime, "peer", ExampleService)

        assert stub.echo({"answer": 42}).result(timeout=1) == {"answer": 42}
        stream = stub.tokens([b"a", b"b"])
        assert list(stream) == [b"a", b"b"]
        stream.cancel()
        assert stream.closed

    assert node.calls == [
        ("unary", "peer", "ExampleService.echo", 12.5),
        ("stream", "peer", "ExampleService.tokens", 12.5),
    ]


def test_transport_selection_and_protected_token_file(monkeypatch, tmp_path):
    monkeypatch.setenv("FABI_NETWORK_TRANSPORT", "IROH")
    assert configured_transport() == "iroh"
    monkeypatch.setenv("FABI_NETWORK_TRANSPORT", "invalid")
    with pytest.raises(ValueError, match="iroh.*lattica"):
        configured_transport()

    monkeypatch.delenv("FABI_RELAY_TOKEN", raising=False)
    token_file = tmp_path / "relay.env"
    token_file.write_text("IROH_RELAY_ACCESS_TOKEN=secret\n", encoding="utf-8")
    token_file.chmod(0o600)
    monkeypatch.setenv("FABI_RELAY_TOKEN_FILE", str(token_file))
    assert _relay_token() == "secret"

    token_file.chmod(0o644)
    if os.name == "nt":
        assert _relay_token() == "secret"
    else:
        with pytest.raises(PermissionError, match="group or others"):
            _relay_token()
