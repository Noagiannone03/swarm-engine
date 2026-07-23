"""Safe Python RPC services over the native Iroh transport.

This module adapts the RPC method metadata already present on Parallax service
handlers without inheriting Lattica's transport or its pickle wire format.
"""

from __future__ import annotations

import inspect
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Iterator, TypeVar, get_type_hints

import msgpack
from google.protobuf.message import Message

_CODEC_NONE = 0
_CODEC_MSGPACK = 1
_CODEC_PROTOBUF = 2
_CODEC_BYTES = 3
_DEFAULT_MAX_PAYLOAD_BYTES = 512 * 1024 * 1024
_DEFAULT_RPC_TIMEOUT_SECONDS = 600.0
_DEFAULT_DISPATCH_WORKERS = 64
_DEFAULT_PENDING_REQUESTS = 1024

_T = TypeVar("_T")
_AUTHENTICATED_RPC_PEER_ID: ContextVar[str | None] = ContextVar(
    "fabi_authenticated_rpc_peer_id",
    default=None,
)


def authenticated_rpc_peer_id() -> str:
    """Return the Iroh endpoint authenticated for the current inbound RPC.

    The value comes from ``Connection.remote_id()`` in the native QUIC
    receiver, never from the application payload. Calls made through a
    transport that cannot provide this identity fail closed.
    """

    peer_id = _AUTHENTICATED_RPC_PEER_ID.get()
    if not peer_id:
        raise RuntimeError("the current RPC has no authenticated Iroh peer identity")
    return peer_id


def _native_module():
    try:
        import fabi_network_native
    except ImportError as error:  # pragma: no cover - depends on release packaging
        raise RuntimeError(
            "fabi-network-native is required; install the platform wheel shipped with Fabi"
        ) from error
    return fabi_network_native


def _is_protobuf_class(value: object) -> bool:
    return inspect.isclass(value) and issubclass(value, Message)


def _protobuf_parameter_type(method: Callable[..., Any]) -> type[Message] | None:
    try:
        hints = get_type_hints(method)
        signature = inspect.signature(method)
    except (NameError, TypeError, ValueError):
        return None
    parameters = list(signature.parameters.values())
    if len(parameters) != 1:
        return None
    candidate = hints.get(parameters[0].name)
    return candidate if _is_protobuf_class(candidate) else None


def _protobuf_return_type(method: Callable[..., Any]) -> type[Message] | None:
    try:
        candidate = get_type_hints(method).get("return")
    except (NameError, TypeError):
        return None
    return candidate if _is_protobuf_class(candidate) else None


def _encode_value(value: object) -> bytes:
    if value is None:
        return bytes([_CODEC_NONE])
    if isinstance(value, Message):
        return bytes([_CODEC_PROTOBUF]) + value.SerializeToString()
    if isinstance(value, bytes):
        return bytes([_CODEC_BYTES]) + value
    try:
        encoded = msgpack.packb(value, use_bin_type=True, strict_types=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"unsupported RPC value type: {type(value).__name__}") from error
    return bytes([_CODEC_MSGPACK]) + encoded


def _decode_value(payload: bytes, protobuf_type: type[Message] | None = None) -> object:
    if not payload:
        raise ValueError("RPC payload is missing its codec byte")
    codec, body = payload[0], payload[1:]
    if codec == _CODEC_NONE:
        if body:
            raise ValueError("null RPC payload must not contain a body")
        return None
    if codec == _CODEC_BYTES:
        return body
    if codec == _CODEC_PROTOBUF:
        if protobuf_type is None:
            raise TypeError("RPC protobuf payload has no declared message type")
        message = protobuf_type()
        message.ParseFromString(body)
        return message
    if codec == _CODEC_MSGPACK:
        if len(body) > _DEFAULT_MAX_PAYLOAD_BYTES:
            raise ValueError("MessagePack RPC payload exceeds the configured maximum")
        return msgpack.unpackb(
            body,
            raw=False,
            strict_map_key=False,
        )
    raise ValueError(f"unsupported RPC codec {codec}")


def _call_value(args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    if not args and not kwargs:
        return None
    if len(args) == 1 and not kwargs:
        return args[0]
    if args and not kwargs:
        return list(args)
    if kwargs and not args:
        return dict(kwargs)
    combined = dict(kwargs)
    combined.update({f"arg{index}": value for index, value in enumerate(args)})
    return combined


def _invoke(method: Callable[..., _T], value: object) -> _T:
    parameters = list(inspect.signature(method).parameters.values())
    if not parameters:
        return method()
    if len(parameters) == 1:
        return method(value)
    if isinstance(value, dict):
        return method(**value)
    if isinstance(value, (list, tuple)):
        return method(*value)
    return method(value)


@dataclass(frozen=True)
class _Method:
    full_name: str
    function: Callable[..., Any]
    request_protobuf: type[Message] | None
    response_protobuf: type[Message] | None
    streaming: bool


def _service_methods(handler: object) -> dict[str, _Method]:
    service_name = type(handler).__name__
    methods: dict[str, _Method] = {}
    for name in dir(type(handler)):
        class_function = getattr(type(handler), name, None)
        if not getattr(class_function, "_is_rpc_method", False):
            continue
        function = getattr(handler, name)
        full_name = f"{service_name}.{name}"
        methods[full_name] = _Method(
            full_name=full_name,
            function=function,
            request_protobuf=_protobuf_parameter_type(function),
            response_protobuf=_protobuf_return_type(function),
            streaming=bool(getattr(class_function, "_is_stream_iter_method", False)),
        )
    if not methods:
        raise ValueError(f"{service_name} does not declare any RPC methods")
    return methods


class _CapacityLimiter:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("pending request capacity must be greater than zero")
        self._semaphore = threading.BoundedSemaphore(capacity)

    def submit(
        self,
        executor: ThreadPoolExecutor,
        function: Callable[..., _T],
        *args: object,
    ) -> Future[_T] | None:
        if not self._semaphore.acquire(blocking=False):
            return None
        try:
            future = executor.submit(function, *args)
        except BaseException:
            self._semaphore.release()
            raise
        future.add_done_callback(lambda _: self._semaphore.release())
        return future


class _DecodedRpcStream(Iterator[object]):
    def __init__(self, native_stream: object, protobuf_type: type[Message] | None):
        self._native_stream = native_stream
        self._protobuf_type = protobuf_type

    def __iter__(self) -> _DecodedRpcStream:
        return self

    def __next__(self) -> object:
        return _decode_value(next(self._native_stream), self._protobuf_type)

    def cancel(self) -> None:
        self._native_stream.cancel()

    @property
    def closed(self) -> bool:
        return bool(self._native_stream.closed)


class RpcServiceStub:
    """Dynamic client for one registered service on a remote endpoint."""

    def __init__(
        self,
        runtime: IrohRpcRuntime,
        peer_id: str,
        service: object | type[object],
    ):
        self._runtime = runtime
        self._peer_id = peer_id
        service_type = service if inspect.isclass(service) else type(service)
        self._service_name = service_type.__name__
        self._method_cache: dict[str, Callable[..., object]] = {}
        self._methods = _service_methods_from_type(service_type)

    def __getattr__(self, name: str) -> Callable[..., object]:
        cached = self._method_cache.get(name)
        if cached is not None:
            return cached
        method = self._methods.get(name)
        if method is None:
            raise AttributeError(f"{self._service_name} has no RPC method {name}")

        def call(*args: object, **kwargs: object) -> object:
            body = _encode_value(_call_value(args, kwargs))
            if method.streaming:
                native_stream = self._runtime._node.call_stream(
                    self._peer_id,
                    method.full_name,
                    body,
                    self._runtime.timeout_seconds,
                )
                return _DecodedRpcStream(native_stream, method.response_protobuf)

            def invoke() -> object:
                response = self._runtime._node.call(
                    self._peer_id,
                    method.full_name,
                    body,
                    self._runtime.timeout_seconds,
                )
                return _decode_value(response, method.response_protobuf)

            return self._runtime._outbound.submit(invoke)

        self._method_cache[name] = call
        return call


def _service_methods_from_type(service_type: type[object]) -> dict[str, _Method]:
    service_name = service_type.__name__
    methods: dict[str, _Method] = {}
    for name in dir(service_type):
        function = getattr(service_type, name, None)
        if not getattr(function, "_is_rpc_method", False):
            continue
        full_name = f"{service_name}.{name}"
        methods[name] = _Method(
            full_name=full_name,
            function=function,
            request_protobuf=_protobuf_parameter_type(function),
            response_protobuf=_protobuf_return_type(function),
            streaming=bool(getattr(function, "_is_stream_iter_method", False)),
        )
    if not methods:
        raise ValueError(f"{service_name} does not declare any RPC methods")
    return methods


class IrohRpcRuntime:
    """Own an Iroh endpoint, registered services and bounded dispatch pools."""

    def __init__(
        self,
        identity_path: str | Path,
        relay_url: str,
        relay_token: str | None = None,
        *,
        force_relay: bool = False,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
        timeout_seconds: float = _DEFAULT_RPC_TIMEOUT_SECONDS,
        dispatch_workers: int = _DEFAULT_DISPATCH_WORKERS,
        max_pending_requests: int = _DEFAULT_PENDING_REQUESTS,
    ):
        if dispatch_workers <= 0:
            raise ValueError("dispatch_workers must be greater than zero")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        native = _native_module()
        self.timeout_seconds = float(timeout_seconds)
        self._node = native.NetworkNode(
            Path(identity_path),
            relay_url,
            relay_token,
            force_relay,
            max_payload_bytes,
            max(1, int(timeout_seconds)),
        )
        self._methods: dict[str, _Method] = {}
        self._methods_lock = threading.Lock()
        self._stop = threading.Event()
        self._dispatch = ThreadPoolExecutor(
            max_workers=dispatch_workers,
            thread_name_prefix="fabi-iroh-rpc",
        )
        self._outbound = ThreadPoolExecutor(
            max_workers=dispatch_workers,
            thread_name_prefix="fabi-iroh-client",
        )
        self._capacity = _CapacityLimiter(max_pending_requests)
        self._receiver = threading.Thread(
            target=self._receive_loop,
            name="fabi-iroh-receiver",
            daemon=True,
        )
        self._receiver.start()

    @property
    def endpoint_id(self) -> str:
        return self._node.endpoint_id

    def register(self, handler: object) -> None:
        methods = _service_methods(handler)
        with self._methods_lock:
            duplicates = self._methods.keys() & methods.keys()
            if duplicates:
                raise ValueError(f"RPC methods already registered: {sorted(duplicates)}")
            self._methods.update(methods)

    def stub(self, peer_id: str, service: object | type[object]) -> RpcServiceStub:
        return RpcServiceStub(self, peer_id, service)

    def paths(self, peer_id: str) -> str:
        return self._node.paths(peer_id)

    def peers(self) -> list[str]:
        return list(self._node.peers())

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._node.close()
        self._receiver.join(timeout=2.0)
        self._dispatch.shutdown(wait=True, cancel_futures=True)
        self._outbound.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> IrohRpcRuntime:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                request = self._node.recv(1.0)
            except TimeoutError:
                continue
            except RuntimeError:
                if not self._stop.is_set():
                    self._stop.set()
                return
            with self._methods_lock:
                method = self._methods.get(request.method)
            if method is None:
                request.fail(f"unknown RPC method {request.method}")
                continue
            submitted = self._capacity.submit(
                self._dispatch, self._dispatch_request, request, method
            )
            if submitted is None:
                request.fail("RPC dispatcher is at capacity")

    @staticmethod
    def _dispatch_request(request: object, method: _Method) -> None:
        iterator = None
        peer_token = _AUTHENTICATED_RPC_PEER_ID.set(
            str(peer_id) if (peer_id := getattr(request, "peer_id", None)) else None
        )
        try:
            value = _decode_value(request.body, method.request_protobuf)
            result = _invoke(method.function, value)
            if not method.streaming:
                request.respond(_encode_value(result))
                return
            iterator = iter(result)
            for chunk in iterator:
                request.send_chunk(_encode_value(chunk))
            request.finish()
        except BaseException as error:
            try:
                request.fail(f"{type(error).__name__}: {error}")
            except RuntimeError:
                pass
        finally:
            _AUTHENTICATED_RPC_PEER_ID.reset(peer_token)
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
