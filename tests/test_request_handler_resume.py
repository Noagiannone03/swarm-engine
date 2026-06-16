"""Integration test for the resilient streaming orchestration in RequestHandler.

The web deps (aiohttp / fastapi / starlette) aren't needed for the logic under
test, so we stub them in sys.modules before importing the handler. This drives
the real async generator `_resilient_stream` end-to-end: a stream that breaks
mid-generation must re-route and resume with no duplicated or lost text.
"""

import asyncio
import json
import sys
import types


# --------------------------------------------------------------------------- #
# Stub the heavy web deps so request_handler imports without them.            #
# --------------------------------------------------------------------------- #
def _install_stubs():
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.ClientTimeout = lambda **kw: None
    sys.modules.setdefault("aiohttp", aiohttp)

    fastapi = types.ModuleType("fastapi")
    responses = types.ModuleType("fastapi.responses")

    class _Resp:
        def __init__(self, *a, **kw):
            self.args, self.kwargs = a, kw

    class StreamingResponse(_Resp):
        def __init__(self, gen, **kw):
            super().__init__(**kw)
            self.body_iterator = gen

    responses.JSONResponse = _Resp
    responses.Response = _Resp
    responses.StreamingResponse = StreamingResponse
    fastapi.responses = responses
    sys.modules.setdefault("fastapi", fastapi)
    sys.modules.setdefault("fastapi.responses", responses)

    starlette = types.ModuleType("starlette")
    concurrency = types.ModuleType("starlette.concurrency")

    async def iterate_in_threadpool(sync_iterable):
        for item in sync_iterable:
            yield item

    concurrency.iterate_in_threadpool = iterate_in_threadpool
    starlette.concurrency = concurrency
    sys.modules.setdefault("starlette", starlette)
    sys.modules.setdefault("starlette.concurrency", concurrency)


_install_stubs()

import backend.server.request_handler as request_handler_module  # noqa: E402
from backend.server.request_handler import RequestHandler  # noqa: E402
from parallax_utils.stream_resume import delta_content, finish_reason, iter_sse_events  # noqa: E402


def _chunk(content=None, *, role=None, fr=None) -> bytes:
    obj = {
        "id": "rid",
        "object": "chat.completion.chunk",
        "model": "m",
        "created": 1,
        "choices": [{"index": 0, "finish_reason": fr, "delta": {"role": role, "content": content}}],
        "usage": {"prompt_tokens": 1, "total_tokens": 2, "completion_tokens": 1},
    }
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode()


class _Stub:
    """Fake worker stub: returns a sync generator of byte chunks per call."""

    def __init__(self, streams):
        self._streams = list(streams)
        self.calls = 0

    def chat_completion(self, request_data):
        stream = self._streams[self.calls]
        self.calls += 1
        return iter(stream)


def _text(chunks):
    out = ""
    for c in chunks:
        for kind, obj in iter_sse_events(c):
            if kind == "data":
                out += delta_content(obj)
    return out


async def _collect(agen):
    return [c async for c in agen]


def _make_handler(stub):
    h = RequestHandler()
    completion_handler = types.SimpleNamespace(get_stub=lambda node_id: stub)
    h.scheduler_manage = types.SimpleNamespace(
        get_routing_table=lambda rid, ts: ["replacement-head"],
        completion_handler=completion_handler,
    )
    return h


def test_resilient_stream_resumes_without_duplication():
    # Original stream: greets, then BREAKS (raises) before finishing.
    def original():
        yield _chunk("The capital ", role="assistant")
        yield _chunk("of France")
        raise ConnectionError("peer dropped mid-stream")

    # Replacement stream: regenerates the full answer from the prompt.
    replacement = [
        _chunk("The capital of France is Paris.", role="assistant"),
        _chunk(None, fr="stop"),
        b"data: [DONE]\n\n",
    ]
    stub = _Stub([replacement])  # the replacement is opened on resume
    handler = _make_handler(stub)

    first_chunk = _chunk("The capital ", role="assistant")

    async def run():
        # Build the "original" iterator the way _open_stream would have.
        from starlette.concurrency import iterate_in_threadpool

        def original_after_first():
            yield _chunk("of France")
            raise ConnectionError("peer dropped mid-stream")

        agen = handler._resilient_stream(
            request_data={"stream": True, "messages": []},
            request_id="rid",
            received_ts=0,
            start_time=0.0,
            response=object(),
            iterator=iterate_in_threadpool(original_after_first()),
            first_chunk=first_chunk,
        )
        return await _collect(agen)

    emitted = asyncio.run(run())
    full = _text(emitted)
    # The client must see the answer exactly once — prefix not duplicated.
    assert full == "The capital of France is Paris.", full
    assert emitted[-1] == b"data: [DONE]\n\n"
    assert stub.calls == 1  # exactly one replacement opened


def test_resilient_stream_passthrough_when_no_break():
    # A clean stream that completes normally must be forwarded unchanged.
    stub = _Stub([])  # never reopened
    handler = _make_handler(stub)
    first_chunk = _chunk("Hello", role="assistant")

    async def run():
        from starlette.concurrency import iterate_in_threadpool

        def rest():
            yield _chunk(" world")
            yield _chunk(None, fr="stop")
            yield b"data: [DONE]\n\n"

        agen = handler._resilient_stream(
            request_data={"stream": True},
            request_id="rid",
            received_ts=0,
            start_time=0.0,
            response=object(),
            iterator=iterate_in_threadpool(rest()),
            first_chunk=first_chunk,
        )
        return await _collect(agen)

    emitted = asyncio.run(run())
    assert _text(emitted) == "Hello world"
    assert emitted[-1] == b"data: [DONE]\n\n"
    assert stub.calls == 0  # no resume happened


def test_resilient_stream_closes_incomplete_stream_with_terminal_chunk():
    stub = _Stub([])  # never reopened; failover disabled below
    handler = _make_handler(stub)
    first_chunk = _chunk("partial", role="assistant")
    old_max_resumes = request_handler_module.MAX_STREAM_RESUMES
    request_handler_module.MAX_STREAM_RESUMES = 0

    async def run():
        from starlette.concurrency import iterate_in_threadpool

        def broken_rest():
            raise ConnectionError("peer dropped before finish")
            yield b""  # pragma: no cover

        agen = handler._resilient_stream(
            request_data={"stream": True},
            request_id="rid",
            received_ts=0,
            start_time=0.0,
            response=object(),
            iterator=iterate_in_threadpool(broken_rest()),
            first_chunk=first_chunk,
        )
        return await _collect(agen)

    try:
        emitted = asyncio.run(run())
    finally:
        request_handler_module.MAX_STREAM_RESUMES = old_max_resumes

    assert emitted[-1] == b"data: [DONE]\n\n"
    terminal_events = [
        obj
        for kind, obj in iter_sse_events(emitted[-2])
        if kind == "data"
    ]
    assert terminal_events
    assert finish_reason(terminal_events[0]) == "length"
    assert _text(emitted) == "partial"
