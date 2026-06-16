import asyncio
import os
import time
from typing import Dict, List, Optional, Tuple

import aiohttp
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from backend.server.constants import NODE_STATUS_AVAILABLE
from backend.server.openai_compat import openai_error_payload, openai_error_response
from parallax_utils.logging_config import get_logger
from parallax_utils.request_metrics import get_request_metrics
from parallax_utils.stream_resume import DONE_BYTES, ResumableSSEStream, terminal_chunk_from

logger = get_logger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)

# Hard cap for the synchronous lattica `response.cancel()`. When the upstream
# worker drops mid-stream, libp2p teardown waits for an ACK that never comes
# and `cancel()` blocks forever — freezing the scheduler's asyncio loop and,
# transitively, every other client request. We run the call in a worker thread
# and abandon it after this many seconds.
UPSTREAM_CANCEL_TIMEOUT_SEC = 2.0

# These upstream first-chunk bodies mean "worker accepted the connection but its
# inner HTTP server isn't ready yet" — treat as a failed attempt and re-route.
_INVALID_FIRST_CHUNKS = ("internal server error", "Not found.")


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "off", "no")


# Bound the wait for the first streamed chunk. Without this a worker that accepts
# the connection but hangs during prefill would wedge the request for AIOHTTP's
# 20h timeout; instead we time out and let the forward-retry loop re-route.
FIRST_CHUNK_TIMEOUT_SEC = _env_float("PARALLAX_FIRST_CHUNK_TIMEOUT_SEC", 120.0)

# Mid-generation failover (port of Petals' inference_session recovery): if the
# pipeline drops AFTER tokens were streamed, re-route to another pipeline and
# resume — suppressing the regenerated prefix so the client sees no duplication.
# Default on; degrades gracefully to ending the stream if anything goes wrong.
STREAM_FAILOVER_ENABLED = _env_bool("PARALLAX_STREAM_FAILOVER", True)
MAX_STREAM_RESUMES = _env_int("PARALLAX_STREAM_MAX_RESUMES", 3)
# Brief pause before re-routing a resume, to let the failure propagate (peer
# ban / heartbeat) so we don't immediately re-pick the same broken pipeline.
RESUME_DELAY_SEC = _env_float("PARALLAX_STREAM_RESUME_DELAY_SEC", 1.0)
_VALID_INCOMPLETE_FINISH_REASONS = {"stop", "length", "content_filter"}
INCOMPLETE_STREAM_FINISH_REASON = (
    os.environ.get("PARALLAX_INCOMPLETE_STREAM_FINISH_REASON", "length").strip() or "length"
)
if INCOMPLETE_STREAM_FINISH_REASON not in _VALID_INCOMPLETE_FINISH_REASONS:
    logger.warning(
        "Ignoring PARALLAX_INCOMPLETE_STREAM_FINISH_REASON=%r; using 'length'",
        INCOMPLETE_STREAM_FINISH_REASON,
    )
    INCOMPLETE_STREAM_FINISH_REASON = "length"


class _NoRoute(Exception):
    """No routable pipeline — carries the HTTP response to return to the client."""

    def __init__(self, status_code: int, content: dict):
        error = content.get("error", "no route")
        if isinstance(error, dict):
            error = error.get("message", "no route")
        super().__init__(error)
        self.status_code = status_code
        self.content = content


async def _safe_cancel_upstream(response, request_id: str) -> None:
    """Cancel a lattica streaming response without blocking the event loop.

    The lattica RPC iterator's `.cancel()` is synchronous and can hang
    indefinitely when the remote peer has already disappeared. Run it in a
    thread with a timeout; log and move on if it doesn't return in time.
    """
    if response is None:
        return
    try:
        await asyncio.wait_for(
            asyncio.to_thread(response.cancel),
            timeout=UPSTREAM_CANCEL_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Timeout cancelling upstream response for %s after %.1fs; abandoning",
            request_id,
            UPSTREAM_CANCEL_TIMEOUT_SEC,
        )
    except Exception as exc:
        logger.warning(
            "Failed to cancel upstream response for %s: %s",
            request_id,
            exc,
        )


class RequestHandler:
    """HTTP request forwarder with scheduler-aware routing and retry logic.

    Behavior for routing resolution:
    - routing_table is None: scheduler has not decided yet -> treat as error for this attempt
    - routing_table is []: all pipelines are full now -> retry up to max attempts
    - routing_table is non-empty: forward to first hop

    On top of the pre-first-token retry (re-route on connect/first-chunk failure),
    streaming responses are *resumable*: a pipeline that drops mid-generation is
    re-routed and the answer continues without the client seeing a duplicate or a
    truncated reply (see parallax_utils.stream_resume).
    """

    MAX_FORWARD_RETRY = 10
    MAX_ROUTING_RETRY = 20
    FORWARD_DELAY_SEC = 10
    RETRY_DELAY_SEC = 5

    def __init__(self):
        self.scheduler_manage = None
        self.stubs = {}

    def set_scheduler_manage(self, scheduler_manage):
        self.scheduler_manage = scheduler_manage

    def get_stub(self, node_id):
        if node_id not in self.stubs:
            self.stubs[node_id] = self.scheduler_manage.completion_handler.get_stub(node_id)
        return self.stubs[node_id]

    async def _resolve_routing(
        self, request_id: str, received_ts: int, *, max_attempts: Optional[int] = None
    ) -> List[str]:
        """Resolve a non-empty routing table, retrying while pipelines are full.

        Raises `_NoRoute` (carrying the client-facing response) when the scheduler
        has no route: None -> 503, repeated empties -> 429, lookup error -> 500.
        """
        limit = self.MAX_ROUTING_RETRY if max_attempts is None else max_attempts
        attempts = 0
        while attempts < limit:
            try:
                routing_table = self.scheduler_manage.get_routing_table(request_id, received_ts)
            except Exception as e:
                logger.exception(f"get_routing_table error: {e}")
                raise _NoRoute(
                    500,
                    openai_error_payload(
                        "Get routing table error",
                        err_type="server_error",
                        code="routing_table_error",
                    ),
                )
            logger.debug(
                f"get_routing_table for request {request_id} return: {routing_table} "
                f"(attempt {attempts + 1})"
            )
            # None -> scheduler has not set yet; treat as hard error (no waiting here)
            if routing_table is None:
                raise _NoRoute(
                    503,
                    openai_error_payload(
                        "Routing pipelines not ready",
                        err_type="server_unavailable",
                        code="routing_not_ready",
                    ),
                )
            if len(routing_table) > 0:
                return routing_table
            # Empty list -> capacity full now, retry after short delay
            attempts += 1
            if attempts < limit:
                await asyncio.sleep(self.RETRY_DELAY_SEC)
        raise _NoRoute(
            429,
            openai_error_payload(
                "All pipelines are busy or not ready. Please retry later.",
                err_type="rate_limit_error",
                code="rate_limit_exceeded",
            ),
        )

    async def _open_stream(
        self, request_data: Dict, request_id: str, received_ts: int, *, routing_max_attempts=None
    ) -> Tuple[object, object, bytes]:
        """Resolve a route, open the upstream stream, and read a valid first chunk.

        Returns `(response, iterator, first_chunk)`. The caller owns `response`
        and must cancel it. Raises on any failure (so retry/resume loops re-route).
        """
        routing_table = await self._resolve_routing(
            request_id, received_ts, max_attempts=routing_max_attempts
        )
        request_data["rid"] = str(request_id)
        request_data["routing_table"] = routing_table
        stub = self.get_stub(routing_table[0])
        response = stub.chat_completion(request_data)
        iterator = iterate_in_threadpool(response)
        try:
            first_chunk = await asyncio.wait_for(
                anext(iterator), timeout=FIRST_CHUNK_TIMEOUT_SEC
            )
        except (asyncio.TimeoutError, StopAsyncIteration) as e:
            await _safe_cancel_upstream(response, request_id)
            raise RuntimeError(f"no first chunk from upstream ({type(e).__name__})") from e

        first_text = first_chunk.decode("utf-8", errors="replace").strip()
        if first_text in _INVALID_FIRST_CHUNKS or first_text.startswith('{"detail":"Not Found"'):
            await _safe_cancel_upstream(response, request_id)
            raise RuntimeError(f"upstream worker returned invalid stream response: {first_text}")
        return response, iterator, first_chunk

    async def _forward_request(self, request_data: Dict, request_id: str, received_ts: int):
        start_time = time.time()
        logger.debug(f"Forwarding request {request_id}; stream={request_data.get('stream', False)}")
        if (
            self.scheduler_manage is None
            or not self.scheduler_manage.get_schedule_status() == NODE_STATUS_AVAILABLE
        ):
            return openai_error_response(
                "Server is not ready",
                status_code=503,
                err_type="server_unavailable",
                code="server_not_ready",
            )

        is_stream = request_data.get("stream", False)

        forward_attempts = 0
        while forward_attempts < self.MAX_FORWARD_RETRY:
            response = None
            try:
                if is_stream:
                    response, iterator, first_chunk = await self._open_stream(
                        request_data, request_id, received_ts
                    )
                    # Ownership of `response` transfers to the streaming generator.
                    resp = StreamingResponse(
                        self._resilient_stream(
                            request_data,
                            request_id,
                            received_ts,
                            start_time,
                            response,
                            iterator,
                            first_chunk,
                        ),
                        media_type="text/event-stream",
                        headers={
                            "X-Content-Type-Options": "nosniff",
                            "Cache-Control": "no-cache",
                        },
                    )
                    logger.debug(f"Streaming response initiated for {request_id}")
                    return resp

                # Non-streaming path.
                routing_table = await self._resolve_routing(request_id, received_ts)
                request_data["rid"] = str(request_id)
                request_data["routing_table"] = routing_table
                stub = self.get_stub(routing_table[0])
                response = stub.chat_completion(request_data)
                content = (
                    await asyncio.wait_for(
                        anext(iterate_in_threadpool(response)), timeout=FIRST_CHUNK_TIMEOUT_SEC
                    )
                ).decode()
                if content.strip() == "internal server error":
                    raise RuntimeError("upstream worker returned internal server error")
                logger.debug(f"Non-stream response completed for {request_id}")
                return Response(content=content, media_type="application/json")
            except _NoRoute as nr:
                return JSONResponse(content=nr.content, status_code=nr.status_code)
            except Exception as e:
                # Release the upstream stream if we still hold it (failure before
                # ownership transferred to the streaming generator).
                if response is not None:
                    await _safe_cancel_upstream(response, request_id)
                    response = None
                forward_attempts += 1
                if forward_attempts < self.MAX_FORWARD_RETRY:
                    await asyncio.sleep(self.FORWARD_DELAY_SEC)
                logger.warning(f"Error in _forward_request: {e}. Retry attempts {forward_attempts}")

        return openai_error_response(
            "Downstream request failed",
            status_code=502,
            err_type="upstream_error",
            code="upstream_error",
        )

    async def _resilient_stream(
        self,
        request_data: Dict,
        request_id: str,
        received_ts: int,
        start_time: float,
        response,
        iterator,
        first_chunk: bytes,
    ):
        """Stream the answer, transparently re-routing if the pipeline drops mid-generation.

        Nominal chunks are forwarded byte-for-byte. On a mid-stream break (iterator
        ends or errors before a finish_reason / [DONE]), and while the answer is a
        plain-text continuation (no tool-calls), we open a replacement upstream and
        resume — emitting only the genuinely-new tail. Anything unexpected ends the
        stream cleanly, i.e. never worse than the pre-failover behavior.
        """
        resumable = ResumableSSEStream()
        first_token_time: Optional[float] = None
        last_token_time: Optional[float] = None
        last_chunk: Optional[bytes] = first_chunk
        streamed_response = response
        resumes_left = MAX_STREAM_RESUMES if STREAM_FAILOVER_ENABLED else 0

        async def _reopen():
            """Cancel the broken upstream and open a replacement. Mutates streamed_response."""
            nonlocal streamed_response
            await _safe_cancel_upstream(streamed_response, request_id)
            streamed_response = None
            if RESUME_DELAY_SEC > 0:
                await asyncio.sleep(RESUME_DELAY_SEC)
            new_resp, new_iter, new_first = await self._open_stream(
                request_data, request_id, received_ts, routing_max_attempts=3
            )
            streamed_response = new_resp
            resumable.begin_resume()
            return new_iter, new_first

        try:
            # Emit the first chunk and the nominal stream (raw passthrough).
            for out in resumable.feed_original(first_chunk):
                yield out

            cur_iter = iterator
            while True:
                stream_error: Optional[Exception] = None
                try:
                    async for chunk in cur_iter:
                        last_token_time = time.time()
                        if first_token_time is None:
                            first_token_time = last_token_time
                        for out in resumable.feed_original(chunk):
                            if out is not None and not out.decode(
                                "utf-8", errors="replace"
                            ).startswith("data: [DONE]"):
                                last_chunk = out
                            yield out
                except Exception as e:
                    stream_error = e

                if resumable.completed:
                    break
                if not (resumable.should_resume() and resumes_left > 0):
                    logger.warning(
                        "stream for %s ended unrecovered: %s",
                        request_id,
                        stream_error if stream_error is not None else "EOF before completion",
                    )
                    break

                # --- mid-generation failover (Petals-style resume) ---
                resumes_left -= 1
                logger.warning(
                    "stream for %s broke mid-generation (%s); resuming, %d attempt(s) left",
                    request_id,
                    f"error: {stream_error}" if stream_error else "EOF before completion",
                    resumes_left,
                )
                try:
                    cur_iter, new_first = await _reopen()
                except Exception as e:
                    logger.warning("resume re-open failed for %s; ending stream: %s", request_id, e)
                    break
                for out in resumable.feed_resumed(new_first):
                    last_chunk = out
                    yield out
                if not resumable.resume_safe:
                    # Replacement produced something we can't safely splice (tool-calls).
                    logger.warning("resume for %s became unsafe; ending stream", request_id)
                    break
                # Continue the outer loop, now consuming the replacement iterator.

            # If we exited without ever forwarding a terminator, close cleanly.
            if not resumable.completed:
                terminal = terminal_chunk_from(
                    last_chunk,
                    finish_reason=INCOMPLETE_STREAM_FINISH_REASON,
                ) if last_chunk is not None else None
                if terminal is not None:
                    yield terminal
                else:
                    logger.warning(
                        "stream for %s ended incomplete without a JSON chunk template",
                        request_id,
                    )
                yield DONE_BYTES
        finally:
            if last_chunk is not None:
                tps, ttft, input_tokens, output_tokens = get_request_metrics(
                    last_chunk, start_time, first_token_time, last_token_time
                )
                if (
                    tps is not None
                    and ttft is not None
                    and input_tokens is not None
                    and output_tokens is not None
                ):
                    logger.info(
                        f"Request ID: {request_id} | TPS: {tps:.2f} |  TTFT: {ttft} ms | "
                        f"Output tokens: {output_tokens} | Input tokens: {input_tokens}"
                    )
            logger.debug(f"stream finished for {request_id}")
            await _safe_cancel_upstream(streamed_response, request_id)

    async def v1_chat_completions(self, request_data: Dict, request_id: str, received_ts: int):
        return await self._forward_request(request_data, request_id, received_ts)
