import asyncio
import time
from typing import Dict

import aiohttp
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from backend.server.constants import NODE_STATUS_AVAILABLE
from parallax_utils.logging_config import get_logger
from parallax_utils.request_metrics import get_request_metrics

logger = get_logger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)

# Hard cap for the synchronous lattica `response.cancel()`. When the upstream
# worker drops mid-stream, libp2p teardown waits for an ACK that never comes
# and `cancel()` blocks forever — freezing the scheduler's asyncio loop and,
# transitively, every other client request. We run the call in a worker thread
# and abandon it after this many seconds.
UPSTREAM_CANCEL_TIMEOUT_SEC = 2.0


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

    async def _forward_request(self, request_data: Dict, request_id: str, received_ts: int):
        start_time = time.time()
        logger.debug(f"Forwarding request {request_id}; stream={request_data.get('stream', False)}")
        if (
            self.scheduler_manage is None
            or not self.scheduler_manage.get_schedule_status() == NODE_STATUS_AVAILABLE
        ):
            return JSONResponse(
                content={"error": "Server is not ready"},
                status_code=500,
            )

        # Try to get a success response
        forward_attempts = 0
        while forward_attempts < self.MAX_FORWARD_RETRY:
            # Try to resolve routing; retry if table is an empty list (capacity full)
            attempts = 0
            routing_table = None
            while attempts < self.MAX_ROUTING_RETRY:
                try:
                    routing_table = self.scheduler_manage.get_routing_table(request_id, received_ts)
                    logger.debug(
                        f"get_routing_table for request {request_id} return: {routing_table} (attempt {attempts+1})"
                    )
                except Exception as e:
                    logger.exception(f"get_routing_table error: {e}")
                    return JSONResponse(
                        content={"error": "Get routing table error"},
                        status_code=500,
                    )

                # None -> scheduler has not set yet; treat as hard error (no waiting here)
                if routing_table is None:
                    return JSONResponse(
                        content={"error": "Routing pipelines not ready"},
                        status_code=503,
                    )

                # Non-empty -> proceed
                if len(routing_table) > 0:
                    break

                # Empty list -> capacity full now, retry after short delay
                attempts += 1
                if attempts < self.MAX_ROUTING_RETRY:
                    # small async delay before re-forwarding
                    await asyncio.sleep(self.RETRY_DELAY_SEC)

            # If still empty after retries, return 429 Too Many Requests
            if routing_table is not None and len(routing_table) == 0:
                return JSONResponse(
                    content={"error": "All pipelines are busy or not ready. Please retry later."},
                    status_code=429,
                )

            # Add request_id and routing_table to request_data
            request_data["rid"] = str(request_id)
            request_data["routing_table"] = routing_table
            stub = self.get_stub(routing_table[0])
            is_stream = request_data.get("stream", False)
            response = None
            try:
                if is_stream:
                    response = stub.chat_completion(request_data)
                    iterator = iterate_in_threadpool(response)
                    first_chunk = await anext(iterator)
                    first_chunk_text = first_chunk.decode("utf-8", errors="replace").strip()
                    if (
                        first_chunk_text == "internal server error"
                        or first_chunk_text == "Not found."
                        or first_chunk_text.startswith('{"detail":"Not Found"')
                    ):
                        # Worker accepted the connection but its inner HTTP
                        # server is not ready (model still loading) — release
                        # the upstream stream before retrying with another peer.
                        await _safe_cancel_upstream(response, request_id)
                        response = None
                        raise RuntimeError(
                            f"upstream worker returned invalid stream response: {first_chunk_text}"
                        )

                    streamed_response = response

                    async def stream_generator():
                        first_token_time = None
                        last_chunk = first_chunk
                        last_token_time = None
                        try:
                            yield first_chunk
                            async for chunk in iterator:
                                last_token_time = time.time()
                                if first_token_time is None:
                                    first_token_time = last_token_time
                                if chunk is not None and not chunk.decode("utf-8").startswith(
                                    "data: [DONE]"
                                ):
                                    last_chunk = chunk
                                yield chunk
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
                                        f"Request ID: {request_id} | TPS: {tps:.2f} |  TTFT: {ttft} ms | Output tokens: {output_tokens} | Input tokens: {input_tokens}"
                                    )
                            logger.debug(f"client disconnected for {request_id}")
                            await _safe_cancel_upstream(streamed_response, request_id)

                    # Ownership of the upstream stream is now transferred to
                    # the generator's finally block.
                    response = None

                    resp = StreamingResponse(
                        stream_generator(),
                        media_type="text/event-stream",
                        headers={
                            "X-Content-Type-Options": "nosniff",
                            "Cache-Control": "no-cache",
                        },
                    )
                    logger.debug(f"Streaming response initiated for {request_id}")
                    return resp
                else:
                    response = stub.chat_completion(request_data)
                    content = (await anext(iterate_in_threadpool(response))).decode()
                    if content.strip() == "internal server error":
                        raise RuntimeError("upstream worker returned internal server error")
                    logger.debug(f"Non-stream response completed for {request_id}")
                    return Response(content=content, media_type="application/json")
            except Exception as e:
                # If we still hold the upstream stream (early failure before
                # ownership was transferred to stream_generator), release it.
                if response is not None:
                    await _safe_cancel_upstream(response, request_id)
                    response = None
                forward_attempts += 1
                if forward_attempts < self.MAX_FORWARD_RETRY:
                    # small async delay before re-forwarding
                    await asyncio.sleep(self.FORWARD_DELAY_SEC)
                logger.warning(f"Error in _forward_request: {e}. Retry attemps {forward_attempts}")

        return JSONResponse(
            content={"error": "Internal server error"},
            status_code=500,
        )

    async def v1_chat_completions(self, request_data: Dict, request_id: str, received_ts: int):
        return await self._forward_request(request_data, request_id, received_ts)
