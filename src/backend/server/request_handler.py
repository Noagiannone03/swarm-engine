import asyncio
import time
from typing import Awaitable, Callable, Dict, List, Optional

import aiohttp
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from backend.server.constants import NODE_STATUS_AVAILABLE
from backend.server.context_admission import ContextRequestError
from backend.server.openai_compat import (
    decode_http_response_envelope,
    openai_error_response,
)
from parallax_utils.logging_config import get_logger
from parallax_utils.request_metrics import get_request_metrics

logger = get_logger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)
PARALLAX_ROUTING_TABLE_XARG = "parallax_routing_table"
PARALLAX_SCHEDULER_REQUEST_ID_XARG = "parallax_scheduler_request_id"


class ClientDisconnectedError(Exception):
    """Raised when the scheduler's HTTP client leaves before a reply is ready."""


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

    def _release_route(self, request_id: str) -> None:
        release = getattr(self.scheduler_manage, "release_routing_table", None)
        if release is not None:
            release(str(request_id))

    async def _wait_for_routing_capacity(self) -> None:
        wait = getattr(self.scheduler_manage, "wait_for_routing_capacity", None)
        if wait is None:
            await asyncio.sleep(self.RETRY_DELAY_SEC)
            return
        await asyncio.to_thread(wait, self.RETRY_DELAY_SEC)

    async def _next_chunk_until_disconnect(
        self,
        response,
        is_disconnected: Optional[Callable[[], Awaitable[bool]]],
    ) -> bytes:
        """Wait for one blocking RPC chunk while observing the inbound HTTP socket."""
        next_chunk = asyncio.create_task(anext(iterate_in_threadpool(response)))
        if is_disconnected is None:
            return await next_chunk

        try:
            while not next_chunk.done():
                if await is_disconnected():
                    response.cancel()
                    next_chunk.cancel()
                    raise ClientDisconnectedError
                await asyncio.sleep(0.1)
            return await next_chunk
        except BaseException:
            if not next_chunk.done():
                next_chunk.cancel()
            raise

    def _get_model_name_for_node(self, node_id: str) -> Optional[str]:
        try:
            scheduler = getattr(self.scheduler_manage, "scheduler", None)
            node = scheduler.get_node(node_id) if scheduler is not None else None
            if node is not None:
                if getattr(node.hardware, "device", None) == "mlx":
                    return node.model_info.mlx_model_name
                return node.model_info.model_name
        except Exception as e:
            logger.debug(f"Unable to resolve model name for node {node_id}: {e}")

        try:
            return self.scheduler_manage.get_model_name()
        except Exception:
            return None

    def _prepare_backend_request(
        self,
        request_data: Dict,
        request_id: str,
        routing_table: List[str],
    ) -> Dict:
        backend_request = dict(request_data)
        backend_request.pop("rid", None)
        backend_request.pop("routing_table", None)

        if not backend_request.get("request_id"):
            backend_request["request_id"] = str(request_id)

        model_name = self._get_model_name_for_node(routing_table[0])
        backend_request["model"] = model_name

        vllm_xargs = backend_request.get("vllm_xargs")
        if vllm_xargs is None:
            vllm_xargs = {}
        elif isinstance(vllm_xargs, dict):
            vllm_xargs = dict(vllm_xargs)
        else:
            logger.warning(
                "Ignoring non-object vllm_xargs for request %s; got %s",
                request_id,
                type(vllm_xargs).__name__,
            )
            vllm_xargs = {}

        vllm_xargs[PARALLAX_ROUTING_TABLE_XARG] = list(routing_table)
        vllm_xargs[PARALLAX_SCHEDULER_REQUEST_ID_XARG] = str(request_id)
        backend_request["vllm_xargs"] = vllm_xargs
        return backend_request

    async def _forward_request(
        self,
        request_data: Dict,
        request_id: str,
        received_ts: int,
        is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    ):
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

        required_context_tokens = 0
        build_budget = getattr(self.scheduler_manage, "build_context_budget", None)
        max_supported_context = getattr(self.scheduler_manage, "max_supported_context_tokens", None)
        if build_budget is not None and max_supported_context is not None:
            try:
                budget = await asyncio.to_thread(build_budget, request_data)
            except ContextRequestError as exc:
                return openai_error_response(
                    str(exc),
                    status_code=400,
                    err_type="invalid_request_error",
                    param="messages",
                    code="context_validation_error",
                )
            except Exception as exc:
                logger.exception("Unable to compute request context budget: %s", exc)
                return openai_error_response(
                    "The model tokenizer is temporarily unavailable",
                    status_code=503,
                    err_type="server_unavailable",
                    code="context_tokenizer_unavailable",
                )

            required_context_tokens = budget.required_tokens
            route_context_limit = max_supported_context()
            if route_context_limit <= 0:
                return openai_error_response(
                    "No context-capable pipeline is ready",
                    status_code=503,
                    err_type="server_unavailable",
                    code="context_route_not_ready",
                )
            if required_context_tokens > route_context_limit:
                return openai_error_response(
                    (
                        f"This request requires {required_context_tokens} tokens "
                        f"({budget.prompt_tokens} prompt + "
                        f"{budget.max_output_tokens} maximum output), but the largest "
                        f"available pipeline supports {route_context_limit} tokens."
                    ),
                    status_code=400,
                    err_type="invalid_request_error",
                    param="messages",
                    code="context_length_exceeded",
                )

        # Try to get a success response
        forward_attempts = 0
        while forward_attempts < self.MAX_FORWARD_RETRY:
            # Try to resolve routing; retry if table is an empty list (capacity full)
            attempts = 0
            routing_table = None
            while attempts < self.MAX_ROUTING_RETRY:
                try:
                    routing_table = self.scheduler_manage.get_routing_table(
                        request_id,
                        received_ts,
                        required_context_tokens,
                    )
                    logger.debug(
                        f"get_routing_table for request {request_id} return: {routing_table} (attempt {attempts+1})"
                    )
                except Exception as e:
                    logger.exception(f"get_routing_table error: {e}")
                    return openai_error_response(
                        "Get routing table error",
                        status_code=500,
                        err_type="server_error",
                        code="routing_table_error",
                    )

                # None -> scheduler has not set yet; treat as hard error (no waiting here)
                if routing_table is None:
                    return openai_error_response(
                        "Routing pipelines not ready",
                        status_code=503,
                        err_type="server_unavailable",
                        code="routing_not_ready",
                    )

                # Non-empty -> proceed
                if len(routing_table) > 0:
                    break

                # Empty list -> capacity full now, retry after short delay
                attempts += 1
                if attempts < self.MAX_ROUTING_RETRY:
                    await self._wait_for_routing_capacity()

            # If still empty after retries, return 429 Too Many Requests
            if routing_table is not None and len(routing_table) == 0:
                return openai_error_response(
                    "All pipelines are busy or not ready. Please retry later.",
                    status_code=429,
                    err_type="rate_limit_error",
                    code="rate_limit_exceeded",
                )

            is_stream = request_data.get("stream", False)
            try:
                backend_request = self._prepare_backend_request(
                    request_data,
                    str(request_id),
                    routing_table,
                )
                stub = self.get_stub(routing_table[0])
                if is_stream:

                    async def stream_generator():
                        response = None
                        first_token_time = None
                        last_chunk = None
                        last_token_time = None
                        try:
                            response = stub.chat_completion(backend_request)
                            iterator = iterate_in_threadpool(response)
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
                            try:
                                if response is not None:
                                    response.cancel()
                            finally:
                                self._release_route(request_id)

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
                    try:
                        response = stub.chat_completion(backend_request)
                        content = await self._next_chunk_until_disconnect(
                            response,
                            is_disconnected,
                        )
                        decoded_response = decode_http_response_envelope(content)
                        if decoded_response is None:
                            status_code = 200
                            content_type = "application/json"
                            body = content
                        else:
                            status_code, content_type, body = decoded_response
                        logger.debug(f"Non-stream response completed for {request_id}")
                        return Response(
                            content=body,
                            status_code=status_code,
                            headers={"content-type": content_type},
                            media_type=None,
                        )
                    finally:
                        self._release_route(request_id)
            except ClientDisconnectedError:
                logger.info("Client disconnected before request %s completed", request_id)
                return Response(status_code=499)
            except Exception as e:
                self._release_route(request_id)
                forward_attempts += 1
                if forward_attempts < self.MAX_FORWARD_RETRY:
                    # small async delay before re-forwarding
                    await asyncio.sleep(self.FORWARD_DELAY_SEC)
                logger.warning(f"Error in _forward_request: {e}. Retry attemps {forward_attempts}")

        return openai_error_response(
            "Downstream request failed",
            status_code=502,
            err_type="upstream_error",
            code="upstream_error",
        )

    async def v1_chat_completions(
        self,
        request_data: Dict,
        request_id: str,
        received_ts: int,
        is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    ):
        return await self._forward_request(
            request_data,
            request_id,
            received_ts,
            is_disconnected,
        )
