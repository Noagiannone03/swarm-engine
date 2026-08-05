import asyncio
import json
import time
from typing import Awaitable, Callable, Dict, List, Optional

import aiohttp
import anyio
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from backend.server.constants import NODE_STATUS_AVAILABLE
from backend.server.context_admission import ContextBudget, ContextRequestError
from backend.server.openai_compat import (
    decode_http_response_envelope,
    openai_error_payload,
    openai_error_response,
)
from backend.server.recovery_stream import (
    OpenAIChatReplayStream,
    OpenAIRecoveryStream,
    RecoveryStreamProtocolError,
)
from parallax_utils.logging_config import get_logger
from parallax_utils.request_metrics import get_request_metrics
from swarm_protocol.contracts import RecoveryLevel
from swarm_protocol.recovery import RecoveryConflict, RecoveryState

logger = get_logger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)
PARALLAX_ROUTING_TABLE_XARG = "parallax_routing_table"
PARALLAX_SCHEDULER_REQUEST_ID_XARG = "parallax_scheduler_request_id"
FABI_ROUTE_ID_XARG = "fabi_route_id"
FABI_ROUTE_EPOCH_XARG = "fabi_route_epoch"
ABORT_RPC_TIMEOUT_SEC = 5.0
TOKENIZE_RPC_TIMEOUT_SEC = 30.0
MAX_EXACT_TOKEN_REPLANS = 2


class ClientDisconnectedError(Exception):
    """Raised when the scheduler's HTTP client leaves before a reply is ready."""


class DownstreamRouteLostError(Exception):
    """Raised when a worker departure invalidates an in-flight scheduler route."""


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
            try:
                release(str(request_id))
            except Exception:
                # Route/permit cleanup is idempotent and their signed TTLs
                # still fail closed. A transient authority outage during
                # cleanup must not corrupt an already committed OpenAI stream.
                logger.warning(
                    "Unable to acknowledge route release for request %s",
                    request_id,
                    exc_info=True,
                )

    @staticmethod
    def _abort_backend_request(stub, backend_request: Dict) -> None:
        """Ask the route head to abort one still-active v3 engine request."""

        xargs = backend_request.get("vllm_xargs")
        if not isinstance(xargs, dict) or not xargs.get(FABI_ROUTE_ID_XARG):
            return
        abort_request = {
            "request_id": str(backend_request["request_id"]),
            "vllm_xargs": {
                PARALLAX_ROUTING_TABLE_XARG: list(xargs.get(PARALLAX_ROUTING_TABLE_XARG, ())),
                PARALLAX_SCHEDULER_REQUEST_ID_XARG: str(
                    xargs.get(PARALLAX_SCHEDULER_REQUEST_ID_XARG, "")
                ),
                FABI_ROUTE_ID_XARG: str(xargs[FABI_ROUTE_ID_XARG]),
                FABI_ROUTE_EPOCH_XARG: int(xargs.get(FABI_ROUTE_EPOCH_XARG, 0)),
            },
        }
        result = stub.abort_completion(abort_request)
        wait = getattr(result, "result", None)
        if wait is not None:
            wait(timeout=ABORT_RPC_TIMEOUT_SEC)

    async def _best_effort_abort_backend_request(self, stub, backend_request: Dict) -> None:
        """Propagate cancellation out of band before releasing its route fence."""

        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._abort_backend_request, stub, backend_request),
                timeout=ABORT_RPC_TIMEOUT_SEC,
            )
        except Exception:
            logger.warning(
                "Unable to explicitly abort backend request %s",
                backend_request.get("request_id"),
                exc_info=True,
            )

    def _route_is_active(self, request_id: str) -> bool:
        is_active = getattr(self.scheduler_manage, "is_routing_table_active", None)
        if is_active is None:
            # Preserve compatibility with scheduler managers that predate route
            # liveness monitoring. Current schedulers always expose the method.
            return True
        try:
            return bool(is_active(str(request_id)))
        except Exception:
            logger.exception("Unable to check scheduler route %s", request_id)
            return True

    async def _wait_for_routing_capacity(self) -> None:
        wait = getattr(self.scheduler_manage, "wait_for_routing_capacity", None)
        if wait is None:
            await asyncio.sleep(self.RETRY_DELAY_SEC)
            return
        await asyncio.to_thread(wait, self.RETRY_DELAY_SEC)

    async def _observe_unmet_context_demand(
        self,
        request_id: str,
        required_context_tokens: int,
    ) -> None:
        """Best-effort placement feedback, isolated from OpenAI admission."""

        observe = getattr(
            self.scheduler_manage,
            "observe_unmet_context_demand",
            None,
        )
        if observe is None:
            return
        try:
            await asyncio.to_thread(
                observe,
                str(request_id),
                int(required_context_tokens),
            )
        except Exception:
            logger.warning(
                "Unable to observe unmet context demand for request %s",
                request_id,
                exc_info=True,
            )

    async def _next_chunk_until_disconnect(
        self,
        response,
        is_disconnected: Optional[Callable[[], Awaitable[bool]]],
        request_id: str,
        iterator=None,
    ) -> bytes:
        """Wait for one RPC chunk while observing client and route liveness."""
        if iterator is None:
            iterator = iterate_in_threadpool(response)
        next_chunk = asyncio.create_task(anext(iterator))

        try:
            while not next_chunk.done():
                if is_disconnected is not None and await is_disconnected():
                    response.cancel()
                    next_chunk.cancel()
                    raise ClientDisconnectedError
                if not self._route_is_active(request_id):
                    response.cancel()
                    next_chunk.cancel()
                    raise DownstreamRouteLostError
                await asyncio.sleep(0.1)
            return await next_chunk
        except BaseException:
            if not next_chunk.done():
                next_chunk.cancel()
            raise

    @staticmethod
    def _stream_error_chunk(
        message: str,
        *,
        err_type: str,
        code: str,
    ) -> bytes:
        """Encode an OpenAI-compatible error as a server-sent event."""
        payload = openai_error_payload(
            message,
            err_type=err_type,
            code=code,
        )
        return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"

    @staticmethod
    def _stream_http_error_chunk(status_code: int, body: bytes) -> bytes:
        """Translate a downstream HTTP failure into one valid OpenAI SSE event.

        The RPC transport carries HTTP failures in a binary envelope. Returning
        the enclosed JSON bytes directly would concatenate them with the next
        ``data:`` line and make standards-compliant EventSource parsers silently
        discard both records.
        """

        fallback_type = "invalid_request_error" if status_code < 500 else "upstream_error"
        fallback_code = (
            "downstream_request_rejected" if status_code < 500 else "upstream_worker_error"
        )
        payload = None
        try:
            decoded = json.loads(body.decode("utf-8"))
            if isinstance(decoded, dict) and isinstance(decoded.get("error"), dict):
                error = decoded["error"]
                message = error.get("message")
                if isinstance(message, str) and message:
                    payload = openai_error_payload(
                        message,
                        err_type=(
                            error.get("type")
                            if isinstance(error.get("type"), str) and error["type"]
                            else fallback_type
                        ),
                        param=error.get("param") if isinstance(error.get("param"), str) else None,
                        code=(
                            error.get("code")
                            if isinstance(error.get("code"), str) and error["code"]
                            else fallback_code
                        ),
                    )
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        if payload is None:
            payload = openai_error_payload(
                f"A worker rejected the generation request (HTTP {status_code}).",
                err_type=fallback_type,
                code=fallback_code,
            )
        return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"

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

        # The scheduler owns the request identity used by route fencing and
        # explicit engine cancellation. Never let a client-selected extension
        # split the HTTP request from its signed route authority.
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
        route_authority = getattr(self.scheduler_manage, "get_route_authority", None)
        authority = route_authority(str(request_id)) if route_authority is not None else None
        if authority is not None:
            vllm_xargs[FABI_ROUTE_ID_XARG] = str(authority["route_id"])
            vllm_xargs[FABI_ROUTE_EPOCH_XARG] = int(authority["epoch"])
        backend_request["vllm_xargs"] = vllm_xargs
        return backend_request

    @staticmethod
    def _frontend_context_budget(
        stub, backend_request: Dict, budget: ContextBudget
    ) -> ContextBudget:
        """Ask the qualified route head for the prompt IDs it will execute."""

        for unsupported in ("documents", "reasoning_effort"):
            if backend_request.get(unsupported) is not None:
                raise ContextRequestError(
                    f"exact frontend tokenization does not support {unsupported}"
                )
        result = stub.tokenize_chat(
            {
                "request_id": backend_request.get("request_id"),
                "vllm_xargs": backend_request.get("vllm_xargs"),
                "request": backend_request,
            }
        )
        wait = getattr(result, "result", None)
        if wait is not None:
            result = wait(timeout=TOKENIZE_RPC_TIMEOUT_SEC)
        if not isinstance(result, dict):
            raise RuntimeError("qualified frontend returned an invalid tokenization response")
        if result.get("ok") is not True:
            detail = result.get("error")
            if not isinstance(detail, str) or not detail:
                detail = "qualified frontend rejected chat tokenization"
            raise ContextRequestError(detail)
        token_ids = result.get("tokens")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or len(token_ids) > 262_144
            or result.get("count") != len(token_ids)
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or token_id > 2**32 - 1
                for token_id in token_ids
            )
        ):
            raise RuntimeError("qualified frontend returned invalid prompt token IDs")
        return ContextBudget(
            prompt_tokens=len(token_ids),
            max_output_tokens=budget.max_output_tokens,
            prompt_token_ids=tuple(token_ids),
        )

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
        budget = ContextBudget(prompt_tokens=0, max_output_tokens=0)
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
                await self._observe_unmet_context_demand(
                    str(request_id),
                    required_context_tokens,
                )
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
        exact_token_replans = 0
        verified_frontend_token_ids: tuple[int, ...] | None = None
        while forward_attempts < self.MAX_FORWARD_RETRY:
            # Try to resolve routing; retry if table is an empty list (capacity full)
            attempts = 0
            routing_table = None
            while attempts < self.MAX_ROUTING_RETRY:
                try:
                    preferred_recovery = getattr(
                        self.scheduler_manage,
                        "preferred_recovery_level",
                        lambda _request: RecoveryLevel.RESTARTABLE,
                    )(request_data)
                    routing_table = self.scheduler_manage.get_routing_table(
                        request_id,
                        received_ts,
                        required_context_tokens,
                        prompt_tokens=budget.prompt_tokens,
                        reserved_output_tokens=budget.max_output_tokens,
                        recovery_level=preferred_recovery,
                    )
                    logger.debug(
                        f"get_routing_table for request {request_id} return: {routing_table} (attempt {attempts + 1})"
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
                requires_exact_tokens = bool(
                    getattr(
                        self.scheduler_manage,
                        "requires_exact_frontend_tokenization",
                        lambda: False,
                    )()
                )
                if requires_exact_tokens:
                    try:
                        frontend_budget = await asyncio.to_thread(
                            self._frontend_context_budget,
                            stub,
                            backend_request,
                            budget,
                        )
                    except ContextRequestError as exc:
                        self._release_route(request_id)
                        return openai_error_response(
                            str(exc),
                            status_code=400,
                            err_type="invalid_request_error",
                            param="messages",
                            code="context_validation_error",
                        )
                    except Exception:
                        self._release_route(request_id)
                        logger.exception(
                            "Unable to obtain exact frontend token IDs for %s",
                            request_id,
                        )
                        return openai_error_response(
                            "The qualified model frontend could not tokenize this request",
                            status_code=503,
                            err_type="server_unavailable",
                            code="frontend_tokenizer_unavailable",
                        )

                    if (
                        verified_frontend_token_ids is not None
                        and frontend_budget.prompt_token_ids != verified_frontend_token_ids
                    ):
                        self._release_route(request_id)
                        logger.error(
                            "Qualified route heads disagree on prompt token IDs for %s",
                            request_id,
                        )
                        return openai_error_response(
                            "Qualified workers disagree on the model tokenizer",
                            status_code=503,
                            err_type="server_unavailable",
                            code="frontend_tokenizer_mismatch",
                        )
                    verified_frontend_token_ids = frontend_budget.prompt_token_ids

                    if frontend_budget.prompt_token_ids != budget.prompt_token_ids:
                        self._release_route(request_id)
                        exact_token_replans += 1
                        if exact_token_replans > MAX_EXACT_TOKEN_REPLANS:
                            return openai_error_response(
                                "Unable to stabilize the exact model token budget",
                                status_code=503,
                                err_type="server_unavailable",
                                code="frontend_tokenizer_unstable",
                            )
                        if frontend_budget.required_tokens > max_supported_context():
                            await self._observe_unmet_context_demand(
                                str(request_id),
                                frontend_budget.required_tokens,
                            )
                            return openai_error_response(
                                (
                                    f"This request requires {frontend_budget.required_tokens} "
                                    f"tokens ({frontend_budget.prompt_tokens} prompt + "
                                    f"{frontend_budget.max_output_tokens} maximum output), but "
                                    "no available pipeline supports that context."
                                ),
                                status_code=400,
                                err_type="invalid_request_error",
                                param="messages",
                                code="context_length_exceeded",
                            )
                        logger.info(
                            "Replanning request %s with qualified frontend budget %s "
                            "(scheduler tokenizer estimated %s)",
                            request_id,
                            frontend_budget.prompt_tokens,
                            budget.prompt_tokens,
                        )
                        budget = frontend_budget
                        required_context_tokens = budget.required_tokens
                        continue
                    budget = frontend_budget

                original_replay_request = dict(backend_request)
                capture_tokens = bool(
                    is_stream
                    and getattr(
                        self.scheduler_manage,
                        "should_capture_generation_tokens",
                        lambda _request_id, _request: False,
                    )(str(request_id), request_data)
                )
                client_requested_token_ids = bool(request_data.get("return_token_ids", False))
                client_requested_reasoning = bool(request_data.get("include_reasoning", True))
                initial_journal_snapshot = None
                if capture_tokens:
                    # Both maintained vLLM frontends return the exact rendered
                    # prompt once and delta token IDs per update. Force
                    # reasoning internally so hidden tokens cannot disappear
                    # from the recovery journal; the sanitizer preserves the
                    # client's original visibility choice.
                    backend_request["return_token_ids"] = True
                    backend_request["include_reasoning"] = True
                    # The route head has already tokenized this exact request
                    # through the authenticated tokenize RPC. Persist that
                    # identity before prefill starts so a worker lost before
                    # its first SSE chunk can still be cold-replanned.
                    begin_before_prefill = getattr(
                        self.scheduler_manage,
                        "begin_generation_journal_before_prefill",
                        None,
                    )
                    if begin_before_prefill is not None:
                        initial_journal_snapshot = begin_before_prefill(
                            str(request_id),
                            prompt_token_ids=budget.prompt_token_ids,
                            request_data=request_data,
                        )
                if is_stream:

                    async def stream_generator():
                        nonlocal backend_request, stub
                        response = None
                        iterator = None
                        first_token_time = None
                        last_chunk = None
                        last_token_time = None
                        stream_finished = False
                        journal_started = initial_journal_snapshot is not None
                        journal_terminal = False
                        prefill_committed = False
                        recovery_epoch = (
                            None
                            if initial_journal_snapshot is None
                            else initial_journal_snapshot.epoch
                        )
                        replay_completion_committed = False
                        recovery_stream = (
                            OpenAIRecoveryStream(
                                expose_token_ids=client_requested_token_ids,
                                expose_reasoning=client_requested_reasoning,
                            )
                            if capture_tokens
                            else None
                        )

                        def finish_journal(
                            state: RecoveryState,
                            failure: str | None = None,
                        ) -> None:
                            nonlocal journal_terminal
                            if not journal_started or journal_terminal or recovery_epoch is None:
                                return
                            try:
                                self.scheduler_manage.finish_generation_journal(
                                    str(request_id),
                                    epoch=recovery_epoch,
                                    state=state,
                                    failure=failure,
                                )
                            except Exception:
                                logger.exception(
                                    "Unable to finalize recovery journal for %s",
                                    request_id,
                                )
                            journal_terminal = True

                        async def promote_stream_recovery() -> bool:
                            """Promote the reserved route and start exact chat replay."""

                            nonlocal backend_request
                            nonlocal iterator
                            nonlocal recovery_epoch
                            nonlocal recovery_stream
                            nonlocal replay_completion_committed
                            nonlocal response
                            nonlocal stub

                            if (
                                not capture_tokens
                                or not journal_started
                                or journal_terminal
                                or recovery_epoch is None
                            ):
                                return False
                            try:
                                if response is not None:
                                    response.cancel()
                                recovering, promoted = await asyncio.to_thread(
                                    self.scheduler_manage.promote_generation_recovery,
                                    str(request_id),
                                    failed_epoch=recovery_epoch,
                                )
                                recovery_epoch = recovering.epoch
                                head = promoted.primary_plan.stages[0].worker_id
                                model_name = self._get_model_name_for_node(head)
                                if not model_name:
                                    raise RecoveryStreamProtocolError(
                                        "promoted route has no qualified model name"
                                    )
                                head, replay_request = await asyncio.to_thread(
                                    self.scheduler_manage.build_generation_replay_request,
                                    str(request_id),
                                    original_request=original_replay_request,
                                    model_name=model_name,
                                )
                                stub = self.get_stub(head)
                                backend_request = replay_request["request"]
                                response = stub.replay_generation(replay_request)
                                iterator = iterate_in_threadpool(response)
                                recovery_stream = OpenAIChatReplayStream(
                                    expected_prompt_token_ids=recovering.spec.prompt_token_ids,
                                    committed_output_token_ids=(
                                        recovering.committed_output_token_ids
                                    ),
                                    expose_token_ids=client_requested_token_ids,
                                    expose_reasoning=client_requested_reasoning,
                                )
                                replay_completion_committed = False
                                logger.info(
                                    "Promoted request %s to recovery route %s at epoch %s "
                                    "after %s committed output tokens",
                                    request_id,
                                    promoted.primary_plan.route_id,
                                    recovery_epoch,
                                    len(recovering.committed_output_token_ids),
                                )
                                return True
                            except Exception:
                                logger.exception(
                                    "Unable to promote exact recovery for request %s",
                                    request_id,
                                )
                                return False

                        try:
                            response = stub.chat_completion(backend_request)
                            iterator = iterate_in_threadpool(response)
                            while True:
                                try:
                                    chunk = await self._next_chunk_until_disconnect(
                                        response,
                                        is_disconnected,
                                        str(request_id),
                                        iterator,
                                    )
                                except StopAsyncIteration:
                                    if not stream_finished and await promote_stream_recovery():
                                        continue
                                    if recovery_stream is not None:
                                        try:
                                            recovery_stream.finalize()
                                        except RecoveryStreamProtocolError as exc:
                                            finish_journal(RecoveryState.FAILED, str(exc))
                                            logger.warning(
                                                "Invalid exact-token stream for request %s: %s",
                                                request_id,
                                                exc,
                                            )
                                    if not stream_finished:
                                        finish_journal(
                                            RecoveryState.FAILED,
                                            "upstream stream ended without a terminal event",
                                        )
                                        logger.warning(
                                            "Upstream stream ended without a terminal event "
                                            "for request %s",
                                            request_id,
                                        )
                                        yield self._stream_error_chunk(
                                            (
                                                "A worker assigned to this request ended the "
                                                "stream unexpectedly. Please retry."
                                            ),
                                            err_type="upstream_error",
                                            code="upstream_worker_lost",
                                        )
                                        yield b"data: [DONE]\n\n"
                                    break
                                except ClientDisconnectedError:
                                    finish_journal(
                                        RecoveryState.ABORTED,
                                        "streaming client disconnected",
                                    )
                                    logger.info(
                                        "Streaming client disconnected during request %s",
                                        request_id,
                                    )
                                    return
                                except DownstreamRouteLostError:
                                    if await promote_stream_recovery():
                                        continue
                                    finish_journal(
                                        RecoveryState.FAILED,
                                        "worker route was lost and exact recovery was unavailable",
                                    )
                                    logger.warning(
                                        "Worker route was lost during streaming request %s",
                                        request_id,
                                    )
                                    yield self._stream_error_chunk(
                                        (
                                            "A worker assigned to this request became "
                                            "unavailable. Please retry."
                                        ),
                                        err_type="upstream_error",
                                        code="upstream_worker_lost",
                                    )
                                    yield b"data: [DONE]\n\n"
                                    return
                                decoded_stream_response = decode_http_response_envelope(chunk)
                                if decoded_stream_response is not None:
                                    status_code, _content_type, body = decoded_stream_response
                                    failure = f"downstream frontend returned HTTP {status_code}"
                                    finish_journal(RecoveryState.FAILED, failure)
                                    logger.warning(
                                        "Worker frontend rejected streaming request %s with "
                                        "HTTP %s",
                                        request_id,
                                        status_code,
                                    )
                                    # This is a complete terminal response, not
                                    # a truncated generation: do not append the
                                    # generic worker-lost error or abort a request
                                    # the frontend has already rejected.
                                    stream_finished = True
                                    yield self._stream_http_error_chunk(status_code, body)
                                    yield b"data: [DONE]\n\n"
                                    return
                                if recovery_stream is None:
                                    if chunk is not None and b"data: [DONE]" in chunk:
                                        stream_finished = True
                                    last_token_time = time.time()
                                    if first_token_time is None:
                                        first_token_time = last_token_time
                                    if chunk is not None and not chunk.decode("utf-8").startswith(
                                        "data: [DONE]"
                                    ):
                                        last_chunk = chunk
                                    yield chunk
                                    continue

                                try:
                                    events = recovery_stream.feed(chunk)
                                    if (
                                        isinstance(recovery_stream, OpenAIChatReplayStream)
                                        and recovery_stream.replay_complete
                                        and not replay_completion_committed
                                    ):
                                        if recovery_epoch is None:
                                            raise RecoveryStreamProtocolError(
                                                "replay completed without a recovery epoch"
                                            )
                                        self.scheduler_manage.complete_generation_replay(
                                            str(request_id),
                                            epoch=recovery_epoch,
                                        )
                                        replay_completion_committed = True
                                    for event in events:
                                        if event.prompt_token_ids is not None:
                                            if event.prompt_token_ids != budget.prompt_token_ids:
                                                raise RecoveryConflict(
                                                    "engine prompt token IDs differ from "
                                                    "the authenticated tokenize RPC"
                                                )
                                            if not journal_started:
                                                snapshot = (
                                                    self.scheduler_manage.begin_generation_journal(
                                                        str(request_id),
                                                        engine_prompt_token_ids=(
                                                            event.prompt_token_ids
                                                        ),
                                                        expected_prompt_token_ids=(
                                                            budget.prompt_token_ids
                                                        ),
                                                        request_data=request_data,
                                                    )
                                                )
                                                recovery_epoch = snapshot.epoch
                                                journal_started = True
                                        if (
                                            event.output_token_ids
                                            or event.finish_reason is not None
                                        ):
                                            if not journal_started or recovery_epoch is None:
                                                raise RecoveryStreamProtocolError(
                                                    "output arrived before exact prompt token IDs"
                                                )
                                            if not prefill_committed:
                                                self.scheduler_manage.commit_generation_prefill(
                                                    str(request_id),
                                                    epoch=recovery_epoch,
                                                )
                                                prefill_committed = True
                                            self.scheduler_manage.commit_generation_tokens(
                                                str(request_id),
                                                epoch=recovery_epoch,
                                                token_ids=event.output_token_ids,
                                            )
                                        if event.output_token_ids:
                                            last_token_time = time.time()
                                            if first_token_time is None:
                                                first_token_time = last_token_time
                                        if event.done:
                                            if not journal_started or recovery_epoch is None:
                                                raise RecoveryStreamProtocolError(
                                                    "[DONE] arrived before exact prompt token IDs"
                                                )
                                            if not prefill_committed:
                                                self.scheduler_manage.commit_generation_prefill(
                                                    str(request_id),
                                                    epoch=recovery_epoch,
                                                )
                                                prefill_committed = True
                                            finish_journal(RecoveryState.COMPLETED)
                                            stream_finished = True
                                        if not event.done:
                                            last_chunk = event.client_bytes
                                        # Token commits above happen before this
                                        # exact event becomes visible to OpenCode.
                                        yield event.client_bytes
                                except (RecoveryStreamProtocolError, RecoveryConflict) as exc:
                                    finish_journal(RecoveryState.FAILED, str(exc))
                                    logger.warning(
                                        "Exact recovery stream rejected for request %s: %s",
                                        request_id,
                                        exc,
                                    )
                                    yield self._stream_error_chunk(
                                        "The generation stream violated its exact recovery contract.",
                                        err_type="upstream_error",
                                        code="recovery_contract_violation",
                                    )
                                    yield b"data: [DONE]\n\n"
                                    return
                        finally:
                            if journal_started and not journal_terminal:
                                finish_journal(
                                    RecoveryState.ABORTED,
                                    "stream ended before a committed terminal event",
                                )
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
                                if not stream_finished:
                                    # Starlette/AnyIO cancels the body task when
                                    # the HTTP client disappears. Engine cleanup
                                    # must finish before the signed route fence
                                    # is released, even inside that cancelled
                                    # request scope.
                                    with anyio.CancelScope(shield=True):
                                        await self._best_effort_abort_backend_request(
                                            stub,
                                            backend_request,
                                        )
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
                    response = None
                    response_finished = False
                    try:
                        response = stub.chat_completion(backend_request)
                        content = await self._next_chunk_until_disconnect(
                            response,
                            is_disconnected,
                            str(request_id),
                        )
                        response_finished = True
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
                        try:
                            if response is not None:
                                response.cancel()
                            if not response_finished:
                                with anyio.CancelScope(shield=True):
                                    await self._best_effort_abort_backend_request(
                                        stub,
                                        backend_request,
                                    )
                        finally:
                            self._release_route(request_id)
            except ClientDisconnectedError:
                logger.info("Client disconnected before request %s completed", request_id)
                return Response(status_code=499)
            except DownstreamRouteLostError:
                logger.warning("Worker route was lost during request %s", request_id)
                return openai_error_response(
                    "A worker assigned to this request became unavailable. Please retry.",
                    status_code=502,
                    err_type="upstream_error",
                    code="upstream_worker_lost",
                )
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
