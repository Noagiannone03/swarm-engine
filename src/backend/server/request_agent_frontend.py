"""Loopback OpenAI frontend driven by the local protocol-v3 Request Agent."""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import os
import threading
import time
import uuid
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.server.constants import NODE_STATUS_AVAILABLE, NODE_STATUS_WAITING
from backend.server.context_admission import ContextBudget, build_context_budget
from backend.server.openai_compat import (
    openai_error_response,
    openai_models_payload,
)
from backend.server.request_handler import RequestHandler
from parallax.utils.model_download import download_model_file
from parallax.utils.model_config import get_model_context_limit, normalize_model_config
from swarm_protocol.artifact_verification import verify_artifact
from swarm_protocol.contracts import ArtifactRole, RecoveryLevel, RequestContract
from swarm_protocol.request_agent import (
    RequestAgentReservation,
    RequestAgentRouteRuntime,
    _account_credential_from_environment,
)
from swarm_protocol.routing import RoutePlanningError

_MAX_OPENAI_REQUEST_BYTES = 16 * 1024 * 1024
_READINESS_CACHE_MS = 1_000


def _environment_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{name} must be a boolean value")


def _snapshot_root(downloaded_path: Path, logical_path: str) -> Path:
    root = downloaded_path
    for _part in Path(logical_path).parts:
        root = root.parent
    return root


def _verified_frontend_assets(bundle, *, local_files_only: bool) -> Path:
    """Materialize and verify only signed architecture/tokenizer artifacts."""

    descriptors = tuple(
        descriptor
        for descriptor in bundle.artifact_index.artifacts
        if descriptor.role in {ArtifactRole.ARCHITECTURE, ArtifactRole.TOKENIZER}
    )
    if not descriptors:
        raise ValueError("model registry bundle has no signed frontend artifacts")
    roots: set[Path] = set()
    for descriptor in descriptors:
        downloaded = download_model_file(
            repo_id=bundle.manifest.model_id,
            filename=descriptor.path,
            local_files_only=local_files_only,
            revision=bundle.manifest.immutable_revision,
        )
        root = _snapshot_root(downloaded, descriptor.path)
        verify_artifact(root, descriptor)
        roots.add(root.resolve())
    if len(roots) != 1:
        raise ValueError("signed frontend artifacts resolved to different model snapshots")
    return roots.pop()


class _RequestAgentCompletionHandler:
    def __init__(self, manager: "RequestAgentOpenAIManager") -> None:
        self._manager = manager

    def get_stub(self, worker_id: str):
        return self._manager.get_completion_stub(worker_id)


class RequestAgentOpenAIManager:
    """Adapter that reuses the qualified OpenAI proxy with local V3 routes."""

    def __init__(
        self,
        runtime: RequestAgentRouteRuntime,
        model_swarm_id: str,
        *,
        tokenizer: Any | None = None,
        model_context_limit: int | None = None,
        local_files_only: bool = False,
        completion_service_type: type | None = None,
    ) -> None:
        self.runtime = runtime
        self.model_swarm_id = model_swarm_id
        self._bundle = runtime.registry.fetch(model_swarm_id)
        if self._bundle.manifest.model_swarm_id != model_swarm_id:
            raise PermissionError("TUF model bundle does not match the selected swarm")
        self.model_name = self._bundle.manifest.model_id
        if tokenizer is None or model_context_limit is None:
            root = _verified_frontend_assets(
                self._bundle,
                local_files_only=local_files_only,
            )
            config_path = root / "config.json"
            if not config_path.is_file():
                raise ValueError("signed model frontend has no config.json")
            try:
                config = normalize_model_config(json.loads(config_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("signed model config.json is invalid") from error
            signed_limit = get_model_context_limit(config)
            if signed_limit is None:
                raise ValueError("signed model config has no finite context limit")
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    root,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            if model_context_limit is None:
                model_context_limit = signed_limit
            elif model_context_limit > signed_limit:
                raise ValueError("configured context limit exceeds the signed model limit")
        if (
            isinstance(model_context_limit, bool)
            or not isinstance(model_context_limit, int)
            or model_context_limit < 2
        ):
            raise ValueError("model context limit must be an integer of at least two tokens")
        self._tokenizer = tokenizer
        self._model_context_limit = model_context_limit
        self._completion_service_type = completion_service_type
        self._endpoint_by_worker: dict[str, str] = {}
        self._stub_by_endpoint: dict[str, object] = {}
        self._readiness_cached_at_ms = 0
        self._readiness_cached_context = 0
        self._closed = False
        self._lock = threading.RLock()
        self.completion_handler = _RequestAgentCompletionHandler(self)

    def get_model_name(self) -> str:
        return self.model_name

    def build_context_budget(self, request_data) -> ContextBudget:
        return build_context_budget(self._tokenizer, request_data)

    def requires_exact_frontend_tokenization(self) -> bool:
        return True

    def _refresh_live_context(self) -> int:
        now_ms = time.monotonic_ns() // 1_000_000
        with self._lock:
            if (
                self._readiness_cached_at_ms > 0
                and now_ms - self._readiness_cached_at_ms <= _READINESS_CACHE_MS
            ):
                return self._readiness_cached_context
        try:
            supported = self.runtime.max_supported_context_tokens(
                self.model_swarm_id,
                self._model_context_limit,
            )
        except (PermissionError, RoutePlanningError, RuntimeError, ValueError):
            supported = 0
        with self._lock:
            self._readiness_cached_at_ms = now_ms
            self._readiness_cached_context = supported
        return supported

    def max_supported_context_tokens(self) -> int:
        return self._refresh_live_context()

    def get_schedule_status(self):
        return NODE_STATUS_AVAILABLE if self._refresh_live_context() >= 2 else NODE_STATUS_WAITING

    def preferred_recovery_level(self, request_data) -> RecoveryLevel:
        del request_data
        # Route replacement is implemented by the Request Agent's replan_cold
        # policy, not by pre-reserving an authoritative scheduler backup.
        return RecoveryLevel.RESTARTABLE

    def get_routing_table(
        self,
        request_id,
        received_ts,
        required_context_tokens: int = 0,
        *,
        prompt_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        recovery_level: RecoveryLevel = RecoveryLevel.RESTARTABLE,
    ):
        del received_ts, recovery_level
        if prompt_tokens is None or reserved_output_tokens is None:
            raise ValueError("Request Agent routing requires exact prompt and output budgets")
        if prompt_tokens + reserved_output_tokens != required_context_tokens:
            raise ValueError("Request Agent token budget is internally inconsistent")
        request = RequestContract(
            request_id=str(request_id),
            model_swarm_id=self.model_swarm_id,
            prompt_tokens=prompt_tokens,
            reserved_output_tokens=reserved_output_tokens,
            recovery_level=RecoveryLevel.RESTARTABLE,
        )
        try:
            reservation = self.runtime.reserve(request)
        except RoutePlanningError:
            return []
        with self._lock:
            for stage in reservation.committed.plan.stages:
                self._endpoint_by_worker[stage.worker_id] = stage.endpoint_id
        return [stage.worker_id for stage in reservation.committed.plan.stages]

    def _active_reservation(self, request_id: str) -> RequestAgentReservation | None:
        return self.runtime.active_reservation(str(request_id))

    def release_routing_table(self, request_id: str) -> bool:
        reservation = self._active_reservation(str(request_id))
        if reservation is None:
            return False
        self.runtime.release(reservation)
        return True

    def is_routing_table_active(self, request_id: str) -> bool:
        return self._active_reservation(str(request_id)) is not None

    def get_route_authority(self, request_id: str) -> dict[str, object] | None:
        reservation = self._active_reservation(str(request_id))
        if reservation is None:
            return None
        plan = reservation.committed.plan
        return {
            "route_id": plan.route_id,
            "epoch": plan.epoch,
            "recovery_policy": reservation.recovery_policy.value,
        }

    def get_completion_stub(self, worker_id: str):
        with self._lock:
            endpoint_id = self._endpoint_by_worker.get(str(worker_id))
            if endpoint_id is None:
                raise RuntimeError("route worker has no authenticated endpoint")
            stub = self._stub_by_endpoint.get(endpoint_id)
            if stub is None:
                service_type = self._completion_service_type
                if service_type is None:
                    from parallax.p2p.server import TransformerConnectionHandler

                    service_type = TransformerConnectionHandler
                stub = self.runtime.transport.stub(
                    endpoint_id,
                    service_type,
                )
                self._stub_by_endpoint[endpoint_id] = stub
            return stub

    def should_capture_generation_tokens(self, request_id: str, request_data) -> bool:
        del request_id, request_data
        # Enabled in the next milestone together with the local durable journal.
        return False

    def wait_for_routing_capacity(self, timeout: float) -> bool:
        if timeout <= 0:
            return self._refresh_live_context() >= 2
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._refresh_live_context() >= 2:
                return True
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        return self._refresh_live_context() >= 2

    def status(self) -> dict[str, object]:
        return {
            **self.runtime.status(),
            "model_swarm_id": self.model_swarm_id,
            "model": self.model_name,
            "max_supported_context_tokens": self._refresh_live_context(),
            "frontend": "openai-local",
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.runtime.close()


def create_request_agent_app(
    manager: RequestAgentOpenAIManager,
    *,
    api_credential: str,
) -> FastAPI:
    if len(api_credential) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in api_credential
    ):
        raise ValueError("local Request Agent API credential must be 32-byte hexadecimal")
    expected_credential = api_credential.lower()
    handler = RequestHandler()
    handler.set_scheduler_manage(manager)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            manager.close()

    app = FastAPI(lifespan=lifespan)

    def authorized(raw_request: Request) -> bool:
        authorization = raw_request.headers.get("authorization", "")
        scheme, separator, credential = authorization.partition(" ")
        return bool(
            separator
            and scheme.lower() == "bearer"
            and hmac.compare_digest(credential.strip().lower(), expected_credential)
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        supported = manager.max_supported_context_tokens()
        return JSONResponse(
            content={"status": "ready" if supported else "waiting"},
            status_code=200 if supported else 503,
        )

    @app.get("/v1/models")
    async def models(raw_request: Request) -> JSONResponse:
        if not authorized(raw_request):
            return openai_error_response(
                "Invalid local API credential",
                status_code=401,
                err_type="authentication_error",
                code="invalid_api_key",
            )
        return JSONResponse(content=openai_models_payload(manager.get_model_name()))

    @app.get("/v1/request-agent/status")
    async def request_agent_status(raw_request: Request) -> JSONResponse:
        if not authorized(raw_request):
            return openai_error_response(
                "Invalid local API credential",
                status_code=401,
                err_type="authentication_error",
                code="invalid_api_key",
            )
        return JSONResponse(content=manager.status())

    @app.post("/v1/chat/completions")
    async def chat_completions(raw_request: Request):
        if not authorized(raw_request):
            return openai_error_response(
                "Invalid local API credential",
                status_code=401,
                err_type="authentication_error",
                code="invalid_api_key",
            )
        body = await raw_request.body()
        if len(body) > _MAX_OPENAI_REQUEST_BYTES:
            return openai_error_response(
                "Request body exceeds 16 MiB",
                status_code=413,
                err_type="invalid_request_error",
                code="request_too_large",
            )
        try:
            request_data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return openai_error_response(
                "Invalid request body",
                status_code=400,
                err_type="invalid_request_error",
                code="invalid_request_error",
            )
        if not isinstance(request_data, dict):
            return openai_error_response(
                "Request body must be a JSON object",
                status_code=400,
                err_type="invalid_request_error",
                code="invalid_request_error",
            )
        request_id = str(uuid.uuid4())
        response = await handler.v1_chat_completions(
            request_data,
            request_id,
            time.time(),
            raw_request.is_disconnected,
        )
        response.headers["X-Request-Id"] = request_id
        if isinstance(response, StreamingResponse):
            response.headers["Cache-Control"] = "no-cache"
        return response

    return app


def _loopback_host(value: str) -> str:
    if value == "localhost":
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Request Agent host must be a loopback address") from error
    if not address.is_loopback:
        raise argparse.ArgumentTypeError("Request Agent frontend must remain loopback-only")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fabi-request-agent")
    parser.add_argument("--host", default="127.0.0.1", type=_loopback_host)
    parser.add_argument("--port", default=7778, type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    model_swarm_id = os.environ.get("FABI_REQUEST_AGENT_MODEL_SWARM_ID", "").strip()
    if len(model_swarm_id) != 64 or any(
        character not in "0123456789abcdef" for character in model_swarm_id
    ):
        parser.error("FABI_REQUEST_AGENT_MODEL_SWARM_ID must be lowercase SHA-256")
    api_credential = _account_credential_from_environment()
    runtime = RequestAgentRouteRuntime.from_environment()
    try:
        manager = RequestAgentOpenAIManager(
            runtime,
            model_swarm_id,
            local_files_only=_environment_flag("FABI_USE_HFCACHE"),
        )
    except BaseException:
        runtime.close()
        raise
    uvicorn.run(
        create_request_agent_app(manager, api_credential=api_credential),
        host=args.host,
        port=args.port,
        access_log=False,
    )
    return 0
