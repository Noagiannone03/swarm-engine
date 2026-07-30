"""Loopback OpenAI frontend driven by the local protocol-v3 Request Agent."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import ipaddress
import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from fabi_network.capability import RouteRecoveryPolicy
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
from swarm_protocol.contracts import (
    ArtifactRole,
    ModelManifest,
    RecoveryLevel,
    RequestContract,
    RoutePlan,
)
from swarm_protocol.recovery import (
    InMemoryRecoveryJournal,
    RecoveryConflict,
    RecoveryState,
    RequestRecoverySnapshot,
    RequestRecoverySpec,
    exact_replay_sampling_params,
    sampling_replay_contract,
)
from swarm_protocol.recovery_sqlite import SqliteRecoveryJournal
from swarm_protocol.request_agent import (
    RequestAgentReservation,
    RequestAgentRouteRuntime,
    _account_credential_from_environment,
)
from swarm_protocol.request_status import RequestPhaseFeed
from swarm_protocol.routing import RoutePlanningError

_MAX_OPENAI_REQUEST_BYTES = 16 * 1024 * 1024
_READINESS_CACHE_MS = 1_000


@dataclass(frozen=True)
class RequestAgentRouteContext:
    manifest: ModelManifest
    primary_plan: RoutePlan


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
        recovery_journal: InMemoryRecoveryJournal | SqliteRecoveryJournal | None = None,
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
        self._owns_recovery_journal = recovery_journal is None
        if recovery_journal is None:
            state_dir = getattr(runtime, "state_dir", None)
            if state_dir is None:
                recovery_journal = InMemoryRecoveryJournal()
            else:
                recovery_journal = SqliteRecoveryJournal(Path(state_dir) / "recovery.sqlite3")
                recovery_journal.abort_unfinished(
                    "local Request Agent restarted before the client stream completed"
                )
        self.recovery_journal = recovery_journal
        self._endpoint_by_worker: dict[str, str] = {}
        self._stub_by_endpoint: dict[str, object] = {}
        self._readiness_cached_at_ms = 0
        self._readiness_cached_context = 0
        self._closed = False
        self._lock = threading.RLock()
        self.request_phases = RequestPhaseFeed()
        set_phase_observer = getattr(runtime, "set_phase_observer", None)
        self._runtime_emits_phases = callable(set_phase_observer)
        if self._runtime_emits_phases:
            set_phase_observer(self._publish_runtime_phase)
        self.completion_handler = _RequestAgentCompletionHandler(self)

    def _publish_runtime_phase(self, request_id: str, phase: str) -> None:
        self.request_phases.publish(request_id, phase)

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
        if request_data.get("stream", False) and sampling_replay_contract(request_data):
            # This describes the replay guarantee, not a pre-reserved backup
            # topology. The route planner still receives RESTARTABLE below so
            # it never immobilizes a second complete pipeline.
            return RecoveryLevel.RECOVERABLE
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
            if not self._runtime_emits_phases:
                self.request_phases.publish(str(request_id), "planning")
            reservation = self.runtime.reserve(request)
        except RoutePlanningError as error:
            self.request_phases.publish(str(request_id), "failed", detail=str(error))
            return []
        except BaseException as error:
            self.request_phases.publish(
                str(request_id),
                "failed",
                detail=f"{type(error).__name__}: {error}",
            )
            raise
        plan = reservation.committed.plan
        self.request_phases.publish(
            str(request_id),
            "prefilling",
            epoch=plan.epoch,
            route_id=plan.route_id,
        )
        with self._lock:
            for stage in plan.stages:
                self._endpoint_by_worker[stage.worker_id] = stage.endpoint_id
        return [stage.worker_id for stage in plan.stages]

    def _active_reservation(self, request_id: str) -> RequestAgentReservation | None:
        return self.runtime.active_reservation(str(request_id))

    def release_routing_table(self, request_id: str) -> bool:
        release_request = getattr(self.runtime, "release_request", None)
        if release_request is not None:
            released = bool(release_request(str(request_id)))
            if released:
                self._publish_release_phase(str(request_id))
            return released
        reservation = self._active_reservation(str(request_id))
        if reservation is None:
            return False
        self.runtime.release(reservation)
        self._publish_release_phase(str(request_id))
        return True

    def _publish_release_phase(self, request_id: str) -> None:
        snapshot = self.recovery_journal.get(str(request_id))
        if snapshot is None or snapshot.state not in {
            RecoveryState.COMPLETED,
            RecoveryState.FAILED,
            RecoveryState.ABORTED,
        }:
            self.request_phases.publish(str(request_id), "released")

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
        reservation = self._active_reservation(str(request_id))
        return bool(
            reservation is not None
            and reservation.recovery_policy == RouteRecoveryPolicy.REPLAN_COLD
            and sampling_replay_contract(request_data) is not None
        )

    def begin_generation_journal(
        self,
        request_id: str,
        *,
        engine_prompt_token_ids: tuple[int, ...],
        expected_prompt_token_ids: tuple[int, ...],
        request_data,
    ) -> RequestRecoverySnapshot:
        """Bind exact engine tokens to the active local route before decode."""

        reservation = self._active_reservation(str(request_id))
        if reservation is None:
            raise RecoveryConflict("request route is no longer active")
        if engine_prompt_token_ids != expected_prompt_token_ids:
            raise RecoveryConflict("engine prompt token IDs differ from local context admission")
        plan = reservation.committed.plan
        if len(engine_prompt_token_ids) != plan.prompt_tokens:
            raise RecoveryConflict("engine prompt token count differs from the reserved route")
        sampling = sampling_replay_contract(request_data)
        if sampling is None:
            raise RecoveryConflict("request sampling is not exactly replayable")
        manifest = self._bundle.manifest
        if manifest.model_swarm_id != plan.model_swarm_id:
            raise RecoveryConflict("active route and trusted manifest identify different swarms")
        return self.recovery_journal.begin(
            RequestRecoverySpec(
                request_id=str(request_id),
                model_swarm_id=manifest.model_swarm_id,
                immutable_revision=manifest.immutable_revision,
                tokenizer_hash=manifest.tokenizer_hash,
                dtype=manifest.dtype,
                prefill_contract_hash=manifest.prefill_contract_hash,
                attention_kv_contract_hash=manifest.attention_kv_contract_hash,
                prompt_token_ids=engine_prompt_token_ids,
                sampling=sampling,
                recovery_level=RecoveryLevel.RECOVERABLE,
                primary_route_id=plan.route_id,
                epoch=plan.epoch,
                reserved_context_tokens=plan.required_context_tokens,
            )
        )

    def begin_generation_journal_before_prefill(
        self,
        request_id: str,
        *,
        prompt_token_ids: tuple[int, ...],
        request_data,
    ) -> RequestRecoverySnapshot:
        """Persist the authenticated route-head tokenization before inference."""

        return self.begin_generation_journal(
            str(request_id),
            engine_prompt_token_ids=prompt_token_ids,
            expected_prompt_token_ids=prompt_token_ids,
            request_data=request_data,
        )

    def commit_generation_prefill(self, request_id: str, *, epoch: int) -> None:
        snapshot = self.recovery_journal.get(str(request_id))
        if snapshot is None:
            raise RecoveryConflict("request is not present in the recovery journal")
        self.recovery_journal.commit_prefill(
            str(request_id),
            epoch=epoch,
            prompt_checksum=snapshot.prompt_checksum,
        )
        self.request_phases.publish(
            str(request_id),
            "decoding",
            epoch=epoch,
            route_id=snapshot.route_ids[-1],
        )

    def commit_generation_tokens(
        self,
        request_id: str,
        *,
        epoch: int,
        token_ids: tuple[int, ...],
    ) -> None:
        if not token_ids:
            return
        commit_batch = getattr(self.recovery_journal, "commit_tokens", None)
        if commit_batch is not None:
            committed_position = self.recovery_journal.committed_position(str(request_id))
            if committed_position is None:
                raise RecoveryConflict("request is not present in the recovery journal")
            commit_batch(
                str(request_id),
                epoch=epoch,
                position=committed_position,
                token_ids=token_ids,
            )
            return
        for token_id in token_ids:
            current = self.recovery_journal.get(str(request_id))
            if current is None:
                raise RecoveryConflict("request is not present in the recovery journal")
            self.recovery_journal.commit_token(
                str(request_id),
                epoch=epoch,
                position=current.committed_position,
                token_id=token_id,
            )

    def finish_generation_journal(
        self,
        request_id: str,
        *,
        epoch: int,
        state: RecoveryState,
        failure: str | None = None,
    ) -> None:
        snapshot = self.recovery_journal.get(str(request_id))
        if snapshot is None:
            return
        self.recovery_journal.finish(
            str(request_id),
            epoch=epoch,
            state=state,
            failure=failure,
        )
        self.request_phases.publish(
            str(request_id),
            state.value,
            epoch=epoch,
            route_id=snapshot.route_ids[-1],
            detail=failure,
        )

    def promote_generation_recovery(
        self,
        request_id: str,
        *,
        failed_epoch: int,
    ) -> tuple[RequestRecoverySnapshot, RequestAgentRouteContext]:
        """Plan a fresh route, then atomically bind the journal to its epoch."""

        snapshot = self.recovery_journal.get(str(request_id))
        if snapshot is None:
            raise RecoveryConflict("request is not present in the recovery journal")
        if snapshot.epoch != failed_epoch:
            raise RecoveryConflict("failed route epoch differs from the recovery journal")
        if not self._runtime_emits_phases:
            self.request_phases.publish(
                str(request_id),
                "recovering",
                epoch=failed_epoch,
                route_id=snapshot.route_ids[-1],
            )
        try:
            reservation = self.runtime.replan_cold(
                str(request_id),
                failed_epoch=failed_epoch,
            )
        except BaseException as error:
            self.request_phases.publish(
                str(request_id),
                "failed",
                epoch=failed_epoch,
                route_id=snapshot.route_ids[-1],
                detail=f"{type(error).__name__}: {error}",
            )
            raise
        plan = reservation.committed.plan
        try:
            recovering = self.recovery_journal.begin_recovery(
                str(request_id),
                failed_epoch=failed_epoch,
                new_epoch=plan.epoch,
                replacement_route_id=plan.route_id,
                retain_recovery_level=True,
            )
        except Exception:
            self.runtime.release_request(str(request_id))
            raise
        with self._lock:
            for stage in plan.stages:
                self._endpoint_by_worker[stage.worker_id] = stage.endpoint_id
        self.request_phases.publish(
            str(request_id),
            "replaying",
            epoch=plan.epoch,
            route_id=plan.route_id,
        )
        return recovering, RequestAgentRouteContext(
            manifest=self._bundle.manifest,
            primary_plan=plan,
        )

    def build_generation_replay_request(
        self,
        request_id: str,
        *,
        original_request: Mapping[str, object],
        model_name: str,
    ) -> tuple[str, dict[str, object]]:
        """Build the token-exact vLLM chat replay on the newly planned route."""

        if not isinstance(original_request, Mapping):
            raise TypeError("original replay request must be a mapping")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("replay model_name must not be empty")
        snapshot = self.recovery_journal.get(str(request_id))
        if snapshot is None or snapshot.state != RecoveryState.RECOVERING:
            raise RecoveryConflict("request is not awaiting exact replay")
        reservation = self._active_reservation(str(request_id))
        if reservation is None:
            raise RecoveryConflict("replacement route is no longer active")
        plan = reservation.committed.plan
        if (
            plan.epoch != snapshot.epoch
            or plan.route_id != snapshot.route_ids[-1]
            or plan.model_swarm_id != snapshot.spec.model_swarm_id
        ):
            raise RecoveryConflict("replacement route differs from the recovery journal fence")

        output_budget = snapshot.spec.reserved_context_tokens - len(snapshot.spec.prompt_token_ids)
        remaining_output_tokens = output_budget - snapshot.committed_position
        if remaining_output_tokens <= 0:
            raise RecoveryConflict("recovered request has no output budget remaining")
        sampling_params = exact_replay_sampling_params(
            snapshot.spec.sampling,
            committed_output_tokens=snapshot.committed_position,
            remaining_output_tokens=remaining_output_tokens,
        )
        routing_table = [stage.worker_id for stage in plan.stages]
        replay_chat_request = dict(original_request)
        replay_chat_request.pop("rid", None)
        replay_chat_request.pop("routing_table", None)
        replay_chat_request.pop("max_tokens", None)
        replay_chat_request["request_id"] = str(request_id)
        replay_chat_request["model"] = model_name
        replay_chat_request["stream"] = True
        replay_chat_request["max_completion_tokens"] = sampling_params.pop("max_tokens")
        replay_chat_request.update(sampling_params)
        replay_chat_request["vllm_xargs"] = {
            "parallax_routing_table": routing_table,
            "parallax_scheduler_request_id": str(request_id),
            "fabi_route_id": plan.route_id,
            "fabi_route_epoch": plan.epoch,
        }
        return plan.stages[0].worker_id, {
            "authority_request_id": str(request_id),
            "request": replay_chat_request,
            "original_prompt_token_ids": list(snapshot.spec.prompt_token_ids),
            "committed_output_token_ids": list(snapshot.committed_output_token_ids),
        }

    def complete_generation_replay(self, request_id: str, *, epoch: int) -> None:
        snapshot = self.recovery_journal.get(str(request_id))
        if snapshot is None:
            raise RecoveryConflict("request is not present in the recovery journal")
        self.recovery_journal.complete_replay(
            str(request_id),
            epoch=epoch,
            sequence_checksum=snapshot.sequence_checksum,
            rng_position=snapshot.rng_position,
        )
        self.request_phases.publish(
            str(request_id),
            "decoding",
            epoch=epoch,
            route_id=snapshot.route_ids[-1],
        )

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
            "recovery_journal": self.recovery_journal.status(),
            "request_phases": self.request_phases.snapshot(),
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        set_phase_observer = getattr(self.runtime, "set_phase_observer", None)
        if callable(set_phase_observer):
            set_phase_observer(None)
        first_error: BaseException | None = None
        try:
            self.runtime.close()
        except BaseException as error:
            first_error = error
        if self._owns_recovery_journal:
            close_journal = getattr(self.recovery_journal, "close", None)
            if callable(close_journal):
                try:
                    close_journal()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        self.request_phases.close()
        if first_error is not None:
            raise first_error


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

    @app.get("/v1/request-agent/events")
    async def request_agent_events(raw_request: Request):
        if not authorized(raw_request):
            return openai_error_response(
                "Invalid local API credential",
                status_code=401,
                err_type="authentication_error",
                code="invalid_api_key",
            )
        raw_last_event_id = raw_request.headers.get("last-event-id")
        if raw_last_event_id is None:
            last_event_id: int | None = None
        elif (
            not raw_last_event_id.isascii()
            or not raw_last_event_id.isdigit()
            or len(raw_last_event_id) > 20
        ):
            return openai_error_response(
                "Invalid Last-Event-ID",
                status_code=400,
                err_type="invalid_request_error",
                code="invalid_last_event_id",
            )
        else:
            last_event_id = int(raw_last_event_id)

        async def phase_stream():
            cursor = last_event_id
            if cursor is None:
                snapshot = manager.request_phases.snapshot()
                cursor = int(snapshot["last_event_id"])
                yield _encode_status_sse("snapshot", cursor, snapshot)
            while not await raw_request.is_disconnected():
                events, gap = await asyncio.to_thread(
                    manager.request_phases.wait_after,
                    cursor,
                    timeout=15.0,
                )
                if gap:
                    snapshot = manager.request_phases.snapshot()
                    cursor = int(snapshot["last_event_id"])
                    yield _encode_status_sse("reset", cursor, snapshot)
                    continue
                if not events:
                    # WHATWG recommends comments to keep intermediaries from
                    # timing out an otherwise quiet event stream. This is not
                    # a worker liveness signal.
                    yield b": keepalive\n\n"
                    continue
                for event in events:
                    cursor = event.event_id
                    yield _encode_status_sse("request-phase", cursor, event.to_dict())

        return StreamingResponse(
            phase_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

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


def _encode_status_sse(event: str, event_id: int, payload: Mapping[str, object]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event_id}\nevent: {event}\ndata: {data}\n\n".encode()


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


def _bound_base_url(server: uvicorn.Server, configured_host: str) -> str:
    """Return the actual loopback URL after Uvicorn has bound its listener."""

    listeners = [
        listener
        for asyncio_server in server.servers
        for listener in (asyncio_server.sockets or ())
    ]
    if len(listeners) != 1:
        raise RuntimeError(
            f"Request Agent expected exactly one listener, found {len(listeners)}"
        )
    address = listeners[0].getsockname()
    if not isinstance(address, tuple) or len(address) < 2:
        raise RuntimeError("Request Agent listener is not a TCP socket")
    port = address[1]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65_535:
        raise RuntimeError("Request Agent listener returned an invalid port")
    host = configured_host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _write_ready_file(path: Path, *, base_url: str) -> None:
    """Atomically publish the process endpoint without exposing credentials."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(
        {
            "schema_version": 1,
            "pid": os.getpid(),
            "base_url": base_url,
        },
        separators=(",", ":"),
    ).encode() + b"\n"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            # Windows ACLs, not POSIX mode bits, are authoritative.
            pass
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


class _ReadyFileServer(uvicorn.Server):
    """Uvicorn server that publishes readiness only after the socket is bound."""

    def __init__(self, config: uvicorn.Config, *, ready_file: Path | None) -> None:
        super().__init__(config)
        self._ready_file = ready_file.expanduser().resolve() if ready_file else None

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started and self._ready_file is not None:
            _write_ready_file(
                self._ready_file,
                base_url=_bound_base_url(self, self.config.host),
            )

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        try:
            await super().shutdown(sockets=sockets)
        finally:
            if self._ready_file is not None:
                self._ready_file.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fabi-request-agent")
    parser.add_argument("--host", default="127.0.0.1", type=_loopback_host)
    parser.add_argument("--port", default=7778, type=int)
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="atomically publish the bound loopback URL after startup",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65_535:
        parser.error("--port must be between 0 and 65535")
    if args.port == 0 and args.ready_file is None:
        parser.error("--port 0 requires --ready-file")
    if args.ready_file is not None:
        args.ready_file.expanduser().resolve().unlink(missing_ok=True)
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
    config = uvicorn.Config(
        create_request_agent_app(manager, api_credential=api_credential),
        host=args.host,
        port=args.port,
        access_log=False,
    )
    _ReadyFileServer(config, ready_file=args.ready_file).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
