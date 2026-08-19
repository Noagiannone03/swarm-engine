"""Skippy executor for signed sparse-GGUF layer packages."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from parallax.p2p.message_util import NativeActivationFrame
from parallax.server.executor.base_executor import BaseExecutor, ExecutorBatchCancelled
from parallax.server.executor.checkpoint_control import SkippyCheckpointExportRegistry
from parallax.server.request import InitialRequest, IntermediateRequest, Request
from parallax.server.skippy_stage_runner import (
    SKIPPY_COOPERATIVE_PREFILL_CHUNK_TOKENS,
    SkippyRequestCancelled,
    SkippyRuntimeStageRunner,
)
from parallax_utils.logging_config import get_logger
from parallax_utils.fabi_events import emit as emit_fabi_event
from swarm_protocol.contracts import LayerSpan, SkippyExactStateKind
from swarm_protocol.kv_snapshot import KvSnapshotCompatibility
from swarm_protocol.recovery import token_sequence_checksum
from swarm_protocol.skippy_execution import materialize_skippy_execution_span
from swarm_protocol.worker_integration import WorkerProtocolV3Reporter

logger = get_logger(__name__)


def _local_prefill_chunk_tokens(
    *,
    is_full_model_stage: bool,
    max_num_tokens_per_batch: int,
) -> int | None:
    """Return the exact local scheduling quantum supported by Skippy.

    A complete replica advances long prompts through ``prefill_tokens`` in
    bounded native chunks before sampling on the final chunk.  The generic
    scheduler must account for that quantum instead of treating the whole
    prompt as one unschedulable batch.  Split pipelines require a negotiated
    end-to-end activation-chunk contract and therefore remain disabled here.
    """

    if not is_full_model_stage:
        return None
    if max_num_tokens_per_batch <= 0:
        raise ValueError("Skippy max tokens per batch must be positive")
    return min(
        SKIPPY_COOPERATIVE_PREFILL_CHUNK_TOKENS,
        max_num_tokens_per_batch,
    )


@dataclass
class _SkippyCacheCapacity:
    """Native runtime admission geometry accepted during successful model open."""

    num_gpu_blocks: int
    block_size: int = 1


@dataclass(frozen=True)
class _SkippyResumeMarker:
    route_id: str
    epoch: int
    token_count: int
    token_prefix_checksum: str


class SkippyExecutor(BaseExecutor):
    """Serve one verified Skippy span through Fabi's normal V3 route lifecycle."""

    def __init__(
        self,
        *,
        model_repo: str,
        start_layer: int,
        end_layer: int,
        model_revision: Optional[str] = None,
        execution_plan_id: Optional[str] = None,
        device: Optional[str] = None,
        use_hfcache: bool = False,
        max_batch_size: Optional[int] = 1,
        max_sequence_length: Optional[int] = None,
        max_num_tokens_per_batch: int = 16384,
        prefill_priority: int = 0,
        micro_batch_ratio: int = 2,
        scheduler_wait_ms: int = 500,
        request_timeout_s: Optional[int] = 600,
        layer_latency_update_every: int = 4096,
        send_to_peer_addr: Optional[str] = None,
        recv_from_peer_addr: Optional[str] = None,
        executor_control_addr: Optional[str] = None,
        executor_input_ipc_addr: Optional[str] = None,
        executor_output_ipc_addr: Optional[str] = None,
        tp_rank: Optional[int] = 0,
        tp_size: Optional[int] = 1,
        dp_rank: Optional[int] = 0,
        dp_size: Optional[int] = 1,
        shared_state: Optional[dict] = None,
        planned_context_tokens: Optional[int] = None,
        conn: Optional[List[Any]] = None,
        enable_lora: Optional[bool] = False,
        lora_paths: Optional[List[str]] = None,
        enable_weight_refit: Optional[bool] = False,
        chunked_prefill_size: Optional[int] = None,
        **_unused: Any,
    ) -> None:
        if device is None:
            raise ValueError("Skippy executor requires an explicit native execution device")
        if not model_revision:
            raise ValueError("Skippy executor requires an immutable model revision")
        if enable_lora or lora_paths:
            raise ValueError("Skippy LoRA execution is not yet qualified")
        if enable_weight_refit:
            raise ValueError("Skippy weight refit is not supported")
        if chunked_prefill_size not in {None, 0}:
            raise ValueError("Skippy chunked prefill is not yet qualified")
        if int(tp_size or 1) != 1 or int(dp_size or 1) != 1:
            raise ValueError("Skippy executor requires tp_size=dp_size=1")
        max_sessions = int(max_batch_size or 1)
        if max_sessions <= 0:
            raise ValueError("Skippy max batch size must be positive")

        reporter = WorkerProtocolV3Reporter.from_environment()
        if reporter is None:
            raise RuntimeError("Skippy execution requires the pinned protocol-v3 registry")
        bundle = reporter.resolve_trusted_bundle(
            model_repo,
            immutable_revision=model_revision,
        )
        context_limit = int(
            planned_context_tokens
            or max_sequence_length
            or bundle.manifest.model_max_context_tokens
        )
        if context_limit <= 0 or context_limit > bundle.manifest.model_max_context_tokens:
            raise ValueError("Skippy context exceeds the signed model limit")

        self._fabi_weights_files_done = 0
        self._fabi_weights_files_total = 0

        def report_download_progress(files_done: int, files_total: int) -> None:
            self._fabi_weights_files_done = files_done
            self._fabi_weights_files_total = files_total
            emit_fabi_event(
                "weights_load_progress",
                start_layer=start_layer,
                end_layer=end_layer,
                files_done=files_done,
                files_total=files_total,
            )

        verified = materialize_skippy_execution_span(
            bundle.artifact_index,
            bundle.manifest,
            LayerSpan(start=start_layer, end=end_layer),
            device=device,
            plan_id=execution_plan_id,
            local_files_only=use_hfcache,
            progress_callback=report_download_progress,
        )
        if context_limit > verified.plan.model_max_context_tokens:
            raise ValueError("Skippy context exceeds the signed execution-plan limit")

        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(
            model_repo,
            revision=model_revision,
            local_files_only=use_hfcache,
            trust_remote_code=False,
        )
        self.config = config.to_dict()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_repo,
            revision=model_revision,
            local_files_only=use_hfcache,
            trust_remote_code=False,
        )
        self.runner = SkippyRuntimeStageRunner(
            verified,
            device=device,
            model_layer_count=bundle.manifest.num_layers,
            max_context_tokens=context_limit,
            max_sessions=max_sessions,
        )
        local_prefill_chunk_tokens = _local_prefill_chunk_tokens(
            is_full_model_stage=self.runner.is_full_model_stage,
            max_num_tokens_per_batch=max_num_tokens_per_batch,
        )
        self.execution_plan = verified.plan
        self.model_manifest = bundle.manifest
        self._checkpoint_import_markers: dict[str, _SkippyResumeMarker] = {}
        self._checkpoint_import_handle_by_request: dict[str, str] = {}
        self._checkpoint_resume_markers: dict[str, _SkippyResumeMarker] = {}
        self._authority_by_engine_request: dict[str, str] = {}
        self.checkpoint_exports = (
            SkippyCheckpointExportRegistry(
                runner=self.runner,
                kv_bytes_per_token=sum(
                    verified.plan.kv_bytes_per_token_by_layer[start_layer:end_layer]
                ),
                layer_start=start_layer,
                layer_end=end_layer,
            )
            if verified.plan.exact_state_kind is not SkippyExactStateKind.DISABLED
            else None
        )
        self.cache_manager = _SkippyCacheCapacity(
            num_gpu_blocks=context_limit * max_sessions,
        )

        super().__init__(
            start_layer=start_layer,
            end_layer=end_layer,
            dtype="float32",
            device=device,
            resolved_dtype="f32",
            max_batch_size=max_sessions,
            max_sequence_length=context_limit,
            max_num_tokens_per_batch=max_num_tokens_per_batch,
            prefill_priority=prefill_priority,
            micro_batch_ratio=micro_batch_ratio,
            scheduler_wait_ms=scheduler_wait_ms,
            request_timeout_s=request_timeout_s,
            layer_latency_update_every=layer_latency_update_every,
            send_to_peer_addr=send_to_peer_addr,
            recv_from_peer_addr=recv_from_peer_addr,
            executor_control_addr=executor_control_addr,
            executor_input_ipc_addr=executor_input_ipc_addr,
            executor_output_ipc_addr=executor_output_ipc_addr,
            tp_rank=tp_rank,
            tp_size=tp_size,
            dp_rank=dp_rank,
            dp_size=dp_size,
            shared_state=shared_state,
            enable_weight_refit=False,
            chunked_prefill_size=local_prefill_chunk_tokens,
            kv_block_size=1,
            conn=conn or [],
        )
        if self.shared_state is not None:
            self.shared_state.update(
                execution_plan_id=verified.plan.plan_id,
                execution_device=device,
                skippy_runtime_release=verified.plan.runtime_release,
                skippy_runtime_abi=verified.plan.runtime_abi_version,
                skippy_backend_device=self.runner.backend_device,
                skippy_execution_artifact_bytes=verified.artifact_bytes,
                skippy_local_prefill_chunk_tokens=local_prefill_chunk_tokens,
            )
        logger.info(
            "Skippy executor ready: plan=%s backend_device=%s span=[%d,%d) "
            "context=%d local_prefill_chunk_tokens=%s",
            verified.plan.plan_id,
            self.runner.backend_device,
            start_layer,
            end_layer,
            context_limit,
            local_prefill_chunk_tokens,
        )

    def handle_input_requests(self, requests: List[Request]) -> None:
        if self.is_first_peer:
            for request in requests:
                if isinstance(request, InitialRequest):
                    self.scheduler.enque_request(request)
                    continue
                if not isinstance(request, IntermediateRequest):
                    raise TypeError(f"first Skippy peer received {type(request)}")
                original = self.scheduler.get_running_request(request.request_id)
                if original is None:
                    logger.warning("Ignoring response for inactive request %s", request.request_id)
                    continue
                if not request.abort and request.next_token_id is not None:
                    original.commit_new_token(request.next_token_id)
                if request.routing_table:
                    original.routing_table = request.routing_table
                self.apply_peer_terminal_status(original, request)
                if self.scheduler.check_and_update_request_status(original):
                    self._release_request(original.request_id)
                    if not self.is_last_peer and not request.abort:
                        self.finished_batch.append(request)
                else:
                    self.scheduler.enque_request(original)
                if self.tp_rank == 0:
                    self.send_engine_core_request_output(
                        request=original,
                        token_id=request.next_token_id,
                    )
            return

        for request in requests:
            if not isinstance(request, IntermediateRequest):
                raise TypeError("non-first Skippy peer requires IntermediateRequest")
            if request.is_finished or request.hidden_states is None:
                self._release_request(request.request_id)
                self.scheduler.evict_request(request.request_id)
                if not self.is_last_peer and not request.abort:
                    self.finished_batch.append(request)
            else:
                if not isinstance(request.hidden_states, NativeActivationFrame):
                    raise TypeError("Skippy peer requires a typed native activation frame")
                self.scheduler.enque_request(request)

    def _prepare_prefill_batch(self, requests: List[Request]) -> Optional[Dict[str, Any]]:
        if not requests:
            return None
        lengths = [len(request.input_ids) for request in requests]
        return {
            "requests": requests,
            "context_lengths": lengths,
            "actual_processed_lengths": lengths,
        }

    def _prepare_decode_batch(self, requests: List[Request]) -> Optional[Dict[str, Any]]:
        if not requests:
            return None
        return {
            "requests": requests,
            "context_lengths": [1] * len(requests),
        }

    def process_batch(
        self,
        prepared_inputs: Dict[str, Any],
        return_decoded_tokens: bool = True,
    ) -> Dict[str, Any]:
        values: list[NativeActivationFrame | int] = []
        for request in prepared_inputs["requests"]:
            native_request_id = self._native_request_id(request)
            if self._request_abort_requested(native_request_id):
                raise ExecutorBatchCancelled([request.request_id])
            resume_marker = self._apply_checkpoint_resume(request, native_request_id)
            input_activation = None if self.is_first_peer else request.hidden_states
            if input_activation is not None and not isinstance(
                input_activation, NativeActivationFrame
            ):
                raise TypeError("Skippy stage received an untyped activation")
            sampling = request.sampling_params if return_decoded_tokens else None
            try:
                if request.is_prefill:
                    result = self.runner.prefill(
                        native_request_id,
                        list(request.input_ids),
                        input_activation,
                        sampling,
                        is_cancelled=lambda rid=native_request_id: self._request_abort_requested(
                            rid
                        ),
                    )
                else:
                    if isinstance(request, InitialRequest):
                        token_id = request.output_ids[-1] if request.output_ids else None
                    else:
                        token_id = request.next_token_id
                    if token_id is None:
                        raise RuntimeError("Skippy decode request has no committed token")
                    result = self.runner.decode(
                        native_request_id,
                        int(token_id),
                        input_activation,
                        sampling,
                    )
            except SkippyRequestCancelled as exc:
                raise ExecutorBatchCancelled([request.request_id]) from exc
            except BaseException:
                if resume_marker is not None:
                    self.runner.release(native_request_id)
                    self._checkpoint_resume_markers.pop(native_request_id, None)
                raise
            if resume_marker is not None:
                self._checkpoint_resume_markers.pop(native_request_id, None)
            if return_decoded_tokens:
                if result.predicted_token is None:
                    raise RuntimeError("final Skippy stage did not sample a token")
                values.append(result.predicted_token)
            else:
                if result.activation is None:
                    raise RuntimeError("non-final Skippy stage did not return an activation")
                values.append(result.activation)
        return {"hidden_states": values, "probs": None}

    def prepare_next_batch_requests(
        self,
        requests: List[Request],
        batch_output: Any,
        context_lengths: Any,
    ) -> List[Request]:
        del context_lengths
        values = batch_output["hidden_states"]
        if len(values) != len(requests):
            raise RuntimeError("Skippy batch output count does not match request count")
        return [
            self._prepare_next_single_request(request, value)
            for request, value in zip(requests, values, strict=True)
        ]

    def _gen_token_id_from_hidden(self, hidden_states: Any) -> Tuple[int, Any]:
        if isinstance(hidden_states, bool) or not isinstance(hidden_states, int):
            raise TypeError("Skippy final stage must return exactly one integer token")
        # Native routes carry the sampled token in the presence-aware protobuf scalar.
        # Sending a Python list through ``hidden_states`` would route it into tensor
        # serialization, which is deliberately forbidden for native activation frames.
        return hidden_states, None

    def _release_request(self, rid: str) -> None:
        native_request_id = self._authority_by_engine_request.pop(rid, rid)
        self.runner.release(native_request_id)
        self._checkpoint_resume_markers.pop(native_request_id, None)
        shared_state = getattr(self, "shared_state", None)
        if shared_state is not None:
            shared_state.clear_request_abort(native_request_id)

    def handle_executor_control(
        self,
        request: dict[str, Any],
        binary_request: bytes | None = None,
    ) -> tuple[dict[str, Any], bytes | None]:
        registry = self.checkpoint_exports
        if registry is None:
            raise RuntimeError("signed Skippy plan does not certify exact warm state")
        command = request.get("command")
        request_id = request.get("request_id")
        if not isinstance(request_id, str):
            raise ValueError("checkpoint control request_id is missing")
        if command == "checkpoint_export_prepare":
            if binary_request is not None:
                raise ValueError("checkpoint export prepare does not accept binary data")
            token_count = request.get("token_count")
            if isinstance(token_count, bool) or not isinstance(token_count, int):
                raise ValueError("checkpoint token_count must be an integer")
            descriptor = registry.prepare(request_id=request_id, token_count=token_count)
            result = descriptor.as_wire_dict()
            compatibility = self._checkpoint_compatibility(result)
            result["compatibility"] = compatibility.to_wire_dict()
            result["compatibility_identity_hash"] = compatibility.identity_hash
            return result, None
        if command == "checkpoint_export_read":
            if binary_request is not None:
                raise ValueError("checkpoint export read does not accept binary data")
            handle = request.get("handle")
            offset = request.get("offset")
            if not isinstance(handle, str):
                raise ValueError("checkpoint export handle is missing")
            if isinstance(offset, bool) or not isinstance(offset, int):
                raise ValueError("checkpoint export offset must be an integer")
            chunk, next_offset, done = registry.read(
                handle=handle,
                request_id=request_id,
                offset=offset,
            )
            return {"next_offset": next_offset, "done": done}, chunk
        if command == "checkpoint_export_drop":
            if binary_request is not None:
                raise ValueError("checkpoint export drop does not accept binary data")
            handle = request.get("handle")
            if not isinstance(handle, str):
                raise ValueError("checkpoint export handle is missing")
            dropped = registry.drop(handle=handle, request_id=request_id)
            return {"dropped": dropped}, None
        if command == "checkpoint_import_begin":
            if binary_request is not None:
                raise ValueError("checkpoint import begin does not accept binary data")
            descriptor = request.get("descriptor")
            if not isinstance(descriptor, dict):
                raise ValueError("checkpoint import descriptor is missing")
            compatibility_payload = descriptor.get("compatibility")
            try:
                source_compatibility = KvSnapshotCompatibility.from_wire_dict(compatibility_payload)
            except (TypeError, ValueError) as error:
                raise ValueError("checkpoint import compatibility is invalid") from error
            claimed_identity_hash = descriptor.get("compatibility_identity_hash")
            if not isinstance(claimed_identity_hash, str) or not hmac.compare_digest(
                claimed_identity_hash,
                source_compatibility.identity_hash,
            ):
                raise ValueError("checkpoint import compatibility identity is invalid")
            source_compatibility.require_compatible(self._checkpoint_compatibility(descriptor))
            resume_marker = self._resume_marker(descriptor)
            handle = registry.begin_import(request_id=request_id, descriptor=descriptor)
            previous_handle = self._checkpoint_import_handle_by_request.get(request_id)
            if previous_handle is not None and previous_handle != handle:
                self._checkpoint_import_markers.pop(previous_handle, None)
            self._checkpoint_import_markers[handle] = resume_marker
            self._checkpoint_import_handle_by_request[request_id] = handle
            return {"handle": handle}, None
        if command == "checkpoint_import_write":
            handle = request.get("handle")
            offset = request.get("offset")
            if not isinstance(handle, str):
                raise ValueError("checkpoint import handle is missing")
            if isinstance(offset, bool) or not isinstance(offset, int):
                raise ValueError("checkpoint import offset must be an integer")
            if binary_request is None:
                raise ValueError("checkpoint import chunk is missing")
            next_offset, done = registry.append_import(
                handle=handle,
                request_id=request_id,
                offset=offset,
                chunk=binary_request,
            )
            return {"next_offset": next_offset, "done": done}, None
        if command == "checkpoint_import_commit":
            if binary_request is not None:
                raise ValueError("checkpoint import commit does not accept binary data")
            handle = request.get("handle")
            if not isinstance(handle, str):
                raise ValueError("checkpoint import handle is missing")
            registry.commit_import(handle=handle, request_id=request_id)
            marker = self._checkpoint_import_markers.pop(handle, None)
            if self._checkpoint_import_handle_by_request.get(request_id) == handle:
                self._checkpoint_import_handle_by_request.pop(request_id, None)
            if marker is None:
                self.runner.release(request_id)
                raise RuntimeError("checkpoint import resume marker is missing")
            self._checkpoint_resume_markers[request_id] = marker
            return {"committed": True}, None
        if command == "checkpoint_import_abort":
            if binary_request is not None:
                raise ValueError("checkpoint import abort does not accept binary data")
            handle = request.get("handle")
            if not isinstance(handle, str):
                raise ValueError("checkpoint import handle is missing")
            aborted = registry.abort_import(handle=handle, request_id=request_id)
            self._checkpoint_import_markers.pop(handle, None)
            if self._checkpoint_import_handle_by_request.get(request_id) == handle:
                self._checkpoint_import_handle_by_request.pop(request_id, None)
            return {"aborted": aborted}, None
        if command == "checkpoint_import_discard":
            if binary_request is not None:
                raise ValueError("checkpoint import discard does not accept binary data")
            marker = self._checkpoint_resume_markers.get(request_id)
            if marker is None:
                return {"discarded": False}, None
            requested = self._resume_marker(request)
            if marker != requested:
                raise PermissionError("checkpoint discard identity differs from import")
            self._checkpoint_resume_markers.pop(request_id, None)
            self.runner.release(request_id)
            return {"discarded": True}, None
        raise ValueError("unknown executor checkpoint command")

    @staticmethod
    def _resume_marker(payload: dict[str, Any]) -> _SkippyResumeMarker:
        route_id = payload.get("resume_route_id")
        epoch = payload.get("resume_route_epoch")
        token_count = payload.get("token_count")
        checksum = payload.get("token_prefix_checksum")
        if not isinstance(route_id, str) or not route_id:
            raise ValueError("checkpoint resume route is missing")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("checkpoint resume epoch is invalid")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
            raise ValueError("checkpoint resume token count is invalid")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError("checkpoint resume token checksum is invalid")
        try:
            bytes.fromhex(checksum)
        except ValueError as error:
            raise ValueError("checkpoint resume token checksum is invalid") from error
        return _SkippyResumeMarker(
            route_id=route_id,
            epoch=epoch,
            token_count=token_count,
            token_prefix_checksum=checksum,
        )

    def _native_request_id(self, request: Request) -> str:
        authority_request_id = getattr(request, "authority_request_id", None)
        native_request_id = (
            authority_request_id
            if isinstance(authority_request_id, str) and authority_request_id
            else request.request_id
        )
        self._authority_by_engine_request[request.request_id] = native_request_id
        return native_request_id

    def _apply_checkpoint_resume(
        self,
        request: Request,
        native_request_id: str,
    ) -> _SkippyResumeMarker | None:
        if not request.is_prefill:
            return None
        marker = self._checkpoint_resume_markers.get(native_request_id)
        if marker is None:
            return None
        if request.route_id != marker.route_id or request.route_epoch != marker.epoch:
            raise PermissionError("checkpoint resume request crossed its route fence")
        full_input_ids = tuple(request.origin_input_ids or request.input_ids)
        if marker.token_count >= len(full_input_ids):
            raise ValueError("checkpoint resume must leave at least one replay token")
        if (
            token_sequence_checksum(full_input_ids[: marker.token_count])
            != marker.token_prefix_checksum
        ):
            raise ValueError("checkpoint resume prefix differs from the recovery journal")
        request.input_ids = list(full_input_ids[marker.token_count :])
        return marker

    def _checkpoint_compatibility(
        self,
        descriptor: dict[str, Any],
    ) -> KvSnapshotCompatibility:
        plan = self.execution_plan
        manifest = self.model_manifest
        return KvSnapshotCompatibility(
            model_swarm_id=manifest.model_swarm_id,
            immutable_revision=manifest.immutable_revision,
            tokenizer_hash=manifest.tokenizer_hash,
            dtype=manifest.dtype,
            prefill_contract_hash=manifest.prefill_contract_hash,
            attention_kv_contract_hash=manifest.attention_kv_contract_hash,
            execution_plan_id=plan.plan_id,
            package_source_sha256=plan.package_source_sha256,
            runtime_release=plan.runtime_release,
            runtime_abi_version=plan.runtime_abi_version,
            layer_start=int(descriptor["layer_start"]),
            layer_end=int(descriptor["layer_end"]),
            state_kind=plan.exact_state_kind,
            page_version=int(descriptor["version"]),
            k_type=int(descriptor["k_type"]),
            v_type=int(descriptor["v_type"]),
            k_row_bytes=int(descriptor["k_row_bytes"]),
            v_row_bytes=int(descriptor["v_row_bytes"]),
            v_element_bytes=int(descriptor["v_element_bytes"]),
            flags=int(descriptor.get("flags", 0)),
        )

    def _request_abort_requested(self, rid: str) -> bool:
        shared_state = getattr(self, "shared_state", None)
        return bool(shared_state is not None and shared_state.request_abort_requested(rid))

    def check_and_refit_weight(self, refit_weight_path: str) -> None:
        if refit_weight_path:
            raise RuntimeError("Skippy weight refit is not supported")
