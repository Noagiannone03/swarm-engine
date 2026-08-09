"""Skippy executor for signed sparse-GGUF layer packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from parallax.p2p.message_util import NativeActivationFrame
from parallax.server.executor.base_executor import BaseExecutor
from parallax.server.request import InitialRequest, IntermediateRequest, Request
from parallax.server.skippy_stage_runner import SkippyRuntimeStageRunner
from parallax_utils.logging_config import get_logger
from swarm_protocol.contracts import LayerSpan
from swarm_protocol.skippy_execution import materialize_skippy_execution_span
from swarm_protocol.worker_integration import WorkerProtocolV3Reporter

logger = get_logger(__name__)


@dataclass
class _SkippyCacheCapacity:
    """Native runtime admission geometry accepted during successful model open."""

    num_gpu_blocks: int
    block_size: int = 1


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
        verified = materialize_skippy_execution_span(
            bundle.artifact_index,
            bundle.manifest,
            LayerSpan(start=start_layer, end=end_layer),
            device=device,
            plan_id=execution_plan_id,
            local_files_only=use_hfcache,
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
        self.execution_plan = verified.plan
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
            executor_input_ipc_addr=executor_input_ipc_addr,
            executor_output_ipc_addr=executor_output_ipc_addr,
            tp_rank=tp_rank,
            tp_size=tp_size,
            dp_rank=dp_rank,
            dp_size=dp_size,
            shared_state=shared_state,
            enable_weight_refit=False,
            chunked_prefill_size=None,
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
            )
        logger.info(
            "Skippy executor ready: plan=%s backend_device=%s span=[%d,%d) context=%d",
            verified.plan.plan_id,
            self.runner.backend_device,
            start_layer,
            end_layer,
            context_limit,
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
            input_activation = None if self.is_first_peer else request.hidden_states
            if input_activation is not None and not isinstance(
                input_activation, NativeActivationFrame
            ):
                raise TypeError("Skippy stage received an untyped activation")
            sampling = request.sampling_params if return_decoded_tokens else None
            if request.is_prefill:
                result = self.runner.prefill(
                    request.request_id,
                    list(request.input_ids),
                    input_activation,
                    sampling,
                )
            else:
                if isinstance(request, InitialRequest):
                    token_id = request.output_ids[-1] if request.output_ids else None
                else:
                    token_id = request.next_token_id
                if token_id is None:
                    raise RuntimeError("Skippy decode request has no committed token")
                result = self.runner.decode(
                    request.request_id,
                    int(token_id),
                    input_activation,
                    sampling,
                )
            if return_decoded_tokens:
                if result.predicted_token is None:
                    raise RuntimeError("final Skippy stage did not sample a token")
                values.append(result.predicted_token)
            else:
                if result.activation is None:
                    raise RuntimeError(
                        "non-final Skippy stage did not return an activation"
                    )
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
        return hidden_states, [hidden_states]

    def _release_request(self, rid: str) -> None:
        self.runner.release(rid)

    def check_and_refit_weight(self, refit_weight_path: str) -> None:
        if refit_weight_path:
            raise RuntimeError("Skippy weight refit is not supported")
