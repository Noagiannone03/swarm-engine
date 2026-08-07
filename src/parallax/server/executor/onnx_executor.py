"""Portable ONNX Runtime executor for signed layer-stage plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from parallax.server.executor.base_executor import BaseExecutor
from parallax.server.onnx_stage_runner import OnnxRuntimeStageRunner
from parallax.server.request import (
    InitialRequest,
    IntermediateRequest,
    Request,
    RequestStatus,
)
from parallax_utils.logging_config import get_logger
from swarm_protocol.contracts import BackendKind, LayerSpan
from swarm_protocol.portable_execution import materialize_execution_span
from swarm_protocol.worker_integration import WorkerProtocolV3Reporter

logger = get_logger(__name__)


@dataclass
class _PortableCacheCapacity:
    """Exact token geometry adapter consumed by BaseExecutor telemetry."""

    num_gpu_blocks: int
    block_size: int = 1


def _penalized_logits(request: Request, logits: np.ndarray, history: list[int]) -> np.ndarray:
    params = request.sampling_params
    if params.repetition_penalty <= 0:
        raise ValueError("repetition penalty must be positive")
    result = np.asarray(logits, dtype=np.float32).copy()
    if history:
        token_ids, counts = np.unique(np.asarray(history, dtype=np.int64), return_counts=True)
        valid = (token_ids >= 0) & (token_ids < result.size)
        token_ids = token_ids[valid]
        counts = counts[valid]
        if params.repetition_penalty != 1.0:
            values = result[token_ids]
            result[token_ids] = np.where(
                values >= 0,
                values / params.repetition_penalty,
                values * params.repetition_penalty,
            )
        result[token_ids] -= params.presence_penalty
        result[token_ids] -= params.frequency_penalty * counts
    return result


def sample_numpy_token(
    request: Request,
    logits: np.ndarray,
    history: list[int],
    rng: np.random.Generator,
) -> tuple[int, float]:
    """Apply Parallax sampling parameters without importing a GPU framework."""

    params = request.sampling_params
    if params.json_schema:
        raise ValueError("portable ONNX structured-output grammar is not yet qualified")
    if not np.isfinite(params.temperature) or params.temperature <= 0:
        raise ValueError("sampling temperature must be finite and positive")
    values = _penalized_logits(request, np.asarray(logits).reshape(-1), history)
    if params.top_k == 1:
        token = int(np.argmax(values))
        return token, 1.0
    values = values / float(params.temperature)
    values -= float(np.max(values))
    probabilities = np.exp(values.astype(np.float64))
    probabilities /= probabilities.sum()

    order = np.argsort(-probabilities)
    ordered = probabilities[order]
    keep = np.ones(ordered.shape, dtype=bool)
    if params.top_k > 0:
        keep[params.top_k :] = False
    if params.top_p < 1.0:
        cumulative_before = np.cumsum(ordered) - ordered
        keep &= cumulative_before < params.top_p
    if params.min_p > 0.0:
        keep &= ordered >= ordered[0] * params.min_p
    filtered = np.where(keep, ordered, 0.0)
    total = float(filtered.sum())
    if not np.isfinite(total) or total <= 0:
        token = int(order[0])
        return token, 1.0
    filtered /= total
    sampled_rank = int(rng.choice(len(order), p=filtered))
    token = int(order[sampled_rank])
    return token, float(filtered[sampled_rank])


class OnnxExecutor(BaseExecutor):
    """Serve one verified ONNX stage span through the normal Parallax pipeline."""

    def __init__(
        self,
        *,
        model_repo: str,
        start_layer: int,
        end_layer: int,
        model_revision: Optional[str] = None,
        execution_plan_id: Optional[str] = None,
        dtype: str = "float16",
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
        kv_block_size: int = 1,
        conn: Optional[List[Any]] = None,
        enable_lora: Optional[bool] = False,
        lora_paths: Optional[List[str]] = None,
        enable_weight_refit: Optional[bool] = False,
        chunked_prefill_size: Optional[int] = None,
        **_unused: Any,
    ) -> None:
        if device is None:
            raise ValueError("portable ONNX executor requires an explicit device")
        if not model_revision:
            raise ValueError("portable ONNX executor requires an immutable model revision")
        if enable_lora or lora_paths:
            raise ValueError("portable ONNX LoRA execution is not yet qualified")
        if enable_weight_refit:
            raise ValueError("portable ONNX weight refit is not supported")
        if chunked_prefill_size not in {None, 0}:
            raise ValueError("portable ONNX chunked prefill is not yet qualified")
        if int(tp_size or 1) != 1 or int(dp_size or 1) != 1:
            raise ValueError("portable ONNX executor currently requires tp_size=dp_size=1")
        max_sessions = int(max_batch_size or 1)
        if max_sessions <= 0:
            raise ValueError("portable ONNX max batch size must be positive")

        reporter = WorkerProtocolV3Reporter.from_environment()
        if reporter is None:
            raise RuntimeError("portable ONNX execution requires the pinned protocol-v3 registry")
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
            raise ValueError("portable ONNX context exceeds the signed model limit")
        verified = materialize_execution_span(
            bundle.artifact_index,
            bundle.manifest,
            LayerSpan(start=start_layer, end=end_layer),
            device=device,
            plan_id=execution_plan_id,
            local_files_only=use_hfcache,
        )

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
        self.runner = OnnxRuntimeStageRunner(
            verified,
            device=device,
            max_context_tokens=context_limit,
            max_sessions=max_sessions,
        )
        self.execution_plan = verified.plan
        self.cache_manager = _PortableCacheCapacity(
            num_gpu_blocks=context_limit * max_sessions,
            block_size=1,
        )
        self._sampling_rng = np.random.default_rng()
        self._token_history: dict[str, list[int]] = {}

        super().__init__(
            start_layer=start_layer,
            end_layer=end_layer,
            dtype=verified.plan.activation_dtype,
            device=device,
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
                portable_execution_artifact_bytes=verified.artifact_bytes,
            )
        logger.info(
            "Portable ONNX executor ready: plan=%s provider=%s span=[%d,%d) context=%d",
            verified.plan.plan_id,
            self.runner.provider.name,
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
                    raise TypeError(f"first peer received unexpected request {type(request)}")
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
                raise TypeError("non-first portable peer requires IntermediateRequest")
            if request.is_finished or request.hidden_states is None:
                self._release_request(request.request_id)
                self.scheduler.evict_request(request.request_id)
                if not self.is_last_peer and not request.abort:
                    self.finished_batch.append(request)
            else:
                self.scheduler.enque_request(request)

    def _prepare_prefill_batch(self, requests: List[Request]) -> Optional[Dict[str, Any]]:
        if not requests:
            return None
        values: list[Any] = []
        lengths: list[int] = []
        for request in requests:
            value = request.input_ids if self.is_first_peer else request.hidden_states
            array = np.asarray(value)
            values.append(value)
            lengths.append(int(array.shape[0]))
        return {
            "requests": requests,
            "values": values,
            "context_lengths": np.asarray(lengths, dtype=np.int32),
            "actual_processed_lengths": np.asarray(lengths, dtype=np.int32),
        }

    def _prepare_decode_batch(self, requests: List[Request]) -> Optional[Dict[str, Any]]:
        if not requests:
            return None
        values = [
            [request.output_ids[-1]] if self.is_first_peer else request.hidden_states
            for request in requests
        ]
        return {
            "requests": requests,
            "values": values,
            "context_lengths": np.ones((len(requests),), dtype=np.int32),
        }

    def process_batch(
        self,
        prepared_inputs: Dict[str, Any],
        return_decoded_tokens: bool = True,
    ) -> Dict[str, Any]:
        requests = prepared_inputs["requests"]
        outputs: list[np.ndarray] = []
        token_ids: list[int] = []
        probabilities: list[float] = []
        for request, values in zip(requests, prepared_inputs["values"], strict=True):
            result = self.runner.forward(request.request_id, values)
            if return_decoded_tokens:
                if not result.is_logits:
                    raise RuntimeError("last portable stage did not produce logits")
                history = self._token_history.setdefault(
                    request.request_id,
                    list(request.origin_input_ids or request.input_ids or []),
                )
                token, probability = sample_numpy_token(
                    request,
                    result.values[0, -1],
                    history,
                    self._sampling_rng,
                )
                history.append(token)
                token_ids.append(token)
                probabilities.append(probability)
            else:
                if result.is_logits:
                    raise RuntimeError("intermediate portable stage unexpectedly produced logits")
                outputs.append(np.asarray(result.values))
        if return_decoded_tokens:
            return {
                "hidden_states": np.asarray(token_ids, dtype=np.uint32),
                "probs": probabilities if any(r.return_probs for r in requests) else None,
            }
        return {
            "hidden_states": np.concatenate(outputs, axis=0),
            "probs": None,
        }

    def _gen_token_id_from_hidden(self, hidden_states) -> Tuple[int, Any]:
        values = np.asarray(hidden_states)
        if values.size != 1:
            raise ValueError("portable final stage must return exactly one token")
        return int(values.reshape(-1)[0]), values.astype(np.int32, copy=False)

    def _release_request(self, rid: str) -> None:
        self.runner.release(rid)
        self._token_history.pop(rid, None)

    def check_and_refit_weight(self, refit_weight_path: str) -> None:
        if refit_weight_path:
            raise RuntimeError("portable ONNX weight refit is not supported")
