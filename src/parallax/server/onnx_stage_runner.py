"""ONNX Runtime execution for a verified portable layer span.

This runner owns the provider sessions and per-request KV state but deliberately
does not own routing or request scheduling. A worker can advertise READY only
after construction succeeds and measured provider assignment matches its signed
execution policy.
"""

from __future__ import annotations

import json
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from parallax.server.backend_capabilities import DeviceKind, device_kind
from parallax_utils.logging_config import get_logger
from swarm_protocol.contracts import ExecutionStageKind
from swarm_protocol.portable_execution import VerifiedExecutionSpan, VerifiedExecutionStage

logger = get_logger(__name__)


@dataclass(frozen=True)
class OrtProviderSpec:
    name: str
    options: dict[str, str]
    disable_cpu_fallback: bool
    sequential_execution: bool
    disable_memory_pattern: bool
    profile_assignment: bool


@dataclass(frozen=True)
class ProviderAssignmentReport:
    accelerator_nodes: int
    cpu_nodes: tuple[str, ...]


@dataclass(frozen=True)
class OnnxForwardResult:
    values: np.ndarray
    is_logits: bool
    cache_tokens: int


@dataclass(frozen=True)
class _LoadedStage:
    verified: VerifiedExecutionStage
    session: Any
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def ort_provider_spec(device: str) -> OrtProviderSpec:
    """Return official provider names/options without silently selecting CPU."""

    kind = device_kind(device)
    _, separator, raw_index = str(device).strip().lower().partition(":")
    index = int(raw_index) if separator else 0
    if kind is DeviceKind.DIRECTML:
        return OrtProviderSpec(
            name="DmlExecutionProvider",
            options={"device_id": str(index)},
            # ORT intentionally keeps two attention-mask shape nodes on CPU
            # for the official Qwen DML graph.  The runner profiles a warm-up
            # and accepts only node-exact exceptions from the signed plan.
            disable_cpu_fallback=False,
            sequential_execution=True,
            disable_memory_pattern=True,
            profile_assignment=True,
        )
    if kind is DeviceKind.OPENVINO:
        return OrtProviderSpec(
            name="OpenVINOExecutionProvider",
            options={"device_type": f"GPU.{index}" if separator else "GPU"},
            disable_cpu_fallback=True,
            sequential_execution=False,
            disable_memory_pattern=False,
            profile_assignment=False,
        )
    if kind is DeviceKind.QNN:
        return OrtProviderSpec(
            name="QNNExecutionProvider",
            options={"backend_type": "htp", "device_id": str(index)},
            disable_cpu_fallback=True,
            sequential_execution=False,
            disable_memory_pattern=False,
            profile_assignment=False,
        )
    if kind is DeviceKind.CPU:
        return OrtProviderSpec(
            name="CPUExecutionProvider",
            options={},
            disable_cpu_fallback=False,
            sequential_execution=False,
            disable_memory_pattern=False,
            profile_assignment=False,
        )
    if kind is DeviceKind.WINML:
        raise RuntimeError(
            "WinML automatic EP selection uses the Windows plugin-device API and is not yet "
            "qualified through ONNX Runtime's Python session surface"
        )
    raise ValueError(f"device {device!r} has no ONNX Runtime stage provider")


def _default_session_factory(path: Path, spec: OrtProviderSpec):
    try:
        import onnxruntime as ort
    except (ImportError, OSError) as exc:
        raise RuntimeError("the qualified ONNX Runtime provider package is not installed") from exc
    available = tuple(str(provider) for provider in ort.get_available_providers())
    if spec.name not in available:
        raise RuntimeError(
            f"requested ONNX Runtime provider {spec.name!r} is unavailable; installed={available}"
        )
    options = ort.SessionOptions()
    if spec.disable_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if spec.sequential_execution:
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if spec.disable_memory_pattern:
        options.enable_mem_pattern = False
    if spec.profile_assignment:
        options.enable_profiling = True
        options.profile_file_prefix = str(
            Path(tempfile.gettempdir()) / f"fabi-ort-assignment-{uuid.uuid4().hex}"
        )
    providers: list[Any] = [(spec.name, spec.options)]
    if not spec.disable_cpu_fallback and spec.name != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=providers,
    )
    active = tuple(str(provider) for provider in session.get_providers())
    if not active or active[0] != spec.name:
        raise RuntimeError(
            f"ONNX Runtime did not activate requested provider {spec.name!r}; active={active}"
        )
    return session


def _profile_node_name(event_name: str) -> str:
    suffix = "_kernel_time"
    return event_name[: -len(suffix)] if event_name.endswith(suffix) else event_name


def validate_profiled_provider_assignment(
    events: list[dict[str, Any]],
    spec: OrtProviderSpec,
    *,
    allowed_cpu_nodes: tuple[str, ...],
    require_accelerator: bool = True,
) -> ProviderAssignmentReport:
    """Reject silent accelerator fallback using ORT's measured node profile."""

    assignments = [
        (
            _profile_node_name(str(event.get("name", ""))),
            str(event.get("args", {}).get("provider", "")),
        )
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    ]
    if not assignments:
        raise RuntimeError("ONNX Runtime provider qualification produced no node assignments")
    accelerator_nodes = sum(provider == spec.name for _, provider in assignments)
    if require_accelerator and accelerator_nodes <= 0:
        raise RuntimeError(f"no graph node executed on requested provider {spec.name}")
    unexpected_providers = sorted(
        {provider for _, provider in assignments if provider not in {spec.name, "CPUExecutionProvider"}}
    )
    if unexpected_providers:
        raise RuntimeError(f"unexpected ONNX Runtime providers executed nodes: {unexpected_providers}")
    cpu_nodes = tuple(sorted({name for name, provider in assignments if provider == "CPUExecutionProvider"}))
    unexpected_cpu = tuple(sorted(set(cpu_nodes) - set(allowed_cpu_nodes)))
    if unexpected_cpu:
        raise RuntimeError(
            "unapproved ONNX Runtime CPU fallback nodes: " + ", ".join(unexpected_cpu)
        )
    return ProviderAssignmentReport(
        accelerator_nodes=accelerator_nodes,
        cpu_nodes=cpu_nodes,
    )


def _session_names(values) -> tuple[str, ...]:
    return tuple(str(value.name) for value in values)


def _stage_sort_key(stage: VerifiedExecutionStage) -> tuple[int, int]:
    descriptor = stage.descriptor
    order = {
        ExecutionStageKind.INPUT: 0,
        ExecutionStageKind.DECODER: 1,
        ExecutionStageKind.OUTPUT: 2,
    }
    return order[descriptor.kind], descriptor.start_layer


class OnnxRuntimeStageRunner:
    """Execute one signed span and maintain atomic per-request KV caches."""

    def __init__(
        self,
        verified: VerifiedExecutionSpan,
        *,
        device: str,
        max_context_tokens: int,
        max_sessions: int,
        session_factory: Callable[[Path, OrtProviderSpec], Any] = _default_session_factory,
        qualify_provider_assignment: bool | None = None,
    ) -> None:
        if max_context_tokens <= 0 or max_sessions <= 0:
            raise ValueError("ONNX runner limits must be positive")
        self.verified = verified
        self.device = device
        self.provider = ort_provider_spec(device)
        self.max_context_tokens = int(max_context_tokens)
        self.max_sessions = int(max_sessions)
        self._lock = threading.RLock()
        self._cache: dict[str, dict[int, tuple[Any, Any]]] = {}
        self._loaded: list[_LoadedStage] = []
        self._device_io_binding = self.provider.name == "DmlExecutionProvider"
        if qualify_provider_assignment is None:
            qualify_provider_assignment = session_factory is _default_session_factory
        self._qualify_provider_assignment = bool(qualify_provider_assignment)

        plan = verified.plan
        self.activation_dtype = np.dtype(plan.activation_dtype)
        self.hidden_size = int(plan.activation_hidden_size)
        self.kv_num_heads = int(plan.kv_num_heads)
        self.kv_head_dim = int(plan.kv_head_dim)
        for stage in sorted(verified.stages, key=_stage_sort_key):
            session = session_factory(stage.graph_path, self.provider)
            loaded = _LoadedStage(
                verified=stage,
                session=session,
                inputs=_session_names(session.get_inputs()),
                outputs=_session_names(session.get_outputs()),
            )
            self._validate_stage_io(loaded)
            if self.provider.profile_assignment and self._qualify_provider_assignment:
                self._profile_and_validate_stage(loaded)
            self._loaded.append(loaded)
        decoder_layers = [
            loaded.verified.descriptor.start_layer
            for loaded in self._loaded
            if loaded.verified.descriptor.kind is ExecutionStageKind.DECODER
        ]
        if decoder_layers != list(range(verified.span.start, verified.span.end)):
            raise ValueError("loaded ONNX decoder sessions do not exactly cover the verified span")

    def _validate_stage_io(self, loaded: _LoadedStage) -> None:
        descriptor = loaded.verified.descriptor
        inputs = set(loaded.inputs)
        outputs = set(loaded.outputs)
        if descriptor.kind is ExecutionStageKind.INPUT:
            if inputs != {"input_ids"} or len(outputs) != 1:
                raise ValueError("portable input endpoint has an unexpected IO contract")
            return
        if descriptor.kind is ExecutionStageKind.OUTPUT:
            if len(inputs) != 1 or outputs != {"logits"}:
                raise ValueError("portable output endpoint has an unexpected IO contract")
            return
        layer = descriptor.start_layer
        required = {
            "attention_mask",
            f"past_key_values.{layer}.key",
            f"past_key_values.{layer}.value",
        }
        if not required <= inputs or len(inputs - required - {"position_ids"}) != 1:
            raise ValueError(f"portable decoder {layer} has an unexpected input contract")
        expected_outputs = {
            f"present.{layer}.key",
            f"present.{layer}.value",
        }
        if not expected_outputs <= outputs or len(outputs - expected_outputs) != 1:
            raise ValueError(f"portable decoder {layer} has an unexpected output contract")

    def _empty_cache(self) -> np.ndarray:
        return np.empty(
            (1, self.kv_num_heads, 0, self.kv_head_dim),
            dtype=self.activation_dtype,
        )

    def _qualification_feed(self, loaded: _LoadedStage) -> dict[str, np.ndarray]:
        descriptor = loaded.verified.descriptor
        if descriptor.kind is ExecutionStageKind.INPUT:
            return {"input_ids": np.zeros((1, 1), dtype=np.int64)}
        activation = self._activation_name(loaded.inputs)
        if descriptor.kind is ExecutionStageKind.OUTPUT:
            return {
                activation: np.zeros(
                    (1, 1, self.hidden_size), dtype=self.activation_dtype
                )
            }
        layer = descriptor.start_layer
        feed = {
            activation: np.zeros(
                (1, 1, self.hidden_size), dtype=self.activation_dtype
            ),
            "attention_mask": np.ones((1, 1), dtype=np.int64),
            f"past_key_values.{layer}.key": self._empty_cache(),
            f"past_key_values.{layer}.value": self._empty_cache(),
        }
        if "position_ids" in loaded.inputs:
            feed["position_ids"] = np.zeros((1, 1), dtype=np.int64)
        return feed

    def _profile_and_validate_stage(self, loaded: _LoadedStage) -> None:
        profile_path: Path | None = None
        try:
            loaded.session.run(
                list(loaded.outputs), self._qualification_feed(loaded)
            )
            raw_path = loaded.session.end_profiling()
            if not raw_path:
                raise RuntimeError("ONNX Runtime did not emit a provider assignment profile")
            profile_path = Path(raw_path)
            events = json.loads(profile_path.read_text(encoding="utf-8"))
            stage_id = loaded.verified.descriptor.stage_id
            report = validate_profiled_provider_assignment(
                events,
                self.provider,
                allowed_cpu_nodes=self.verified.plan.allowed_cpu_fallback_nodes,
                require_accelerator=(
                    stage_id not in self.verified.plan.allowed_cpu_only_stages
                ),
            )
            logger.info(
                "Qualified %s stage %s: %d accelerator nodes, CPU exceptions=%s",
                self.provider.name,
                stage_id,
                report.accelerator_nodes,
                report.cpu_nodes,
            )
        finally:
            if profile_path is not None:
                profile_path.unlink(missing_ok=True)

    @staticmethod
    def _activation_name(names: tuple[str, ...]) -> str:
        candidates = [
            name
            for name in names
            if name not in {"attention_mask", "position_ids"}
            and not name.startswith("past_key_values.")
            and not name.startswith("present.")
        ]
        if len(candidates) != 1:
            raise ValueError(f"could not identify one activation tensor in {names}")
        return candidates[0]

    def _normalize_input(self, values: Any, *, token_ids: bool) -> np.ndarray:
        result = np.asarray(values, dtype=np.int64 if token_ids else self.activation_dtype)
        if token_ids:
            if result.ndim == 1:
                result = result[None, :]
            if result.ndim != 2:
                raise ValueError("portable input tokens must have shape [sequence] or [1, sequence]")
        else:
            if result.ndim == 2:
                result = result[None, :, :]
            if result.ndim != 3 or result.shape[0] != 1 or result.shape[-1] != self.hidden_size:
                raise ValueError(
                    f"portable activations must have shape [sequence, {self.hidden_size}]"
                )
        if result.shape[0] != 1 or result.shape[1] <= 0:
            raise ValueError("portable runner currently requires one non-empty request per call")
        return np.ascontiguousarray(result)

    def _past_length(self, request_cache: dict[int, tuple[Any, Any]]) -> int:
        lengths = {self._value_shape(pair[0])[2] for pair in request_cache.values()}
        if any(
            self._value_shape(pair[0]) != self._value_shape(pair[1])
            for pair in request_cache.values()
        ):
            raise RuntimeError("portable key/value cache shapes diverged")
        if len(lengths) > 1:
            raise RuntimeError("portable decoder layers have different cache lengths")
        return next(iter(lengths), 0)

    @staticmethod
    def _value_shape(value: Any) -> tuple[int, ...]:
        shape = value.shape() if callable(getattr(value, "shape", None)) else value.shape
        return tuple(int(item) for item in shape)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        numpy_method = getattr(value, "numpy", None)
        if not callable(numpy_method):
            raise TypeError("ONNX Runtime returned an unsupported tensor value")
        return np.asarray(numpy_method())

    def _run_stage(
        self,
        loaded: _LoadedStage,
        output_names: list[str],
        feed: dict[str, Any],
        *,
        host_output_names: frozenset[str] = frozenset(),
    ) -> list[Any]:
        """Run a stage while retaining intermediates on DirectML when possible."""

        if not self._device_io_binding:
            return list(loaded.session.run(output_names, feed))
        binding = loaded.session.io_binding()
        for name, value in feed.items():
            if callable(getattr(value, "device_name", None)):
                binding.bind_ortvalue_input(name, value)
            else:
                binding.bind_cpu_input(name, np.ascontiguousarray(value))
        device_id = int(self.provider.options.get("device_id", "0"))
        for name in output_names:
            if name in host_output_names:
                binding.bind_output(name, "cpu")
            else:
                binding.bind_output(name, "dml", device_id)
        loaded.session.run_with_iobinding(binding)
        return list(binding.get_outputs())

    def forward(self, request_id: str, values: Any) -> OnnxForwardResult:
        if not request_id:
            raise ValueError("request id must be non-empty")
        with self._lock:
            if request_id not in self._cache and len(self._cache) >= self.max_sessions:
                raise RuntimeError("portable runner has no free request session")
            current_cache = self._cache.get(request_id, {})
            past_length = self._past_length(current_cache)
            has_input = any(
                loaded.verified.descriptor.kind is ExecutionStageKind.INPUT
                for loaded in self._loaded
            )
            hidden = self._normalize_input(values, token_ids=has_input)
            sequence_length = int(hidden.shape[1])
            total_length = past_length + sequence_length
            if total_length > self.max_context_tokens:
                raise RuntimeError(
                    f"portable request context {total_length} exceeds {self.max_context_tokens}"
                )
            attention_mask = np.ones((1, total_length), dtype=np.int64)
            position_ids = np.arange(
                past_length, total_length, dtype=np.int64
            ).reshape(1, sequence_length)
            pending_cache = dict(current_cache)
            is_logits = False

            for stage_index, loaded in enumerate(self._loaded):
                descriptor = loaded.verified.descriptor
                is_last_loaded = stage_index == len(self._loaded) - 1
                if descriptor.kind is ExecutionStageKind.INPUT:
                    hidden = self._run_stage(
                        loaded, list(loaded.outputs), {"input_ids": hidden}
                    )[0]
                    continue
                if descriptor.kind is ExecutionStageKind.OUTPUT:
                    activation = self._activation_name(loaded.inputs)
                    hidden = self._run_stage(
                        loaded,
                        ["logits"],
                        {activation: hidden},
                        host_output_names=frozenset({"logits"}),
                    )[0]
                    is_logits = True
                    continue
                layer = descriptor.start_layer
                key, value = current_cache.get(
                    layer, (self._empty_cache(), self._empty_cache())
                )
                activation = self._activation_name(loaded.inputs)
                feed: dict[str, Any] = {
                    activation: hidden,
                    "attention_mask": attention_mask,
                    f"past_key_values.{layer}.key": key,
                    f"past_key_values.{layer}.value": value,
                }
                if "position_ids" in loaded.inputs:
                    feed["position_ids"] = position_ids
                activation_output = self._activation_name(loaded.outputs)
                output_values = self._run_stage(
                    loaded,
                    list(loaded.outputs),
                    feed,
                    host_output_names=(
                        frozenset({activation_output})
                        if is_last_loaded
                        else frozenset()
                    ),
                )
                output_by_name = dict(zip(loaded.outputs, output_values, strict=True))
                hidden = output_by_name[activation_output]
                pending_cache[layer] = (
                    output_by_name[f"present.{layer}.key"],
                    output_by_name[f"present.{layer}.value"],
                )

            # Commit only after every stage succeeds, so a provider exception
            # cannot leave half the local span at a newer decode position.
            self._cache[request_id] = pending_cache
            host_hidden = self._to_numpy(hidden)
            values_out = host_hidden if is_logits else host_hidden[0]
            return OnnxForwardResult(
                values=np.asarray(values_out),
                is_logits=is_logits,
                cache_tokens=total_length,
            )

    def release(self, request_id: str) -> None:
        with self._lock:
            self._cache.pop(request_id, None)

    def reset(self, request_id: str) -> None:
        self.release(request_id)

    @property
    def active_sessions(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def resident_kv_bytes(self) -> int:
        with self._lock:
            return sum(
                (
                    int(np.prod(self._value_shape(key), dtype=np.int64))
                    + int(np.prod(self._value_shape(value), dtype=np.int64))
                )
                * self.activation_dtype.itemsize
                for request_cache in self._cache.values()
                for key, value in request_cache.values()
            )
