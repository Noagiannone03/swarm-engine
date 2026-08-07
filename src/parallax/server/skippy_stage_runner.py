"""Verified Skippy staged execution through Fabi's native Rust bridge."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parallax.p2p.message_util import NativeActivationFrame
from parallax.server.backend_capabilities import DeviceKind, device_kind
from parallax.server.sampling.sampling_params import SamplingParams
from swarm_protocol.skippy_execution import VerifiedSkippySpan


_BACKEND_BY_DEVICE = {
    DeviceKind.CPU: "cpu",
    DeviceKind.CUDA: "cuda",
    DeviceKind.METAL: "metal",
    DeviceKind.ROCM: "rocm",
    DeviceKind.VULKAN: "vulkan",
}
SKIPPY_MESH_RELEASE = "0.74.0"
SKIPPY_RUNTIME_ABI = "0.1.32"


@dataclass(frozen=True)
class SkippyForwardResult:
    activation: NativeActivationFrame
    predicted_token: int | None


def _mesh_release_version(value: str) -> str:
    normalized = value.strip()
    for prefix in ("mesh-llm/v", "mesh-llm/", "v"):
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix)
    return normalized


def _runtime_search_roots() -> tuple[Path, ...]:
    explicit = os.environ.get("FABI_SKIPPY_NATIVE_RUNTIME_DIR", "").strip()
    if explicit:
        return (Path(explicit).expanduser(),)
    executable = Path(sys.executable).resolve()
    package_root = Path(__file__).resolve().parents[3]
    candidates = (
        executable.parent / "native-runtimes",
        executable.parent.parent / "native-runtimes",
        package_root / "native-runtimes",
    )
    return tuple(dict.fromkeys(candidates))


def _runtime_manifest(root: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    runtime = payload.get("runtime")
    return runtime if isinstance(runtime, dict) else None


def discover_skippy_native_runtime(
    *,
    mesh_release: str,
    runtime_abi: str,
    backend: str,
) -> Path:
    """Find one product-bundled runtime; ambiguous installations fail closed."""

    matches: list[Path] = []
    searched: list[Path] = []
    for search_root in _runtime_search_roots():
        searched.append(search_root)
        candidates = [search_root]
        if search_root.is_dir():
            candidates.extend(path for path in search_root.iterdir() if path.is_dir())
        for candidate in candidates:
            manifest = _runtime_manifest(candidate)
            if manifest is None:
                continue
            manifest_backend = manifest.get("backend")
            if isinstance(manifest_backend, dict):
                manifest_backend = manifest_backend.get("kind")
            if (
                manifest.get("mesh_version") == mesh_release
                and manifest.get("skippy_abi") == runtime_abi
                and manifest_backend == backend
            ):
                matches.append(candidate)
    unique = tuple(dict.fromkeys(path.resolve() for path in matches))
    if not unique:
        locations = ", ".join(str(path) for path in searched)
        raise RuntimeError(
            f"no bundled Skippy {backend} runtime matches Mesh {mesh_release} / ABI "
            f"{runtime_abi}; searched: {locations}"
        )
    if len(unique) != 1:
        raise RuntimeError(
            "multiple bundled Skippy runtimes match the signed execution plan: "
            + ", ".join(str(path) for path in unique)
        )
    return unique[0]


def _backend_device(device: str) -> str:
    kind = device_kind(device)
    _, separator, raw_index = device.strip().lower().partition(":")
    index = int(raw_index) if separator else 0
    prefixes = {
        DeviceKind.CPU: "CPU",
        DeviceKind.CUDA: "CUDA",
        # llama.cpp/Skippy exposes Apple GPUs as MTL0, MTL1, ... . Keep the
        # execution identifier exact: the human-facing "metal:0" spelling is
        # not accepted by the native runtime's strict device selector.
        DeviceKind.METAL: "MTL",
        DeviceKind.ROCM: "HIP",
        DeviceKind.VULKAN: "Vulkan",
    }
    try:
        prefix = prefixes[kind]
    except KeyError as exc:
        raise ValueError(f"device {device!r} has no Skippy backend-device mapping") from exc
    return prefix if kind is DeviceKind.CPU else f"{prefix}{index}"


def _sampling_kwargs(params: SamplingParams) -> dict[str, Any]:
    if params.json_schema:
        raise ValueError("Skippy structured-output grammar is not yet qualified")
    return {
        "sample": True,
        "temperature": float(params.temperature),
        "top_p": float(params.top_p),
        "top_k": max(0, int(params.top_k)),
        "min_p": float(params.min_p),
        "presence_penalty": float(params.presence_penalty),
        "frequency_penalty": float(params.frequency_penalty),
        "repeat_penalty": float(params.repetition_penalty),
        "penalty_last_n": -1,
    }


def _to_native_frame(native: Any, frame: NativeActivationFrame | None) -> Any | None:
    if frame is None:
        return None
    return native.SkippyActivationFrame(
        frame.payload,
        version=frame.version,
        dtype=frame.dtype,
        layout=frame.layout,
        producer_stage_index=frame.producer_stage_index,
        layer_start=frame.layer_start,
        layer_end=frame.layer_end,
        token_count=frame.token_count,
        sequence_count=frame.sequence_count,
        flags=frame.flags,
    )


def _from_native_frame(frame: Any) -> NativeActivationFrame:
    return NativeActivationFrame(
        version=int(frame.version),
        dtype=str(frame.dtype),
        layout=str(frame.layout),
        producer_stage_index=int(frame.producer_stage_index),
        layer_start=int(frame.layer_start),
        layer_end=int(frame.layer_end),
        token_count=int(frame.token_count),
        sequence_count=int(frame.sequence_count),
        flags=int(frame.flags),
        payload=bytes(frame.payload()),
    )


class SkippyRuntimeStageRunner:
    """Own one native layer span and generation-scoped KV sessions."""

    def __init__(
        self,
        verified: VerifiedSkippySpan,
        *,
        device: str,
        model_layer_count: int,
        max_context_tokens: int,
        max_sessions: int,
        native_module: Any | None = None,
        runtime_root: Path | None = None,
    ) -> None:
        if max_context_tokens <= 0 or max_sessions <= 0:
            raise ValueError("Skippy runner limits must be positive")
        if native_module is None:
            try:
                import fabi_network_native as native_module
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "the qualified Fabi native wheel with Skippy support is not installed"
                ) from exc
        self.native = native_module
        self.verified = verified
        self.device = device
        self.max_context_tokens = int(max_context_tokens)
        kind = device_kind(device)
        try:
            backend = _BACKEND_BY_DEVICE[kind]
        except KeyError as exc:
            raise ValueError(f"device {device!r} is not a Skippy execution device") from exc
        mesh_release = _mesh_release_version(verified.plan.runtime_release)
        runtime_root = runtime_root or discover_skippy_native_runtime(
            mesh_release=mesh_release,
            runtime_abi=verified.plan.runtime_abi_version,
            backend=backend,
        )
        devices = native_module.load_skippy_native_runtime(
            runtime_root,
            mesh_release,
            verified.plan.runtime_abi_version,
            backend,
        )
        if verified.plan.format == "gguf-direct":
            geometry = native_module.inspect_skippy_source_geometry(
                list(verified.part_paths),
                verified.plan.cache_type_k,
                verified.plan.cache_type_v,
            )
        else:
            geometry = native_module.inspect_skippy_package_geometry(
                verified.geometry_path,
                verified.plan.cache_type_k,
                verified.plan.cache_type_v,
            )
        if geometry.layer_count != model_layer_count:
            raise RuntimeError("Skippy GGUF layer count differs from the signed plan")
        if geometry.activation_width != verified.plan.activation_width:
            raise RuntimeError("Skippy GGUF activation width differs from the signed plan")
        if geometry.context_length < verified.plan.model_max_context_tokens:
            raise RuntimeError("Skippy GGUF context is smaller than the signed plan")
        if geometry.kv_bytes_per_token != sum(verified.plan.kv_bytes_per_token_by_layer):
            raise RuntimeError("Skippy GGUF KV geometry differs from the signed plan")
        if verified.plan.format == "gguf-direct" and tuple(
            int(value) for value in geometry.static_bytes_by_layer
        ) != verified.plan.direct_static_bytes_by_layer:
            raise RuntimeError("Skippy GGUF tensor geometry differs from the signed plan")
        self.max_sessions = int(max_sessions)
        selected_backend_device = _backend_device(device)
        identifiers = {
            value
            for native_device in devices
            for value in (native_device.name, native_device.device_id)
            if value
        }
        if selected_backend_device not in identifiers:
            raise RuntimeError(
                f"Skippy runtime did not expose selected device {selected_backend_device!r}; "
                f"available={sorted(identifiers)}"
            )
        self.stage = native_module.SkippyStage(
            list(verified.part_paths),
            stage_index=verified.span.start,
            layer_start=verified.span.start,
            layer_end=verified.span.end,
            model_layer_count=model_layer_count,
            context_tokens=max_context_tokens,
            lane_count=max_sessions,
            selected_backend_device=selected_backend_device,
            cache_type_k=verified.plan.cache_type_k,
            cache_type_v=verified.plan.cache_type_v,
            load_mode=(
                "runtime_slice"
                if verified.plan.format == "gguf-direct"
                else "layer_package"
            ),
        )
        self.runtime_root = runtime_root
        self.backend_device = selected_backend_device

    def prefill(
        self,
        request_id: str,
        token_ids: list[int],
        activation: NativeActivationFrame | None,
        sampling_params: SamplingParams | None,
    ) -> SkippyForwardResult:
        native_input = _to_native_frame(self.native, activation)
        kwargs = {} if sampling_params is None else _sampling_kwargs(sampling_params)
        output = self.stage.prefill(request_id, token_ids, native_input, **kwargs)
        return SkippyForwardResult(
            activation=_from_native_frame(output.activation),
            predicted_token=output.predicted_token,
        )

    def decode(
        self,
        request_id: str,
        token_id: int,
        activation: NativeActivationFrame | None,
        sampling_params: SamplingParams | None,
    ) -> SkippyForwardResult:
        native_input = _to_native_frame(self.native, activation)
        kwargs = {} if sampling_params is None else _sampling_kwargs(sampling_params)
        output = self.stage.decode(request_id, token_id, native_input, **kwargs)
        return SkippyForwardResult(
            activation=_from_native_frame(output.activation),
            predicted_token=output.predicted_token,
        )

    def release(self, request_id: str) -> None:
        self.stage.drop_session(request_id)

    def export_full_state(self, request_id: str) -> bytes:
        return bytes(self.stage.export_full_state(request_id))

    def import_full_state(self, request_id: str, payload: bytes, token_count: int) -> None:
        self.stage.import_full_state(request_id, payload, token_count)
