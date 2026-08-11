"""Verified Skippy staged execution through Fabi's native Rust bridge."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
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
SKIPPY_MESH_RELEASE = "0.75.1"
SKIPPY_RUNTIME_ABI = "0.1.35"
SKIPPY_COOPERATIVE_PREFILL_CHUNK_TOKENS = 512


class SkippyRequestCancelled(RuntimeError):
    """Raised after a native chunk boundary observes a fenced abort."""


@dataclass(frozen=True)
class SkippyForwardResult:
    # A terminal Skippy stage returns the sampled token with the ABI's
    # canonical empty activation descriptor (dtype=unknown, payload_bytes=0).
    # There is no downstream peer in that case, so model the absence instead
    # of fabricating an activation dtype for a payload that does not exist.
    activation: NativeActivationFrame | None
    predicted_token: int | None


@dataclass(frozen=True)
class SkippyKvPage:
    """Exact Mesh KV page plus the native layout descriptor required to restore it."""

    version: int
    layer_start: int
    layer_end: int
    token_start: int
    token_count: int
    layer_count: int
    k_type: int
    v_type: int
    k_row_bytes: int
    v_row_bytes: int
    v_element_bytes: int
    flags: int
    payload: bytes

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError("Skippy KV page has no ABI version")
        if self.layer_start < 0 or self.layer_end <= self.layer_start:
            raise ValueError("Skippy KV page has an invalid layer range")
        if self.layer_count != self.layer_end - self.layer_start:
            raise ValueError("Skippy KV page layer count differs from its range")
        if self.token_start < 0 or self.token_count <= 0:
            raise ValueError("Skippy KV page has an invalid token range")
        if not self.payload:
            raise ValueError("Skippy KV page payload is empty")


@dataclass(frozen=True)
class SkippyKvPageExport:
    """Native-owned KV page exposed through bounded transient chunks."""

    version: int
    layer_start: int
    layer_end: int
    token_start: int
    token_count: int
    layer_count: int
    k_type: int
    v_type: int
    k_row_bytes: int
    v_row_bytes: int
    v_element_bytes: int
    flags: int
    payload_bytes: int
    payload_sha256: str
    _native_page: Any

    def __post_init__(self) -> None:
        if self.payload_bytes <= 0:
            raise ValueError("Skippy KV export payload is empty")
        if len(self.payload_sha256) != 64:
            raise ValueError("Skippy KV export has no SHA-256 digest")

    def read_chunk(self, offset: int, length: int) -> bytes:
        if offset < 0 or length <= 0:
            raise ValueError("Skippy KV chunk range is invalid")
        end = offset + length
        if end > self.payload_bytes:
            raise ValueError("Skippy KV chunk range exceeds payload")
        return bytes(self._native_page.payload_slice(offset, length))


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
    candidates = [
        executable.parent / "native-runtimes",
        executable.parent.parent / "native-runtimes",
        package_root / "native-runtimes",
    ]
    # Release archives install the engine as <runtime>/{python-base,
    # parallax-venv, parallax-src} with the audited bundle at
    # <runtime>/native-runtimes. Seen from the interpreter that root is one
    # level further up on POSIX (bin/ adds a level); seen from the sources it
    # is the parallax-src sibling.
    if len(executable.parents) > 2:
        candidates.append(executable.parents[2] / "native-runtimes")
    candidates.append(package_root.parent / "native-runtimes")
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


def _from_native_frame(
    frame: Any,
    *,
    terminal_sample: bool,
) -> NativeActivationFrame | None:
    dtype = str(frame.dtype)
    layout = str(frame.layout)
    payload = bytes(frame.payload())
    if dtype == "unknown":
        if not terminal_sample:
            raise ValueError("non-terminal Skippy stage returned an empty activation")
        if layout != "opaque" or payload:
            raise ValueError("terminal Skippy activation sentinel is malformed")
        if int(frame.version) <= 0:
            raise ValueError("terminal Skippy activation sentinel has no ABI version")
        if int(frame.token_count) <= 0 or int(frame.sequence_count) <= 0:
            raise ValueError("terminal Skippy activation sentinel has invalid dimensions")
        return None
    return NativeActivationFrame(
        version=int(frame.version),
        dtype=dtype,
        layout=layout,
        producer_stage_index=int(frame.producer_stage_index),
        layer_start=int(frame.layer_start),
        layer_end=int(frame.layer_end),
        token_count=int(frame.token_count),
        sequence_count=int(frame.sequence_count),
        flags=int(frame.flags),
        payload=payload,
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
        if (
            verified.plan.format == "gguf-direct"
            and tuple(int(value) for value in geometry.static_bytes_by_layer)
            != verified.plan.direct_static_bytes_by_layer
        ):
            raise RuntimeError("Skippy GGUF tensor geometry differs from the signed plan")
        self.max_sessions = int(max_sessions)
        self.is_full_model_stage = (
            verified.span.start == 0 and verified.span.end == model_layer_count
        )
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
                "runtime_slice" if verified.plan.format == "gguf-direct" else "layer_package"
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
        *,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> SkippyForwardResult:
        if not token_ids:
            raise ValueError("Skippy prefill token list is empty")
        native_input = _to_native_frame(self.native, activation)
        kwargs = {} if sampling_params is None else _sampling_kwargs(sampling_params)

        # Mesh's maintained OpenAI frontend advances prompt ingestion in
        # bounded chunks and checks its cancellation token between chunks.
        # Do the same for a complete local replica.  Intermediate pipeline
        # stages still need their activation frame for every chunk, so their
        # distributed chunk protocol is deliberately not approximated here.
        cooperative = (
            self.is_full_model_stage
            and native_input is None
            and sampling_params is not None
            and len(token_ids) > SKIPPY_COOPERATIVE_PREFILL_CHUNK_TOKENS
        )
        if cooperative:
            chunks = tuple(
                token_ids[offset : offset + SKIPPY_COOPERATIVE_PREFILL_CHUNK_TOKENS]
                for offset in range(
                    0,
                    len(token_ids),
                    SKIPPY_COOPERATIVE_PREFILL_CHUNK_TOKENS,
                )
            )
            for chunk in chunks[:-1]:
                self._raise_if_cancelled(request_id, is_cancelled)
                self.stage.prefill_tokens(request_id, chunk)
                self._raise_if_cancelled(request_id, is_cancelled)
            self._raise_if_cancelled(request_id, is_cancelled)
            output = self.stage.prefill(request_id, chunks[-1], None, **kwargs)
            self._raise_if_cancelled(request_id, is_cancelled)
        else:
            self._raise_if_cancelled(request_id, is_cancelled)
            output = self.stage.prefill(request_id, token_ids, native_input, **kwargs)
            self._raise_if_cancelled(request_id, is_cancelled)
        predicted_token = output.predicted_token
        return SkippyForwardResult(
            activation=_from_native_frame(
                output.activation,
                terminal_sample=predicted_token is not None,
            ),
            predicted_token=predicted_token,
        )

    def _raise_if_cancelled(
        self,
        request_id: str,
        is_cancelled: Callable[[], bool] | None,
    ) -> None:
        if is_cancelled is None or not is_cancelled():
            return
        self.stage.drop_session(request_id)
        raise SkippyRequestCancelled(f"Skippy request {request_id} was cancelled")

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
        predicted_token = output.predicted_token
        return SkippyForwardResult(
            activation=_from_native_frame(
                output.activation,
                terminal_sample=predicted_token is not None,
            ),
            predicted_token=predicted_token,
        )

    def release(self, request_id: str) -> None:
        self.stage.drop_session(request_id)

    def export_full_state(self, request_id: str) -> bytes:
        return bytes(self.stage.export_full_state(request_id))

    def import_full_state(self, request_id: str, payload: bytes, token_count: int) -> None:
        self.stage.import_full_state(request_id, payload, token_count)

    def export_kv_page(
        self,
        request_id: str,
        *,
        token_start: int,
        token_count: int,
    ) -> SkippyKvPage:
        native_page = self.stage.export_kv_page(request_id, token_start, token_count)
        payload = bytes(native_page.payload())
        if len(payload) != int(native_page.payload_bytes):
            raise RuntimeError("native Skippy KV page payload length changed during export")
        return SkippyKvPage(
            version=int(native_page.version),
            layer_start=int(native_page.layer_start),
            layer_end=int(native_page.layer_end),
            token_start=int(native_page.token_start),
            token_count=int(native_page.token_count),
            layer_count=int(native_page.layer_count),
            k_type=int(native_page.k_type),
            v_type=int(native_page.v_type),
            k_row_bytes=int(native_page.k_row_bytes),
            v_row_bytes=int(native_page.v_row_bytes),
            v_element_bytes=int(native_page.v_element_bytes),
            flags=int(native_page.flags),
            payload=payload,
        )

    def prepare_kv_page_export(
        self,
        request_id: str,
        *,
        token_start: int,
        token_count: int,
    ) -> SkippyKvPageExport:
        """Keep the native page resident and expose it without a full Python copy."""

        native_page = self.stage.export_kv_page(request_id, token_start, token_count)
        return SkippyKvPageExport(
            version=int(native_page.version),
            layer_start=int(native_page.layer_start),
            layer_end=int(native_page.layer_end),
            token_start=int(native_page.token_start),
            token_count=int(native_page.token_count),
            layer_count=int(native_page.layer_count),
            k_type=int(native_page.k_type),
            v_type=int(native_page.v_type),
            k_row_bytes=int(native_page.k_row_bytes),
            v_row_bytes=int(native_page.v_row_bytes),
            v_element_bytes=int(native_page.v_element_bytes),
            flags=int(native_page.flags),
            payload_bytes=int(native_page.payload_bytes),
            payload_sha256=str(native_page.payload_sha256),
            _native_page=native_page,
        )

    def import_kv_page(self, request_id: str, page: SkippyKvPage) -> None:
        if page.layer_start != self.verified.span.start or page.layer_end != self.verified.span.end:
            raise ValueError("Skippy KV page does not match the runner's signed layer span")
        native_page = self.native.SkippyKvPage(
            page.payload,
            version=page.version,
            layer_start=page.layer_start,
            layer_end=page.layer_end,
            token_start=page.token_start,
            token_count=page.token_count,
            layer_count=page.layer_count,
            k_type=page.k_type,
            v_type=page.v_type,
            k_row_bytes=page.k_row_bytes,
            v_row_bytes=page.v_row_bytes,
            v_element_bytes=page.v_element_bytes,
            flags=page.flags,
        )
        self.stage.import_kv_page(request_id, native_page)

    def begin_kv_page_import(self, descriptor: dict[str, int | str]) -> Any:
        """Allocate one native builder after the executor admitted its exact size."""

        return self.native.SkippyKvPageBuilder(
            version=int(descriptor["version"]),
            layer_start=int(descriptor["layer_start"]),
            layer_end=int(descriptor["layer_end"]),
            token_start=int(descriptor["token_start"]),
            token_count=int(descriptor["token_count"]),
            layer_count=int(descriptor["layer_count"]),
            k_type=int(descriptor["k_type"]),
            v_type=int(descriptor["v_type"]),
            k_row_bytes=int(descriptor["k_row_bytes"]),
            v_row_bytes=int(descriptor["v_row_bytes"]),
            v_element_bytes=int(descriptor["v_element_bytes"]),
            flags=int(descriptor.get("flags", 0)),
            payload_bytes=int(descriptor["payload_bytes"]),
            payload_sha256=str(descriptor["payload_sha256"]),
        )

    @staticmethod
    def append_kv_page_import(builder: Any, chunk: bytes) -> int:
        builder.append(chunk)
        return int(builder.bytes_received)

    def commit_kv_page_import(self, request_id: str, builder: Any) -> None:
        page = builder.finish()
        self.stage.import_kv_page(request_id, page)
