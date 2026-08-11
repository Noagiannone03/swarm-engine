import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from parallax.p2p.message_util import NativeActivationFrame
from parallax.server.sampling.sampling_params import SamplingParams
from parallax.server.skippy_stage_runner import (
    SKIPPY_COOPERATIVE_PREFILL_CHUNK_TOKENS,
    SkippyRequestCancelled,
    SkippyRuntimeStageRunner,
    _backend_device,
    discover_skippy_native_runtime,
)
from swarm_protocol.contracts import LayerSpan


class _FakeActivation:
    def __init__(
        self,
        payload,
        *,
        version,
        dtype,
        layout,
        producer_stage_index,
        layer_start,
        layer_end,
        token_count,
        sequence_count,
        flags=0,
    ):
        self._payload = bytes(payload)
        self.version = version
        self.dtype = dtype
        self.layout = layout
        self.producer_stage_index = producer_stage_index
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.token_count = token_count
        self.sequence_count = sequence_count
        self.flags = flags

    def payload(self):
        return self._payload


class _FakeKvPage:
    def __init__(
        self,
        payload,
        *,
        version,
        layer_start,
        layer_end,
        token_start,
        token_count,
        layer_count,
        k_type,
        v_type,
        k_row_bytes,
        v_row_bytes,
        v_element_bytes,
        flags=0,
    ):
        self._payload = bytes(payload)
        self.version = version
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.token_start = token_start
        self.token_count = token_count
        self.layer_count = layer_count
        self.k_type = k_type
        self.v_type = v_type
        self.k_row_bytes = k_row_bytes
        self.v_row_bytes = v_row_bytes
        self.v_element_bytes = v_element_bytes
        self.payload_bytes = len(self._payload)
        self.flags = flags

    def payload(self):
        return self._payload

    def payload_slice(self, offset, length):
        return self._payload[offset : offset + length]

    @property
    def payload_sha256(self):
        return hashlib.sha256(self._payload).hexdigest()


class _FakeKvPageBuilder:
    def __init__(self, **descriptor):
        self.descriptor = descriptor
        self.payload = bytearray()

    @property
    def bytes_received(self):
        return len(self.payload)

    def append(self, chunk):
        self.payload.extend(chunk)

    def finish(self):
        assert len(self.payload) == self.descriptor["payload_bytes"]
        assert hashlib.sha256(self.payload).hexdigest() == self.descriptor["payload_sha256"]
        return _FakeKvPage(
            self.payload,
            version=self.descriptor["version"],
            layer_start=self.descriptor["layer_start"],
            layer_end=self.descriptor["layer_end"],
            token_start=self.descriptor["token_start"],
            token_count=self.descriptor["token_count"],
            layer_count=self.descriptor["layer_count"],
            k_type=self.descriptor["k_type"],
            v_type=self.descriptor["v_type"],
            k_row_bytes=self.descriptor["k_row_bytes"],
            v_row_bytes=self.descriptor["v_row_bytes"],
            v_element_bytes=self.descriptor["v_element_bytes"],
            flags=self.descriptor["flags"],
        )


class _FakeStage:
    def __init__(self, parts, **kwargs):
        self.parts = parts
        self.kwargs = kwargs
        self.calls = []
        self.prefill_token_calls = []
        self.dropped = []

    def _output(self, token_ids, input_frame, sample):
        self.calls.append((list(token_ids), input_frame, sample))
        return SimpleNamespace(
            predicted_token=42 if sample else None,
            activation=_FakeActivation(
                b"" if sample else b"native-output",
                version=1,
                dtype="unknown" if sample else "f32",
                layout="opaque" if sample else "token_major",
                producer_stage_index=0,
                layer_start=0,
                layer_end=2,
                token_count=len(token_ids),
                sequence_count=1,
            ),
        )

    def prefill(self, request_id, token_ids, input_frame, **kwargs):
        del request_id
        return self._output(token_ids, input_frame, kwargs.get("sample", False))

    def decode(self, request_id, token_id, input_frame, **kwargs):
        del request_id
        return self._output([token_id], input_frame, kwargs.get("sample", False))

    def prefill_tokens(self, request_id, token_ids):
        self.prefill_token_calls.append((request_id, list(token_ids)))

    def drop_session(self, request_id):
        self.dropped.append(request_id)
        return True

    def export_full_state(self, request_id):
        return request_id.encode()

    def import_full_state(self, request_id, payload, token_count):
        self.imported = (request_id, payload, token_count)

    def export_kv_page(self, request_id, token_start, token_count):
        self.exported_kv = (request_id, token_start, token_count)
        return _FakeKvPage(
            b"exact-kv",
            version=1,
            layer_start=0,
            layer_end=2,
            token_start=token_start,
            token_count=token_count,
            layer_count=2,
            k_type=1,
            v_type=1,
            k_row_bytes=128,
            v_row_bytes=128,
            v_element_bytes=2,
        )

    def import_kv_page(self, request_id, page):
        self.imported_kv = (request_id, page)


class _FakeNative:
    SkippyActivationFrame = _FakeActivation
    SkippyKvPage = _FakeKvPage
    SkippyKvPageBuilder = _FakeKvPageBuilder

    def __init__(self):
        self.loaded = None
        self.stage = None

    def load_skippy_native_runtime(self, root, release, abi, backend):
        self.loaded = (Path(root), release, abi, backend)
        return [
            SimpleNamespace(
                name="Vulkan0",
                device_id="Vulkan0",
            )
        ]

    def inspect_skippy_package_geometry(self, path, cache_type_k, cache_type_v):
        assert Path(path).name == "metadata.gguf"
        assert (cache_type_k, cache_type_v) == ("f16", "f16")
        return SimpleNamespace(
            architecture="qwen3",
            context_length=32768,
            activation_width=1024,
            layer_count=2,
            kv_bytes_per_token=512,
        )

    def inspect_skippy_source_geometry(self, paths, cache_type_k, cache_type_v):
        assert [Path(path).name for path in paths] == ["model.gguf"]
        assert (cache_type_k, cache_type_v) == ("f16", "f16")
        return SimpleNamespace(
            architecture="qwen3",
            context_length=32768,
            activation_width=1024,
            layer_count=2,
            kv_bytes_per_token=512,
            static_bytes_by_layer=(700, 800),
        )

    def SkippyStage(self, parts, **kwargs):
        self.stage = _FakeStage(parts, **kwargs)
        return self.stage


def _verified(tmp_path):
    parts = []
    for name in ("metadata.gguf", "layer-0.gguf", "layer-1.gguf"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        parts.append(path)
    return SimpleNamespace(
        plan=SimpleNamespace(
            runtime_release="mesh-llm/v0.75.1",
            runtime_abi_version="0.1.35",
            format="gguf-layer-package",
            shared_metadata_path="metadata.gguf",
            cache_type_k="f16",
            cache_type_v="f16",
            activation_width=1024,
            model_max_context_tokens=32768,
            kv_bytes_per_token_by_layer=(256, 256),
        ),
        span=LayerSpan(start=0, end=2),
        package_root=tmp_path,
        geometry_path=tmp_path / "metadata.gguf",
        part_paths=tuple(parts),
    )


def test_runner_reuses_native_stage_and_typed_activations(tmp_path):
    native = _FakeNative()
    runner = SkippyRuntimeStageRunner(
        _verified(tmp_path),
        device="vulkan:0",
        model_layer_count=2,
        max_context_tokens=32768,
        max_sessions=2,
        native_module=native,
        runtime_root=tmp_path,
    )

    prefill = runner.prefill("request-1", [1, 2], None, None)
    decode = runner.decode(
        "request-1",
        7,
        NativeActivationFrame(
            version=1,
            dtype="f32",
            layout="token_major",
            producer_stage_index=0,
            layer_start=0,
            layer_end=2,
            token_count=1,
            sequence_count=1,
            flags=0,
            payload=b"input",
        ),
        SamplingParams(temperature=0, min_p=0.1),
    )

    assert native.loaded == (tmp_path, "0.75.1", "0.1.35", "vulkan")
    assert native.stage.kwargs["selected_backend_device"] == "Vulkan0"
    assert native.stage.kwargs["load_mode"] == "layer_package"
    assert prefill.predicted_token is None
    assert prefill.activation.payload == b"native-output"
    assert decode.predicted_token == 42
    assert decode.activation is None
    assert isinstance(native.stage.calls[-1][1], _FakeActivation)
    runner.release("request-1")
    assert native.stage.dropped == ["request-1"]


def test_runner_round_trips_exact_native_kv_descriptor(tmp_path):
    native = _FakeNative()
    runner = SkippyRuntimeStageRunner(
        _verified(tmp_path),
        device="vulkan:0",
        model_layer_count=2,
        max_context_tokens=32768,
        max_sessions=2,
        native_module=native,
        runtime_root=tmp_path,
    )

    page = runner.export_kv_page("request-kv", token_start=0, token_count=96)
    runner.import_kv_page("replacement-kv", page)

    assert native.stage.exported_kv == ("request-kv", 0, 96)
    imported_request, imported_page = native.stage.imported_kv
    assert imported_request == "replacement-kv"
    assert imported_page.payload() == b"exact-kv"
    assert imported_page.token_start == 0
    assert imported_page.token_count == 96
    assert imported_page.layer_start == 0
    assert imported_page.layer_end == 2


def test_runner_reads_native_kv_export_in_bounded_chunks(tmp_path):
    native = _FakeNative()
    runner = SkippyRuntimeStageRunner(
        _verified(tmp_path),
        device="vulkan:0",
        model_layer_count=2,
        max_context_tokens=32768,
        max_sessions=2,
        native_module=native,
        runtime_root=tmp_path,
    )

    export = runner.prepare_kv_page_export("request-kv", token_start=0, token_count=96)

    assert export.payload_bytes == len(b"exact-kv")
    assert export.payload_sha256 == hashlib.sha256(b"exact-kv").hexdigest()
    assert export.read_chunk(0, 5) == b"exact"
    assert export.read_chunk(5, 3) == b"-kv"
    with pytest.raises(ValueError, match="exceeds"):
        export.read_chunk(5, 4)


def test_runner_builds_native_kv_import_incrementally(tmp_path):
    native = _FakeNative()
    runner = SkippyRuntimeStageRunner(
        _verified(tmp_path),
        device="vulkan:0",
        model_layer_count=2,
        max_context_tokens=32768,
        max_sessions=2,
        native_module=native,
        runtime_root=tmp_path,
    )
    payload = b"exact-kv"
    descriptor = {
        "version": 1,
        "layer_start": 0,
        "layer_end": 2,
        "token_start": 0,
        "token_count": 96,
        "layer_count": 2,
        "k_type": 1,
        "v_type": 1,
        "k_row_bytes": 128,
        "v_row_bytes": 128,
        "v_element_bytes": 2,
        "flags": 0,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }

    builder = runner.begin_kv_page_import(descriptor)
    assert runner.append_kv_page_import(builder, payload[:4]) == 4
    assert runner.append_kv_page_import(builder, payload[4:]) == len(payload)
    runner.commit_kv_page_import("replacement-kv", builder)

    imported_request, imported_page = native.stage.imported_kv
    assert imported_request == "replacement-kv"
    assert imported_page.payload() == payload


def test_runner_rejects_empty_activation_from_non_terminal_stage(tmp_path):
    native = _FakeNative()
    runner = SkippyRuntimeStageRunner(
        _verified(tmp_path),
        device="vulkan:0",
        model_layer_count=2,
        max_context_tokens=32768,
        max_sessions=1,
        native_module=native,
        runtime_root=tmp_path,
    )
    native.stage._output = lambda token_ids, input_frame, sample: SimpleNamespace(
        predicted_token=None,
        activation=_FakeActivation(
            b"",
            version=1,
            dtype="unknown",
            layout="opaque",
            producer_stage_index=0,
            layer_start=0,
            layer_end=2,
            token_count=len(token_ids),
            sequence_count=1,
        ),
    )

    with pytest.raises(ValueError, match="non-terminal"):
        runner.prefill("request-empty", [1], None, None)


def test_full_replica_prefill_uses_bounded_native_chunks(tmp_path):
    native = _FakeNative()
    runner = SkippyRuntimeStageRunner(
        _verified(tmp_path),
        device="vulkan:0",
        model_layer_count=2,
        max_context_tokens=32768,
        max_sessions=1,
        native_module=native,
        runtime_root=tmp_path,
    )
    token_ids = list(range(SKIPPY_COOPERATIVE_PREFILL_CHUNK_TOKENS * 2 + 17))

    result = runner.prefill(
        "long-request",
        token_ids,
        None,
        SamplingParams(temperature=0),
        is_cancelled=lambda: False,
    )

    assert [len(call[1]) for call in native.stage.prefill_token_calls] == [512, 512]
    assert len(native.stage.calls) == 1
    assert len(native.stage.calls[0][0]) == 17
    assert native.stage.calls[0][2] is True
    assert result.predicted_token == 42
    assert result.activation is None


def test_full_replica_prefill_observes_abort_between_native_chunks(tmp_path):
    native = _FakeNative()
    runner = SkippyRuntimeStageRunner(
        _verified(tmp_path),
        device="vulkan:0",
        model_layer_count=2,
        max_context_tokens=32768,
        max_sessions=1,
        native_module=native,
        runtime_root=tmp_path,
    )

    with pytest.raises(SkippyRequestCancelled, match="was cancelled"):
        runner.prefill(
            "cancelled-request",
            list(range(SKIPPY_COOPERATIVE_PREFILL_CHUNK_TOKENS * 2)),
            None,
            SamplingParams(temperature=0),
            is_cancelled=lambda: bool(native.stage.prefill_token_calls),
        )

    assert [len(call[1]) for call in native.stage.prefill_token_calls] == [512]
    assert native.stage.calls == []
    assert native.stage.dropped == ["cancelled-request"]


def test_direct_gguf_runner_uses_runtime_slice_without_a_package(tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"gguf")
    verified = SimpleNamespace(
        plan=SimpleNamespace(
            runtime_release="mesh-llm/v0.75.1",
            runtime_abi_version="0.1.35",
            format="gguf-direct",
            cache_type_k="f16",
            cache_type_v="f16",
            activation_width=1024,
            model_max_context_tokens=32768,
            kv_bytes_per_token_by_layer=(256, 256),
            direct_static_bytes_by_layer=(700, 800),
        ),
        span=LayerSpan(start=0, end=2),
        package_root=tmp_path,
        geometry_path=source,
        part_paths=(source,),
    )
    native = _FakeNative()
    SkippyRuntimeStageRunner(
        verified,
        device="vulkan:0",
        model_layer_count=2,
        max_context_tokens=32768,
        max_sessions=1,
        native_module=native,
        runtime_root=tmp_path,
    )

    assert native.stage.kwargs["load_mode"] == "runtime_slice"


def test_runtime_discovery_is_exact_and_ambiguous_installations_fail(tmp_path, monkeypatch):
    def write_runtime(name, *, abi="0.1.35"):
        root = tmp_path / name
        root.mkdir()
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "mesh_version": "0.75.1",
                        "skippy_abi": abi,
                        "backend": {"kind": "vulkan"},
                    }
                }
            ),
            encoding="utf-8",
        )
        return root

    expected = write_runtime("expected")
    write_runtime("wrong-abi", abi="0.1.31")
    monkeypatch.setenv("FABI_SKIPPY_NATIVE_RUNTIME_DIR", str(expected))

    assert (
        discover_skippy_native_runtime(
            mesh_release="0.75.1",
            runtime_abi="0.1.35",
            backend="vulkan",
        )
        == expected.resolve()
    )

    monkeypatch.setenv("FABI_SKIPPY_NATIVE_RUNTIME_DIR", str(tmp_path))
    write_runtime("duplicate")
    with pytest.raises(RuntimeError, match="multiple bundled"):
        discover_skippy_native_runtime(
            mesh_release="0.75.1",
            runtime_abi="0.1.35",
            backend="vulkan",
        )


def test_runtime_discovery_covers_installed_product_layout(tmp_path, monkeypatch):
    # release-build.sh installs <runtime>/{python-base,parallax-venv,
    # parallax-src} next to <runtime>/native-runtimes; on POSIX the resolved
    # interpreter lives one bin/ level below its environment root.
    interpreter = tmp_path / "runtime" / "python-base" / "bin" / "python3.12"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    bundle = (
        tmp_path / "runtime" / "native-runtimes" / "meshllm-native-runtime-darwin-aarch64-metal"
    )
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "runtime": {
                    "mesh_version": "0.75.1",
                    "skippy_abi": "0.1.35",
                    "backend": {"kind": "metal"},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("FABI_SKIPPY_NATIVE_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(sys, "executable", str(interpreter))

    assert (
        discover_skippy_native_runtime(
            mesh_release="0.75.1",
            runtime_abi="0.1.35",
            backend="metal",
        )
        == bundle.resolve()
    )


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cuda:2", "CUDA2"),
        ("rocm:1", "HIP1"),
        ("metal", "MTL0"),
        ("vulkan:3", "Vulkan3"),
        ("cpu", "CPU"),
    ],
)
def test_backend_device_names_follow_skippy(device, expected):
    assert _backend_device(device) == expected
