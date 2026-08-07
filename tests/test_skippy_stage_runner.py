import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from parallax.p2p.message_util import NativeActivationFrame
from parallax.server.sampling.sampling_params import SamplingParams
from parallax.server.skippy_stage_runner import (
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


class _FakeStage:
    def __init__(self, parts, **kwargs):
        self.parts = parts
        self.kwargs = kwargs
        self.calls = []
        self.dropped = []

    def _output(self, token_ids, input_frame, sample):
        self.calls.append((list(token_ids), input_frame, sample))
        return SimpleNamespace(
            predicted_token=42 if sample else None,
            activation=_FakeActivation(
                b"native-output",
                version=1,
                dtype="f32",
                layout="token_major",
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

    def drop_session(self, request_id):
        self.dropped.append(request_id)
        return True

    def export_full_state(self, request_id):
        return request_id.encode()

    def import_full_state(self, request_id, payload, token_count):
        self.imported = (request_id, payload, token_count)


class _FakeNative:
    SkippyActivationFrame = _FakeActivation

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
            runtime_release="mesh-llm/v0.74.0",
            runtime_abi_version="0.1.32",
        ),
        span=LayerSpan(start=0, end=2),
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

    assert native.loaded == (tmp_path, "0.74.0", "0.1.32", "vulkan")
    assert native.stage.kwargs["selected_backend_device"] == "Vulkan0"
    assert prefill.predicted_token is None
    assert prefill.activation.payload == b"native-output"
    assert decode.predicted_token == 42
    assert isinstance(native.stage.calls[-1][1], _FakeActivation)
    runner.release("request-1")
    assert native.stage.dropped == ["request-1"]


def test_runtime_discovery_is_exact_and_ambiguous_installations_fail(tmp_path, monkeypatch):
    def write_runtime(name, *, abi="0.1.32"):
        root = tmp_path / name
        root.mkdir()
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "mesh_version": "0.74.0",
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

    assert discover_skippy_native_runtime(
        mesh_release="0.74.0",
        runtime_abi="0.1.32",
        backend="vulkan",
    ) == expected.resolve()

    monkeypatch.setenv("FABI_SKIPPY_NATIVE_RUNTIME_DIR", str(tmp_path))
    write_runtime("duplicate")
    with pytest.raises(RuntimeError, match="multiple bundled"):
        discover_skippy_native_runtime(
            mesh_release="0.74.0",
            runtime_abi="0.1.32",
            backend="vulkan",
        )


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cuda:2", "CUDA2"),
        ("rocm:1", "HIP1"),
        ("metal", "Metal0"),
        ("vulkan:3", "Vulkan3"),
        ("cpu", "CPU"),
    ],
)
def test_backend_device_names_follow_skippy(device, expected):
    assert _backend_device(device) == expected
