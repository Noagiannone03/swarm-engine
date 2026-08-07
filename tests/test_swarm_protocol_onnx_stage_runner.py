from types import SimpleNamespace

import numpy as np
import pytest

from parallax.server.onnx_stage_runner import (
    OnnxRuntimeStageRunner,
    ort_provider_spec,
    validate_profiled_provider_assignment,
)
from swarm_protocol.contracts import (
    BackendKind,
    ExecutionProviderKind,
    ExecutionStageDescriptor,
    ExecutionStageKind,
    LayerSpan,
    ModelExecutionPlan,
    OnnxExportTarget,
)
from swarm_protocol.portable_execution import VerifiedExecutionSpan, VerifiedExecutionStage


HASH = "0" * 64


class FakeSession:
    def __init__(self, kind, layer=None, *, fail=False):
        self.kind = kind
        self.layer = layer
        self.fail = fail
        if kind == "input":
            self.inputs = ("input_ids",)
            self.outputs = ("fabi.hidden_states.0",)
        elif kind == "output":
            self.inputs = ("fabi.hidden_states.2",)
            self.outputs = ("logits",)
        else:
            self.inputs = (
                f"fabi.hidden_states.{layer}",
                "attention_mask",
                "position_ids",
                f"past_key_values.{layer}.key",
                f"past_key_values.{layer}.value",
            )
            self.outputs = (
                f"fabi.hidden_states.{layer + 1}",
                f"present.{layer}.key",
                f"present.{layer}.value",
            )

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self.inputs]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in self.outputs]

    def run(self, outputs, feed):
        if self.fail:
            raise RuntimeError("provider failure")
        if self.kind == "input":
            tokens = feed["input_ids"]
            return [np.repeat(tokens[..., None].astype(np.float32), 8, axis=-1)]
        if self.kind == "output":
            hidden = feed[self.inputs[0]]
            return [np.concatenate([hidden, hidden[..., :2]], axis=-1)]
        hidden = feed[self.inputs[0]] + 1
        old_key = feed[f"past_key_values.{self.layer}.key"]
        old_value = feed[f"past_key_values.{self.layer}.value"]
        added = np.zeros((1, 1, hidden.shape[1], 8), dtype=np.float32)
        return [
            hidden,
            np.concatenate([old_key, added], axis=2),
            np.concatenate([old_value, added], axis=2),
        ]


def descriptor(kind, start, end):
    stage_id = kind if kind != "decoder" else f"decoder-{start:03d}"
    return ExecutionStageDescriptor(
        stage_id=stage_id,
        kind=ExecutionStageKind(kind),
        start_layer=start,
        end_layer=end,
        graph_path=f"execution/{stage_id}.onnx",
        io_contract_hash=HASH,
    )


def verified_span(tmp_path, *, fail_layer=None):
    descriptors = (
        descriptor("input", 0, 0),
        descriptor("decoder", 0, 1),
        descriptor("decoder", 1, 2),
        descriptor("output", 2, 2),
    )
    stages = tuple(
        VerifiedExecutionStage(
            descriptor=item,
            graph_path=tmp_path / f"{item.stage_id}.onnx",
            external_data_paths=(),
        )
        for item in descriptors
    )
    plan = ModelExecutionPlan(
        plan_id="cpu-test",
        backend=BackendKind.ONNXRUNTIME,
        precision="fp32",
        quantization="none",
        exporter="microsoft/onnxruntime-genai",
        exporter_revision="a" * 40,
        artifact_repository_id="fabi/test",
        artifact_revision="b" * 40,
        export_target=OnnxExportTarget.ORT_GENAI_CPU,
        activation_dtype="float32",
        activation_hidden_size=8,
        kv_num_heads=1,
        kv_head_dim=8,
        providers=(ExecutionProviderKind.CPU,),
        stages=descriptors,
    )

    def factory(path, _provider):
        name = path.stem
        if name == "input":
            return FakeSession("input")
        if name == "output":
            return FakeSession("output")
        layer = int(name.rsplit("-", 1)[1])
        return FakeSession("decoder", layer, fail=layer == fail_layer)

    return (
        VerifiedExecutionSpan(
            plan=plan,
            span=LayerSpan(start=0, end=2),
            stages=stages,
            artifact_bytes=0,
        ),
        factory,
    )


def test_provider_configuration_profiles_directml_cpu_fallback():
    spec = ort_provider_spec("directml:2")

    assert spec.name == "DmlExecutionProvider"
    assert spec.options == {"device_id": "2"}
    assert not spec.disable_cpu_fallback
    assert spec.sequential_execution
    assert spec.disable_memory_pattern
    assert spec.profile_assignment


def test_profiled_assignment_accepts_only_signed_cpu_nodes():
    spec = ort_provider_spec("directml:0")
    events = [
        {
            "cat": "Node",
            "name": "/model/MatMul_kernel_time",
            "args": {"provider": "DmlExecutionProvider"},
        },
        {
            "cat": "Node",
            "name": "/mask/Gather_kernel_time",
            "args": {"provider": "CPUExecutionProvider"},
        },
    ]

    report = validate_profiled_provider_assignment(
        events, spec, allowed_cpu_nodes=("/mask/Gather",)
    )
    assert report.accelerator_nodes == 1
    assert report.cpu_nodes == ("/mask/Gather",)
    with pytest.raises(RuntimeError, match="unapproved"):
        validate_profiled_provider_assignment(events, spec, allowed_cpu_nodes=())


def test_profiled_assignment_allows_only_an_explicit_cpu_only_endpoint():
    spec = ort_provider_spec("directml:0")
    events = [
        {
            "cat": "Node",
            "name": "/embed/GatherBlockQuantized_kernel_time",
            "args": {"provider": "CPUExecutionProvider"},
        }
    ]
    report = validate_profiled_provider_assignment(
        events,
        spec,
        allowed_cpu_nodes=("/embed/GatherBlockQuantized",),
        require_accelerator=False,
    )
    assert report.accelerator_nodes == 0
    with pytest.raises(RuntimeError, match="no graph node"):
        validate_profiled_provider_assignment(
            events,
            spec,
            allowed_cpu_nodes=("/embed/GatherBlockQuantized",),
        )


def test_runner_commits_prefill_and_decode_kv_atomically(tmp_path):
    verified, factory = verified_span(tmp_path)
    runner = OnnxRuntimeStageRunner(
        verified,
        device="cpu",
        max_context_tokens=8,
        max_sessions=1,
        session_factory=factory,
    )

    prefill = runner.forward("request-1", [1, 2, 3])
    decode = runner.forward("request-1", [4])

    assert prefill.is_logits and prefill.values.shape == (1, 3, 10)
    assert decode.is_logits and decode.values.shape == (1, 1, 10)
    assert decode.cache_tokens == 4
    assert runner.resident_kv_bytes == 2 * 2 * 4 * 8 * 4
    runner.release("request-1")
    assert runner.active_sessions == 0


def test_runner_does_not_commit_partial_kv_after_provider_failure(tmp_path):
    verified, factory = verified_span(tmp_path, fail_layer=1)
    runner = OnnxRuntimeStageRunner(
        verified,
        device="cpu",
        max_context_tokens=8,
        max_sessions=1,
        session_factory=factory,
    )

    with pytest.raises(RuntimeError, match="provider failure"):
        runner.forward("request-1", [1, 2])

    assert runner.active_sessions == 0
    assert runner.resident_kv_bytes == 0


def test_runner_enforces_context_and_session_limits(tmp_path):
    verified, factory = verified_span(tmp_path)
    runner = OnnxRuntimeStageRunner(
        verified,
        device="cpu",
        max_context_tokens=2,
        max_sessions=1,
        session_factory=factory,
    )
    runner.forward("request-1", [1, 2])

    with pytest.raises(RuntimeError, match="no free request session"):
        runner.forward("request-2", [1])
    with pytest.raises(RuntimeError, match="exceeds"):
        runner.forward("request-1", [3])
