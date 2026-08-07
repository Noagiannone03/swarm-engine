import hashlib
import json

import pytest

from swarm_protocol.contracts import (
    ArtifactDescriptor,
    ArtifactRole,
    BackendKind,
    ExecutionProviderKind,
    ExecutionStageDescriptor,
    ExecutionStageKind,
    ModelArtifactIndex,
    ModelExecutionPlan,
    ModelManifest,
    OnnxExportTarget,
)
from swarm_protocol.model_manifest import artifact_collection_hash, execution_plan_hash
from swarm_protocol.registry import ModelRegistryBundle
from swarm_protocol.contracts import LayerSpan
from swarm_protocol.portable_execution import (
    execution_provider_for_device,
    materialize_execution_span,
    portable_span_static_bytes,
    required_execution_storage_bytes,
    select_execution_plan,
    stages_for_span,
    verify_execution_span,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def artifact(path: str, role: ArtifactRole, content: str) -> ArtifactDescriptor:
    encoded = content.encode()
    return ArtifactDescriptor(
        path=path,
        size=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
        media_type="application/onnx" if path.endswith(".onnx") else "application/octet-stream",
        role=role,
    )


def portable_index(*, decoder_ranges=((0, 1), (1, 2))) -> ModelArtifactIndex:
    artifacts = tuple(
        sorted(
            (
                artifact("config.json", ArtifactRole.ARCHITECTURE, "config"),
                artifact("execution/decoder-000.onnx", ArtifactRole.EXECUTION_GRAPH, "decoder0"),
                artifact("execution/decoder-001.onnx", ArtifactRole.EXECUTION_GRAPH, "decoder1"),
                artifact("execution/input.onnx", ArtifactRole.EXECUTION_GRAPH, "input"),
                artifact("execution/output.onnx", ArtifactRole.EXECUTION_GRAPH, "output"),
                artifact("execution/weights.data", ArtifactRole.EXECUTION_DATA, "portable-weights"),
                artifact("model.safetensors", ArtifactRole.WEIGHT, "weights"),
                artifact("tokenizer.json", ArtifactRole.TOKENIZER, "tokenizer"),
            ),
            key=lambda item: item.path,
        )
    )
    decoder_paths = ("execution/decoder-000.onnx", "execution/decoder-001.onnx")
    stages = [
        ExecutionStageDescriptor(
            stage_id="input",
            kind=ExecutionStageKind.INPUT,
            start_layer=0,
            end_layer=0,
            graph_path="execution/input.onnx",
            external_data_paths=("execution/weights.data",),
            io_contract_hash=digest("input-io"),
        )
    ]
    for index, (start, end) in enumerate(decoder_ranges):
        stages.append(
            ExecutionStageDescriptor(
                stage_id=f"decoder-{index:03d}",
                kind=ExecutionStageKind.DECODER,
                start_layer=start,
                end_layer=end,
                graph_path=decoder_paths[index],
                external_data_paths=("execution/weights.data",),
                io_contract_hash=digest(f"decoder-{index}-io"),
            )
        )
    stages.append(
        ExecutionStageDescriptor(
            stage_id="output",
            kind=ExecutionStageKind.OUTPUT,
            start_layer=2,
            end_layer=2,
            graph_path="execution/output.onnx",
            external_data_paths=("execution/weights.data",),
            io_contract_hash=digest("output-io"),
        )
    )
    plan = ModelExecutionPlan(
        plan_id="onnx-int4-v1",
        backend=BackendKind.ONNXRUNTIME,
        precision="int4",
        quantization="rtn-block-32",
        exporter="microsoft/onnxruntime-genai",
        exporter_revision="d" * 40,
        artifact_repository_id="fabi-ai/Qwen3-4B-onnx-stages",
        artifact_revision="a" * 40,
        export_target=OnnxExportTarget.ORT_GENAI_DML,
        activation_dtype="float16",
        activation_hidden_size=2560,
        kv_num_heads=8,
        kv_head_dim=128,
        providers=(ExecutionProviderKind.DIRECTML, ExecutionProviderKind.WINML),
        stages=tuple(stages),
    )
    return ModelArtifactIndex(
        model_id="Qwen/Qwen3-4B",
        immutable_revision="revision-1",
        artifacts=artifacts,
        execution_plans=(plan,),
    )


def manifest(index: ModelArtifactIndex, *, bind_plan: bool = True) -> ModelManifest:
    return ModelManifest(
        model_id=index.model_id,
        immutable_revision=index.immutable_revision,
        architecture_graph_hash=artifact_collection_hash(index, ArtifactRole.ARCHITECTURE),
        tokenizer_hash=artifact_collection_hash(index, ArtifactRole.TOKENIZER),
        weight_collection_hash=artifact_collection_hash(index, ArtifactRole.WEIGHT),
        weight_format="safetensors",
        quantization="bf16",
        dtype="bfloat16",
        num_layers=2,
        model_max_context_tokens=32768,
        context_classes=(16384, 32768),
        activation_bytes_per_token=4096,
        kv_bytes_per_token_by_layer=(256, 256),
        weight_bytes_by_layer=(1024, 1024),
        input_endpoint_weight_bytes=128,
        output_endpoint_weight_bytes=128,
        rope_context_contract_hash=digest("rope"),
        attention_kv_contract_hash=digest("kv"),
        prefill_contract_hash=digest("prefill"),
        wire_protocol_version=1,
        execution_plan_hash=execution_plan_hash(index) if bind_plan else None,
    )


def test_signed_portable_plan_tiles_layers_and_validates():
    index = portable_index()

    bundle = ModelRegistryBundle(manifest=manifest(index), artifact_index=index)

    assert bundle.artifact_index.execution_plans[0].providers == (
        ExecutionProviderKind.DIRECTML,
        ExecutionProviderKind.WINML,
    )


def test_portable_static_bytes_reject_stage_cuts_without_hiding_contract_errors():
    index = portable_index(decoder_ranges=((0, 2),))
    model = manifest(index)
    plan = index.execution_plans[0]

    assert (
        portable_span_static_bytes(
            index,
            plan,
            model,
            LayerSpan(start=0, end=1),
        )
        is None
    )
    assert portable_span_static_bytes(
        index,
        plan,
        model,
        LayerSpan(start=0, end=2),
    ) == required_execution_storage_bytes(
        index,
        plan,
        model,
        LayerSpan(start=0, end=2),
    )


def test_execution_plan_cpu_fallback_policy_is_signed_and_stage_exact():
    base = portable_index().execution_plans[0]
    plan = base.model_copy(
        update={
            "allowed_cpu_fallback_nodes": ("/mask/Gather",),
            "allowed_cpu_only_stages": ("input",),
        }
    )
    validated = ModelExecutionPlan.model_validate(plan.model_dump())
    assert validated.allowed_cpu_only_stages == ("input",)

    with pytest.raises(ValueError, match="unknown stages"):
        ModelExecutionPlan.model_validate(
            plan.model_copy(update={"allowed_cpu_only_stages": ("missing",)}).model_dump()
        )


@pytest.mark.parametrize("field", ["exporter_revision", "artifact_revision"])
def test_portable_plan_rejects_mutable_revisions(field):
    plan = portable_index().execution_plans[0]

    with pytest.raises(ValueError):
        plan.model_copy(update={field: "main"}).__class__.model_validate(
            plan.model_dump(mode="json") | {field: "main"}
        )


def test_portable_plan_requires_manifest_binding():
    index = portable_index()

    with pytest.raises(ValueError, match="not bound"):
        ModelRegistryBundle(manifest=manifest(index, bind_plan=False), artifact_index=index)


def test_dml_export_cannot_claim_unqualified_openvino_provider():
    plan = portable_index().execution_plans[0]
    payload = plan.model_dump(mode="json")
    payload["providers"] = ["openvino"]

    with pytest.raises(ValueError, match="cannot claim providers"):
        ModelExecutionPlan.model_validate(payload)


def test_portable_plan_rejects_layer_gaps():
    index = portable_index(decoder_ranges=((0, 1), (2, 3)))

    with pytest.raises(ValueError, match="tile model layers exactly"):
        ModelRegistryBundle(manifest=manifest(index), artifact_index=index)


def test_portable_plan_hash_binds_referenced_graph_bytes():
    index = portable_index()
    changed_artifacts = tuple(
        (
            item.model_copy(update={"sha256": digest("tampered")})
            if item.path == "execution/input.onnx"
            else item
        )
        for item in index.artifacts
    )
    changed = ModelArtifactIndex(
        model_id=index.model_id,
        immutable_revision=index.immutable_revision,
        artifacts=changed_artifacts,
        execution_plans=index.execution_plans,
    )

    with pytest.raises(ValueError, match="execution plan hash"):
        ModelRegistryBundle(manifest=manifest(index), artifact_index=changed)


def test_legacy_model_swarm_identity_omits_absent_execution_plan_field():
    index = portable_index()
    legacy = manifest(index, bind_plan=False)
    old_shape = legacy.model_dump(mode="json", exclude={"execution_plan_hash"})
    expected = hashlib.sha256(
        json.dumps(old_shape, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()

    assert legacy.model_swarm_id == expected


def test_portable_plan_selection_is_device_specific_and_ambiguous_plans_fail():
    index = portable_index()

    assert execution_provider_for_device("directml:0") is ExecutionProviderKind.DIRECTML
    assert select_execution_plan(index, device="winml").plan_id == "onnx-int4-v1"
    with pytest.raises(ValueError, match="no portable execution provider"):
        execution_provider_for_device("cuda")
    with pytest.raises(ValueError, match="supports qnn"):
        select_execution_plan(index, device="qnn")


def test_worker_span_selects_only_owned_endpoints_and_aligned_decoders():
    index = portable_index()
    model = manifest(index)
    plan = index.execution_plans[0]

    first = stages_for_span(plan, model, LayerSpan(start=0, end=1))
    last = stages_for_span(plan, model, LayerSpan(start=1, end=2))

    assert [stage.kind for stage in first] == [
        ExecutionStageKind.INPUT,
        ExecutionStageKind.DECODER,
    ]
    assert [stage.kind for stage in last] == [
        ExecutionStageKind.DECODER,
        ExecutionStageKind.OUTPUT,
    ]


def test_worker_span_rejects_partial_signed_chunks():
    index = portable_index(decoder_ranges=((0, 2),))
    # The fixture allocates a second decoder graph but the plan intentionally
    # publishes one two-layer chunk.
    model = manifest(index)

    with pytest.raises(ValueError, match="not aligned"):
        stages_for_span(index.execution_plans[0], model, LayerSpan(start=0, end=1))


def test_execution_span_verifies_and_deduplicates_shared_external_data(tmp_path):
    index = portable_index()
    contents = {
        "config.json": "config",
        "execution/decoder-000.onnx": "decoder0",
        "execution/decoder-001.onnx": "decoder1",
        "execution/input.onnx": "input",
        "execution/output.onnx": "output",
        "execution/weights.data": "portable-weights",
        "model.safetensors": "weights",
        "tokenizer.json": "tokenizer",
    }
    for path, content in contents.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    model = manifest(index)
    span = LayerSpan(start=0, end=1)
    verified = verify_execution_span(
        tmp_path,
        index,
        model,
        span,
        device="directml",
    )

    expected_bytes = sum(
        item.size
        for item in index.artifacts
        if item.path
        in {
            "execution/input.onnx",
            "execution/decoder-000.onnx",
            "execution/weights.data",
        }
    )
    assert verified.artifact_bytes == expected_bytes
    assert required_execution_storage_bytes(index, verified.plan, model, span) == expected_bytes
    assert verified.weight_hashes == verified.artifact_hashes


def test_materialize_execution_span_downloads_only_signed_owned_files(tmp_path):
    index = portable_index()
    contents = {
        "execution/decoder-000.onnx": "decoder0",
        "execution/input.onnx": "input",
        "execution/weights.data": "portable-weights",
    }
    for path, content in contents.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return tmp_path

    verified = materialize_execution_span(
        index,
        manifest(index),
        LayerSpan(start=0, end=1),
        device="directml:0",
        snapshot_downloader=download,
    )

    assert calls == [
        {
            "repo_id": "fabi-ai/Qwen3-4B-onnx-stages",
            "revision": "a" * 40,
            "allow_patterns": [
                "execution/decoder-000.onnx",
                "execution/input.onnx",
                "execution/weights.data",
            ],
            "local_files_only": False,
            "token": None,
            "max_workers": 4,
        }
    ]
    assert verified.span == LayerSpan(start=0, end=1)
    assert len(verified.artifact_hashes) == 3
