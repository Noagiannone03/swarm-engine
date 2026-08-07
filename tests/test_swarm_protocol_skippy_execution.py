import hashlib
import json
from types import SimpleNamespace

import pytest

from swarm_protocol.contracts import (
    ArtifactDescriptor,
    ArtifactRole,
    ExecutionProviderKind,
    LayerSpan,
    ModelArtifactIndex,
    ModelManifest,
    SkippyExecutionPlan,
    SkippyRuntimeFeature,
)
from swarm_protocol.model_manifest import (
    artifact_collection_hash,
    execution_plan_hash,
    execution_plan_identity_hash,
)
from swarm_protocol.registry import ModelRegistryBundle
from swarm_protocol.skippy_package_import import attach_skippy_direct_gguf, attach_skippy_package
from swarm_protocol.skippy_execution import (
    materialize_skippy_execution_span,
    required_skippy_storage_bytes,
    select_skippy_execution_plan,
    skippy_execution_provider_for_device,
    skippy_span_static_bytes,
    verify_skippy_execution_span,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact(path: str, role: ArtifactRole, payload: bytes) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        path=path,
        size=len(payload),
        sha256=_sha(payload),
        media_type="application/vnd.gguf" if path.endswith(".gguf") else "application/json",
        role=role,
    )


def _fixture(*, package_layer_count: int = 2):
    contents = {
        "config.json": b"config",
        "model.safetensors": b"source-weights",
        "tokenizer.json": b"tokenizer",
        "shared/metadata.gguf": b"metadata",
        "shared/embeddings.gguf": b"embeddings",
        "shared/output.gguf": b"output",
        "layers/layer-000.gguf": b"layer-0",
        "layers/layer-001.gguf": b"layer-1",
    }

    def package_entry(path: str) -> dict[str, object]:
        payload = contents[path]
        return {
            "path": path,
            "tensor_count": 0 if path.endswith("metadata.gguf") else 1,
            "tensor_bytes": 0 if path.endswith("metadata.gguf") else len(payload),
            "artifact_bytes": len(payload),
            "sha256": _sha(payload),
        }

    package = {
        "schema_version": 1,
        "model_id": "unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
        "source_model": {
            "path": "/source/model.gguf",
            "sha256": "a" * 64,
        },
        "format": "layer-package",
        "layer_count": package_layer_count,
        "activation_width": 1024,
        "shared": {
            "metadata": package_entry("shared/metadata.gguf"),
            "embeddings": package_entry("shared/embeddings.gguf"),
            "output": package_entry("shared/output.gguf"),
        },
        "layers": [
            package_entry("layers/layer-000.gguf") | {"layer_index": 0},
            package_entry("layers/layer-001.gguf") | {"layer_index": 1},
        ],
        "skippy_abi_version": "0.1.24",
    }
    contents["model-package.json"] = json.dumps(
        package,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    roles = {
        "config.json": ArtifactRole.ARCHITECTURE,
        "model.safetensors": ArtifactRole.WEIGHT,
        "tokenizer.json": ArtifactRole.TOKENIZER,
        "model-package.json": ArtifactRole.EXECUTION_PACKAGE_MANIFEST,
        "shared/metadata.gguf": ArtifactRole.EXECUTION_SHARED,
        "shared/embeddings.gguf": ArtifactRole.EXECUTION_SHARED,
        "shared/output.gguf": ArtifactRole.EXECUTION_SHARED,
        "layers/layer-000.gguf": ArtifactRole.EXECUTION_LAYER,
        "layers/layer-001.gguf": ArtifactRole.EXECUTION_LAYER,
    }
    artifacts = tuple(
        sorted(
            (_artifact(path, roles[path], payload) for path, payload in contents.items()),
            key=lambda descriptor: descriptor.path,
        )
    )
    plan = SkippyExecutionPlan(
        plan_id="skippy-q4-k-m-v1",
        quantization="Q4_K_M",
        package_repository_id="meshllm/Qwen3-0.6B-Q4_K_M-layers",
        package_revision="1" * 40,
        package_manifest_path="model-package.json",
        package_manifest_sha256=_sha(contents["model-package.json"]),
        package_model_id="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
        package_source_sha256="a" * 64,
        package_abi_version="0.1.24",
        runtime_release="mesh-llm/v0.74.0",
        runtime_abi_version="0.1.32",
        required_runtime_features=(
            SkippyRuntimeFeature.ACTIVATION_FRAME,
            SkippyRuntimeFeature.BACKEND_DEVICES,
            SkippyRuntimeFeature.GENERATION_SIGNALS,
            SkippyRuntimeFeature.LAYER_PACKAGE,
            SkippyRuntimeFeature.SESSION_RESET,
        ),
        activation_width=1024,
        activation_bytes_per_token=4096,
        cache_type_k="f16",
        cache_type_v="f16",
        kv_bytes_per_token_by_layer=(256, 256),
        model_max_context_tokens=32768,
        providers=(
            ExecutionProviderKind.CPU,
            ExecutionProviderKind.CUDA,
            ExecutionProviderKind.METAL,
            ExecutionProviderKind.ROCM,
            ExecutionProviderKind.VULKAN,
        ),
        shared_metadata_path="shared/metadata.gguf",
        embeddings_path="shared/embeddings.gguf",
        output_path="shared/output.gguf",
        layer_paths=("layers/layer-000.gguf", "layers/layer-001.gguf"),
    )
    index = ModelArtifactIndex(
        model_id="Qwen/Qwen3-0.6B",
        immutable_revision="2" * 40,
        artifacts=artifacts,
        execution_plans=(plan,),
    )
    model = ModelManifest(
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
        activation_bytes_per_token=2048,
        kv_bytes_per_token_by_layer=(256, 256),
        weight_bytes_by_layer=(1024, 1024),
        input_endpoint_weight_bytes=128,
        output_endpoint_weight_bytes=128,
        rope_context_contract_hash="3" * 64,
        attention_kv_contract_hash="4" * 64,
        prefill_contract_hash="5" * 64,
        wire_protocol_version=1,
        execution_plan_hash=execution_plan_hash(index),
    )
    return contents, index, model


def _write_contents(root, contents) -> None:
    for path, payload in contents.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def test_skippy_plan_is_bound_by_registry_and_supports_native_devices():
    _, index, model = _fixture()

    bundle = ModelRegistryBundle(manifest=model, artifact_index=index)

    assert select_skippy_execution_plan(index, device="vulkan:0").plan_id == "skippy-q4-k-m-v1"
    assert skippy_execution_provider_for_device("metal") is ExecutionProviderKind.METAL
    assert bundle.manifest.execution_plan_hash == execution_plan_hash(index)
    identity = execution_plan_identity_hash(index, index.execution_plans[0])
    assert len(identity) == 64


def test_skippy_span_downloads_only_owned_layers_and_boundary(tmp_path):
    contents, index, model = _fixture()
    _write_contents(tmp_path, contents)
    calls = []

    def snapshot_downloader(**kwargs):
        calls.append(kwargs)
        return tmp_path

    span = LayerSpan(start=0, end=1)
    verified = materialize_skippy_execution_span(
        index,
        model,
        span,
        device="vulkan:0",
        snapshot_downloader=snapshot_downloader,
    )

    assert calls[0]["allow_patterns"] == [
        "model-package.json",
        "shared/metadata.gguf",
        "shared/embeddings.gguf",
        "layers/layer-000.gguf",
    ]
    assert [path.relative_to(tmp_path).as_posix() for path in verified.part_paths] == [
        "shared/metadata.gguf",
        "shared/embeddings.gguf",
        "layers/layer-000.gguf",
    ]
    assert verified.artifact_bytes == required_skippy_storage_bytes(
        index,
        verified.plan,
        model,
        span,
    )


def test_skippy_final_span_gets_output_but_not_embeddings(tmp_path):
    contents, index, model = _fixture()
    _write_contents(tmp_path, contents)

    verified = verify_skippy_execution_span(
        tmp_path,
        index,
        model,
        LayerSpan(start=1, end=2),
        device="cpu",
    )

    assert [path.relative_to(tmp_path).as_posix() for path in verified.part_paths] == [
        "shared/metadata.gguf",
        "layers/layer-001.gguf",
        "shared/output.gguf",
    ]


def test_skippy_manifest_semantics_are_verified_after_file_hash(tmp_path):
    contents, index, model = _fixture(package_layer_count=1)
    _write_contents(tmp_path, contents)

    with pytest.raises(ValueError, match="layer_count"):
        verify_skippy_execution_span(
            tmp_path,
            index,
            model,
            LayerSpan(start=0, end=1),
            device="cuda:0",
        )


def test_skippy_plan_rejects_newer_package_minor_abi():
    _, index, _ = _fixture()
    payload = index.execution_plans[0].model_dump(mode="json")
    payload["package_abi_version"] = "0.2.0"

    with pytest.raises(ValueError, match="incompatible"):
        SkippyExecutionPlan.model_validate(payload)


def test_existing_hub_layer_package_is_imported_at_immutable_revision(tmp_path):
    contents, populated_index, populated_model = _fixture()
    execution_roles = {
        ArtifactRole.EXECUTION_LAYER,
        ArtifactRole.EXECUTION_PACKAGE_MANIFEST,
        ArtifactRole.EXECUTION_SHARED,
    }
    base_index = ModelArtifactIndex(
        model_id=populated_index.model_id,
        immutable_revision=populated_index.immutable_revision,
        artifacts=tuple(
            artifact
            for artifact in populated_index.artifacts
            if artifact.role not in execution_roles
        ),
    )
    base_model = populated_model.model_copy(update={"execution_plan_hash": None})
    base_bundle = ModelRegistryBundle(manifest=base_model, artifact_index=base_index)
    package_path = tmp_path / "model-package.json"
    package_path.write_bytes(contents["model-package.json"])

    siblings = []
    for artifact in populated_index.artifacts:
        if artifact.role not in execution_roles:
            continue
        lfs = None
        if artifact.path != "model-package.json":
            lfs = SimpleNamespace(sha256=artifact.sha256, size=artifact.size)
        siblings.append(SimpleNamespace(rfilename=artifact.path, size=artifact.size, lfs=lfs))
    api = SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(sha="9" * 40, siblings=siblings)
    )

    attached = attach_skippy_package(
        base_bundle,
        package_repository_id="meshllm/Qwen3-0.6B-Q4_K_M-layers",
        package_revision="main",
        plan_id="skippy-q4-k-m-v1",
        runtime_release="mesh-llm/v0.74.0",
        runtime_abi_version="0.1.32",
        api=api,
        downloader=lambda **kwargs: str(package_path),
    )

    plan = attached.artifact_index.execution_plans[0]
    assert isinstance(plan, SkippyExecutionPlan)
    assert plan.package_revision == "9" * 40
    assert plan.package_repository_id == "meshllm/Qwen3-0.6B-Q4_K_M-layers"
    assert len(plan.layer_paths) == 2
    assert attached.manifest.execution_plan_hash == execution_plan_hash(attached.artifact_index)


def test_direct_gguf_needs_no_layer_package_and_loads_as_runtime_slice(tmp_path):
    _, populated_index, populated_model = _fixture()
    execution_roles = {
        ArtifactRole.EXECUTION_LAYER,
        ArtifactRole.EXECUTION_PACKAGE_MANIFEST,
        ArtifactRole.EXECUTION_SHARED,
    }
    base_index = ModelArtifactIndex(
        model_id=populated_index.model_id,
        immutable_revision=populated_index.immutable_revision,
        artifacts=tuple(
            artifact
            for artifact in populated_index.artifacts
            if artifact.role not in execution_roles
        ),
    )
    base_bundle = ModelRegistryBundle(
        manifest=populated_model.model_copy(update={"execution_plan_hash": None}),
        artifact_index=base_index,
    )
    payload = b"ordinary-gguf-source"
    source = tmp_path / "model-q4.gguf"
    source.write_bytes(payload)
    sibling = SimpleNamespace(
        rfilename="model-q4.gguf",
        size=len(payload),
        lfs=SimpleNamespace(sha256=_sha(payload), size=len(payload)),
    )
    api = SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(
            sha="8" * 40,
            siblings=(sibling,),
        )
    )
    geometry = SimpleNamespace(
        activation_width=1024,
        context_length=32768,
        kv_bytes_per_token=512,
        layer_count=2,
        static_bytes_by_layer=(700, 800),
    )
    attached = attach_skippy_direct_gguf(
        base_bundle,
        repository_id="community/Qwen3-0.6B-GGUF",
        revision="main",
        source_paths=("model-q4.gguf",),
        plan_id="skippy-direct-q4",
        quantization="Q4_K_M",
        runtime_release="mesh-llm/v0.74.0",
        runtime_abi_version="0.1.32",
        api=api,
        downloader=lambda **kwargs: str(source),
        geometry_inspector=lambda paths, cache_k, cache_v: geometry,
    )

    plan = attached.artifact_index.execution_plans[0]
    assert isinstance(plan, SkippyExecutionPlan)
    assert plan.format == "gguf-direct"
    assert plan.source_model_paths == ("model-q4.gguf",)
    assert SkippyRuntimeFeature.RUNTIME_SLICE in plan.required_runtime_features
    assert SkippyRuntimeFeature.LAYER_PACKAGE not in plan.required_runtime_features
    assert skippy_span_static_bytes(
        attached.artifact_index,
        plan,
        attached.manifest,
        LayerSpan(start=0, end=1),
    ) == 700

    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "model-q4.gguf").write_bytes(payload)
    verified = materialize_skippy_execution_span(
        attached.artifact_index,
        attached.manifest,
        LayerSpan(start=1, end=2),
        device="cpu",
        snapshot_downloader=lambda **kwargs: root,
    )
    assert verified.part_paths == (root / "model-q4.gguf",)
    assert verified.geometry_path == root / "model-q4.gguf"
    assert verified.artifact_bytes == len(payload)
