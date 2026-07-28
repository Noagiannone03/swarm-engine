import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from parallax.utils import selective_safetensors
from parallax.utils.selective_safetensors import (
    materialize_tensor_span,
    selective_receipt_path,
)
from swarm_protocol import (
    ArtifactDescriptor,
    ArtifactRole,
    LayerSpan,
    ModelArtifactIndex,
    ModelManifest,
    TensorArtifactDescriptor,
    artifact_collection_hash,
    verify_worker_span,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _descriptor(root: Path, path: str, role: ArtifactRole) -> ArtifactDescriptor:
    content = (root / path).read_bytes()
    return ArtifactDescriptor(
        path=path,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        media_type=(
            "application/vnd.safetensors" if path.endswith(".safetensors") else "application/json"
        ),
        role=role,
        xet_file_hash="ab" * 32 if path.endswith(".safetensors") else None,
    )


def _fixture(root: Path) -> tuple[ModelArtifactIndex, ModelManifest, bytes]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "hidden_size": 4,
                "tie_word_embeddings": False,
            }
        )
    )
    (root / "tokenizer.json").write_text("{}")
    tensors = {
        "model.embed_tokens.weight": np.arange(24, dtype=np.float32).reshape(6, 4),
        "model.layers.0.self_attn.q_proj.weight": np.arange(16, dtype=np.float32).reshape(4, 4),
        "model.layers.1.self_attn.q_proj.weight": np.arange(16, 32, dtype=np.float32).reshape(4, 4),
        "model.norm.weight": np.arange(4, dtype=np.float32),
        "lm_head.weight": np.arange(24, 48, dtype=np.float32).reshape(6, 4),
    }
    source_path = root / "model.safetensors"
    save_file(tensors, source_path)
    source_bytes = source_path.read_bytes()
    header_length = struct.unpack("<Q", source_bytes[:8])[0]
    header = json.loads(source_bytes[8 : 8 + header_length])
    data_start = 8 + header_length
    tensor_descriptors = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        start, end = metadata["data_offsets"]
        payload = source_bytes[data_start + start : data_start + end]
        tensor_descriptors.append(
            TensorArtifactDescriptor(
                name=name,
                source_path="model.safetensors",
                offset=data_start + start,
                length=end - start,
                sha256=hashlib.sha256(payload).hexdigest(),
                dtype=metadata["dtype"],
                shape=tuple(metadata["shape"]),
            )
        )
    artifacts = tuple(
        sorted(
            (
                _descriptor(root, "config.json", ArtifactRole.ARCHITECTURE),
                _descriptor(root, "tokenizer.json", ArtifactRole.TOKENIZER),
                _descriptor(root, "model.safetensors", ArtifactRole.WEIGHT),
            ),
            key=lambda item: item.path,
        )
    )
    index = ModelArtifactIndex(
        model_id="test/selective",
        immutable_revision=REVISION,
        artifacts=artifacts,
        tensors=tuple(sorted(tensor_descriptors, key=lambda item: item.name)),
    )
    manifest = ModelManifest(
        model_id=index.model_id,
        immutable_revision=REVISION,
        architecture_graph_hash=artifact_collection_hash(index, ArtifactRole.ARCHITECTURE),
        tokenizer_hash=artifact_collection_hash(index, ArtifactRole.TOKENIZER),
        weight_collection_hash=artifact_collection_hash(index, ArtifactRole.WEIGHT),
        weight_format="safetensors",
        quantization="unquantized",
        dtype="float32",
        num_layers=2,
        activation_bytes_per_token=16,
        kv_bytes_per_token_by_layer=(8, 8),
        rope_context_contract_hash="1" * 64,
        attention_kv_contract_hash="2" * 64,
        prefill_contract_hash="3" * 64,
        wire_protocol_version=1,
    )
    return index, manifest, source_bytes


def _keys(path: Path) -> set[str]:
    with safe_open(path, framework="numpy") as model:
        return set(model.keys())


def test_layer_packs_are_reused_and_reallocation_fetches_only_new_layers(tmp_path):
    metadata_root = tmp_path / "metadata"
    cache_root = tmp_path / "cache"
    index, manifest, _ = _fixture(metadata_root)

    projection = materialize_tensor_span(
        repo_id=index.model_id,
        immutable_revision=REVISION,
        metadata_root=metadata_root,
        artifact_index=index,
        start_layer=0,
        end_layer=1,
        local_files_only=True,
        cache_root=cache_root,
    )
    layer_zero = projection / "model-fabi-layer-00000.safetensors"
    embedding = projection / "model-fabi-embedding.safetensors"
    assert _keys(layer_zero) == {"model.layers.0.self_attn.q_proj.weight"}
    assert _keys(embedding) == {"model.embed_tokens.weight"}
    assert not (projection / "model-fabi-layer-00001.safetensors").exists()
    first_mtime = layer_zero.stat().st_mtime_ns

    same_projection = materialize_tensor_span(
        repo_id=index.model_id,
        immutable_revision=REVISION,
        metadata_root=metadata_root,
        artifact_index=index,
        start_layer=0,
        end_layer=2,
        local_files_only=True,
        cache_root=cache_root,
    )
    assert same_projection == projection
    assert layer_zero.stat().st_mtime_ns == first_mtime
    assert _keys(projection / "model-fabi-layer-00001.safetensors") == {
        "model.layers.1.self_attn.q_proj.weight"
    }
    assert _keys(projection / "model-fabi-output.safetensors") == {
        "lm_head.weight",
        "model.norm.weight",
    }

    verified = verify_worker_span(
        projection,
        index,
        manifest,
        LayerSpan(start=0, end=2),
        include_tokenizer=True,
    )
    assert {path.name for path in verified.weight_paths} == {
        "model.safetensors.index.json",
        "model-fabi-embedding.safetensors",
        "model-fabi-layer-00000.safetensors",
        "model-fabi-layer-00001.safetensors",
        "model-fabi-output.safetensors",
    }


def test_signed_tensor_hash_rejects_pack_and_receipt_tampering(tmp_path):
    metadata_root = tmp_path / "metadata"
    index, manifest, _ = _fixture(metadata_root)
    projection = materialize_tensor_span(
        repo_id=index.model_id,
        immutable_revision=REVISION,
        metadata_root=metadata_root,
        artifact_index=index,
        start_layer=0,
        end_layer=1,
        local_files_only=True,
        cache_root=tmp_path / "cache",
    )
    pack = projection / "model-fabi-layer-00000.safetensors"
    content = bytearray(pack.read_bytes())
    content[-1] ^= 1
    pack.write_bytes(content)
    receipt_path = selective_receipt_path(projection)
    receipt = json.loads(receipt_path.read_bytes())
    receipt["packs"][pack.name]["sha256"] = hashlib.sha256(content).hexdigest()
    receipt_path.write_text(json.dumps(receipt, separators=(",", ":"), sort_keys=True))

    with pytest.raises(ValueError, match="projected tensor SHA-256 mismatch"):
        verify_worker_span(projection, index, manifest, LayerSpan(start=0, end=1))


def test_offline_growth_fails_when_neither_pack_nor_full_source_is_cached(tmp_path):
    metadata_root = tmp_path / "metadata"
    index, _, _ = _fixture(metadata_root)
    projection = materialize_tensor_span(
        repo_id=index.model_id,
        immutable_revision=REVISION,
        metadata_root=metadata_root,
        artifact_index=index,
        start_layer=0,
        end_layer=1,
        local_files_only=True,
        cache_root=tmp_path / "cache",
    )
    (metadata_root / "model.safetensors").unlink()

    assert (
        materialize_tensor_span(
            repo_id=index.model_id,
            immutable_revision=REVISION,
            metadata_root=metadata_root,
            artifact_index=index,
            start_layer=0,
            end_layer=1,
            local_files_only=True,
            cache_root=tmp_path / "cache",
        )
        == projection
    )
    with pytest.raises(FileNotFoundError, match="not available offline"):
        materialize_tensor_span(
            repo_id=index.model_id,
            immutable_revision=REVISION,
            metadata_root=metadata_root,
            artifact_index=index,
            start_layer=0,
            end_layer=2,
            local_files_only=True,
            cache_root=tmp_path / "cache",
        )


def test_remote_materialization_requests_only_selected_signed_ranges(tmp_path, monkeypatch):
    metadata_root = tmp_path / "metadata"
    index, _, source_bytes = _fixture(metadata_root)
    (metadata_root / "model.safetensors").unlink()
    requested = []

    class Group:
        def download_stream(self, file_info, *, start, end):
            requested.append((start, end))
            yield source_bytes[start:end]

    monkeypatch.setattr(selective_safetensors._RangeSources, "_xet_group", lambda *_: Group())
    projection = materialize_tensor_span(
        repo_id=index.model_id,
        immutable_revision=REVISION,
        metadata_root=metadata_root,
        artifact_index=index,
        start_layer=1,
        end_layer=2,
        cache_root=tmp_path / "cache",
    )

    selected_names = {
        "model.layers.1.self_attn.q_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    selected = [tensor for tensor in index.tensors if tensor.name in selected_names]
    assert sum(end - start for start, end in requested) == sum(tensor.length for tensor in selected)
    assert not (projection / "model-fabi-layer-00000.safetensors").exists()
