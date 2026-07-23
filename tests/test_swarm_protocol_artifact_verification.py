import hashlib
import json
from pathlib import Path

import pytest

from swarm_protocol import (
    ArtifactDescriptor,
    ArtifactRole,
    LayerSpan,
    ModelArtifactIndex,
    ModelManifest,
    artifact_collection_hash,
    required_weight_descriptors,
    verify_worker_span,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _write(root: Path, path: str, content: bytes, role: ArtifactRole) -> ArtifactDescriptor:
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return ArtifactDescriptor(
        path=path,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        media_type="application/octet-stream",
        role=role,
    )


def _fixture(root: Path):
    config = json.dumps(
        {"num_hidden_layers": 4, "hidden_size": 64, "tie_word_embeddings": False}
    ).encode()
    tokenizer = b'{"tokenizer":"v1"}'
    shard_a = b"weights-a"
    shard_b = b"weights-b"
    weight_map = {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.2.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        "model.layers.3.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        "model.norm.weight": "model-00002-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    }
    index_bytes = json.dumps({"weight_map": weight_map}, sort_keys=True).encode()
    artifacts = tuple(
        sorted(
            (
                _write(root, "config.json", config, ArtifactRole.ARCHITECTURE),
                _write(root, "tokenizer.json", tokenizer, ArtifactRole.TOKENIZER),
                _write(
                    root,
                    "model.safetensors.index.json",
                    index_bytes,
                    ArtifactRole.WEIGHT,
                ),
                _write(
                    root,
                    "model-00001-of-00002.safetensors",
                    shard_a,
                    ArtifactRole.WEIGHT,
                ),
                _write(
                    root,
                    "model-00002-of-00002.safetensors",
                    shard_b,
                    ArtifactRole.WEIGHT,
                ),
            ),
            key=lambda artifact: artifact.path,
        )
    )
    artifact_index = ModelArtifactIndex(
        model_id="test/model",
        immutable_revision=REVISION,
        artifacts=artifacts,
    )
    manifest = ModelManifest(
        model_id="test/model",
        immutable_revision=REVISION,
        architecture_graph_hash=artifact_collection_hash(artifact_index, ArtifactRole.ARCHITECTURE),
        tokenizer_hash=artifact_collection_hash(artifact_index, ArtifactRole.TOKENIZER),
        weight_collection_hash=artifact_collection_hash(artifact_index, ArtifactRole.WEIGHT),
        weight_format="safetensors",
        quantization="unquantized",
        dtype="bfloat16",
        num_layers=4,
        activation_bytes_per_token=128,
        kv_bytes_per_token_by_layer=(64,) * 4,
        rope_context_contract_hash="1" * 64,
        attention_kv_contract_hash="2" * 64,
        prefill_contract_hash="3" * 64,
        wire_protocol_version=1,
    )
    return artifact_index, manifest


def test_layer_span_maps_to_exact_authenticated_weight_shards(tmp_path):
    artifact_index, manifest = _fixture(tmp_path)

    first = required_weight_descriptors(
        tmp_path, artifact_index, manifest, LayerSpan(start=0, end=2)
    )
    last = required_weight_descriptors(
        tmp_path, artifact_index, manifest, LayerSpan(start=2, end=4)
    )

    assert [artifact.path for artifact in first] == [
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
    ]
    assert [artifact.path for artifact in last] == [
        "model.safetensors.index.json",
        "model-00002-of-00002.safetensors",
    ]


def test_verified_span_hashes_are_ready_for_span_lease_binding(tmp_path):
    artifact_index, manifest = _fixture(tmp_path)
    verified = verify_worker_span(
        tmp_path,
        artifact_index,
        manifest,
        LayerSpan(start=0, end=2),
        include_tokenizer=True,
    )

    assert {path.name for path in verified.runtime_paths} == {"config.json", "tokenizer.json"}
    assert [path.name for path in verified.weight_paths] == [
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
    ]
    assert verified.weight_hashes == tuple(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in verified.weight_paths
    )


def test_worker_rejects_tampered_checkpoint_before_advertising(tmp_path):
    artifact_index, manifest = _fixture(tmp_path)
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"corrupt!!")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_worker_span(tmp_path, artifact_index, manifest, LayerSpan(start=0, end=2))


def test_worker_rejects_weight_map_reference_outside_signed_index(tmp_path):
    artifact_index, manifest = _fixture(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_bytes())
    index_data["weight_map"]["model.layers.0.weight"] = "unsigned.safetensors"
    content = json.dumps(index_data, sort_keys=True).encode()
    index_path.write_bytes(content)

    # Rebind the index digest so this test reaches the signed-reference check rather than the
    # earlier local-tampering check.
    artifacts = tuple(
        (
            ArtifactDescriptor(
                path=artifact.path,
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                media_type=artifact.media_type,
                role=artifact.role,
            )
            if artifact.path == "model.safetensors.index.json"
            else artifact
        )
        for artifact in artifact_index.artifacts
    )
    artifact_index = artifact_index.model_copy(update={"artifacts": artifacts})

    with pytest.raises(ValueError, match="unsigned checkpoint"):
        required_weight_descriptors(tmp_path, artifact_index, manifest, LayerSpan(start=0, end=2))


def test_worker_accepts_official_cache_symlink_only_when_target_hash_matches(tmp_path):
    artifact_index, manifest = _fixture(tmp_path)
    outside = tmp_path.parent / "outside-model-weight"
    outside.write_bytes(b"weights-a")
    checkpoint = tmp_path / "model-00001-of-00002.safetensors"
    checkpoint.unlink()
    checkpoint.symlink_to(outside)

    verified = verify_worker_span(tmp_path, artifact_index, manifest, LayerSpan(start=0, end=2))
    assert outside.resolve() in verified.weight_paths

    outside.write_bytes(b"not-the-signed-weight")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        verify_worker_span(tmp_path, artifact_index, manifest, LayerSpan(start=0, end=2))
