import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from swarm_protocol import ArtifactRole, build_hub_model_bundle
from swarm_protocol.contracts import ArtifactDescriptor
from swarm_protocol.model_manifest import (
    _get_safetensors_metadata_with_backoff,
    _hash_xet_tensor_ranges,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _lfs(content: bytes) -> SimpleNamespace:
    return SimpleNamespace(size=len(content), sha256=hashlib.sha256(content).hexdigest())


def _sibling(path: str, content: bytes, *, lfs: bool) -> SimpleNamespace:
    return SimpleNamespace(
        rfilename=path,
        size=len(content),
        blob_id="git-oid-is-not-a-content-sha256",
        lfs=_lfs(content) if lfs else None,
    )


class FakeApi:
    def __init__(self, siblings, *, metadata_revision: str = REVISION):
        self.siblings = siblings
        self.metadata_revision = metadata_revision
        self.calls = []

    def model_info(self, repo_id, **kwargs):
        self.calls.append((repo_id, kwargs))
        if kwargs.get("files_metadata"):
            return SimpleNamespace(sha=self.metadata_revision, siblings=self.siblings)
        return SimpleNamespace(sha=REVISION)


def _fixture():
    files = {
        "README.md": b"editorial changes must not fork the swarm",
        "config.json": json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "max_position_embeddings": 32768,
                "rope_theta": 1_000_000,
            },
            sort_keys=True,
        ).encode(),
        "modeling_qwen3.py": b"class Model: pass\n",
        "tokenizer.json": b'{"model":"tokenizer"}',
        "model.safetensors": b"weight-bytes-are-not-downloaded-in-this-test",
    }
    siblings = [
        _sibling("README.md", files["README.md"], lfs=False),
        _sibling("config.json", files["config.json"], lfs=False),
        _sibling("modeling_qwen3.py", files["modeling_qwen3.py"], lfs=False),
        _sibling("tokenizer.json", files["tokenizer.json"], lfs=True),
        _sibling("model.safetensors", files["model.safetensors"], lfs=True),
    ]
    reads = []

    def reader(repo_id, path, revision, token):
        reads.append((repo_id, path, revision, token))
        return files[path]

    return files, siblings, reads, reader


def test_builder_resolves_moving_ref_then_hashes_exact_runtime_artifacts():
    files, siblings, reads, reader = _fixture()
    api = FakeApi(siblings)

    bundle = build_hub_model_bundle(
        "Qwen/Qwen3-1.7B",
        revision="main",
        quantization="bf16",
        dtype="bfloat16",
        api=api,
        artifact_reader=reader,
    )

    assert [call[1] for call in api.calls] == [
        {"revision": "main", "token": None},
        {"revision": REVISION, "files_metadata": True, "token": None},
    ]
    assert [read[1] for read in reads] == ["config.json", "modeling_qwen3.py"]
    assert bundle.manifest.immutable_revision == REVISION
    assert bundle.manifest.num_layers == 28
    assert bundle.manifest.activation_bytes_per_token == 2048 * 2
    assert {artifact.path for artifact in bundle.artifact_index.artifacts} == {
        "config.json",
        "model.safetensors",
        "modeling_qwen3.py",
        "tokenizer.json",
    }
    weights = [
        artifact
        for artifact in bundle.artifact_index.artifacts
        if artifact.role is ArtifactRole.WEIGHT
    ]
    assert weights[0].sha256 == hashlib.sha256(files["model.safetensors"]).hexdigest()
    assert (
        bundle.manifest.model_swarm_id
        == build_hub_model_bundle(
            "Qwen/Qwen3-1.7B",
            revision=REVISION,
            quantization="bf16",
            dtype="bfloat16",
            api=FakeApi(list(reversed(siblings))),
            artifact_reader=reader,
        ).manifest.model_swarm_id
    )
    assert (
        bundle.manifest.model_swarm_id
        == build_hub_model_bundle(
            "Qwen/Qwen3-1.7B",
            revision=REVISION,
            quantization="bfloat16",
            dtype="bf16",
            api=FakeApi(siblings),
            artifact_reader=reader,
        ).manifest.model_swarm_id
    )


def test_builder_never_uses_git_oid_as_sha256_and_detects_download_tampering():
    _, siblings, _, reader = _fixture()

    def tampered_reader(repo_id, path, revision, token):
        content = reader(repo_id, path, revision, token)
        return b"tampered" if path == "config.json" else content

    with pytest.raises(ValueError, match="downloaded size"):
        build_hub_model_bundle(
            "Qwen/Qwen3-1.7B",
            quantization="bf16",
            dtype="bfloat16",
            api=FakeApi(siblings),
            artifact_reader=tampered_reader,
        )


def test_builder_rejects_metadata_from_a_different_commit():
    _, siblings, _, reader = _fixture()
    with pytest.raises(ValueError, match="does not match"):
        build_hub_model_bundle(
            "Qwen/Qwen3-1.7B",
            quantization="bf16",
            dtype="bfloat16",
            api=FakeApi(siblings, metadata_revision="f" * 40),
            artifact_reader=reader,
        )


def test_builder_rejects_unsafe_paths_before_reading():
    _, siblings, reads, reader = _fixture()
    siblings.insert(0, _sibling("../config.json", b"{}", lfs=False))
    with pytest.raises(ValueError, match="unsafe"):
        build_hub_model_bundle(
            "Qwen/Qwen3-1.7B",
            quantization="bf16",
            dtype="bfloat16",
            api=FakeApi(siblings),
            artifact_reader=reader,
        )
    assert reads == []


def test_builder_rejects_inconsistent_lfs_size():
    _, siblings, _, reader = _fixture()
    weight = next(sibling for sibling in siblings if sibling.rfilename == "model.safetensors")
    weight.lfs.size += 1
    with pytest.raises(ValueError, match="LFS size"):
        build_hub_model_bundle(
            "Qwen/Qwen3-1.7B",
            quantization="bf16",
            dtype="bfloat16",
            api=FakeApi(siblings),
            artifact_reader=reader,
        )


def test_runtime_code_and_dtype_fork_the_swarm_but_readme_does_not():
    files, siblings, _, reader = _fixture()
    base = build_hub_model_bundle(
        "Qwen/Qwen3-1.7B",
        quantization="bf16",
        dtype="bfloat16",
        api=FakeApi(siblings),
        artifact_reader=reader,
    )

    changed_files = dict(files)
    changed_files["modeling_qwen3.py"] = b"class DifferentModel: pass\n"
    changed_siblings = [
        _sibling(
            sibling.rfilename,
            changed_files[sibling.rfilename],
            lfs=sibling.lfs is not None,
        )
        for sibling in siblings
    ]

    def changed_reader(repo_id, path, revision, token):
        return changed_files[path]

    changed_code = build_hub_model_bundle(
        "Qwen/Qwen3-1.7B",
        quantization="bf16",
        dtype="bfloat16",
        api=FakeApi(changed_siblings),
        artifact_reader=changed_reader,
    )
    changed_dtype = build_hub_model_bundle(
        "Qwen/Qwen3-1.7B",
        quantization="bf16",
        dtype="float32",
        api=FakeApi(siblings),
        artifact_reader=reader,
    )

    assert changed_code.manifest.architecture_graph_hash != base.manifest.architecture_graph_hash
    assert changed_code.manifest.model_swarm_id != base.manifest.model_swarm_id
    assert changed_dtype.manifest.model_swarm_id != base.manifest.model_swarm_id


def test_builder_rejects_dtype_that_cuda_loaders_would_ignore():
    files, siblings, _, reader = _fixture()
    config = json.loads(files["config.json"])
    config["torch_dtype"] = "bfloat16"
    files["config.json"] = json.dumps(config, sort_keys=True).encode()
    siblings = [
        _sibling(sibling.rfilename, files[sibling.rfilename], lfs=sibling.lfs is not None)
        for sibling in siblings
    ]

    with pytest.raises(ValueError, match="does not match"):
        build_hub_model_bundle(
            "Qwen/Qwen3-1.7B",
            quantization="bf16",
            dtype="float16",
            api=FakeApi(siblings),
            artifact_reader=reader,
        )


def test_xet_stream_hashes_each_tensor_across_transport_chunk_boundaries(monkeypatch):
    payload = b"abcdefghijkl"
    ranges = [
        ("tensor.a", 0, 3, object()),
        ("tensor.empty", 3, 3, object()),
        ("tensor.b", 3, 8, object()),
        ("tensor.c", 8, 12, object()),
    ]

    class Group:
        def download_stream(self, file_info, *, start, end):
            assert (start, end) == (8, 20)
            yield payload[:2]
            yield payload[2:7]
            yield payload[7:]

    class Session:
        def new_download_stream_group(self, **kwargs):
            assert kwargs["token_refresh_url"] == "https://xet.example/refresh"
            return Group()

    monkeypatch.setattr("swarm_protocol.model_manifest.get_xet_session", lambda: Session())
    source = ArtifactDescriptor(
        path="model.safetensors",
        size=20,
        sha256="10" * 32,
        media_type="application/vnd.safetensors",
        role=ArtifactRole.WEIGHT,
    )

    hashes = _hash_xet_tensor_ranges(
        source=source,
        xet_file_hash="20" * 32,
        refresh_route="https://xet.example/refresh",
        headers={},
        data_section_offset=8,
        relative_ranges=ranges,
    )

    assert hashes == {
        "tensor.a": hashlib.sha256(b"abc").hexdigest(),
        "tensor.empty": hashlib.sha256(b"").hexdigest(),
        "tensor.b": hashlib.sha256(b"defgh").hexdigest(),
        "tensor.c": hashlib.sha256(b"ijkl").hexdigest(),
    }


def test_safetensors_metadata_retries_only_transient_hub_network_errors(monkeypatch):
    calls = 0
    sleeps = []

    class Api:
        def get_safetensors_metadata(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise httpx.ConnectError("temporary TLS disconnect")
            return "metadata"

    monkeypatch.setattr("swarm_protocol.model_manifest.time.sleep", sleeps.append)
    result = _get_safetensors_metadata_with_backoff(
        client=Api(),
        repo_id="test/model",
        immutable_revision=REVISION,
        token=None,
    )

    assert result == "metadata"
    assert calls == 3
    assert sleeps == [1, 2]


def test_builder_signs_absolute_tensor_ranges_and_xet_identity(monkeypatch):
    files, siblings, _, reader = _fixture()
    source = next(item for item in siblings if item.rfilename == "model.safetensors")
    source.size = 128
    source.lfs.size = 128
    source.lfs.sha256 = "30" * 32

    class SelectiveApi(FakeApi):
        def get_safetensors_metadata(self, repo_id, **kwargs):
            return SimpleNamespace(
                files_metadata={
                    "model.safetensors": SimpleNamespace(
                        tensors={
                            "model.layers.0.weight": SimpleNamespace(
                                data_offsets=(0, 16), dtype="F32", shape=(2, 2)
                            ),
                            "model.layers.1.weight": SimpleNamespace(
                                data_offsets=(16, 32), dtype="F32", shape=(2, 2)
                            ),
                        }
                    )
                }
            )

    monkeypatch.setattr(
        "swarm_protocol.model_manifest.get_hf_file_metadata",
        lambda *args, **kwargs: SimpleNamespace(
            size=128,
            etag='"' + "30" * 32 + '"',
            xet_file_data=SimpleNamespace(
                file_hash="40" * 32,
                refresh_route="https://xet.example/refresh",
            ),
        ),
    )
    monkeypatch.setattr(
        "swarm_protocol.model_manifest._hash_xet_tensor_ranges",
        lambda **kwargs: {
            "model.layers.0.weight": "50" * 32,
            "model.layers.1.weight": "60" * 32,
        },
    )

    bundle = build_hub_model_bundle(
        "Qwen/Qwen3-1.7B",
        revision=REVISION,
        quantization="bf16",
        dtype="bfloat16",
        api=SelectiveApi(siblings),
        artifact_reader=reader,
        include_selective_weight_index=True,
    )

    signed_source = next(
        artifact
        for artifact in bundle.artifact_index.artifacts
        if artifact.path == "model.safetensors"
    )
    assert signed_source.xet_file_hash == "40" * 32
    assert [
        (tensor.name, tensor.offset, tensor.length, tensor.sha256)
        for tensor in bundle.artifact_index.tensors
    ] == [
        ("model.layers.0.weight", 96, 16, "50" * 32),
        ("model.layers.1.weight", 112, 16, "60" * 32),
    ]
