import functools
import hashlib
import json
import os
import stat
import threading
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from swarm_protocol import (
    ArtifactDescriptor,
    ArtifactRole,
    ExecutionProviderKind,
    ModelArtifactIndex,
    ModelManifest,
    ModelRegistryBundle,
    RouteAuthorityKey,
    RouteAuthorityKeyset,
    TrustedModelRegistry,
    artifact_collection_hash,
)
from swarm_protocol.model_manifest import execution_plan_hash
from swarm_protocol.onnx_stage_builder import portable_build_inventory_hash
from swarm_protocol.registry_operator import (
    _bundle_summary,
    attach_portable_execution,
    attach_portable_execution_file,
    build_skippy_package_bundle_file,
    generate_passphrase_file,
    generate_route_authority,
    generate_staging_keys,
    initialize_staging_registry,
    load_staging_keys,
    load_staging_timestamp_signers,
    publish_staging_registry,
    read_passphrase_file,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"
PASSPHRASE = b"correct horse battery staple"


def _bundle() -> ModelRegistryBundle:
    files = (
        ("config.json", b'{"num_hidden_layers":2}', ArtifactRole.ARCHITECTURE),
        ("model.safetensors", b"weights", ArtifactRole.WEIGHT),
        ("tokenizer.json", b"tokenizer", ArtifactRole.TOKENIZER),
    )
    index = ModelArtifactIndex(
        model_id="test/operator",
        immutable_revision=REVISION,
        artifacts=tuple(
            ArtifactDescriptor(
                path=path,
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                media_type="application/octet-stream",
                role=role,
            )
            for path, content, role in sorted(files)
        ),
    )
    return ModelRegistryBundle(
        manifest=ModelManifest(
            model_id=index.model_id,
            immutable_revision=index.immutable_revision,
            architecture_graph_hash=artifact_collection_hash(index, ArtifactRole.ARCHITECTURE),
            tokenizer_hash=artifact_collection_hash(index, ArtifactRole.TOKENIZER),
            weight_collection_hash=artifact_collection_hash(index, ArtifactRole.WEIGHT),
            weight_format="safetensors",
            quantization="unquantized",
            dtype="bfloat16",
            num_layers=2,
            model_max_context_tokens=65_536,
            context_classes=(4_096, 8_192, 16_384, 32_768, 65_536),
            activation_bytes_per_token=128,
            kv_bytes_per_token_by_layer=(64, 64),
            rope_context_contract_hash="1" * 64,
            attention_kv_contract_hash="2" * 64,
            prefill_contract_hash="3" * 64,
            wire_protocol_version=1,
        ),
        artifact_index=index,
    )


def _write_bundle(path, bundle):
    path.write_bytes(bundle.canonical_bytes())


def test_build_skippy_package_bundle_skips_source_tensor_scan(tmp_path, monkeypatch):
    base = _bundle()
    calls = {}

    def fake_build(model_id, **kwargs):
        calls["build"] = (model_id, kwargs)
        return SimpleNamespace(manifest=base.manifest, artifact_index=base.artifact_index)

    def fake_attach(bundle, **kwargs):
        calls["attach"] = (bundle, kwargs)
        return bundle

    monkeypatch.setattr(
        "swarm_protocol.registry_operator.build_hub_model_bundle",
        fake_build,
    )
    monkeypatch.setattr(
        "swarm_protocol.registry_operator.attach_skippy_package",
        fake_attach,
    )
    output = tmp_path / "skippy-package.json"
    result = build_skippy_package_bundle_file(
        output,
        model_id="Qwen/Qwen3-30B-A3B",
        revision="a" * 40,
        package_repository_id="meshllm/Qwen3-30B-A3B-Q4_K_M-layers",
        package_revision="b" * 40,
        plan_id="skippy-q4-k-m-v1",
        quantization="Q4_K_M",
        dtype="bfloat16",
        runtime_release="mesh-llm/v0.74.0",
        runtime_abi_version="0.1.32",
        providers=(ExecutionProviderKind.CUDA, ExecutionProviderKind.METAL),
    )

    assert result == base
    assert ModelRegistryBundle.model_validate_json(output.read_bytes()) == base
    assert calls["build"] == (
        "Qwen/Qwen3-30B-A3B",
        {
            "revision": "a" * 40,
            "quantization": "Q4_K_M",
            "dtype": "bfloat16",
            "token": None,
            "include_weight_profile": False,
            "include_selective_weight_index": False,
        },
    )
    assert calls["attach"][0] == base
    assert calls["attach"][1]["package_repository_id"] == ("meshllm/Qwen3-30B-A3B-Q4_K_M-layers")
    assert calls["attach"][1]["package_revision"] == "b" * 40
    assert calls["attach"][1]["expected_quantization"] == "Q4_K_M"


def _portable_inventory(root):
    contents = {
        "execution/decoder-000.data": b"decoder-zero-data",
        "execution/decoder-000.onnx": b"decoder-zero",
        "execution/decoder-001.data": b"decoder-one-data",
        "execution/decoder-001.onnx": b"decoder-one",
        "execution/input.onnx": b"input",
        "execution/output.onnx": b"output",
    }
    for relative_path, payload in contents.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    artifacts = []
    for relative_path, payload in sorted(contents.items()):
        graph = relative_path.endswith(".onnx")
        artifacts.append(
            {
                "path": relative_path,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "media_type": "application/onnx" if graph else "application/octet-stream",
                "role": "execution_graph" if graph else "execution_data",
            }
        )
    stages = [
        {
            "stage_id": "input",
            "kind": "input",
            "start_layer": 0,
            "end_layer": 0,
            "graph_path": "execution/input.onnx",
            "external_data_paths": [],
            "io_contract_hash": "a" * 64,
            "inputs": ["input_ids"],
            "outputs": ["hidden.0"],
        },
        {
            "stage_id": "decoder-000",
            "kind": "decoder",
            "start_layer": 0,
            "end_layer": 1,
            "graph_path": "execution/decoder-000.onnx",
            "external_data_paths": ["execution/decoder-000.data"],
            "io_contract_hash": "b" * 64,
            "inputs": ["hidden.0"],
            "outputs": ["hidden.1"],
        },
        {
            "stage_id": "decoder-001",
            "kind": "decoder",
            "start_layer": 1,
            "end_layer": 2,
            "graph_path": "execution/decoder-001.onnx",
            "external_data_paths": ["execution/decoder-001.data"],
            "io_contract_hash": "c" * 64,
            "inputs": ["hidden.1"],
            "outputs": ["hidden.2"],
        },
        {
            "stage_id": "output",
            "kind": "output",
            "start_layer": 2,
            "end_layer": 2,
            "graph_path": "execution/output.onnx",
            "external_data_paths": [],
            "io_contract_hash": "d" * 64,
            "inputs": ["hidden.2"],
            "outputs": ["logits"],
        },
    ]
    inventory = {
        "format_version": 1,
        "builder": "fabi/swarm-engine/portable-onnx-stage-builder",
        "source_model_id": "test/operator",
        "source_model_revision": REVISION,
        "exporter": "microsoft/onnxruntime-genai",
        "exporter_revision": "e" * 40,
        "target_execution_provider": "dml",
        "execution_geometry": {
            "activation_dtype": "float16",
            "activation_hidden_size": 128,
            "kv_num_heads": 2,
            "kv_head_dim": 64,
        },
        "provider_assignment_policy": {
            "allowed_cpu_fallback_nodes": ["/mask/Gather"],
            "allowed_cpu_only_stages": ["input"],
            "require_profiled_assignment": True,
        },
        "num_layers": 2,
        "shared_initializers": [],
        "source_artifacts": [
            {
                "path": "model.onnx",
                "size": len(b"source-model"),
                "sha256": hashlib.sha256(b"source-model").hexdigest(),
            }
        ],
        "artifacts": artifacts,
        "stages": stages,
    }
    inventory["inventory_hash"] = portable_build_inventory_hash(inventory)
    return inventory


def _attach_portable(bundle, inventory, artifact_root, *, source_root=None):
    if source_root is None:
        source_root = artifact_root.parent / f"{artifact_root.name}-source"
        source_root.mkdir(parents=True, exist_ok=True)
        (source_root / "model.onnx").write_bytes(b"source-model")
    return attach_portable_execution(
        bundle,
        inventory,
        artifact_root=artifact_root,
        source_root=source_root,
        artifact_repository_id="fabi-ai/test-operator-onnx",
        artifact_revision="f" * 40,
        plan_id="onnx-dml-int4-v1",
        precision="int4",
        quantization="rtn-block-32",
    )


def test_bundle_summary_exposes_signed_context_contract():
    summary = _bundle_summary(_bundle())

    assert summary["model_max_context_tokens"] == 65_536
    assert summary["context_classes"] == [4_096, 8_192, 16_384, 32_768, 65_536]


def test_attach_portable_execution_verifies_bytes_and_binds_plan(tmp_path):
    artifact_root = tmp_path / "portable"
    inventory = _portable_inventory(artifact_root)
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "model.onnx").write_bytes(b"source-model")

    bundle = _attach_portable(_bundle(), inventory, artifact_root, source_root=source_root)

    assert bundle.manifest.execution_plan_hash == execution_plan_hash(bundle.artifact_index)
    plan = bundle.artifact_index.execution_plans[0]
    assert plan.providers == (ExecutionProviderKind.DIRECTML,)
    assert plan.execution_granularity_layers == 1
    assert plan.artifact_revision == "f" * 40
    assert _bundle_summary(bundle)["signed_execution_bytes"] == sum(
        artifact["size"] for artifact in inventory["artifacts"]
    )


def test_attach_portable_execution_file_is_atomic_and_never_replaces_inputs(tmp_path):
    artifact_root = tmp_path / "portable"
    inventory = _portable_inventory(artifact_root)
    inventory_path = artifact_root / "portable-build.json"
    inventory_path.write_text(json.dumps(inventory))
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "model.onnx").write_bytes(b"source-model")
    bundle_path = tmp_path / "base.json"
    _write_bundle(bundle_path, _bundle())
    output = tmp_path / "portable-bundle.json"

    result = attach_portable_execution_file(
        output,
        bundle_path=bundle_path,
        inventory_path=inventory_path,
        artifact_root=artifact_root,
        source_root=source_root,
        artifact_repository_id="fabi-ai/test-operator-onnx",
        artifact_revision="f" * 40,
        plan_id="onnx-dml-int4-v1",
        precision="int4",
        quantization="rtn-block-32",
    )

    assert ModelRegistryBundle.model_validate_json(output.read_bytes()) == result
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        attach_portable_execution_file(
            output,
            bundle_path=bundle_path,
            inventory_path=inventory_path,
            artifact_root=artifact_root,
            source_root=source_root,
            artifact_repository_id="fabi-ai/test-operator-onnx",
            artifact_revision="f" * 40,
            plan_id="onnx-dml-int4-v1",
            precision="int4",
            quantization="rtn-block-32",
        )


def test_attach_portable_execution_rejects_tampering_and_wrong_provenance(tmp_path):
    artifact_root = tmp_path / "portable"
    inventory = _portable_inventory(artifact_root)
    (artifact_root / "execution/decoder-000.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
        _attach_portable(_bundle(), inventory, artifact_root)

    clean_root = tmp_path / "clean"
    inventory = _portable_inventory(clean_root)
    inventory["source_model_id"] = "other/model"
    inventory["inventory_hash"] = portable_build_inventory_hash(inventory)
    with pytest.raises(ValueError, match="different models"):
        _attach_portable(_bundle(), inventory, clean_root)

    inventory = _portable_inventory(tmp_path / "hash-mismatch")
    inventory["inventory_hash"] = "0" * 64
    with pytest.raises(ValueError, match="inventory hash"):
        _attach_portable(_bundle(), inventory, tmp_path / "hash-mismatch")


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return None


@contextmanager
def _serve(directory):
    handler = functools.partial(_QuietHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_staging_keys_are_encrypted_private_and_never_overwritten(tmp_path):
    key_dir = tmp_path / "keys"
    signers = generate_staging_keys(key_dir, PASSPHRASE)

    assert signers.root[0].public_key.keyid
    for role in ("root", "targets", "snapshot", "timestamp"):
        path = key_dir / f"{role}.pem"
        payload = path.read_bytes()
        assert b"BEGIN ENCRYPTED PRIVATE KEY" in payload
        assert load_pem_private_key(payload, password=PASSPHRASE)
        if os.name != "nt":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    if os.name != "nt":
        assert stat.S_IMODE(key_dir.stat().st_mode) == 0o700

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_staging_keys(key_dir, PASSPHRASE)


def test_staging_keys_reject_wrong_password_and_public_permissions(tmp_path):
    key_dir = tmp_path / "keys"
    generate_staging_keys(key_dir, PASSPHRASE)

    with pytest.raises(ValueError):
        load_staging_keys(key_dir, b"wrong password")

    if os.name != "nt":
        root = key_dir / "root.pem"
        root.chmod(0o644)
        with pytest.raises(PermissionError, match="group or others"):
            load_staging_keys(key_dir, PASSPHRASE)


def test_timestamp_signer_can_refresh_without_other_private_roles(tmp_path):
    bundle_path = tmp_path / "bundle.json"
    _write_bundle(bundle_path, _bundle())
    repository = tmp_path / "repository"
    key_dir = tmp_path / "keys"
    initialize_staging_registry(
        repository,
        key_dir,
        (bundle_path,),
        PASSPHRASE,
        bootstrap_root_output=tmp_path / "root.json",
    )
    previous_snapshot = (repository / "metadata" / "1.snapshot.json").read_bytes()
    previous_targets = (repository / "metadata" / "1.targets.json").read_bytes()

    from swarm_protocol.registry import TufTimestampRefresher

    version = TufTimestampRefresher(
        repository,
        load_staging_timestamp_signers(key_dir, PASSPHRASE),
    ).refresh()

    assert version == 2
    assert (repository / "metadata" / "1.snapshot.json").read_bytes() == previous_snapshot
    assert (repository / "metadata" / "1.targets.json").read_bytes() == previous_targets
    assert not (repository / "metadata" / "2.snapshot.json").exists()
    timestamp = __import__("json").loads((repository / "metadata" / "timestamp.json").read_bytes())
    assert timestamp["signed"]["version"] == 2
    assert timestamp["signed"]["meta"]["snapshot.json"]["version"] == 1


def test_passphrase_file_must_be_private_and_nonempty(tmp_path):
    secret = tmp_path / "passphrase"
    secret.write_bytes(PASSPHRASE + b"\n")
    secret.chmod(0o600)
    assert read_passphrase_file(secret) == PASSPHRASE

    secret.write_bytes(b"\n")
    with pytest.raises(ValueError, match="must not be empty"):
        read_passphrase_file(secret)

    if os.name != "nt":
        secret.write_bytes(PASSPHRASE)
        secret.chmod(0o640)
        with pytest.raises(PermissionError, match="group or others"):
            read_passphrase_file(secret)


def test_generated_passphrase_is_private_high_entropy_and_never_overwritten(tmp_path):
    secret = tmp_path / "secrets" / "registry-passphrase"
    generate_passphrase_file(secret)

    payload = read_passphrase_file(secret)
    assert len(payload) >= 43
    if os.name != "nt":
        assert stat.S_IMODE(secret.stat().st_mode) == 0o600
        assert stat.S_IMODE(secret.parent.stat().st_mode) == 0o700

    with pytest.raises(FileExistsError):
        generate_passphrase_file(secret)


def test_generated_route_authority_is_private_public_and_never_overwritten(tmp_path):
    private_key = tmp_path / "secrets" / "route-authority.key"
    keyset_path = tmp_path / "public" / "route-authorities.json"

    keyset = generate_route_authority(
        private_key,
        keyset_path,
        generation=7,
        valid_for_days=30,
        clock_skew_seconds=60,
        now_ms=1_800_000_000_000,
    )

    payload = private_key.read_text().strip()
    assert len(payload) == 64
    assert payload == payload.lower()
    assert all(character in "0123456789abcdef" for character in payload)
    assert keyset == RouteAuthorityKeyset.model_validate_json(keyset_path.read_bytes())
    assert keyset.generation == 7
    assert keyset.issued_at_ms == 1_799_999_940_000
    assert keyset.expires_at_ms == 1_802_592_000_000
    assert keyset.keys[0].not_before_ms == keyset.issued_at_ms
    assert keyset.keys[0].not_after_ms == keyset.expires_at_ms
    if os.name != "nt":
        assert stat.S_IMODE(private_key.stat().st_mode) == 0o600
        assert stat.S_IMODE(private_key.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(keyset_path.stat().st_mode) == 0o644

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_route_authority(private_key, tmp_path / "other.json")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_route_authority(tmp_path / "other.key", keyset_path)


def test_route_authority_generation_rejects_invalid_contracts(tmp_path):
    private_key = tmp_path / "authority.key"
    keyset = tmp_path / "authority.json"
    with pytest.raises(ValueError, match="different"):
        generate_route_authority(private_key, private_key)
    with pytest.raises(ValueError, match="generation"):
        generate_route_authority(private_key, keyset, generation=0)
    with pytest.raises(ValueError, match="validity"):
        generate_route_authority(private_key, keyset, valid_for_days=0)
    with pytest.raises(ValueError, match="clock skew"):
        generate_route_authority(private_key, keyset, clock_skew_seconds=3601)


def test_staging_registry_initializes_publishes_and_verifies(tmp_path):
    bundle = _bundle()
    bundle_path = tmp_path / "bundle.json"
    _write_bundle(bundle_path, bundle)
    repository = tmp_path / "repository"
    key_dir = tmp_path / "keys"
    bootstrap = tmp_path / "bootstrap-root.json"
    authority_public_key = "aa" * 32
    authorities = RouteAuthorityKeyset(
        generation=1,
        issued_at_ms=1_000,
        expires_at_ms=100_000,
        keys=(
            RouteAuthorityKey(
                key_id=hashlib.sha256(bytes.fromhex(authority_public_key)).hexdigest(),
                public_key=authority_public_key,
                not_before_ms=1_000,
                not_after_ms=100_000,
            ),
        ),
    )
    authorities_path = tmp_path / "route-authorities.json"
    authorities_path.write_bytes(authorities.canonical_bytes())

    assert initialize_staging_registry(
        repository,
        key_dir,
        (bundle_path,),
        PASSPHRASE,
        bootstrap_root_output=bootstrap,
        route_authority_path=authorities_path,
    ) == (bundle,)
    assert bootstrap.read_bytes() == (repository / "metadata" / "1.root.json").read_bytes()
    if os.name != "nt":
        public_files = tuple((repository / "metadata").iterdir()) + tuple(
            path for path in (repository / "targets").rglob("*") if path.is_file()
        )
        assert public_files
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o644 for path in public_files)

    version, published = publish_staging_registry(
        repository,
        key_dir,
        (bundle_path,),
        PASSPHRASE,
        route_authority_path=authorities_path,
    )
    assert version == 2
    assert published == (bundle,)

    with _serve(repository) as base_url:
        client = TrustedModelRegistry(
            tmp_path / "client",
            metadata_base_url=f"{base_url}/metadata/",
            target_base_url=f"{base_url}/targets/",
            bootstrap_root=bootstrap.read_bytes(),
        )
        assert client.fetch(bundle.model_swarm_id) == bundle
        assert client.route_authorities() == authorities


def test_staging_registry_refuses_nonempty_repository_before_key_generation(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "unrelated").write_text("keep")
    bundle_path = tmp_path / "bundle.json"
    _write_bundle(bundle_path, _bundle())
    key_dir = tmp_path / "keys"

    with pytest.raises(FileExistsError, match="non-empty"):
        initialize_staging_registry(
            repository,
            key_dir,
            (bundle_path,),
            PASSPHRASE,
            bootstrap_root_output=tmp_path / "root.json",
        )
    assert not key_dir.exists()


def test_staging_registry_never_overwrites_bootstrap_root(tmp_path):
    bundle_path = tmp_path / "bundle.json"
    _write_bundle(bundle_path, _bundle())
    bootstrap = tmp_path / "root.json"
    bootstrap.write_bytes(b"pinned-existing-root")
    key_dir = tmp_path / "keys"

    with pytest.raises(FileExistsError, match="bootstrap root"):
        initialize_staging_registry(
            tmp_path / "repository",
            key_dir,
            (bundle_path,),
            PASSPHRASE,
            bootstrap_root_output=bootstrap,
        )
    assert bootstrap.read_bytes() == b"pinned-existing-root"
    assert not key_dir.exists()
