import functools
import hashlib
import os
import stat
import threading
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from swarm_protocol import (
    ArtifactDescriptor,
    ArtifactRole,
    ModelArtifactIndex,
    ModelManifest,
    ModelRegistryBundle,
    TrustedModelRegistry,
    artifact_collection_hash,
)
from swarm_protocol.registry_operator import (
    generate_passphrase_file,
    generate_staging_keys,
    initialize_staging_registry,
    load_staging_keys,
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


def test_staging_registry_initializes_publishes_and_verifies(tmp_path):
    bundle = _bundle()
    bundle_path = tmp_path / "bundle.json"
    _write_bundle(bundle_path, bundle)
    repository = tmp_path / "repository"
    key_dir = tmp_path / "keys"
    bootstrap = tmp_path / "bootstrap-root.json"

    assert initialize_staging_registry(
        repository,
        key_dir,
        (bundle_path,),
        PASSPHRASE,
        bootstrap_root_output=bootstrap,
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
