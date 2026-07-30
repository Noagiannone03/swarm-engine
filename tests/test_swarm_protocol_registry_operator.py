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
    RouteAuthorityKey,
    RouteAuthorityKeyset,
    TrustedModelRegistry,
    artifact_collection_hash,
)
from swarm_protocol.registry_operator import (
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
