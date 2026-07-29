import functools
import hashlib
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
from securesystemslib.signer import CryptoSigner
from tuf.api.exceptions import (
    BadVersionNumberError,
    DownloadError,
    ExpiredMetadataError,
)

from swarm_protocol import (
    ArtifactDescriptor,
    ArtifactRole,
    ModelArtifactIndex,
    ModelManifest,
    ModelRegistryBundle,
    RegistryExpiryPolicy,
    RegistryRoleSigners,
    RouteAuthorityKey,
    RouteAuthorityKeyset,
    TrustedModelRegistry,
    TufRegistryPublisher,
    artifact_collection_hash,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _bundle(*, tokenizer_bytes: bytes = b"tokenizer") -> ModelRegistryBundle:
    files = (
        ("config.json", b'{"num_hidden_layers":4}', ArtifactRole.ARCHITECTURE),
        ("model.safetensors", b"weights", ArtifactRole.WEIGHT),
        ("tokenizer.json", tokenizer_bytes, ArtifactRole.TOKENIZER),
    )
    index = ModelArtifactIndex(
        model_id="test/model",
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
    manifest = ModelManifest(
        model_id=index.model_id,
        immutable_revision=index.immutable_revision,
        architecture_graph_hash=artifact_collection_hash(index, ArtifactRole.ARCHITECTURE),
        tokenizer_hash=artifact_collection_hash(index, ArtifactRole.TOKENIZER),
        weight_collection_hash=artifact_collection_hash(index, ArtifactRole.WEIGHT),
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
    return ModelRegistryBundle(manifest=manifest, artifact_index=index)


def _route_authorities() -> RouteAuthorityKeyset:
    public_key = "aa" * 32
    return RouteAuthorityKeyset(
        generation=3,
        issued_at_ms=1_000,
        expires_at_ms=100_000,
        keys=(
            RouteAuthorityKey(
                key_id=hashlib.sha256(bytes.fromhex(public_key)).hexdigest(),
                public_key=public_key,
                not_before_ms=500,
                not_after_ms=90_000,
            ),
        ),
        revoked_identifiers=("bb" * 64,),
    )


def _signers(*, root_threshold: int = 1) -> RegistryRoleSigners:
    root = tuple(CryptoSigner.generate_ed25519() for _ in range(root_threshold))
    return RegistryRoleSigners(
        root=root,
        targets=(CryptoSigner.generate_ed25519(),),
        snapshot=(CryptoSigner.generate_ed25519(),),
        timestamp=(CryptoSigner.generate_ed25519(),),
        root_threshold=root_threshold,
    )


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


def _client(state_dir, base_url, root_bytes):
    return TrustedModelRegistry(
        state_dir,
        metadata_base_url=f"{base_url}/metadata/",
        target_base_url=f"{base_url}/targets/",
        bootstrap_root=root_bytes,
    )


def test_tuf_registry_authenticates_bundle_and_survives_root_rotation(tmp_path):
    repository = tmp_path / "repository"
    old_signers = _signers(root_threshold=2)
    publisher = TufRegistryPublisher(repository, old_signers)
    bundle = _bundle()
    root_bytes = publisher.initialize((bundle,), now=datetime.now(timezone.utc))

    with _serve(repository) as base_url:
        client = _client(tmp_path / "client", base_url, root_bytes)
        assert client.fetch(bundle.model_swarm_id) == bundle
        projected_root = tmp_path / "client" / "metadata" / "root.json"
        assert not projected_root.is_symlink()
        assert projected_root.read_bytes() == root_bytes
        assert client.resolve("test/model", immutable_revision=REVISION) == bundle
        assert client.catalog().models[0].model_swarm_id == bundle.model_swarm_id

        new_signers = _signers(root_threshold=2)
        publisher.rotate_root(new_signers, now=datetime.now(timezone.utc))
        new_publisher = TufRegistryPublisher(repository, new_signers)
        assert new_publisher.publish((bundle,), now=datetime.now(timezone.utc)) == 2

        # The original embedded root follows the dual-signed chain and accepts the new roles.
        assert client.fetch(bundle.model_swarm_id) == bundle
        assert not projected_root.is_symlink()
        assert projected_root.read_bytes() == (repository / "metadata" / "2.root.json").read_bytes()


def test_tuf_registry_authenticates_route_authority_rotation_and_revocations(tmp_path):
    repository = tmp_path / "repository"
    publisher = TufRegistryPublisher(repository, _signers())
    bundle = _bundle()
    keyset = _route_authorities()
    root_bytes = publisher.initialize((bundle,), route_authorities=keyset)

    with _serve(repository) as base_url:
        client = _client(tmp_path / "client", base_url, root_bytes)
        trusted = client.route_authorities()
        assert trusted == keyset
        assert trusted.active_public_keys(50_000) == {
            keyset.keys[0].key_id: keyset.keys[0].public_key
        }
        assert trusted.revoked_identifiers == ("bb" * 64,)

        rotated_key = "cc" * 32
        rotated = RouteAuthorityKeyset(
            generation=4,
            issued_at_ms=50_000,
            expires_at_ms=150_000,
            keys=(
                RouteAuthorityKey(
                    key_id=hashlib.sha256(bytes.fromhex(rotated_key)).hexdigest(),
                    public_key=rotated_key,
                    not_before_ms=50_000,
                    not_after_ms=140_000,
                ),
            ),
            revoked_identifiers=("bb" * 64, "dd" * 64),
        )
        assert publisher.publish((bundle,), route_authorities=rotated) == 2
        assert client.route_authorities() == rotated


def test_route_authority_keyset_rejects_substituted_key_ids_and_stale_windows():
    with pytest.raises(ValueError, match="key ID"):
        RouteAuthorityKey(
            key_id="00" * 32,
            public_key="aa" * 32,
            not_before_ms=1,
            not_after_ms=2,
        )

    keyset = _route_authorities()
    with pytest.raises(ValueError, match="not currently valid"):
        keyset.active_public_keys(keyset.expires_at_ms)


def test_tuf_registry_rejects_target_tampering(tmp_path):
    repository = tmp_path / "repository"
    publisher = TufRegistryPublisher(repository, _signers())
    bundle = _bundle()
    root_bytes = publisher.initialize((bundle,), now=datetime.now(timezone.utc))
    target_file = next((repository / "targets" / "models").iterdir())
    target_file.write_bytes(target_file.read_bytes() + b"tampered")

    with _serve(repository) as base_url:
        with pytest.raises(DownloadError, match="length|hash|Downloaded"):
            _client(tmp_path / "client", base_url, root_bytes).fetch(bundle.model_swarm_id)


def test_tuf_registry_rejects_expired_timestamp(tmp_path):
    repository = tmp_path / "repository"
    publisher = TufRegistryPublisher(
        repository,
        _signers(),
        expiry=RegistryExpiryPolicy(
            root=timedelta(days=1),
            targets=timedelta(days=1),
            snapshot=timedelta(days=1),
            timestamp=timedelta(seconds=-1),
        ),
    )
    bundle = _bundle()
    root_bytes = publisher.initialize((bundle,), now=datetime.now(timezone.utc))

    with _serve(repository) as base_url:
        with pytest.raises(ExpiredMetadataError):
            _client(tmp_path / "client", base_url, root_bytes).fetch(bundle.model_swarm_id)


def test_tuf_registry_rejects_timestamp_rollback(tmp_path):
    repository = tmp_path / "repository"
    publisher = TufRegistryPublisher(repository, _signers())
    bundle = _bundle()
    now = datetime.now(timezone.utc)
    root_bytes = publisher.initialize((bundle,), now=now)
    old_timestamp = (repository / "metadata" / "timestamp.json").read_bytes()

    with _serve(repository) as base_url:
        client = _client(tmp_path / "client", base_url, root_bytes)
        assert client.fetch(bundle.model_swarm_id) == bundle
        publisher.publish((bundle,), now=now + timedelta(minutes=1))
        assert client.fetch(bundle.model_swarm_id) == bundle

        (repository / "metadata" / "timestamp.json").write_bytes(old_timestamp)
        with pytest.raises(BadVersionNumberError):
            client.fetch(bundle.model_swarm_id)


def test_registry_bundle_rejects_manifest_index_substitution():
    bundle = _bundle()
    replacement = _bundle(tokenizer_bytes=b"different tokenizer")

    with pytest.raises(ValueError, match="tokenizer collection hash"):
        ModelRegistryBundle(
            manifest=bundle.manifest,
            artifact_index=replacement.artifact_index,
        )


def test_registry_refuses_unsatisfied_signature_threshold():
    signers = _signers()
    invalid = RegistryRoleSigners(
        root=signers.root,
        targets=signers.targets,
        snapshot=signers.snapshot,
        timestamp=signers.timestamp,
        root_threshold=2,
    )

    with pytest.raises(ValueError, match="threshold 2"):
        invalid.validate()
