import functools
import hashlib
import json
import shutil
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
    TufTimestampRefresher,
    artifact_collection_hash,
    inspect_repository_metadata,
    synchronize_repository_timestamp,
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
        model_max_context_tokens=65_536,
        context_classes=(4_096, 8_192, 16_384, 32_768, 65_536),
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
        projected_root = client.metadata_dir / "root.json"
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


def test_tuf_registry_isolates_independent_bootstrap_authorities(tmp_path):
    first_repository = tmp_path / "first-repository"
    second_repository = tmp_path / "second-repository"
    bundle = _bundle()
    first_root = TufRegistryPublisher(first_repository, _signers()).initialize((bundle,))
    second_root = TufRegistryPublisher(second_repository, _signers()).initialize((bundle,))
    assert first_root != second_root

    state_dir = tmp_path / "shared-client-state"
    with _serve(first_repository) as first_url, _serve(second_repository) as second_url:
        first = _client(state_dir, first_url, first_root)
        second = _client(state_dir, second_url, second_root)
        assert first.fetch(bundle.model_swarm_id) == bundle
        assert second.fetch(bundle.model_swarm_id) == bundle

    assert first.cache_dir != second.cache_dir
    assert first.metadata_dir.joinpath("timestamp.json").is_file()
    assert second.metadata_dir.joinpath("timestamp.json").is_file()
    assert first.metadata_dir.joinpath("root.json").read_bytes() == first_root
    assert second.metadata_dir.joinpath("root.json").read_bytes() == second_root


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


def test_full_publish_advances_timestamp_after_independent_refreshes(tmp_path):
    repository = tmp_path / "repository"
    signers = _signers()
    publisher = TufRegistryPublisher(repository, signers)
    bundle = _bundle()
    now = datetime.now(timezone.utc)
    root_bytes = publisher.initialize((bundle,), now=now)
    refresher = TufTimestampRefresher(repository, signers.timestamp)

    assert refresher.refresh(now=now + timedelta(minutes=1)) == 2
    assert refresher.refresh(now=now + timedelta(minutes=2)) == 3

    with _serve(repository) as base_url:
        client = _client(tmp_path / "client", base_url, root_bytes)
        assert client.fetch(bundle.model_swarm_id) == bundle

        assert publisher.publish((bundle,), now=now + timedelta(minutes=3)) == 2
        assert client.fetch(bundle.model_swarm_id) == bundle

    timestamp = __import__("json").loads((repository / "metadata" / "timestamp.json").read_bytes())
    assert timestamp["signed"]["version"] == 4
    assert timestamp["signed"]["meta"]["snapshot.json"]["version"] == 2


def test_metadata_inspection_authenticates_roles_and_warns_before_offline_expiry(tmp_path):
    repository = tmp_path / "repository"
    signers = _signers()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    root_bytes = TufRegistryPublisher(repository, signers).initialize((_bundle(),), now=now)

    healthy = inspect_repository_metadata(repository, root_bytes, now=now + timedelta(hours=1))
    assert healthy.status == "healthy"
    assert healthy.requires_attention is False
    assert [role.role for role in healthy.roles] == ["root", "targets", "snapshot", "timestamp"]

    TufTimestampRefresher(repository, signers.timestamp).refresh(
        now=now + timedelta(days=5, hours=12)
    )
    warning = inspect_repository_metadata(
        repository,
        root_bytes,
        now=now + timedelta(days=5, hours=13),
    )
    assert warning.status == "offline_renewal_required"
    assert warning.requires_attention is True
    assert next(role for role in warning.roles if role.role == "snapshot").renewal_required
    assert not next(role for role in warning.roles if role.role == "timestamp").renewal_required


def test_metadata_inspection_rejects_snapshot_hash_substitution(tmp_path):
    repository = tmp_path / "repository"
    signers = _signers()
    root_bytes = TufRegistryPublisher(repository, signers).initialize((_bundle(),))
    snapshot_path = repository / "metadata" / "1.snapshot.json"
    snapshot_path.write_bytes(snapshot_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="exact referenced snapshot"):
        inspect_repository_metadata(repository, root_bytes)


def test_metadata_inspection_follows_rotation_from_pinned_bootstrap_root(tmp_path):
    repository = tmp_path / "repository"
    old_signers = _signers(root_threshold=2)
    publisher = TufRegistryPublisher(repository, old_signers)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    bootstrap_root = publisher.initialize((_bundle(),), now=now)
    new_signers = _signers(root_threshold=2)
    publisher.rotate_root(new_signers, now=now + timedelta(minutes=1))
    TufRegistryPublisher(repository, new_signers).publish(
        (_bundle(),),
        now=now + timedelta(minutes=2),
    )

    status = inspect_repository_metadata(
        repository,
        bootstrap_root,
        now=now + timedelta(minutes=3),
    )

    assert status.status == "healthy"
    assert next(role for role in status.roles if role.role == "root").version == 2
    assert next(role for role in status.roles if role.role == "snapshot").version == 2


def test_offline_publish_synchronizes_the_authenticated_online_timestamp(tmp_path):
    operator_repository = tmp_path / "operator"
    online_repository = tmp_path / "online"
    signers = _signers()
    bundle = _bundle()
    now = datetime.now(timezone.utc)
    publisher = TufRegistryPublisher(operator_repository, signers)
    publisher.initialize((bundle,), now=now)
    shutil.copytree(operator_repository, online_repository)

    assert TufTimestampRefresher(online_repository, signers.timestamp).refresh(
        now=now + timedelta(minutes=1)
    ) == 2
    version, changed = synchronize_repository_timestamp(
        operator_repository,
        (online_repository / "metadata" / "timestamp.json").read_bytes(),
        now=now + timedelta(minutes=2),
    )
    assert (version, changed) == (2, True)

    # The full offline publish now advances beyond the real online version.
    assert publisher.publish((bundle,), now=now + timedelta(minutes=3)) == 2
    timestamp = json.loads((operator_repository / "metadata" / "timestamp.json").read_bytes())
    assert timestamp["signed"]["version"] == 3
    assert timestamp["signed"]["meta"]["snapshot.json"]["version"] == 2


def test_timestamp_synchronization_rejects_same_version_equivocation(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    signers = _signers()
    now = datetime.now(timezone.utc)
    TufRegistryPublisher(first, signers).initialize((_bundle(),), now=now)
    shutil.copytree(first, second)
    assert TufTimestampRefresher(first, signers.timestamp).refresh(
        now=now + timedelta(minutes=1)
    ) == 2
    assert TufTimestampRefresher(second, signers.timestamp).refresh(
        now=now + timedelta(minutes=2)
    ) == 2

    with pytest.raises(ValueError, match="equivocation"):
        synchronize_repository_timestamp(
            first,
            (second / "metadata" / "timestamp.json").read_bytes(),
            now=now + timedelta(minutes=3),
        )


def test_timestamp_synchronization_never_imports_an_unknown_snapshot(tmp_path):
    operator_repository = tmp_path / "operator"
    online_repository = tmp_path / "online"
    signers = _signers()
    now = datetime.now(timezone.utc)
    TufRegistryPublisher(operator_repository, signers).initialize((_bundle(),), now=now)
    shutil.copytree(operator_repository, online_repository)
    TufRegistryPublisher(online_repository, signers).publish(
        (_bundle(tokenizer_bytes=b"new revision"),),
        now=now + timedelta(minutes=1),
    )

    with pytest.raises(ValueError, match="unavailable snapshot version 2"):
        synchronize_repository_timestamp(
            operator_repository,
            (online_repository / "metadata" / "timestamp.json").read_bytes(),
            now=now + timedelta(minutes=2),
        )


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
