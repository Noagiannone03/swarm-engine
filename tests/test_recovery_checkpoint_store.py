from types import SimpleNamespace

import pytest

from backend.server.recovery_checkpoint_store import (
    CheckpointStoreUnavailable,
    EncryptedRecoveryCheckpointStore,
    canonical_checkpoint_aad,
)


def usage(*, total=100_000, free=80_000):
    return SimpleNamespace(total=total, used=total - free, free=free)


def test_checkpoint_store_streams_encrypted_atomic_roundtrip(tmp_path):
    store = EncryptedRecoveryCheckpointStore(
        tmp_path,
        minimum_free_bytes=100,
        maximum_store_bytes=10_000,
        disk_usage=lambda _path: usage(),
    )
    aad = canonical_checkpoint_aad({"request_id": "request-1", "epoch": 7, "checkpoint_index": 3})
    writer = store.begin(checkpoint_id="request-1/stage-0", plaintext_bytes=9, aad=aad)
    writer.write(b"native")
    writer.write(b"-kv")
    checkpoint = writer.finalize()

    assert checkpoint.path.read_bytes() != b"native-kv"
    assert list(tmp_path.glob(".checkpoint-*.tmp")) == []
    assert b"".join(store.read_chunks(checkpoint, aad=aad, chunk_bytes=3)) == b"native-kv"

    store.close()
    assert not checkpoint.path.exists()


def test_checkpoint_store_rejects_disk_pressure_before_creating_a_file(tmp_path):
    store = EncryptedRecoveryCheckpointStore(
        tmp_path,
        minimum_free_bytes=100,
        maximum_store_bytes=10_000,
        disk_usage=lambda _path: usage(free=105),
    )

    with pytest.raises(CheckpointStoreUnavailable, match="disk reserve"):
        store.begin(checkpoint_id="request-1/stage-0", plaintext_bytes=10, aad=b"identity")

    assert list(tmp_path.iterdir()) == []


def test_checkpoint_store_detects_tampering_and_never_reuses_stale_files(tmp_path):
    stale = tmp_path / "checkpoint-stale.bin"
    stale.write_bytes(b"stale")
    store = EncryptedRecoveryCheckpointStore(
        tmp_path,
        minimum_free_bytes=0,
        maximum_store_bytes=10_000,
        disk_usage=lambda _path: usage(),
    )
    assert not stale.exists()

    writer = store.begin(checkpoint_id="request-1/stage-0", plaintext_bytes=4, aad=b"identity")
    writer.write(b"page")
    checkpoint = writer.finalize()
    content = bytearray(checkpoint.path.read_bytes())
    content[-17] ^= 1
    checkpoint.path.write_bytes(content)

    with pytest.raises(Exception):
        b"".join(store.read_chunks(checkpoint, aad=b"identity", chunk_bytes=2))


def test_checkpoint_store_aborts_incomplete_transaction_and_releases_capacity(tmp_path):
    store = EncryptedRecoveryCheckpointStore(
        tmp_path,
        minimum_free_bytes=0,
        maximum_store_bytes=4,
        disk_usage=lambda _path: usage(),
    )
    writer = store.begin(checkpoint_id="request-1/stage-0", plaintext_bytes=4, aad=b"identity")
    writer.write(b"no")
    with pytest.raises(ValueError, match="ended before"):
        writer.finalize()

    replacement = store.begin(
        checkpoint_id="request-1/stage-0",
        plaintext_bytes=4,
        aad=b"identity",
    )
    replacement.write(b"page")
    replacement.finalize()
