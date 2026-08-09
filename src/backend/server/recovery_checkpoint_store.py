"""Ephemeral encrypted storage for exact recovery checkpoints.

The Request Agent keeps the replay journal as the source of truth.  This store
only retains optional native KV pages that can shorten recovery.  Files are
stream-encrypted, atomically published, bounded by a disk reserve, and removed
when the process exits because the encryption key is intentionally never
persisted.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import struct
import threading

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_MAGIC = b"FABIKV01"
_NONCE_BYTES = 12
_TAG_BYTES = 16
_HEADER = struct.Struct(">8s12s32sQ")
_MAX_CHECKPOINT_BYTES = 16 * 1024**3
_DEFAULT_CHUNK_BYTES = 4 * 1024**2
_MIN_FREE_FLOOR_BYTES = 1024**3
_MIN_FREE_CAP_BYTES = 10 * 1024**3
_MAX_STORE_FLOOR_BYTES = 512 * 1024**2
_MAX_STORE_CAP_BYTES = 64 * 1024**3


class CheckpointStoreUnavailable(RuntimeError):
    """A warm checkpoint cannot be retained without violating disk safety."""


@dataclass(frozen=True)
class StoredRecoveryCheckpoint:
    checkpoint_id: str
    path: Path
    plaintext_bytes: int
    aad_sha256: str


def canonical_checkpoint_aad(payload: Mapping[str, object]) -> bytes:
    """Encode checkpoint identity for authenticated, deterministic storage."""

    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint authenticated data must be a mapping")
    try:
        encoded = json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint authenticated data is not canonical JSON") from error
    if not encoded or len(encoded) > 1024 * 1024:
        raise ValueError("checkpoint authenticated data exceeds its size bound")
    return encoded


class EncryptedCheckpointWriter:
    """One sequential AES-256-GCM transaction."""

    def __init__(
        self,
        store: EncryptedRecoveryCheckpointStore,
        *,
        checkpoint_id: str,
        plaintext_bytes: int,
        aad: bytes,
    ) -> None:
        self._store = store
        self.checkpoint_id = checkpoint_id
        self.plaintext_bytes = plaintext_bytes
        self._aad = aad
        self._aad_sha256 = hashlib.sha256(aad).digest()
        self._nonce = secrets.token_bytes(_NONCE_BYTES)
        token = secrets.token_hex(24)
        self._temporary_path = store.root / f".checkpoint-{token}.tmp"
        self._final_path = store.root / f"checkpoint-{token}.bin"
        descriptor = os.open(
            self._temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        self._file = os.fdopen(descriptor, "wb")
        self._file.write(_HEADER.pack(_MAGIC, self._nonce, self._aad_sha256, plaintext_bytes))
        cipher = Cipher(algorithms.AES(store._key), modes.GCM(self._nonce))
        self._encryptor = cipher.encryptor()
        self._encryptor.authenticate_additional_data(aad)
        self._written = 0
        self._finished = False

    def write(self, chunk: bytes) -> None:
        if self._finished:
            raise RuntimeError("checkpoint transaction is already closed")
        if not isinstance(chunk, bytes) or not chunk:
            raise ValueError("checkpoint chunk must be non-empty bytes")
        if self._written + len(chunk) > self.plaintext_bytes:
            raise ValueError("checkpoint payload exceeds its reserved length")
        encrypted = self._encryptor.update(chunk)
        self._file.write(encrypted)
        self._written += len(chunk)

    def finalize(self) -> StoredRecoveryCheckpoint:
        if self._finished:
            raise RuntimeError("checkpoint transaction is already closed")
        if self._written != self.plaintext_bytes:
            self.abort()
            raise ValueError("checkpoint payload ended before its reserved length")
        try:
            tail = self._encryptor.finalize()
            self._file.write(tail)
            self._file.write(self._encryptor.tag)
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            os.replace(self._temporary_path, self._final_path)
            stored = StoredRecoveryCheckpoint(
                checkpoint_id=self.checkpoint_id,
                path=self._final_path,
                plaintext_bytes=self.plaintext_bytes,
                aad_sha256=self._aad_sha256.hex(),
            )
            self._store._commit(self, stored)
            self._finished = True
            return stored
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self._file.close()
        finally:
            for path in (self._temporary_path, self._final_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            self._store._abort(self)


class EncryptedRecoveryCheckpointStore:
    """Process-local encrypted checkpoint cache with exact byte admission."""

    def __init__(
        self,
        root: Path,
        *,
        minimum_free_bytes: int | None = None,
        maximum_store_bytes: int | None = None,
        disk_usage=shutil.disk_usage,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass
        usage = disk_usage(self.root)
        self.minimum_free_bytes = (
            max(_MIN_FREE_FLOOR_BYTES, min(_MIN_FREE_CAP_BYTES, usage.total // 50))
            if minimum_free_bytes is None
            else int(minimum_free_bytes)
        )
        self.maximum_store_bytes = (
            max(_MAX_STORE_FLOOR_BYTES, min(_MAX_STORE_CAP_BYTES, usage.total // 10))
            if maximum_store_bytes is None
            else int(maximum_store_bytes)
        )
        if self.minimum_free_bytes < 0:
            raise ValueError("checkpoint minimum free bytes cannot be negative")
        if self.maximum_store_bytes <= 0:
            raise ValueError("checkpoint store limit must be positive")
        self._disk_usage = disk_usage
        self._key = secrets.token_bytes(32)
        self._lock = threading.RLock()
        self._reserved_bytes = 0
        self._stored: dict[str, StoredRecoveryCheckpoint] = {}
        self._writers: set[EncryptedCheckpointWriter] = set()
        self._closed = False
        self._remove_owned_files()

    def begin(
        self,
        *,
        checkpoint_id: str,
        plaintext_bytes: int,
        aad: bytes,
    ) -> EncryptedCheckpointWriter:
        if not checkpoint_id or len(checkpoint_id) > 512:
            raise ValueError("checkpoint id is invalid")
        if plaintext_bytes <= 0 or plaintext_bytes > _MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint payload is outside its size bound")
        if not isinstance(aad, bytes) or not aad:
            raise ValueError("checkpoint authenticated data must be non-empty bytes")
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint store is closed")
            current_bytes = sum(item.plaintext_bytes for item in self._stored.values())
            growth = plaintext_bytes
            previous = self._stored.get(checkpoint_id)
            if previous is not None:
                growth = max(0, plaintext_bytes - previous.plaintext_bytes)
            if current_bytes + self._reserved_bytes + growth > self.maximum_store_bytes:
                raise CheckpointStoreUnavailable("checkpoint cache quota would be exceeded")
            free_bytes = self._disk_usage(self.root).free
            if free_bytes - self._reserved_bytes - plaintext_bytes < self.minimum_free_bytes:
                raise CheckpointStoreUnavailable("checkpoint would violate the disk reserve")
            self._reserved_bytes += plaintext_bytes
            try:
                writer = EncryptedCheckpointWriter(
                    self,
                    checkpoint_id=checkpoint_id,
                    plaintext_bytes=plaintext_bytes,
                    aad=aad,
                )
            except BaseException:
                self._reserved_bytes -= plaintext_bytes
                raise
            self._writers.add(writer)
            return writer

    def read_chunks(
        self,
        checkpoint: StoredRecoveryCheckpoint,
        *,
        aad: bytes,
        chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    ) -> Iterator[bytes]:
        """Yield plaintext chunks and authenticate the complete file at EOF."""

        if chunk_bytes <= 0 or chunk_bytes > _DEFAULT_CHUNK_BYTES:
            raise ValueError("checkpoint read chunk exceeds its size bound")
        if hashlib.sha256(aad).hexdigest() != checkpoint.aad_sha256:
            raise ValueError("checkpoint authenticated data differs from its identity")
        with checkpoint.path.open("rb") as source:
            header = source.read(_HEADER.size)
            if len(header) != _HEADER.size:
                raise ValueError("checkpoint header is truncated")
            magic, nonce, aad_sha256, plaintext_bytes = _HEADER.unpack(header)
            if (
                magic != _MAGIC
                or aad_sha256.hex() != checkpoint.aad_sha256
                or plaintext_bytes != checkpoint.plaintext_bytes
            ):
                raise ValueError("checkpoint header differs from its committed identity")
            expected_file_bytes = _HEADER.size + plaintext_bytes + _TAG_BYTES
            if checkpoint.path.stat().st_size != expected_file_bytes:
                raise ValueError("checkpoint encrypted length changed")
            source.seek(-_TAG_BYTES, os.SEEK_END)
            tag = source.read(_TAG_BYTES)
            source.seek(_HEADER.size)
            decryptor = Cipher(algorithms.AES(self._key), modes.GCM(nonce, tag)).decryptor()
            decryptor.authenticate_additional_data(aad)
            remaining = plaintext_bytes
            while remaining:
                encrypted = source.read(min(chunk_bytes, remaining))
                if not encrypted:
                    raise ValueError("checkpoint ciphertext is truncated")
                remaining -= len(encrypted)
                plaintext = decryptor.update(encrypted)
                if plaintext:
                    yield plaintext
            tail = decryptor.finalize()
            if tail:
                yield tail

    def remove(self, checkpoint_id: str) -> bool:
        with self._lock:
            checkpoint = self._stored.pop(checkpoint_id, None)
        if checkpoint is None:
            return False
        try:
            checkpoint.path.unlink()
        except FileNotFoundError:
            pass
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            writers = tuple(self._writers)
        for writer in writers:
            writer.abort()
        self._remove_owned_files()
        with self._lock:
            self._stored.clear()
            self._key = b""

    def _commit(
        self,
        writer: EncryptedCheckpointWriter,
        checkpoint: StoredRecoveryCheckpoint,
    ) -> None:
        with self._lock:
            self._writers.discard(writer)
            self._reserved_bytes -= writer.plaintext_bytes
            previous = self._stored.get(checkpoint.checkpoint_id)
            self._stored[checkpoint.checkpoint_id] = checkpoint
        if previous is not None and previous.path != checkpoint.path:
            try:
                previous.path.unlink()
            except FileNotFoundError:
                pass

    def _abort(self, writer: EncryptedCheckpointWriter) -> None:
        with self._lock:
            if writer in self._writers:
                self._writers.remove(writer)
                self._reserved_bytes -= writer.plaintext_bytes

    def _remove_owned_files(self) -> None:
        for pattern in ("checkpoint-*.bin", ".checkpoint-*.tmp"):
            for path in self.root.glob(pattern):
                if path.is_file() and not path.is_symlink():
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
