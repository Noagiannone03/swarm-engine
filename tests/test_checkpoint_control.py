import hashlib
from types import SimpleNamespace

import pytest

from parallax.server.executor.checkpoint_control import (
    CheckpointControlError,
    CheckpointMemoryUnavailable,
    SkippyCheckpointExportRegistry,
)


class _Page:
    version = 1
    layer_start = 4
    layer_end = 8
    token_start = 0
    layer_count = 4
    k_type = 1
    v_type = 1
    k_row_bytes = 8
    v_row_bytes = 8
    v_element_bytes = 2
    flags = 0

    def __init__(self, payload: bytes, token_count: int):
        self.payload = payload
        self.token_count = token_count
        self.payload_bytes = len(payload)
        self.payload_sha256 = hashlib.sha256(payload).hexdigest()

    def read_chunk(self, offset: int, length: int) -> bytes:
        return self.payload[offset : offset + length]


class _Runner:
    def __init__(self, bytes_per_token: int):
        self.bytes_per_token = bytes_per_token
        self.calls = []
        self.imports = []

    def prepare_kv_page_export(self, request_id, *, token_start, token_count):
        self.calls.append((request_id, token_start, token_count))
        return _Page(b"x" * (self.bytes_per_token * token_count), token_count)

    def begin_kv_page_import(self, descriptor):
        return SimpleNamespace(descriptor=descriptor, payload=bytearray())

    @staticmethod
    def append_kv_page_import(builder, chunk):
        builder.payload.extend(chunk)
        return len(builder.payload)

    def commit_kv_page_import(self, request_id, builder):
        assert hashlib.sha256(builder.payload).hexdigest() == builder.descriptor["payload_sha256"]
        self.imports.append((request_id, bytes(builder.payload)))


def _registry(*, available=10_000, chunk_bytes=7, bytes_per_token=3):
    runner = _Runner(bytes_per_token)
    registry = SkippyCheckpointExportRegistry(
        runner=runner,
        kv_bytes_per_token=bytes_per_token,
        layer_start=4,
        layer_end=8,
        memory_snapshot=lambda: SimpleNamespace(total=20_000, available=available),
        reserve_bytes=lambda total, free: 1_000,
        chunk_bytes=chunk_bytes,
    )
    return registry, runner


def test_checkpoint_export_is_sequential_and_chunk_bounded():
    registry, runner = _registry()
    descriptor = registry.prepare(request_id="request-1", token_count=8)

    assert runner.calls == [("request-1", 0, 8)]
    assert descriptor.payload_bytes == 24
    payload = bytearray()
    offset = 0
    done = False
    while not done:
        chunk, offset, done = registry.read(
            handle=descriptor.handle,
            request_id="request-1",
            offset=offset,
        )
        assert 0 < len(chunk) <= 7
        payload.extend(chunk)
    assert bytes(payload) == b"x" * 24
    assert hashlib.sha256(payload).hexdigest() == descriptor.payload_sha256


def test_checkpoint_export_rejects_random_access_and_cross_request_handles():
    registry, _ = _registry()
    descriptor = registry.prepare(request_id="request-1", token_count=8)

    with pytest.raises(CheckpointControlError, match="sequentially"):
        registry.read(handle=descriptor.handle, request_id="request-1", offset=1)
    with pytest.raises(CheckpointControlError, match="another request"):
        registry.read(handle=descriptor.handle, request_id="request-2", offset=0)


def test_checkpoint_export_fails_closed_before_native_allocation_under_pressure():
    registry, runner = _registry(available=1_020, chunk_bytes=7)

    with pytest.raises(CheckpointMemoryUnavailable, match="reserve"):
        registry.prepare(request_id="request-1", token_count=8)
    assert runner.calls == []


def test_new_checkpoint_for_same_request_releases_old_handle():
    registry, _ = _registry()
    first = registry.prepare(request_id="request-1", token_count=8)
    second = registry.prepare(request_id="request-1", token_count=9)

    assert first.handle != second.handle
    with pytest.raises(CheckpointControlError, match="unknown"):
        registry.read(handle=first.handle, request_id="request-1", offset=0)
    assert registry.drop(handle=second.handle, request_id="request-1") is True
    assert registry.drop(handle=second.handle, request_id="request-1") is False


def test_checkpoint_import_is_sequential_digest_bound_and_committed_once():
    registry, runner = _registry()
    payload = b"x" * 24
    descriptor = {
        "version": 1,
        "layer_start": 4,
        "layer_end": 8,
        "token_start": 0,
        "token_count": 8,
        "layer_count": 4,
        "k_type": 1,
        "v_type": 1,
        "k_row_bytes": 8,
        "v_row_bytes": 8,
        "v_element_bytes": 2,
        "flags": 0,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    handle = registry.begin_import(request_id="replacement", descriptor=descriptor)

    offset, done = registry.append_import(
        handle=handle,
        request_id="replacement",
        offset=0,
        chunk=payload[:7],
    )
    assert (offset, done) == (7, False)
    with pytest.raises(CheckpointControlError, match="sequential"):
        registry.append_import(
            handle=handle,
            request_id="replacement",
            offset=0,
            chunk=payload[7:14],
        )
    while not done:
        chunk = payload[offset : offset + 7]
        offset, done = registry.append_import(
            handle=handle,
            request_id="replacement",
            offset=offset,
            chunk=chunk,
        )
    registry.commit_import(handle=handle, request_id="replacement")

    assert runner.imports == [("replacement", payload)]
    with pytest.raises(CheckpointControlError, match="unknown"):
        registry.commit_import(handle=handle, request_id="replacement")


def test_checkpoint_import_rejects_wrong_span_and_signed_size():
    registry, runner = _registry()
    descriptor = {
        "version": 1,
        "layer_start": 0,
        "layer_end": 4,
        "token_start": 0,
        "token_count": 8,
        "layer_count": 4,
        "k_type": 1,
        "v_type": 1,
        "k_row_bytes": 8,
        "v_row_bytes": 8,
        "v_element_bytes": 2,
        "flags": 0,
        "payload_bytes": 24,
        "payload_sha256": "a" * 64,
    }
    with pytest.raises(CheckpointControlError, match="different layer span"):
        registry.begin_import(request_id="replacement", descriptor=descriptor)
    descriptor["layer_start"] = 4
    descriptor["layer_end"] = 8
    descriptor["payload_bytes"] = 25
    with pytest.raises(CheckpointControlError, match="signed execution geometry"):
        registry.begin_import(request_id="replacement", descriptor=descriptor)
    assert runner.imports == []
