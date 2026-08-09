"""Bounded local control for exact Skippy recovery checkpoints.

The native runtime owns exported pages.  This registry exposes one sequential
reader per request so the P2P process can relay small chunks without making a
second full Python copy.  Checkpointing is an optimisation: memory pressure or
any descriptor mismatch rejects the warm path and leaves cold replay intact.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import psutil

from parallax.server.memory_budget import configured_system_reserve_bytes

CHECKPOINT_CHUNK_BYTES = 4 * 1024 * 1024
MAX_CHECKPOINT_EXPORTS = 8


class CheckpointControlError(RuntimeError):
    """The optional warm-checkpoint path could not serve a control request."""


class CheckpointMemoryUnavailable(CheckpointControlError):
    """Creating another native page would violate the host-memory reserve."""


@dataclass(frozen=True)
class CheckpointExportDescriptor:
    handle: str
    request_id: str
    version: int
    layer_start: int
    layer_end: int
    token_start: int
    token_count: int
    layer_count: int
    k_type: int
    v_type: int
    k_row_bytes: int
    v_row_bytes: int
    v_element_bytes: int
    flags: int
    payload_bytes: int
    payload_sha256: str
    chunk_bytes: int = CHECKPOINT_CHUNK_BYTES

    def as_wire_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass
class _RetainedExport:
    descriptor: CheckpointExportDescriptor
    page: Any
    next_offset: int = 0


@dataclass
class _RetainedImport:
    request_id: str
    builder: Any
    payload_bytes: int
    next_offset: int = 0


class SkippyCheckpointExportRegistry:
    """Retain native pages and expose strictly sequential bounded reads."""

    def __init__(
        self,
        *,
        runner: Any,
        kv_bytes_per_token: int,
        layer_start: int,
        layer_end: int,
        memory_snapshot: Callable[[], Any] = psutil.virtual_memory,
        reserve_bytes: Callable[[int, int], int] = configured_system_reserve_bytes,
        chunk_bytes: int = CHECKPOINT_CHUNK_BYTES,
        max_exports: int = MAX_CHECKPOINT_EXPORTS,
    ) -> None:
        if kv_bytes_per_token <= 0:
            raise ValueError("checkpoint KV geometry must be positive")
        if layer_start < 0 or layer_end <= layer_start:
            raise ValueError("checkpoint layer span is invalid")
        if chunk_bytes <= 0 or chunk_bytes > CHECKPOINT_CHUNK_BYTES:
            raise ValueError("checkpoint chunk size exceeds the wire bound")
        if max_exports <= 0:
            raise ValueError("checkpoint export count must be positive")
        self._runner = runner
        self._kv_bytes_per_token = int(kv_bytes_per_token)
        self._layer_start = int(layer_start)
        self._layer_end = int(layer_end)
        self._memory_snapshot = memory_snapshot
        self._reserve_bytes = reserve_bytes
        self._chunk_bytes = int(chunk_bytes)
        self._max_exports = int(max_exports)
        self._by_handle: dict[str, _RetainedExport] = {}
        self._handle_by_request: dict[str, str] = {}
        self._imports_by_handle: dict[str, _RetainedImport] = {}
        self._import_handle_by_request: dict[str, str] = {}

    def prepare(self, *, request_id: str, token_count: int) -> CheckpointExportDescriptor:
        if not request_id or len(request_id) > 256:
            raise ValueError("checkpoint request_id is invalid")
        if token_count <= 0:
            raise ValueError("checkpoint token count must be positive")

        previous = self._handle_by_request.get(request_id)
        if previous is None and len(self._by_handle) >= self._max_exports:
            raise CheckpointControlError("checkpoint export registry is full")

        estimated_bytes = self._kv_bytes_per_token * int(token_count)
        self._admit_allocation(estimated_bytes)

        page = self._runner.prepare_kv_page_export(
            request_id,
            token_start=0,
            token_count=int(token_count),
        )
        if page.payload_bytes != estimated_bytes:
            raise CheckpointControlError(
                "native KV page size differs from the signed execution geometry"
            )

        handle = secrets.token_hex(24)
        descriptor = CheckpointExportDescriptor(
            handle=handle,
            request_id=request_id,
            version=page.version,
            layer_start=page.layer_start,
            layer_end=page.layer_end,
            token_start=page.token_start,
            token_count=page.token_count,
            layer_count=page.layer_count,
            k_type=page.k_type,
            v_type=page.v_type,
            k_row_bytes=page.k_row_bytes,
            v_row_bytes=page.v_row_bytes,
            v_element_bytes=page.v_element_bytes,
            flags=page.flags,
            payload_bytes=page.payload_bytes,
            payload_sha256=page.payload_sha256,
            chunk_bytes=self._chunk_bytes,
        )
        if previous is not None:
            self.drop(handle=previous, request_id=request_id)
        self._by_handle[handle] = _RetainedExport(descriptor=descriptor, page=page)
        self._handle_by_request[request_id] = handle
        return descriptor

    def read(
        self,
        *,
        handle: str,
        request_id: str,
        offset: int,
    ) -> tuple[bytes, int, bool]:
        retained = self._require(handle=handle, request_id=request_id)
        if offset != retained.next_offset:
            raise CheckpointControlError("checkpoint chunks must be read sequentially")
        remaining = retained.descriptor.payload_bytes - offset
        if remaining <= 0:
            raise CheckpointControlError("checkpoint export is already complete")
        length = min(self._chunk_bytes, remaining)
        chunk = retained.page.read_chunk(offset, length)
        if len(chunk) != length:
            raise CheckpointControlError("native checkpoint chunk length changed")
        next_offset = offset + length
        retained.next_offset = next_offset
        done = next_offset == retained.descriptor.payload_bytes
        return chunk, next_offset, done

    def drop(self, *, handle: str, request_id: str) -> bool:
        retained = self._by_handle.get(handle)
        if retained is None:
            return False
        if retained.descriptor.request_id != request_id:
            raise CheckpointControlError("checkpoint handle belongs to another request")
        del self._by_handle[handle]
        self._handle_by_request.pop(request_id, None)
        return True

    def clear(self) -> None:
        self._by_handle.clear()
        self._handle_by_request.clear()
        self._imports_by_handle.clear()
        self._import_handle_by_request.clear()

    def drop_request(self, request_id: str) -> bool:
        handle = self._handle_by_request.get(request_id)
        if handle is None:
            return False
        return self.drop(handle=handle, request_id=request_id)

    def begin_import(
        self,
        *,
        request_id: str,
        descriptor: dict[str, int | str],
    ) -> str:
        if not request_id or len(request_id) > 256:
            raise ValueError("checkpoint request_id is invalid")
        required_ints = (
            "version",
            "layer_start",
            "layer_end",
            "token_start",
            "token_count",
            "layer_count",
            "k_type",
            "v_type",
            "k_row_bytes",
            "v_row_bytes",
            "v_element_bytes",
            "flags",
            "payload_bytes",
        )
        values: dict[str, int | str] = {}
        for name in required_ints:
            value = descriptor.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"checkpoint import {name} must be an integer")
            values[name] = value
        digest = descriptor.get("payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("checkpoint import SHA-256 is invalid")
        values["payload_sha256"] = digest
        if values["layer_start"] != self._layer_start or values["layer_end"] != self._layer_end:
            raise CheckpointControlError("checkpoint import targets a different layer span")
        if values["layer_count"] != self._layer_end - self._layer_start:
            raise CheckpointControlError("checkpoint import layer count differs from its span")
        if values["token_start"] != 0 or values["token_count"] <= 0:
            raise CheckpointControlError("checkpoint import token range is invalid")
        expected_bytes = self._kv_bytes_per_token * int(values["token_count"])
        if values["payload_bytes"] != expected_bytes:
            raise CheckpointControlError(
                "checkpoint import size differs from the signed execution geometry"
            )
        previous = self._import_handle_by_request.get(request_id)
        if previous is None and len(self._imports_by_handle) >= self._max_exports:
            raise CheckpointControlError("checkpoint import registry is full")
        self._admit_allocation(expected_bytes)
        builder = self._runner.begin_kv_page_import(values)
        handle = secrets.token_hex(24)
        if previous is not None:
            self.abort_import(handle=previous, request_id=request_id)
        self._imports_by_handle[handle] = _RetainedImport(
            request_id=request_id,
            builder=builder,
            payload_bytes=expected_bytes,
        )
        self._import_handle_by_request[request_id] = handle
        return handle

    def append_import(
        self,
        *,
        handle: str,
        request_id: str,
        offset: int,
        chunk: bytes,
    ) -> tuple[int, bool]:
        retained = self._require_import(handle=handle, request_id=request_id)
        if offset != retained.next_offset:
            raise CheckpointControlError("checkpoint import chunks must be sequential")
        if not chunk or len(chunk) > self._chunk_bytes:
            raise CheckpointControlError("checkpoint import chunk exceeds the wire bound")
        if offset + len(chunk) > retained.payload_bytes:
            raise CheckpointControlError("checkpoint import exceeds its declared payload")
        received = int(self._runner.append_kv_page_import(retained.builder, chunk))
        if received != offset + len(chunk):
            raise CheckpointControlError("native checkpoint import boundary changed")
        retained.next_offset = received
        return received, received == retained.payload_bytes

    def commit_import(self, *, handle: str, request_id: str) -> None:
        retained = self._require_import(handle=handle, request_id=request_id)
        if retained.next_offset != retained.payload_bytes:
            raise CheckpointControlError("checkpoint import payload is incomplete")
        self._runner.commit_kv_page_import(request_id, retained.builder)
        del self._imports_by_handle[handle]
        self._import_handle_by_request.pop(request_id, None)

    def abort_import(self, *, handle: str, request_id: str) -> bool:
        retained = self._imports_by_handle.get(handle)
        if retained is None:
            return False
        if retained.request_id != request_id:
            raise CheckpointControlError("checkpoint import handle belongs to another request")
        del self._imports_by_handle[handle]
        self._import_handle_by_request.pop(request_id, None)
        return True

    def _admit_allocation(self, payload_bytes: int) -> None:
        memory = self._memory_snapshot()
        total = int(memory.total)
        available = int(memory.available)
        reserve = int(self._reserve_bytes(total, available))
        if payload_bytes + self._chunk_bytes > max(0, available - reserve):
            raise CheckpointMemoryUnavailable(
                "checkpoint would cross the configured host-memory reserve"
            )

    def _require(self, *, handle: str, request_id: str) -> _RetainedExport:
        retained = self._by_handle.get(handle)
        if retained is None:
            raise CheckpointControlError("unknown checkpoint export handle")
        if retained.descriptor.request_id != request_id:
            raise CheckpointControlError("checkpoint handle belongs to another request")
        return retained

    def _require_import(self, *, handle: str, request_id: str) -> _RetainedImport:
        retained = self._imports_by_handle.get(handle)
        if retained is None:
            raise CheckpointControlError("unknown checkpoint import handle")
        if retained.request_id != request_id:
            raise CheckpointControlError("checkpoint import handle belongs to another request")
        return retained
