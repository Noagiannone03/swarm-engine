"""Adaptive full-route checkpoints and exact warm recovery orchestration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Any, Callable

from backend.server.recovery_checkpoint_store import (
    EncryptedRecoveryCheckpointStore,
    StoredRecoveryCheckpoint,
    canonical_checkpoint_aad,
)
from swarm_protocol.contracts import RoutePlan
from swarm_protocol.kv_snapshot import KvSnapshotCompatibility
from swarm_protocol.recovery import (
    RequestRecoverySnapshot,
    RecoveryState,
    token_sequence_checksum,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageRecoveryCheckpoint:
    source_worker_id: str
    layer_start: int
    layer_end: int
    descriptor: dict[str, object]
    stored: StoredRecoveryCheckpoint
    aad: bytes


@dataclass(frozen=True)
class RouteRecoveryCheckpoint:
    request_id: str
    model_swarm_id: str
    source_route_id: str
    source_epoch: int
    checkpoint_index: int
    token_count: int
    token_prefix_checksum: str
    stages: tuple[StageRecoveryCheckpoint, ...]


@dataclass
class _CheckpointPolicy:
    checkpoint_index: int = 0
    next_token_count: int = 0
    last_observed_position: int = 0
    last_observed_ns: int = 0
    output_tokens_per_second: float = 0.0
    disabled_epoch: int | None = None


@dataclass(frozen=True)
class _CheckpointCandidate:
    snapshot: RequestRecoverySnapshot
    plan: RoutePlan


def _rpc_result(value: Any) -> dict[str, object]:
    result_method = getattr(value, "result", None)
    result = result_method() if callable(result_method) else value
    if not isinstance(result, dict):
        raise RuntimeError("checkpoint RPC returned a non-object response")
    return result


def _route_authority(plan: RoutePlan) -> dict[str, object]:
    return {
        "fabi_route_id": plan.route_id,
        "fabi_route_epoch": plan.epoch,
        "parallax_routing_table": [stage.worker_id for stage in plan.stages],
    }


class RecoveryCheckpointCoordinator:
    """Create optional checkpoints without blocking SSE token publication."""

    def __init__(
        self,
        *,
        store: EncryptedRecoveryCheckpointStore,
        get_stub: Callable[[str], Any],
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.store = store
        self._get_stub = get_stub
        self._monotonic_ns = monotonic_ns
        self._condition = threading.Condition(threading.RLock())
        self._pending: dict[str, _CheckpointCandidate] = {}
        self._pending_order: deque[str] = deque()
        self._inflight: set[str] = set()
        self._latest: dict[str, RouteRecoveryCheckpoint] = {}
        self._policy: dict[str, _CheckpointPolicy] = {}
        self._finished_requests: set[str] = set()
        self._failed_epoch_by_request: dict[str, int] = {}
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="FabiRecoveryCheckpoint",
            daemon=True,
        )
        self._worker.start()

    def observe_committed(
        self,
        snapshot: RequestRecoverySnapshot,
        plan: RoutePlan,
    ) -> bool:
        """Queue the newest safe boundary using measured checkpoint overhead."""

        if snapshot.state != RecoveryState.DECODING or snapshot.epoch != plan.epoch:
            return False
        token_count = len(snapshot.replay_token_ids) - 1
        if token_count <= 0:
            return False
        now_ns = self._monotonic_ns()
        with self._condition:
            if self._closed:
                return False
            policy = self._policy.setdefault(snapshot.spec.request_id, _CheckpointPolicy())
            self._finished_requests.discard(snapshot.spec.request_id)
            if plan.epoch <= self._failed_epoch_by_request.get(snapshot.spec.request_id, -1):
                return False
            if policy.disabled_epoch == plan.epoch:
                return False
            if (
                policy.last_observed_ns
                and snapshot.committed_position > policy.last_observed_position
            ):
                elapsed = max((now_ns - policy.last_observed_ns) / 1_000_000_000, 1e-9)
                instantaneous = (
                    snapshot.committed_position - policy.last_observed_position
                ) / elapsed
                policy.output_tokens_per_second = (
                    instantaneous
                    if policy.output_tokens_per_second <= 0
                    else policy.output_tokens_per_second * 0.75 + instantaneous * 0.25
                )
            policy.last_observed_ns = now_ns
            policy.last_observed_position = snapshot.committed_position
            if token_count < policy.next_token_count:
                return False
            request_id = snapshot.spec.request_id
            candidate = _CheckpointCandidate(snapshot=snapshot, plan=plan)
            if request_id in self._pending:
                self._pending[request_id] = candidate
                return True
            self._pending[request_id] = candidate
            self._pending_order.append(request_id)
            self._condition.notify()
            return True

    def checkpoint_for_recovery(
        self,
        *,
        request_id: str,
        failed_epoch: int,
        replay_token_ids: tuple[int, ...],
    ) -> RouteRecoveryCheckpoint | None:
        with self._condition:
            previous_failed_epoch = self._failed_epoch_by_request.get(request_id, -1)
            self._failed_epoch_by_request[request_id] = max(
                previous_failed_epoch,
                failed_epoch,
            )
            self._pending.pop(request_id, None)
            checkpoint = self._latest.get(request_id)
        if checkpoint is None or checkpoint.source_epoch != failed_epoch:
            return None
        if checkpoint.token_count >= len(replay_token_ids):
            return None
        if (
            token_sequence_checksum(replay_token_ids[: checkpoint.token_count])
            != checkpoint.token_prefix_checksum
        ):
            return None
        return checkpoint

    def restore(
        self,
        checkpoint: RouteRecoveryCheckpoint,
        *,
        snapshot: RequestRecoverySnapshot,
        plan: RoutePlan,
    ) -> bool:
        """Import a complete compatible route or erase every partial import."""

        if (
            checkpoint.request_id != snapshot.spec.request_id
            or checkpoint.model_swarm_id != snapshot.spec.model_swarm_id
            or snapshot.epoch != plan.epoch
        ):
            return False
        by_span = {(stage.layer_start, stage.layer_end): stage for stage in checkpoint.stages}
        targets = []
        for stage in plan.stages:
            key = (stage.effective_span.start, stage.effective_span.end)
            source = by_span.get(key)
            if source is None:
                return False
            targets.append((stage, source))
        if len(targets) != len(checkpoint.stages):
            return False

        authority = _route_authority(plan)
        begun: list[tuple[Any, str]] = []
        try:
            for target, source in targets:
                stub = self._get_stub(target.worker_id)
                begin = _rpc_result(
                    stub.begin_recovery_checkpoint_import(
                        {
                            "request_id": checkpoint.request_id,
                            "route_authority": authority,
                            "descriptor": source.descriptor,
                        }
                    )
                )
                handle = begin.get("handle")
                if not isinstance(handle, str) or not handle:
                    raise RuntimeError("checkpoint import returned no handle")
                begun.append((stub, handle))
                offset = 0
                for chunk in self.store.read_chunks(source.stored, aad=source.aad):
                    written = _rpc_result(
                        stub.write_recovery_checkpoint_import(
                            {
                                "request_id": checkpoint.request_id,
                                "route_authority": authority,
                                "handle": handle,
                                "offset": offset,
                                "chunk": chunk,
                            }
                        )
                    )
                    next_offset = written.get("next_offset")
                    if next_offset != offset + len(chunk):
                        raise RuntimeError("checkpoint import offset diverged")
                    offset = int(next_offset)
                if offset != source.stored.plaintext_bytes:
                    raise RuntimeError("checkpoint import ended at the wrong byte boundary")
                committed = _rpc_result(
                    stub.commit_recovery_checkpoint_import(
                        {
                            "request_id": checkpoint.request_id,
                            "route_authority": authority,
                            "handle": handle,
                        }
                    )
                )
                if committed.get("committed") is not True:
                    raise RuntimeError("checkpoint import was not committed")
                begun.pop()
            return True
        except BaseException:
            logger.warning(
                "Warm checkpoint restore failed for %s; using exact cold replay",
                checkpoint.request_id,
                exc_info=True,
            )
            for stub, handle in begun:
                try:
                    _rpc_result(
                        stub.abort_recovery_checkpoint_import(
                            {
                                "request_id": checkpoint.request_id,
                                "route_authority": authority,
                                "handle": handle,
                            }
                        )
                    )
                except BaseException:
                    logger.debug("Unable to abort partial checkpoint import", exc_info=True)
            # A stage may have committed immediately before its response failed.
            # Discard on every target, not only those acknowledged above.
            for target, _source in targets:
                try:
                    stub = self._get_stub(target.worker_id)
                except BaseException:
                    continue
                self._discard_import_best_effort(stub, checkpoint, authority)
            return False

    def finish_request(self, request_id: str) -> None:
        with self._condition:
            checkpoint = self._latest.pop(request_id, None)
            self._pending.pop(request_id, None)
            self._policy.pop(request_id, None)
            self._failed_epoch_by_request.pop(request_id, None)
            if request_id in self._inflight:
                self._finished_requests.add(request_id)
            else:
                self._finished_requests.discard(request_id)
        if checkpoint is not None:
            self._remove_checkpoint(checkpoint)

    def status(self) -> dict[str, object]:
        with self._condition:
            return {
                "enabled": not self._closed,
                "ready_requests": len(self._latest),
                "pending_requests": len(self._pending),
                "inflight_requests": len(self._inflight),
            }

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._pending.clear()
            self._pending_order.clear()
            self._condition.notify_all()
        self._worker.join(timeout=1.0)
        if not self._worker.is_alive():
            self.store.close()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._pending_order and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    request_id = self._pending_order.popleft()
                    candidate = self._pending.pop(request_id, None)
                    if candidate is None:
                        continue
                    policy = self._policy.setdefault(request_id, _CheckpointPolicy())
                    candidate_token_count = len(candidate.snapshot.replay_token_ids) - 1
                    if (
                        policy.disabled_epoch == candidate.plan.epoch
                        or candidate_token_count < policy.next_token_count
                    ):
                        continue
                    self._inflight.add(request_id)
                    checkpoint_index = policy.checkpoint_index
                started_ns = self._monotonic_ns()
                checkpoint = None
                try:
                    checkpoint = self._capture(candidate, checkpoint_index=checkpoint_index)
                except BaseException:
                    logger.warning(
                        "Warm checkpoint capture failed for %s; cold replay remains available",
                        request_id,
                        exc_info=True,
                    )
                completed_ns = self._monotonic_ns()
                with self._condition:
                    self._inflight.discard(request_id)
                    failed_epoch = self._failed_epoch_by_request.get(request_id, -1)
                    if (
                        request_id in self._finished_requests
                        or self._closed
                        or candidate.plan.epoch <= failed_epoch
                    ):
                        self._finished_requests.discard(request_id)
                        if checkpoint is not None:
                            self._remove_checkpoint(checkpoint)
                        continue
                    policy = self._policy.setdefault(request_id, _CheckpointPolicy())
                    if checkpoint is None:
                        policy.disabled_epoch = candidate.plan.epoch
                        continue
                    previous = self._latest.get(request_id)
                    self._latest[request_id] = checkpoint
                    policy.checkpoint_index += 1
                    copy_seconds = max((completed_ns - started_ns) / 1_000_000_000, 0.0)
                    tokens_during_copy = max(
                        1,
                        math.ceil(policy.output_tokens_per_second * copy_seconds),
                    )
                    # A full-prefix checkpoint is written again only after at
                    # least an equal amount of new KV has accumulated.  This
                    # geometric cadence bounds write amplification while the
                    # measured in-flight token count raises the gap further on
                    # slow links or busy workers.
                    policy.next_token_count = checkpoint.token_count + max(
                        checkpoint.token_count,
                        tokens_during_copy * 2,
                    )
                if previous is not None:
                    self._remove_checkpoint(previous)
        finally:
            self.store.close()

    def _capture(
        self,
        candidate: _CheckpointCandidate,
        *,
        checkpoint_index: int,
    ) -> RouteRecoveryCheckpoint:
        snapshot = candidate.snapshot
        plan = candidate.plan
        token_count = len(snapshot.replay_token_ids) - 1
        token_checksum = token_sequence_checksum(snapshot.replay_token_ids[:token_count])
        authority = _route_authority(plan)
        captured: list[StageRecoveryCheckpoint] = []
        try:
            for stage_number, stage in enumerate(plan.stages):
                with self._condition:
                    if self._closed:
                        raise RuntimeError("checkpoint coordinator is closing")
                stub = self._get_stub(stage.worker_id)
                prepared = _rpc_result(
                    stub.prepare_recovery_checkpoint(
                        {
                            "request_id": snapshot.spec.request_id,
                            "route_authority": authority,
                            "token_count": token_count,
                        }
                    )
                )
                handle = prepared.get("handle")
                if not isinstance(handle, str) or not handle:
                    raise RuntimeError("checkpoint export returned no handle")
                try:
                    descriptor = dict(prepared)
                    descriptor.pop("handle", None)
                    self._validate_descriptor(
                        descriptor,
                        snapshot=snapshot,
                        layer_start=stage.effective_span.start,
                        layer_end=stage.effective_span.end,
                        token_count=token_count,
                    )
                    descriptor["token_prefix_checksum"] = token_checksum
                    aad = canonical_checkpoint_aad(
                        {
                            "request_id": snapshot.spec.request_id,
                            "model_swarm_id": snapshot.spec.model_swarm_id,
                            "source_route_id": plan.route_id,
                            "source_epoch": plan.epoch,
                            "checkpoint_index": checkpoint_index,
                            "token_count": token_count,
                            "token_prefix_checksum": token_checksum,
                            "source_worker_id": stage.worker_id,
                            "stage_number": stage_number,
                            "descriptor": descriptor,
                        }
                    )
                    payload_bytes = int(descriptor["payload_bytes"])
                    checkpoint_id = (
                        f"{snapshot.spec.request_id}:{checkpoint_index}:"
                        f"{stage.effective_span.start}-{stage.effective_span.end}"
                    )
                    writer = self.store.begin(
                        checkpoint_id=checkpoint_id,
                        plaintext_bytes=payload_bytes,
                        aad=aad,
                    )
                    try:
                        offset = 0
                        while offset < payload_bytes:
                            response = _rpc_result(
                                stub.read_recovery_checkpoint(
                                    {
                                        "request_id": snapshot.spec.request_id,
                                        "route_authority": authority,
                                        "handle": handle,
                                        "offset": offset,
                                    }
                                )
                            )
                            chunk = response.get("chunk")
                            next_offset = response.get("next_offset")
                            if not isinstance(chunk, bytes) or not chunk:
                                raise RuntimeError("checkpoint export returned an empty chunk")
                            if next_offset != offset + len(chunk):
                                raise RuntimeError("checkpoint export offset diverged")
                            writer.write(chunk)
                            offset = int(next_offset)
                            done = response.get("done")
                            if not isinstance(done, bool) or done != (offset == payload_bytes):
                                raise RuntimeError("checkpoint export completion flag diverged")
                        stored = writer.finalize()
                    except BaseException:
                        writer.abort()
                        raise
                    captured.append(
                        StageRecoveryCheckpoint(
                            source_worker_id=stage.worker_id,
                            layer_start=stage.effective_span.start,
                            layer_end=stage.effective_span.end,
                            descriptor=descriptor,
                            stored=stored,
                            aad=aad,
                        )
                    )
                finally:
                    try:
                        _rpc_result(
                            stub.drop_recovery_checkpoint(
                                {
                                    "request_id": snapshot.spec.request_id,
                                    "route_authority": authority,
                                    "handle": handle,
                                }
                            )
                        )
                    except BaseException:
                        logger.debug("Unable to drop source checkpoint handle", exc_info=True)
            return RouteRecoveryCheckpoint(
                request_id=snapshot.spec.request_id,
                model_swarm_id=snapshot.spec.model_swarm_id,
                source_route_id=plan.route_id,
                source_epoch=plan.epoch,
                checkpoint_index=checkpoint_index,
                token_count=token_count,
                token_prefix_checksum=token_checksum,
                stages=tuple(captured),
            )
        except BaseException:
            for stage in captured:
                self.store.remove(stage.stored.checkpoint_id)
            raise

    @staticmethod
    def _validate_descriptor(
        descriptor: dict[str, object],
        *,
        snapshot: RequestRecoverySnapshot,
        layer_start: int,
        layer_end: int,
        token_count: int,
    ) -> None:
        if (
            descriptor.get("layer_start") != layer_start
            or descriptor.get("layer_end") != layer_end
            or descriptor.get("token_start") != 0
            or descriptor.get("token_count") != token_count
        ):
            raise ValueError("checkpoint descriptor differs from its route stage")
        payload_bytes = descriptor.get("payload_bytes")
        payload_sha256 = descriptor.get("payload_sha256")
        if (
            isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes <= 0
        ):
            raise ValueError("checkpoint payload size is invalid")
        if not isinstance(payload_sha256, str) or len(payload_sha256) != 64:
            raise ValueError("checkpoint payload digest is invalid")
        try:
            bytes.fromhex(payload_sha256)
        except ValueError as error:
            raise ValueError("checkpoint payload digest is invalid") from error
        if payload_sha256.lower() != payload_sha256:
            raise ValueError("checkpoint payload digest is invalid")
        compatibility = KvSnapshotCompatibility.from_wire_dict(descriptor.get("compatibility"))
        if (
            compatibility.model_swarm_id != snapshot.spec.model_swarm_id
            or compatibility.immutable_revision != snapshot.spec.immutable_revision
            or compatibility.tokenizer_hash != snapshot.spec.tokenizer_hash
            or compatibility.dtype != snapshot.spec.dtype
            or compatibility.prefill_contract_hash != snapshot.spec.prefill_contract_hash
            or compatibility.attention_kv_contract_hash != snapshot.spec.attention_kv_contract_hash
        ):
            raise ValueError("checkpoint compatibility differs from the recovery journal")
        identity = descriptor.get("compatibility_identity_hash")
        if identity != compatibility.identity_hash:
            raise ValueError("checkpoint compatibility identity is invalid")

    def _discard_import_best_effort(
        self,
        stub: Any,
        checkpoint: RouteRecoveryCheckpoint,
        authority: dict[str, object],
    ) -> None:
        try:
            _rpc_result(
                stub.discard_recovery_checkpoint_import(
                    {
                        "request_id": checkpoint.request_id,
                        "route_authority": authority,
                        "token_count": checkpoint.token_count,
                        "token_prefix_checksum": checkpoint.token_prefix_checksum,
                    }
                )
            )
        except BaseException:
            logger.debug("Unable to discard imported checkpoint", exc_info=True)

    def _remove_checkpoint(self, checkpoint: RouteRecoveryCheckpoint) -> None:
        for stage in checkpoint.stages:
            self.store.remove(stage.stored.checkpoint_id)
