import hashlib
import time
from types import SimpleNamespace

from backend.server.recovery_checkpoint import RecoveryCheckpointCoordinator
from backend.server.recovery_checkpoint_store import EncryptedRecoveryCheckpointStore
from swarm_protocol.contracts import SkippyExactStateKind
from swarm_protocol.kv_snapshot import KvSnapshotCompatibility
from swarm_protocol.recovery import RecoveryState


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def span(start: int, end: int):
    return SimpleNamespace(start=start, end=end)


def plan(*, route_id: str, epoch: int, workers: tuple[str, str]):
    return SimpleNamespace(
        route_id=route_id,
        epoch=epoch,
        stages=(
            SimpleNamespace(worker_id=workers[0], effective_span=span(0, 2)),
            SimpleNamespace(worker_id=workers[1], effective_span=span(2, 4)),
        ),
    )


def snapshot(*, epoch: int):
    spec = SimpleNamespace(
        request_id="request-1",
        model_swarm_id=sha("swarm"),
        immutable_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        tokenizer_hash=sha("tokenizer"),
        dtype="bfloat16",
        prefill_contract_hash=sha("prefill"),
        attention_kv_contract_hash=sha("attention"),
    )
    replay = (10, 20, 30, 40, 41)
    return SimpleNamespace(
        spec=spec,
        state=RecoveryState.DECODING,
        epoch=epoch,
        replay_token_ids=replay,
        committed_position=2,
    )


class FakeCheckpointStub:
    def __init__(self, *, worker_id: str, layer_start: int, layer_end: int, fail_commit=False):
        self.worker_id = worker_id
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.payload = f"kv:{layer_start}-{layer_end}".encode()
        self.fail_commit = fail_commit
        self.imported = bytearray()
        self.discards = 0
        self.drops = 0

    def compatibility(self):
        return KvSnapshotCompatibility(
            model_swarm_id=sha("swarm"),
            immutable_revision="c1899de289a04d12100db370d81485cdf75e47ca",
            tokenizer_hash=sha("tokenizer"),
            dtype="bfloat16",
            prefill_contract_hash=sha("prefill"),
            attention_kv_contract_hash=sha("attention"),
            execution_plan_id="skippy-qwen3-q4",
            package_source_sha256=sha("weights"),
            runtime_release="mesh-llm/v0.75.1",
            runtime_abi_version="0.1.35",
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            state_kind=SkippyExactStateKind.DENSE_ATTENTION_KV,
            page_version=1,
            k_type=1,
            v_type=1,
            k_row_bytes=128,
            v_row_bytes=128,
            v_element_bytes=2,
        )

    def prepare_recovery_checkpoint(self, request):
        compatibility = self.compatibility()
        return {
            "handle": f"export:{self.worker_id}",
            "version": 1,
            "request_id": request["request_id"],
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "token_start": 0,
            "token_count": request["token_count"],
            "layer_count": self.layer_end - self.layer_start,
            "k_type": 1,
            "v_type": 1,
            "k_row_bytes": 128,
            "v_row_bytes": 128,
            "v_element_bytes": 2,
            "flags": 0,
            "payload_bytes": len(self.payload),
            "payload_sha256": hashlib.sha256(self.payload).hexdigest(),
            "chunk_bytes": 4 * 1024 * 1024,
            "compatibility": compatibility.to_wire_dict(),
            "compatibility_identity_hash": compatibility.identity_hash,
        }

    def read_recovery_checkpoint(self, request):
        offset = request["offset"]
        chunk = self.payload[offset:]
        return {"chunk": chunk, "next_offset": len(self.payload), "done": True}

    def drop_recovery_checkpoint(self, _request):
        self.drops += 1
        return {"dropped": True}

    def begin_recovery_checkpoint_import(self, _request):
        self.imported.clear()
        return {"handle": f"import:{self.worker_id}"}

    def write_recovery_checkpoint_import(self, request):
        assert request["offset"] == len(self.imported)
        self.imported.extend(request["chunk"])
        return {"next_offset": len(self.imported), "done": True}

    def commit_recovery_checkpoint_import(self, _request):
        if self.fail_commit:
            raise RuntimeError("simulated import failure")
        return {"committed": True}

    def abort_recovery_checkpoint_import(self, _request):
        self.imported.clear()
        return {"aborted": True}

    def discard_recovery_checkpoint_import(self, _request):
        self.discards += 1
        self.imported.clear()
        return {"discarded": True}


def wait_for_checkpoint(coordinator, replay_token_ids):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if coordinator.status()["ready_requests"] == 1:
            checkpoint = coordinator.checkpoint_for_recovery(
                request_id="request-1",
                failed_epoch=7,
                replay_token_ids=replay_token_ids,
            )
            assert checkpoint is not None
            return checkpoint
        time.sleep(0.01)
    raise AssertionError("checkpoint was not captured")


def test_coordinator_captures_every_stage_and_restores_only_an_exact_route(tmp_path):
    source_a = FakeCheckpointStub(worker_id="source-a", layer_start=0, layer_end=2)
    source_b = FakeCheckpointStub(worker_id="source-b", layer_start=2, layer_end=4)
    target_a = FakeCheckpointStub(worker_id="target-a", layer_start=0, layer_end=2)
    target_b = FakeCheckpointStub(worker_id="target-b", layer_start=2, layer_end=4)
    stubs = {stub.worker_id: stub for stub in (source_a, source_b, target_a, target_b)}
    store = EncryptedRecoveryCheckpointStore(
        tmp_path,
        minimum_free_bytes=0,
        maximum_store_bytes=1024 * 1024,
    )
    coordinator = RecoveryCheckpointCoordinator(store=store, get_stub=stubs.__getitem__)
    current = snapshot(epoch=7)

    assert coordinator.observe_committed(
        current,
        plan=plan(route_id="route-7", epoch=7, workers=("source-a", "source-b")),
    )
    checkpoint = wait_for_checkpoint(coordinator, current.replay_token_ids)
    assert coordinator._policy["request-1"].next_token_count >= checkpoint.token_count * 2
    assert not coordinator.observe_committed(
        current,
        plan=plan(route_id="route-7", epoch=7, workers=("source-a", "source-b")),
    )
    recovering = snapshot(epoch=8)

    assert coordinator.restore(
        checkpoint,
        snapshot=recovering,
        plan=plan(route_id="route-8", epoch=8, workers=("target-a", "target-b")),
    )
    assert bytes(target_a.imported) == source_a.payload
    assert bytes(target_b.imported) == source_b.payload
    assert source_a.drops == source_b.drops == 1
    coordinator.close()


def test_coordinator_discards_all_partial_state_when_one_stage_fails(tmp_path):
    source_a = FakeCheckpointStub(worker_id="source-a", layer_start=0, layer_end=2)
    source_b = FakeCheckpointStub(worker_id="source-b", layer_start=2, layer_end=4)
    target_a = FakeCheckpointStub(worker_id="target-a", layer_start=0, layer_end=2)
    target_b = FakeCheckpointStub(
        worker_id="target-b", layer_start=2, layer_end=4, fail_commit=True
    )
    stubs = {stub.worker_id: stub for stub in (source_a, source_b, target_a, target_b)}
    store = EncryptedRecoveryCheckpointStore(
        tmp_path,
        minimum_free_bytes=0,
        maximum_store_bytes=1024 * 1024,
    )
    coordinator = RecoveryCheckpointCoordinator(store=store, get_stub=stubs.__getitem__)
    current = snapshot(epoch=7)
    coordinator.observe_committed(
        current,
        plan=plan(route_id="route-7", epoch=7, workers=("source-a", "source-b")),
    )
    checkpoint = wait_for_checkpoint(coordinator, current.replay_token_ids)

    assert not coordinator.restore(
        checkpoint,
        snapshot=snapshot(epoch=8),
        plan=plan(route_id="route-8", epoch=8, workers=("target-a", "target-b")),
    )
    assert target_a.imported == target_b.imported == b""
    assert target_a.discards >= 1
    assert target_b.discards >= 1
    coordinator.close()
