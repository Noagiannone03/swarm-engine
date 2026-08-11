import hashlib

import pytest

from swarm_protocol.kv_snapshot import (
    KvSnapshotCompatibility,
    KvSnapshotEnvelope,
    KvSnapshotIncompatible,
    KvStateKind,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _compatibility(**updates) -> KvSnapshotCompatibility:
    values = {
        "model_swarm_id": _sha("swarm"),
        "immutable_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "tokenizer_hash": _sha("tokenizer"),
        "dtype": "bfloat16",
        "prefill_contract_hash": _sha("prefill"),
        "attention_kv_contract_hash": _sha("attention"),
        "execution_plan_id": "skippy-qwen3-q4",
        "package_source_sha256": _sha("weights"),
        "runtime_release": "mesh-llm/v0.75.1",
        "runtime_abi_version": "0.1.35",
        "layer_start": 0,
        "layer_end": 28,
        "state_kind": KvStateKind.DENSE_ATTENTION_KV,
        "page_version": 1,
        "k_type": 1,
        "v_type": 1,
        "k_row_bytes": 128,
        "v_row_bytes": 128,
        "v_element_bytes": 2,
    }
    values.update(updates)
    return KvSnapshotCompatibility(**values)


def test_snapshot_accepts_exact_target_and_returns_only_uncheckpointed_suffix():
    compatibility = _compatibility()
    payload = b"native-kv-page"
    checkpoint = KvSnapshotEnvelope.create(
        compatibility=compatibility,
        request_id="request-1",
        source_worker_id="worker-a",
        source_route_id="route-a",
        source_epoch=7,
        checkpoint_index=3,
        replay_token_ids=(10, 20, 30),
        payload=payload,
    )

    compatibility.require_compatible(_compatibility())
    checkpoint.validate_payload(payload)
    assert checkpoint.resume_suffix((10, 20, 30, 40, 50)) == (40, 50)
    assert compatibility.identity_hash == _compatibility().identity_hash
    assert KvSnapshotCompatibility.from_wire_dict(compatibility.to_wire_dict()) == compatibility


def test_snapshot_rejects_runtime_or_native_layout_mismatch():
    source = _compatibility()

    with pytest.raises(KvSnapshotIncompatible, match="different KV identity"):
        source.require_compatible(_compatibility(runtime_abi_version="0.1.33"))
    with pytest.raises(KvSnapshotIncompatible, match="different KV identity"):
        source.require_compatible(_compatibility(k_row_bytes=256))


def test_snapshot_rejects_corrupt_payload_and_divergent_journal():
    checkpoint = KvSnapshotEnvelope.create(
        compatibility=_compatibility(),
        request_id="request-1",
        source_worker_id="worker-a",
        source_route_id="route-a",
        source_epoch=7,
        checkpoint_index=3,
        replay_token_ids=(10, 20, 30),
        payload=b"native-kv-page",
    )

    with pytest.raises(KvSnapshotIncompatible, match="digest"):
        checkpoint.validate_payload(b"native-kv-pagf")
    with pytest.raises(KvSnapshotIncompatible, match="differs from the journal"):
        checkpoint.resume_suffix((10, 99, 30, 40))
