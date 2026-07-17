import time
from types import SimpleNamespace

from parallax.p2p.server import GradientServer, ServerState, _resolve_worker_key_path


def test_worker_key_path_is_persistent_and_private(monkeypatch, tmp_path):
    key_path = tmp_path / "fabi" / "identity"
    monkeypatch.setenv("PARALLAX_KEY_PATH", str(key_path))

    assert _resolve_worker_key_path() == str(key_path)
    assert key_path.is_dir()
    assert key_path.stat().st_mode & 0o777 == 0o700


def test_manual_assignment_is_preserved_in_heartbeat(monkeypatch):
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
        block_start_index=2,
        block_end_index=28,
        max_batch_size=1,
        max_sequence_length=2048,
    )
    server.lattica = SimpleNamespace(peer_id=lambda: "worker-peer")
    server.rtt_last_update = time.time()
    server.status = ServerState.READY
    monkeypatch.setattr(
        "parallax.p2p.server.detect_node_hardware",
        lambda node_id: {"node_id": node_id},
    )

    heartbeat = server.get_node_info(is_update=True)

    assert heartbeat["manual_layer_assignment"] is True
    assert heartbeat["start_layer"] == 2
    assert heartbeat["end_layer"] == 28
