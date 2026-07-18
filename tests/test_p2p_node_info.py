import time
from types import SimpleNamespace

from parallax.p2p.server import GradientServer, ServerState, _resolve_worker_key_path


def test_worker_key_path_is_persistent_and_private(monkeypatch, tmp_path):
    key_path = tmp_path / "fabi" / "identity"
    monkeypatch.setenv("PARALLAX_KEY_PATH", str(key_path))

    assert _resolve_worker_key_path() == str(key_path)
    assert key_path.is_dir()
    assert key_path.stat().st_mode & 0o777 == 0o700


def test_shutdown_notifies_scheduler_when_shared_state_is_already_closed(monkeypatch):
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
    )
    leaves = []
    closed = []
    server.scheduler_stub = SimpleNamespace(node_leave=leaves.append)
    server.rtt_last_update = time.time()
    server.lattica = SimpleNamespace(
        peer_id=lambda: "worker-peer",
        get_all_peers=lambda: [],
        close=lambda: closed.append(True),
    )
    server._shared_state = SimpleNamespace(
        update=lambda **values: (_ for _ in ()).throw(EOFError()),
        get_status=lambda: (_ for _ in ()).throw(BrokenPipeError()),
    )
    monkeypatch.setattr(
        "parallax.p2p.server.detect_node_hardware",
        lambda node_id: {"node_id": node_id},
    )

    server.shutdown()

    assert leaves[0]["node_id"] == "worker-peer"
    assert leaves[0]["status"] == ServerState.OFFLINE.value
    assert closed == [True]
    assert server._shared_state is None
    assert server.status is ServerState.OFFLINE


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


def test_worker_advertises_frontend_capability(monkeypatch):
    monkeypatch.setattr(
        "parallax.p2p.server.vllm_rust_frontend_available",
        lambda: False,
    )
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
    )
    server.lattica = SimpleNamespace(peer_id=lambda: "worker-peer")
    server.rtt_last_update = time.time()
    monkeypatch.setattr(
        "parallax.p2p.server.detect_node_hardware",
        lambda node_id: {"node_id": node_id},
    )

    node_info = server.get_node_info()

    assert node_info["supports_frontend"] is False


def test_worker_advertises_runtime_chunked_prefill_capability(monkeypatch):
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
        gpu_backend="vllm",
        chunked_prefill_size=1024,
    )
    server.lattica = SimpleNamespace(peer_id=lambda: "worker-peer")
    server.rtt_last_update = time.time()
    monkeypatch.setattr(
        "parallax.p2p.server.detect_node_hardware",
        lambda node_id: {"node_id": node_id, "device": "cuda"},
    )

    node_info = server.get_node_info()

    assert node_info["supports_chunked_prefill"] is False
    assert node_info["preferred_chunked_prefill_size"] == 1024
    assert node_info["chunked_prefill_size"] == 0


def test_mlx_worker_keeps_chunked_prefill_enabled(monkeypatch):
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
        gpu_backend="vllm",
        chunked_prefill_size=1024,
    )
    server.lattica = SimpleNamespace(peer_id=lambda: "worker-peer")
    server.rtt_last_update = time.time()
    monkeypatch.setattr(
        "parallax.p2p.server.detect_node_hardware",
        lambda node_id: {"node_id": node_id, "device": "mlx"},
    )

    node_info = server.get_node_info()

    assert node_info["supports_chunked_prefill"] is True
    assert node_info["preferred_chunked_prefill_size"] == 1024
    assert node_info["chunked_prefill_size"] == 1024
