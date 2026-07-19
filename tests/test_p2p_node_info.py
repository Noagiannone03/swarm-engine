import time
from types import SimpleNamespace

from parallax.p2p.server import (
    GradientServer,
    ServerState,
    TransformerConnectionHandler,
    _resolve_worker_key_path,
)


class ProbeFuture:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def result(self, timeout=None):
        del timeout
        if self.error is not None:
            raise self.error
        return self.value


class ProbeStub:
    def __init__(self, future):
        self.future = future

    def rpc_health(self, request):
        assert request == {}
        return self.future


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


def test_worker_cannot_report_ready_when_required_frontend_is_dead():
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
    )
    values = {
        "frontend_required": True,
        "frontend_alive": False,
    }
    server._shared_state = SimpleNamespace(
        get=lambda key, default=None: values.get(key, default),
        get_status=lambda: ServerState.READY.value,
    )

    assert server._get_status() == ServerState.INITIALIZING.value


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


def test_worker_advertises_executor_measured_kv_geometry(monkeypatch):
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
        max_batch_size=8,
    )
    server.lattica = SimpleNamespace(peer_id=lambda: "worker-peer")
    server.rtt_last_update = time.time()
    values = {
        "max_concurrent_requests": 6,
        "kv_cache_token_capacity": 123456,
        "kv_cache_block_size": 64,
    }
    server._shared_state = SimpleNamespace(
        get=lambda key, default=None: values.get(key, default),
        get_metrics=lambda: {},
        get_status=lambda: ServerState.READY.value,
    )
    monkeypatch.setattr(
        "parallax.p2p.server.detect_node_hardware",
        lambda node_id: {"node_id": node_id, "device": "mlx"},
    )

    node_info = server.get_node_info(is_update=True)

    assert node_info["max_concurrent_requests"] == 6
    assert node_info["kv_cache_token_capacity"] == 123456
    assert node_info["kv_cache_block_size"] == 64


def test_worker_sends_account_credential_only_when_configured(monkeypatch):
    credential = "ab" * 32
    monkeypatch.setenv("FABI_ACCOUNT_TOKEN", credential)
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
    )
    server.lattica = SimpleNamespace(peer_id=lambda: "worker-peer")
    server.rtt_last_update = time.time()
    monkeypatch.setattr(
        "parallax.p2p.server.detect_node_hardware",
        lambda node_id: {"node_id": node_id, "device": "mlx"},
    )

    assert server.get_node_info()["account_token"] == credential

    monkeypatch.delenv("FABI_ACCOUNT_TOKEN")
    anonymous = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
    )
    anonymous.lattica = server.lattica
    anonymous.rtt_last_update = time.time()
    assert "account_token" not in anonymous.get_node_info()


def test_worker_reports_only_outbound_peers_reachable_by_registered_rpc():
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
    )
    server.lattica = SimpleNamespace(peer_id=lambda: "worker-peer")
    server.connection_handler = object()
    server.outbound_peer_ids = ["direct-peer", "relay-only-peer"]
    stubs = {
        "direct-peer": ProbeStub(ProbeFuture({"peer_id": "direct-peer"})),
        "relay-only-peer": ProbeStub(ProbeFuture(error=RuntimeError("relay only"))),
    }
    server.get_stub = stubs.__getitem__

    assert server._probe_outbound_peers() == ["direct-peer"]
    assert server.direct_peer_ids == ["direct-peer"]


def test_heartbeat_uses_cached_topology_without_running_network_probes(monkeypatch):
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
        max_batch_size=1,
    )
    server.lattica = SimpleNamespace(peer_id=lambda: "worker-peer")
    server.rtt_last_update = time.time()
    server.direct_peer_ids = ["qualified-peer"]
    server._probe_outbound_peers = lambda: (_ for _ in ()).throw(
        AssertionError("network probes must not run in the heartbeat path")
    )
    monkeypatch.setattr(
        "parallax.p2p.server.detect_node_hardware",
        lambda node_id: {"node_id": node_id, "device": "mlx"},
    )

    heartbeat = server.get_node_info(is_update=True)

    assert heartbeat["direct_peer_ids"] == ["qualified-peer"]


def test_transformer_health_rpc_returns_registered_peer_identity():
    handler = TransformerConnectionHandler.__new__(TransformerConnectionHandler)
    handler.lattica_instance = SimpleNamespace(peer_id=lambda: "worker-peer")

    assert handler.rpc_health({}) == {"peer_id": "worker-peer"}
