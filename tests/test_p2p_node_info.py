import threading
import time
from types import SimpleNamespace

import pytest

from parallax.p2p.proto import forward_pb2
from parallax.p2p.server import (
    GradientServer,
    ServerState,
    TransformerConnectionHandler,
    _resolve_worker_key_path,
    send_notify,
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
    def __init__(self, future, link_future=None):
        self.future = future
        self.link_future = link_future

    def rpc_health(self, request):
        assert request == {}
        return self.future

    def rpc_link_probe(self, request):
        assert isinstance(request, bytes)
        if self.link_future is None:
            raise RuntimeError("link probe is not configured")
        return self.link_future


class RecordingSocket:
    def __init__(self, error=None):
        self.error = error
        self.messages = []

    def send_multipart(self, message):
        if self.error is not None:
            raise self.error
        self.messages.append(message)


def build_forward_handler(socket):
    handler = object.__new__(TransformerConnectionHandler)
    handler._recv_from_peer_lock = threading.Lock()
    handler._recv_from_peer = socket
    handler.notify_url = None
    handler.block_start_index = None
    handler.block_end_index = None
    return handler


def test_disabled_notification_does_not_require_an_assigned_span():
    request = forward_pb2.ForwardRequest()
    request.reqs.add(rid="request-1")

    assert send_notify(None, None, None, request, "started") is None


def test_dynamic_span_handler_enqueues_after_standby_assignment():
    socket = RecordingSocket()
    handler = build_forward_handler(socket)
    handler.update_serving_span(1, 28)
    request = forward_pb2.ForwardRequest()
    request.reqs.add(rid="request-1")

    handler.rpc_pp_forward(request)

    assert handler.block_start_index == 1
    assert handler.block_end_index == 28
    assert socket.messages == [[b"forward", request.SerializeToString()]]


def test_forward_admission_uses_signed_authority_id_not_engine_request_id(monkeypatch):
    socket = RecordingSocket()
    handler = build_forward_handler(socket)
    calls = []
    handler.execution_admission = SimpleNamespace(
        authorize_forward=lambda **values: calls.append(values)
    )
    monkeypatch.setattr("parallax.p2p.server.authenticated_rpc_peer_id", lambda: "mac-endpoint")
    request = forward_pb2.ForwardRequest()
    request.reqs.add(
        rid="chatcmpl-engine-request",
        authority_request_id="scheduler-request",
        route_id="route-7",
        route_epoch=7,
        routing_table=["mac-worker", "rtx-worker"],
    )

    handler.rpc_pp_forward(request)

    assert calls == [
        {
            "request_id": "scheduler-request",
            "route_id": "route-7",
            "epoch": 7,
            "routing_table": ("mac-worker", "rtx-worker"),
            "caller_endpoint_id": "mac-endpoint",
        }
    ]
    assert socket.messages == [[b"forward", request.SerializeToString()]]


def test_autonomous_span_reload_fences_ingress_and_updates_shared_generation():
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
        block_start_index=0,
        block_end_index=2,
    )
    values = {}

    class State:
        def get(self, key, default=None):
            return values.get(key, default)

        def update(self, **changes):
            values.update(changes)

    spans = []
    server._shared_state = State()
    server.swarm_v3_execution_admission = object()
    server.connection_handler = SimpleNamespace(
        update_serving_span=lambda start, end: spans.append((start, end))
    )

    server._apply_v3_span_reload(
        span=SimpleNamespace(start=2, end=4),
        generation=7,
    )

    assert spans == [(2, 4)]
    assert server.block_start_index == 2
    assert server.block_end_index == 4
    assert server.status is ServerState.INITIALIZING
    assert values["_layer_allocation_changed"] is True
    assert values["swarm_v3_placement_generation"] == 7
    assert values["swarm_v3_placement_phase"] == "building"


def test_forward_enqueue_failure_is_not_reported_as_success():
    handler = build_forward_handler(RecordingSocket(error=RuntimeError("enqueue failed")))
    request = forward_pb2.ForwardRequest()
    request.reqs.add(rid="request-1")

    with pytest.raises(RuntimeError, match="enqueue failed"):
        handler.rpc_pp_forward(request)


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


def test_worker_heartbeat_carries_non_blocking_v3_shadow_report(monkeypatch):
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
        block_start_index=0,
        block_end_index=4,
        gpu_backend="sglang",
    )
    server.lattica = SimpleNamespace(peer_id=lambda: "worker-peer")
    server.rtt_last_update = time.time()
    server.model_name = "test/model"
    server.model_revision = "0123456789abcdef0123456789abcdef01234567"
    captured = {}
    server.swarm_v3_reporter = SimpleNamespace(
        snapshot=lambda serving: captured.setdefault(
            "report", {"mode": "shadow", "state": "verifying", "span": serving.span}
        )
    )
    values = {
        "max_concurrent_requests": 2,
        "kv_cache_token_capacity": 4096,
        "kv_cache_block_size": 16,
        "memory_contract_failure": None,
        "memory_pressure": "normal",
        "memory_pressure_resources": {},
    }
    server._shared_state = SimpleNamespace(
        get=lambda key, default=None: values.get(key, default),
        get_status=lambda: ServerState.READY.value,
        get_metrics=lambda: {"current_requests": 0},
    )
    monkeypatch.setattr(
        "parallax.p2p.server.detect_node_hardware",
        lambda node_id: {
            "node_id": node_id,
            "device": "cuda",
            "usable_memory_bytes": 8 * 1024**3,
        },
    )

    heartbeat = server.get_node_info(is_update=True)

    assert heartbeat["swarm_v3"]["state"] == "verifying"
    assert captured["report"]["span"].start == 0
    assert captured["report"]["span"].end == 4


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


def test_worker_capacity_envelope_is_immutable_for_process_generation(monkeypatch):
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
    )
    server.lattica = SimpleNamespace(peer_id=lambda: "worker-peer")
    server.rtt_last_update = time.time()
    detections = [
        {
            "node_id": "worker-peer",
            "device": "mlx",
            "usable_memory_bytes": 9_000,
            "system_available_memory_bytes": 11_000,
        },
        {
            "node_id": "worker-peer",
            "device": "mlx",
            "usable_memory_bytes": 1_000,
            "system_available_memory_bytes": 2_000,
        },
    ]
    calls = 0

    def detect(_node_id):
        nonlocal calls
        result = detections[calls]
        calls += 1
        return result

    monkeypatch.setattr("parallax.p2p.server.detect_node_hardware", detect)

    initial = server.get_node_info()
    heartbeat = server.get_node_info(is_update=True)

    assert calls == 1
    assert initial["hardware"]["usable_memory_bytes"] == 9_000
    assert heartbeat["hardware"]["usable_memory_bytes"] == 9_000
    initial["hardware"]["usable_memory_bytes"] = 0
    assert server.get_node_info(is_update=True)["hardware"]["usable_memory_bytes"] == 9_000


def test_transformer_health_rpc_returns_registered_peer_identity():
    handler = TransformerConnectionHandler.__new__(TransformerConnectionHandler)
    handler.lattica_instance = SimpleNamespace(peer_id=lambda: "worker-peer")

    assert handler.rpc_health({}) == {"peer_id": "worker-peer"}


def test_link_probe_accepts_only_bounded_payload_from_assigned_iroh_peer(monkeypatch):
    handler = TransformerConnectionHandler.__new__(TransformerConnectionHandler)
    handler.iroh_transport = SimpleNamespace(peer_id=lambda: "receiving-worker")
    handler.link_probe_authorizer = lambda peer_id: peer_id == "sending-worker"
    handler._link_probe_lock = threading.Lock()
    handler._link_probe_last_received = {}
    monkeypatch.setattr(
        "parallax.p2p.server.authenticated_rpc_peer_id",
        lambda: "sending-worker",
    )
    payload = bytes(64 * 1024)

    assert handler.rpc_link_probe(payload) == {
        "peer_id": "receiving-worker",
        "received_bytes": len(payload),
    }

    with pytest.raises(RuntimeError, match="rate limited"):
        handler.rpc_link_probe(payload)


def test_link_probe_rejects_unassigned_peer_before_reading_payload(monkeypatch):
    handler = TransformerConnectionHandler.__new__(TransformerConnectionHandler)
    handler.iroh_transport = SimpleNamespace(peer_id=lambda: "receiving-worker")
    handler.link_probe_authorizer = lambda peer_id: False
    handler._link_probe_lock = threading.Lock()
    handler._link_probe_last_received = {}
    monkeypatch.setattr(
        "parallax.p2p.server.authenticated_rpc_peer_id",
        lambda: "unassigned-worker",
    )

    with pytest.raises(PermissionError, match="not an assigned peer"):
        handler.rpc_link_probe(bytes(64 * 1024))


def test_worker_calibrates_cold_link_with_application_goodput(monkeypatch):
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
    )
    server.iroh_transport = object()
    server.link_probe_bytes = 64 * 1024
    server.link_probe_payload = bytes(server.link_probe_bytes)
    server.get_stub = lambda peer_id: ProbeStub(
        ProbeFuture({"peer_id": peer_id}),
        ProbeFuture(
            {
                "peer_id": peer_id,
                "received_bytes": server.link_probe_bytes,
            }
        ),
    )
    timestamps = iter((1_000_000_000, 1_100_000_000))
    monkeypatch.setattr(
        "parallax.p2p.server.time.perf_counter_ns",
        lambda: next(timestamps),
    )

    assert server._probe_peer_goodput("next-worker") is True
    assert server.link_throughputs["next-worker"]["bytes_per_second"] == pytest.approx(655_360.0)
    assert server._probe_peer_goodput("next-worker") is False


def test_worker_advertises_qualified_link_when_goodput_sample_expires(monkeypatch):
    server = GradientServer(
        recv_from_peer_addr="",
        send_to_peer_addr="",
        scheduler_addr="scheduler-peer",
    )
    server.lattica = SimpleNamespace(peer_id=lambda: "mac")
    server.reachable_peer_ids = ["rtx"]
    server.direct_peer_ids = ["rtx"]
    server.relayed_peer_ids = []
    server.rtts = {"rtx": 12.5}
    server.link_path_observed_at_ms = {"rtx": 200_000}
    server.link_throughputs = {
        "rtx": {
            "bytes_per_second": 100_000_000.0,
            "measured_at_ms": 1,
        }
    }
    monkeypatch.setattr(
        "parallax.p2p.server.time.time_ns",
        lambda: 205_000 * 1_000_000,
    )

    [metric] = server._v3_outgoing_link_metrics()

    assert metric.from_worker_id == "mac"
    assert metric.to_worker_id == "rtx"
    assert metric.throughput_bytes_per_second is None
    assert metric.throughput_measured_at_ms is None
    assert metric.measured_at_ms == 200_000
    assert metric.expires_at_ms == 215_000


def test_worker_builds_iroh_with_explicit_scheduler_endpoint(monkeypatch):
    transport = SimpleNamespace(peer_id=lambda: "worker-endpoint")
    monkeypatch.setattr(
        "parallax.p2p.server.IrohTransport.from_environment",
        lambda role: transport,
    )
    server = GradientServer.__new__(GradientServer)
    server.scheduler_addr = "scheduler-endpoint"

    assert server._build_iroh() is True
    assert server.iroh_transport is transport
    assert server.lattica is transport
    assert server.scheduler_peer_id == "scheduler-endpoint"


def test_worker_qualifies_scheduler_before_reading_connection_telemetry():
    path = {"kind": "relay", "selected": True, "rtt_ms": 42.0}
    transport = SimpleNamespace(selected_path=lambda peer_id: path)
    server = GradientServer.__new__(GradientServer)
    server.iroh_transport = transport
    server.scheduler_peer_id = "scheduler-endpoint"
    server.scheduler_stub = ProbeStub(ProbeFuture({"peer_id": "scheduler-endpoint"}))

    server._qualify_iroh_scheduler()


def test_worker_rejects_wrong_scheduler_health_identity():
    server = GradientServer.__new__(GradientServer)
    server.iroh_transport = SimpleNamespace(selected_path=lambda peer_id: None)
    server.scheduler_peer_id = "scheduler-endpoint"
    server.scheduler_stub = ProbeStub(ProbeFuture({"peer_id": "different-endpoint"}))

    with pytest.raises(RuntimeError, match="wrong endpoint identity"):
        server._qualify_iroh_scheduler()


@pytest.mark.parametrize("scheduler_addr", [None, "auto", "/ip4/127.0.0.1/tcp/1"])
def test_worker_iroh_rejects_implicit_or_lattica_scheduler_address(scheduler_addr):
    server = GradientServer.__new__(GradientServer)
    server.scheduler_addr = scheduler_addr

    with pytest.raises(ValueError, match="explicit scheduler endpoint ID|not a Lattica"):
        server._build_iroh()
