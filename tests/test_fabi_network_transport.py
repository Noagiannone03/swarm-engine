from __future__ import annotations

import pytest

from fabi_network.transport import (
    IrohTransport,
    _catalog_bootstrap_addresses,
    _trusted_demand_publishers,
)


class FakeNode:
    endpoint_id = "endpoint"

    def __init__(self) -> None:
        self.started = None
        self.bootstrapped = False
        self.stopped = False

    def start_catalog_dht(
        self,
        identity_path,
        server_mode,
        listen_address,
        bootstraps,
        replication_factor,
        query_timeout_ms,
        bootstrap_interval_seconds,
    ):
        self.started = {
            "identity_path": identity_path,
            "server_mode": server_mode,
            "listen_address": listen_address,
            "bootstraps": bootstraps,
            "replication_factor": replication_factor,
            "query_timeout_ms": query_timeout_ms,
            "bootstrap_interval_seconds": bootstrap_interval_seconds,
        }
        return "catalog-peer", "/ip4/127.0.0.1/tcp/4242"

    def catalog_bootstrap(self):
        self.bootstrapped = True

    def catalog_key(self, kind, model_swarm_id, worker_id, region_id, publisher=None):
        return f"{kind}:{model_swarm_id}:{worker_id}:{region_id}:{publisher}"

    def stop_catalog_dht(self):
        self.stopped = True


class FakeRuntime:
    instances = []

    def __init__(self, identity_path, relay_url, relay_token, *, force_relay):
        self.identity_path = identity_path
        self.relay_url = relay_url
        self.relay_token = relay_token
        self.force_relay = force_relay
        self.endpoint_id = "endpoint"
        self._node = FakeNode()
        self.closed = False
        self.instances.append(self)

    def close(self):
        self.closed = True


def base_environment(monkeypatch):
    FakeRuntime.instances.clear()
    monkeypatch.setattr("fabi_network.transport.IrohRpcRuntime", FakeRuntime)
    monkeypatch.setenv("FABI_RELAY_URL", "https://relay.invalid")
    monkeypatch.setenv("FABI_RELAY_TOKEN", "token")
    monkeypatch.delenv("FABI_RELAY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("FABI_SWARM_V3_DEMAND_AUTHORITIES", raising=False)


def test_catalogue_bootstrap_parser_accepts_json_or_newline_lists(monkeypatch):
    monkeypatch.setenv("FABI_CATALOG_DHT_BOOTSTRAPS", '["/dns/a/tcp/1", "/dns/b/tcp/2"]')
    assert _catalog_bootstrap_addresses() == ["/dns/a/tcp/1", "/dns/b/tcp/2"]


def test_trusted_demand_publishers_require_a_bounded_json_mapping(monkeypatch):
    monkeypatch.setenv(
        "FABI_SWARM_V3_DEMAND_AUTHORITIES",
        '{"eu-west":"endpoint-a","local":"endpoint-b"}',
    )
    assert _trusted_demand_publishers() == {
        "eu-west": "endpoint-a",
        "local": "endpoint-b",
    }
    monkeypatch.setenv("FABI_SWARM_V3_DEMAND_AUTHORITIES", "[]")
    with pytest.raises(ValueError, match="map at most 32 regions"):
        _trusted_demand_publishers()
    monkeypatch.setenv(
        "FABI_SWARM_V3_DEMAND_AUTHORITIES",
        '{"eu-west":"endpoint-a"," eu-west ":"endpoint-b"}',
    )
    with pytest.raises(ValueError, match="duplicate normalized regions"):
        _trusted_demand_publishers()
    monkeypatch.setenv("FABI_CATALOG_DHT_BOOTSTRAPS", "/dns/a/tcp/1\n/dns/b/tcp/2")
    assert _catalog_bootstrap_addresses() == ["/dns/a/tcp/1", "/dns/b/tcp/2"]


def test_catalogue_client_fails_closed_without_bootstrap_and_closes_rpc(monkeypatch):
    base_environment(monkeypatch)
    monkeypatch.setenv("FABI_CATALOG_DHT_MODE", "client")
    monkeypatch.delenv("FABI_CATALOG_DHT_BOOTSTRAPS", raising=False)

    with pytest.raises(ValueError, match="bootstrap"):
        IrohTransport.from_environment("worker")

    assert FakeRuntime.instances[0].closed


def test_catalogue_server_starts_embedded_dht_and_closes_it(monkeypatch):
    base_environment(monkeypatch)
    monkeypatch.setenv("FABI_CATALOG_DHT_MODE", "server")
    monkeypatch.setenv("FABI_CATALOG_DHT_BOOTSTRAPS", "/dns/bootstrap/tcp/4242")

    transport = IrohTransport.from_environment("scheduler")
    runtime = FakeRuntime.instances[0]
    try:
        assert transport.catalog_peer_id == "catalog-peer"
        assert transport.catalog_listen_address == "/ip4/127.0.0.1/tcp/4242"
        assert transport.catalog_discovery is not None
        assert runtime._node.started["server_mode"] is True
        assert runtime._node.started["replication_factor"] == 15
        assert runtime._node.started["query_timeout_ms"] == 25_000
        assert runtime._node.started["bootstrap_interval_seconds"] == 30
        assert runtime._node.bootstrapped
    finally:
        transport.close()

    assert runtime._node.stopped
    assert runtime.closed


def test_connected_scheduler_is_the_default_signed_demand_authority(monkeypatch):
    base_environment(monkeypatch)
    monkeypatch.setenv("FABI_CATALOG_DHT_MODE", "client")
    monkeypatch.setenv("FABI_CATALOG_DHT_BOOTSTRAPS", "/dns/bootstrap/tcp/4242")

    transport = IrohTransport.from_environment(
        "worker",
        trusted_demand_publishers={"global": "scheduler-endpoint"},
    )
    try:
        assert transport.catalog_discovery is not None
        assert transport.catalog_discovery._trusted_demand_publishers == {
            "global": "scheduler-endpoint"
        }
    finally:
        transport.close()

    monkeypatch.setenv(
        "FABI_SWARM_V3_DEMAND_AUTHORITIES",
        '{"global":"different-endpoint"}',
    )
    with pytest.raises(ValueError, match="conflicts with the connected scheduler"):
        IrohTransport.from_environment(
            "worker",
            trusted_demand_publishers={"global": "scheduler-endpoint"},
        )


def test_catalogue_bootstrap_interval_is_configurable_and_positive(monkeypatch):
    base_environment(monkeypatch)
    monkeypatch.setenv("FABI_CATALOG_DHT_MODE", "client")
    monkeypatch.setenv("FABI_CATALOG_DHT_BOOTSTRAPS", "/dns/bootstrap/tcp/4242")
    monkeypatch.setenv("FABI_CATALOG_DHT_BOOTSTRAP_INTERVAL_SECONDS", "45")

    transport = IrohTransport.from_environment("worker")
    try:
        assert FakeRuntime.instances[0]._node.started["bootstrap_interval_seconds"] == 45
    finally:
        transport.close()

    monkeypatch.setenv("FABI_CATALOG_DHT_BOOTSTRAP_INTERVAL_SECONDS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        IrohTransport.from_environment("worker")
    assert FakeRuntime.instances[-1].closed


def test_automatic_relay_enrollment_removes_client_relay_token(monkeypatch):
    base_environment(monkeypatch)
    monkeypatch.delenv("FABI_RELAY_TOKEN", raising=False)
    monkeypatch.delenv("FABI_RELAY_TOKEN_FILE", raising=False)
    monkeypatch.setenv("FABI_CATALOG_DHT_MODE", "off")

    class FakeEnrollment:
        def __init__(self):
            self.enrolled = False
            self.started_with = None
            self.closed = False

        def enroll(self):
            self.enrolled = True
            return object()

        def start_refresh(self, lease):
            self.started_with = lease

        def close(self):
            self.closed = True

    enrollment = FakeEnrollment()
    monkeypatch.setattr(
        "fabi_network.transport.RelayEnrollmentClient.from_environment",
        lambda identity_path: enrollment,
    )

    transport = IrohTransport.from_environment("worker")
    assert enrollment.enrolled
    assert enrollment.started_with is not None
    assert FakeRuntime.instances[-1].relay_token is None
    transport.close()
    assert enrollment.closed
