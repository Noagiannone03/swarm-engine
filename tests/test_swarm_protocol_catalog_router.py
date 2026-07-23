from types import SimpleNamespace

import pytest

from swarm_protocol.catalog_router import CatalogRouter


class FakeTransport:
    def __init__(self, *, attached: bool = True):
        self.catalog_discovery = object() if attached else None
        self.catalog_peer_id = "catalog-peer" if attached else None
        self.catalog_listen_address = "/ip4/127.0.0.1/tcp/19191" if attached else None
        self.runtime = SimpleNamespace(endpoint_id="iroh-endpoint")
        self.closed = False

    def peer_id(self) -> str:
        return self.runtime.endpoint_id

    def close(self) -> None:
        self.closed = True


def test_catalog_router_exposes_public_identity_and_closes() -> None:
    transport = FakeTransport()
    router = CatalogRouter.from_environment(lambda role: transport)

    assert router.summary() == {
        "status": "ready",
        "role": "catalog_router",
        "iroh_endpoint_id": "iroh-endpoint",
        "catalog_peer_id": "catalog-peer",
        "catalog_listen_address": "/ip4/127.0.0.1/tcp/19191",
    }

    router.close()
    assert transport.closed


def test_catalog_router_fails_closed_without_server_catalogue() -> None:
    transport = FakeTransport(attached=False)

    with pytest.raises(RuntimeError, match="FABI_CATALOG_DHT_MODE=server"):
        CatalogRouter.from_environment(lambda role: transport)

    assert transport.closed
