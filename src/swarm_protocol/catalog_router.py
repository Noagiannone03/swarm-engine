"""Long-lived routing node for the protocol-v3 soft-state catalogue.

Workers run Kademlia in client mode and therefore never become accidental
infrastructure.  This process gives operators one explicit server-mode role
that can be replicated independently from schedulers and model executors.
"""

from __future__ import annotations

import json
import signal
import threading
from dataclasses import dataclass
from typing import Callable, Sequence

from fabi_network.transport import IrohTransport


@dataclass
class CatalogRouter:
    """Own one server-mode catalogue participant and its transport lifecycle."""

    transport: IrohTransport

    @classmethod
    def from_environment(
        cls,
        factory: Callable[[str], IrohTransport] = IrohTransport.from_environment,
    ) -> "CatalogRouter":
        transport = factory("catalog-router")
        if (
            transport.catalog_discovery is None
            or transport.catalog_peer_id is None
            or transport.catalog_listen_address is None
        ):
            transport.close()
            raise RuntimeError(
                "catalog router requires FABI_CATALOG_DHT_MODE=server and a listen address"
            )
        return cls(transport)

    def summary(self) -> dict[str, str]:
        assert self.transport.catalog_peer_id is not None
        assert self.transport.catalog_listen_address is not None
        return {
            "status": "ready",
            "role": "catalog_router",
            "iroh_endpoint_id": self.transport.peer_id(),
            "catalog_peer_id": self.transport.catalog_peer_id,
            "catalog_listen_address": self.transport.catalog_listen_address,
        }

    def close(self) -> None:
        self.transport.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run until SIGINT/SIGTERM while emitting no credentials or private paths."""

    if argv:
        raise ValueError("fabi-catalog-router is configured exclusively through FABI_* env vars")
    router = CatalogRouter.from_environment()
    stopped = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stopped.set()

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        print(json.dumps(router.summary(), sort_keys=True), flush=True)
        stopped.wait()
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        router.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
