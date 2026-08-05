"""Transport selection and the small compatibility surface used by Parallax."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from .rpc import IrohRpcRuntime, RpcServiceStub
from .relay_enrollment import RelayEnrollmentClient
from swarm_protocol.dht_discovery import DhtDiscoveryStore

IROH_TRANSPORT = "iroh"
LATTICA_TRANSPORT = "lattica"


def configured_transport() -> str:
    value = os.environ.get("FABI_NETWORK_TRANSPORT", LATTICA_TRANSPORT).strip().lower()
    if value not in {IROH_TRANSPORT, LATTICA_TRANSPORT}:
        raise ValueError("FABI_NETWORK_TRANSPORT must be 'iroh' or 'lattica'")
    return value


def using_iroh() -> bool:
    return configured_transport() == IROH_TRANSPORT


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _relay_token(*, required: bool = True) -> str | None:
    token = os.environ.get("FABI_RELAY_TOKEN", "").strip()
    token_file = os.environ.get("FABI_RELAY_TOKEN_FILE", "").strip()
    if token and token_file:
        raise ValueError("set only one of FABI_RELAY_TOKEN and FABI_RELAY_TOKEN_FILE")
    if token:
        return token
    if not token_file:
        if required:
            raise ValueError(
                "FABI_RELAY_TOKEN or FABI_RELAY_TOKEN_FILE is required when automatic enrollment is disabled"
            )
        return None
    path = Path(token_file).expanduser()
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError(f"relay token file {path} must not be accessible by group or others")
    value = path.read_text(encoding="utf-8").strip()
    if value.startswith("IROH_RELAY_ACCESS_TOKEN="):
        value = value.removeprefix("IROH_RELAY_ACCESS_TOKEN=").strip()
    if not value:
        raise ValueError(f"relay token file {path} is empty")
    return value


def _catalog_bootstrap_addresses() -> list[str]:
    raw = os.environ.get("FABI_CATALOG_DHT_BOOTSTRAPS", "").strip()
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        decoded = [line.strip() for line in raw.splitlines() if line.strip()]
    if not isinstance(decoded, list) or any(
        not isinstance(address, str) or not address.strip() for address in decoded
    ):
        raise ValueError("FABI_CATALOG_DHT_BOOTSTRAPS must be a JSON array or newline list")
    return [address.strip() for address in decoded]


def _trusted_demand_publishers() -> dict[str, str]:
    """Parse region-to-endpoint pins provisioned with the signed runtime."""

    raw = os.environ.get("FABI_SWARM_V3_DEMAND_AUTHORITIES", "").strip()
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("FABI_SWARM_V3_DEMAND_AUTHORITIES must be a JSON object") from exc
    if (
        not isinstance(decoded, dict)
        or len(decoded) > 32
        or any(
            not isinstance(region, str)
            or not region.strip()
            or len(region) > 128
            or not isinstance(endpoint, str)
            or not endpoint.strip()
            for region, endpoint in decoded.items()
        )
    ):
        raise ValueError(
            "FABI_SWARM_V3_DEMAND_AUTHORITIES must map at most 32 regions to endpoint IDs"
        )
    normalized: dict[str, str] = {}
    for region, endpoint in decoded.items():
        region_id = region.strip()
        if region_id in normalized:
            raise ValueError("FABI_SWARM_V3_DEMAND_AUTHORITIES has duplicate normalized regions")
        normalized[region_id] = endpoint.strip()
    return normalized


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class IrohTransport:
    """Authenticated RPC plus an optional native peer-discovery catalogue.

    Model bytes remain a separate content-plane concern. The embedded DHT
    carries only bounded signed manifests and short-lived membership records.
    """

    def __init__(
        self,
        runtime: IrohRpcRuntime,
        relay_enrollment: RelayEnrollmentClient | None = None,
    ):
        self.runtime = runtime
        self.relay_enrollment = relay_enrollment
        self.catalog_discovery: DhtDiscoveryStore | None = None
        self.catalog_peer_id: str | None = None
        self.catalog_listen_address: str | None = None

    @classmethod
    def from_environment(cls, role: str) -> IrohTransport:
        relay_url = os.environ.get("FABI_RELAY_URL", "").strip()
        if not relay_url:
            raise ValueError("FABI_RELAY_URL is required for the Iroh transport")
        configured_identity = os.environ.get("FABI_NETWORK_IDENTITY_PATH", "").strip()
        identity_path = Path(configured_identity or f"~/.fabi/network/{role}.key").expanduser()
        enrollment = RelayEnrollmentClient.from_environment(identity_path)
        relay_token = _relay_token(required=enrollment is None)
        initial_lease = enrollment.enroll() if enrollment is not None else None
        try:
            runtime = IrohRpcRuntime(
                identity_path,
                relay_url,
                relay_token,
                force_relay=_env_bool("FABI_FORCE_RELAY"),
            )
        except BaseException:
            if enrollment is not None:
                enrollment.close()
            raise
        transport = cls(runtime, enrollment)
        try:
            if enrollment is not None and initial_lease is not None:
                enrollment.start_refresh(initial_lease)
            transport._start_catalog_from_environment(role)
        except BaseException:
            if enrollment is not None:
                enrollment.close()
            runtime.close()
            raise
        return transport

    def _start_catalog_from_environment(self, role: str) -> None:
        mode = os.environ.get("FABI_CATALOG_DHT_MODE", "off").strip().lower()
        if mode in {"", "off", "disabled"}:
            return
        if mode not in {"client", "server"}:
            raise ValueError("FABI_CATALOG_DHT_MODE must be off, client, or server")
        bootstraps = _catalog_bootstrap_addresses()
        if mode == "client" and not bootstraps:
            raise ValueError("catalogue DHT clients require at least one bootstrap address")
        configured_identity = os.environ.get("FABI_CATALOG_DHT_IDENTITY_PATH", "").strip()
        identity_path = Path(
            configured_identity or f"~/.fabi/network/{role}-catalog.key"
        ).expanduser()
        default_listen = (
            "/ip4/0.0.0.0/tcp/0" if mode == "server" else "/ip4/127.0.0.1/tcp/0"
        )
        listen_address = os.environ.get(
            "FABI_CATALOG_DHT_LISTEN_ADDRESS",
            default_listen,
        ).strip()
        peer_id, bound_address = self.runtime._node.start_catalog_dht(
            identity_path,
            mode == "server",
            listen_address,
            bootstraps,
            15,
            25_000,
            _positive_env_int("FABI_CATALOG_DHT_BOOTSTRAP_INTERVAL_SECONDS", 30),
        )
        if bootstraps:
            self.runtime._node.catalog_bootstrap()
        self.catalog_peer_id = str(peer_id)
        self.catalog_listen_address = str(bound_address)
        trusted_demand_publishers = _trusted_demand_publishers()
        # Parse every pinned Iroh identity through the native implementation at
        # startup. A typo must fail the runtime configuration once, not become
        # an endless stream of shadow lookup failures.
        for region_id, publisher in trusted_demand_publishers.items():
            self.runtime._node.catalog_key(
                "context_demand",
                "0" * 64,
                None,
                region_id,
                publisher,
            )
        self.catalog_discovery = DhtDiscoveryStore(
            self.runtime._node,
            self.catalog_peer_id,
            trusted_demand_publishers=trusted_demand_publishers,
        )

    def peer_id(self) -> str:
        return self.runtime.endpoint_id

    def register(self, handler: object) -> None:
        self.runtime.register(handler)

    def stub(self, peer_id: str, service: object | type[object]) -> RpcServiceStub:
        return self.runtime.stub(peer_id, service)

    def get_all_peers(self) -> list[str]:
        return self.runtime.peers()

    def selected_path(self, peer_id: str) -> dict[str, Any] | None:
        paths = json.loads(self.runtime.paths(peer_id))
        selected = next((path for path in paths if path.get("selected")), None)
        if selected is not None:
            return selected
        return (
            min(paths, key=lambda path: float(path.get("rtt_ms", float("inf")))) if paths else None
        )

    def get_peer_rtt(self, peer_id: str) -> float:
        path = self.selected_path(peer_id)
        if path is None:
            raise RuntimeError(f"peer {peer_id} has no active Iroh path")
        return float(path["rtt_ms"]) / 1000.0

    def sign_control_payload(self, payload: bytes) -> bytes:
        """Sign exact v3 control bytes with this endpoint's stable identity."""

        return bytes(self.runtime._node.sign_control_payload(payload))

    def verify_control_payload(
        self,
        signer_endpoint_id: str,
        payload: bytes,
        signature: bytes,
    ) -> None:
        """Verify exact v3 control bytes against the claimed endpoint identity."""

        self.runtime._node.verify_control_payload(signer_endpoint_id, payload, signature)

    def close(self) -> None:
        if self.relay_enrollment is not None:
            self.relay_enrollment.close()
        if self.catalog_discovery is not None:
            self.runtime._node.stop_catalog_dht()
            self.catalog_discovery = None
        self.runtime.close()

    def store(self, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise RuntimeError("Iroh transport does not provide a DHT; use the central scheduler")

    def get(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("Iroh transport does not provide a DHT; use the central scheduler")

    def get_block(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError(
            "Iroh RPC transport does not implement content-addressed weight distribution"
        )
