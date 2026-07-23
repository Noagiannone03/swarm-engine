"""Transport selection and the small compatibility surface used by Parallax."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from .rpc import IrohRpcRuntime, RpcServiceStub

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


def _relay_token() -> str:
    token = os.environ.get("FABI_RELAY_TOKEN", "").strip()
    token_file = os.environ.get("FABI_RELAY_TOKEN_FILE", "").strip()
    if token and token_file:
        raise ValueError("set only one of FABI_RELAY_TOKEN and FABI_RELAY_TOKEN_FILE")
    if token:
        return token
    if not token_file:
        raise ValueError(
            "FABI_RELAY_TOKEN or FABI_RELAY_TOKEN_FILE is required for the authenticated relay"
        )
    path = Path(token_file).expanduser()
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError(f"relay token file {path} must not be accessible by group or others")
    value = path.read_text(encoding="utf-8").strip()
    if value.startswith("IROH_RELAY_ACCESS_TOKEN="):
        value = value.removeprefix("IROH_RELAY_ACCESS_TOKEN=").strip()
    if not value:
        raise ValueError(f"relay token file {path} is empty")
    return value


class IrohTransport:
    """Runtime facade for centrally scheduled Parallax components.

    DHT and content-addressed weight distribution are deliberately absent: the
    scheduler owns discovery/routing, while weight distribution remains a
    separate content-plane concern.
    """

    def __init__(self, runtime: IrohRpcRuntime):
        self.runtime = runtime

    @classmethod
    def from_environment(cls, role: str) -> IrohTransport:
        relay_url = os.environ.get("FABI_RELAY_URL", "").strip()
        if not relay_url:
            raise ValueError("FABI_RELAY_URL is required for the Iroh transport")
        relay_token = _relay_token()
        configured_identity = os.environ.get("FABI_NETWORK_IDENTITY_PATH", "").strip()
        identity_path = Path(configured_identity or f"~/.fabi/network/{role}.key").expanduser()
        runtime = IrohRpcRuntime(
            identity_path,
            relay_url,
            relay_token,
            force_relay=_env_bool("FABI_FORCE_RELAY"),
        )
        return cls(runtime)

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
