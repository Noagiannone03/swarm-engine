"""Signed, bounded control messages for protocol-v3 route admission."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from swarm_protocol.contracts import ContractModel

_MAX_CONTROL_PAYLOAD_BYTES = 256 * 1024
_T = TypeVar("_T", bound=ContractModel)


class ControlMessageKind(str, Enum):
    ROUTE_PLAN = "route_plan"
    RESERVATION_COMMAND = "reservation_command"
    RESERVATION_LEASE = "reservation_lease"


class ControlCrypto(Protocol):
    def peer_id(self) -> str: ...

    def sign_control_payload(self, payload: bytes) -> bytes: ...

    def verify_control_payload(
        self,
        signer_endpoint_id: str,
        payload: bytes,
        signature: bytes,
    ) -> None: ...


class SignedControlMessage(BaseModel):
    """Exact signed bytes plus the endpoint identity that must verify them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ControlMessageKind
    signer_endpoint_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: bytes = Field(min_length=1, max_length=_MAX_CONTROL_PAYLOAD_BYTES)
    signature: bytes = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_payload(self) -> "SignedControlMessage":
        if len(self.payload) > _MAX_CONTROL_PAYLOAD_BYTES:
            raise ValueError("control payload exceeds the protocol maximum")
        return self


def sign_control_contract(
    contract: ContractModel,
    *,
    kind: ControlMessageKind,
    crypto: ControlCrypto,
) -> SignedControlMessage:
    """Serialize once, sign those exact bytes, and transport them unchanged."""

    payload = contract.model_dump_json().encode("utf-8")
    return SignedControlMessage(
        kind=kind,
        signer_endpoint_id=crypto.peer_id(),
        payload=payload,
        signature=crypto.sign_control_payload(payload),
    )


def verify_control_contract(
    message: SignedControlMessage | dict[str, object],
    *,
    expected_kind: ControlMessageKind,
    expected_signer_endpoint_id: str,
    contract_type: type[_T],
    crypto: ControlCrypto,
) -> _T:
    """Authenticate exact bytes before parsing their application fields."""

    envelope = (
        message
        if isinstance(message, SignedControlMessage)
        else SignedControlMessage.model_validate(message)
    )
    if envelope.kind != expected_kind:
        raise ValueError(
            f"expected {expected_kind.value} control message, got {envelope.kind.value}"
        )
    if envelope.signer_endpoint_id != expected_signer_endpoint_id:
        raise PermissionError("control message signer is not the expected endpoint")
    crypto.verify_control_payload(
        envelope.signer_endpoint_id,
        envelope.payload,
        envelope.signature,
    )
    return contract_type.model_validate_json(envelope.payload)
