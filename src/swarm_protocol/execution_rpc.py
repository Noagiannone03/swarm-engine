"""Iroh RPC binding for signed worker-local protocol-v3 admission."""

from __future__ import annotations

from lattica import rpc_method

from fabi_network.rpc import authenticated_rpc_peer_id
from swarm_protocol.control import SignedControlMessage
from swarm_protocol.execution import WorkerExecutionAdmission


def control_message_to_wire(message: SignedControlMessage) -> dict[str, object]:
    return {
        "kind": message.kind.value,
        "signer_endpoint_id": message.signer_endpoint_id,
        "payload": message.payload,
        "signature": message.signature,
    }


class WorkerExecutionControlService:
    """Small control-only service registered on an authenticated Iroh endpoint."""

    def __init__(self, admission: WorkerExecutionAdmission) -> None:
        self.admission = admission

    @rpc_method
    def prepare(self, message: dict[str, object]) -> dict[str, object]:
        response = self.admission.prepare(
            message,
            caller_endpoint_id=authenticated_rpc_peer_id(),
        )
        return control_message_to_wire(response)

    @rpc_method
    def command(self, message: dict[str, object]) -> dict[str, object] | None:
        response = self.admission.apply_command(
            message,
            caller_endpoint_id=authenticated_rpc_peer_id(),
        )
        return None if response is None else control_message_to_wire(response)
