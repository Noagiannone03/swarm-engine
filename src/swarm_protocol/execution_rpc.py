"""Iroh RPC binding for signed worker-local protocol-v3 admission."""

from __future__ import annotations

from collections.abc import Callable

from lattica import rpc_method

from fabi_network.rpc import authenticated_rpc_peer_id
from swarm_protocol.control import SignedControlMessage
from swarm_protocol.execution import WorkerExecutionAdmission
from swarm_protocol.route_authority import RouteAdmissionEnvelope
from swarm_protocol.speculative import SpeculativeVerifyResponse, SpeculativeVerifyWindow


def control_message_to_wire(message: SignedControlMessage) -> dict[str, object]:
    return {
        "kind": message.kind.value,
        "signer_endpoint_id": message.signer_endpoint_id,
        "payload": message.payload,
        "signature": message.signature,
    }


def route_admission_to_wire(envelope: RouteAdmissionEnvelope) -> dict[str, object]:
    return {
        "signed_plan": control_message_to_wire(envelope.signed_plan),
        "authority_key_id": envelope.authority_key_id,
        "capability_token": envelope.capability_token,
        "permit_id": envelope.permit_id,
        "account_id": envelope.account_id,
        "authorization_generation": envelope.authorization_generation,
        "expires_at_ms": envelope.expires_at_ms,
        "recovery_policy": envelope.recovery_policy.value,
    }


def route_renewal_to_wire(
    command: SignedControlMessage,
    admission: RouteAdmissionEnvelope,
) -> dict[str, object]:
    return {
        "signed_command": control_message_to_wire(command),
        "admission": route_admission_to_wire(admission),
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


class WorkerSpeculativeVerifyService:
    """Dormant bounded RPC for unpublished target-model verification."""

    def __init__(
        self,
        admission: WorkerExecutionAdmission,
        verify_window: Callable[[SpeculativeVerifyWindow], SpeculativeVerifyResponse],
    ) -> None:
        self.admission = admission
        self._verify_window = verify_window

    @rpc_method
    def verify(self, message: dict[str, object]) -> dict[str, object]:
        window = SpeculativeVerifyWindow.model_validate(message)
        self.admission.authorize_speculative_verify(
            request_id=window.request_id,
            route_id=window.route_id,
            epoch=window.epoch,
            route_plan_digest=window.route_plan_digest,
            caller_endpoint_id=authenticated_rpc_peer_id(),
        )
        response = self._verify_window(window)
        if not isinstance(response, SpeculativeVerifyResponse):
            response = SpeculativeVerifyResponse.model_validate(response)
        window.verify_response(response)
        return response.model_dump(mode="json")
